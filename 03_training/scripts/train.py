"""Train the YOLO11s detection and drivable-area segmentation model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask import (DEFAULT_DATA, DEFAULT_WEIGHTS, MultiTaskDataset, MultiTaskYOLO11s,
                       collate_batch, evaluate_model, segmentation_loss, to_device)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Local COCO-pretrained yolo11s.pt")
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint made by this script")
    parser.add_argument("--output", type=Path, default=Path("runs/yolo11s_multitask"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seg-weight", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Use FP32 when mixed precision is unstable")
    parser.add_argument("--p2-detect", action="store_true", help="Add P2/4 detection scale")
    parser.add_argument("--copy-paste-bank", type=Path, help="Train-only SAM cutout bank")
    parser.add_argument("--copy-paste-prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--train-limit", type=int, help="Limit images for a smoke test")
    parser.add_argument("--val-limit", type=int, help="Limit validation images for a smoke test")
    return parser.parse_args()


def main():
    args = arguments()
    if args.epochs < 1 or args.batch < 1 or args.accumulate < 1 or args.seg_weight < 0:
        raise ValueError("epochs/batch/accumulate must be positive and seg-weight must be nonnegative")
    if not 0 <= args.copy_paste_prob <= 1:
        raise ValueError("copy-paste-prob must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    size = (args.height, args.width)
    train_set = MultiTaskDataset(args.data, "train", size, augment=True, limit=args.train_limit,
                                 copy_paste_bank=args.copy_paste_bank,
                                 copy_paste_prob=args.copy_paste_prob)
    val_set = MultiTaskDataset(args.data, "val", size, limit=args.val_limit)
    loader_options = {"batch_size": args.batch, "num_workers": args.workers,
                      "collate_fn": collate_batch, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = MultiTaskYOLO11s(pretrained=None if args.resume else args.weights,
                              p2_detect=args.p2_detect).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    start_epoch, best_score, stale = 0, -math.inf, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_score = state["best_score"]
        stale = state.get("stale", 0)
        saved_size = (state["config"]["height"], state["config"]["width"])
        if saved_size != size:
            raise ValueError(f"Resume image size {saved_size} differs from requested {size}")
        if state["config"].get("p2_detect", False) != args.p2_detect:
            raise ValueError("Resume detector architecture differs from --p2-detect")
        if state["config"].get("accumulate", 1) != args.accumulate:
            raise ValueError("Resume gradient accumulation differs from --accumulate")
        saved_bank = state["config"].get("copy_paste_bank")
        current_bank = str(args.copy_paste_bank.resolve()) if args.copy_paste_bank else None
        if saved_bank != current_bank or state["config"].get("copy_paste_prob", 0.3) != args.copy_paste_prob:
            raise ValueError("Resume Copy-Paste settings differ from checkpoint")

    print(f"device={device} train={len(train_set)} val={len(val_set)} size={size} "
          f"batch={args.batch} accumulate={args.accumulate} p2_detect={args.p2_detect}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        optimizer_steps = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}",
                        disable=not sys.stderr.isatty())
        for batch_index, batch in enumerate(progress):
            batch = to_device(batch, device)
            if batch_index % args.accumulate == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                det_pred, seg_logits = model(batch["img"])
                det_loss, _ = model.detection_loss(det_pred, batch)
                seg_loss = (segmentation_loss(seg_logits, batch["mask"])
                            if args.seg_weight else det_loss.new_zeros(()))
                loss = det_loss + args.seg_weight * seg_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {loss.item()}")
            group_start = batch_index - batch_index % args.accumulate
            group_size = min(args.accumulate, len(progress) - group_start)
            scaler.scale(loss / group_size).backward()
            if batch_index - group_start + 1 == group_size:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += not amp or scaler.get_scale() >= old_scale
            n = batch["img"].shape[0]
            totals += np.array([loss.item(), det_loss.item(), seg_loss.item()]) * n
            progress.set_postfix(loss=f"{loss.item():.3f}", det=f"{det_loss.item():.3f}",
                                 seg=f"{seg_loss.item():.3f}")
        if optimizer_steps == 0:
            raise FloatingPointError("AMP skipped every optimizer step; try a lower learning rate or disable AMP")
        scheduler.step()
        scores = evaluate_model(model, val_loader, device, segmentation=args.seg_weight > 0)
        selection_metric = "detection_map50_95" if args.seg_weight == 0 else "joint_map50_95_iou"
        score = (scores["detection"]["map50_95"] if args.seg_weight == 0 else
                 (scores["detection"]["map50_95"] + scores["segmentation"]["iou"]) / 2)
        improved = score > best_score
        best_score = max(best_score, score)
        stale = 0 if improved else stale + 1
        record = {"epoch": epoch + 1, "optimizer_steps": optimizer_steps, "amp": amp,
                  "loss": (totals / len(train_set)).tolist(),
                  "lr": scheduler.get_last_lr()[0], "selection_metric": selection_metric,
                  "score": score, **scores}
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_score": best_score, "stale": stale,
                 "config": {"height": args.height, "width": args.width, "seg_weight": args.seg_weight,
                            "p2_detect": args.p2_detect, "accumulate": args.accumulate,
                            "copy_paste_bank": str(args.copy_paste_bank.resolve()) if args.copy_paste_bank else None,
                            "copy_paste_prob": args.copy_paste_prob,
                            "amp": amp,
                            "data": str(args.data), "classes": ["person", "vehicle"]}}
        torch.save(state, args.output / "last.pt")
        if improved:
            torch.save(state, args.output / "best.pt")
        print(json.dumps(record, ensure_ascii=False))
        if stale >= args.patience:
            print(f"Early stopping after {stale} epochs without improvement")
            break


if __name__ == "__main__":
    main()
