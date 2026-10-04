"""Fine-tune the sensitivity-selected joint-pruned H2 detector."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PRUNING_ROOT = ROOT / "04_pruning"
sys.path.insert(0, str(ROOT / "03_training/scripts"))
sys.path.insert(0, str(PRUNING_ROOT))
from checkpoint_compat import register_pickle_aliases

register_pickle_aliases()
from multitask import (MultiTaskDataset, collate_batch, evaluate_model, segmentation_loss,
                       to_device)
from architectures import FeatureFusionHead


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PRUNING_ROOT / "sensitivity_full/joint_probe.pt")
    parser.add_argument("--output", type=Path, default=PRUNING_ROOT / "sensitivity_full/joint_finetuned")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    size = (config["height"], config["width"])
    train_data = MultiTaskDataset(config["data"], "train", size, augment=True,
                                  copy_paste_bank=config["copy_paste_bank"],
                                  copy_paste_prob=config["copy_paste_prob"])
    val_data = MultiTaskDataset(config["data"], "val", size)
    loader_options = {"batch_size": args.batch, "num_workers": args.workers,
                      "collate_fn": collate_batch, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model = checkpoint["model"].to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.lr * .01)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    last_path = args.output / "last.pt"
    best_path = args.output / "best.pt"
    first_epoch, best_score, stale = 0, -math.inf, 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        first_epoch, best_score, stale = state["epoch"], state["best_score"], state["stale"]
    for epoch in range(first_epoch, args.epochs):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        start = time.perf_counter()
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                det, seg = model(batch["img"])
                det_loss, _ = model.detection_loss(det, batch)
                seg_loss = segmentation_loss(seg, batch["mask"])
                loss = det_loss + config["seg_weight"] * seg_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            totals += np.array([loss.item(), det_loss.item(), seg_loss.item()]) * len(batch["img"])
        scheduler.step()
        metrics = evaluate_model(model, val_loader, device)
        score = (metrics["detection"]["map50_95"] + metrics["segmentation"]["iou"]) / 2
        improved = score > best_score
        best_score = max(best_score, score)
        stale = 0 if improved else stale + 1
        record = {"epoch": epoch + 1, "seconds": time.perf_counter() - start,
                  "loss": (totals / len(train_data)).tolist(), "lr": scheduler.get_last_lr()[0],
                  "score": score, "best_score": best_score, "stale": stale, **metrics}
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch + 1, "best_score": best_score, "stale": stale}
        torch.save(state, last_path)
        if improved:
            torch.save({"model": model, "config": config, "source": str(args.source),
                        "epoch": epoch + 1, "selection_score": score}, best_path)
            save_json(args.output / "val_metrics.json", metrics)
        print(f"epoch={epoch + 1} loss={record['loss'][0]:.4f} "
              f"mAP={metrics['detection']['map50_95']:.4f} "
              f"person={metrics['detection']['per_class']['person']['map50_95']:.4f} "
              f"IoU={metrics['segmentation']['iou']:.4f} best={best_score:.4f} "
              f"seconds={record['seconds']:.1f}", flush=True)
        if stale >= args.patience:
            break
    print(f"saved {best_path}", flush=True)


if __name__ == "__main__":
    main()
