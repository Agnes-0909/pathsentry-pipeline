"""Run the three structured-pruning experiments on the B1 multitask model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch_pruning as tp
from thop import profile
from torch.utils.data import DataLoader
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(__file__).resolve().parents[2]
PRUNING_ROOT = ROOT / "04_pruning"
sys.path.insert(0, str(ROOT / "03_training/scripts"))
sys.path.insert(0, str(PRUNING_ROOT))
from checkpoint_compat import register_pickle_aliases

register_pickle_aliases()
from multitask import (MultiTaskDataset, MultiTaskYOLO11s, collate_batch, evaluate_model,
                       segmentation_loss, to_device)
from visualize_predictions import selected_indices

BASE = ROOT / "03_training/runs/p01/b1/best.pt"
OUTPUT = PRUNING_ROOT / "experiments"
STAGES = {"s1": ("p02_s1_10pct", 0.10), "s2": ("p02_s2_20pct", 0.20),
          "s3": ("p02_s3_30pct", 0.30)}
ROOT_LAYERS = ("1.conv", "2.cv2.conv", "3.conv", "4.cv2.conv", "5.conv",
               "6.cv2.conv", "7.conv", "8.cv2.conv", "9.cv2.conv", "10.cv2.conv",
               "13.cv2.conv", "16.cv2.conv", "17.conv", "19.cv2.conv",
               "20.conv", "22.cv2.conv")
P3_PROTECTED = {"1.conv", "2.cv2.conv", "3.conv", "4.cv2.conv", "13.cv2.conv",
                "16.cv2.conv", "17.conv"}


def save_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_slim(path: Path, device: torch.device):
    # These full-module checkpoints are generated locally by this script and are trusted inputs.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"].to(device), checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_stats(model, size, device):
    model.eval()
    sample = torch.zeros(1, 3, *size, device=device)
    with torch.inference_mode():
        macs, params = profile(model, inputs=(sample,), verbose=False)
        det, seg = model(sample)
    for module in model.modules():
        module._buffers.pop("total_ops", None)
        module._buffers.pop("total_params", None)
    actual_params = sum(parameter.numel() for parameter in model.parameters())
    if int(params) != actual_params:
        raise RuntimeError(f"THOP counted {int(params)} parameters, expected {actual_params}")
    return {"parameters": int(params), "gmacs": macs / 1e9,
            "detection_shape": list(det[0].shape), "segmentation_shape": list(seg.shape)}


def prune_model(model, strength: float, device: torch.device):
    model.eval()
    manifest = []
    for suffix in ROOT_LAYERS:
        name = "detector.model." + suffix
        layer = dict(model.named_modules())[name]
        original = layer.out_channels
        ratio = strength * (0.5 if suffix in P3_PROTECTED else 1.0)
        keep = max(32, round(original * (1 - ratio) / 8) * 8)
        count = original - min(original, keep)
        if count == 0:
            manifest.append({"layer": name, "before": original, "after": original,
                             "ratio": ratio, "removed_indices": []})
            continue
        importance = layer.weight.detach().abs().sum(dim=(1, 2, 3))
        indices = torch.argsort(importance)[:count].tolist()
        graph = tp.DependencyGraph().build_dependency(
            model, torch.zeros(1, 3, 64, 64, device=device), verbose=False)
        group = graph.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Invalid pruning group for {name}")
        group.prune()
        manifest.append({"layer": name, "before": original,
                         "after": layer.out_channels, "ratio": ratio,
                         "removed_indices": sorted(indices)})
    return manifest


@torch.no_grad()
def calibrate_bn(model, loader, device, batches: int):
    model.train()
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        model(batch["img"].to(device).float() / 255)
    model.eval()


@torch.inference_mode()
def benchmark(model, dataset, device, samples=32, warmup=10):
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int)
    images = [dataset[int(i)]["img"].unsqueeze(0).to(device).float() / 255 for i in indices]

    def infer(image):
        det, seg = model(image)
        non_max_suppression(det, conf_thres=0.25, iou_thres=0.7, nc=2, max_det=300)
        _ = seg.sigmoid() > 0.5

    for i in range(warmup):
        infer(images[i % len(images)])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for image in images:
        start = time.perf_counter()
        infer(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    return {"samples": len(images), "mean_ms": float(np.mean(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "peak_cuda_mib": (torch.cuda.max_memory_allocated(device) / 1024**2
                              if device.type == "cuda" else None)}


@torch.inference_mode()
def visualize(model, dataset, device, output: Path, count=12):
    output.mkdir(exist_ok=True)
    thumbnails = []
    for index in selected_indices(dataset, count):
        sample = dataset[index]
        image = sample["img"].permute(1, 2, 0).numpy()
        det, logits = model(sample["img"].unsqueeze(0).to(device).float() / 255)
        predictions = non_max_suppression(det, conf_thres=0.05, iou_thres=0.7,
                                          nc=2, multi_label=True, max_det=300)[0]
        seg = F.interpolate(logits.float(), size=image.shape[:2], mode="bilinear",
                            align_corners=False)[0, 0].sigmoid().cpu().numpy() > 0.5
        result = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        overlay = result.copy()
        overlay[seg] = (90, 190, 70)
        result = cv2.addWeighted(result, 0.72, overlay, 0.28, 0)
        h, w = image.shape[:2]
        for class_id, cx, cy, bw, bh in torch.cat((sample["cls"], sample["bboxes"]), 1).numpy():
            cv2.rectangle(result, (round((cx - bw / 2) * w), round((cy - bh / 2) * h)),
                          (round((cx + bw / 2) * w), round((cy + bh / 2) * h)), (255, 255, 255), 1)
        for x1, y1, x2, y2, score, class_id in predictions.cpu().numpy():
            color = (30, 30, 230) if int(class_id) == 0 else (240, 190, 20)
            cv2.rectangle(result, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
            if int(class_id) == 0:
                cv2.putText(result, f"person {score:.2f}", (round(x1), max(18, round(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(result, Path(sample["path"]).name, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(output / Path(sample["path"]).name), result)
        thumbnails.append(cv2.resize(result, (480, 416), interpolation=cv2.INTER_AREA))
    for start in range(0, len(thumbnails), 6):
        rows = []
        for i in range(start, min(start + 6, len(thumbnails)), 2):
            pair = thumbnails[i:i + 2]
            rows.append(np.concatenate(pair if len(pair) == 2 else [pair[0], np.zeros_like(pair[0])], 1))
        cv2.imwrite(str(output / f"contact_{start // 6:02d}.jpg"), np.concatenate(rows, 0))


def update_record(output: Path, baseline: dict):
    lines = ["# P02 结构化剪枝实验记录", "", "基线：P01-B1，960×832，YOLO11s 多任务。"
             "S1/S2/S3 均从同一 B1 权重独立剪枝、独立微调。", "",
             "| 实验 | 状态 | epoch | 参数量 | GMACs | mAP50-95 | person AP | 分割 IoU | 延迟 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
             f"| B1 | 基线 | - | {baseline['parameters']:,} | {baseline['gmacs']:.2f} | "
             f"{baseline['detection']['map50_95']:.4f} | "
             f"{baseline['detection']['per_class']['person']['map50_95']:.4f} | "
             f"{baseline['segmentation']['iou']:.4f} | {baseline['mean_ms']:.2f} |"]
    for stage, (directory, _) in STAGES.items():
        folder = output / directory
        status = json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {}
        stats = json.loads((folder / "model_stats.json").read_text()) if (folder / "model_stats.json").exists() else {}
        metrics = json.loads((folder / "val_metrics.json").read_text()) if (folder / "val_metrics.json").exists() else {}
        speed = json.loads((folder / "benchmark.json").read_text()) if (folder / "benchmark.json").exists() else {}
        det = metrics.get("detection", {})
        seg = metrics.get("segmentation", {})
        cells = [stage.upper(), status.get("state", "pending"), str(status.get("epoch", "-")),
                 f"{stats['parameters']:,}" if stats else "-",
                 f"{stats['gmacs']:.2f}" if stats else "-",
                 f"{det['map50_95']:.4f}" if det else "-",
                 f"{det['per_class']['person']['map50_95']:.4f}" if det else "-",
                 f"{seg['iou']:.4f}" if seg else "-",
                 f"{speed['mean_ms']:.2f}" if speed else "-"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "选型门槛：mAP50-95 下降 ≤0.005；person AP 下降 ≤0.003；"
              "分割 IoU 下降 ≤0.002；延迟下降 ≥20% 或实际计算量/参数量下降。", "",
              "训练：B1 原始训练划分与 SAM 审核版 Copy-Paste，概率 0.3；batch 2；"
              "分割权重 5；AdamW，初始学习率 5e-5；最多 30 epoch，耐心 10。", "",
              "每组目录保存剪枝清单、校准后指标、微调历史、最优验证指标、速度与可视化。"
              "仅最终通过门槛的候选进入测试集评测。", ""]
    (PRUNING_ROOT / "EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8")


def status(folder: Path, state: str, epoch: int, output: Path, baseline: dict, **extra):
    save_json(folder / "status.json", {"state": state, "epoch": epoch,
                                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), **extra})
    update_record(output, baseline)


def run_stage(stage: str, args, base_state: dict, baseline: dict, device: torch.device):
    dirname, strength = STAGES[stage]
    folder = args.output / dirname
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    config = base_state["config"]
    size = (config["height"], config["width"])
    train_data = MultiTaskDataset(config["data"], "train", size, augment=True,
                                  copy_paste_bank=config["copy_paste_bank"],
                                  copy_paste_prob=config["copy_paste_prob"])
    calibration_data = MultiTaskDataset(config["data"], "train", size)
    val_data = MultiTaskDataset(config["data"], "val", size)
    loader_args = {"batch_size": args.batch, "num_workers": args.workers,
                   "collate_fn": collate_batch, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    calibration_loader = DataLoader(calibration_data, shuffle=False, **loader_args)
    pruned_path = folder / "pruned.pt"
    last_path = folder / "finetuned_last.pt"
    best_path = folder / "finetuned_best.pt"

    if pruned_path.exists():
        model, _ = load_slim(pruned_path, device)
        print(f"[{stage}] loading existing pruned model", flush=True)
    else:
        model = MultiTaskYOLO11s(p2_detect=False).to(device)
        model.load_state_dict(base_state["model"])
        status(folder, "pruning", 0, args.output, baseline)
        manifest = prune_model(model, strength, device)
        calibrate_bn(model, calibration_loader, device, args.calibration_batches)
        stats = model_stats(model, size, device)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError(f"{stage}: non-finite weight after pruning")
        save_json(folder / "prune_manifest.json", {"stage": stage, "target_ratio": strength,
                   "p3_ratio": strength * 0.5, "layers": manifest})
        torch.save({"model": model, "config": config, "stage": stage}, pruned_path)
        stats["pruned_file_mib"] = pruned_path.stat().st_size / 1024**2
        save_json(folder / "model_stats.json", stats)
        metrics = evaluate_model(model, val_loader, device)
        save_json(folder / "unfinetuned_val_metrics.json", metrics)
        status(folder, "pruned", 0, args.output, baseline)
        print(f"[{stage}] pruned params={stats['parameters']:,} gmacs={stats['gmacs']:.2f}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    first_epoch, best_score, stale = 0, -math.inf, 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        first_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        stale = checkpoint["stale"]
        print(f"[{stage}] resuming at epoch {first_epoch + 1}", flush=True)

    if first_epoch < args.epochs and stale < args.patience:
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
                    raise FloatingPointError(f"{stage}: non-finite loss in epoch {epoch + 1}")
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
            best_score = max(score, best_score)
            stale = 0 if improved else stale + 1
            record = {"epoch": epoch + 1, "seconds": time.perf_counter() - start,
                      "loss": (totals / len(train_data)).tolist(), "lr": scheduler.get_last_lr()[0],
                      "selection_score": score, "best_score": best_score,
                      "stale": stale, **metrics}
            with (folder / "history.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch + 1, "best_score": best_score, "stale": stale}, last_path)
            if improved:
                torch.save({"model": model, "config": config, "stage": stage,
                            "epoch": epoch + 1, "selection_score": score}, best_path)
                save_json(folder / "val_metrics.json", metrics)
            status(folder, "training", epoch + 1, args.output, baseline,
                   best_score=best_score, stale=stale)
            print(f"[{stage}] epoch={epoch + 1} loss={record['loss'][0]:.4f} "
                  f"mAP={metrics['detection']['map50_95']:.4f} "
                  f"IoU={metrics['segmentation']['iou']:.4f} "
                  f"best={best_score:.4f} seconds={record['seconds']:.1f}", flush=True)
            if stale >= args.patience:
                break

    model, checkpoint = load_slim(best_path, device)
    metrics = json.loads((folder / "val_metrics.json").read_text())
    speed = benchmark(model, val_data, device)
    save_json(folder / "benchmark.json", speed)
    visualize(model, val_data, device, folder / "visuals")
    passed = (metrics["detection"]["map50_95"] >= baseline["detection"]["map50_95"] - .005
              and metrics["detection"]["per_class"]["person"]["map50_95"] >=
              baseline["detection"]["per_class"]["person"]["map50_95"] - .003
              and metrics["segmentation"]["iou"] >= baseline["segmentation"]["iou"] - .002)
    stats = json.loads((folder / "model_stats.json").read_text())
    efficiency = (speed["mean_ms"] <= baseline["mean_ms"] * .8
                  or stats["parameters"] < baseline["parameters"]
                  or stats["gmacs"] < baseline["gmacs"])
    status(folder, "complete", checkpoint["epoch"], args.output, baseline,
           quality_pass=passed, efficiency_pass=efficiency,
           acceptable=passed and efficiency)
    print(f"[{stage}] complete quality_pass={passed} efficiency_pass={efficiency}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.patience < 1:
        parser.error("epochs, batch and patience must be positive")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(BASE, map_location="cpu", weights_only=True)
    base_val = json.loads((BASE.parent / "val_metrics.json").read_text())
    base_speed = json.loads((BASE.parent / "benchmark.json").read_text())
    baseline = {"parameters": 10003415, "gmacs": 31.88884608, **base_val, **base_speed}
    update_record(args.output, baseline)
    for stage in args.stages:
        folder = args.output / STAGES[stage][0]
        current = json.loads((folder / "status.json").read_text()) if (folder / "status.json").exists() else {}
        if current.get("state") == "complete":
            print(f"[{stage}] already complete", flush=True)
            continue
        try:
            run_stage(stage, args, base_state, baseline, device)
        except Exception as error:
            folder.mkdir(parents=True, exist_ok=True)
            status(folder, "failed", current.get("epoch", 0), args.output, baseline,
                   error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
