"""Compare a 64-channel segmentation head and a P3-output segmentation head."""

from __future__ import annotations

import argparse
import json
import math
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
sys.path.insert(0, str(PRUNING_ROOT / "scripts"))
from checkpoint_compat import register_pickle_aliases

register_pickle_aliases()
from multitask import (MultiTaskDataset, MultiTaskYOLO11s, collate_batch, evaluate_model,
                       segmentation_loss, to_device)
from architectures import FeatureFusionHead, transfer_p3, transfer_width64
from run_experiments import (BASE, OUTPUT, benchmark, load_slim, model_stats,
                             save_json, set_seed, visualize)

VARIANTS = {"h1": "p02_h1_width64", "h2": "p02_h2_p3_output"}
RECORD = PRUNING_ROOT / "ARCHITECTURE_EXPERIMENTS.md"


def create_variant(kind: str, base_state: dict, device: torch.device):
    model = MultiTaskYOLO11s().to(device)
    model.load_state_dict(base_state["model"])
    old = model.segmenter
    if kind == "h1":
        head = FeatureFusionHead((2, 16, 19, 22), (128, 128, 256, 512), 64).to(device)
        transfer = {"kept_channels": transfer_width64(old, head)}
    else:
        head = FeatureFusionHead((16, 19, 22), (128, 256, 512), 128,
                                 upsample_output=True).to(device)
        transfer_p3(old, head)
        transfer = {"source_lateral": [1, 2, 3], "source_refine": [0, 1],
                    "output_upsample": "bilinear x2"}
    model.segmenter = head
    for parameter in model.detector.parameters():
        parameter.requires_grad_(False)
    model.detector.eval()
    return model, transfer


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_record(baseline: dict):
    lines = ["# P02 分割头结构实验", "",
             "H1、H2 各自从 P01-B1 初始化，只改分割头并冻结检测网络。检测输入、训练数据、"
             "Copy-Paste、分割损失和验证集不变；速度是相同脚本、同一 GPU 会话中的配对测量中位数。", "",
             "| 模型 | 状态 | 最佳/已训练 epoch | 参数量 | GMACs | mAP50-95 | person AP | 分割 IoU | 延迟 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
             f"| B1 | 基线 | - | 10,003,415 | 31.89 | "
             f"{baseline['detection']['map50_95']:.4f} | "
             f"{baseline['detection']['per_class']['person']['map50_95']:.4f} | "
             f"{baseline['segmentation']['iou']:.4f} | {baseline['mean_ms']:.2f} |"]
    for kind, directory in VARIANTS.items():
        folder = OUTPUT / directory
        status = read_json(folder / "status.json") or {}
        stats = read_json(folder / "model_stats.json") or {}
        metrics = read_json(folder / "val_metrics.json") or {}
        speed = read_json(folder / "paired_benchmark.json") or {}
        det = metrics.get("detection", {})
        seg = metrics.get("segmentation", {})
        cells = [kind.upper(), status.get("state", "pending"),
                 str(status.get("best_epoch", status.get("epoch", "-"))),
                 f"{stats['parameters']:,}" if stats else "-",
                 f"{stats['gmacs']:.2f}" if stats else "-",
                 f"{det['map50_95']:.4f}" if det else "-",
                 f"{det['per_class']['person']['map50_95']:.4f}" if det else "-",
                 f"{seg['iou']:.4f}" if seg else "-",
                 f"{speed['candidate_median_ms']:.2f}" if speed else "-"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "H1：P2/P3/P4/P5 融合宽度 128→64，按训练后通道分数迁移权重。"
              "H2：移除 P2 lateral 和最高分辨率 refine，在 P3 生成 logits 后双线性上采样 2 倍。", "",
              "训练：仅更新分割头；batch 2、960×832、SAM 审核版 Copy-Paste 概率 0.3、"
              "AdamW 初始学习率 1e-4、最多 30 epoch、耐心 10；每轮以验证集分割 IoU 选权重。", "",
              "质量门槛沿用 P02：相对 B1，mAP50-95 下降 ≤0.005，person AP 下降 ≤0.003，"
              "分割 IoU 下降 ≤0.002。速度目标为同口径端到端延迟下降 ≥20%，未达到时如实记录。", "",
              "各组目录包含初始化权重、逐轮训练日志、最佳权重、验证指标、配对速度和固定样本图。"
              "本轮仅使用验证集，测试集不参与结构选择。", ""]
    RECORD.write_text("\n".join(lines), encoding="utf-8")


def set_status(folder: Path, baseline: dict, state: str, epoch: int, **extra):
    save_json(folder / "status.json", {"state": state, "epoch": epoch,
                                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), **extra})
    write_record(baseline)


def paired_benchmark(candidate, base_state, dataset, device):
    base = MultiTaskYOLO11s().to(device)
    base.load_state_dict(base_state["model"])
    base.eval()
    candidate.eval()
    reference = []
    tested = []
    for repeat in range(3):
        if repeat % 2:
            tested.append(benchmark(candidate, dataset, device))
            reference.append(benchmark(base, dataset, device))
        else:
            reference.append(benchmark(base, dataset, device))
            tested.append(benchmark(candidate, dataset, device))
    base_ms = float(np.median([item["mean_ms"] for item in reference]))
    candidate_ms = float(np.median([item["mean_ms"] for item in tested]))
    return {"baseline_median_ms": base_ms, "candidate_median_ms": candidate_ms,
            "speedup_percent": 100 * (1 - candidate_ms / base_ms),
            "baseline_runs": reference, "candidate_runs": tested}


def run_variant(kind: str, args, base_state: dict, baseline: dict, device: torch.device):
    folder = OUTPUT / VARIANTS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    config = base_state["config"]
    size = (config["height"], config["width"])
    train_data = MultiTaskDataset(config["data"], "train", size, augment=True,
                                  copy_paste_bank=config["copy_paste_bank"],
                                  copy_paste_prob=config["copy_paste_prob"])
    val_data = MultiTaskDataset(config["data"], "val", size)
    loader_args = {"batch_size": args.batch, "num_workers": args.workers,
                   "collate_fn": collate_batch, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    initial_path = folder / "initialized.pt"
    last_path = folder / "finetuned_last.pt"
    best_path = folder / "finetuned_best.pt"
    if initial_path.exists():
        model, _ = load_slim(initial_path, device)
        print(f"[{kind}] loading initialized model", flush=True)
    else:
        set_status(folder, baseline, "building", 0)
        model, transfer = create_variant(kind, base_state, device)
        stats = model_stats(model, size, device)
        save_json(folder / "architecture.json", {"variant": kind, "transfer": transfer,
                   "detector_frozen": True, "source": str(BASE),
                   "output_size": stats["segmentation_shape"][-2:]})
        torch.save({"model": model, "config": config, "variant": kind}, initial_path)
        stats["initialized_file_mib"] = initial_path.stat().st_size / 1024**2
        save_json(folder / "model_stats.json", stats)
        metrics = evaluate_model(model, val_loader, device)
        save_json(folder / "initialized_val_metrics.json", metrics)
        set_status(folder, baseline, "initialized", 0)
        print(f"[{kind}] initialized params={stats['parameters']:,} gmacs={stats['gmacs']:.2f} "
              f"iou={metrics['segmentation']['iou']:.4f}", flush=True)

    optimizer = torch.optim.AdamW(model.segmenter.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.lr * 0.01)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    first_epoch, best_iou, stale, best_epoch = 0, -math.inf, 0, 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        first_epoch = checkpoint["epoch"]
        best_iou = checkpoint["best_iou"]
        stale = checkpoint["stale"]
        best_epoch = checkpoint["best_epoch"]
        print(f"[{kind}] resuming at epoch {first_epoch + 1}", flush=True)

    completed_epoch = first_epoch
    if first_epoch < args.epochs and stale < args.patience:
        for epoch in range(first_epoch, args.epochs):
            model.train()
            model.detector.eval()
            total_seg_loss = 0.0
            start = time.perf_counter()
            for batch in train_loader:
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    _, logits = model(batch["img"])
                    seg_loss = segmentation_loss(logits, batch["mask"])
                    loss = config["seg_weight"] * seg_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{kind}: non-finite loss in epoch {epoch + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.segmenter.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                total_seg_loss += seg_loss.item() * len(batch["img"])
            scheduler.step()
            metrics = evaluate_model(model, val_loader, device)
            iou = metrics["segmentation"]["iou"]
            improved = iou > best_iou
            best_iou = max(best_iou, iou)
            stale = 0 if improved else stale + 1
            if improved:
                best_epoch = epoch + 1
            record = {"epoch": epoch + 1, "seconds": time.perf_counter() - start,
                      "seg_loss": total_seg_loss / len(train_data),
                      "lr": scheduler.get_last_lr()[0], "best_epoch": best_epoch,
                      "best_iou": best_iou, "stale": stale, **metrics}
            with (folder / "history.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch + 1, "best_iou": best_iou, "stale": stale,
                        "best_epoch": best_epoch}, last_path)
            if improved:
                torch.save({"model": model, "config": config, "variant": kind,
                            "epoch": epoch + 1, "selection_iou": iou}, best_path)
                save_json(folder / "val_metrics.json", metrics)
            completed_epoch = epoch + 1
            set_status(folder, baseline, "training", epoch + 1,
                       best_epoch=best_epoch, best_iou=best_iou, stale=stale)
            print(f"[{kind}] epoch={epoch + 1} seg_loss={record['seg_loss']:.5f} "
                  f"iou={iou:.5f} best={best_iou:.5f} seconds={record['seconds']:.1f}", flush=True)
            if stale >= args.patience:
                break

    model, checkpoint = load_slim(best_path, device)
    metrics = read_json(folder / "val_metrics.json")
    speed = paired_benchmark(model, base_state, val_data, device)
    save_json(folder / "paired_benchmark.json", speed)
    visualize(model, val_data, device, folder / "visuals")
    quality = (metrics["detection"]["map50_95"] >= baseline["detection"]["map50_95"] - .005
               and metrics["detection"]["per_class"]["person"]["map50_95"] >=
               baseline["detection"]["per_class"]["person"]["map50_95"] - .003
               and metrics["segmentation"]["iou"] >= baseline["segmentation"]["iou"] - .002)
    set_status(folder, baseline, "complete", completed_epoch,
               best_epoch=checkpoint["epoch"], quality_pass=quality,
               speed_20pct_pass=speed["speedup_percent"] >= 20)
    print(f"[{kind}] complete quality_pass={quality} speedup={speed['speedup_percent']:.1f}%",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_state = torch.load(BASE, map_location="cpu", weights_only=True)
    base_val = read_json(BASE.parent / "val_metrics.json")
    base_speed = read_json(BASE.parent / "benchmark.json")
    baseline = {**base_val, **base_speed}
    write_record(baseline)
    for kind in args.variants:
        folder = OUTPUT / VARIANTS[kind]
        current = read_json(folder / "status.json") or {}
        if current.get("state") == "complete":
            print(f"[{kind}] already complete", flush=True)
            continue
        try:
            run_variant(kind, args, base_state, baseline, device)
        except Exception as error:
            folder.mkdir(parents=True, exist_ok=True)
            previous = read_json(folder / "status.json") or {}
            set_status(folder, baseline, "failed", previous.get("epoch", 0),
                       error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
