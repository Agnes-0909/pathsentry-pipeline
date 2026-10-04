"""Layer-wise detection sensitivity scan starting from the H2 architecture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp
from thop import profile
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PRUNING_ROOT = ROOT / "04_pruning"
sys.path.insert(0, str(ROOT / "03_training/scripts"))
sys.path.insert(0, str(PRUNING_ROOT))
from checkpoint_compat import register_pickle_aliases

register_pickle_aliases()
from multitask import MultiTaskDataset, collate_batch, evaluate_model
from architectures import FeatureFusionHead  # Required for trusted checkpoint unpickling.

SOURCE = PRUNING_ROOT / "experiments/p02_h2_p3_output/finetuned_best.pt"
DEFAULT_OUTPUT = PRUNING_ROOT / "sensitivity"
CANDIDATES = (
    "detector.model.2.cv2.conv", "detector.model.3.conv", "detector.model.4.cv2.conv",
    "detector.model.5.conv", "detector.model.6.cv2.conv", "detector.model.6.m.0.cv3.conv",
    "detector.model.7.conv", "detector.model.8.cv2.conv", "detector.model.8.m.0.cv3.conv",
    "detector.model.9.cv2.conv", "detector.model.10.cv2.conv",
    "detector.model.10.m.0.ffn.1.conv", "detector.model.13.cv2.conv",
    "detector.model.16.cv2.conv", "detector.model.17.conv", "detector.model.19.cv2.conv",
    "detector.model.20.conv", "detector.model.22.cv2.conv",
    "detector.model.22.m.0.cv3.conv",
)


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stats(model, size, device):
    model.eval()
    sample = torch.zeros(1, 3, *size, device=device)
    with torch.inference_mode():
        macs, params = profile(model, inputs=(sample,), verbose=False)
        detection, segmentation = model(sample)
    for module in model.modules():
        module._buffers.pop("total_ops", None)
        module._buffers.pop("total_params", None)
    return {"parameters": int(sum(p.numel() for p in model.parameters())),
            "gmacs": macs / 1e9, "detection_shape": list(detection[0].shape),
            "segmentation_shape": list(segmentation.shape)}


def prune_one(model, name, device, channels):
    layer = dict(model.named_modules()).get(name)
    if layer is None or not isinstance(layer, torch.nn.Conv2d):
        raise ValueError(f"Not a convolution layer: {name}")
    if layer.groups != 1 or layer.out_channels <= channels + 16:
        raise ValueError(f"Layer cannot safely remove {channels} channels: {name}")
    importance = layer.weight.detach().abs().sum(dim=(1, 2, 3))
    indices = torch.argsort(importance)[:channels].tolist()
    graph = tp.DependencyGraph().build_dependency(
        model, torch.zeros(1, 3, 64, 64, device=device), verbose=False)
    group = graph.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
    if not graph.check_pruning_group(group):
        raise RuntimeError(f"Invalid dependency group: {name}")
    group.prune()
    return {"removed_channels": channels, "indices": sorted(indices),
            "before": layer.out_channels + channels, "after": layer.out_channels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.channels < 1 or args.batch < 1:
        parser.error("channels and batch must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    size = (config["height"], config["width"])
    dataset = MultiTaskDataset(config["data"], "val", size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        collate_fn=collate_batch)
    base_model = checkpoint["model"].to(device).eval()
    # H2 was fine-tuned with the detector frozen; dependency tracing needs trainable parameters.
    for parameter in base_model.detector.parameters():
        parameter.requires_grad_(True)
    base_metrics = evaluate_model(base_model, loader, device)
    base_stats = stats(base_model, size, device)
    save_json(args.output / "baseline.json", {"source": str(args.source),
                                                "metrics": base_metrics, "stats": base_stats})
    results = []
    candidates = CANDIDATES[:args.limit] if args.limit else CANDIDATES
    for number, name in enumerate(candidates, 1):
        started = time.perf_counter()
        print(f"[{number}/{len(candidates)}] {name}", flush=True)
        model = torch.load(args.source, map_location="cpu", weights_only=False)["model"].to(device).eval()
        for parameter in model.detector.parameters():
            parameter.requires_grad_(True)
        try:
            layer = dict(model.named_modules()).get(name)
            if layer is None:
                raise ValueError("missing layer")
            before = layer.out_channels
            manifest = prune_one(model, name, device, args.channels)
            model_stats = stats(model, size, device)
            metrics = evaluate_model(model, loader, device)
            det = metrics["detection"]
            result = {"layer": name, "status": "ok", "seconds": time.perf_counter() - started,
                      "before": before, "manifest": manifest, "stats": model_stats,
                      "metrics": metrics,
                      "delta_map50_95": det["map50_95"] - base_metrics["detection"]["map50_95"],
                      "delta_person_ap": det["per_class"]["person"]["map50_95"] -
                      base_metrics["detection"]["per_class"]["person"]["map50_95"],
                      "delta_iou": metrics["segmentation"]["iou"] - base_metrics["segmentation"]["iou"]}
        except Exception as error:
            result = {"layer": name, "status": "failed", "seconds": time.perf_counter() - started,
                      "error": f"{type(error).__name__}: {error}"}
        results.append(result)
        save_json(args.output / "results.json", {"source": str(args.source),
                   "channels_removed_each": args.channels, "baseline": base_metrics,
                   "baseline_stats": base_stats, "results": results})
        print(json.dumps({"layer": name, "status": result["status"],
                          "delta_map50_95": result.get("delta_map50_95"),
                          "delta_person_ap": result.get("delta_person_ap"),
                          "delta_iou": result.get("delta_iou"),
                          "gmacs": result.get("stats", {}).get("gmacs")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
