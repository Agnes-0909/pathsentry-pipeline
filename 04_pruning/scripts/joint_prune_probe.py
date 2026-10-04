"""Probe a conservative joint pruning set selected by layer-wise sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from architectures import FeatureFusionHead

SOURCE = PRUNING_ROOT / "experiments/p02_h2_p3_output/finetuned_best.pt"
LAYERS = ("detector.model.5.conv", "detector.model.6.cv2.conv", "detector.model.7.conv",
          "detector.model.8.cv2.conv", "detector.model.9.cv2.conv",
          "detector.model.10.cv2.conv", "detector.model.17.conv",
          "detector.model.19.cv2.conv", "detector.model.20.conv",
          "detector.model.22.cv2.conv")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stats(model, size, device):
    model.eval()
    sample = torch.zeros(1, 3, *size, device=device)
    with torch.inference_mode():
        macs, _ = profile(model, inputs=(sample,), verbose=False)
        det, seg = model(sample)
    for module in model.modules():
        module._buffers.pop("total_ops", None)
        module._buffers.pop("total_params", None)
    return {"parameters": sum(p.numel() for p in model.parameters()), "gmacs": macs / 1e9,
            "detection_shape": list(det[0].shape), "segmentation_shape": list(seg.shape)}


def prune_layer(model, name, channels, device):
    layer = dict(model.named_modules())[name]
    importance = layer.weight.detach().abs().sum(dim=(1, 2, 3))
    indices = torch.argsort(importance)[:channels].tolist()
    graph = tp.DependencyGraph().build_dependency(
        model, torch.zeros(1, 3, 64, 64, device=device), verbose=False)
    group = graph.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
    if not graph.check_pruning_group(group):
        raise RuntimeError(f"Invalid dependency group: {name}")
    group.prune()
    return {"layer": name, "before": layer.out_channels + channels,
            "after": layer.out_channels, "removed_indices": sorted(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    size = (config["height"], config["width"])
    dataset = MultiTaskDataset(config["data"], "val", size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda", collate_fn=collate_batch)
    model = checkpoint["model"].to(device).eval()
    for parameter in model.detector.parameters():
        parameter.requires_grad_(True)
    baseline = {"metrics": evaluate_model(model, loader, device), "stats": stats(model, size, device)}
    manifest = []
    for name in LAYERS:
        print(f"pruning {name}", flush=True)
        manifest.append(prune_layer(model, name, args.channels, device))
    candidate = {"metrics": evaluate_model(model, loader, device), "stats": stats(model, size, device)}
    save_json(args.output.with_suffix(".json"), {"source": str(args.source), "layers": LAYERS,
               "channels_each": args.channels, "manifest": manifest,
               "baseline": baseline, "candidate": candidate})
    torch.save({"model": model, "config": config, "manifest": manifest}, args.output)
    print(json.dumps({"baseline": baseline["stats"], "candidate": candidate["stats"],
                      "map50_95": candidate["metrics"]["detection"]["map50_95"],
                      "person_ap": candidate["metrics"]["detection"]["per_class"]["person"]["map50_95"],
                      "iou": candidate["metrics"]["segmentation"]["iou"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
