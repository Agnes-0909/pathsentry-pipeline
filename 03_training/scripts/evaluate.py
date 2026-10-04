"""Evaluate a YOLO11s multitask checkpoint on val or test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multitask import DEFAULT_DATA, MultiTaskDataset, MultiTaskYOLO11s, collate_batch, evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.001, help="Low threshold for AP calculation")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--limit", type=int, help="Limit images for a smoke test")
    parser.add_argument("--output", type=Path, help="Optional metrics JSON path")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    size = (state["config"]["height"], state["config"]["width"])
    dataset = MultiTaskDataset(args.data, args.split, size, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda", collate_fn=collate_batch)
    model = MultiTaskYOLO11s(p2_detect=state["config"].get("p2_detect", False)).to(device)
    model.load_state_dict(state["model"])
    result = {"split": args.split, "checkpoint": str(args.checkpoint),
              **evaluate_model(model, loader, device, conf=args.conf, iou=args.iou,
                               segmentation=state["config"].get("seg_weight", 1) > 0)}
    message = json.dumps(result, ensure_ascii=False, indent=2)
    print(message)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(message + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
