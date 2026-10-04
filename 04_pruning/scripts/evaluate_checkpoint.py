"""Evaluate a trusted full-module pruning checkpoint on val or test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_training/scripts"))
sys.path.insert(0, str(ROOT / "04_pruning"))
from checkpoint_compat import register_pickle_aliases

register_pickle_aliases()
from multitask import MultiTaskDataset, collate_batch, evaluate_model
from architectures import FeatureFusionHead  # Required to unpickle architecture checkpoints.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    device = torch.device(args.device)
    # Full-module pickle loading is intended only for locally generated, trusted checkpoints.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = checkpoint["model"].to(device)
    dataset = MultiTaskDataset(config["data"], args.split,
                               (config["height"], config["width"]))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        collate_fn=collate_batch)
    metrics = {"split": args.split, "checkpoint": str(args.checkpoint),
               **evaluate_model(model, loader, device)}
    payload = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
