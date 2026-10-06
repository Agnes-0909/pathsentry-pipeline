"""Export a trusted multitask checkpoint in the opset supported by RDK X5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_training/scripts"))
sys.path.insert(0, str(ROOT / "04_pruning"))
from checkpoint_compat import register_pickle_aliases

register_pickle_aliases()
from multitask import MultiTaskYOLO11s


class DualOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        detection, segmentation = self.model(images)
        if isinstance(detection, (tuple, list)):
            detection = detection[0]
        return detection, segmentation


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    if isinstance(model, dict):
        model_instance = MultiTaskYOLO11s()
        model_instance.load_state_dict(model)
        model = model_instance
    return model.eval(), checkpoint.get("config", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()
    model, _ = load_model(args.checkpoint)
    wrapper = DualOutputWrapper(model).eval()
    example = torch.zeros(1, 3, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, example, str(args.output), opset_version=11,
            input_names=["images"], output_names=["detection", "segmentation"],
            do_constant_folding=True, dynamo=False,
        )
    print(args.output)


if __name__ == "__main__":
    main()
