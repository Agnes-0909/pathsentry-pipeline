"""Export the custom multitask model with an RDK-friendly raw detection head.

The detector emits per-scale class logits and DFL regression logits. DFL
decoding, sigmoid and NMS are intentionally kept outside the BPU graph.
"""

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
from export_opset11 import load_model


class RawDetect(nn.Module):
    """Official X5 ordering: cls NHWC, bbox NHWC for each detection scale."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.cv2 = head.cv2
        self.cv3 = head.cv3
        self.nl = head.nl
        self.i = head.i
        self.f = head.f

    def forward(self, features):
        outputs = []
        for i in range(self.nl):
            outputs.append(self.cv3[i](features[i]).permute(0, 2, 3, 1).contiguous())
            outputs.append(self.cv2[i](features[i]).permute(0, 2, 3, 1).contiguous())
        return tuple(outputs)


class RawHeadWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        head = model.detector.model[-1]
        model.detector.model[-1] = RawDetect(head)

    def forward(self, images):
        outputs = self.model(images)
        detection, segmentation = outputs
        return (*detection, segmentation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    model, _ = load_model(args.checkpoint)
    wrapper = RawHeadWrapper(model).eval()
    example = torch.zeros(1, 3, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example,
            str(args.output),
            opset_version=11,
            input_names=["images"],
            output_names=["cls_0", "bbox_0", "cls_1", "bbox_1", "cls_2", "bbox_2", "segmentation"],
            do_constant_folding=True,
            dynamo=False,
        )
    print(args.output)


if __name__ == "__main__":
    main()
