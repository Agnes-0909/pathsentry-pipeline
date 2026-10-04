"""Benchmark model plus NMS latency and peak CUDA memory on fixed val images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression

from multitask import DEFAULT_DATA, MultiTaskDataset, MultiTaskYOLO11s


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        raise ValueError("samples must be positive and warmup nonnegative")
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    size = (state["config"]["height"], state["config"]["width"])
    dataset = MultiTaskDataset(args.data, "val", size)
    indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int)
    images = [dataset[int(index)]["img"].unsqueeze(0).to(device).float() / 255 for index in indices]
    model = MultiTaskYOLO11s(p2_detect=state["config"].get("p2_detect", False)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    def infer(image):
        det, seg = model(image)
        non_max_suppression(det, conf_thres=0.25, iou_thres=0.7, nc=2, max_det=300)
        _ = seg.sigmoid() > 0.5

    for i in range(args.warmup):
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
    result = {"checkpoint": str(args.checkpoint), "input_size": [size[1], size[0]],
              "batch": 1, "samples": len(images), "mean_ms": float(np.mean(timings)),
              "p95_ms": float(np.percentile(timings, 95)),
              "peak_cuda_mib": (torch.cuda.max_memory_allocated(device) / 1024**2
                                if device.type == "cuda" else None)}
    message = json.dumps(result, indent=2)
    print(message)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(message + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
