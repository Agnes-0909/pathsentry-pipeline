"""Validate raw-head DFL decoding against the existing decoded ONNX export."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from dfl_postprocess import decode_dfl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    decoded = ort.InferenceSession(str(args.decoded), providers=["CPUExecutionProvider"])
    raw = ort.InferenceSession(str(args.raw), providers=["CPUExecutionProvider"])
    paths = sorted(args.images.glob("*.jpg"))[:args.limit]
    cosines = []
    max_errors = []
    for path in paths:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (960, 832)).astype(np.float32) / 255.0
        tensor = image.transpose(2, 0, 1)[None]
        x_dec = decoded.run(None, {decoded.get_inputs()[0].name: tensor})[0]
        raw_out = raw.run(None, {raw.get_inputs()[0].name: tensor})
        x_raw = decode_dfl(raw_out[0::2][:3], raw_out[1::2][:3])
        a, b = x_dec.ravel(), x_raw.ravel()
        cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        max_errors.append(float(np.max(np.abs(a - b))))
    print(f"images={len(paths)} cosine_mean={np.mean(cosines):.9f} max_error={np.max(max_errors):.6f}")
    print(f"cosine_min={np.min(cosines):.9f} max_error_mean={np.mean(max_errors):.6f}")


if __name__ == "__main__":
    main()
