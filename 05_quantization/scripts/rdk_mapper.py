"""RDK X5 mapper entry point.

This is the D-Robotics rdk_model_zoo mapper flow adapted only by filename;
the conversion logic is kept in sync with the official X5 sample.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-images", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cal-sample-num", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--optimize-level", default="O3")
    args = parser.parse_args()
    args.cal_images = args.cal_images.resolve()
    args.onnx = args.onnx.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.onnx.is_file() or not args.cal_images.is_dir():
        raise FileNotFoundError("ONNX or calibration image directory is missing")
    subprocess.run(["hb_mapper", "--version"], check=True)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) != 4 or not all(isinstance(x, int) for x in input_shape):
        raise ValueError(f"Expected static NCHW input, got {input_shape}")
    _, _, height, width = input_shape
    names = sorted(p for p in args.cal_images.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(names) < args.cal_sample_num:
        args.cal_sample_num = len(names)
    names = names[:args.cal_sample_num]
    if not names:
        raise ValueError("No calibration images found")
    work = args.output_dir / "mapper_workspace"
    if work.exists():
        shutil.rmtree(work)
    cal_dir = work / "calibration_data_float32"
    cal_dir.mkdir(parents=True)
    for index, path in enumerate(names):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read calibration image: {path}")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        # Keep calibration tensors in the original 0..255 RGB range.  The
        # mapper applies input_parameters.scale_value (1/255) exactly once.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))[None]
        image.tofile(cal_dir / f"{index:04d}.rgbchw")
    config = work / "config.yaml"
    config.write_text(f"""model_parameters:
  onnx_model: '{args.onnx}'
  march: 'bayes-e'
  layer_out_dump: False
  working_dir: '{work / 'bpu_model_output'}'
  output_model_file_prefix: '{args.onnx.stem}_bayese_960x832'
input_parameters:
  input_name: 'images'
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{cal_dir}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8
compiler_parameters:
  jobs: {args.jobs}
  compile_mode: 'latency'
  debug: True
  optimize_level: '{args.optimize_level}'
""", encoding="utf-8")
    logging.info("calibration files: %d", len(names))
    subprocess.run(["hb_mapper", "makertbin", "--config", str(config), "--model-type", "onnx"],
                   cwd=work, check=True)
    generated = work / "bpu_model_output" / f"{args.onnx.stem}_bayese_960x832.bin"
    if not generated.is_file():
        raise FileNotFoundError(f"Expected generated model not found: {generated}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, args.output_dir / generated.name)
    print(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
