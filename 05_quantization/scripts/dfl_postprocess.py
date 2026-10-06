"""Decode official Ultralytics DFL raw-head outputs outside the BPU graph."""

from __future__ import annotations

import numpy as np


def make_anchors(height: int, width: int, stride: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(width, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    return np.stack((x.reshape(-1), y.reshape(-1)), axis=0)


def decode_dfl(cls_outputs, bbox_outputs, strides=(8, 16, 32), reg_max=16):
    """Return the same ``[B, 4+nc, N]`` tensor as Ultralytics Detect."""
    boxes = []
    scores = []
    anchors = []
    stride_values = []
    for cls_nhwc, bbox_nhwc, stride in zip(cls_outputs, bbox_outputs, strides):
        cls_nhwc = np.asarray(cls_nhwc, dtype=np.float32)
        bbox_nhwc = np.asarray(bbox_nhwc, dtype=np.float32)
        b, h, w, nc = cls_nhwc.shape
        if bbox_nhwc.shape != (b, h, w, 4 * reg_max):
            raise ValueError(f"Unexpected bbox shape {bbox_nhwc.shape}")
        # NHWC -> B,C,N, matching Detect.forward_head().
        cls = cls_nhwc.transpose(0, 3, 1, 2).reshape(b, nc, h * w)
        reg = bbox_nhwc.transpose(0, 3, 1, 2).reshape(b, 4, reg_max, h * w)
        reg = np.exp(reg - reg.max(axis=2, keepdims=True))
        reg /= reg.sum(axis=2, keepdims=True)
        reg *= np.arange(reg_max, dtype=np.float32)[None, None, :, None]
        reg = reg.sum(axis=2)
        boxes.append(reg)
        scores.append(cls)
        anchors.append(make_anchors(h, w, stride))
        stride_values.append(np.full((1, h * w), stride, dtype=np.float32))

    distances = np.concatenate(boxes, axis=2)
    scores = np.concatenate(scores, axis=2)
    anchor_points = np.concatenate(anchors, axis=1)[None]
    stride_tensor = np.concatenate(stride_values, axis=1)[None]
    lt, rb = np.split(distances, 2, axis=1)
    center = anchor_points
    xyxy = np.concatenate((center - lt, center + rb), axis=1) * stride_tensor
    xywh = np.concatenate(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), axis=1)
    scores = np.clip(scores, -80.0, 80.0)
    return np.concatenate((xywh, 1 / (1 + np.exp(-scores))), axis=1)
