from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from .engine import Region, SegmentResult

PALETTE = [
    (233, 82, 71),
    (77, 144, 254),
    (56, 172, 120),
    (247, 183, 49),
    (168, 85, 247),
    (20, 184, 166),
    (244, 114, 182),
    (245, 158, 11),
]


def parse_prompts(text: str) -> list[str]:
    prompts: list[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        for chunk in line.split(","):
            prompt = chunk.strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def mask_to_polygons(mask: np.ndarray) -> list[list[list[int]]]:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError("mask must be 2D")
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[int]]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        points = contour.reshape(-1, 2)
        polygons.append([[int(x), int(y)] for x, y in points.tolist()])
    return polygons


def merge_region_masks(regions: Iterable["Region"]) -> np.ndarray:
    merged: np.ndarray | None = None
    for region in regions:
        mask = np.asarray(region.mask, dtype=bool)
        merged = mask if merged is None else np.logical_or(merged, mask)
    if merged is None:
        return np.zeros((0, 0), dtype=bool)
    return merged


def clean_drivable_mask(
    mask: np.ndarray,
    *,
    close_kernel_size: int = 41,
    min_component_area_ratio: float = 0.00025,
) -> np.ndarray:
    """Stabilize road masks across markings, reflections, and surface colors.

    SAM can split one road into several regions when a painted marking or a
    strong reflection crosses it.  Work on the binary mask only: no RGB/color
    heuristic is used here.  Small disconnected predictions are removed, and
    each remaining component is filled across narrow internal gaps row by row.
    """
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError("mask must be 2D")
    if not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)

    kernel_size = max(1, int(close_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    height, width = closed.shape
    min_area = max(64, int(height * width * max(0.0, min_component_area_ratio)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        return closed.astype(bool)

    kept = np.zeros_like(closed, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area and label != largest_label:
            continue
        component = labels == label
        ys, xs = np.where(component)
        if len(xs) == 0:
            continue
        # Fill gaps inside this component without joining separate road-side
        # components or extending the mask beyond its predicted boundary.
        for y in range(int(ys.min()), int(ys.max()) + 1):
            row_xs = np.flatnonzero(component[y])
            if len(row_xs) >= 2:
                kept[y, int(row_xs[0]) : int(row_xs[-1]) + 1] = True
        kept[component] = True
    return kept


def render_overlay(image: Image.Image, regions: Iterable["Region"], alpha: float = 0.45) -> Image.Image:
    base = np.array(image.convert("RGBA"), dtype=np.float32)
    for index, region in enumerate(regions):
        color = np.array(PALETTE[index % len(PALETTE)] + (255,), dtype=np.float32)
        mask = np.asarray(region.mask, dtype=bool)
        base[mask, :3] = base[mask, :3] * (1.0 - alpha) + color[:3] * alpha
        base[mask, 3] = 255.0

    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGBA")
    draw = ImageDraw.Draw(overlay)
    for index, region in enumerate(regions):
        color = PALETTE[index % len(PALETTE)]
        x0, y0, x1, y1 = region.box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{region.prompt} {region.score:.2f}"
        tx = max(0, int(x0))
        ty = max(0, int(y0) - 18)
        bbox = draw.textbbox((tx, ty), label)
        draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=(0, 0, 0, 160))
        draw.text((tx + 4, ty + 1), label, fill=(255, 255, 255, 255))
    return overlay


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest(image_name: str, result: "SegmentResult", mask_files: list[str]) -> dict:
    return {
        "image_name": image_name,
        "width": result.width,
        "height": result.height,
        "region_count": len(result.regions),
        "regions": [
            {
                "prompt": region.prompt,
                "score": region.score,
                "box": [float(v) for v in region.box],
                "polygons": region.polygons,
                "mask_file": mask_files[index],
            }
            for index, region in enumerate(result.regions)
        ],
    }
