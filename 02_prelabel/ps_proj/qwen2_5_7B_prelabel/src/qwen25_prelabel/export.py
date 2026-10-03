from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .schema import Detection, ImageAnnotation

PALETTE = {
    "person": (231, 76, 60),
    "vehicle": (52, 152, 219),
    "other_obstacle": (241, 196, 15),
}


def yolo_line(detection: Detection, *, width: int, height: int) -> str:
    x0, y0, x1, y1 = detection.bbox
    box_width = x1 - x0
    box_height = y1 - y0
    x_center = x0 + box_width / 2.0
    y_center = y0 + box_height / 2.0
    return (
        f"{detection.class_id} "
        f"{x_center / width:.6f} "
        f"{y_center / height:.6f} "
        f"{box_width / width:.6f} "
        f"{box_height / height:.6f}"
    )


def write_annotation_outputs(
    annotation: ImageAnnotation,
    output_dir: Path,
    *,
    write_yolo: bool,
    write_overlay: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples" / annotation.image.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    payload = annotation.to_dict()
    _write_json(sample_dir / "manifest.json", payload)
    with (output_dir / "annotations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if write_yolo:
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        lines = [yolo_line(detection, width=annotation.width, height=annotation.height) for detection in annotation.detections]
        (labels_dir / f"{annotation.image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if write_overlay:
        render_overlay(annotation).save(sample_dir / "overlay.png")


def render_overlay(annotation: ImageAnnotation) -> Image.Image:
    with Image.open(annotation.image) as raw_image:
        image = raw_image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for detection in annotation.detections:
        color = PALETTE[detection.label]
        x0, y0, x1, y1 = detection.bbox
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)
        label = f"{detection.label} {detection.confidence:.2f}"
        bbox = draw.textbbox((x0, max(0, y0 - 16)), label)
        draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=(0, 0, 0, 180))
        draw.text((bbox[0], bbox[1]), label, fill=(255, 255, 255, 255))
    return image


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
