"""Export the curated LabelMe splits to Ultralytics YOLO datasets.

The LabelMe JSON files remain the source annotations.  This exporter creates
hard-linked images and derived YOLO ``.txt`` labels for detection and
segmentation training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


SPLITS = ("train", "val", "test")
DETECTION_NAMES = ("person", "vehicle")
DRIVABLE_NAMES = ("drivable_area",)
SEGMENTATION_NAMES = ("drivable_area", "person", "vehicle")


def _normalise(value: float, limit: float) -> float:
    return min(1.0, max(0.0, value / limit))


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _box_points(points: list[list[float]], width: int, height: int) -> list[tuple[float, float]]:
    if len(points) != 2:
        raise ValueError(f"rectangle must have two points, got {len(points)}")
    (x0, y0), (x1, y1) = points
    left, right = sorted((_normalise(float(x0), width), _normalise(float(x1), width)))
    top, bottom = sorted((_normalise(float(y0), height), _normalise(float(y1), height)))
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _polygon_points(points: list[list[float]], width: int, height: int) -> list[tuple[float, float]]:
    if len(points) < 3:
        raise ValueError(f"polygon must have at least three points, got {len(points)}")
    return [
        (_normalise(float(x), width), _normalise(float(y), height))
        for x, y in points
    ]


def _detection_line(shape: dict, width: int, height: int, class_ids: dict[str, int]) -> str:
    points = shape["points"]
    if shape.get("shape_type") != "rectangle" or len(points) != 2:
        raise ValueError("detection export expects rectangle shapes")
    (x0, y0), (x1, y1) = points
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    left = max(0.0, min(width, left))
    right = max(0.0, min(width, right))
    top = max(0.0, min(height, top))
    bottom = max(0.0, min(height, bottom))
    if right <= left or bottom <= top:
        raise ValueError("rectangle has zero area")
    cx = _normalise((left + right) / 2.0, width)
    cy = _normalise((top + bottom) / 2.0, height)
    box_w = (right - left) / width
    box_h = (bottom - top) / height
    return f"{class_ids[shape['label']]} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"


def _segmentation_line(shape: dict, width: int, height: int, class_ids: dict[str, int]) -> str:
    if shape.get("shape_type") == "polygon":
        points = _polygon_points(shape["points"], width, height)
    elif shape.get("shape_type") == "rectangle":
        # YOLO-seg requires a polygon.  A box is represented by its four corners.
        points = _box_points(shape["points"], width, height)
    else:
        raise ValueError(f"unsupported shape type: {shape.get('shape_type')}")
    values = " ".join(f"{value:.6f}" for point in points for value in point)
    return f"{class_ids[shape['label']]} {values}"


def _write_yaml(path: Path, names: tuple[str, ...]) -> None:
    path.write_text(
        f"path: {json.dumps(str(path.parent.resolve()), ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {json.dumps(list(names), ensure_ascii=False)}\n",
        encoding="utf-8",
    )


def export_multitask(source: Path, destination: Path, *, replace: bool) -> dict[str, int]:
    """Export paired detection labels and semantic drivable-area masks.

    This layout is intended for a custom shared-backbone/two-head trainer:
    ``labels_det`` supervises the detection head and ``masks_seg`` supervises
    a binary semantic segmentation head.  It is deliberately separate from
    the standard Ultralytics instance-segmentation export.
    """
    class_ids = {name: index for index, name in enumerate(DETECTION_NAMES)}
    if destination.exists():
        if not replace:
            raise FileExistsError(f"{destination} exists; use --replace to rebuild it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    counts = {"images": 0, "detection_objects": 0, "drivable_polygons": 0}
    manifest_rows: list[tuple[str, str, str, str, str]] = []
    for split in SPLITS:
        image_dir = destination / "images" / split
        label_dir = destination / "labels_det" / split
        mask_dir = destination / "masks_seg" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for annotation_path in sorted((source / split / "left").glob("*.json")):
            record = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_path = annotation_path.with_suffix(".jpg")
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            if record.get("imagePath") != image_path.name:
                raise ValueError(f"{annotation_path}: imagePath does not match image name")
            width, height = int(record["imageWidth"]), int(record["imageHeight"])
            detection_lines: list[str] = []
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            polygon_count = 0
            for shape in record.get("shapes", []):
                label = shape.get("label")
                if shape.get("shape_type") == "rectangle" and label in class_ids:
                    detection_lines.append(_detection_line(shape, width, height, class_ids))
                elif shape.get("shape_type") == "polygon" and label == "drivable_area":
                    points = [(round(float(x)), round(float(y))) for x, y in shape["points"]]
                    if len(points) >= 3:
                        draw.polygon(points, fill=1)
                        polygon_count += 1
            target_image = image_dir / image_path.name
            target_label = label_dir / annotation_path.with_suffix(".txt").name
            target_mask = mask_dir / annotation_path.with_suffix(".png").name
            _link_or_copy(image_path, target_image)
            target_label.write_text(
                "\n".join(detection_lines) + ("\n" if detection_lines else ""),
                encoding="utf-8",
            )
            mask.save(target_mask)
            manifest_rows.append(
                (split, image_path.name, f"labels_det/{split}/{target_label.name}",
                 f"masks_seg/{split}/{target_mask.name}", annotation_path.name)
            )
            counts["images"] += 1
            counts["detection_objects"] += len(detection_lines)
            counts["drivable_polygons"] += polygon_count
    with (destination / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "image", "detection_label", "segmentation_mask", "labelme_json"))
        writer.writerows(manifest_rows)
    (destination / "dataset.yaml").write_text(
        "# Custom two-head dataset: detection labels + semantic drivable-area masks\n"
        f"path: {json.dumps(str(destination.resolve()), ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "det_labels: labels_det\n"
        "seg_masks: masks_seg\n"
        "det_names: [person, vehicle]\n"
        "seg_names: [background, drivable_area]\n",
        encoding="utf-8",
    )
    return counts


def export(source: Path, destination: Path, *, mode: str, replace: bool) -> dict[str, int]:
    if mode == "detect":
        names = DETECTION_NAMES
    elif mode == "drivable_seg":
        names = DRIVABLE_NAMES
    else:
        names = SEGMENTATION_NAMES
    class_ids = {name: index for index, name in enumerate(names)}
    if destination.exists():
        if not replace:
            raise FileExistsError(f"{destination} exists; use --replace to rebuild it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    counts = {"images": 0, "labels": 0, "objects": 0}
    for split in SPLITS:
        image_dir = destination / "images" / split
        label_dir = destination / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for annotation_path in sorted((source / split / "left").glob("*.json")):
            record = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_path = annotation_path.with_suffix(".jpg")
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            if record.get("imagePath") != image_path.name:
                raise ValueError(f"{annotation_path}: imagePath does not match image name")
            width, height = int(record["imageWidth"]), int(record["imageHeight"])
            shapes = record.get("shapes", [])
            lines: list[str] = []
            for shape in shapes:
                label = shape.get("label")
                if mode == "detect":
                    if label not in class_ids:
                        continue
                    lines.append(_detection_line(shape, width, height, class_ids))
                else:
                    if label not in class_ids:
                        continue
                    lines.append(_segmentation_line(shape, width, height, class_ids))
            target_image = image_dir / image_path.name
            target_label = label_dir / annotation_path.with_suffix(".txt").name
            _link_or_copy(image_path, target_image)
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts["images"] += 1
            counts["labels"] += 1
            counts["objects"] += len(lines)
    _write_yaml(destination / "data.yaml", names)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = Path(__file__).resolve().parents[1] / "labeled_data" / "labelme_check"
    parser.add_argument(
        "--source",
        type=Path,
        default=data_root / "splits",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root,
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    for mode in ("detect", "drivable_seg", "seg"):
        output = args.output_root / f"yolo_{mode}"
        counts = export(args.source, output, mode=mode, replace=args.replace)
        print(mode, counts, output)
    multitask_output = args.output_root / "yolo_multitask"
    counts = export_multitask(args.source, multitask_output, replace=args.replace)
    print("multitask", counts, multitask_output)


if __name__ == "__main__":
    main()
