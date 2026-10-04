"""Build a reviewed training cutout bank without altering the raw SAM output."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from cutout_contact_sheet import make_contact_sheet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "runs/p01/materials/sam3_1"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/p01/materials/sam3_1_curated"
DEFAULT_REJECTS = Path(__file__).with_name("p01_copy_paste_rejects.json")
DEFAULT_WEIGHTS = Path("/home/huace/下载/yolo11s.pt")


def matching_confidences(prediction, target):
    boxes = prediction.boxes.xyxy.cpu().numpy()
    classes = prediction.boxes.cls.cpu().numpy().astype(int)
    scores = prediction.boxes.conf.cpu().numpy()
    target = np.asarray(target, dtype=float)
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], target[2:]) -
                              np.maximum(boxes[:, :2], target[:2])).prod(axis=1)
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    overlap = intersection / np.minimum(target_area, box_area).clip(min=1)
    person = max((float(score) for score, cls, cover in zip(scores, classes, overlap)
                  if cls == 0 and cover >= 0.3), default=0.0)
    vehicle = max((float(score) for score, cls, cover in zip(scores, classes, overlap)
                   if cls in (1, 2, 3, 5, 7) and cover >= 0.3), default=0.0)
    return person, vehicle


def person_cross_check(rows, data, weights):
    by_image = defaultdict(list)
    for row in rows:
        if row["class_id"] == 0:
            by_image[row["source_image"]].append(row)
    detector = YOLO(str(weights))
    quality = []
    for image, records in by_image.items():
        prediction = detector.predict(str(data / "images/train" / image), imgsz=1280,
                                      conf=0.01, verbose=False)[0]
        for row in records:
            person, vehicle = matching_confidences(prediction, row["tight_box"])
            x1, y1, x2, y2 = row["tight_box"]
            quality.append({"id": row["id"], "person_conf": person,
                            "vehicle_conf": vehicle,
                            "aspect_ratio": (x2 - x1) / max(y2 - y1, 1)})
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "dataset/yolo_multitask")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--rejects", type=Path, default=DEFAULT_REJECTS)
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    if raw == output:
        raise ValueError("Curated bank must differ from the raw SAM bank")
    rows = [json.loads(line) for line in (raw / "manifest.jsonl").read_text().splitlines() if line]
    exclusions = json.loads(args.rejects.read_text(encoding="utf-8"))
    qc_path = raw / "person_yolo_qc.json"
    if qc_path.exists():
        quality = json.loads(qc_path.read_text(encoding="utf-8"))
    else:
        quality = person_cross_check(rows, args.data.resolve(), args.weights)
        qc_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    qc = {item["id"]: item for item in quality}
    if set(qc) != {row["id"] for row in rows if row["class_id"] == 0}:
        raise ValueError("Person cross-check does not match the raw SAM bank")

    kept, rejected = [], Counter()
    for row in rows:
        identifier = row["id"]
        frame = int(Path(row["source_image"]).stem)
        x1, y1, x2, y2 = row["tight_box"]
        if row["class_id"] == 0:
            info = qc[identifier]
            reason = ("manual_review" if identifier in exclusions["person_ids"] or
                      any(start <= frame <= end for start, end in exclusions["person_frame_ranges"])
                      else "low_person_confirmation" if info["person_conf"] < 0.03
                      else "person_shape" if info["aspect_ratio"] > 0.9 or y2 - y1 < 32
                      else "low_sam_score" if row["sam_score"] < 0.7 else None)
        else:
            reason = ("manual_review" if identifier in exclusions["vehicle_ids"]
                      else "wide_or_grouped_vehicle" if (x2 - x1) / max(y2 - y1, 1) > 2.0
                      else "low_sam_score" if row["sam_score"] < 0.8 else None)
        if reason:
            rejected[f"{['person', 'vehicle'][row['class_id']]}:{reason}"] += 1
            continue
        kept.append(row)
        target = output / row["crop"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw / row["crop"], target)

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    make_contact_sheet(kept, output)
    counts = Counter("person" if row["class_id"] == 0 else "vehicle" for row in kept)
    summary = {"source_split": "train", "raw_bank": str(raw),
               "cross_check_weights": str(args.weights), "person_conf_min": 0.03,
               "person_max_aspect_ratio": 0.9, "person_min_height": 32,
               "vehicle_max_aspect_ratio": 2.0,
               "manual_rejects": str(args.rejects.resolve()),
               "counts": dict(counts), "total_cutouts": len(kept),
               "rejected": dict(rejected)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
