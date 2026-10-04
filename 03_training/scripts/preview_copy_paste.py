"""Render before/after examples from the train-only Copy-Paste bank."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from copy_paste import CopyPasteBank
from multitask import DEFAULT_DATA


def draw_boxes(image, labels, original_count, title):
    height, width = image.shape[:2]
    image = image.copy()
    for index, (class_id, cx, cy, bw, bh) in enumerate(labels):
        x1 = round((cx - bw / 2) * width)
        y1 = round((cy - bh / 2) * height)
        x2 = round((cx + bw / 2) * width)
        y2 = round((cy + bh / 2) * height)
        color = (20, 30, 230) if index >= original_count else (20, 190, 30)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, "person" if class_id == 0 else "vehicle", (x1, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(image, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(image, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 1)
    return cv2.resize(image, (480, 408), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("runs/p01/materials/previews"))
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    data_config = args.data.resolve()
    import yaml
    root = Path(yaml.safe_load(data_config.read_text(encoding="utf-8"))["path"])
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "train"]
    bank = CopyPasteBank(args.bank, probability=1)
    chosen = []
    for index in np.random.permutation(len(rows)):
        row = rows[index]
        image = cv2.imread(str(root / "images/train" / row["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / row["segmentation_mask"]), cv2.IMREAD_UNCHANGED)
        lines = (root / row["detection_label"]).read_text(encoding="utf-8").splitlines()
        labels = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32).reshape(-1, 5)
        augmented, augmented_mask, new_labels = bank.apply(image, mask, labels, row["image"])
        if len(new_labels) == len(labels):
            continue
        if np.any((augmented_mask != mask) & (augmented_mask != 0)):
            raise ValueError(f"Unexpected mask class after augmentation: {row['image']}")
        before = draw_boxes(image, labels, len(labels), f"{row['image']} original")
        after = draw_boxes(augmented, new_labels, len(labels), f"+{len(new_labels)-len(labels)} pasted")
        chosen.append(np.concatenate((before, after), axis=1))
        if len(chosen) >= args.pairs:
            break
    if not chosen:
        raise RuntimeError("No Copy-Paste examples could be generated")
    args.output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(chosen), 6):
        sheet = np.concatenate(chosen[start:start + 6], axis=0)
        path = args.output / f"preview_{start // 6:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(path)


if __name__ == "__main__":
    main()
