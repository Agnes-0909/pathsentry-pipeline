"""Train-only Copy-Paste for detection boxes and drivable-area masks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


class CopyPasteBank:
    def __init__(self, root: str | Path, probability: float = 0.3, max_objects: int = 2):
        if not 0 <= probability <= 1 or max_objects < 1:
            raise ValueError("Invalid Copy-Paste probability or max_objects")
        self.root = Path(root).resolve()
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        self.by_class = {class_id: [row for row in records if row["class_id"] == class_id]
                         for class_id in (0, 1)}
        self.by_clip = {}
        for class_id, class_records in self.by_class.items():
            clips = defaultdict(list)
            for row in class_records:
                clips[int(Path(row["source_image"]).stem) // 100].append(row)
            self.by_clip[class_id] = list(clips.values())
        if not any(self.by_class.values()):
            raise ValueError(f"No cutouts in {manifest}")
        for row in records:
            if not (self.root / row["crop"]).is_file():
                raise FileNotFoundError(self.root / row["crop"])
        self.probability = probability
        self.max_objects = max_objects

    @staticmethod
    def _overlap(candidate, labels, width, height):
        if not len(labels):
            return False
        cx, cy, bw, bh = (labels[:, 1:] * [width, height, width, height]).T
        original = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), axis=1)
        x1 = np.maximum(candidate[0], original[:, 0])
        y1 = np.maximum(candidate[1], original[:, 1])
        x2 = np.minimum(candidate[2], original[:, 2])
        y2 = np.minimum(candidate[3], original[:, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
        existing_area = bw * bh
        return bool(np.any(intersection / np.minimum(candidate_area, existing_area).clip(min=1) > 0.1))

    def apply(self, image: np.ndarray, mask: np.ndarray, labels: np.ndarray, source_image: str):
        if np.random.random() >= self.probability:
            return image, mask, labels
        height, width = mask.shape
        road = (mask == 1).astype(np.uint8)
        road_nearby = cv2.dilate(road, np.ones((31, 31), np.uint8))
        image, mask, labels = image.copy(), mask.copy(), labels.copy()
        object_count = np.random.randint(1, self.max_objects + 1)
        for _ in range(object_count):
            class_id = 0 if self.by_class[0] and (not self.by_class[1] or np.random.random() < 0.8) else 1
            if not self.by_class[class_id]:
                continue
            for _ in range(12):
                clips = self.by_clip[class_id]
                clip = clips[np.random.randint(len(clips))]
                record = clip[np.random.randint(len(clip))]
                if record["source_image"] == source_image:
                    continue
                crop = cv2.imread(str(self.root / record["crop"]), cv2.IMREAD_UNCHANGED)
                if crop is None or crop.ndim != 3 or crop.shape[2] != 4:
                    raise ValueError(f"Invalid RGBA cutout: {record['crop']}")
                source_w, source_h = record["image_size"]
                source_bottom = record["tight_box"][3] / source_h
                target_bottom = float(np.clip(source_bottom + np.random.uniform(-0.07, 0.07), 0.08, 0.98))
                scale = (width / source_w) * np.clip(target_bottom / max(source_bottom, 0.1), 0.7, 1.3)
                scale *= np.random.uniform(0.85, 1.15)
                new_w, new_h = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
                if new_w >= width or new_h >= height:
                    continue
                center_x = record["tight_box"][0] / source_w + record["tight_box"][2] / source_w
                center_x = float(np.clip(center_x / 2 + np.random.uniform(-0.18, 0.18), 0.01, 0.99))
                left = round(center_x * width - new_w / 2)
                top = round(target_bottom * height - new_h)
                if left < 0 or top < 0 or left + new_w > width or top + new_h > height:
                    continue
                foot_x = min(width - 1, max(0, left + new_w // 2))
                foot_y = min(height - 1, max(0, top + new_h))
                if class_id == 1 and not road[foot_y, foot_x]:
                    continue
                if class_id == 0 and not road_nearby[foot_y, foot_x]:
                    continue
                candidate = (left, top, left + new_w, top + new_h)
                if self._overlap(candidate, labels, width, height):
                    continue
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                alpha = crop[:, :, 3:4].astype(np.float32) / 255
                if (alpha > 0.5).sum() < 32:
                    continue
                roi = image[top:top + new_h, left:left + new_w]
                roi[:] = (roi * (1 - alpha) + crop[:, :, :3] * alpha).astype(np.uint8)
                visible = alpha[:, :, 0] > 0.5
                mask[top:top + new_h, left:left + new_w][visible] = 0
                ys, xs = np.nonzero(visible)
                x1, x2 = left + int(xs.min()), left + int(xs.max()) + 1
                y1, y2 = top + int(ys.min()), top + int(ys.max()) + 1
                added = np.array([[class_id, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                                   (x2 - x1) / width, (y2 - y1) / height]], dtype=np.float32)
                labels = np.concatenate((labels, added), axis=0)
                break
        return image, mask, labels
