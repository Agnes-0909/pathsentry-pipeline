from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CLASS_TO_ID = {
    "person": 0,
    "vehicle": 1,
    "other_obstacle": 2,
}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}

LABEL_ALIASES = {
    "pedestrian": "person",
    "people": "person",
    "human": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "van": "vehicle",
    "motorcycle": "vehicle",
    "bike": "vehicle",
    "bicycle": "vehicle",
    "cyclist": "vehicle",
    "obstacle": "other_obstacle",
    "barrier": "other_obstacle",
    "cone": "other_obstacle",
    "traffic_cone": "other_obstacle",
    "debris": "other_obstacle",
    "animal": "other_obstacle",
}


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    class_id: int
    bbox: list[float]
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ImageAnnotation:
    image: Path
    width: int
    height: int
    detections: list[Detection]
    raw_response: str = ""
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "image": str(self.image),
            "width": self.width,
            "height": self.height,
            "detections": [detection.to_dict() for detection in self.detections],
            "raw_response": self.raw_response,
        }
        if self.parse_error:
            payload["parse_error"] = self.parse_error
        return payload


def normalize_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = LABEL_ALIASES.get(normalized, normalized)
    if normalized in CLASS_TO_ID:
        return normalized
    return None


def to_float_list(values: object) -> list[float] | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            return None
    return result


def normalize_bbox(values: list[float], width: int, height: int) -> list[float]:
    if all(0.0 <= value <= 1.0 for value in values):
        x0, y0, x1, y1 = values
        values = [x0 * width, y0 * height, x1 * width, y1 * height]
    x0 = min(max(values[0], 0.0), float(width))
    y0 = min(max(values[1], 0.0), float(height))
    x1 = min(max(values[2], 0.0), float(width))
    y1 = min(max(values[3], 0.0), float(height))
    return [x0, y0, x1, y1]


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
