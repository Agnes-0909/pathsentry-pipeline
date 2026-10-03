from __future__ import annotations

import json
from typing import Any

from .schema import CLASS_TO_ID, Detection, bbox_area, normalize_bbox, normalize_label, to_float_list


def parse_annotation_response(
    text: str,
    *,
    width: int,
    height: int,
    min_confidence: float,
    min_box_area: float,
) -> tuple[list[Detection], str | None]:
    try:
        payload = _load_json_payload(text)
    except ValueError as exc:
        return [], str(exc)

    items = payload.get("detections", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        return [], "JSON payload must contain a detections list"

    detections: list[Detection] = []
    for item in items:
        detection = _parse_detection(item, width, height, min_confidence, min_box_area)
        if detection is not None:
            detections.append(detection)
    return detections, None


def _load_json_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("No JSON payload found in empty response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        start = stripped.find("[")
        end = stripped.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object in model response: {exc}") from exc


def _parse_detection(
    item: object,
    width: int,
    height: int,
    min_confidence: float,
    min_box_area: float,
) -> Detection | None:
    if not isinstance(item, dict):
        return None

    label = normalize_label(item.get("label") or item.get("class"))
    if label is None:
        return None

    confidence = _to_float(item.get("confidence", item.get("score", 1.0)))
    if confidence is None or confidence < min_confidence:
        return None

    raw_bbox = item.get("bbox") or item.get("box")
    bbox_values = to_float_list(raw_bbox)
    if bbox_values is None:
        return None
    bbox = normalize_bbox(bbox_values, width, height)
    if bbox_area(bbox) < min_box_area:
        return None

    reason = item.get("reason", "")
    return Detection(
        label=label,
        class_id=CLASS_TO_ID[label],
        bbox=bbox,
        confidence=float(confidence),
        reason=reason if isinstance(reason, str) else "",
    )


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
