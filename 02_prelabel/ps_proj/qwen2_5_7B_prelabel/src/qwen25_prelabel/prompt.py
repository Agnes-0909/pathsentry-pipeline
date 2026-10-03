from __future__ import annotations


def build_obstacle_prompt() -> str:
    return """You are labeling obstacles for autonomous driving obstacle avoidance.

Detect only objects that are on, touching, or likely to enter the visible drivable road area and may block the ego vehicle path.

Use exactly these labels:
- person: pedestrians or people on or near the drivable area.
- vehicle: cars, trucks, buses, vans, motorcycles, bicycles, and riders occupying the drivable area.
- other_obstacle: cones, barriers, debris, animals, carts, roadblocks, fallen objects, or any physical obstacle on the drivable area.

Ignore sky, buildings, lane markings, traffic lights or signs, road texture, and objects far outside the drivable area.

Return JSON only. Do not include markdown or prose. The schema is:
{
  "detections": [
    {
      "label": "person|vehicle|other_obstacle",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.0,
      "reason": "short reason tied to drivable-area risk"
    }
  ]
}

Bounding boxes must use absolute image pixel coordinates."""
