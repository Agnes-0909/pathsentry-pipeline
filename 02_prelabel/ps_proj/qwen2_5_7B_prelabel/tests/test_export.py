from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from qwen25_prelabel.export import write_annotation_outputs, yolo_line
from qwen25_prelabel.prompt import build_obstacle_prompt
from qwen25_prelabel.schema import Detection, ImageAnnotation


class ExportTest(unittest.TestCase):
    def test_yolo_line_converts_absolute_box(self) -> None:
        detection = Detection(
            label="vehicle",
            class_id=1,
            bbox=[100.0, 50.0, 300.0, 250.0],
            confidence=0.75,
        )

        self.assertEqual(yolo_line(detection, width=400, height=400), "1 0.500000 0.375000 0.500000 0.500000")

    def test_write_annotation_outputs_creates_manifest_jsonl_yolo_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "000001.jpg"
            Image.new("RGB", (400, 300), color=(20, 20, 20)).save(image_path)
            annotation = ImageAnnotation(
                image=image_path,
                width=400,
                height=300,
                detections=[
                    Detection(
                        label="person",
                        class_id=0,
                        bbox=[10.0, 20.0, 50.0, 120.0],
                        confidence=0.8,
                        reason="on drivable area",
                    )
                ],
                raw_response='{"detections":[]}',
            )

            write_annotation_outputs(annotation, root / "out", write_yolo=True, write_overlay=True)

            manifest = json.loads((root / "out" / "samples" / "000001" / "manifest.json").read_text())
            jsonl = (root / "out" / "annotations.jsonl").read_text().strip()
            yolo = (root / "out" / "labels" / "000001.txt").read_text().strip()

            self.assertEqual(manifest["detections"][0]["label"], "person")
            self.assertIn('"image":', jsonl)
            self.assertEqual(yolo, "0 0.075000 0.233333 0.100000 0.333333")
            self.assertTrue((root / "out" / "samples" / "000001" / "overlay.png").exists())

    def test_prompt_mentions_drivable_area_and_json(self) -> None:
        prompt = build_obstacle_prompt()

        self.assertIn("JSON", prompt)
        self.assertIn("drivable", prompt)
        self.assertIn("person", prompt)
        self.assertIn("vehicle", prompt)
        self.assertIn("other_obstacle", prompt)


if __name__ == "__main__":
    unittest.main()
