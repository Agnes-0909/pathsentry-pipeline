from __future__ import annotations

import unittest

from qwen25_prelabel.parser import parse_annotation_response


class ParserTest(unittest.TestCase):
    def test_parse_annotation_response_extracts_and_filters_detections(self) -> None:
        text = """
        ```json
        {
          "detections": [
            {"label": "vehicle", "bbox": [100, 120, 260, 320], "confidence": 0.91, "reason": "on lane ahead"},
            {"label": "animal", "bbox": [10, 10, 20, 20], "confidence": 0.99},
            {"label": "person", "bbox": [-5, 20, 40, 80], "confidence": 0.80},
            {"label": "other_obstacle", "bbox": [50, 60, 55, 65], "confidence": 0.95},
            {"label": "vehicle", "bbox": [300, 120, 280, 320], "confidence": 0.95},
            {"label": "vehicle", "bbox": [400, 120, 460, 220], "confidence": 0.31}
          ]
        }
        ```
        """

        detections, error = parse_annotation_response(
            text,
            width=640,
            height=480,
            min_confidence=0.5,
            min_box_area=101.0,
        )

        self.assertIsNone(error)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].label, "vehicle")
        self.assertEqual(detections[0].class_id, 1)
        self.assertEqual(detections[0].bbox, [100.0, 120.0, 260.0, 320.0])
        self.assertEqual(detections[1].label, "person")
        self.assertEqual(detections[1].bbox, [0.0, 20.0, 40.0, 80.0])

    def test_parse_annotation_response_scales_normalized_boxes(self) -> None:
        text = '{"detections":[{"label":"car","bbox":[0.25,0.25,0.5,0.75],"confidence":0.9}]}'

        detections, error = parse_annotation_response(
            text,
            width=800,
            height=400,
            min_confidence=0.5,
            min_box_area=100.0,
        )

        self.assertIsNone(error)
        self.assertEqual(detections[0].bbox, [200.0, 100.0, 400.0, 300.0])

    def test_parse_annotation_response_reports_missing_json(self) -> None:
        detections, error = parse_annotation_response(
            "no json here",
            width=640,
            height=480,
            min_confidence=0.5,
            min_box_area=100.0,
        )

        self.assertEqual(detections, [])
        self.assertIn("JSON", error or "")


if __name__ == "__main__":
    unittest.main()
