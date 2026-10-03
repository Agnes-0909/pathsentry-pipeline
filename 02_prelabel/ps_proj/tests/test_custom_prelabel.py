from __future__ import annotations

import unittest
from pathlib import Path

from custom_prelabel import build_labelme_record, labelme_output_path


class CustomPrelabelTest(unittest.TestCase):
    def test_labelme_output_path_uses_image_stem(self) -> None:
        output_dir = Path("/tmp/left_label")
        image_path = Path("/home/huace/new_storage/ps_datasets/custom/left/005054.jpg")

        self.assertEqual(labelme_output_path(image_path, output_dir), output_dir / "005054.json")

    def test_build_labelme_record_merges_sam_polygons_and_qwen_boxes(self) -> None:
        image_path = Path("/home/huace/new_storage/ps_datasets/custom/left/005054.jpg")
        record = build_labelme_record(
            image_path=image_path,
            width=1280,
            height=1088,
            drivable_polygons=[[[0, 100], [10, 100], [10, 120], [0, 120]]],
            detections=[
                {
                    "label": "person",
                    "bbox": [20.0, 30.0, 40.0, 50.0],
                    "confidence": 0.88,
                    "reason": "on road",
                }
            ],
        )

        self.assertEqual(record["imagePath"], str(image_path))
        self.assertIsNone(record["imageData"])
        self.assertEqual(record["imageHeight"], 1088)
        self.assertEqual(record["imageWidth"], 1280)
        self.assertEqual([shape["label"] for shape in record["shapes"]], ["drivable_area", "person"])
        self.assertEqual(record["shapes"][0]["shape_type"], "polygon")
        self.assertEqual(record["shapes"][1]["shape_type"], "rectangle")
        self.assertEqual(record["shapes"][1]["flags"]["source"], "qwen")


if __name__ == "__main__":
    unittest.main()
