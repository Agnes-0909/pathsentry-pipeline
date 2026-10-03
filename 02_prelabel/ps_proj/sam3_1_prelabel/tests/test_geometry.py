from __future__ import annotations

import unittest

import numpy as np

from dataclasses import dataclass

from sam3_prelabel.geometry import clean_drivable_mask, mask_to_polygons, merge_region_masks, parse_prompts


class GeometryTest(unittest.TestCase):
    def test_parse_prompts(self) -> None:
        self.assertEqual(parse_prompts("car, person\nbike"), ["car", "person", "bike"])

    def test_mask_to_polygons(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 3:7] = 1
        polygons = mask_to_polygons(mask)
        self.assertTrue(polygons)
        self.assertTrue(all(len(poly) >= 3 for poly in polygons))

    def test_merge_region_masks(self) -> None:
        @dataclass
        class RegionLike:
            mask: np.ndarray

        a = RegionLike(mask=np.array([[0, 1], [0, 0]], dtype=bool))
        b = RegionLike(mask=np.array([[0, 0], [1, 0]], dtype=bool))
        merged = merge_region_masks([a, b])
        self.assertTrue(merged[0, 1])
        self.assertTrue(merged[1, 0])

    def test_clean_drivable_mask_bridges_marking_and_drops_small_fragments(self) -> None:
        mask = np.zeros((100, 120), dtype=bool)
        mask[45:100, 10:110] = True
        mask[55:60, 10:110] = False
        mask[10:12, 3:8] = True

        cleaned = clean_drivable_mask(mask, close_kernel_size=11, min_component_area_ratio=0.01)

        self.assertTrue(cleaned[57, 50])
        self.assertFalse(cleaned[10, 4])


if __name__ == "__main__":
    unittest.main()
