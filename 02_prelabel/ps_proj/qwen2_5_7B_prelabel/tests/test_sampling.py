from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen25_prelabel.sampling import collect_images, sample_images


class SamplingTest(unittest.TestCase):
    def test_collect_images_filters_supported_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "b.jpg").write_bytes(b"")
            (root / "c.jpg").write_bytes(b"1")
            (root / "a.txt").write_bytes(b"")
            (root / "nested" / "d.PNG").write_bytes(b"2")

            images = collect_images(root, recursive=True)

            self.assertEqual([path.name for path in images], ["c.jpg", "d.PNG"])

    def test_sample_images_is_seeded_and_returns_all_when_sample_large(self) -> None:
        images = [Path(f"{idx:03d}.jpg") for idx in range(10)]

        sampled = sample_images(images, sample_size=4, seed=7)
        sampled_again = sample_images(images, sample_size=4, seed=7)
        all_images = sample_images(images, sample_size=999, seed=7)

        self.assertEqual([path.name for path in sampled], ["005.jpg", "002.jpg", "006.jpg", "009.jpg"])
        self.assertEqual(sampled, sampled_again)
        self.assertEqual(all_images, images)


if __name__ == "__main__":
    unittest.main()
