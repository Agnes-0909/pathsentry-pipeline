from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from sam3_prelabel.cli import collect_images, output_root_for_image, resolve_prompts


class CliHelpersTest(unittest.TestCase):
    def test_collect_images_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "a" / "one.jpg").write_bytes(b"")
            (root / "a" / "two.txt").write_bytes(b"")
            (root / "b.png").write_bytes(b"")

            images = collect_images(root, recursive=True)
            self.assertEqual([p.name for p in images], ["one.jpg", "b.png"])

            out = output_root_for_image(root, root / "a" / "one.jpg", root / "outputs")
            self.assertEqual(out, root / "outputs" / "a" / "one")

    def test_resolve_prompts(self) -> None:
        args = argparse.Namespace(prompts="car, person\nbike", prompt_file=None)
        self.assertEqual(resolve_prompts(args), ["car", "person", "bike"])


if __name__ == "__main__":
    unittest.main()

