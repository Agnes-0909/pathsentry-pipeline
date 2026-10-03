from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

from qwen25_prelabel.cli import build_parser, main


class CliTest(unittest.TestCase):
    def test_cli_defaults_to_zero_confidence_filter_for_prelabeling(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.min_confidence, 0.0)

    def test_cli_dry_run_writes_sample_manifest_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            for index in range(5):
                Image.new("RGB", (16, 12), color=(index, index, index)).save(images / f"{index:03d}.jpg")

            output_dir = root / "out"
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        str(images),
                        "--output-dir",
                        str(output_dir),
                        "--sample-size",
                        "3",
                        "--seed",
                        "7",
                        "--dry-run",
                    ]
                )

            manifest = json.loads((output_dir / "sample_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["count"], 3)
            self.assertEqual([Path(path).name for path in manifest["images"]], ["002.jpg", "001.jpg", "003.jpg"])


if __name__ == "__main__":
    unittest.main()
