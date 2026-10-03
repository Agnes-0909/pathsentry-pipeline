from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen25_prelabel.engine import ModelCompatibilityError, is_vision_language_model, load_model_config


class EngineTest(unittest.TestCase):
    def test_is_vision_language_model_rejects_text_only_qwen2_config(self) -> None:
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
        }

        self.assertFalse(is_vision_language_model(config))

    def test_is_vision_language_model_accepts_qwen2_5_vl_config(self) -> None:
        config = {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
        }

        self.assertTrue(is_vision_language_model(config))

    def test_load_model_config_raises_for_non_vl_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ModelCompatibilityError, "vision-language"):
                load_model_config(model_dir, require_vision=True)


if __name__ == "__main__":
    unittest.main()
