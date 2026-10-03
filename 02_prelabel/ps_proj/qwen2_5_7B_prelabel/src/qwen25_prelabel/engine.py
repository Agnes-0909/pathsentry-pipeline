from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .parser import parse_annotation_response
from .prompt import build_obstacle_prompt
from .schema import ImageAnnotation


class ModelCompatibilityError(RuntimeError):
    pass


def load_model_config(model_dir: Path, *, require_vision: bool) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if require_vision and not is_vision_language_model(config):
        model_type = config.get("model_type", "unknown")
        architectures = ", ".join(config.get("architectures", [])) or "unknown"
        raise ModelCompatibilityError(
            "The configured model is not a supported vision-language checkpoint. "
            f"model_type={model_type}, architectures={architectures}. "
            "Use a Qwen2.5-VL model directory for image obstacle detection."
        )
    return config


def is_vision_language_model(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(str(item).lower() for item in config.get("architectures", []))
    haystack = f"{model_type} {architectures}"
    return "qwen2_5_vl" in haystack or "vision" in haystack or "conditionalgeneration" in haystack


class VllmObstacleDetector:
    def __init__(
        self,
        *,
        model_dir: Path,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = 8192,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        min_confidence: float = 0.35,
        min_box_area: float = 16.0,
        max_image_pixels: int | None = 401408,
        cpu_offload_gb: float = 0.0,
    ) -> None:
        self.model_dir = model_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.min_box_area = min_box_area
        self.max_image_pixels = max_image_pixels
        self.cpu_offload_gb = cpu_offload_gb
        self._llm: Any | None = None
        self._sampling_params: Any | None = None
        self._processor: Any | None = None
        load_model_config(model_dir, require_vision=True)

    def detect(self, image_path: Path) -> ImageAnnotation:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")

        llm, sampling_params, processor = self._load_runtime()
        prompt = self._build_prompt(processor, image_path)
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        raw_response = _extract_text(outputs)
        detections, parse_error = parse_annotation_response(
            raw_response,
            width=image.width,
            height=image.height,
            min_confidence=self.min_confidence,
            min_box_area=self.min_box_area,
        )
        return ImageAnnotation(
            image=image_path.resolve(),
            width=image.width,
            height=image.height,
            detections=detections,
            raw_response=raw_response,
            parse_error=parse_error,
        )

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        if self._llm is not None and self._sampling_params is not None and self._processor is not None:
            return self._llm, self._sampling_params, self._processor
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the vLLM runtime. Run through `conda run -n vllm` and "
                "verify that the CUDA libraries from the env are present on LD_LIBRARY_PATH. "
                "If libcudart.so.13 is missing, add "
                "`$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib`."
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": str(self.model_dir),
            "runner": "generate",
            "trust_remote_code": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": "bfloat16",
            "limit_mm_per_prompt": {"image": 1},
        }
        if self.max_image_pixels is not None:
            llm_kwargs["mm_processor_kwargs"] = {
                "max_pixels": int(self.max_image_pixels),
            }
        if self.cpu_offload_gb > 0:
            llm_kwargs["cpu_offload_gb"] = float(self.cpu_offload_gb)
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        self._processor = AutoProcessor.from_pretrained(str(self.model_dir), trust_remote_code=True)
        return self._llm, self._sampling_params, self._processor

    def _build_prompt(self, processor: Any, image_path: Path) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.resolve().as_uri()},
                    {"type": "text", "text": build_obstacle_prompt()},
                ],
            }
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_text(outputs: Any) -> str:
    if not outputs:
        return ""
    first = outputs[0]
    if not getattr(first, "outputs", None):
        return ""
    return str(first.outputs[0].text).strip()
