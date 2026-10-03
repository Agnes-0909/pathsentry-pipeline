from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .geometry import mask_to_polygons


@dataclass(slots=True)
class Region:
    prompt: str
    score: float
    box: list[float]
    mask: np.ndarray
    polygons: list[list[list[int]]]


@dataclass(slots=True)
class SegmentResult:
    width: int
    height: int
    regions: list[Region]


class Sam3Engine:
    def __init__(
        self,
        *,
        device: str,
        checkpoint_path: str | None = None,
        bpe_path: str | None = None,
        load_from_hf: bool = True,
        default_confidence_threshold: float = 0.5,
        compile_model: bool = False,
    ) -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.load_from_hf = load_from_hf
        self.default_confidence_threshold = default_confidence_threshold
        self.compile_model = compile_model
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self._model = build_sam3_image_model(
                    bpe_path=self.bpe_path,
                    device=self.device,
                    eval_mode=True,
                    checkpoint_path=self.checkpoint_path,
                    load_from_HF=self.load_from_hf,
                    enable_segmentation=True,
                    enable_inst_interactivity=False,
                    compile=self.compile_model,
                )
        return self._model

    def segment(
        self,
        image: Image.Image,
        prompts: list[str],
        *,
        confidence_threshold: float | None = None,
        max_regions: int | None = None,
    ) -> SegmentResult:
        model = self._load_model()
        threshold = confidence_threshold if confidence_threshold is not None else self.default_confidence_threshold
        processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=threshold,
        )
        base_state = processor.set_image(image)
        regions: list[Region] = []
        for prompt in prompts:
            state = {
                "original_height": base_state["original_height"],
                "original_width": base_state["original_width"],
                "backbone_out": dict(base_state["backbone_out"]),
            }
            result = processor.set_text_prompt(prompt, state)
            regions.extend(self._regions_from_state(prompt, result))

        regions.sort(key=lambda region: region.score, reverse=True)
        if max_regions is not None:
            regions = regions[:max_regions]
        return SegmentResult(width=image.width, height=image.height, regions=regions)

    def _regions_from_state(self, prompt: str, state: dict[str, Any]) -> list[Region]:
        masks = self._to_numpy(state.get("masks"))
        boxes = self._to_numpy(state.get("boxes"))
        scores = self._to_numpy(state.get("scores"))
        if masks.size == 0 or boxes.size == 0 or scores.size == 0:
            return []

        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
        elif masks.ndim == 3 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        elif masks.ndim != 3:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        count = min(len(masks), len(boxes), len(scores))
        regions: list[Region] = []
        for index in range(count):
            mask = np.asarray(masks[index], dtype=bool)
            polygons = mask_to_polygons(mask)
            regions.append(
                Region(
                    prompt=prompt,
                    score=float(scores[index]),
                    box=[float(v) for v in boxes[index].tolist()],
                    mask=mask,
                    polygons=polygons,
                )
            )
        return regions

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.asarray([])
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

