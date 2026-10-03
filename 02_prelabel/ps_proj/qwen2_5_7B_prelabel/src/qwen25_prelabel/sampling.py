from __future__ import annotations

import random
from pathlib import Path

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(path: Path, recursive: bool = False) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES and path.stat().st_size > 0 else []
    pattern = "**/*" if recursive else "*"
    return [
        image_path
        for image_path in sorted(path.glob(pattern))
        if image_path.is_file()
        and image_path.suffix.lower() in SUPPORTED_SUFFIXES
        and image_path.stat().st_size > 0
    ]


def sample_images(images: list[Path], sample_size: int | None, seed: int) -> list[Path]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(images):
        return list(images)
    return random.Random(seed).sample(images, sample_size)
