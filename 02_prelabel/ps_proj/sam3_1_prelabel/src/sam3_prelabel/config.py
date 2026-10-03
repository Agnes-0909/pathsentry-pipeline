from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import torch


_BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"


def _default_bpe_path() -> Path:
    try:
        import open_clip

        open_clip_path = Path(open_clip.__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
        if open_clip_path.exists():
            return open_clip_path
    except Exception:
        pass

    asset_path = Path(__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if asset_path.exists():
        return asset_path
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_BPE_URL, asset_path)
    return asset_path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    device: str = field(default_factory=lambda: os.getenv("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path: str | None = field(default_factory=lambda: os.getenv("SAM3_CHECKPOINT_PATH") or None)
    bpe_path: str | None = field(default_factory=lambda: os.getenv("SAM3_BPE_PATH") or None)
    load_from_hf: bool = field(default_factory=lambda: _env_bool("SAM3_LOAD_FROM_HF", True))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5")))
    max_regions: int = field(default_factory=lambda: int(os.getenv("SAM3_MAX_REGIONS", "64")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("SAM3_OUTPUT_DIR", "outputs")))

    def __post_init__(self) -> None:
        if self.bpe_path is None:
            self.bpe_path = str(_default_bpe_path())
        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
