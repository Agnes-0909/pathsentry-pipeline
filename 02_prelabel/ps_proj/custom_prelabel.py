from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SAM_SRC = PROJECT_ROOT / "sam3_1_prelabel" / "src"
QWEN_SRC = PROJECT_ROOT / "qwen2_5_7B_prelabel" / "src"

for source_root in (SAM_SRC, QWEN_SRC):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

from qwen25_prelabel.sampling import collect_images  # noqa: E402
from sam3_prelabel.geometry import (  # noqa: E402
    clean_drivable_mask,
    mask_to_polygons,
    merge_region_masks,
    parse_prompts,
    write_json,
)

DEFAULT_LEFT_INPUT = Path("/home/huace/new_storage/ps_datasets/custom/left")
DEFAULT_RIGHT_INPUT = Path("/home/huace/new_storage/ps_datasets/custom/right")
DEFAULT_LEFT_OUTPUT = Path("/home/huace/new_storage/ps_datasets/custom/left_label")
DEFAULT_RIGHT_OUTPUT = Path("/home/huace/new_storage/ps_datasets/custom/right_label")
DEFAULT_WORK_ROOT = PROJECT_ROOT / "prelabel_work"
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "sam3_1" / "sam3.1_multiplex.pt"
DEFAULT_QWEN_MODEL_DIR = PROJECT_ROOT / "qwen2_5_7B_prelabel" / "model_dir"


@dataclass(frozen=True, slots=True)
class SideJob:
    name: str
    input_dir: Path
    output_dir: Path
    work_dir: Path


def labelme_output_path(image_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{image_path.stem}.json"


def stage_output_path(stage_dir: Path, image_path: Path) -> Path:
    return stage_dir / f"{image_path.stem}.json"


def build_labelme_record(
    *,
    image_path: Path,
    width: int,
    height: int,
    drivable_polygons: list[list[list[int]]],
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    shapes: list[dict[str, Any]] = []
    for polygon in drivable_polygons:
        shapes.append(
            {
                "label": "drivable_area",
                "points": [[int(x), int(y)] for x, y in polygon],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {"source": "sam"},
            }
        )

    for detection in detections:
        bbox = detection.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x0, y0, x1, y1 = (float(value) for value in bbox)
        flags = {
            "source": "qwen",
            "confidence": float(detection.get("confidence", 0.0)),
        }
        reason = detection.get("reason")
        if isinstance(reason, str) and reason:
            flags["reason"] = reason

        shapes.append(
            {
                "label": str(detection.get("label", "other_obstacle")),
                "points": [[x0, y0], [x1, y1]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": flags,
            }
        )

    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def _default_sam_bpe_path() -> Path:
    try:
        import open_clip

        open_clip_path = Path(open_clip.__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
        if open_clip_path.exists():
            return open_clip_path
    except Exception:
        pass

    asset_path = PROJECT_ROOT / "sam3_1_prelabel" / "src" / "sam3_prelabel" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if asset_path.exists():
        return asset_path

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    urllib.request.urlretrieve("https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz", asset_path)
    return asset_path


def _default_sam_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return os.getenv("SAM3_DEVICE", "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM + Qwen prelabel merge to LabelMe")
    parser.add_argument("--left-input", type=Path, default=DEFAULT_LEFT_INPUT)
    parser.add_argument("--right-input", type=Path, default=DEFAULT_RIGHT_INPUT)
    parser.add_argument("--left-output", type=Path, default=DEFAULT_LEFT_OUTPUT)
    parser.add_argument("--right-output", type=Path, default=DEFAULT_RIGHT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--sam-checkpoint-path", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-bpe-path", type=Path, default=None)
    parser.add_argument("--sam-load-from-hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam-device", default=os.getenv("SAM3_DEVICE") or _default_sam_device())
    parser.add_argument(
        "--sam-prompts",
        default="drivable road surface regardless of color or lane markings\nroadway\npavement",
    )
    parser.add_argument("--sam-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam-max-regions", type=int, default=64)
    parser.add_argument("--sam-close-kernel-size", type=int, default=41)
    parser.add_argument("--sam-min-component-area-ratio", type=float, default=0.00025)
    parser.add_argument("--qwen-model-dir", type=Path, default=DEFAULT_QWEN_MODEL_DIR)
    parser.add_argument("--qwen-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--qwen-max-model-len", type=int, default=4096)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=512)
    parser.add_argument("--qwen-max-image-pixels", type=int, default=401408)
    parser.add_argument("--qwen-cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-min-confidence", type=float, default=0.0)
    parser.add_argument("--qwen-min-box-area", type=float, default=16.0)
    parser.add_argument("--sides", choices=("left", "right", "both"), default="both")
    parser.add_argument("--skip-sam", action="store_true", help="reuse existing SAM stage files")
    parser.add_argument("--skip-qwen", action="store_true", help="reuse existing Qwen stage files")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing stage files and labelme JSON")
    return parser


def make_jobs(args: argparse.Namespace) -> list[SideJob]:
    jobs = [
        SideJob(
            name="left",
            input_dir=args.left_input,
            output_dir=args.left_output,
            work_dir=args.work_dir / "left",
        ),
        SideJob(
            name="right",
            input_dir=args.right_input,
            output_dir=args.right_output,
            work_dir=args.work_dir / "right",
        ),
    ]
    if args.sides == "left":
        return jobs[:1]
    if args.sides == "right":
        return jobs[1:]
    return jobs


def _load_sam_engine(args: argparse.Namespace):
    from sam3_prelabel.config import Settings as SamSettings
    from sam3_prelabel.engine import Sam3Engine

    settings = SamSettings()
    bpe_path = Path(args.sam_bpe_path) if args.sam_bpe_path else Path(settings.bpe_path)
    checkpoint_path = Path(args.sam_checkpoint_path) if args.sam_checkpoint_path else None
    return Sam3Engine(
        device=args.sam_device,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        bpe_path=str(bpe_path),
        load_from_hf=args.sam_load_from_hf,
        default_confidence_threshold=args.sam_confidence_threshold,
        compile_model=False,
    )


def _load_qwen_detector(args: argparse.Namespace):
    from qwen25_prelabel.engine import VllmObstacleDetector

    return VllmObstacleDetector(
        model_dir=args.qwen_model_dir,
        tensor_parallel_size=args.qwen_tensor_parallel_size,
        gpu_memory_utilization=args.qwen_gpu_memory_utilization,
        max_model_len=args.qwen_max_model_len,
        max_new_tokens=args.qwen_max_new_tokens,
        max_image_pixels=args.qwen_max_image_pixels,
        cpu_offload_gb=args.qwen_cpu_offload_gb,
        temperature=args.qwen_temperature,
        min_confidence=args.qwen_min_confidence,
        min_box_area=args.qwen_min_box_area,
    )


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _progress(stage: str, side: str, index: int, total: int, image_path: Path) -> None:
    if index == 1 or index == total or index % 250 == 0:
        print(f"[{stage}][{side}] {index}/{total} {image_path.name}")


def run_sam_stage(job: SideJob, args: argparse.Namespace, prompts: list[str], engine: Any) -> list[Path]:
    images = collect_images(job.input_dir, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No supported images found in: {job.input_dir}")

    stage_dir = job.work_dir / "sam"
    stage_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, image_path in enumerate(images, start=1):
        stage_path = stage_output_path(stage_dir, image_path)
        outputs.append(stage_path)
        if stage_path.exists() and not args.overwrite:
            continue

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        result = engine.segment(
            image,
            prompts,
            confidence_threshold=args.sam_confidence_threshold,
            max_regions=args.sam_max_regions,
        )
        merged_mask = clean_drivable_mask(
            merge_region_masks(result.regions),
            close_kernel_size=args.sam_close_kernel_size,
            min_component_area_ratio=args.sam_min_component_area_ratio,
        )
        drivable_polygons = mask_to_polygons(merged_mask) if merged_mask.size else []

        payload = {
            "image": str(image_path.resolve()),
            "width": image.width,
            "height": image.height,
            "prompts": prompts,
            "mask_postprocess": {
                "close_kernel_size": args.sam_close_kernel_size,
                "min_component_area_ratio": args.sam_min_component_area_ratio,
            },
            "region_count": len(result.regions),
            "drivable_polygons": drivable_polygons,
            "regions": [
                {
                    "prompt": region.prompt,
                    "score": region.score,
                    "box": [float(value) for value in region.box],
                    "polygons": region.polygons,
                }
                for region in result.regions
            ],
        }
        write_json(stage_path, payload)
        _progress("sam", job.name, index, len(images), image_path)
    return outputs


def run_qwen_stage(job: SideJob, args: argparse.Namespace, detector: Any) -> list[Path]:
    images = collect_images(job.input_dir, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No supported images found in: {job.input_dir}")

    stage_dir = job.work_dir / "qwen"
    stage_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, image_path in enumerate(images, start=1):
        stage_path = stage_output_path(stage_dir, image_path)
        outputs.append(stage_path)
        if stage_path.exists() and not args.overwrite:
            continue

        annotation = detector.detect(image_path)
        payload = {
            "image": str(annotation.image),
            "width": annotation.width,
            "height": annotation.height,
            "detections": [detection.to_dict() for detection in annotation.detections],
            "parse_error": annotation.parse_error,
        }
        write_json(stage_path, payload)
        _progress("qwen", job.name, index, len(images), image_path)
    return outputs


def merge_stage_outputs(job: SideJob, args: argparse.Namespace) -> list[Path]:
    images = collect_images(job.input_dir, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No supported images found in: {job.input_dir}")

    sam_stage_dir = job.work_dir / "sam"
    qwen_stage_dir = job.work_dir / "qwen"
    job.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, image_path in enumerate(images, start=1):
        final_path = labelme_output_path(image_path, job.output_dir)
        outputs.append(final_path)
        if final_path.exists() and not args.overwrite:
            continue

        sam_stage_path = stage_output_path(sam_stage_dir, image_path)
        qwen_stage_path = stage_output_path(qwen_stage_dir, image_path)
        if not sam_stage_path.exists():
            raise FileNotFoundError(f"Missing SAM stage file: {sam_stage_path}")
        if not qwen_stage_path.exists():
            raise FileNotFoundError(f"Missing Qwen stage file: {qwen_stage_path}")

        sam_stage = json.loads(sam_stage_path.read_text(encoding="utf-8"))
        qwen_stage = json.loads(qwen_stage_path.read_text(encoding="utf-8"))
        record = build_labelme_record(
            image_path=Path(sam_stage["image"]),
            width=int(sam_stage["width"]),
            height=int(sam_stage["height"]),
            drivable_polygons=[[[int(x), int(y)] for x, y in polygon] for polygon in sam_stage.get("drivable_polygons", [])],
            detections=list(qwen_stage.get("detections", [])),
        )
        write_json(final_path, record)
        _progress("merge", job.name, index, len(images), image_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    jobs = make_jobs(args)
    for job in jobs:
        if not job.input_dir.exists():
            raise SystemExit(f"Input path does not exist: {job.input_dir}")

    prompts = parse_prompts(args.sam_prompts)
    if not prompts:
        raise SystemExit("At least one SAM prompt is required.")

    if not args.skip_sam:
        sam_engine = _load_sam_engine(args)
        for job in jobs:
            print(f"== SAM stage: {job.name} ==")
            run_sam_stage(job, args, prompts, sam_engine)
        del sam_engine
        _release_gpu_memory()
    else:
        for job in jobs:
            expected = len(collect_images(job.input_dir, recursive=args.recursive))
            actual = len(list((job.work_dir / "sam").glob("*.json")))
            if actual < expected:
                raise SystemExit(
                    f"Cannot skip SAM for {job.name}: found {actual}/{expected} stage files."
                )
        print("== SAM stage: reused existing files ==")

    if not args.skip_qwen:
        qwen_detector = _load_qwen_detector(args)
        for job in jobs:
            print(f"== Qwen stage: {job.name} ==")
            run_qwen_stage(job, args, qwen_detector)
        del qwen_detector
        _release_gpu_memory()
    else:
        for job in jobs:
            expected = len(collect_images(job.input_dir, recursive=args.recursive))
            actual = len(list((job.work_dir / "qwen").glob("*.json")))
            if actual < expected:
                raise SystemExit(
                    f"Cannot skip Qwen for {job.name}: found {actual}/{expected} stage files."
                )
        print("== Qwen stage: reused existing files ==")

    for job in jobs:
        print(f"== Merge stage: {job.name} ==")
        merge_stage_outputs(job, args)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
