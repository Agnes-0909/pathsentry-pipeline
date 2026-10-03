from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import ModelCompatibilityError, VllmObstacleDetector
from .export import write_annotation_outputs
from .sampling import collect_images, sample_images

DEFAULT_MODEL_DIR = Path("/home/huace/new_storage/ps_proj/qwen2_5_7B_prelabel/model_dir")
DEFAULT_INPUT_DIR = Path("/home/huace/new_storage/ps_datasets/custom/left")
DEFAULT_OUTPUT_DIR = Path("/home/huace/new_storage/ps_proj/qwen2_5_7B_prelabel/outputs/left_sample_200")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL vLLM obstacle prelabeling")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT_DIR, help="image file or directory")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recursive", action="store_true", help="scan input directory recursively")
    parser.add_argument("--sample-size", type=int, default=200, help="number of images to sample; <=0 means all")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-box-area", type=float, default=16.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--no-yolo", action="store_true", help="do not write YOLO label files")
    parser.add_argument("--no-overlay", action="store_true", help="do not write overlay PNG files")
    parser.add_argument("--dry-run", action="store_true", help="sample images and write sample_manifest.json without loading vLLM")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input path does not exist: {args.input}")

    images = collect_images(args.input, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No supported images found in: {args.input}")

    selected = sample_images(images, sample_size=args.sample_size, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_manifest(args.output_dir / "sample_manifest.json", selected, args.seed)
    if args.dry_run:
        print(f"Dry run selected {len(selected)} images. Manifest: {args.output_dir / 'sample_manifest.json'}")
        return 0

    annotations_path = args.output_dir / "annotations.jsonl"
    if annotations_path.exists():
        annotations_path.unlink()

    try:
        detector = VllmObstacleDetector(
            model_dir=args.model_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_confidence=args.min_confidence,
            min_box_area=args.min_box_area,
        )
    except ModelCompatibilityError as exc:
        raise SystemExit(str(exc)) from exc

    for index, image_path in enumerate(selected, start=1):
        try:
            annotation = detector.detect(image_path)
        except Exception as exc:
            print(f"[{index}/{len(selected)}] {image_path.name}: skipped ({exc})")
            continue

        write_annotation_outputs(
            annotation,
            args.output_dir,
            write_yolo=not args.no_yolo,
            write_overlay=not args.no_overlay,
        )
        print(f"[{index}/{len(selected)}] {image_path.name}: {len(annotation.detections)} detections")

    return 0


def _write_sample_manifest(path: Path, images: list[Path], seed: int) -> None:
    payload = {
        "seed": seed,
        "count": len(images),
        "images": [str(image.resolve()) for image in images],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
