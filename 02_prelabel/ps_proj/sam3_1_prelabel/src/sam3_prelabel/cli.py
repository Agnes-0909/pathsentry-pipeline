from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .engine import Region, Sam3Engine
from .geometry import build_manifest, merge_region_masks, parse_prompts, render_overlay, write_json

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Offline SAM 3.1 image prelabeling")
    parser.add_argument("input", type=Path, help="image file or directory")
    parser.add_argument("--prompts", help="prompt text, separated by commas or newlines")
    parser.add_argument("--prompt-file", type=Path, help="text file containing prompts")
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir, help="output directory")
    parser.add_argument("--recursive", action="store_true", help="scan input directory recursively")
    parser.add_argument("--confidence-threshold", type=float, default=settings.confidence_threshold)
    parser.add_argument("--max-regions", type=int, default=settings.max_regions)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--checkpoint-path", default=settings.checkpoint_path)
    parser.add_argument("--bpe-path", default=settings.bpe_path)
    parser.add_argument("--load-from-hf", action=argparse.BooleanOptionalAction, default=settings.load_from_hf)
    parser.add_argument("--compile", action="store_true", help="enable torch compile for supported models")
    return parser


def collect_images(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return images


def resolve_prompts(args: argparse.Namespace) -> list[str]:
    text = args.prompts or ""
    if args.prompt_file:
        text = args.prompt_file.read_text(encoding="utf-8")
    prompts = parse_prompts(text)
    if not prompts:
        raise SystemExit("At least one prompt is required via --prompts or --prompt-file.")
    return prompts


def output_root_for_image(input_root: Path, image_path: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        return output_dir / image_path.stem
    relative = image_path.relative_to(input_root)
    return output_dir / relative.with_suffix("")


def process_image(
    engine: Sam3Engine,
    image_path: Path,
    output_root: Path,
    prompts: list[str],
    confidence_threshold: float,
    max_regions: int,
) -> dict:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    result = engine.segment(
        image,
        prompts,
        confidence_threshold=confidence_threshold,
        max_regions=max_regions,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "masks").mkdir(parents=True, exist_ok=True)
    image.save(output_root / "original.png")

    overlay = render_overlay(image, result.regions)
    overlay.save(output_root / "overlay.png")

    merged_mask = merge_region_masks(result.regions)
    if merged_mask.size:
        Image.fromarray((merged_mask.astype(np.uint8) * 255), mode="L").save(output_root / "drivable_mask.png")
        merged_regions = [Region(
            prompt="merged",
            score=1.0,
            box=[0.0, 0.0, float(image.width), float(image.height)],
            mask=merged_mask,
            polygons=[],
        )] if result.regions else []
        if merged_regions:
            render_overlay(image, merged_regions).save(output_root / "drivable_overlay.png")

    mask_files: list[str] = []
    for index, region in enumerate(result.regions):
        mask_name = f"masks/mask_{index:03d}.png"
        Image.fromarray((region.mask.astype(np.uint8) * 255), mode="L").save(output_root / mask_name)
        mask_files.append(mask_name)

    manifest = build_manifest(image_path.name, result, mask_files)
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    prompts = resolve_prompts(args)
    images = collect_images(input_path, args.recursive)
    if not images:
        raise SystemExit("No images found.")

    engine = Sam3Engine(
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        bpe_path=args.bpe_path,
        load_from_hf=args.load_from_hf,
        default_confidence_threshold=args.confidence_threshold,
        compile_model=args.compile,
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        sample_dir = output_root_for_image(input_path, image_path, output_dir)
        manifest = process_image(
            engine=engine,
            image_path=image_path,
            output_root=sample_dir,
            prompts=prompts,
            confidence_threshold=args.confidence_threshold,
            max_regions=args.max_regions,
        )
        print(f"{image_path} -> {sample_dir} ({manifest['region_count']} regions)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
