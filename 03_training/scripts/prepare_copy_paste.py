"""Create train-only object cutouts from YOLO boxes with the local SAM 3.1 model.

Run this script with the vllm Python environment, which contains sam3.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from cutout_contact_sheet import make_contact_sheet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "dataset/yolo_multitask"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/p01/materials/sam3_1"
DEFAULT_CHECKPOINT = Path("/home/huace/new_storage/ps_proj/prelabel/sam3_1/sam3.1_multiplex.pt")
CLASS_NAMES = {0: "person", 1: "vehicle"}


def default_bpe_path():
    spec = importlib.util.find_spec("open_clip")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("open_clip is required to locate the SAM 3.1 BPE vocabulary")
    path = Path(spec.origin).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bpe-path", type=Path, default=default_bpe_path())
    parser.add_argument("--person-stride", type=int, default=3)
    parser.add_argument("--vehicle-stride", type=int, default=10)
    parser.add_argument("--max-images", type=int, help="For a smoke test only")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def xyxy(cx, cy, width, height, image_width, image_height):
    return [
        int(round((cx - width / 2) * image_width)),
        int(round((cy - height / 2) * image_height)),
        int(round((cx + width / 2) * image_width)),
        int(round((cy + height / 2) * image_height)),
    ]


def usable_cutout(mask, label_box):
    ys, xs = np.nonzero(mask)
    if len(xs) < 32:
        return None
    tight = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    iou = box_iou(tight, label_box)
    x1, y1, x2, y2 = label_box
    box_area = max(1, (x2 - x1) * (y2 - y1))
    area_ratio = len(xs) / box_area
    inside = mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].sum() / len(xs)
    if iou < 0.5 or not 0.12 <= area_ratio <= 1.25 or inside < 0.75:
        return None
    if tight[2] - tight[0] < 8 or tight[3] - tight[1] < 8:
        return None
    return tight, {"box_iou": float(iou), "area_ratio": float(area_ratio), "inside_box": float(inside)}


def main():
    args = arguments()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The installed SAM 3.1 package requires CUDA for image inference")
    if args.person_stride < 1 or args.vehicle_stride < 1:
        raise ValueError("strides must be positive")
    data = args.data.resolve()
    output = args.output.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    with (data / "manifest.csv").open(newline="", encoding="utf-8") as file:
        samples = [row for row in csv.DictReader(file) if row["split"] == "train"]
    selected = []
    for row in samples:
        frame_id = int(Path(row["image"]).stem)
        labels = []
        for index, line in enumerate((data / row["detection_label"]).read_text(encoding="utf-8").splitlines()):
            class_id, cx, cy, width, height = map(float, line.split())
            if int(class_id) == 1 and width / max(height, 1e-6) > 3:
                continue
            stride = args.person_stride if int(class_id) == 0 else args.vehicle_stride
            if frame_id % stride == 0:
                labels.append((index, int(class_id), cx, cy, width, height))
        if labels:
            selected.append((row, labels))
    if args.max_images is not None:
        selected = selected[:args.max_images]
    print(f"Selected {len(selected)} training images for SAM 3.1", flush=True)
    if not selected:
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    existing = []
    if manifest_path.exists():
        existing = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    seen = {row["id"] for row in existing}
    model = build_sam3_image_model(device=args.device, checkpoint_path=str(args.checkpoint),
                                    bpe_path=str(args.bpe_path),
                                    load_from_HF=False, enable_segmentation=True,
                                    enable_inst_interactivity=False)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.05)
    counters = Counter()
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for image_index, (row, labels) in enumerate(selected, 1):
            remaining = [item for item in labels if f"{Path(row['image']).stem}_{item[0]}" not in seen]
            if not remaining:
                continue
            with Image.open(data / "images/train" / row["image"]) as image_file:
                image = image_file.convert("RGB")
            rgb = np.asarray(image)
            state = processor.set_image(image)
            for label_index, class_id, cx, cy, width, height in remaining:
                identifier = f"{Path(row['image']).stem}_{label_index}"
                target_box = xyxy(cx, cy, width, height, image.width, image.height)
                processor.reset_all_prompts(state)
                result = processor.add_geometric_prompt([cx, cy, width, height], True, state)
                masks = result["masks"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                candidates = []
                for mask, score in zip(masks, scores):
                    mask = np.asarray(mask).squeeze().astype(bool)
                    quality = usable_cutout(mask, target_box)
                    if quality is not None:
                        tight, stats = quality
                        candidates.append((stats["box_iou"] + 0.1 * float(score), mask, tight, stats, float(score)))
                if not candidates:
                    counters[f"rejected_{CLASS_NAMES[class_id]}"] += 1
                    continue
                _, mask, tight, stats, score = max(candidates, key=lambda item: item[0])
                x1, y1, x2, y2 = tight
                rgba = np.dstack((rgb[y1:y2, x1:x2], mask[y1:y2, x1:x2].astype(np.uint8) * 255))
                crop = f"crops/{CLASS_NAMES[class_id]}/{identifier}.png"
                (output / crop).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgba, "RGBA").save(output / crop)
                record = {"id": identifier, "class_id": class_id, "source_image": row["image"],
                          "crop": crop, "source_box": target_box, "tight_box": tight,
                          "image_size": [image.width, image.height], "sam_score": score, **stats}
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                existing.append(record)
                seen.add(identifier)
                counters[CLASS_NAMES[class_id]] += 1
            if image_index % 50 == 0:
                print(f"{image_index}/{len(selected)} images: {dict(counters)}", flush=True)
    make_contact_sheet(existing, output)
    totals = Counter(CLASS_NAMES[row["class_id"]] for row in existing)
    summary = {"source_split": "train", "checkpoint": str(args.checkpoint),
               "person_stride": args.person_stride, "vehicle_stride": args.vehicle_stride,
               "selected_images": len(selected), "counts": dict(totals),
               "new_or_rejected_this_run": dict(counters), "total_cutouts": len(existing)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
