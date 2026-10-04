"""Save fixed validation examples with detection and drivable-area predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils.nms import non_max_suppression

from multitask import DEFAULT_DATA, MultiTaskDataset, MultiTaskYOLO11s


def selected_indices(dataset, count):
    positives = []
    negatives = []
    for index, row in enumerate(dataset.rows):
        label_path = dataset.root / row["detection_label"]
        if any(line.startswith("0 ") for line in label_path.read_text(encoding="utf-8").splitlines()):
            positives.append(index)
        else:
            negatives.append(index)
    chosen = []
    for group in (positives, negatives):
        n = min(count // 2, len(group))
        if n:
            chosen.extend(group[i] for i in np.linspace(0, len(group) - 1, n, dtype=int))
    return sorted(set(chosen))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    size = (state["config"]["height"], state["config"]["width"])
    dataset = MultiTaskDataset(args.data, args.split, size)
    device = torch.device(args.device)
    model = MultiTaskYOLO11s(p2_detect=state["config"].get("p2_detect", False)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for index in selected_indices(dataset, args.count):
        sample = dataset[index]
        image = sample["img"].permute(1, 2, 0).numpy()
        tensor = sample["img"].unsqueeze(0).to(device).float() / 255
        det, logits = model(tensor)
        predictions = non_max_suppression(det, conf_thres=args.conf, iou_thres=0.7, nc=2,
                                          multi_label=True, max_det=300)[0]
        seg = F.interpolate(logits.float(), size=image.shape[:2], mode="bilinear",
                            align_corners=False)[0, 0].sigmoid().cpu().numpy() > 0.5
        output = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        overlay = output.copy()
        overlay[seg] = (90, 190, 70)
        output = cv2.addWeighted(output, 0.72, overlay, 0.28, 0)
        height, width = image.shape[:2]
        labels = np.concatenate((sample["cls"].numpy(), sample["bboxes"].numpy()), axis=1)
        for class_id, cx, cy, bw, bh in labels:
            x1, y1 = round((cx - bw / 2) * width), round((cy - bh / 2) * height)
            x2, y2 = round((cx + bw / 2) * width), round((cy + bh / 2) * height)
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 1)
        for x1, y1, x2, y2, score, class_id in predictions.cpu().numpy():
            color = (30, 30, 230) if int(class_id) == 0 else (240, 190, 20)
            cv2.rectangle(output, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
            if int(class_id) == 0:
                cv2.putText(output, f"person {score:.2f}",
                            (round(x1), max(18, round(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        cv2.putText(output, Path(sample["path"]).name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imwrite(str(args.output / Path(sample["path"]).name), output)
        thumbnails.append(cv2.resize(output, (480, 416), interpolation=cv2.INTER_AREA))
    for start in range(0, len(thumbnails), 6):
        rows = []
        for i in range(start, min(start + 6, len(thumbnails)), 2):
            pair = thumbnails[i:i + 2]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            rows.append(np.concatenate(pair, axis=1))
        cv2.imwrite(str(args.output / f"contact_{start // 6:02d}.jpg"), np.concatenate(rows, axis=0))
    print(f"Saved {len(thumbnails)} samples to {args.output}")


if __name__ == "__main__":
    main()
