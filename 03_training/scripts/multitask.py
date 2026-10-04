"""YOLO11s detection plus binary drivable-area segmentation utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.nms import non_max_suppression

from copy_paste import CopyPasteBank


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "dataset/yolo_multitask/dataset.yaml"
DEFAULT_WEIGHTS = Path("/home/huace/下载/yolo11s.pt")
P2_MODEL = SCRIPT_DIR / "models/yolo11-p2.yaml"
CLASS_NAMES = {0: "person", 1: "vehicle"}


class SegmentationHead(nn.Module):
    """Fuse YOLO11s P5/P4/P3 features with a P2 skip connection."""

    def __init__(self, feature_indices=(2, 16, 19, 22)):
        super().__init__()
        self.feature_indices = feature_indices
        self.lateral = nn.ModuleList(nn.Conv2d(c, 128, 1) for c in (128, 128, 256, 512))
        self.refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            )
            for _ in range(3)
        )
        self.out = nn.Conv2d(128, 1, 1)

    def forward(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        p2, p3, p4, p5 = [layer(features[i]) for layer, i in zip(self.lateral, self.feature_indices)]
        x = p5
        for skip, block in zip((p4, p3, p2), self.refine):
            x = block(F.interpolate(x, size=skip.shape[-2:], mode="nearest") + skip)
        return self.out(x)


class MultiTaskYOLO11s(nn.Module):
    """YOLO11s detection graph with a semantic segmentation branch."""

    def __init__(self, pretrained: str | Path | None = None, p2_detect: bool = False):
        super().__init__()
        self.p2_detect = p2_detect
        if p2_detect:
            model_config = yaml.safe_load(P2_MODEL.read_text(encoding="utf-8"))
            model_config["scale"] = "s"
        else:
            model_config = "yolo11s.yaml"
        self.detector = DetectionModel(model_config, nc=2, verbose=False)
        self.detector.names = CLASS_NAMES
        self.detector.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        if pretrained is not None:
            pretrained = Path(pretrained).expanduser()
            if not pretrained.is_file():
                raise FileNotFoundError(f"YOLO11s weights not found: {pretrained}")
            source = YOLO(str(pretrained)).model
            if p2_detect:
                self._load_p2_pretrained(source)
            else:
                self.detector.load(source, verbose=True)
        self.feature_indices = (2, 22, 25, 28) if p2_detect else (2, 16, 19, 22)
        self.segmenter = SegmentationHead(self.feature_indices)

    def _load_p2_pretrained(self, source):
        source_state = source.float().state_dict()
        target_state = self.detector.state_dict()
        mapping = {17: 23, 19: 25, 20: 26, 22: 28}
        transfer = {}
        for key, value in source_state.items():
            parts = key.split(".")
            if parts[0] != "model":
                continue
            index = int(parts[1])
            if index <= 16:
                dest = key
            elif index in mapping:
                dest = ".".join(("model", str(mapping[index]), *parts[2:]))
            elif index == 23 and len(parts) > 3 and parts[2] in ("cv2", "cv3"):
                dest = ".".join(("model", "29", parts[2], str(int(parts[3]) + 1), *parts[4:]))
            elif index == 23 and parts[2] == "dfl":
                dest = ".".join(("model", "29", *parts[2:]))
            else:
                continue
            if dest in target_state and target_state[dest].shape == value.shape:
                transfer[dest] = value
        self.detector.load_state_dict(transfer, strict=False)
        print(f"Transferred {len(transfer)}/{len(target_state)} tensors from YOLO11s to P2 detector")

    def forward(self, images: torch.Tensor):
        saved = []
        features = {}
        x = images
        for layer in self.detector.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.detector.save else None)
            if layer.i in self.feature_indices:
                features[layer.i] = x
        return x, self.segmenter(features)

    def detection_loss(self, predictions, batch: dict[str, torch.Tensor]):
        labels = {key: batch[key] for key in ("img", "batch_idx", "cls", "bboxes")}
        loss, items = self.detector.loss(labels, preds=predictions)
        return loss.sum() / batch["img"].shape[0], items


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")
    valid = target != 255
    labels = target.clamp(0, 1)
    if not valid.any():
        return logits.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    bce = (bce * valid).sum() / valid.sum()
    probabilities = logits.float().sigmoid() * valid
    labels = labels * valid
    dice = 1 - (2 * (probabilities * labels).sum() + 1) / (probabilities.sum() + labels.sum() + 1)
    return 0.5 * (bce + dice)


class MultiTaskDataset(Dataset):
    def __init__(self, config: str | Path, split: str, size: tuple[int, int], augment=False, limit=None,
                 copy_paste_bank: str | Path | None = None, copy_paste_prob: float = 0.3):
        self.config = Path(config).resolve()
        data = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        self.root = Path(data["path"]).expanduser()
        if not self.root.is_absolute():
            self.root = (self.config.parent / self.root).resolve()
        self.split = split
        self.height, self.width = size
        if self.height % 32 or self.width % 32:
            raise ValueError("Image height and width must be multiples of 32")
        self.augment = augment
        if copy_paste_bank is not None and (split != "train" or not augment):
            raise ValueError("Copy-Paste is only allowed for augmented train data")
        self.copy_paste = CopyPasteBank(copy_paste_bank, copy_paste_prob) if copy_paste_bank else None
        with (self.root / "manifest.csv").open(newline="", encoding="utf-8") as file:
            self.rows = [row for row in csv.DictReader(file) if row["split"] == split]
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No samples in split {split!r}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.root / "images" / self.split / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / row["segmentation_mask"]), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise OSError(f"Could not read image or mask for {image_path}")
        h, w = image.shape[:2]
        if mask.shape != (h, w) or mask.ndim != 2 or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"Invalid binary mask for {image_path}")

        label_path = self.root / row["detection_label"]
        text = label_path.read_text(encoding="utf-8").strip()
        labels = np.array([list(map(float, line.split())) for line in text.splitlines()], dtype=np.float32).reshape(-1, 5)
        if len(labels) and (not np.isin(labels[:, 0], (0, 1)).all() or not ((labels[:, 1:] >= 0) & (labels[:, 1:] <= 1)).all()):
            raise ValueError(f"Invalid detection label: {label_path}")

        if self.copy_paste is not None:
            image, mask, labels = self.copy_paste.apply(image, mask, labels, row["image"])

        scale = min(self.width / w, self.height / h)
        new_w, new_h = round(w * scale), round(h * scale)
        left, top = (self.width - new_w) // 2, (self.height - new_h) // 2
        image = cv2.copyMakeBorder(
            cv2.resize(image, (new_w, new_h)), top, self.height - new_h - top,
            left, self.width - new_w - left, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        mask = cv2.copyMakeBorder(
            cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST),
            top, self.height - new_h - top, left, self.width - new_w - left,
            cv2.BORDER_CONSTANT, value=255
        )
        if len(labels):
            labels[:, 1] = (labels[:, 1] * w * scale + left) / self.width
            labels[:, 2] = (labels[:, 2] * h * scale + top) / self.height
            labels[:, 3] *= new_w / self.width
            labels[:, 4] *= new_h / self.height
        if self.augment:
            if np.random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
                labels[:, 1] = 1 - labels[:, 1]
            if np.random.random() < 0.8:
                image = np.clip(image.astype(np.float32) * np.random.uniform(0.85, 1.15)
                                + np.random.uniform(-15, 15), 0, 255).astype(np.uint8)
        image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())
        return {
            "img": image,
            "mask": torch.from_numpy(mask.copy()),
            "cls": torch.from_numpy(labels[:, :1].copy()),
            "bboxes": torch.from_numpy(labels[:, 1:].copy()),
            "path": str(image_path),
        }


def collate_batch(samples):
    return {
        "img": torch.stack([s["img"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
        "cls": torch.cat([s["cls"] for s in samples]),
        "bboxes": torch.cat([s["bboxes"] for s in samples]),
        "batch_idx": torch.cat([
            torch.full((len(s["cls"]),), i, dtype=torch.float32) for i, s in enumerate(samples)
        ]),
        "path": [s["path"] for s in samples],
    }


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()} | {"img": batch["img"].to(device, non_blocking=True).float() / 255}


def _match_detections(prediction: torch.Tensor, target_boxes: torch.Tensor, target_cls: torch.Tensor):
    correct = np.zeros((len(prediction), 10), dtype=bool)
    if not len(prediction) or not len(target_boxes):
        return correct
    iou = box_iou(target_boxes, prediction[:, :4])
    iou *= target_cls[:, None] == prediction[:, 5]
    values = iou.cpu().numpy()
    for column, threshold in enumerate(np.linspace(0.5, 0.95, 10)):
        matches = np.argwhere(values >= threshold)
        if len(matches):
            matches = matches[np.argsort(values[matches[:, 0], matches[:, 1]])[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1], column] = True
    return correct


@torch.inference_mode()
def evaluate_model(model, loader, device, conf=0.001, iou=0.7, segmentation=True):
    model.eval()
    metrics = DetMetrics(names=CLASS_NAMES)
    seg_tp = seg_fp = seg_fn = 0
    for batch in loader:
        batch = to_device(batch, device)
        det_output, seg_logits = model(batch["img"])
        predictions = non_max_suppression(det_output, conf_thres=conf, iou_thres=iou,
                                          nc=2, multi_label=True, max_det=300)
        if segmentation:
            masks = batch["mask"]
            seg_pred = F.interpolate(seg_logits.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)[:, 0] > 0
            valid = masks != 255
            truth = masks == 1
            seg_tp += int((seg_pred & truth & valid).sum())
            seg_fp += int((seg_pred & ~truth & valid).sum())
            seg_fn += int((~seg_pred & truth & valid).sum())

        h, w = batch["img"].shape[-2:]
        for j, pred in enumerate(predictions):
            selected = batch["batch_idx"] == j
            cls = batch["cls"][selected, 0]
            boxes = batch["bboxes"][selected]
            xyxy = torch.empty_like(boxes)
            xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
            xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
            xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
            xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
            metrics.update_stats({
                "tp": _match_detections(pred, xyxy, cls),
                "conf": pred[:, 4].cpu().numpy(),
                "pred_cls": pred[:, 5].cpu().numpy(),
                "target_cls": cls.cpu().numpy(),
                "target_img": np.unique(cls.cpu().numpy()),
            })
    metrics.process()
    precision, recall, map50, map5095 = metrics.mean_results()
    per_class = {}
    for position, class_id in enumerate(metrics.ap_class_index):
        p, r, a50, a5095 = metrics.class_result(position)
        per_class[CLASS_NAMES[int(class_id)]] = {"precision": float(p), "recall": float(r),
                                                  "map50": float(a50), "map50_95": float(a5095)}
    return {
        "detection": {"precision": float(precision), "recall": float(recall),
                      "map50": float(map50), "map50_95": float(map5095), "per_class": per_class},
        "segmentation": ({"iou": seg_tp / max(seg_tp + seg_fp + seg_fn, 1),
                          "dice": 2 * seg_tp / max(2 * seg_tp + seg_fp + seg_fn, 1)}
                         if segmentation else None),
        "samples": len(loader.dataset),
    }
