"""Plot P01 validation metrics and training curves from saved JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "runs/p01/figures"
GROUPS = ["E0", "A1", "A2", "A3", "B1", "C1"]
METRIC_PATHS = {
    "E0": ROOT / "runs/yolo11s_experiments/final/e0/val_metrics.json",
    "A1": ROOT / "runs/p01/a1/val_metrics.json",
    "A2": ROOT / "runs/p01/a2/val_metrics.json",
    "A3": ROOT / "runs/p01/a3/val_metrics.json",
    "B1": ROOT / "runs/p01/b1/val_metrics.json",
    "C1": ROOT / "runs/p01/c1/val_metrics.json",
}
HISTORY_PATHS = {
    "E0": ROOT / "runs/yolo11s_experiments/final/e0/history.jsonl",
    **{name: ROOT / f"runs/p01/{name.lower()}/history.jsonl" for name in GROUPS[1:]},
}


def load_metrics():
    values = {}
    for name in GROUPS:
        item = json.loads(METRIC_PATHS[name].read_text(encoding="utf-8"))
        detection = item["detection"]
        values[name] = {
            "map50_95": detection["map50_95"],
            "person": detection["per_class"]["person"]["map50_95"],
            "vehicle": detection["per_class"]["vehicle"]["map50_95"],
            "seg_iou": item["segmentation"]["iou"],
        }
    return values


def plot_metric_bars(values):
    metrics = [
        ("map50_95", "Detection mAP50-95", "#2878b5"),
        ("person", "Person AP50-95", "#d95f02"),
        ("vehicle", "Vehicle AP50-95", "#1b9e77"),
        ("seg_iou", "Drivable-area IoU", "#7570b3"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (key, title, color) in zip(axes.flat, metrics):
        values_for_plot = [values[name][key] for name in GROUPS]
        bars = axis.bar(GROUPS, values_for_plot, color=color, alpha=0.88)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        top = max(values_for_plot) * 1.24 if key != "seg_iou" else 1.01
        bottom = 0 if key != "seg_iou" else min(values_for_plot) - 0.001
        axis.set_ylim(bottom, top)
        for bar, value in zip(bars, values_for_plot):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (top - bottom) * 0.015,
                      f"{value:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)
    figure.suptitle("P01 validation comparison", fontsize=14)
    figure.savefig(OUTPUT / "p01_metric_comparison.png", dpi=180)
    plt.close(figure)


def plot_curves():
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    colors = {name: color for name, color in zip(GROUPS, plt.cm.tab10.colors)}
    specs = [
        ("detection", "map50_95", "Validation mAP50-95"),
        ("detection", "person", "Validation person AP50-95"),
        ("segmentation", "iou", "Validation drivable-area IoU"),
    ]
    for axis, (section, metric, title) in zip(axes, specs):
        for name in GROUPS:
            rows = [json.loads(line) for line in HISTORY_PATHS[name].read_text(encoding="utf-8").splitlines() if line]
            epochs = [row["epoch"] for row in rows]
            if section == "detection" and metric == "person":
                series = [row["detection"]["per_class"]["person"]["map50_95"] for row in rows]
            elif section == "detection":
                series = [row["detection"][metric] for row in rows]
            else:
                series = [row["segmentation"][metric] for row in rows]
            axis.plot(epochs, series, label=name, color=colors[name], linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        if metric != "iou":
            axis.set_ylim(bottom=0)
    axes[0].set_ylabel("score")
    axes[2].legend(ncol=2, fontsize=8, loc="lower right")
    figure.suptitle("P01 validation learning curves", fontsize=14)
    figure.savefig(OUTPUT / "p01_learning_curves.png", dpi=180)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values = load_metrics()
    plot_metric_bars(values)
    plot_curves()
    print(OUTPUT / "p01_metric_comparison.png")
    print(OUTPUT / "p01_learning_curves.png")


if __name__ == "__main__":
    main()
