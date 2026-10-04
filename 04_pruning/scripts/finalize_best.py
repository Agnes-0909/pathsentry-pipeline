"""Finalize the speed-selected H2 model with inference fusion and ONNX export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import ultralytics
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PRUNING_ROOT = ROOT / "04_pruning"
sys.path.insert(0, str(ROOT / "03_training/scripts"))
sys.path.insert(0, str(PRUNING_ROOT))
sys.path.insert(0, str(PRUNING_ROOT / "scripts"))
from checkpoint_compat import register_pickle_aliases

register_pickle_aliases()
from multitask import MultiTaskDataset, MultiTaskYOLO11s, collate_batch, evaluate_model
from architectures import FeatureFusionHead
from run_experiments import benchmark, visualize

SOURCE = PRUNING_ROOT / "experiments/p02_h2_p3_output/finetuned_best.pt"
BASE = ROOT / "03_training/runs/p01/b1/best.pt"
OUTPUT = PRUNING_ROOT / "final"


class DualOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        detection, segmentation = self.model(images)
        if isinstance(detection, (tuple, list)):
            detection = detection[0]
        return detection, segmentation


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    hash_value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def fuse_model(model):
    model.eval()
    model.detector.fuse(verbose=False)
    for block in model.segmenter.refine:
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = nn.Identity()
    return model.eval()


def run_metrics(model, config, split, device):
    dataset = MultiTaskDataset(config["data"], split,
                               (config["height"], config["width"]))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4,
                        pin_memory=device.type == "cuda", collate_fn=collate_batch)
    return {"split": split, **evaluate_model(model, loader, device)}, dataset


def run_benchmarks(source, base_state, dataset, device):
    plain_b1 = MultiTaskYOLO11s().to(device).eval()
    plain_b1.load_state_dict(base_state["model"])
    fused_b1 = MultiTaskYOLO11s().to(device).eval()
    fused_b1.load_state_dict(base_state["model"])
    fuse_model(fused_b1)
    plain_h2 = torch.load(source, map_location="cpu", weights_only=False)["model"].to(device).eval()
    fused_h2 = fuse_model(torch.load(source, map_location="cpu", weights_only=False)["model"].to(device))
    models = {"b1_plain": plain_b1, "b1_fused": fused_b1,
              "h2_plain": plain_h2, "h2_fused": fused_h2}
    runs = {name: [] for name in models}
    names = list(models)
    for repeat in range(4):
        for name in names[repeat:] + names[:repeat]:
            runs[name].append(benchmark(models[name], dataset, device))
    medians = {name: float(np.median([run["mean_ms"] for run in items]))
               for name, items in runs.items()}
    return {"input_size": [dataset.width, dataset.height], "batch": 1,
            "samples_per_run": min(32, len(dataset)), "runs": runs, "median_ms": medians,
            "h2_vs_b1_unfused_pct": 100 * (1 - medians["h2_plain"] / medians["b1_plain"]),
            "h2_vs_b1_fused_pct": 100 * (1 - medians["h2_fused"] / medians["b1_fused"]),
            "fused_h2_vs_plain_b1_pct": 100 * (1 - medians["h2_fused"] / medians["b1_plain"])}


def export_onnx(model, config, output, data):
    model = model.cpu().eval()
    wrapper = DualOutputWrapper(model).eval()
    size = (config["height"], config["width"])
    example = torch.zeros(1, 3, *size)
    with torch.inference_mode():
        torch.onnx.export(wrapper, example, str(output), opset_version=17,
                          input_names=["images"], output_names=["detection", "segmentation"],
                          do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    samples = [data[i]["img"].unsqueeze(0).float() / 255 for i in (0, len(data) // 2)]
    comparisons = []
    with torch.inference_mode():
        for sample in samples:
            torch_det, torch_seg = wrapper(sample)
            ort_det, ort_seg = session.run(None, {"images": sample.numpy()})
            det_diff = np.abs(torch_det.numpy() - ort_det)
            seg_diff = np.abs(torch_seg.numpy() - ort_seg)
            torch_mask = torch_seg.numpy() > 0
            ort_mask = ort_seg > 0
            comparisons.append({"detection_shape": list(ort_det.shape),
                                "segmentation_shape": list(ort_seg.shape),
                                "detection_mean_abs_diff": float(det_diff.mean()),
                                "detection_max_abs_diff": float(det_diff.max()),
                                "segmentation_mean_abs_diff": float(seg_diff.mean()),
                                "segmentation_max_abs_diff": float(seg_diff.max()),
                                "mask_agreement": float((torch_mask == ort_mask).mean())})
    if not all(item["detection_shape"] == [1, 6, 16380] and
               item["segmentation_shape"] == [1, 1, 208, 240] and
               item["mask_agreement"] > .999 for item in comparisons):
        raise RuntimeError("ONNX output shape or segmentation agreement check failed")
    return {"onnx_path": str(output), "onnx_sha256": digest(output),
            "opset": 17, "static_input_shape": [1, 3, *size],
            "output_names": [output.name for output in session.get_outputs()],
            "runtime_provider": "CPUExecutionProvider", "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    base_state = torch.load(BASE, map_location="cpu", weights_only=True)
    config = source["config"]
    fused = fuse_model(source["model"].to(device))
    val_metrics, val_data = run_metrics(fused, config, "val", device)
    test_metrics, _ = run_metrics(fused, config, "test", device)
    save_json(args.output / "val_metrics.json", val_metrics)
    save_json(args.output / "test_metrics.json", test_metrics)
    benchmark_result = run_benchmarks(args.source, base_state, val_data, device)
    save_json(args.output / "benchmark.json", benchmark_result)
    visualize(fused, val_data, device, args.output / "visuals")
    artifact = args.output / "best_fused.pt"
    torch.save({"model": fused.cpu(), "config": config, "source": str(args.source),
                "fusion": "Ultralytics detector Conv/BN and segmentation refine Conv/BN"}, artifact)
    reloaded = torch.load(artifact, map_location="cpu", weights_only=False)["model"].eval()
    with torch.inference_mode():
        detection, segmentation = reloaded(torch.zeros(1, 3, config["height"], config["width"]))
    if detection[0].shape != (1, 6, 16380) or segmentation.shape != (1, 1, 208, 240):
        raise RuntimeError("Reloaded fused checkpoint output shape mismatch")
    save_json(args.output / "artifact.json", {"source": str(args.source),
              "source_sha256": digest(args.source), "fused_path": str(artifact),
              "fused_sha256": digest(artifact), "fused_bytes": artifact.stat().st_size,
              "input_shape": [1, 3, config["height"], config["width"]],
              "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
              "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    onnx_result = export_onnx(reloaded, config, args.output / "best_fused.onnx", val_data)
    save_json(args.output / "onnx_validation.json", onnx_result)
    print(json.dumps({"artifact": str(artifact), "benchmark": benchmark_result["median_ms"],
                      "val": val_metrics["detection"]["map50_95"],
                      "val_iou": val_metrics["segmentation"]["iou"],
                      "test_iou": test_metrics["segmentation"]["iou"],
                      "onnx": onnx_result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
