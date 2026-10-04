"""Run five YOLO11s training experiments in sequence and select final candidates."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from multitask import DEFAULT_DATA, DEFAULT_WEIGHTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "scripts/evaluate.py"


@dataclass(frozen=True)
class Experiment:
    name: str
    width: int
    height: int
    seg_weight: float
    description: str


EXPERIMENTS = (
    Experiment("e0", 960, 832, 5.0, "multitask baseline"),
    Experiment("e1", 960, 832, 1.0, "low segmentation weight"),
    Experiment("e2", 960, 832, 10.0, "high segmentation weight"),
    Experiment("e3", 640, 544, 5.0, "lower input resolution"),
    Experiment("e4", 960, 832, 0.0, "detection-only control"),
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "pilot", "final"), default="all")
    parser.add_argument("--experiments", nargs="+", choices=[e.name for e in EXPERIMENTS],
                        default=[e.name for e in EXPERIMENTS])
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs/yolo11s_experiments")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--pilot-epochs", type=int, default=20)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=2, help="Multitask candidates retrained from pretrained weights")
    parser.add_argument("--full-det-baseline", action="store_true", help="Also retrain E4 for 100 epochs")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-limit", type=int, help="Small dataset subset for an end-to-end smoke test")
    parser.add_argument("--val-limit", type=int, help="Small validation subset for an end-to-end smoke test")
    parser.add_argument("--test-limit", type=int, help="Small test subset for an end-to-end smoke test")
    parser.add_argument("--resume-incomplete", action="store_true", help="Resume incomplete runs from last.pt")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or writing files")
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def best_record(run_dir: Path) -> dict:
    history = records(run_dir / "history.jsonl")
    if not history or not (run_dir / "best.pt").is_file():
        raise RuntimeError(f"Missing training results in {run_dir}")
    return max(history, key=lambda row: row["score"])


def run_command(command: list[str], dry_run: bool):
    print("$", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_experiment(exp: Experiment, stage: str, epochs: int, args, output_root: Path):
    run_dir = output_root / stage / exp.name
    config = {"experiment": exp.name, "stage": stage, "epochs": epochs,
              "width": exp.width, "height": exp.height, "seg_weight": exp.seg_weight,
              "data": str(resolved(args.data)), "weights": str(resolved(args.weights)),
              "batch": args.batch, "workers": args.workers, "device": args.device,
              "seed": args.seed, "lr": args.lr, "weight_decay": args.weight_decay,
              "train_limit": args.train_limit, "val_limit": args.val_limit}
    command = [sys.executable, str(TRAIN_SCRIPT), "--data", config["data"],
               "--weights", config["weights"], "--output", str(run_dir),
               "--epochs", str(epochs), "--width", str(exp.width), "--height", str(exp.height),
               "--seg-weight", str(exp.seg_weight), "--batch", str(args.batch),
               "--workers", str(args.workers), "--device", args.device,
               "--seed", str(args.seed), "--lr", str(args.lr),
               "--weight-decay", str(args.weight_decay), "--patience", str(epochs + 1)]
    if args.train_limit is not None:
        command += ["--train-limit", str(args.train_limit)]
    if args.val_limit is not None:
        command += ["--val-limit", str(args.val_limit)]
    if args.dry_run:
        run_command(command, True)
        return run_dir

    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(f"Existing run has different settings: {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Existing run has no run_config.json: {run_dir}")
    history = records(run_dir / "history.jsonl")
    if history and history[-1]["epoch"] >= epochs:
        print(f"Skipping completed {stage}/{exp.name}: {run_dir}", flush=True)
        return run_dir
    if history:
        last = run_dir / "last.pt"
        if not args.resume_incomplete or not last.is_file():
            raise RuntimeError(f"Incomplete run in {run_dir}; use --resume-incomplete")
        command += ["--resume", str(last)]
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run_command(command, False)
    if not records(run_dir / "history.jsonl") or records(run_dir / "history.jsonl")[-1]["epoch"] != epochs:
        raise RuntimeError(f"Training did not finish {epochs} epochs: {run_dir}")
    return run_dir


def evaluate_checkpoint(checkpoint: Path, split: str, args, output: Path):
    command = [sys.executable, str(EVALUATE_SCRIPT), "--checkpoint", str(checkpoint),
               "--data", str(resolved(args.data)), "--split", split, "--batch", str(args.batch),
               "--workers", str(args.workers), "--device", args.device, "--output", str(output)]
    limit = args.test_limit if split == "test" else args.val_limit
    if limit is not None:
        command += ["--limit", str(limit)]
    run_command(command, args.dry_run)


def main():
    args = arguments()
    if args.pilot_epochs < 1 or args.final_epochs < 1 or args.top_k < 1 or args.batch < 1:
        raise ValueError("epoch counts, top-k and batch must be positive")
    selected = [e for e in EXPERIMENTS if e.name in args.experiments]
    output_root = resolved(args.output_root)
    summary = {"pilot": {}, "final": {}, "winner": None}

    if args.stage in ("all", "pilot"):
        for exp in selected:
            print(f"\nPilot {exp.name}: {exp.description}", flush=True)
            run_dir = train_experiment(exp, "pilot", args.pilot_epochs, args, output_root)
            if not args.dry_run:
                summary["pilot"][exp.name] = best_record(run_dir)
        if not args.dry_run:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "pilot_summary.json").write_text(
                json.dumps(summary["pilot"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.stage in ("all", "final"):
        if args.dry_run and args.stage == "all":
            print("Final candidates are selected after pilot metrics are available.")
            return
        pilot = {e.name: best_record(output_root / "pilot" / e.name) for e in selected}
        summary["pilot"] = pilot
        candidates = [e for e in selected if e.seg_weight > 0]
        if not candidates:
            raise ValueError("Final selection needs at least one multitask experiment (E0-E3)")
        candidates.sort(key=lambda e: pilot[e.name]["score"], reverse=True)
        finalists = candidates[:args.top_k]
        if args.full_det_baseline and any(e.name == "e4" for e in selected):
            finalists += [next(e for e in selected if e.name == "e4")]
        print("Finalists:", ", ".join(e.name for e in finalists), flush=True)
        for exp in finalists:
            print(f"\nFinal {exp.name}: {exp.description}", flush=True)
            run_dir = train_experiment(exp, "final", args.final_epochs, args, output_root)
            evaluate_checkpoint(run_dir / "best.pt", "val", args, run_dir / "val_metrics.json")
            if not args.dry_run:
                summary["final"][exp.name] = best_record(run_dir)
        if args.dry_run:
            return
        winner = max((e for e in finalists if e.seg_weight > 0),
                     key=lambda e: summary["final"][e.name]["score"])
        summary["winner"] = winner.name
        winner_dir = output_root / "final" / winner.name
        evaluate_checkpoint(winner_dir / "best.pt", "test", args, winner_dir / "test_metrics.json")

    if not args.dry_run:
        path = output_root / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults: {path}", flush=True)


if __name__ == "__main__":
    main()
