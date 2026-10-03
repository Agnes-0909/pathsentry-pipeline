#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/huace/new_storage/ps_proj/qwen2_5_7B_prelabel"
INPUT_DIR="/home/huace/new_storage/ps_datasets/custom/left"
MODEL_DIR="${PROJECT_DIR}/model_dir"
OUTPUT_DIR="${PROJECT_DIR}/outputs/left_sample_200"
VLLM_ENV="${VLLM_ENV:-/home/huace/miniconda3/envs/vllm}"

cd "${PROJECT_DIR}"

export LD_LIBRARY_PATH="${VLLM_ENV}/lib/python3.12/site-packages/nvidia/cu13/lib:${VLLM_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

conda run -n vllm qwen25-prelabel "${INPUT_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-size 200 \
  --min-confidence 0.0 \
  --seed 20260830
