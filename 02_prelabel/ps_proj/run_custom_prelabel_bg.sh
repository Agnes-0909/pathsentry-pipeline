#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/huace/new_storage/ps_proj"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/custom_prelabel_${TS}.log"
PID_FILE="$LOG_DIR/custom_prelabel_${TS}.pid"

RUNNER="$(mktemp "$LOG_DIR/custom_prelabel_runner_XXXXXX.sh")"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/huace/miniconda3/etc/profile.d/conda.sh
export VLLM_ENV="/home/huace/miniconda3/envs/vllm"
export LD_LIBRARY_PATH="\$VLLM_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:\$VLLM_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
exec conda run --no-capture-output -n vllm python -u "$ROOT/custom_prelabel.py" "\$@"
EOF
chmod +x "$RUNNER"

setsid nohup "$RUNNER" "$@" >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
echo "$PID" >"$PID_FILE"
echo "PID=$PID"
echo "LOG=$LOG_FILE"
echo "PID_FILE=$PID_FILE"
