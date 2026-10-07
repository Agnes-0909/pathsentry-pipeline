#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/configs/rdk_x5.env"
BOARD=${BOARD:-root@192.168.1.12}; REMOTE=${REMOTE:-/opt/cv_pipeline}; FRAMES=${FRAMES:-100}; HOST=${CAMERA_HOST:-0}
PASS=${BOARD_PASSWORD:-root}; SOCK=/tmp/cv_pipeline_left.sock; START_DELAY=$INFERENCE_START_DELAY
BIN_DIR=${BIN_DIR:-$REMOTE/bin}
case "$CPU_GOVERNOR" in performance|schedutil|keep) ;; *) echo "CPU_GOVERNOR must be performance, schedutil or keep" >&2; exit 2;; esac
case "$SAVE_RESULTS" in async|sync|off) ;; *) echo "SAVE_RESULTS must be async, sync or off" >&2; exit 2;; esac
case "$LOG_LEVEL" in info|debug) ;; *) echo "LOG_LEVEL must be info or debug" >&2; exit 2;; esac
if [[ "$RT_PRIORITY" -gt 0 ]]; then
  CAPTURE_LAUNCH="chrt -r $RT_PRIORITY taskset -c $CAPTURE_CPUS"
  INFERENCE_LAUNCH="chrt -r $RT_PRIORITY taskset -c $INFERENCE_CPUS"
else
  CAPTURE_LAUNCH="taskset -c $CAPTURE_CPUS"
  INFERENCE_LAUNCH="taskset -c $INFERENCE_CPUS"
fi
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$BOARD" "export CV_LOG_LEVEL='$LOG_LEVEL'; export LD_LIBRARY_PATH='$REMOTE/3rdlibrary/DNN/lib:$REMOTE/3rdlibrary/RDK_CAMERA/lib:/usr/lib:/usr/hobot/lib'; echo performance > /sys/class/devfreq/3a000000.bpu/governor 2>/dev/null || true; if [ '$CPU_GOVERNOR' != keep ]; then echo '$CPU_GOVERNOR' > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor || exit 1; fi; pkill -x capture_service || true; pkill -x inference_service || true; rm -f '$SOCK'; mkdir -p '$REMOTE/logs/results'; : > '$REMOTE/logs/capture.log'; : > '$REMOTE/logs/inference.log'; rm -f '$REMOTE/logs/results'/frame_*.json '$REMOTE/logs/results'/frame_*.pgm; nohup $CAPTURE_LAUNCH '$BIN_DIR/capture_service' --host '$HOST' --socket '$SOCK' --frames '$FRAMES' --warmup '$WARMUP' --queue-capacity '$QUEUE_CAPACITY' --log '$REMOTE/logs/capture.log' >/dev/null 2>&1 & sleep '$START_DELAY'; nohup $INFERENCE_LAUNCH '$BIN_DIR/inference_service' --socket '$SOCK' --model '$REMOTE/models/h2_raw_head.bin' --log '$REMOTE/logs/inference.log' --output-dir '$REMOTE/logs/results' --save-results '$SAVE_RESULTS' >/dev/null 2>&1 &"
