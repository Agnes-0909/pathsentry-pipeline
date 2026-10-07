#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); BOARD=${BOARD:-root@192.168.1.12}; REMOTE=${REMOTE:-/opt/cv_pipeline}; PASS=${BOARD_PASSWORD:-root}; OUT=${OUT:-$ROOT/logs/board_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"
for name in capture.log inference.log; do for n in 1 2 3; do sshpass -p "$PASS" scp -q -o StrictHostKeyChecking=no "$BOARD:$REMOTE/logs/$name" "$OUT/$name" && break || sleep 1; done; done
mkdir -p "$OUT/results"
for n in 1 2 3; do sshpass -p "$PASS" scp -q -r -o StrictHostKeyChecking=no "$BOARD:$REMOTE/logs/results/." "$OUT/results/" && break || sleep 1; done
echo "$OUT"
