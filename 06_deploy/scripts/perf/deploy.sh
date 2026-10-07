#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-$ROOT/../3rdlibrary}
BOARD=${BOARD:-root@192.168.1.12}; REMOTE=${REMOTE:-/opt/cv_pipeline}; PASS=${BOARD_PASSWORD:-root}
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$BOARD" "mkdir -p '$REMOTE/bin' '$REMOTE/models' '$REMOTE/logs' '$REMOTE/3rdlibrary/DNN' '$REMOTE/3rdlibrary/RDK_CAMERA'"
copy_retry(){ local src=$1 dst=$2; for n in 1 2 3; do sshpass -p "$PASS" scp -q -o StrictHostKeyChecking=no "$src" "$BOARD:$dst" && return 0; sleep 1; done; return 1; }
copy_retry "$ROOT/build_aarch64/capture_service" "$REMOTE/bin/capture_service"
copy_retry "$ROOT/build_aarch64/inference_service" "$REMOTE/bin/inference_service"
copy_retry "$ROOT/models/h2_raw_head.bin" "$REMOTE/models/h2_raw_head.bin"
( cd "$THIRD_PARTY_ROOT/DNN" && COPYFILE_DISABLE=1 tar cf - lib ) | sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$BOARD" "tar xf - -C '$REMOTE/3rdlibrary/DNN'"
( cd "$THIRD_PARTY_ROOT/RDK_CAMERA" && COPYFILE_DISABLE=1 tar cf - lib ) | sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$BOARD" "tar xf - -C '$REMOTE/3rdlibrary/RDK_CAMERA'"
