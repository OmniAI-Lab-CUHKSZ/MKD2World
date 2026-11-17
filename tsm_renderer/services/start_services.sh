#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${MD_RENDER_SERVICE:-}" ]]; then
  cat >&2 <<'EOF'
start_services.sh no longer starts a bundled YOLO/OCR/Qwen service group.

Start each service as an independent foreground process instead:

  MD_RENDER_SERVICE=yolo YOLO_DEVICE=cuda:0 YOLO_PORT=8001 \
    tsm_renderer/services/start_services.sh

  MD_RENDER_SERVICE=ocr OCR_CUDA_VISIBLE_DEVICES=0 OCR_PORT=8002 \
    tsm_renderer/services/start_services.sh

  MD_RENDER_SERVICE=qwen QWEN_CUDA_VISIBLE_DEVICES=0 QWEN_PORT=8000 \
    tsm_renderer/services/start_services.sh

The script uses exec, so it does not remain as a parent supervisor process.
EOF
  exit 2
fi

exec "${SCRIPT_DIR}/run_single_service.sh" "${MD_RENDER_SERVICE}"
