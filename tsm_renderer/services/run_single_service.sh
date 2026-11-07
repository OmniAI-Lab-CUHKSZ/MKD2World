#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ENVS_DIR="${PROJECT_ENVS_DIR:-${PROJECT_ROOT}/envs}"
SERVICE="${1:-${MD_RENDER_SERVICE:-}}"

if [[ -z "${SERVICE}" ]]; then
  echo "usage: $0 yolo|ocr|qwen" >&2
  exit 2
fi

case "${SERVICE}" in
  yolo)
    YOLO_ENV="${YOLO_ENV:-${PROJECT_ENVS_DIR}/yolo}"
    YOLO_PYTHON="${YOLO_PYTHON:-${YOLO_ENV}/bin/python}"
    YOLO_PORT="${YOLO_PORT:-8001}"
    YOLO_DEVICE="${YOLO_DEVICE:-cuda:0}"
    exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      PYTHONUNBUFFERED=1 \
      YOLO_PORT="${YOLO_PORT}" \
      YOLO_DEVICE="${YOLO_DEVICE}" \
      "${YOLO_PYTHON}" "${SCRIPT_DIR}/yolo_service.py"
    ;;
  ocr)
    OCR_ENV="${OCR_ENV:-${PROJECT_ENVS_DIR}/gui_ocr_gpu}"
    OCR_PYTHON="${OCR_PYTHON:-${OCR_ENV}/bin/python}"
    OCR_PORT="${OCR_PORT:-8002}"
    OCR_CUDA_VISIBLE_DEVICES="${OCR_CUDA_VISIBLE_DEVICES:-0}"
    OCR_PROCESS_MODE="${OCR_PROCESS_MODE:-inline}"
    OCR_WORKERS="${OCR_WORKERS:-1}"
    OCR_READY_TIMEOUT="${OCR_READY_TIMEOUT:-900}"
    exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="${OCR_CUDA_VISIBLE_DEVICES}" \
      OCR_PORT="${OCR_PORT}" \
      OCR_PROCESS_MODE="${OCR_PROCESS_MODE}" \
      OCR_WORKERS="${OCR_WORKERS}" \
      OCR_READY_TIMEOUT="${OCR_READY_TIMEOUT}" \
      OCR_USE_DOC_ORIENTATION_CLASSIFY="${OCR_USE_DOC_ORIENTATION_CLASSIFY:-0}" \
      OCR_USE_DOC_UNWARPING="${OCR_USE_DOC_UNWARPING:-0}" \
      OCR_USE_TEXTLINE_ORIENTATION="${OCR_USE_TEXTLINE_ORIENTATION:-0}" \
      "${OCR_PYTHON}" "${SCRIPT_DIR}/ocr_vl_service.py"
    ;;
  qwen)
    QWEN_ENV="${QWEN_ENV:-${PROJECT_ENVS_DIR}/qwen3.5}"
    QWEN_VLLM="${QWEN_VLLM:-${QWEN_ENV}/bin/vllm}"
    QWEN_MODEL="${QWEN_MODEL:-MKD2WORLD_EXTERNAL_ROOT/models/Qwen3.5-0.8B}"
    QWEN_PORT="${QWEN_PORT:-8000}"
    QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-0}"
    QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen08b}"
    QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.1}"
    QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-2048}"
    QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-2048}"
    QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-32}"
    exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
      PATH="${QWEN_ENV}/bin:${PATH}" \
      "${QWEN_VLLM}" serve "${QWEN_MODEL}" \
        --host 0.0.0.0 \
        --port "${QWEN_PORT}" \
        --served-model-name "${QWEN_SERVED_MODEL_NAME}" \
        --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${QWEN_MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${QWEN_MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${QWEN_MAX_NUM_SEQS}" \
        ${QWEN_EXTRA_ARGS:-}
    ;;
  *)
    echo "unknown tsm_renderer service: ${SERVICE}" >&2
    exit 2
    ;;
esac
