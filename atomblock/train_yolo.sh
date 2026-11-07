#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-MKD2WORLD_EXTERNAL_ROOT/envs/yolo/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/tsm_renderer/training/train_atomblock_yolo.py}"

DATA="${DATA:-MKD2WORLD_EXTERNAL_ROOT/datasets/AtomBlock-WebUI/yolo-dataset/data.yaml}"
MODEL="${MODEL:-MKD2WORLD_EXTERNAL_ROOT/models/YOLO26/yolo26m.pt}"
EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256:-}"
PROJECT="${PROJECT:-MKD2WORLD_EXTERNAL_ROOT/output/yolo/atomblock}"
NAME="${NAME:-yolo26m_img1920_seed0_8gpu_b64}"
DEPLOY_PATH="${DEPLOY_PATH:-MKD2WORLD_EXTERNAL_ROOT/models/yolo/atomblock_yolo26m_best.pt}"
RESUME="${RESUME:-}"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1920}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0,1,2,3,4,5,6,7}"
EVAL_DEVICE="${EVAL_DEVICE:-0}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-0}"
PATIENCE="${PATIENCE:-30}"
SAVE_PERIOD="${SAVE_PERIOD:-2}"
FRACTION="${FRACTION:-1.0}"
DEPLOY_CONF="${DEPLOY_CONF:-0.1}"
DEPLOY_IOU="${DEPLOY_IOU:-0.1}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
MAX_AUTO_GPUS="${MAX_AUTO_GPUS:-0}"
REQUIRE_EXCLUSIVE_GPUS="${REQUIRE_EXCLUSIVE_GPUS:-0}"
MIN_FREE_PER_GPU_MB="${MIN_FREE_PER_GPU_MB:-0}"

RECT="${RECT:-0}"
VAL="${VAL:-1}"
PLOTS="${PLOTS:-1}"
EXIST_OK="${EXIST_OK:-1}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

DRY_RUN="${DRY_RUN:-1}"
RUN_IN_TMUX="${RUN_IN_TMUX:-0}"
TMUX_SESSION="${TMUX_SESSION:-atomblock_yolo26_train}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/yolo/atomblock}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log}"

args=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  --data "${DATA}"
  --model "${MODEL}"
  --project "${PROJECT}"
  --name "${NAME}"
  --deploy-path "${DEPLOY_PATH}"
  --epochs "${EPOCHS}"
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --device "${DEVICE}"
  --eval-device "${EVAL_DEVICE}"
  --workers "${WORKERS}"
  --seed "${SEED}"
  --patience "${PATIENCE}"
  --save-period "${SAVE_PERIOD}"
  --fraction "${FRACTION}"
  --deploy-conf "${DEPLOY_CONF}"
  --deploy-iou "${DEPLOY_IOU}"
  --min-free-mb "${MIN_FREE_MB}"
  --max-auto-gpus "${MAX_AUTO_GPUS}"
)

[[ -n "${RESUME}" ]] && args+=(--resume "${RESUME}")
[[ -n "${EXPECTED_MODEL_SHA256}" ]] && args+=(--expected-model-sha256 "${EXPECTED_MODEL_SHA256}")
[[ "${RECT}" == "1" || "${RECT}" == "true" ]] && args+=(--rect) || args+=(--no-rect)
[[ "${VAL}" == "1" || "${VAL}" == "true" ]] && args+=(--val) || args+=(--no-val)
[[ "${PLOTS}" == "1" || "${PLOTS}" == "true" ]] && args+=(--plots) || args+=(--no-plots)
[[ "${EXIST_OK}" == "1" || "${EXIST_OK}" == "true" ]] && args+=(--exist-ok)
[[ "${SKIP_DEPLOY}" == "1" || "${SKIP_DEPLOY}" == "true" ]] && args+=(--skip-deploy)
[[ "${SKIP_EVAL}" == "1" || "${SKIP_EVAL}" == "true" ]] && args+=(--skip-eval)
[[ "${SKIP_EXPORT}" == "1" || "${SKIP_EXPORT}" == "true" ]] && args+=(--skip-export)
[[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]] && args+=(--dry-run)

printf 'YOLO train command:\n'
printf ' %q' "${args[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

preflight_explicit_gpus() {
  local requested_device="$1"
  if [[ "${requested_device}" == "cpu" || "${requested_device}" == auto* ]]; then
    return 0
  fi

  declare -A uuid_by_index=()
  declare -A free_by_index=()
  declare -A occupied_by_uuid=()
  local index uuid free_mb pid
  while IFS=',' read -r index uuid free_mb; do
    index="${index//[[:space:]]/}"
    uuid="${uuid//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    [[ -n "${index}" && -n "${uuid}" ]] || continue
    uuid_by_index["${index}"]="${uuid}"
    free_by_index["${index}"]="${free_mb}"
  done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits)

  while IFS=',' read -r uuid pid; do
    uuid="${uuid//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    [[ -n "${uuid}" && -n "${pid}" ]] || continue
    occupied_by_uuid["${uuid}"]+="${pid} "
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)

  local failed=0
  IFS=',' read -ra requested_indices <<< "${requested_device}"
  for index in "${requested_indices[@]}"; do
    index="${index//[[:space:]]/}"
    uuid="${uuid_by_index[${index}]:-}"
    free_mb="${free_by_index[${index}]:-0}"
    if [[ -z "${uuid}" ]]; then
      echo "GPU preflight failed: requested GPU ${index} was not reported by nvidia-smi" >&2
      failed=1
      continue
    fi
    if [[ "${MIN_FREE_PER_GPU_MB}" -gt 0 && "${free_mb}" -lt "${MIN_FREE_PER_GPU_MB}" ]]; then
      echo "GPU preflight failed: GPU ${index} has ${free_mb} MiB free, needs ${MIN_FREE_PER_GPU_MB} MiB" >&2
      failed=1
    fi
    if [[ "${REQUIRE_EXCLUSIVE_GPUS}" == "1" || "${REQUIRE_EXCLUSIVE_GPUS}" == "true" ]]; then
      if [[ -n "${occupied_by_uuid[${uuid}]:-}" ]]; then
        echo "GPU preflight failed: GPU ${index} has compute PID(s) ${occupied_by_uuid[${uuid}]}" >&2
        failed=1
      fi
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    return 3
  fi
  echo "GPU preflight passed: device=${requested_device}, exclusive=${REQUIRE_EXCLUSIVE_GPUS}, min_free_mb=${MIN_FREE_PER_GPU_MB}"
}

preflight_explicit_gpus "${DEVICE}"

if [[ "${RUN_IN_TMUX}" == "1" || "${RUN_IN_TMUX}" == "true" ]]; then
  mkdir -p "${LOG_DIR}"
  tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
  command_text="$(printf ' %q' "${args[@]}")"
  tmux new-session -d -s "${TMUX_SESSION}" \
    "cd ${PROJECT_ROOT@Q} && export PYTHONPATH=${PYTHONPATH@Q} PYTHONUNBUFFERED=1 &&${command_text} 2>&1 | tee ${LOG_FILE@Q}"
  echo "tmux session: ${TMUX_SESSION}"
  echo "log: ${LOG_FILE}"
else
  "${args[@]}"
fi
