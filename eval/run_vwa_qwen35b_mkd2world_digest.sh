set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${MKD2WORLD_PYTHON:-python3}"
VLLM="${MKD2WORLD_VLLM:-vllm}"
AGENT_BASE="${MKD2WORLD_AGENT_BASE:?Set MKD2WORLD_AGENT_BASE to Qwen3.5-35B-A3B}"
WM_BASE="${MKD2WORLD_WM_BASE:?Set MKD2WORLD_WM_BASE to Qwen3.5-9B}"
WM_ADAPTER="${MKD2WORLD_WM_ADAPTER:?Set MKD2WORLD_WM_ADAPTER to the MKD2World step-700 adapter}"
OUT="${MKD2WORLD_OUTPUT_DIR:?Set MKD2WORLD_OUTPUT_DIR outside this repository}"
TASKS="${MKD2WORLD_VWA_TASKS:?Set MKD2WORLD_VWA_TASKS to a task json or a directory of task json files}"
AGENT_PORT="${MKD2WORLD_AGENT_PORT:-18310}"
WM_PORT="${MKD2WORLD_WM_PORT:-18340}"
AGENT_GPUS="${MKD2WORLD_AGENT_GPUS:-0,1}"
AGENT_TP="${MKD2WORLD_AGENT_TP:-}"
WM_GPU="${MKD2WORLD_WM_GPU:-2}"
VWA_ROOT="${MKD2WORLD_VWA_ROOT:-envs/visualwebarena}"

if [[ -z "$AGENT_TP" ]]; then
  AGENT_TP="$(awk -F, "{print NF}" <<<"$AGENT_GPUS")"
fi

mapfile -t TASK_FILES < <(if [[ -d "$TASKS" ]]; then find "$TASKS" -name "*.json" | sort; else printf "%s\n" "$TASKS"; fi)
if [[ "${#TASK_FILES[@]}" -eq 0 ]]; then
  echo "No VWA task json files in $TASKS" >&2
  exit 1
fi

mkdir -p "$OUT"
export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${MKD2WORLD_PLAYWRIGHT_BROWSERS:-}" ]]; then
  export PLAYWRIGHT_BROWSERS_PATH="$MKD2WORLD_PLAYWRIGHT_BROWSERS"
fi
if [[ -n "${MKD2WORLD_BROWSER_LIB:-}" ]]; then
  export LD_LIBRARY_PATH="${MKD2WORLD_BROWSER_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$PYTHON" - "$ROOT/eval/examples/pipeline_vwa_mkd2world_p21.json" "$OUT/pipeline.json" "$AGENT_PORT" "$WM_PORT" "$VWA_ROOT" << "PY"
import json, sys
src, dst, agent_port, wm_port, vwa_root = sys.argv[1:]
data = json.load(open(src, encoding="utf-8"))
data["pipeline"]["agent"]["base_url"] = f"http://127.0.0.1:{agent_port}/v1"
data["pipeline"]["local_world_model"]["base_url"] = f"http://127.0.0.1:{wm_port}/v1"
data["pipeline"]["environment"]["options"]["vwa_root"] = vwa_root
json.dump(data, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(dst, "a", encoding="utf-8").write("\n")
PY

wait_model() {
  local url="$1" name="$2" n
  for n in $(seq 1 180); do
    if curl --noproxy "*" -fsS --max-time 5 "$url/v1/models" | grep -F "$name" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "service not ready: $url ($name)" >&2
  return 1
}

if ! curl --noproxy "*" -fsS --max-time 3 "http://127.0.0.1:${AGENT_PORT}/v1/models" >/dev/null; then
  CUDA_VISIBLE_DEVICES="$AGENT_GPUS" "$VLLM" serve "$AGENT_BASE" \
    --host 127.0.0.1 --port "$AGENT_PORT" \
    --served-model-name qwen3.5-35b-a3b \
    --tensor-parallel-size "$AGENT_TP" \
    --gpu-memory-utilization "${MKD2WORLD_AGENT_GPU_UTIL:-0.55}" \
    --max-model-len 32768 \
    --max-num-seqs "${MKD2WORLD_AGENT_MAX_SEQS:-2}" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt "{\"image\": 4}" \
    >"$OUT/agent_vllm.log" 2>&1 &
  echo $! >"$OUT/agent_vllm.pid"
fi

if ! curl --noproxy "*" -fsS --max-time 3 "http://127.0.0.1:${WM_PORT}/v1/models" >/dev/null; then
  CUDA_VISIBLE_DEVICES="$WM_GPU" "$VLLM" serve "$WM_BASE" \
    --host 127.0.0.1 --port "$WM_PORT" \
    --served-model-name mkd2world-9b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${MKD2WORLD_WM_GPU_UTIL:-0.55}" \
    --max-model-len 60000 \
    --max-num-seqs "${MKD2WORLD_WM_MAX_SEQS:-2}" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt "{\"image\": 1}" \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "mkd2world-9b=${WM_ADAPTER}" \
    >"$OUT/wm_vllm.log" 2>&1 &
  echo $! >"$OUT/wm_vllm.pid"
fi

wait_model "http://127.0.0.1:${AGENT_PORT}" qwen3.5-35b-a3b
wait_model "http://127.0.0.1:${WM_PORT}" mkd2world-9b

TASK_ARGS=()
for task in "${TASK_FILES[@]}"; do
  TASK_ARGS+=(--task-config "$task")
done

echo "VWA site containers are not started by this script." | tee "$OUT/experiment_config.txt"
"$PYTHON" -u "$ROOT/eval/vwa_mkd2world_p21_runner.py" \
  --config "$OUT/pipeline.json" \
  --benchmark visualwebarena \
  --output-dir "$OUT/runs" \
  --method agent_plus_local_wm \
  --max-steps 50 \
  --no-progress-limit 4 \
  --repeat-action-limit 5 \
  --enable-local-wm \
  --no-enable-memory \
  "${TASK_ARGS[@]}" \
  >"$OUT/runner.log" 2>&1
echo "runner_exit=$?" | tee -a "$OUT/runner.log"
