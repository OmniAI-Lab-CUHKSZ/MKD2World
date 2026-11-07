#!/usr/bin/env bash
# Online-Mind2Web stable-109: bare Qwen3.5-9B, no TSM adapter, plus MKD2World with P2.1 digest.
# Sampling matches configs/om2w_qwen9b_mkd2world_digest.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/configs/om2w_qwen9b_mkd2world_digest.json"
PYTHON="${MKD2WORLD_PYTHON:-python3}"
VLLM="${MKD2WORLD_VLLM:-vllm}"
AGENT_BASE="${MKD2WORLD_AGENT_BASE:?Set MKD2WORLD_AGENT_BASE to Qwen3.5-9B}"
WM_BASE="${MKD2WORLD_WM_BASE:?Set MKD2WORLD_WM_BASE to Qwen3.5-9B}"
WM_ADAPTER="${MKD2WORLD_WM_ADAPTER:?Set MKD2WORLD_WM_ADAPTER to the MKD2World step-700 adapter}"
OUT="${MKD2WORLD_OUTPUT_DIR:?Set MKD2WORLD_OUTPUT_DIR outside this repository}"
AGENT_PORT="${MKD2WORLD_AGENT_PORT:-18310}"
WM_PORT="${MKD2WORLD_WM_PORT:-18340}"
AGENT_GPU="${MKD2WORLD_AGENT_GPU:-0}"
WM_GPU="${MKD2WORLD_WM_GPU:-1}"
CONCURRENCY="${MKD2WORLD_CONCURRENCY:-1}"

mkdir -p "$OUT"
cp "$CFG" "$OUT/experiment_config.json"
{
  echo "name=om2w_qwen9b_mkd2world_digest"
  echo "agent_base=$AGENT_BASE"
  echo "agent_adapter=none"
  echo "agent_temperature=0"
  echo "agent_thinking=true"
  echo "agent_reasoning_effort=medium"
  echo "agent_thinking_token_budget=4096"
  echo "agent_context_tokens=32768"
  echo "agent_max_tokens=5000"
  echo "wm_base=$WM_BASE"
  echo "wm_adapter=$WM_ADAPTER"
  echo "wm_temperature=0"
  echo "wm_thinking=false"
  echo "wm_max_length=60000"
  echo "wm_max_new_tokens=6000"
  echo "strategy=p2_digest"
  echo "viewport=1280x720"
  echo "max_steps=50"
  echo "no_progress_limit=4"
  echo "action_wait_ms=2500"
  echo "concurrency=$CONCURRENCY"
} >"$OUT/experiment_config.txt"

export MKD2WORLD_VLLM_ENDPOINTS="http://127.0.0.1:${WM_PORT}"
export MKD2WORLD_VLLM_MODEL="${MKD2WORLD_VLLM_MODEL:-mkd2world-9b}"
export MKD2WORLD_VLLM_TIMEOUT="${MKD2WORLD_VLLM_TIMEOUT:-300}"
export MKD2WORLD_WORLD_PROMPT="$ROOT/mkd2world/prompts/templatesV3/train_localwm/localwm_next.md"
export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${MKD2WORLD_PLAYWRIGHT_BROWSERS:-}" ]]; then
  export PLAYWRIGHT_BROWSERS_PATH="$MKD2WORLD_PLAYWRIGHT_BROWSERS"
fi
if [[ -n "${MKD2WORLD_BROWSER_LIB:-}" ]]; then
  export LD_LIBRARY_PATH="${MKD2WORLD_BROWSER_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

wait_model() {
  local url="$1" name="$2" n
  for n in $(seq 1 180); do
    if curl --noproxy '*' -fsS --max-time 5 "$url/v1/models" | grep -F "$name" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "service not ready: $url ($name)" >&2
  return 1
}

if ! curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${AGENT_PORT}/v1/models" >/dev/null; then
  CUDA_VISIBLE_DEVICES="$AGENT_GPU" "$VLLM" serve "$AGENT_BASE" \
    --host 127.0.0.1 --port "$AGENT_PORT" \
    --served-model-name qwen3.5-9b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${MKD2WORLD_AGENT_GPU_UTIL:-0.50}" \
    --max-model-len 32768 \
    --max-num-seqs "${MKD2WORLD_AGENT_MAX_SEQS:-2}" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 2}' \
    >"$OUT/agent_vllm.log" 2>&1 &
  echo $! >"$OUT/agent_vllm.pid"
fi

if ! curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${WM_PORT}/v1/models" >/dev/null; then
  CUDA_VISIBLE_DEVICES="$WM_GPU" "$VLLM" serve "$WM_BASE" \
    --host 127.0.0.1 --port "$WM_PORT" \
    --served-model-name mkd2world-9b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${MKD2WORLD_WM_GPU_UTIL:-0.55}" \
    --max-model-len 60000 \
    --max-num-seqs "${MKD2WORLD_WM_MAX_SEQS:-2}" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 2}' \
    --enable-lora \
    --max-lora-rank 32 \
    --lora-modules "mkd2world-9b=${WM_ADAPTER}" \
    >"$OUT/wm_vllm.log" 2>&1 &
  echo $! >"$OUT/wm_vllm.pid"
fi

wait_model "http://127.0.0.1:${AGENT_PORT}" qwen3.5-9b
wait_model "http://127.0.0.1:${WM_PORT}" mkd2world-9b

RESUME_ARGS=(--no-resume)
if [[ "${MKD2WORLD_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

"$PYTHON" -u "$ROOT/eval/run_online_mind2web_mkd2world_vllm.py" \
  --dataset "$ROOT/eval/online_mind2web/tasks.json" \
  --schema "$ROOT/eval/online_mind2web/submission_schema.json" \
  --task-manifest "$ROOT/eval/online_mind2web/stable109_task_ids.json" \
  --output-dir "$OUT/attempts" \
  --endpoint "http://127.0.0.1:${AGENT_PORT}" \
  --model qwen3.5-9b \
  --model-path "$AGENT_BASE" \
  --agent-strategy p2_digest \
  --prompt "$ROOT/mkd2world/prompts/p2.1/candidate.md" \
  --digest-prompt "$ROOT/mkd2world/prompts/p2.1/digest.md" \
  --best-prompt "$ROOT/mkd2world/prompts/p2.1/best.md" \
  --digest-contract-prompt-dir "$ROOT/mkd2world/prompts/p2.1" \
  --local-base-model "$WM_BASE" \
  --local-adapter "$WM_ADAPTER" \
  --local-max-length 60000 \
  --local-max-new-tokens 6000 \
  --local-wm-prompt-key localwm_next \
  --local-wm-output-state-mode replayed_next \
  --local-wm-devices cuda:0 \
  --no-gui-agent-use-current-md \
  --mode screenshot \
  --concurrency "$CONCURRENCY" \
  --max-steps 50 \
  --no-progress-limit 4 \
  --viewport-width 1280 \
  --viewport-height 720 \
  --browser-disable-http2 \
  --browser-user-agent "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36" \
  --navigation-retries 2 \
  --entry-wait-until domcontentloaded \
  --action-wait-ms 2500 \
  --model-retries 2 \
  --enable-thinking \
  --reasoning-effort medium \
  --thinking-token-budget 4096 \
  --agent-context-tokens 32768 \
  --max-completion-tokens 5000 \
  --temperature 0 \
  "${RESUME_ARGS[@]}" \
  >"$OUT/runner.log" 2>&1
echo "runner_exit=$?" | tee -a "$OUT/runner.log"
