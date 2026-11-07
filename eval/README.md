# Online evaluation

## Online-Mind2Web, 109 tasks

`online_mind2web/stable109_task_ids.json` is the 109-task subset used in the paper: 25 easy, 48 medium, 36 hard. `submission_schema.json` is the trajectory file schema.

The decision loop is candidate, then MKD2World for every candidate when more than one is proposed, then digest, then best. A single high-confidence candidate skips the world model. Check that loop without a website:

```bash
PYTHONPATH=. python eval/check_online_mind2web_109.py
```

A real run needs an OpenAI-compatible GUI agent and an MKD2World server loaded with `localwm_next`. Output length is 6000 tokens. Playwright must be able to launch Chromium.

```bash
export MKD2WORLD_VLLM_ENDPOINTS=http://127.0.0.1:8000
export MKD2WORLD_VLLM_MODEL=mkd2world-9b
export OPENAI_API_KEY=...
PYTHONPATH=. python eval/run_online_mind2web_mkd2world_vllm.py \
  --endpoint http://127.0.0.1:8001 \
  --model gui-agent \
  --agent-strategy p2_digest \
  --task-manifest eval/online_mind2web/stable109_task_ids.json \
  --output-dir /tmp/om2w-run \
  --local-max-new-tokens 6000 \
  --local-wm-prompt-key localwm_next \
  --local-wm-output-state-mode replayed_next
```

`--agent-strategy p2_digest` is the paper loop. `--mode` only selects screenshot input.

## Bare 9B plus MKD2World with digest

`configs/om2w_qwen9b_mkd2world_digest.json` follows the 2026-09-12 Qwen3.5-9B plus MKD2World P2.1 digest run. The agent is the base Qwen3.5-9B with no TSM adapter. Agent temperature is 0, thinking is on at medium with a 4096-token budget, context is 32768, and the completion limit is 5000. The world model is the step-700 adapter, temperature 0, thinking off, context 60000, and 6000 new tokens. Viewport is 1280x720. Prompts are `p2.1` candidate, digest, and best.

```bash
export MKD2WORLD_AGENT_BASE=/path/to/Qwen3.5-9B
export MKD2WORLD_WM_BASE=/path/to/Qwen3.5-9B
export MKD2WORLD_WM_ADAPTER=/path/to/mkd2world
export MKD2WORLD_OUTPUT_DIR=/path/outside/this/repo
bash eval/run_om2w_qwen9b_mkd2world_digest.sh
```

The recorded run used two shards and three browsers per shard. This script defaults to one agent GPU and concurrency 1. Set `MKD2WORLD_AGENT_GPU`, `MKD2WORLD_WM_GPU`, and `MKD2WORLD_CONCURRENCY` to match the machine.

## Bare 35B plus MKD2World with digest

`configs/om2w_qwen35b_mkd2world_digest.json` is the last complete Online-Mind2Web chain with a bare Qwen3.5-35B-A3B agent, no TSM adapter, MKD2World, and P2.1 digest. Agent temperature is 0, thinking is on at medium with a 4096-token budget, context is 32768, and the completion limit is 5000. The world model uses temperature 0, thinking off, context 60000, and 6000 new tokens. Viewport is 1280x720.

```bash
export MKD2WORLD_AGENT_BASE=/path/to/Qwen3.5-35B-A3B
export MKD2WORLD_WM_BASE=/path/to/Qwen3.5-9B
export MKD2WORLD_WM_ADAPTER=/path/to/mkd2world
export MKD2WORLD_OUTPUT_DIR=/path/outside/this/repo
bash eval/run_om2w_qwen35b_mkd2world_digest.sh
```

This script defaults to one agent process and concurrency 1. Set `MKD2WORLD_AGENT_GPUS`, `MKD2WORLD_AGENT_TP`, `MKD2WORLD_WM_GPU`, and `MKD2WORLD_CONCURRENCY` to match the machine.

## VisualWebArena, 233 tasks

`vwa_mkd2world_p21_runner.py` uses the same candidate, world-model, digest, and best calls. The shopping, forum, and classifieds containers are not in this tree, so a full 233-task run needs those sites separately. Sampling defaults are in `vwa_eval_defaults.json`: agent temperature 1, thinking on, 50 steps, world model temperature 0 and 6000 tokens.

`eval/examples/pipeline_vwa_mkd2world_p21.json` and `eval/run_vwa_qwen35b_mkd2world_digest.sh` start the bare 35B agent and MKD2World digest loop. Agent temperature is 1. Viewport is 1280x2048. The script does not start the website containers. Set `MKD2WORLD_VWA_TASKS` to the task json files and `MKD2WORLD_VWA_ROOT` to an existing VisualWebArena checkout whose sites are already running.
