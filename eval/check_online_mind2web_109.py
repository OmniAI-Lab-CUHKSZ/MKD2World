from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "eval/online_mind2web/tasks.json"
MANIFEST = ROOT / "eval/online_mind2web/stable109_task_ids.json"
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x04"
    b"\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

CLICK_A = {"type": "left_click", "coordinate": [100, 120]}
CLICK_B = {"type": "left_click", "coordinate": [400, 220]}
FORECAST = "[Environment: Web]\n- [Go](button)<100,100,80,30>\n"


def check_task_split() -> None:
    import eval.run_online_mind2web_vanilla as core

    ids = core.load_task_ids(MANIFEST)
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    by_id = {str(row["task_id"]): row for row in tasks}
    missing = [item for item in ids if item not in by_id]
    if missing:
        raise SystemExit(f"stable109 ids missing from tasks: {len(missing)}")
    counts = Counter(str(by_id[item]["level"]) for item in ids)
    expected = {"easy": 25, "medium": 48, "hard": 36}
    if dict(counts) != expected or len(ids) != 109:
        raise SystemExit(f"stable109 split {len(ids)} {dict(counts)}")
    print("stable109", len(ids), expected, flush=True)


class _Handler(BaseHTTPRequestHandler):
    stages: list[str] = []

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        system = ""
        user = ""
        for message in messages:
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            if message.get("role") == "system":
                system = text
            else:
                user += text
        if "Digest mode" in system:
            stage = "digest"
            body = {
                "predicted_change": "The button stays visible.",
                "intent_alignment": "partial",
                "task_progress": "neutral",
                "evidence_refs": ["- [Go](button)<100,100,80,30>"],
            }
        elif "Select the candidate" in system:
            stage = "best"
            body = {
                "best_action": {
                    "candidate_index": 1,
                    "unified_action": CLICK_B,
                    "action_intent": "Open the second control.",
                }
            }
        elif "Generate candidate" in system:
            stage = "candidate"
            body = {
                "candidate_actions": [
                    {
                        "unified_action": CLICK_A,
                        "action_intent": "Open the first control.",
                        "confidence": 4,
                    },
                    {
                        "unified_action": CLICK_B,
                        "action_intent": "Open the second control.",
                        "confidence": 5,
                    },
                ]
            }
        elif "Predict the complete visible interface" in user or "WM_Local" in system:
            stage = "world"
            body_text = FORECAST
            self._send(body_text, stage)
            return
        else:
            stage = "unknown"
            body = {}
        self._send(json.dumps(body), stage)

    def _send(self, content: str, stage: str) -> None:
        _Handler.stages.append(stage)
        encoded = json.dumps(
            {
                "id": "chk",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def runner_config(endpoint: str):
    import eval.run_online_mind2web_vanilla as core

    return core.RunnerConfig(
        endpoint=endpoint,
        model="gui-agent",
        prompt="candidate",
        mode="screenshot",
        markdown_endpoint=None,
        width=1280,
        height=2048,
        max_steps=50,
        navigation_timeout_ms=30000,
        action_wait_ms=1000,
        model_timeout_seconds=30,
        model_retries=0,
        reasoning_effort="medium",
        thinking_token_budget=4096,
        max_completion_tokens=5000,
        temperature=1.0,
        enable_thinking=True,
        agent_context_tokens=32768,
        no_progress_limit=4,
        compact_vanilla_input=True,
        gui_agent_use_current_md=False,
    )


async def run_om2w(endpoint: str) -> None:
    import eval.run_online_mind2web_mkd2world_vllm as world
    import eval.run_online_mind2web_vanilla as core

    os.environ["MKD2WORLD_VLLM_ENDPOINTS"] = endpoint
    os.environ["MKD2WORLD_VLLM_MODEL"] = "mkd2world-9b"
    core.P2DigestAgent.__bases__ = (world.VllmLocalWMAgent,)
    prompts = ROOT / "mkd2world/prompts/p2.1"
    _Handler.stages = []
    agent = core.P2DigestAgent(
        runner_config(endpoint),
        candidate_prompt=(prompts / "candidate.md").read_text(encoding="utf-8"),
        digest_prompt=(prompts / "digest.md").read_text(encoding="utf-8"),
        best_prompt=(prompts / "best.md").read_text(encoding="utf-8"),
        local_base_model=Path("Qwen3.5-9B"),
        local_adapter=None,
        local_base_only=False,
        local_batch_size=1,
        local_max_length=60000,
        local_max_new_tokens=6000,
        local_wm_prompt_key="localwm_next",
        local_wm_output_state_mode="replayed_next",
        local_wm_devices=("vllm:0",),
        policy_version="p2.1",
    )
    try:
        selected, _raw, _latency = await agent.choose(
            "Find a recipe",
            "https://example.test/",
            [],
            PNG,
            "",
        )
    finally:
        await agent.close()
    action = selected["selected_action"]["unified_action"]
    if _Handler.stages[0] != "candidate" or _Handler.stages[-1] != "best":
        raise SystemExit(f"OM2W stage order {_Handler.stages}")
    if _Handler.stages.count("world") != 2 or _Handler.stages.count("digest") != 2:
        raise SystemExit(f"OM2W stages {_Handler.stages}")
    if action.get("type") != "left_click" or action.get("coordinate") != [400, 220]:
        raise SystemExit(f"OM2W selected {action}")
    print("om2w_chain", _Handler.stages, action["type"], flush=True)


async def run_vwa(endpoint: str) -> None:
    import eval.vwa_mkd2world_p21_runner as vwa

    prompts = ROOT / "mkd2world/prompts/p2.1"
    world_prompt = (
        ROOT / "mkd2world/prompts/templatesV3/train_localwm/localwm_next.md"
    ).read_text(encoding="utf-8")
    _Handler.stages = []
    agent = vwa.RemoteP2DigestAgent(
        runner_config(endpoint),
        candidate_prompt=(prompts / "candidate.md").read_text(encoding="utf-8"),
        digest_prompt=(prompts / "digest.md").read_text(encoding="utf-8"),
        best_prompt=(prompts / "best.md").read_text(encoding="utf-8"),
        world_prompt=world_prompt,
        world_endpoints=(endpoint,),
        world_model="mkd2world-9b",
        world_max_tokens=6000,
        world_timeout_seconds=30,
        policy_version="p2.1",
    )
    try:
        selected, _raw, _latency = await agent.choose(
            "Find a recipe",
            "https://example.test/",
            [],
            PNG,
            "",
        )
    finally:
        await agent.close()
    action = selected["selected_action"]["unified_action"]
    if _Handler.stages.count("world") != 2 or _Handler.stages[-1] != "best":
        raise SystemExit(f"VWA stages {_Handler.stages}")
    if action.get("coordinate") != [400, 220]:
        raise SystemExit(f"VWA selected {action}")
    print("vwa_chain", _Handler.stages, flush=True)


def main() -> None:
    check_task_split()
    server, endpoint = serve()
    try:
        asyncio.run(run_om2w(endpoint))
        asyncio.run(run_vwa(endpoint))
    finally:
        server.shutdown()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
