from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "eval"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

import run_online_mind2web_vanilla as core
from mkd2world.contracts import LocalPrediction
from mkd2world.local_wm_runtime import decode_local_prediction
from mkd2world.training.prompts import build_local_world_model_user_text


def vllm_endpoints() -> tuple[str, ...]:
    raw = os.environ.get("MKD2WORLD_VLLM_ENDPOINTS", "").strip()
    return tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())


@dataclass
class _RemoteLocalWMMetadata:
    prompt_key: str = "localwm_next"
    delta_state_mode: str = "replayed_next"

    def telemetry_snapshot(self) -> dict[str, int]:
        return {}


class VllmLocalWMAgent(core.LocalWMAgent):
    """LocalWMAgent with HTTP MKD2World transport and no in-process 9B load."""

    def __init__(
        self,
        config: core.RunnerConfig,
        *,
        candidate_prompt: str,
        best_prompt: str,
        local_base_model: Path,
        local_adapter: Path | None,
        local_base_only: bool,
        local_batch_size: int,
        local_max_length: int,
        local_max_new_tokens: int,
        local_wm_prompt_key: str,
        local_wm_output_state_mode: str,
        local_wm_devices: Sequence[str],
        v3_contract: bool = False,
    ) -> None:
        del local_base_model, local_adapter, local_base_only, local_batch_size
        del local_max_length, local_wm_devices
        core.VanillaAgent.__init__(self, config)
        endpoints = vllm_endpoints()
        if not endpoints:
            raise ValueError("MKD2WORLD_VLLM_ENDPOINTS is required")
        if local_wm_prompt_key != "localwm_next":
            raise ValueError("vLLM local_wm requires localwm_next")
        if local_wm_output_state_mode != "replayed_next":
            raise ValueError("vLLM local_wm requires replayed_next")
        if local_max_new_tokens != 6000:
            raise ValueError("MKD2World max output must be 6000")
        prompt_path = os.environ.get(
            "MKD2WORLD_WORLD_PROMPT",
            str(_ROOT / "mkd2world/prompts/templatesV3/train_localwm/localwm_next.md"),
        )
        self.candidate_prompt = candidate_prompt
        self.best_prompt = best_prompt
        self.v3_contract = v3_contract
        self._world_prompt = Path(prompt_path).read_text(encoding="utf-8")
        self._world_model = os.environ.get("MKD2WORLD_VLLM_MODEL", "mkd2world-9b")
        self._world_max_tokens = 6000
        timeout = int(os.environ.get("MKD2WORLD_VLLM_TIMEOUT", "300"))
        self._wm_clients = tuple(
            AsyncOpenAI(
                base_url=endpoint + "/v1",
                api_key="none",
                timeout=timeout,
                max_retries=0,
            )
            for endpoint in endpoints
        )
        self._wm_cursor = 0
        self.local_wm_devices = tuple(f"vllm:{index}" for index in range(len(self._wm_clients)))
        self.local_wm = _RemoteLocalWMMetadata(
            prompt_key=local_wm_prompt_key,
            delta_state_mode=local_wm_output_state_mode,
        )
        self.local_wms = [self.local_wm]

    async def preload(self) -> None:
        print(
            "LOCAL_WM_REPLICA_POOL_READY "
            + json.dumps(
                {
                    "replica_count": len(self._wm_clients),
                    "transport": "vllm_http",
                    "endpoints": list(vllm_endpoints()),
                    "world_model": self._world_model,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    async def close(self) -> None:
        for client in self._wm_clients:
            await client.close()
        await core.VanillaAgent.close(self)

    async def _http_predict_one(self, replica_index: int, current_state: Any, candidate: Any) -> LocalPrediction:
        client = self._wm_clients[replica_index % len(self._wm_clients)]
        sample = {
            "candidate_action_json": json.dumps(
                candidate.unified_action.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            )
        }
        user_text = build_local_world_model_user_text(sample, prompt_key="localwm_next")
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if current_state.screenshot:
            content.insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(current_state.screenshot).decode("ascii"),
                        "detail": "high",
                    },
                },
            )
        response = await client.chat.completions.create(
            model=self._world_model,
            messages=[
                {"role": "system", "content": self._world_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=self._world_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (response.choices[0].message.content or "").strip()
        analysis, predicted_markdown = decode_local_prediction(
            raw,
            current_md=current_state.markdown or "",
            prompt_key="localwm_next",
            delta_state_mode="replayed_next",
        )
        return LocalPrediction(
            candidate_action=candidate,
            action_analysis=analysis,
            predicted_markdown=predicted_markdown,
            raw_output=raw,
            predicted_delta=None,
            output_state_mode="replayed_next",
        )

    async def _predict_with_local_wm_replica(
        self,
        current_state: Any,
        candidates: Sequence[Any],
    ) -> tuple[list[Any], dict[str, int], dict[str, int], int, str]:
        if not candidates:
            return [], {}, {}, 0, "vllm:none"
        start = self._wm_cursor
        self._wm_cursor += len(candidates)
        predictions = list(
            await asyncio.gather(
                *[
                    self._http_predict_one(start + index, current_state, candidate)
                    for index, candidate in enumerate(candidates)
                ]
            )
        )
        dummy = {"http_calls": len(predictions)}
        replica = start % len(self._wm_clients)
        return predictions, dummy, dummy, replica, f"vllm:{replica}"


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        core.main()
        return
    if not vllm_endpoints():
        raise SystemExit("Set MKD2WORLD_VLLM_ENDPOINTS to the world-model server.")
    core.LocalWMAgent = VllmLocalWMAgent
    # P2DigestAgent subclasses LocalWMAgent at import time; rebind so digest
    # also uses HTTP MKD2World instead of in-process 9B weights.
    if hasattr(core, "P2DigestAgent"):
        core.P2DigestAgent.__bases__ = (VllmLocalWMAgent,)
    core.main()


if __name__ == "__main__":
    main()
