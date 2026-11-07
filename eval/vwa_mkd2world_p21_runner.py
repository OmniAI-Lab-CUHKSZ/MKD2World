from __future__ import annotations

import asyncio
import base64
import enum
import json
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence


if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return str(self.value)

    enum.StrEnum = StrEnum

from openai import AsyncOpenAI

import mkd2world.experiments.online_sr as online_sr
from mkd2world.actions import UnifiedAction
from mkd2world.contracts import CandidateAction, GuiState, LocalPrediction
from mkd2world.local_wm_runtime import decode_local_prediction
from mkd2world.memory import NoOpMemoryStore
from mkd2world.pipeline import PipelineRunner
from mkd2world.training.prompts import build_local_world_model_user_text
from mkd2world.vwa_runtime import map_vwa_url_to_real
from mkd2world.world_model import NoOpGlobalWorldModel, NoOpLocalWorldModel

from eval import run_online_mind2web_vanilla as p21_core
from eval import vwa_bare_qwen_runner as bare_vwa


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class _RemoteLocalWMMetadata:
    prompt_key: str = "localwm_next"
    delta_state_mode: str = "replayed_next"


class RemoteP2DigestAgent(p21_core.P2DigestAgent):

    def __init__(
        self,
        config: p21_core.RunnerConfig,
        *,
        candidate_prompt: str,
        digest_prompt: str,
        best_prompt: str,
        world_prompt: str,
        world_endpoints: Sequence[str],
        world_model: str,
        world_max_tokens: int,
        world_timeout_seconds: int,
        policy_version: str = "p2.1",
    ) -> None:

        p21_core.VanillaAgent.__init__(self, config)
        if not world_endpoints:
            raise ValueError("MKD2World requires at least one inference endpoint")
        if world_max_tokens != 6000:
            raise ValueError("MKD2World world-model max output must be 6000")
        if policy_version not in {"p2.1", "p2.4"}:
            raise ValueError(f"unsupported MKD2World policy version: {policy_version}")
        self.agent_clients = tuple(AsyncOpenAI(base_url=config.endpoint.rstrip("/") + "/v1", api_key=os.environ.get("OPENAI_API_KEY", "none"), timeout=config.model_timeout_seconds) for _ in range(3))
        self._agent_client_cursor = 0
        self.candidate_prompt = candidate_prompt
        self.digest_prompt = digest_prompt
        self.best_prompt = best_prompt
        self.policy_version = policy_version
        self.world_prompt = world_prompt
        self.world_endpoints = tuple(endpoint.rstrip("/") for endpoint in world_endpoints)
        self.world_model = world_model
        self.world_max_tokens = world_max_tokens
        self.world_clients = tuple(
            AsyncOpenAI(
                base_url=endpoint.rstrip("/") + "/v1",
                api_key="none",
                timeout=world_timeout_seconds,
            )
            for endpoint in self.world_endpoints
        )
        self._available_world_clients: asyncio.Queue[int] = asyncio.Queue()
        for index in range(len(self.world_clients)):
            self._available_world_clients.put_nowait(index)
        self.local_wm = _RemoteLocalWMMetadata()
        self.local_wms: list[Any] = []
        self.reference_images: list[dict[str, Any]] = []
        self._telemetry: dict[str, Any] = {
            "gui_calls": 0,
            "world_calls": 0,
            "world_prompt_tokens": 0,
            "world_completion_tokens": 0,
            "world_output_limit_hits": 0,
            "best_intent_normalizations": 0,
            "configured_world_max_tokens": world_max_tokens,
        }
        self._stage_latency_requests: dict[str, list[float]] = {}

    async def preload(self) -> None:
        return None

    def set_reference_images(self, rows: Sequence[dict[str, Any]]) -> None:
        self.reference_images = [dict(row) for row in rows]
        if len(self.reference_images) > 3:
            raise ValueError("VWA supports at most three task-reference images")

    def telemetry_snapshot(self) -> dict[str, Any]:
        return dict(self._telemetry)

    def reset_telemetry(self) -> None:
        self._telemetry.update(
            {
                "gui_calls": 0,
                "world_calls": 0,
                "world_prompt_tokens": 0,
                "world_completion_tokens": 0,
                "world_output_limit_hits": 0,
                "best_intent_normalizations": 0,
            }
        )

    @staticmethod
    def _normalize_redundant_best_intent(
        raw: str,
        request_payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:

        actions = request_payload.get("candidate_actions")
        intents = request_payload.get("candidate_actions_intent")
        if not isinstance(actions, list) or not isinstance(intents, list):
            return None
        if len(actions) != len(intents) or not actions:
            return None
        try:
            response_payload = p21_core._parse_json_object(raw)
        except Exception:
            return None
        best = response_payload.get("best_action")
        if not isinstance(best, dict):
            return None
        candidate_index = best.get("candidate_index")
        if type(candidate_index) is not int:
            return None
        if not 0 <= candidate_index < len(actions):
            return None
        if best.get("unified_action") != actions[candidate_index]:
            return None
        expected_intent = intents[candidate_index]
        observed_intent = best.get("action_intent")
        if observed_intent == expected_intent or not isinstance(expected_intent, str):
            return None
        normalized = dict(response_payload)
        normalized_best = dict(best)
        normalized_best["action_intent"] = expected_intent
        normalized["best_action"] = normalized_best
        warning = {
            "kind": "best_action_intent_paraphrase",
            "candidate_index_zero_based": candidate_index,
            "decision_identity_verified_by": [
                "candidate_index",
                "exact_unified_action",
            ],
        }
        return json.dumps(normalized, ensure_ascii=False), warning

    async def close(self) -> None:
        for client in self.world_clients:
            await client.close()
        for client in self.agent_clients:
            await client.close()
        await p21_core.VanillaAgent.close(self)

    async def _chat_json_uninstrumented(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        screenshot: bytes,
        response_format: dict[str, Any],
        parser: Any,
        correction: str,
    ) -> tuple[Any, str, dict[str, Any]]:
        payload = dict(payload)
        if self.reference_images:
            payload["task_reference_images"] = [
                str(item.get("label") or f"task_reference_{index}")
                for index, item in enumerate(self.reference_images)
            ]
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            {
                "type": "text",
                "text": "Attached image current_screenshot:",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(screenshot).decode("ascii"),
                    "detail": "high",
                },
            },
        ]
        for index, item in enumerate(self.reference_images):
            label = str(item.get("label") or f"task_reference_{index}")
            content.extend(
                [
                    {"type": "text", "text": f"Attached task reference image {label}:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(item["data"]).decode("ascii"),
                            "detail": "high",
                        },
                    },
                ]
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        last_error: Exception | None = None
        request_max_tokens = self.config.max_completion_tokens
        for attempt in range(self.config.model_retries + 1):
            try:
                extra_body: dict[str, Any] = {
                    "chat_template_kwargs": {
                        "enable_thinking": self.config.enable_thinking
                    },
                }
                if self.config.enable_thinking:
                    extra_body["reasoning_effort"] = self.config.reasoning_effort
                    extra_body["thinking_token_budget"] = min(
                        self.config.thinking_token_budget,
                        max(1, request_max_tokens - 1),
                    )
                client = self.agent_clients[self._agent_client_cursor % len(self.agent_clients)]
                self._agent_client_cursor += 1
                response = await client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=request_max_tokens,
                    response_format=response_format,
                    extra_body=extra_body,
                )
                self._telemetry["gui_calls"] += 1
                choice = response.choices[0]
                raw = choice.message.content or ""
                try:
                    parsed = parser(raw)
                    usage = response.usage
                    return parsed, raw, {
                        "finish_reason": choice.finish_reason,
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                        "output_limit_hit": choice.finish_reason == "length",
                    }
                except Exception as exc:
                    normalized = self._normalize_redundant_best_intent(raw, payload)
                    if normalized is not None:
                        normalized_raw, warning = normalized
                        try:
                            parsed = parser(normalized_raw)
                        except Exception:
                            pass
                        else:
                            self._telemetry["best_intent_normalizations"] += 1
                            usage = response.usage
                            return parsed, raw, {
                                "finish_reason": choice.finish_reason,
                                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(
                                    usage, "completion_tokens", None
                                ),
                                "total_tokens": getattr(usage, "total_tokens", None),
                                "output_limit_hit": choice.finish_reason == "length",
                                "content_validation_warning": warning,
                            }
                    last_error = ValueError(f"{exc}; raw_response={raw[:2000]!r}")
                    if attempt < self.config.model_retries:
                        messages.extend(
                            [
                                {"role": "assistant", "content": raw},
                                {
                                    "role": "user",
                                    "content": f"{correction} Validation error: {exc}",
                                },
                            ]
                        )
                        await asyncio.sleep(2**attempt)
            except Exception as exc:
                last_error = exc
                adjusted = p21_core.adjust_max_completion_tokens_from_context_error(
                    exc, request_max_tokens
                )
                if adjusted < request_max_tokens:
                    request_max_tokens = adjusted
                if attempt < self.config.model_retries:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"GUI Agent failed after {self.config.model_retries + 1} attempts: {last_error}"
        )

    async def _chat_json(self, **kwargs):
        system_prompt = kwargs.get("system_prompt")
        if system_prompt == self.candidate_prompt:
            stage = "candidate_gui_agent"
        elif system_prompt == self.digest_prompt:
            stage = "digest_request"
        elif system_prompt == self.best_prompt:
            stage = "best_gui_agent"
        else:
            stage = "unclassified_gui_agent"
        started = time.monotonic()
        try:
            return await self._chat_json_uninstrumented(**kwargs)
        finally:
            self._stage_latency_requests.setdefault(stage, []).append(
                time.monotonic() - started
            )

    async def _digest_candidates(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return await super()._digest_candidates(*args, **kwargs)
        finally:
            self._stage_latency_requests.setdefault("digest_parallel_wall", []).append(
                time.monotonic() - started
            )

    async def _predict_with_local_wm_replica_uninstrumented(
        self,
        current_state: GuiState,
        candidates: Sequence[CandidateAction],
    ) -> tuple[list[LocalPrediction], dict[str, Any], dict[str, Any], int, str]:
        replica_index = await self._available_world_clients.get()
        client = self.world_clients[replica_index]
        before = self.telemetry_snapshot()

        async def predict(candidate: CandidateAction) -> LocalPrediction:
            sample = {
                "candidate_action_json": json.dumps(
                    candidate.unified_action.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            }
            user_text = build_local_world_model_user_text(
                sample, prompt_key="localwm_next"
            )
            image_url = "data:image/png;base64," + base64.b64encode(
                current_state.screenshot or b""
            ).decode("ascii")
            response = await client.chat.completions.create(
                model=self.world_model,
                messages=[
                    {"role": "system", "content": self.world_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": "high"},
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                temperature=0,
                max_tokens=self.world_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choice = response.choices[0]
            raw = (choice.message.content or "").strip()
            usage = response.usage
            self._telemetry["world_calls"] += 1
            self._telemetry["world_prompt_tokens"] += int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            self._telemetry["world_completion_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            if choice.finish_reason == "length":
                self._telemetry["world_output_limit_hits"] += 1
            analysis, predicted_markdown = decode_local_prediction(
                raw,
                current_md=current_state.markdown,
                prompt_key="localwm_next",
                delta_state_mode="replayed_next",
            )
            return LocalPrediction(
                candidate_action=candidate,
                action_analysis=analysis,
                predicted_markdown=predicted_markdown,
                raw_output=raw,
                output_state_mode="replayed_next",
            )

        try:
            predictions = list(await asyncio.gather(*(predict(item) for item in candidates)))
            return (
                predictions,
                before,
                self.telemetry_snapshot(),
                replica_index,
                self.world_endpoints[replica_index],
            )
        finally:
            self._available_world_clients.put_nowait(replica_index)

    async def _predict_with_local_wm_replica(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return await self._predict_with_local_wm_replica_uninstrumented(
                *args, **kwargs
            )
        finally:
            self._stage_latency_requests.setdefault("mkd2world_parallel_wall", []).append(
                time.monotonic() - started
            )

    async def choose(self, *args, **kwargs):
        self._stage_latency_requests = {}
        selected, diagnostics_raw, latency = await super().choose(*args, **kwargs)
        diagnostics = json.loads(diagnostics_raw)
        request_latencies = {
            key: [round(value, 6) for value in values]
            for key, values in self._stage_latency_requests.items()
        }
        wall_components = {
            "candidate_gui_agent": sum(
                self._stage_latency_requests.get("candidate_gui_agent", [])
            ),
            "mkd2world": sum(
                self._stage_latency_requests.get("mkd2world_parallel_wall", [])
            ),
            "digest": sum(
                self._stage_latency_requests.get("digest_parallel_wall", [])
            ),
            "best_gui_agent": sum(
                self._stage_latency_requests.get("best_gui_agent", [])
            ),
        }
        accounted = sum(wall_components.values())
        diagnostics["stage_latency_seconds"] = {
            "wall": {
                key: round(value, 6) for key, value in wall_components.items()
            },
            "digest_request_latencies": request_latencies.get(
                "digest_request", []
            ),
            "pipeline_total": round(latency, 6),
            "unattributed_overhead": round(max(0.0, latency - accounted), 6),
        }
        return selected, json.dumps(diagnostics, ensure_ascii=False), latency


class P21DirectAgentAdapter:
    def __init__(self, agent: RemoteP2DigestAgent, audit_path: Path) -> None:
        self.agent = agent
        self.audit_path = audit_path
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self._run_event_loop,
            name="vwa-p21-async-loop",
            daemon=True,
        )
        self.loop_thread.start()
        self.task_config = ""
        self.step_index = 0

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def set_task(self, task: str | Path, reference_images: Sequence[dict[str, Any]]) -> None:
        self.task_config = str(Path(task).resolve())
        self.step_index = 0
        self.agent.set_reference_images(reference_images)

    def telemetry_snapshot(self) -> dict[str, Any]:
        return self.agent.telemetry_snapshot()

    def reset_telemetry(self) -> None:
        self.agent.reset_telemetry()

    def direct_action(self, context, **_kwargs) -> CandidateAction:
        history = [
            json.dumps(item.get("action", item), ensure_ascii=False, sort_keys=True)
            for item in context.current_subgoal_trace
        ]
        future = asyncio.run_coroutine_threadsafe(
            self.agent.choose(
                context.goal_text,
                map_vwa_url_to_real(context.current_state.url)
                if context.current_state.url
                else "",
                history,
                context.current_state.screenshot or b"",
                "",
            ),
            self.loop,
        )
        selected_payload, diagnostics_raw, latency = future.result()
        selected = selected_payload["selected_action"]
        diagnostics = json.loads(diagnostics_raw)
        row = {
            "task_config": self.task_config,
            "step_index": self.step_index,
            "latency_seconds": latency,
            "selected_action": selected,
            "diagnostics": diagnostics,
            "reference_image_count": len(self.agent.reference_images),
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.step_index += 1
        return CandidateAction(
            unified_action=UnifiedAction.from_dict(selected["unified_action"]),
            action_intent=str(selected.get("action_intent") or ""),
            evidence=diagnostics_raw,
            confidence=int(selected.get("confidence", 0)),
        )

    def close(self) -> None:
        if not self.loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(self.agent.close(), self.loop)
            future.result()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=30)
            self.loop.close()


class P21VwaEnvironment(bare_vwa.ScreenshotOnlyVwaEnvironment):
    def __init__(self, *, agent_adapter: P21DirectAgentAdapter, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent_adapter = agent_adapter

    def reset(self, task=None) -> GuiState:
        state = super().reset(task)
        self.agent_adapter.set_task(task, bare_vwa.ACTIVE_TASK_IMAGES)
        return state

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.agent_adapter.close()


def build_p21_pipeline(config) -> PipelineRunner:
    try:
        from eval.vwa_eval_defaults import apply_to_experiment_config
    except ImportError:
        from vwa_eval_defaults import apply_to_experiment_config
    config = apply_to_experiment_config(config)
    if os.environ.get("VWA_GUI_OWL") == "1":
        agent = config.pipeline.agent
        options = dict(agent.options or {})
        options["enable_thinking"] = False
        agent = replace(agent, options=options)
        config = replace(config, pipeline=replace(config.pipeline, agent=agent))
    agent_options = dict(config.pipeline.agent.options)
    world_options = dict(config.pipeline.local_world_model.options)
    prompt_paths = {
        name: _resolve_project_path(str(agent_options[f"{name}_prompt"]))
        for name in ("candidate", "digest", "best")
    }
    world_prompt_path = _resolve_project_path(str(world_options["prompt"]))
    agent_endpoint = os.environ.get(
        "VWA_P21_AGENT_ENDPOINT", config.pipeline.agent.base_url.removesuffix("/v1")
    )
    endpoint_text = os.environ.get(
        "VWA_P21_WORLD_ENDPOINTS", config.pipeline.local_world_model.base_url.removesuffix("/v1")
    )
    world_endpoints = tuple(item.strip() for item in endpoint_text.split(",") if item.strip())
    runner_config = p21_core.RunnerConfig(
        endpoint=agent_endpoint,
        model=config.pipeline.agent.model,
        prompt=prompt_paths["candidate"].read_text(encoding="utf-8"),
        mode="screenshot",
        markdown_endpoint=None,
        width=int(config.pipeline.environment.options.get("width", 1280)),
        height=int(config.pipeline.environment.options.get("height", 2048)),
        max_steps=50,
        navigation_timeout_ms=30000,
        action_wait_ms=1000,
        model_timeout_seconds=int(config.pipeline.agent.timeout_seconds),
        model_retries=int(config.pipeline.agent.retry_attempts),
        reasoning_effort=str(agent_options.get("reasoning_effort") or "medium"),
        thinking_token_budget=int(agent_options.get("thinking_token_budget", 4096)),
        max_completion_tokens=int(config.pipeline.agent.max_tokens),
        temperature=float(config.pipeline.agent.temperature),
        enable_thinking=bool(agent_options.get("enable_thinking", True)),
        agent_context_tokens=int(agent_options.get("max_context_tokens", 32768)),
        no_progress_limit=4,
        compact_vanilla_input=True,
        gui_agent_use_current_md=False,
    )
    remote_agent = RemoteP2DigestAgent(
        runner_config,
        candidate_prompt=prompt_paths["candidate"].read_text(encoding="utf-8"),
        digest_prompt=prompt_paths["digest"].read_text(encoding="utf-8"),
        best_prompt=prompt_paths["best"].read_text(encoding="utf-8"),
        world_prompt=world_prompt_path.read_text(encoding="utf-8"),
        world_endpoints=world_endpoints,
        world_model=config.pipeline.local_world_model.model,
        world_max_tokens=int(config.pipeline.local_world_model.max_tokens),
        world_timeout_seconds=int(config.pipeline.local_world_model.timeout_seconds),
        policy_version=str(agent_options.get("policy_version") or "p2.1"),
    )
    audit_path = Path(
        os.environ.get("VWA_P21_AUDIT_PATH", "output/visualwebarena/mkd2world_p21_policy_audit.jsonl")
    )
    adapter = P21DirectAgentAdapter(remote_agent, audit_path)
    env_options = config.pipeline.environment.options
    environment = P21VwaEnvironment(
        agent_adapter=adapter,
        vwa_root=env_options.get("vwa_root", "envs/visualwebarena"),
        headless=bool(env_options.get("headless", True)),
        viewport_width=int(env_options.get("width", 1280)),
        viewport_height=int(env_options.get("height", 2048)),
        observation_type="image",
        sleep_after_execution=float(env_options.get("sleep_after_execution", 1.0)),
        site_urls=dict(env_options.get("site_urls") or {}),
    )
    return PipelineRunner(
        agent=adapter,
        global_world_model=NoOpGlobalWorldModel(),
        local_world_model=NoOpLocalWorldModel(),
        environment=environment,
        memory=NoOpMemoryStore(),
        mode="vanilla",
    )


online_sr.build_pipeline_runner = build_p21_pipeline


if __name__ == "__main__":
    online_sr.main()
