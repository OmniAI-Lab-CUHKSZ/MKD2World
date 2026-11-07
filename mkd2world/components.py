"""Component factories for configurable experiment runners."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mkd2world.agent import LlmGuiAgent
from mkd2world.config import ComponentConfig, PipelineConfig, RunConfig
from mkd2world.contracts import CandidateAction, CandidateSubgoal, GuiState
from mkd2world.envs import (
    ONLINE_MIND2WEB_SUPPORTED_ACTIONS,
    OSWORLD_SUPPORTED_ACTIONS,
    VWA_SUPPORTED_ACTIONS,
    OSWorldActionAdapter,
    OnlineMind2WebActionAdapter,
    StaticConversionEnvironment,
    VisualWebArenaActionAdapter,
)
from mkd2world.llm import LLMClient, LLMConfig
from mkd2world.local_wm_runtime import TransformersLocalWorldModel
from mkd2world.vwa_runtime import VisualWebArenaEnvironment
from mkd2world.memory import MemoryStore, NoOpMemoryStore
from mkd2world.pipeline import DryRunEnvironment, PipelineRunner
from mkd2world.world_model import LlmGlobalWorldModel, LlmLocalWorldModel, NoOpGlobalWorldModel, NoOpLocalWorldModel
from mkd2world.actions import UnifiedAction


LLM_KINDS = {"llm", "openai", "openai_compatible", "anthropic"}
NOOP_KINDS = {"", "noop", "disabled", "none"}


def _component_effectively_enabled(component: ComponentConfig, top_level_enabled: bool = True) -> bool:
    return bool(top_level_enabled and component.enabled and component.kind not in NOOP_KINDS)


def llm_config_from_component(component: ComponentConfig, *, default_provider: str = "openai_compatible") -> LLMConfig:
    provider = component.provider or (component.kind if component.kind in LLM_KINDS - {"llm"} else default_provider)
    return LLMConfig(
        provider=provider,
        model=component.model,
        api_key_env=component.api_key_env or None,
        base_url=component.base_url or None,
        temperature=component.temperature if component.temperature is not None else 0.7,
        max_tokens=component.max_tokens if component.max_tokens is not None else 4096,
        timeout_seconds=component.timeout_seconds,
        retry_attempts=component.retry_attempts,
        enable_thinking=bool(component.options.get("enable_thinking", False)),
        reasoning_effort=(
            str(component.options["reasoning_effort"])
            if component.options.get("reasoning_effort") is not None
            else None
        ),
        thinking_token_budget=(
            int(component.options["thinking_token_budget"])
            if component.options.get("thinking_token_budget") is not None
            else None
        ),
    )


def build_llm_client(component: ComponentConfig) -> LLMClient:
    if component.kind in NOOP_KINDS or not component.enabled:
        raise ValueError("Cannot build an LLM client from a disabled/noop component")
    if component.kind not in LLM_KINDS:
        raise ValueError(f"Unsupported LLM component kind: {component.kind}")
    if not component.model:
        raise ValueError("LLM component requires model")
    return LLMClient(llm_config_from_component(component))


def build_memory_store(component: ComponentConfig | None) -> MemoryStore:
    return NoOpMemoryStore()


class ScriptedGuiAgent:
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        options = dict(options or {})
        self.subgoal = CandidateSubgoal(
            subgoal_text=str(options.get("subgoal_text") or "Continue the task"),
            global_plan=str(options.get("global_plan") or "Execute the scripted action."),
            completion_condition=str(options.get("completion_condition") or "The scripted step has run."),
            evidence=str(options.get("evidence") or "Scripted dry-run agent."),
            confidence=int(options.get("confidence", 10)),
        )
        action_payload = options.get("action") if isinstance(options.get("action"), dict) else {"type": "wait"}
        self.action = CandidateAction(
            unified_action=UnifiedAction.from_dict(action_payload),
            action_intent=str(options.get("action_intent") or "execute scripted action"),
            evidence=str(options.get("action_evidence") or "Scripted dry-run action."),
            confidence=int(options.get("action_confidence", 10)),
        )

    def candidate_subgoals(self, context) -> list[CandidateSubgoal]:
        return [self.subgoal]

    def best_subgoal(self, context, candidates, advice) -> CandidateSubgoal:
        return candidates[0]

    def candidate_actions(self, context, current_subgoal_text: str | None = None) -> list[CandidateAction]:
        return [self.action]

    def best_action(self, context, candidates, predictions) -> CandidateAction:
        return candidates[0]

    def direct_action(self, context, *, prompt_key: str = "gui_agent_vanilla_md", current_subgoal_text: str | None = None) -> CandidateAction:
        return self.action


class AdapterLocalWorldModel(NoOpLocalWorldModel):
    def __init__(self, adapter_path: str) -> None:
        self.adapter_path = adapter_path

    def predict_next_state(self, current_state, candidate_action, *, memory=(), guidance_mode=None):
        prediction = super().predict_next_state(
            current_state,
            candidate_action,
            memory=memory,
            guidance_mode=guidance_mode,
        )
        return prediction.__class__(
            candidate_action=prediction.candidate_action,
            action_analysis=f"Local WM adapter is configured at {self.adapter_path}; runtime inference backend is not loaded in this process.",
            predicted_markdown=prediction.predicted_markdown,
            raw_output=prediction.raw_output,
        )


def build_gui_agent(component: ComponentConfig):
    if component.kind in NOOP_KINDS or not component.enabled:
        raise ValueError("GUI agent cannot be disabled for an executable pipeline")
    if component.kind == "scripted":
        return ScriptedGuiAgent(component.options)
    return LlmGuiAgent(build_llm_client(component))


def build_global_world_model(component: ComponentConfig | None):
    if component is None or component.kind in NOOP_KINDS or not component.enabled:
        return NoOpGlobalWorldModel()
    return LlmGlobalWorldModel(build_llm_client(component))


def build_local_world_model(component: ComponentConfig | None):
    if component is None or component.kind in NOOP_KINDS or not component.enabled:
        return NoOpLocalWorldModel()
    if component.kind in {"adapter", "transformers_adapter"}:
        if not component.adapter_path:
            raise ValueError("Local WM adapter component requires adapter_path")
        return TransformersLocalWorldModel(
            base_model=component.model or str(component.options.get("base_model") or "models/Qwen3.5-9B"),
            adapter_path=component.adapter_path,
            device_map=component.options.get("device_map", "auto"),
            torch_dtype=str(component.options.get("torch_dtype") or "bfloat16"),
            max_length=int(component.options.get("max_length", 60000)),
            max_new_tokens=int(component.options.get("max_new_tokens", 60000)),
            image_min_pixels=int(component.options.get("image_min_pixels", 256 * 28 * 28)),
            image_max_pixels=int(component.options.get("image_max_pixels", 1024 * 28 * 28)),
            batch_size=int(component.options.get("batch_size", 1)),
            prompt_key=str(component.options.get("prompt_key") or "world_local"),
        )
    if component.kind == "adapter_stub":
        if not component.adapter_path:
            raise ValueError("Local WM adapter component requires adapter_path")
        return AdapterLocalWorldModel(component.adapter_path)
    return LlmLocalWorldModel(build_llm_client(component))


def build_environment(component: ComponentConfig | None):
    if component is None or component.kind in NOOP_KINDS or not component.enabled or component.kind == "dry_run":
        options = component.options if component is not None else {}
        state = GuiState(
            screenshot=None,
            markdown=str(options.get("markdown") or "[Environment: Web]"),
            url=str(options.get("url") or "") or None,
            environment=str(options.get("environment") or "Web"),
        )
        return DryRunEnvironment(state)
    options = component.options
    if component.kind in {"visualwebarena_runtime", "vwa_runtime"}:
        return VisualWebArenaEnvironment(
            vwa_root=options.get("vwa_root", "envs/visualwebarena"),
            headless=bool(options.get("headless", True)),
            viewport_width=int(options.get("width", 1280)),
            viewport_height=int(options.get("height", 720)),
            sleep_after_execution=float(options.get("sleep_after_execution", 1.0)),
            render_config=dict(options.get("render_config") or {}),
            site_urls=dict(options.get("site_urls") or {}),
            eval_captioning_model=str(
                options.get("eval_captioning_model") or "Salesforce/blip2-flan-t5-xl"
            ),
            eval_captioning_model_device=str(
                options.get("eval_captioning_model_device") or "cpu"
            ),
        )
    width = int(options.get("width", 1000))
    height = int(options.get("height", 1000))
    state = GuiState(
        screenshot=None,
        markdown=str(options.get("markdown") or "[Environment: Web]"),
        url=str(options.get("url") or "") or None,
        environment=str(options.get("environment") or "Web"),
    )
    after_markdown = str(options["after_markdown"]) if "after_markdown" in options else None
    if component.kind in {"visualwebarena", "vwa"}:
        return StaticConversionEnvironment(
            state=state,
            supported_actions=VWA_SUPPORTED_ACTIONS,
            converter=VisualWebArenaActionAdapter(width=width, height=height),
            after_markdown=after_markdown,
        )
    if component.kind in {"online_mind2web", "online-mind2web"}:
        return StaticConversionEnvironment(
            state=state,
            supported_actions=ONLINE_MIND2WEB_SUPPORTED_ACTIONS,
            converter=OnlineMind2WebActionAdapter(width=width, height=height),
            after_markdown=after_markdown,
        )
    if component.kind == "osworld":
        return StaticConversionEnvironment(
            state=state,
            supported_actions=OSWORLD_SUPPORTED_ACTIONS,
            converter=OSWorldActionAdapter(width=width, height=height),
            after_markdown=after_markdown,
        )
    raise ValueError(f"Unsupported executable environment component kind: {component.kind}")


def build_pipeline_runner(config: RunConfig | PipelineConfig) -> PipelineRunner:
    pipeline = config.pipeline if isinstance(config, RunConfig) else config
    memory = build_memory_store(pipeline.memory if pipeline.enable_memory else None)
    global_wm = build_global_world_model(pipeline.global_world_model if pipeline.enable_global_wm else None)
    local_wm = build_local_world_model(pipeline.local_world_model if pipeline.enable_local_wm else None)
    return PipelineRunner(
        agent=build_gui_agent(pipeline.agent),
        global_world_model=global_wm,
        local_world_model=local_wm,
        environment=build_environment(pipeline.environment),
        memory=memory,
        mode=pipeline.mode,
    )


def pipeline_component_report(config: RunConfig | PipelineConfig) -> dict[str, Any]:
    pipeline = config.pipeline if isinstance(config, RunConfig) else config
    return {
        "agent": asdict(pipeline.agent),
        "global_world_model": asdict(pipeline.global_world_model),
        "local_world_model": asdict(pipeline.local_world_model),
        "memory": asdict(pipeline.memory),
        "environment": asdict(pipeline.environment),
        "mode": pipeline.mode,
        "enabled": {
            "global_world_model": _component_effectively_enabled(pipeline.global_world_model, pipeline.enable_global_wm),
            "local_world_model": _component_effectively_enabled(pipeline.local_world_model, pipeline.enable_local_wm),
            "memory": _component_effectively_enabled(pipeline.memory, pipeline.enable_memory),
        },
    }
