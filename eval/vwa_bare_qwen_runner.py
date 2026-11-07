from __future__ import annotations

import enum
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from PIL import Image


if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return str(self.value)

    enum.StrEnum = StrEnum

import mkd2world.experiments.online_sr as online_sr
from mkd2world.actions import ActionType, UnifiedAction
from mkd2world.agent import AgentContext, LlmGuiAgent
from mkd2world.components import build_pipeline_runner as build_default_pipeline
from mkd2world.contracts import CandidateAction, GuiState
from mkd2world.vwa_runtime import VisualWebArenaEnvironment, map_vwa_url_to_real


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "mkd2world/prompts/templatesV3/vanilla/gui_agent.md"
ACTIVE_TASK_IMAGES: list[dict] = []


class ExactV3PromptRegistry:
    def __init__(self, prompt_path: Path) -> None:
        self.prompt_path = prompt_path

    def load(self, key: str) -> str:
        if key != "gui_agent_vanilla":
            raise KeyError(f"Bare Qwen only permits gui_agent_vanilla, got {key}")
        return self.prompt_path.read_text(encoding="utf-8")


class ExactV3GuiAgent(LlmGuiAgent):
    PARSING_FAILURE_LIMIT = 3

    def _base_payload(self, context: AgentContext) -> dict:
        payload = LlmGuiAgent._base_payload(context)
        payload["current_url"] = (
            map_vwa_url_to_real(context.current_state.url)
            if context.current_state.url
            else None
        )
        if ACTIVE_TASK_IMAGES:
            payload["_images"] = list(ACTIVE_TASK_IMAGES)
        return payload

    def direct_action(self, context: AgentContext, **kwargs) -> CandidateAction:
        if not context.current_subgoal_trace:
            self._consecutive_parsing_failures = 0
        try:
            candidate = super().direct_action(context, **kwargs)
            action_type = candidate.unified_action.type.value
            if action_type not in context.supported_unified_actions:
                raise ValueError(f"unsupported action type: {action_type}")
            self._consecutive_parsing_failures = 0
            return candidate
        except (KeyError, TypeError, ValueError) as exc:
            self._consecutive_parsing_failures = (
                getattr(self, "_consecutive_parsing_failures", 0) + 1
            )
            reason = f"Invalid agent action: {exc}"
            if self._consecutive_parsing_failures < self.PARSING_FAILURE_LIMIT:
                return CandidateAction(
                    unified_action=UnifiedAction(ActionType.WAIT, {}),
                    action_intent=reason,
                    evidence=(
                        "The model response violated the frozen action contract; "
                        "VWA records a parsing-failure no-op."
                    ),
                    confidence=0,
                )
            return CandidateAction(
                unified_action=UnifiedAction(ActionType.FAIL, {"reason": reason}),
                action_intent=reason,
                evidence="VWA parsing-failure threshold reached.",
                confidence=0,
            )


class ScreenshotOnlyVwaEnvironment(VisualWebArenaEnvironment):
    def reset(self, task=None) -> GuiState:
        ACTIVE_TASK_IMAGES.clear()
        if isinstance(task, (str, Path)):
            task_path = Path(task).resolve()
            task_payload = json.loads(task_path.read_text(encoding="utf-8"))
            image_paths = task_payload.get("image")
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for index, image_path in enumerate(image_paths or []):
                candidate = Path(image_path)
                if not candidate.is_absolute():
                    project_candidate = PROJECT_ROOT / candidate
                    candidate = (
                        project_candidate
                        if project_candidate.is_file()
                        else PROJECT_ROOT / "envs/visualwebarena" / candidate
                    )
                with Image.open(candidate) as reference:
                    from io import BytesIO

                    buffer = BytesIO()
                    reference.convert("RGB").save(buffer, format="PNG")
                    ACTIVE_TASK_IMAGES.append(
                        {"label": f"task_reference_{index}", "data": buffer.getvalue()}
                    )
        return super().reset(task)

    def _state_from_observation(self, observation: dict, info: dict) -> GuiState:
        image_array = observation.get("image")
        if image_array is None:
            raise ValueError("VisualWebArena observation has no image")
        image = Image.fromarray(image_array).convert("RGB")
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        screenshot = buffer.getvalue()
        page = info.get("page") or getattr(self._env, "page", None)
        url = str(getattr(page, "url", "") or "") or None
        digest = hashlib.sha256(screenshot).hexdigest()[:16]
        return GuiState(
            screenshot=screenshot,
            markdown=f"[Environment: Web]\nurl={url or ''}\nscreenshot={digest}",
            url=url,
            environment="Web",
        )


def build_bare_pipeline(config):
    try:
        from eval.vwa_eval_defaults import apply_to_experiment_config
    except ImportError:
        from vwa_eval_defaults import apply_to_experiment_config
    config = apply_to_experiment_config(config)
    if os.environ.get("VWA_GUI_OWL") == "1":
        options = dict(config.pipeline.agent.options or {})
        options["enable_thinking"] = False
        agent = replace(config.pipeline.agent, options=options)
        config = replace(config, pipeline=replace(config.pipeline, agent=agent))
    agent_updates: dict[str, str] = {}
    endpoint = os.environ.get("VWA_BARE_AGENT_ENDPOINT")
    if endpoint:
        base = endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        agent_updates["base_url"] = base
    model = os.environ.get("VWA_BARE_AGENT_MODEL")
    if model:
        agent_updates["model"] = model
    if agent_updates:
        agent = replace(config.pipeline.agent, **agent_updates)
        pipeline_cfg = replace(config.pipeline, agent=agent)
        config = replace(config, pipeline=pipeline_cfg)
    pipeline = build_default_pipeline(config)
    pipeline.agent = ExactV3GuiAgent(
        client=pipeline.agent.client,
        prompts=ExactV3PromptRegistry(PROMPT_PATH),
    )
    options = config.pipeline.environment.options
    pipeline.environment = ScreenshotOnlyVwaEnvironment(
        vwa_root=options.get("vwa_root", "envs/visualwebarena"),
        headless=bool(options.get("headless", True)),
        viewport_width=int(options.get("width", 1280)),
        viewport_height=int(options.get("height", 720)),
        observation_type="image",
        sleep_after_execution=float(options.get("sleep_after_execution", 1.0)),
        site_urls=dict(options.get("site_urls") or {}),
    )
    pipeline.mode = "vanilla"
    return pipeline


online_sr.build_pipeline_runner = build_bare_pipeline


if __name__ == "__main__":
    online_sr.main()
