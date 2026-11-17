from __future__ import annotations

import json
from typing import Any

from mkd2world.actions import UnifiedAction
from mkd2world.prompts import PromptRegistry
from mkd2world.training.markdown import build_assistant_output


LOCALWM_ROUTED_PROMPT_KEY = "localwm_routed"
LOCALWM_TASK_PROMPT_KEYS = {
    "describe": "localwm_cur",
    "forward_full": "localwm_next",
    "forward_delta": "localwm_delta",
}


def resolve_local_world_model_prompt_key(sample: dict[str, Any], prompt_key: str) -> str:
    """Resolve the concrete system prompt for one Local WM sample."""
    if prompt_key != LOCALWM_ROUTED_PROMPT_KEY:
        return prompt_key
    task_type = str(sample.get("task_type") or "").strip()
    try:
        return LOCALWM_TASK_PROMPT_KEYS[task_type]
    except KeyError as exc:
        raise ValueError(
            f"{LOCALWM_ROUTED_PROMPT_KEY} requires one of "
            f"{sorted(LOCALWM_TASK_PROMPT_KEYS)}; got task_type={task_type!r}"
        ) from exc


def concrete_local_world_model_prompt_keys(prompt_key: str) -> tuple[str, ...]:
    if prompt_key == LOCALWM_ROUTED_PROMPT_KEY:
        return tuple(dict.fromkeys(LOCALWM_TASK_PROMPT_KEYS.values()))
    return (prompt_key,)


def _canonical_action_text(value: object, *, field_name: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    value = dict(value)
    value.pop("raw", None)
    return json.dumps(UnifiedAction.from_dict(value).to_dict(), ensure_ascii=False, sort_keys=True)


def candidate_action_text(sample: dict[str, Any]) -> str:
    if sample.get("candidate_action_json"):
        return _canonical_action_text(sample["candidate_action_json"], field_name="candidate_action_json")
    if "candidate_action" in sample:
        return _canonical_action_text(sample["candidate_action"], field_name="candidate_action")
    if "action" in sample:
        return _canonical_action_text(sample["action"], field_name="action")
    return "{}"


def local_world_model_target(sample: dict[str, Any], *, prompt_key: str) -> str:
    """Resolve the supervised target without overwriting the raw next-state label."""
    concrete_prompt_key = resolve_local_world_model_prompt_key(sample, prompt_key)
    if concrete_prompt_key == "localwm_delta":
        delta = str(sample.get("target_delta_md_clean") or "").strip()
        if delta:
            return delta
    if concrete_prompt_key == "localwm_next":
        clean_next = str(sample.get("target_next_md_clean") or "").strip()
        if clean_next:
            return clean_next
    return str(sample.get("target_next_md") or "").strip()


def build_local_world_model_user_text(
    sample: dict[str, Any],
    *,
    prompt_key: str = "world_local",
) -> str:
    prompt_key = resolve_local_world_model_prompt_key(sample, prompt_key)
    lines = [
        "## Current Screenshot",
        "The attached image is the current screenshot before the action.",
        "",
    ]
    if prompt_key == "localwm_cur":
        return "\n".join(lines)
    if prompt_key in {"localwm_next", "localwm_delta"}:
        lines.extend(
            [
                "## Candidate Action",
                candidate_action_text(sample),
                "",
                (
                    "Predict the complete visible interface state after this candidate action."
                    if prompt_key == "localwm_next"
                    else "Predict only the visible interface changes caused by this candidate action."
                ),
            ]
        )
        return "\n".join(lines)
    if prompt_key != "localwm_next":
        lines.extend(
            [
                "## Current Interface Markdown",
                str(sample["current_md"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Candidate Action",
            candidate_action_text(sample),
            "",
            "Predict the complete visible interface state after this candidate action.",
        ]
    )
    if sample.get("guidance_mode"):
        lines.extend(
            [
                "",
                "## Memory Guidance Mode",
                str(sample["guidance_mode"]),
                "",
                "## Retrieved Memory",
                json.dumps(sample.get("retrieved_memory"), ensure_ascii=False, sort_keys=True),
            ]
        )
    return "\n".join(lines)


def build_local_world_model_messages(
    sample: dict[str, Any],
    *,
    include_output: bool = False,
    include_image: bool = True,
    prompt_key: str = "world_local",
) -> list[dict]:
    concrete_prompt_key = resolve_local_world_model_prompt_key(sample, prompt_key)
    prompt = PromptRegistry.from_project().load(concrete_prompt_key)
    user_content = []
    if include_image and sample.get("current_image_path"):
        user_content.append({"type": "image", "image": str(sample["current_image_path"])})
    user_content.append(
        {
            "type": "text",
            "text": build_local_world_model_user_text(sample, prompt_key=concrete_prompt_key),
        }
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": user_content},
    ]
    if include_output:
        supervised_target = local_world_model_target(sample, prompt_key=concrete_prompt_key)
        if not supervised_target:
            raise ValueError("include_output=True requires a non-empty supervised target")
        target = build_assistant_output(sample.get("action_analysis", ""), supervised_target)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def apply_chat_template(processor, messages: list[dict], **kwargs):
    if hasattr(processor, "apply_chat_template"):
        kwargs.setdefault("enable_thinking", False)
        try:
            return processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return processor.apply_chat_template(messages, **kwargs)
    return messages
