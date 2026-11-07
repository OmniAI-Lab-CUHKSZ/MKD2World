"""GUI Agent interfaces and LLM-backed skeleton implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from mkd2world.actions import ActionType, UnifiedAction
from mkd2world.contracts import CandidateAction, CandidateSubgoal, GlobalAdvice, GuiState, LocalPrediction
from mkd2world.messages import json_messages, parse_json_object
from mkd2world.prompts import PromptRegistry


class ChatClient(Protocol):
    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> str:
        ...


@dataclass
class AgentContext:
    goal_text: str
    current_state: GuiState
    subgoal_history: Sequence[dict] = field(default_factory=tuple)
    current_subgoal_trace: Sequence[dict] = field(default_factory=tuple)
    supported_unified_actions: Sequence[str] = field(default_factory=lambda: tuple(item.value for item in ActionType))


class GuiAgent(Protocol):
    def candidate_subgoals(self, context: AgentContext) -> list[CandidateSubgoal]:
        ...

    def best_subgoal(self, context: AgentContext, candidates: Sequence[CandidateSubgoal], advice: Sequence[GlobalAdvice]) -> CandidateSubgoal:
        ...

    def candidate_actions(self, context: AgentContext, current_subgoal_text: str | None = None) -> list[CandidateAction]:
        ...

    def best_action(self, context: AgentContext, candidates: Sequence[CandidateAction], predictions: Sequence[LocalPrediction]) -> CandidateAction:
        ...

    def direct_action(self, context: AgentContext, *, prompt_key: str = "gui_agent_vanilla_md", current_subgoal_text: str | None = None) -> CandidateAction:
        ...


def _confidence(value: object) -> int:
    try:
        return max(0, min(10, int(value)))
    except Exception:
        return 0


def _parse_action(payload: dict) -> CandidateAction:
    return CandidateAction(
        unified_action=UnifiedAction.from_dict(payload["unified_action"]),
        action_intent=str(payload.get("action_intent", "")),
        evidence=str(payload.get("evidence", "")),
        confidence=_confidence(payload.get("confidence")),
    )


def _json_object(raw: str, *, stage: str) -> dict:
    return parse_json_object(raw, stage=stage)


def _candidate_index(payload: dict, candidates: Sequence[object], *, stage: str) -> int:
    if not candidates:
        raise ValueError(f"{stage} cannot select from an empty candidate list")
    try:
        index = int(payload["candidate_index"])
    except Exception as exc:
        raise ValueError(f"{stage} candidate_index must be an integer") from exc
    if index < 0 or index >= len(candidates):
        raise ValueError(f"{stage} candidate_index {index} out of range for {len(candidates)} candidates")
    return index


def _selected_action_index(payload: dict, candidates: Sequence[CandidateAction]) -> int:
    """Resolve a best-action response without trusting an ambiguous index."""

    reported = _candidate_index(payload, candidates, stage="best_action")
    try:
        returned = UnifiedAction.from_dict(payload["unified_action"]).to_dict()
    except Exception as exc:
        raise ValueError("best_action must repeat one candidate unified_action exactly") from exc
    matching = [
        index for index, candidate in enumerate(candidates)
        if candidate.unified_action.to_dict() == returned
    ]
    if not matching:
        raise ValueError("best_action unified_action does not match any candidate action")
    return reported if reported in matching else matching[0]


def _selected_subgoal_index(payload: dict, candidates: Sequence[CandidateSubgoal]) -> int:
    """Use the repeated immutable subgoal fields as the authoritative choice."""

    reported = _candidate_index(payload, candidates, stage="best_subgoal")
    subgoal_text = str(payload.get("subgoal_text", ""))
    completion_condition = str(payload.get("completion_condition", ""))
    matching = [
        index for index, candidate in enumerate(candidates)
        if candidate.subgoal_text == subgoal_text
        and candidate.completion_condition == completion_condition
    ]
    if not matching:
        raise ValueError(
            "best_subgoal must repeat one candidate subgoal_text and completion_condition exactly"
        )
    return reported if reported in matching else matching[0]


@dataclass
class LlmGuiAgent:
    client: ChatClient
    prompts: PromptRegistry = field(default_factory=PromptRegistry.from_project)

    def candidate_subgoals(self, context: AgentContext) -> list[CandidateSubgoal]:
        payload = self._base_payload(context)
        raw = self.client.chat(json_messages(self.prompts.load("gui_agent_subgoal_candidates"), payload))
        data = _json_object(raw, stage="candidate_subgoals")
        return [
            CandidateSubgoal(
                subgoal_text=str(item.get("subgoal_text", "")),
                global_plan=str(item.get("global_plans", "")),
                completion_condition=str(item.get("completion_condition", "")),
                evidence=str(item.get("evidence", "")),
                confidence=_confidence(item.get("confidence")),
            )
            for item in data.get("candidate_subgoals", [])
        ]

    def best_subgoal(self, context: AgentContext, candidates: Sequence[CandidateSubgoal], advice: Sequence[GlobalAdvice]) -> CandidateSubgoal:
        payload = self._base_payload(context) | {
            "candidate_subgoals": [item.__dict__ for item in candidates],
            "path_per_subgoal": [item.__dict__ for item in advice],
        }
        raw = self.client.chat(json_messages(self.prompts.load("gui_agent_subgoal_best"), payload))
        item = _json_object(raw, stage="best_subgoal")["best_subgoal"]
        index = _selected_subgoal_index(item, candidates)
        base = candidates[index]
        return CandidateSubgoal(
            subgoal_text=base.subgoal_text,
            global_plan=base.global_plan,
            completion_condition=base.completion_condition,
            evidence=str(item.get("evidence", "")),
            confidence=_confidence(item.get("confidence")),
        )

    def candidate_actions(self, context: AgentContext, current_subgoal_text: str | None = None) -> list[CandidateAction]:
        payload = self._base_payload(context) | {"current_subgoal_text": current_subgoal_text or ""}
        raw = self.client.chat(json_messages(self.prompts.load("gui_agent_action_candidates"), payload))
        data = _json_object(raw, stage="candidate_actions")
        return [_parse_action(item) for item in data.get("candidate_actions", [])]

    def best_action(self, context: AgentContext, candidates: Sequence[CandidateAction], predictions: Sequence[LocalPrediction]) -> CandidateAction:
        payload = self._base_payload(context) | {
            "candidate_actions": [item.unified_action.to_dict() | {"action_intent": item.action_intent} for item in candidates],
            "predicted_state": [item.predicted_markdown for item in predictions],
        }
        raw = self.client.chat(json_messages(self.prompts.load("gui_agent_action_best"), payload))
        item = _json_object(raw, stage="best_action")["best_action"]
        index = _selected_action_index(item, candidates)
        return candidates[index]

    def direct_action(self, context: AgentContext, *, prompt_key: str = "gui_agent_vanilla_md", current_subgoal_text: str | None = None) -> CandidateAction:
        payload = self._base_payload(context) | {"current_subgoal_text": current_subgoal_text or ""}
        if prompt_key == "gui_agent_vanilla":
            payload.pop("current_md", None)
        raw = self.client.chat(json_messages(self.prompts.load(prompt_key), payload))
        item = _json_object(raw, stage="direct_action")["selected_action"]
        return _parse_action(item)

    @staticmethod
    def _base_payload(context: AgentContext) -> dict:
        return {
            "goal_text": context.goal_text,
            "current_md": context.current_state.markdown,
            "current_screenshot": None,
            "_screenshot": context.current_state.screenshot,
            "subgoal_history": list(context.subgoal_history),
            "current_subgoal_trace": list(context.current_subgoal_trace),
            "supported_unified_actions": list(context.supported_unified_actions),
        }
