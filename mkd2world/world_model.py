"""Global and local world model interfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from mkd2world.agent import ChatClient
from mkd2world.contracts import CandidateAction, CandidateSubgoal, GlobalAdvice, GuiState, LocalPrediction
def memory_rows_for_prompt(memory: Sequence) -> list[dict]:
    rows: list[dict] = []
    for item in memory:
        if isinstance(item, dict):
            rows.append(dict(item))
            continue
        payload = getattr(item, "to_prompt_payload", None)
        rows.append(payload() if callable(payload) else {})
    return rows


def local_memory_reference(memory: Sequence) -> dict | None:
    if not memory:
        return None
    rows = memory_rows_for_prompt(memory[:1])
    return rows[0] if rows else None


from mkd2world.messages import json_messages, parse_json_object
from mkd2world.prompts import PromptRegistry


def global_guidance_mode(memory: Sequence[dict]) -> list[str]:
    modes = []
    for item in memory:
        mode = getattr(item, "guidance_mode", None)
        if mode is None and isinstance(item, dict):
            mode = item.get("guidance_mode")
        if mode in {"traj_reference", "subgoal_reference"} and mode not in modes:
            modes.append(str(mode))
    return modes or ["no_reference"]


def local_guidance_mode(memory: Sequence[dict]) -> str:
    if not memory:
        return "no_reference"
    first = memory[0]
    mode = getattr(first, "guidance_mode", None)
    if mode is None and isinstance(first, dict):
        mode = first.get("guidance_mode")
    if mode in {"strict_reference", "soft_reference", "no_reference"}:
        return str(mode)
    observed_next_md = getattr(first, "observed_next_md", None)
    if observed_next_md is None and isinstance(first, dict):
        observed_next_md = first.get("observed_next_md")
    if observed_next_md:
        return "strict_reference"
    return "soft_reference"


def _confidence(value: object) -> int:
    try:
        return max(0, min(10, int(value)))
    except Exception:
        return 0


def _aligned_global_advice(
    rows: object,
    candidate_subgoals: Sequence[CandidateSubgoal],
) -> list[GlobalAdvice]:
    """Align Global-WM rows by exact subgoal text, never by a possibly wrong index."""

    if not isinstance(rows, list):
        raise ValueError("global world model path_per_subgoal must be a list")
    texts = [item.subgoal_text for item in candidate_subgoals]
    if len(set(texts)) != len(texts):
        raise ValueError("candidate subgoal texts must be unique for Global-WM alignment")
    if len(rows) != len(texts):
        raise ValueError(
            f"global world model returned {len(rows)} rows for {len(texts)} candidate subgoals"
        )
    by_index: dict[int, GlobalAdvice] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("global world model advice row must be an object")
        try:
            reported_index = int(row["candidate_index"])
        except Exception as exc:
            raise ValueError("global world model candidate_index must be an integer") from exc
        if reported_index < 0 or reported_index >= len(texts):
            raise ValueError(
                f"global world model candidate_index {reported_index} out of range for {len(texts)} candidates"
            )
        returned_text = str(row.get("candidate_subgoal_text", ""))
        try:
            authoritative_index = texts.index(returned_text)
        except ValueError as exc:
            raise ValueError(
                "global world model candidate_subgoal_text must exactly match one input candidate"
            ) from exc
        if authoritative_index in by_index:
            raise ValueError("global world model returned duplicate advice for one candidate subgoal")
        by_index[authoritative_index] = GlobalAdvice(
            candidate_index=authoritative_index,
            candidate_subgoal_text=texts[authoritative_index],
            predicted_path=str(row.get("predicted_path", "")),
            expected_end_state=str(row.get("expected_end_state", "")),
            confidence=_confidence(row.get("confidence", 0)),
        )
    if len(by_index) != len(texts):
        raise ValueError("global world model omitted one or more candidate subgoals")
    return [by_index[index] for index in range(len(texts))]


class GlobalWorldModel(Protocol):
    def predict_paths(
        self,
        goal_text: str,
        current_state: GuiState,
        candidate_subgoals: Sequence[CandidateSubgoal],
        *,
        memory: Sequence[dict] = (),
        subgoal_history: Sequence[dict] = (),
    ) -> list[GlobalAdvice]:
        ...


class LocalWorldModel(Protocol):
    def predict_next_state(
        self,
        current_state: GuiState,
        candidate_action: CandidateAction,
        *,
        memory: Sequence[dict] = (),
        guidance_mode: str | None = None,
    ) -> LocalPrediction:
        ...


@dataclass
class NoOpGlobalWorldModel:
    def predict_paths(
        self,
        goal_text: str,
        current_state: GuiState,
        candidate_subgoals: Sequence[CandidateSubgoal],
        *,
        memory: Sequence[dict] = (),
        subgoal_history: Sequence[dict] = (),
    ) -> list[GlobalAdvice]:
        return []


@dataclass
class NoOpLocalWorldModel:
    def predict_next_state(
        self,
        current_state: GuiState,
        candidate_action: CandidateAction,
        *,
        memory: Sequence[dict] = (),
        guidance_mode: str | None = None,
    ) -> LocalPrediction:
        return LocalPrediction(
            candidate_action=candidate_action,
            action_analysis="Local World Model disabled.",
            predicted_markdown=current_state.markdown,
            raw_output=current_state.markdown,
        )


def split_local_prediction(raw_output: str) -> tuple[str, str]:
    marker = "PREDICTED_MARKDOWN:"
    if marker not in raw_output:
        return "", raw_output.strip()
    analysis = raw_output.split(marker, 1)[0].replace("ACTION_ANALYSIS:", "").strip()
    predicted_markdown = raw_output.split(marker, 1)[1].strip()
    return analysis, predicted_markdown


@dataclass
class LlmGlobalWorldModel:
    client: ChatClient
    prompts: PromptRegistry = field(default_factory=PromptRegistry.from_project)

    def predict_paths(
        self,
        goal_text: str,
        current_state: GuiState,
        candidate_subgoals: Sequence[CandidateSubgoal],
        *,
        memory: Sequence[dict] = (),
        subgoal_history: Sequence[dict] = (),
    ) -> list[GlobalAdvice]:
        prompt = self.prompts.load_pair("world_global", with_memory=bool(memory))
        memory_rows = memory_rows_for_prompt(memory)
        payload = {
            "goal_text": goal_text,
            "current_md": current_state.markdown,
            "subgoal_history": list(subgoal_history),
            "candidate_subgoals": [
                {
                    "candidate_index": index,
                    "candidate_subgoal_text": item.subgoal_text,
                    "completion_condition": item.completion_condition,
                    "evidence": item.evidence,
                    "confidence": item.confidence,
                }
                for index, item in enumerate(candidate_subgoals)
            ],
            "guidance_mode": global_guidance_mode(memory),
            "retrieved_memory": memory_rows if memory_rows else None,
        }
        raw = self.client.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ]
        )
        data = parse_json_object(raw, stage="global world model")
        return _aligned_global_advice(data.get("path_per_subgoal"), candidate_subgoals)


@dataclass
class LlmLocalWorldModel:
    client: ChatClient
    prompts: PromptRegistry = field(default_factory=PromptRegistry.from_project)

    def predict_next_state(
        self,
        current_state: GuiState,
        candidate_action: CandidateAction,
        *,
        memory: Sequence[dict] = (),
        guidance_mode: str | None = None,
    ) -> LocalPrediction:
        prompt = self.prompts.load_pair("world_local", with_memory=bool(memory))
        mode = guidance_mode or local_guidance_mode(memory)
        reference = local_memory_reference(memory)
        payload = {
            "current_md": current_state.markdown,
            "current_screenshot": None,
            "_screenshot": current_state.screenshot,
            "candidate_action": candidate_action.unified_action.to_dict(),
            "guidance_mode": mode,
            "retrieved_memory": reference,
        }
        raw = self.client.chat(json_messages(prompt, payload))
        analysis, predicted_markdown = split_local_prediction(raw)
        return LocalPrediction(
            candidate_action=candidate_action,
            action_analysis=analysis,
            predicted_markdown=predicted_markdown,
            raw_output=raw,
        )
