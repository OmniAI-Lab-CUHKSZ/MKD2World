
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from mkd2world.actions import UnifiedAction


@dataclass(frozen=True)
class GuiState:
    screenshot: bytes | None
    markdown: str
    url: str | None = None
    environment: str = "Web"


@dataclass(frozen=True)
class CandidateSubgoal:
    subgoal_text: str
    global_plan: str
    completion_condition: str
    evidence: str
    confidence: int


@dataclass(frozen=True)
class CandidateAction:
    unified_action: UnifiedAction
    action_intent: str
    evidence: str
    confidence: int


@dataclass(frozen=True)
class GlobalAdvice:
    candidate_index: int
    candidate_subgoal_text: str
    predicted_path: str
    expected_end_state: str
    confidence: int


@dataclass(frozen=True)
class LocalPrediction:
    candidate_action: CandidateAction
    action_analysis: str
    predicted_markdown: str
    raw_output: str
    predicted_delta: str | None = None
    output_state_mode: str = "replayed_next"


@dataclass(frozen=True)
class StepRecord:
    before_state: GuiState
    action: UnifiedAction
    action_intent: str
    after_state: GuiState
    subgoal_text: str | None = None
    step_order: int | None = None


@dataclass(frozen=True)
class PipelineStepResult:
    selected_action: CandidateAction
    before_state: GuiState
    after_state: GuiState
    candidate_subgoals: Sequence[CandidateSubgoal] = field(default_factory=tuple)
    global_advice: Sequence[GlobalAdvice] = field(default_factory=tuple)
    candidate_actions: Sequence[CandidateAction] = field(default_factory=tuple)
    local_predictions: Sequence[LocalPrediction] = field(default_factory=tuple)
    memory_retrievals: dict[str, Any] = field(default_factory=dict)
    memory_updates: dict[str, Any] = field(default_factory=dict)
    model_calls: dict[str, int] = field(default_factory=dict)
    model_output_statistics: dict[str, Any] = field(default_factory=dict)
    stage_errors: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineTaskResult:
    goal_text: str
    success: bool
    terminal_reason: str
    step_results: Sequence[PipelineStepResult] = field(default_factory=tuple)
    trajectory_memory: Any = None
    evaluator_score: float | None = None
    error: str | None = None
    model_calls: dict[str, int] = field(default_factory=dict)
    model_output_statistics: dict[str, Any] = field(default_factory=dict)
