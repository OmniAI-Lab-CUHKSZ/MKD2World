from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from mkd2world.contracts import StepRecord


@dataclass(frozen=True)
class TrajectoryMemoryUpdate:
    subgoal_completion_conditions: tuple[dict, ...] = ()
    trajectory_summary: str = "None"
    success_pattern: str = "None"
    failure_reason: str = "None"
    manager_error: str | None = None


class MemoryStore(Protocol):
    def retrieve_global(self, goal_text: str, current_md: str, candidate_subgoals: list[str]) -> list[dict]:
        ...

    def retrieve_local(self, current_md: str, action: dict, current_url: str | None = None) -> list[dict]:
        ...

    def record_step(self, record: StepRecord) -> dict:
        ...

    def finish_trajectory(
        self,
        goal_text: str,
        subgoal_history: list[dict],
        success: bool,
        step_buffer: Sequence[StepRecord],
    ) -> TrajectoryMemoryUpdate:
        ...


@dataclass
class NoOpMemoryStore:
    recorded_steps: list[StepRecord] = field(default_factory=list)

    def retrieve_global(self, goal_text: str, current_md: str, candidate_subgoals: list[str]) -> list[dict]:
        return []

    def retrieve_local(self, current_md: str, action: dict, current_url: str | None = None) -> list[dict]:
        return []

    def record_step(self, record: StepRecord) -> dict:
        self.recorded_steps.append(record)
        return {"stored": False, "reason": "no-op memory store"}

    def finish_trajectory(
        self,
        goal_text: str,
        subgoal_history: list[dict],
        success: bool,
        step_buffer: Sequence[StepRecord],
    ) -> TrajectoryMemoryUpdate:
        return TrajectoryMemoryUpdate(
            success_pattern="None" if not success else "No-op store does not summarize trajectories",
            failure_reason="None" if success else "No-op store does not summarize trajectories",
        )
