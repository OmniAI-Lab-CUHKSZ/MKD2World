"""Single-step pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from mkd2world.actions import ActionType, action_to_prompt_dict
from mkd2world.agent import AgentContext, GuiAgent
from mkd2world.contracts import (
    GuiState,
    LocalPrediction,
    PipelineStepResult,
    PipelineTaskResult,
    StepRecord,
)
from mkd2world.envs import EnvironmentAdapter
from mkd2world.memory import MemoryStore, NoOpMemoryStore, TrajectoryMemoryUpdate
from mkd2world.world_model import GlobalWorldModel, LocalWorldModel


TERMINAL_ACTION_TYPES = {ActionType.FINISH, ActionType.FAIL}
DIRECT_MODE_PROMPTS = {
    "vanilla": "gui_agent_vanilla",
    "vanilla_md": "gui_agent_vanilla_md",
}
PIPELINE_MODES = frozenset({"full", "local_wm", *DIRECT_MODE_PROMPTS})


def pipeline_model_statistics(pipeline: "PipelineRunner") -> dict[str, Any]:
    """Return cumulative output/token statistics without coupling components to the runner."""
    components = {
        "gui_agent": pipeline.agent,
        "global_world_model": pipeline.global_world_model,
        "local_world_model": pipeline.local_world_model,
        "trajectory_manager": getattr(pipeline.memory, "trajectory_manager", None),
    }
    rows: dict[str, Any] = {}
    for name, component in components.items():
        if component is None:
            continue
        source = getattr(component, "client", component)
        snapshot = getattr(source, "telemetry_snapshot", None)
        if callable(snapshot):
            rows[name] = snapshot()
    return rows


def reset_pipeline_model_statistics(pipeline: "PipelineRunner") -> None:
    """Reset per-task client counters when one pipeline is reused for a task list."""
    components = (
        pipeline.agent,
        pipeline.global_world_model,
        pipeline.local_world_model,
        getattr(pipeline.memory, "trajectory_manager", None),
    )
    for component in components:
        source = getattr(component, "client", component)
        reset = getattr(source, "reset_telemetry", None)
        if callable(reset):
            reset()


@dataclass
class PipelineRunner:
    agent: GuiAgent
    global_world_model: GlobalWorldModel
    local_world_model: LocalWorldModel
    environment: EnvironmentAdapter
    memory: MemoryStore = field(default_factory=NoOpMemoryStore)
    mode: str = "full"
    subgoal_history: list[dict] = field(default_factory=list)
    current_subgoal_trace: list[dict] = field(default_factory=list)
    step_buffer: list[StepRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in PIPELINE_MODES:
            raise ValueError(f"Unsupported pipeline mode: {self.mode}")

    def run_step(self, goal_text: str, current_state: GuiState | None = None) -> PipelineStepResult:
        total_started = perf_counter()
        timings: dict[str, float] = {}
        model_calls: dict[str, int] = {}
        stage_errors: dict[str, str] = {}
        before_state = current_state or self.environment.observe()
        context = AgentContext(
            goal_text=goal_text,
            current_state=before_state,
            subgoal_history=tuple(self.subgoal_history),
            current_subgoal_trace=tuple(self.current_subgoal_trace),
            supported_unified_actions=tuple(self.environment.supported_actions),
        )

        if self.mode in DIRECT_MODE_PROMPTS:
            direct_action = getattr(self.agent, "direct_action", None)
            if direct_action is None:
                raise TypeError(f"Agent does not support direct_action required by pipeline mode: {self.mode}")
            started = perf_counter()
            selected = direct_action(context, prompt_key=DIRECT_MODE_PROMPTS[self.mode], current_subgoal_text=None)
            timings["agent_direct"] = perf_counter() - started
            model_calls["gui_agent"] = 1
            return self._execute_selected(
                goal_text,
                before_state,
                selected,
                subgoal_text=None,
                completion_condition="",
                candidate_actions=(selected,),
                model_calls=model_calls,
                timings=timings,
                total_started=total_started,
            )

        if self.mode == "local_wm":
            # Local-WM-only ablation has no subgoal/global path. The Agent emits
            # actions directly from the task and current state.
            subgoals = []
            advice = []
            global_memory = []
            selected_subgoal_text = None
            completion_condition = ""
            action_context = context
            timings["agent_candidate_subgoals"] = 0.0
            timings["memory_global_retrieval"] = 0.0
            timings["global_world_model"] = 0.0
            timings["agent_best_subgoal"] = 0.0
        else:
            started = perf_counter()
            subgoals = self.agent.candidate_subgoals(context)
            timings["agent_candidate_subgoals"] = perf_counter() - started
            model_calls["gui_agent"] = 1
            if not subgoals:
                raise ValueError("GUI Agent returned no candidate subgoals")
            started = perf_counter()
            global_memory = self.memory.retrieve_global(
                goal_text, before_state.markdown, [item.subgoal_text for item in subgoals]
            )
            timings["memory_global_retrieval"] = perf_counter() - started
            if self.global_world_model.__class__.__name__.startswith("NoOp"):
                advice = []
                timings["global_world_model"] = 0.0
            else:
                started = perf_counter()
                advice = self.global_world_model.predict_paths(
                    goal_text,
                    before_state,
                    subgoals,
                    memory=global_memory,
                    subgoal_history=context.subgoal_history,
                )
                timings["global_world_model"] = perf_counter() - started
                model_calls["global_world_model"] = 1
            started = perf_counter()
            try:
                best_subgoal = self.agent.best_subgoal(context, subgoals, advice)
            except Exception as exc:
                if len(subgoals) != 1:
                    raise
                best_subgoal = subgoals[0]
                stage_errors["agent_best_subgoal"] = f"{type(exc).__name__}: {exc}"
            timings["agent_best_subgoal"] = perf_counter() - started
            model_calls["gui_agent"] += 1
            if not any(item.subgoal_text == best_subgoal.subgoal_text for item in subgoals):
                raise ValueError("GUI Agent selected a subgoal outside the candidate set")
            selected_subgoal_text = best_subgoal.subgoal_text
            completion_condition = best_subgoal.completion_condition
            self._record_active_subgoal(selected_subgoal_text, completion_condition)
            action_context = AgentContext(
                goal_text=goal_text,
                current_state=before_state,
                subgoal_history=tuple(self.subgoal_history),
                current_subgoal_trace=tuple(self.current_subgoal_trace),
                supported_unified_actions=tuple(self.environment.supported_actions),
            )

        started = perf_counter()
        actions = self.agent.candidate_actions(
            action_context, current_subgoal_text=selected_subgoal_text
        )
        timings["agent_candidate_actions"] = perf_counter() - started
        model_calls["gui_agent"] = model_calls.get("gui_agent", 0) + 1
        actions = [action for action in actions if action.unified_action.type.value in self.environment.supported_actions]
        if not actions:
            raise ValueError("GUI Agent returned no supported candidate actions")
        # One candidate means the Agent is already certain. Confidence remains
        # diagnostic only: Local WM is used whenever the Agent emits alternatives.
        if len(actions) == 1:
            timings["memory_local_retrieval"] = 0.0
            timings["local_world_model"] = 0.0
            timings["agent_best_action"] = 0.0
            return self._execute_selected(
                goal_text,
                before_state,
                actions[0],
                subgoal_text=selected_subgoal_text,
                completion_condition=completion_condition,
                candidate_subgoals=tuple(subgoals),
                global_advice=tuple(advice),
                candidate_actions=tuple(actions),
                local_predictions=(),
                memory_retrievals={"global": list(global_memory), "local": []},
                model_calls=model_calls,
                stage_errors=stage_errors,
                timings=timings,
                total_started=total_started,
            )
        predictions: list[LocalPrediction | None] = [None] * len(actions)
        local_memory_records: list[dict[str, Any]] = []
        local_retrieval_time = 0.0
        local_world_model_time = 0.0
        batched_predict = getattr(self.local_world_model, "predict_next_states", None)
        use_batched_predict = callable(batched_predict) and int(getattr(self.local_world_model, "batch_size", 1)) > 1
        batch_indices: list[int] = []
        batch_actions = []
        batch_memories = []
        for action_index, action in enumerate(actions):
            if action.unified_action.type in TERMINAL_ACTION_TYPES:
                predictions[action_index] = LocalPrediction(
                    candidate_action=action,
                    action_analysis="Terminal action; Local World Model prediction skipped.",
                    predicted_markdown=before_state.markdown,
                    raw_output=before_state.markdown,
                )
                local_memory_records.append(
                    {"action": action.unified_action.to_dict(), "skipped": "terminal_action", "hits": []}
                )
                continue
            started = perf_counter()
            local_memory = self.memory.retrieve_local(
                before_state.markdown,
                action.unified_action.to_dict(),
                current_url=before_state.url,
            )
            local_retrieval_time += perf_counter() - started
            local_memory_records.append(
                {"action": action.unified_action.to_dict(), "hits": list(local_memory)}
            )
            if use_batched_predict:
                batch_indices.append(action_index)
                batch_actions.append(action)
                batch_memories.append(local_memory)
                continue
            started = perf_counter()
            predictions[action_index] = self.local_world_model.predict_next_state(
                before_state, action, memory=local_memory
            )
            model_calls["local_world_model"] = model_calls.get("local_world_model", 0) + 1
            local_world_model_time += perf_counter() - started
        if batch_actions:
            started = perf_counter()
            batch_predictions = batched_predict(before_state, batch_actions, memories=batch_memories)
            local_world_model_time += perf_counter() - started
            if len(batch_predictions) != len(batch_indices):
                raise ValueError("Local WM batched prediction count does not match candidate actions")
            for action_index, prediction in zip(batch_indices, batch_predictions):
                predictions[action_index] = prediction
            model_calls["local_world_model"] = model_calls.get("local_world_model", 0) + len(batch_actions)
        if any(prediction is None for prediction in predictions):
            raise RuntimeError("Local WM prediction alignment contains an empty slot")
        aligned_predictions = tuple(prediction for prediction in predictions if prediction is not None)
        timings["memory_local_retrieval"] = local_retrieval_time
        timings["local_world_model"] = local_world_model_time

        started = perf_counter()
        try:
            selected = self.agent.best_action(action_context, actions, aligned_predictions)
        except Exception as exc:
            if len(actions) != 1:
                raise
            selected = actions[0]
            stage_errors["agent_best_action"] = f"{type(exc).__name__}: {exc}"
        timings["agent_best_action"] = perf_counter() - started
        model_calls["gui_agent"] += 1
        if selected not in actions:
            raise ValueError("GUI Agent selected an action outside the supported candidate set")
        return self._execute_selected(
            goal_text,
            before_state,
            selected,
            subgoal_text=selected_subgoal_text,
            completion_condition=completion_condition,
            candidate_subgoals=tuple(subgoals),
            global_advice=tuple(advice),
            candidate_actions=tuple(actions),
            local_predictions=aligned_predictions,
            memory_retrievals={"global": list(global_memory), "local": local_memory_records},
            model_calls=model_calls,
            stage_errors=stage_errors,
            timings=timings,
            total_started=total_started,
        )

    def _execute_selected(
        self,
        goal_text: str,
        before_state: GuiState,
        selected,
        *,
        subgoal_text: str | None,
        completion_condition: str,
        candidate_subgoals=(),
        global_advice=(),
        candidate_actions=(),
        local_predictions=(),
        memory_retrievals: dict[str, Any] | None = None,
        model_calls: dict[str, int] | None = None,
        stage_errors: dict[str, str] | None = None,
        timings: dict[str, float] | None = None,
        total_started: float | None = None,
    ) -> PipelineStepResult:
        timings = dict(timings or {})
        if selected.unified_action.type.value not in self.environment.supported_actions:
            raise ValueError(f"Environment does not support action type: {selected.unified_action.type.value}")
        started = perf_counter()
        after_state = self.environment.execute(selected.unified_action)
        timings["environment_execute"] = perf_counter() - started
        step_order = len(self.step_buffer)
        record = StepRecord(
            before_state=before_state,
            action=selected.unified_action,
            action_intent=selected.action_intent,
            after_state=after_state,
            subgoal_text=subgoal_text,
            step_order=step_order,
        )
        started = perf_counter()
        update = self.memory.record_step(record)
        timings["memory_record_step"] = perf_counter() - started
        self.step_buffer.append(record)
        trace_item = {
            "action": action_to_prompt_dict(selected.unified_action),
            "action_intent": selected.action_intent,
            "observed_next_md": after_state.markdown,
        }
        if subgoal_text is not None:
            trace_item["subgoal_text"] = subgoal_text
            trace_item["step_order"] = step_order
        self.current_subgoal_trace.append(trace_item)
        if total_started is not None:
            timings["total"] = perf_counter() - total_started
        return PipelineStepResult(
            selected_action=selected,
            before_state=before_state,
            after_state=after_state,
            candidate_subgoals=tuple(candidate_subgoals),
            global_advice=tuple(global_advice),
            candidate_actions=tuple(candidate_actions),
            local_predictions=tuple(local_predictions),
            memory_retrievals=dict(memory_retrievals or {}),
            memory_updates=update,
            model_calls=dict(model_calls or {}),
            model_output_statistics=pipeline_model_statistics(self),
            stage_errors=dict(stage_errors or {}),
            timings=timings,
        )

    def finish_trajectory(self, goal_text: str, success: bool) -> TrajectoryMemoryUpdate:
        update = self.memory.finish_trajectory(
            goal_text=goal_text,
            subgoal_history=list(self.subgoal_history),
            success=success,
            step_buffer=list(self.step_buffer),
        )
        self.subgoal_history.clear()
        self.current_subgoal_trace.clear()
        self.step_buffer.clear()
        return update

    def _record_active_subgoal(self, subgoal_text: str, completion_condition: str) -> None:
        if (
            self.subgoal_history
            and self.subgoal_history[-1].get("subgoal_text") == subgoal_text
            and self.subgoal_history[-1].get("status") == "active"
        ):
            if completion_condition and not self.subgoal_history[-1].get("completion_condition"):
                self.subgoal_history[-1]["completion_condition"] = completion_condition
            return
        if self.subgoal_history and self.subgoal_history[-1].get("status") == "active":
            self.subgoal_history[-1]["status"] = "interrupted"
            self.subgoal_history[-1]["interrupted_reason"] = "A different subgoal was selected before completion."
            self.current_subgoal_trace.clear()
        self.subgoal_history.append(
            {
                "subgoal_text": subgoal_text,
                "completion_condition": completion_condition,
                "status": "active",
            }
        )

def _evaluator_score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "reward", "success"):
            if key in value:
                return _evaluator_score(value[key])
    return 0.0


def _unified_actions_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Match VisualWebArena's repeated-action semantics on unified actions."""
    if left.get("type") != right.get("type"):
        return False
    action_type = left.get("type")
    if action_type == ActionType.SCROLL.value:
        left_y = float(left.get("delta_y", 0))
        right_y = float(right.get("delta_y", 0))
        left_x = float(left.get("delta_x", 0))
        right_x = float(right.get("delta_x", 0))
        left_axis = "y" if abs(left_y) >= abs(left_x) else "x"
        right_axis = "y" if abs(right_y) >= abs(right_x) else "x"
        if left_axis != right_axis:
            return False
        left_delta = left_y if left_axis == "y" else left_x
        right_delta = right_y if right_axis == "y" else right_x
        return (left_delta >= 0) == (right_delta >= 0)
    return left == right


@dataclass
class PipelineTaskRunner:
    pipeline: PipelineRunner
    max_steps: int = 50
    no_progress_limit: int = 4
    repeat_action_limit: int = 0
    event_sink: Callable[[dict[str, Any]], None] | None = None

    def run(self, goal_text: str, *, task: Any = None, close_environment: bool = False) -> PipelineTaskResult:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        # Reset per-trajectory state; the runner is reused across tasks.
        self.pipeline.subgoal_history.clear()
        self.pipeline.current_subgoal_trace.clear()
        self.pipeline.step_buffer.clear()
        reset_pipeline_model_statistics(self.pipeline)
        environment = self.pipeline.environment
        step_results: list[PipelineStepResult] = []
        terminal_reason = "max_steps"
        error = None
        evaluator_score = 0.0
        current_state: GuiState | None = None
        trajectory_update = None
        finalized = False
        unchanged_steps = 0
        action_history: list[dict[str, Any]] = []
        task_model_calls: dict[str, int] = {}
        try:
            reset = getattr(environment, "reset", None)
            current_state = reset(task) if callable(reset) else environment.observe()
            self._event("task_started", goal_text=goal_text)
            for _ in range(self.max_steps):
                result = self.pipeline.run_step(goal_text, current_state=current_state)
                step_results.append(result)
                for name, count in result.model_calls.items():
                    task_model_calls[name] = task_model_calls.get(name, 0) + int(count)
                self._event(
                    "step_finished",
                    step=len(step_results) - 1,
                    action=result.selected_action.unified_action.to_dict(),
                    timings=result.timings,
                    memory_retrievals=result.memory_retrievals,
                    memory_updates=result.memory_updates,
                    model_calls=result.model_calls,
                    model_output_statistics=result.model_output_statistics,
                    stage_errors=result.stage_errors,
                )
                current_state = result.after_state
                action_type = result.selected_action.unified_action.type
                if action_type in TERMINAL_ACTION_TYPES:
                    terminal_reason = action_type.value
                    break
                if bool(getattr(environment, "done", False)):
                    terminal_reason = "environment_done"
                    break
                action_payload = action_to_prompt_dict(result.selected_action.unified_action)
                action_history.append(action_payload)
                if self.repeat_action_limit > 0:
                    if action_payload.get("type") == ActionType.TYPE_TEXT.value:
                        repeated = sum(
                            _unified_actions_equivalent(item, action_payload)
                            for item in action_history
                        )
                        repeat_limit_hit = repeated >= self.repeat_action_limit
                    else:
                        recent = action_history[-self.repeat_action_limit :]
                        repeat_limit_hit = (
                            len(recent) >= self.repeat_action_limit
                            and all(
                                _unified_actions_equivalent(item, action_payload)
                                for item in recent
                            )
                        )
                    if repeat_limit_hit:
                        terminal_reason = "repeated_action"
                        break
                if result.before_state.markdown == result.after_state.markdown:
                    unchanged_steps += 1
                else:
                    unchanged_steps = 0
                if self.no_progress_limit > 0 and unchanged_steps >= self.no_progress_limit:
                    terminal_reason = "repeated_no_progress"
                    break
            evaluate = getattr(environment, "evaluate", None)
            evaluator_score = _evaluator_score(evaluate() if callable(evaluate) else False)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            terminal_reason = "error"
            self._event("task_error", error=error)
            try:
                evaluate = getattr(environment, "evaluate", None)
                evaluator_score = _evaluator_score(evaluate() if callable(evaluate) else False)
            except Exception:
                evaluator_score = 0.0
        finally:
            success = evaluator_score > 0
            self._finalize_active_subgoal(success, terminal_reason)
            try:
                trajectory_manager = getattr(self.pipeline.memory, "trajectory_manager", None)
                if trajectory_manager is not None and trajectory_manager.__class__.__name__.startswith("Llm"):
                    task_model_calls["trajectory_manager"] = task_model_calls.get("trajectory_manager", 0) + 1
                trajectory_update = self.pipeline.finish_trajectory(goal_text, success)
                finalized = True
            except Exception as exc:
                finish_error = f"{type(exc).__name__}: {exc}"
                error = f"{error}; trajectory_finalize={finish_error}" if error else f"trajectory_finalize={finish_error}"
            if close_environment:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()
        self._event(
            "task_finished",
            success=evaluator_score > 0,
            evaluator_score=evaluator_score,
            terminal_reason=terminal_reason,
            finalized=finalized,
            error=error,
            model_calls=task_model_calls,
            model_output_statistics=pipeline_model_statistics(self.pipeline),
        )
        return PipelineTaskResult(
            goal_text=goal_text,
            success=evaluator_score > 0,
            terminal_reason=terminal_reason,
            step_results=tuple(step_results),
            trajectory_memory=trajectory_update,
            evaluator_score=evaluator_score,
            error=error,
            model_calls=task_model_calls,
            model_output_statistics=pipeline_model_statistics(self.pipeline),
        )

    def _finalize_active_subgoal(self, success: bool, terminal_reason: str) -> None:
        for item in reversed(self.pipeline.subgoal_history):
            if item.get("status") != "active":
                continue
            item["status"] = "completed" if success else "failed"
            item["confidence"] = 10
            item["evidence"] = (
                "Benchmark evaluator marked the task successful."
                if success
                else f"Task ended without evaluator success: {terminal_reason}."
            )
            item["completed_at_step_order"] = (
                len(self.pipeline.step_buffer) - 1 if success and self.pipeline.step_buffer else None
            )
            break

    def _event(self, event: str, **payload: Any) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, **payload})


@dataclass
class DryRunEnvironment:
    state: GuiState

    @property
    def supported_actions(self) -> tuple[str, ...]:
        return tuple(item.value for item in ActionType)

    def observe(self) -> GuiState:
        return self.state

    def execute(self, action) -> GuiState:
        return self.state
