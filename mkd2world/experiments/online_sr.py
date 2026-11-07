from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from mkd2world.components import build_pipeline_runner, pipeline_component_report
from mkd2world.experiments.common import csv_values, load_config_with_overrides, missing_path, optional_path
from mkd2world.pipeline import PipelineTaskRunner


ONLINE_METRICS = (
    "success_rate",
    "step_count",
    "fail_reason",
    "local_wm_calls",
    "global_wm_calls",
)
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_config(args: argparse.Namespace):
    methods = csv_values(args.methods) or ((args.method,) if args.method else None)
    overrides: dict[str, Any] = {
        "experiment.name": "online_success_rate",
        "experiment.dataset": args.benchmark or None,
        "experiment.output_dir": str(args.output_dir) if args.output_dir else None,
        "experiment.max_tasks": args.max_tasks,
        "experiment.methods": methods,
        "experiment.metrics": ONLINE_METRICS,
        "experiment.dry_run": args.dry_run,
        "pipeline.agent.kind": args.agent_kind or None,
        "pipeline.agent.model": args.agent_model or None,
        "pipeline.global_world_model.kind": args.global_wm_kind or None,
        "pipeline.global_world_model.model": args.global_wm_model or None,
        "pipeline.local_world_model.kind": args.local_wm_kind or None,
        "pipeline.local_world_model.model": args.local_wm_model or None,
        "pipeline.local_world_model.adapter_path": str(args.local_wm_adapter_path) if args.local_wm_adapter_path else None,
        "pipeline.memory.kind": args.memory_kind or None,
        "pipeline.environment.kind": "vwa_runtime" if args.benchmark == "visualwebarena" else (args.benchmark or None),
        "pipeline.enable_global_wm": args.enable_global_wm,
        "pipeline.enable_local_wm": args.enable_local_wm,
        "pipeline.enable_memory": args.enable_memory,
    }
    return load_config_with_overrides(args.config, overrides)


def build_dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    config = build_config(args)
    missing = [
        item
        for item in (
            missing_path(config.experiment.dataset, "benchmark"),
            missing_path(config.experiment.output_dir, "output_dir"),
        )
        if item is not None
    ]
    return {
        "status": "dry_run",
        "experiment": "online_success_rate",
        "can_run_online": False,
        "missing_assets": missing,
        "benchmark": config.experiment.dataset,
        "methods": list(config.experiment.methods or ("agent",)),
        "planned_metrics": list(ONLINE_METRICS),
        "planned_ablations": [
            "agent_only",
            "agent_plus_local_wm",
            "agent_plus_global_wm",
        ],
        "pipeline": asdict(config.pipeline),
        "components": pipeline_component_report(config),
        "planned_flow": [
            "copy frozen indexed Memory into the run directory",
            "instantiate modules from pipeline config",
            "run each task with the selected method or ablation",
            "persist per-step traces and aggregate success-rate metrics",
        ],
        "config": asdict(config),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan online success-rate experiments for Online-Mind2Web/VWA.")
    parser.add_argument("--config", type=optional_path)
    parser.add_argument("--benchmark", choices=["online_mind2web", "visualwebarena"])
    parser.add_argument("--output-dir", "--output_dir", type=optional_path)
    parser.add_argument("--method")
    parser.add_argument("--methods")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--agent-kind")
    parser.add_argument("--agent-model")
    parser.add_argument("--global-wm-kind")
    parser.add_argument("--global-wm-model")
    parser.add_argument("--local-wm-kind")
    parser.add_argument("--local-wm-model")
    parser.add_argument("--local-wm-adapter-path", "--local_wm_adapter_path", type=optional_path)
    parser.add_argument("--memory-kind")
    parser.add_argument("--task-config", type=Path, action="append", default=[])
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--no-progress-limit", type=int, default=0)
    parser.add_argument("--repeat-action-limit", type=int, default=5)
    parser.add_argument("--frozen-memory-db", type=Path)
    parser.add_argument("--enable-global-wm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-local-wm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _task_goal(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"VWA task config must contain an object: {path}")
    goal = str(payload.get("intent") or payload.get("goal") or "").strip()
    if not goal:
        raise ValueError(f"VWA task config has no intent: {path}")
    return goal, payload


def _prepare_runtime_config(config, args: argparse.Namespace, output_dir: Path):
    pipeline = config.pipeline
    environment = replace(pipeline.environment, kind="vwa_runtime")
    memory = replace(pipeline.memory, kind="noop", enabled=False, adapter_path="")
    pipeline = replace(pipeline, environment=environment, memory=memory, enable_memory=False)
    return replace(config, pipeline=pipeline), None


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    if args.benchmark != "visualwebarena":
        raise NotImplementedError("The real task runner currently supports VisualWebArena first")
    task_paths = [path.resolve() for path in args.task_config]
    if not task_paths:
        raise ValueError("At least one --task-config is required")
    missing = [str(path) for path in task_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing VWA task configs: {missing}")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    config = build_config(args)
    run_id = args.run_id or datetime.now().strftime("vwa_%Y%m%d_%H%M%S")
    base_output = Path(config.experiment.output_dir or "output/pipeline_runs")
    output_dir = base_output if base_output.name == run_id else base_output / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    config, memory_copy = _prepare_runtime_config(config, args, output_dir)
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if memory_copy is not None:
        (output_dir / "memory/copy_report.json").write_text(
            json.dumps(memory_copy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    events_path = output_dir / "events.jsonl"
    results_path = output_dir / "task_results.jsonl"
    task_context: dict[str, Any] = {}

    def event_sink(event: dict[str, Any]) -> None:
        row = {"time": _timestamp(), **task_context, **event}
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    pipeline = build_pipeline_runner(config)
    task_runner = PipelineTaskRunner(
        pipeline,
        max_steps=args.max_steps,
        no_progress_limit=args.no_progress_limit,
        repeat_action_limit=args.repeat_action_limit,
        event_sink=event_sink,
    )
    task_rows = []
    try:
        for index, task_path in enumerate(task_paths):
            goal_text, task_payload = _task_goal(task_path)
            task_context.clear()
            task_context.update(
                {
                    "task_index": index,
                    "task_id": task_payload.get("task_id"),
                    "task_config": str(task_path),
                }
            )
            result = task_runner.run(goal_text, task=task_path)
            row = {
                **task_context,
                "goal_text": goal_text,
                "success": result.success,
                "evaluator_score": result.evaluator_score,
                "terminal_reason": result.terminal_reason,
                "error": result.error,
                "step_count": len(result.step_results),
                "actions": [step.selected_action.unified_action.to_dict() for step in result.step_results],
                "step_timings": [step.timings for step in result.step_results],
                "model_calls": result.model_calls,
                "model_output_statistics": result.model_output_statistics,
                "trajectory_memory": result.trajectory_memory,
            }
            task_rows.append(row)
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    finally:
        close = getattr(pipeline.environment, "close", None)
        if callable(close):
            close()
    summary = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "benchmark": args.benchmark,
        "task_count": len(task_rows),
        "success_count": sum(bool(row["success"]) for row in task_rows),
        "success_rate": (
            sum(bool(row["success"]) for row in task_rows) / len(task_rows) if task_rows else 0.0
        ),
        "total_steps": sum(int(row["step_count"]) for row in task_rows),
        "created_at": _timestamp(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_dry_run_report(args), ensure_ascii=False, indent=2, default=str))
        return
    print(json.dumps(run_online(args), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
