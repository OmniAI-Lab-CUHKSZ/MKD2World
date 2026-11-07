from __future__ import annotations

import argparse
import enum
import json
import multiprocessing as mp
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return str(self.value)

    enum.StrEnum = StrEnum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mkd2world.experiments import online_sr  # noqa: E402
from mkd2world.pipeline import PipelineTaskRunner  # noqa: E402
from eval import vwa_mkd2world_p21_runner as mkd2world  # noqa: E402


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def task_goal(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    goal = str(payload.get("intent") or payload.get("goal") or "").strip()
    if not goal:
        raise ValueError(f"VWA task config has no intent: {path}")
    return goal, payload


def make_args(config_path: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark="visualwebarena",
        task_config=[],
        config=config_path,
        output_dir=output_dir,
        method="agent_plus_local_wm",
        methods=None,
        max_tasks=None,
        agent_kind=None,
        agent_model=None,
        global_wm_kind=None,
        global_wm_model=None,
        local_wm_kind=None,
        local_wm_model=None,
        local_wm_adapter_path=None,
        memory_kind=None,
        run_id=output_dir.name,
        max_steps=50,
        no_progress_limit=4,
        repeat_action_limit=5,
        frozen_memory_db=None,
        enable_global_wm=False,
        enable_local_wm=False,
        enable_memory=False,
        dry_run=False,
    )


def worker(
    worker_id: int,
    config_path: str,
    agent_endpoint: str,
    world_endpoints: str,
    tasks: mp.Queue,
    output_root: str,
) -> None:
    os.environ["VWA_P21_AGENT_ENDPOINT"] = agent_endpoint
    os.environ["VWA_P21_WORLD_ENDPOINTS"] = world_endpoints
    worker_dir = Path(output_root) / f"worker{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VWA_P21_AUDIT_PATH"] = str(worker_dir / "p21_policy_audit.jsonl")
    result_path = worker_dir / "task_results.jsonl"
    event_path = worker_dir / "events.jsonl"
    args = make_args(Path(config_path), worker_dir)
    config = online_sr.build_config(args)
    config, _ = online_sr._prepare_runtime_config(config, args, worker_dir)

    def event_sink(event: dict[str, Any]) -> None:
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"time": timestamp(), **event}, ensure_ascii=False, default=json_default)
                + "\n"
            )

    pipeline = None
    try:
        pipeline = mkd2world.build_p21_pipeline(config)
        runner = PipelineTaskRunner(
            pipeline,
            max_steps=50,
            no_progress_limit=4,
            repeat_action_limit=5,
            event_sink=event_sink,
        )
        while True:
            task = tasks.get()
            if task is None:
                break
            task_path = Path(task)
            try:
                goal_text, task_payload = task_goal(task_path)
                result = runner.run(goal_text, task=task_path)
                row = {
                    "worker_id": worker_id,
                    "task_id": task_payload.get("task_id"),
                    "task_config": str(task_path),
                    "goal_text": goal_text,
                    "success": result.success,
                    "evaluator_score": result.evaluator_score,
                    "terminal_reason": result.terminal_reason,
                    "error": result.error,
                    "step_count": len(result.step_results),
                    "actions": [
                        step.selected_action.unified_action.to_dict()
                        for step in result.step_results
                    ],
                    "step_timings": [step.timings for step in result.step_results],
                    "model_calls": result.model_calls,
                    "model_output_statistics": result.model_output_statistics,
                    "trajectory_memory": result.trajectory_memory,
                }
            except Exception as exc:
                row = {
                    "worker_id": worker_id,
                    "task_config": str(task_path),
                    "success": False,
                    "terminal_reason": "worker_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "step_count": 0,
                }
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    finally:
        close = getattr(getattr(pipeline, "environment", None), "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agent-endpoints", required=True)
    parser.add_argument("--world-endpoints", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    task_paths = [
        line.strip()
        for line in args.task_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not task_paths:
        raise SystemExit("task list is empty")
    agent_endpoints = [item.strip() for item in args.agent_endpoints.split(",") if item.strip()]
    if not agent_endpoints:
        raise SystemExit("at least one agent endpoint is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    for task in task_paths:
        queue.put(task)
    for _ in agent_endpoints:
        queue.put(None)
    processes = [
        ctx.Process(
            target=worker,
            args=(
                i,
                str(args.config),
                endpoint,
                args.world_endpoints,
                queue,
                str(args.output_dir),
            ),
            name=f"mkd2world-worker-{i}",
        )
        for i, endpoint in enumerate(agent_endpoints)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process for process in processes if process.exitcode != 0]
    if failed:
        raise SystemExit(f"worker failures: {[(p.name, p.exitcode) for p in failed]}")
    files = sorted(args.output_dir.glob("worker*/task_results.jsonl"))
    rows = []
    for path in files:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    success = sum(bool(row.get("success")) for row in rows)
    summary = {
        "run_id": args.run_id,
        "output_dir": str(args.output_dir),
        "task_count": len(rows),
        "unique_task_count": len({row.get("task_config") for row in rows}),
        "success_count": success,
        "success_rate": success / len(rows) if rows else 0.0,
        "worker_count": len(agent_endpoints),
        "dynamic_task_pool": True,
        "strict_complete": len(rows) == len(task_paths)
        and len({row.get("task_config") for row in rows}) == len(task_paths),
        "created_at": timestamp(),
        "result_files": [str(path) for path in files],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["strict_complete"]:
        raise SystemExit(24)


if __name__ == "__main__":
    main()
