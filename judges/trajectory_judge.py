from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORIES = (
    ROOT / "output/online_mind2web/qwen35b_dreamer_fixed155_20260724"
)
DEFAULT_MASK = (
    ROOT
    / "output/online_mind2web/comparisons/qwen35b_fixed155_20260723"
    / "environment_valid_mask_155.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output/online_mind2web/gpt56_sol_judge"
    / "qwen35b_dreamer_fixed155_20260724"
)
DEFAULT_OFFICIAL_WEBJUDGE = (
    ROOT
    / "output/online_mind2web/webjudge"
    / "qwen35b_dreamer_fixed155_20260724_webjudge7b"
)
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_OPENAI_MODEL = "openai/gpt-5.6-sol"
DEFAULT_REASONING = "low"
DEFAULT_PROXY = "http://127.0.0.1:11080"
DEFAULT_OPENAI_BASE_URL = ""
DEFAULT_OPENAI_API_KEY_FILE = "MKD2WORLD_EXTERNAL_ROOT/secrets/openai_compatible_api_key"
VALID_TECHNICAL_TYPES = {"none", "environment", "model", "both"}
IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rel_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_mask(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_ids = payload.get("included_task_ids")
    if not isinstance(task_ids, list):
        raise RuntimeError(f"{path} has no included_task_ids list")
    task_ids = [str(value) for value in task_ids]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise RuntimeError(
            f"Frozen mask must contain a non-empty list of unique IDs, found "
            f"{len(task_ids)} rows / {len(set(task_ids))} unique"
        )
    if payload.get("included_tasks") != len(task_ids):
        raise RuntimeError(
            f"{path} declares included_tasks={payload.get('included_tasks')!r}, "
            f"but contains {len(task_ids)} task IDs"
        )
    return payload, task_ids


def load_task(trajectories: Path, task_id: str) -> dict[str, Any]:
    task_dir = trajectories / task_id
    if not task_dir.is_dir():
        lane_matches = [
            lane_dir / task_id
            for lane_dir in sorted(trajectories.glob("lane_*"))
            if (lane_dir / task_id).is_dir()
        ]

        complete_matches = [
            p for p in lane_matches
            if all((p / name).is_file() for name in ("result.json", "run_metadata.json", "raw_steps.json"))
        ]
        if complete_matches:
            task_dir = complete_matches[0]
        elif len(lane_matches) != 1:
            raise FileNotFoundError(
                f"Could not resolve exactly one trajectory directory for {task_id}: "
                f"{lane_matches}"
            )
        else:
            task_dir = lane_matches[0]
    result_path = task_dir / "result.json"
    metadata_path = task_dir / "run_metadata.json"
    raw_steps_path = task_dir / "raw_steps.json"
    for path in (result_path, metadata_path, raw_steps_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_steps = json.loads(raw_steps_path.read_text(encoding="utf-8"))
    if result.get("task_id") != task_id:
        raise RuntimeError(
            f"{result_path}: task_id={result.get('task_id')!r}, expected {task_id!r}"
        )
    screenshots = sorted(
        (task_dir / "trajectory").glob("*.png"),
        key=lambda path: int(path.stem),
    )
    history = result.get("action_history")
    if not isinstance(history, list) or not screenshots:
        raise RuntimeError(f"{task_id}: empty action history or screenshot sequence")
    screenshot_names = [path.name for path in screenshots]
    history_screenshots = [
        str(row.get("screenshot"))
        for row in history
        if isinstance(row, dict) and row.get("screenshot")
    ]
    missing = sorted(set(history_screenshots) - set(screenshot_names))
    if missing:
        raise RuntimeError(f"{task_id}: history references missing screenshots: {missing}")
    execution_errors = [
        {"step": row.get("step"), "execution_error": row.get("execution_error")}
        for row in raw_steps
        if isinstance(row, dict) and row.get("execution_error")
    ]
    return {
        "task_dir": task_dir,
        "result": result,
        "metadata": metadata,
        "raw_steps": raw_steps,
        "screenshots": screenshots,
        "execution_errors": execution_errors,
    }


def build_prompt(task_id: str, task: dict[str, Any]) -> str:
    result = task["result"]
    metadata = task["metadata"]
    screenshot_names = [path.name for path in task["screenshots"]]
    evidence = {
        "task_id": task_id,
        "task_instruction": result.get("task"),
        "agent_final_answer": result.get("agent_final_answer"),
        "reference_length": result.get("reference_length"),
        "run_metadata": metadata,
        "action_history": result.get("action_history"),
        "runtime_diagnostics": {
            "raw_step_count": len(task["raw_steps"]),
            "execution_errors": task["execution_errors"],
        },
    }
    return "\n".join(
        [
            "# Independent Online-Mind2Web trajectory judgment",
            "",
            "You are an independent strict judge. Inspect this one trajectory only.",
            "The attached screenshots are ordered chronologically and correspond exactly to:",
            json.dumps(screenshot_names, ensure_ascii=False),
            "",
            "Use the screenshots and action log as evidence. The agent's final answer and",
            "`agent_finish` status are untrusted claims, not proof of success.",
            "",
            "## Success rule",
            "",
            "- Return success=1 only when the visual/action evidence proves every material",
            "  requirement in the task instruction was completed.",
            "- For cart/form/state-changing tasks, require evidence of the resulting state;",
            "  reaching a relevant page or merely attempting the last action is insufficient.",
            "- For comparison or information tasks, require that the requested information",
            "  was actually reached and visible/readable enough to answer the instruction.",
            "- If evidence is ambiguous, incomplete, contradicted, or ends at max steps, use 0.",
            "- Do not award success just because an action has action_status=SUCCESS; that only",
            "  means browser execution returned, not that the benchmark goal was achieved.",
            "",
            "## Technical-failure rule",
            "",
            "- environment_technical_failure=true only for objective environment failures",
            "  such as DNS/network/browser failure, persistent server error, inaccessible",
            "  page, or an external blocking challenge that prevents evaluation.",
            "- model_technical_failure=true only for model/pipeline failures such as empty or",
            "  malformed model output, generation/API failure, action parse failure, or an",
            "  execution_error attributable to the agent pipeline.",
            "- Ordinary bad planning, wrong clicks, repetition, premature finish, and max_steps",
            "  are task failures but are NOT model technical failures.",
            "- A successful task may still have a transient technical failure only if the",
            "  trajectory clearly recovered and ultimately proves completion.",
            "",
            "## Evidence payload",
            "",
            json.dumps(evidence, ensure_ascii=False, indent=2),
            "",
            "## Output contract",
            "",
            "Return exactly one JSON object and nothing else:",
            "{",
            f'  "task_id": {json.dumps(task_id)},',
            '  "success": 0,',
            '  "reason": "one concise evidence-based sentence",',
            '  "key_evidence_steps": [0],',
            '  "key_evidence_screenshots": ["0000.png"],',
            '  "environment_technical_failure": false,',
            '  "model_technical_failure": false,',
            '  "technical_failure_type": "none",',
            '  "technical_failure_reason": ""',
            "}",
            "",
            "The success value must be integer 0 or 1. Evidence steps/screenshots must exist",
            "in the supplied trajectory. technical_failure_type must be one of none,",
            "environment, model, both and must agree with the two booleans.",
            "",
        ]
    )


def extract_agent_messages(events_path: Path) -> list[str]:
    messages: list[str] = []
    if not events_path.is_file():
        return messages
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ):
            messages.append(item["text"])
    return messages


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = candidate.find("{")
        if start < 0:
            raise
        value, _ = decoder.raw_decode(candidate[start:])
    if not isinstance(value, dict):
        raise ValueError("judge output is not a JSON object")
    return value


def validate_judgment(
    raw: dict[str, Any],
    task_id: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    if str(raw.get("task_id")) != task_id:
        raise ValueError(f"wrong task_id {raw.get('task_id')!r}")
    success = raw.get("success")
    if isinstance(success, bool):
        success = int(success)
    if success not in (0, 1):
        raise ValueError(f"success must be 0 or 1, got {success!r}")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    env_failure = raw.get("environment_technical_failure")
    model_failure = raw.get("model_technical_failure")
    if not isinstance(env_failure, bool) or not isinstance(model_failure, bool):
        raise ValueError("technical failure fields must be booleans")
    expected_type = (
        "both"
        if env_failure and model_failure
        else "environment"
        if env_failure
        else "model"
        if model_failure
        else "none"
    )
    failure_type = str(raw.get("technical_failure_type", "")).lower()
    if failure_type not in VALID_TECHNICAL_TYPES or failure_type != expected_type:
        raise ValueError(
            f"technical_failure_type={failure_type!r}, expected {expected_type!r}"
        )
    failure_reason = raw.get("technical_failure_reason", "")
    if not isinstance(failure_reason, str):
        raise ValueError("technical_failure_reason must be a string")
    if expected_type != "none" and not failure_reason.strip():
        raise ValueError("technical failure requires a non-empty technical_failure_reason")
    steps = raw.get("key_evidence_steps")
    screenshots = raw.get("key_evidence_screenshots")
    if not isinstance(steps, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in steps
    ):
        raise ValueError("key_evidence_steps must be a list of integers")
    if not isinstance(screenshots, list) or not all(
        isinstance(value, str) for value in screenshots
    ):
        raise ValueError("key_evidence_screenshots must be a list of strings")
    valid_steps = {
        int(row["step"])
        for row in task["result"]["action_history"]
        if isinstance(row, dict) and isinstance(row.get("step"), int)
    }
    valid_screenshots = {path.name for path in task["screenshots"]}
    if not set(steps).issubset(valid_steps):
        raise ValueError(f"unknown evidence steps: {sorted(set(steps) - valid_steps)}")
    if not set(screenshots).issubset(valid_screenshots):
        raise ValueError(
            f"unknown evidence screenshots: {sorted(set(screenshots) - valid_screenshots)}"
        )
    if not steps or not screenshots:
        raise ValueError("at least one evidence step and screenshot are required")
    return {
        "task_id": task_id,
        "success": int(success),
        "reason": reason.strip(),
        "key_evidence_steps": steps,
        "key_evidence_screenshots": screenshots,
        "environment_technical_failure": env_failure,
        "model_technical_failure": model_failure,
        "technical_failure_type": expected_type,
        "technical_failure_reason": failure_reason.strip(),
    }


def encode_image_part(path: Path, detail: str) -> dict[str, Any]:
    mime = IMAGE_MIME.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    image_url: dict[str, Any] = {"url": f"data:{mime};base64,{encoded}"}
    if detail and detail != "auto":
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def read_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"empty API key file: {path}")
    return key


def isolated_worker_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.environ.get("HOME"):
        env["HOME"] = os.environ["HOME"]
    if os.environ.get("SSL_CERT_FILE"):
        env["SSL_CERT_FILE"] = os.environ["SSL_CERT_FILE"]
    if os.environ.get("REQUESTS_CA_BUNDLE"):
        env["REQUESTS_CA_BUNDLE"] = os.environ["REQUESTS_CA_BUNDLE"]
    return env


def run_openai_compatible_worker(args: argparse.Namespace) -> dict[str, Any]:
    from openai import OpenAI
    import httpx

    prompt_path = args.prompt_path
    output_path = args.output_path
    events_path = args.events_path
    stderr_path = args.stderr_path
    images = list(args.image or [])
    if prompt_path is None or output_path is None or events_path is None:
        raise ValueError("openai_compatible-worker requires --prompt-path --output-path --events-path")
    if not images:
        raise ValueError("openai_compatible-worker requires at least one --image")
    prompt = prompt_path.read_text(encoding="utf-8")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in images:
        content.append(encode_image_part(path, args.image_detail))
    payload = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    timeout = None if args.timeout_seconds == 0 else args.timeout_seconds
    http_timeout = httpx.Timeout(timeout or 1800.0, connect=30.0)
    http_client = httpx.Client(
        proxy=args.proxy_url or None,
        timeout=http_timeout,
        trust_env=False,
    )
    client = OpenAI(
        base_url=args.base_url,
        api_key=read_api_key(args.api_key_file),
        http_client=http_client,
        max_retries=0,
    )
    last_error = ""
    returncode = 1
    max_tries = max(1, args.transport_retries)
    try:
        for try_index in range(1, max_tries + 1):
            started_try = time.monotonic()
            try:
                response = client.chat.completions.create(**payload)
                output = (response.choices[0].message.content or "").strip()
                usage = response.usage.model_dump() if response.usage is not None else {}
                events_path.write_text(
                    json.dumps(
                        {
                            "type": "openai_compatible.completed",
                            "try": try_index,
                            "model": args.model,
                            "reasoning_effort": args.reasoning_effort,
                            "image_count": len(images),
                            "image_detail": args.image_detail,
                            "elapsed_seconds": round(time.monotonic() - started_try, 3),
                            "usage": usage,
                            "finish_reason": response.choices[0].finish_reason,
                            "context_isolation": "one ephemeral worker process, one user message, this trajectory only",
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                output_path.write_text(output + ("\n" if output else ""), encoding="utf-8")
                returncode = 0
                last_error = ""
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "type": "openai_compatible.error",
                                "try": try_index,
                                "error": last_error,
                                "elapsed_seconds": round(time.monotonic() - started_try, 3),
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
                if try_index < max_tries:
                    time.sleep(min(30, 2 ** try_index))
        if returncode != 0:
            print(last_error, file=sys.stderr)
    finally:
        http_client.close()
    if returncode != 0:
        raise SystemExit(1)
    return {"returncode": 0, "image_count": len(images)}


def run_attempt_openai_compatible(
    args: argparse.Namespace,
    task_id: str,
    task: dict[str, Any],
    attempt_number: int,
    attempt_dir: Path,
    prompt_path: Path,
    output_path: Path,
    events_path: Path,
    stderr_path: Path,
) -> tuple[int, bool, str]:
    isolated_root = Path(
        os.environ.get("GPT_JUDGE_ISOLATED_ROOT", "/tmp/gpt56_sol_m2w_judge")
    )
    isolated_root.mkdir(parents=True, exist_ok=True)
    isolated_cwd = Path(tempfile.mkdtemp(prefix=f"{task_id}_", dir=isolated_root))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "openai_compatible-worker",
        "--backend",
        "openai_compatible",
        "--prompt-path",
        str(prompt_path),
        "--output-path",
        str(output_path),
        "--events-path",
        str(events_path),
        "--stderr-path",
        str(stderr_path),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--base-url",
        args.base_url,
        "--api-key-file",
        str(args.api_key_file),
        "--proxy-url",
        args.proxy_url,
        "--image-detail",
        args.image_detail,
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--transport-retries",
        str(args.transport_retries),
    ]
    for path in task["screenshots"]:
        command.extend(["--image", str(path)])
    timed_out = False
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=isolated_cwd,
            env=isolated_worker_env(),
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            wait_timeout = None if args.timeout_seconds == 0 else args.timeout_seconds
            returncode = process.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    last_error = ""
    if returncode != 0 and stderr_path.is_file():
        last_error = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
    return returncode, timed_out, last_error


def event_error_text(events_path: Path) -> str:
    messages: list[str] = []
    if not events_path.is_file():
        return ""
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") in {"error", "turn.failed"}:
            messages.append(
                json.dumps(
                    event.get("error", event.get("message", "")),
                    ensure_ascii=False,
                    default=str,
                )
            )
    return "\n".join(messages)


def run_attempt(
    args: argparse.Namespace,
    task_id: str,
    task: dict[str, Any],
    attempt_number: int,
) -> dict[str, Any]:
    attempt_dir = args.output_dir / "attempts" / task_id / f"attempt_{attempt_number:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    prompt_path = attempt_dir / "prompt.md"
    output_path = attempt_dir / "last_message.txt"
    events_path = attempt_dir / "events.jsonl"
    stderr_path = attempt_dir / "stderr.log"
    prompt_path.write_text(build_prompt(task_id, task), encoding="utf-8")
    started = time.monotonic()
    timed_out = False
    if args.backend == "openai_compatible":
        returncode, timed_out, transport_error = run_attempt_openai_compatible(
            args,
            task_id,
            task,
            attempt_number,
            attempt_dir,
            prompt_path,
            output_path,
            events_path,
            stderr_path,
        )
        if transport_error and not stderr_path.is_file():
            stderr_path.write_text(transport_error + "\n", encoding="utf-8")
    else:
        isolated_root = Path(
            os.environ.get("GPT_JUDGE_ISOLATED_ROOT", "/tmp/gpt56_sol_m2w_judge")
        )
        isolated_root.mkdir(parents=True, exist_ok=True)
        isolated_cwd = Path(tempfile.mkdtemp(prefix=f"{task_id}_", dir=isolated_root))
        command = [
            args.codex_bin,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--model",
            args.model,
            "--config",
            f'model_reasoning_effort="{args.reasoning_effort}"',
            "--sandbox",
            "read-only",
            "--cd",
            str(isolated_cwd),
            "--image",
            *(str(path) for path in task["screenshots"]),
            "--json",
            "--output-last-message",
            str(output_path),
            "-",
        ]
        child_env = os.environ.copy()
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            child_env[key] = args.proxy_url
        with (
            prompt_path.open("r", encoding="utf-8") as prompt,
            events_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                command,
                stdin=prompt,
                stdout=stdout,
                stderr=stderr,
                text=True,
                cwd=isolated_cwd,
                env=child_env,
                start_new_session=True,
            )
            try:
                wait_timeout = None if args.timeout_seconds == 0 else args.timeout_seconds
                returncode = process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
    output = (
        output_path.read_text(encoding="utf-8", errors="replace")
        if output_path.is_file()
        else ""
    )
    if not output.strip():
        messages = extract_agent_messages(events_path)
        if messages:
            output = messages[-1]
            output_path.write_text(output, encoding="utf-8")
    parse_error = ""
    judgment: dict[str, Any] | None = None
    if returncode == 0 and output.strip():
        try:
            judgment = validate_judgment(
                extract_json_object(output),
                task_id,
                task,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = str(exc)
    else:
        parse_error = "empty output" if not output.strip() else f"returncode={returncode}"
    status = {
        "task_id": task_id,
        "attempt": attempt_number,
        "completed_at": now(),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": len(task["screenshots"]),
        "output_nonempty": bool(output.strip()),
        "parse_valid": judgment is not None,
        "parse_error": parse_error,
        "event_errors": event_error_text(events_path),
        "prompt_path": rel_to_root(prompt_path),
        "output_path": rel_to_root(output_path),
        "events_path": rel_to_root(events_path),
        "stderr_path": rel_to_root(stderr_path),
    }
    write_json(attempt_dir / "status.json", status)
    if judgment is not None:
        judgment.update(
            {
                "difficulty": str(task["metadata"].get("difficulty", "unknown")).lower(),
                "website": task["metadata"].get("website"),
                "trajectory_status": task["metadata"].get("status"),
                "step_count": task["metadata"].get("step_count"),
                "judge_model": args.model,
                "judge_backend": args.backend,
                "judge_reasoning_effort": args.reasoning_effort,
                "judge_attempt": attempt_number,
                "judge_completed_at": now(),
                "source_trajectory_dir": rel_to_root(task["task_dir"]),
            }
        )
        write_json(args.output_dir / "judgments" / f"{task_id}.json", judgment)
    return status


def attempts_so_far(output_dir: Path, task_id: str) -> int:
    return len(list((output_dir / "attempts" / task_id).glob("attempt_*")))


def run_one(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    judgment_path = args.output_dir / "judgments" / f"{task_id}.json"
    if judgment_path.is_file():
        return {"task_id": task_id, "skipped": True, "parse_valid": True}
    task = load_task(args.trajectories_dir, task_id)
    statuses: list[dict[str, Any]] = []
    attempt_number = attempts_so_far(args.output_dir, task_id) + 1
    per_invoke = args.attempts_per_invocation
    while args.max_attempts == 0 or attempt_number <= args.max_attempts:
        status = run_attempt(args, task_id, task, attempt_number)
        statuses.append(status)
        if status["parse_valid"]:
            break
        attempt_number += 1
        if per_invoke is not None and len(statuses) >= per_invoke:
            break
    return {
        "task_id": task_id,
        "skipped": False,
        "parse_valid": bool(statuses and statuses[-1]["parse_valid"]),
        "attempts_this_run": len(statuses),
        "statuses": statuses,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    mask, task_ids = read_mask(args.mask_path)
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    for position, task_id in enumerate(task_ids):
        try:
            task = load_task(args.trajectories_dir, task_id)
        except (FileNotFoundError, RuntimeError):
            missing.append(task_id)
            continue
        rows.append(
            {
                "position": position,
                "task_id": task_id,
                "difficulty": task["metadata"].get("difficulty"),
                "trajectory_status": task["metadata"].get("status"),
                "step_count": task["metadata"].get("step_count"),
                "image_count": len(task["screenshots"]),
            }
        )
    if (missing or len(rows) != len(task_ids)) and not args.available_only:
        raise RuntimeError(f"trajectory preflight missing/invalid IDs: {missing}")
    if not rows:
        raise RuntimeError("trajectory preflight found no complete trajectories")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "created_at": now(),
            "task_count": len(rows),
            "task_ids": [str(row["task_id"]) for row in rows],
            "frozen_mask_task_count": len(task_ids),
            "available_only": args.available_only,
            "missing_task_ids": missing,
            "tasks": rows,
        },
    )
    config = {
        "created_at": now(),
        "backend": args.backend,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "max_attempts": args.max_attempts,
        "timeout_seconds": args.timeout_seconds,
        "trajectories_dir": str(args.trajectories_dir),
        "mask_path": str(args.mask_path),
        "mask_sha256": sha256(args.mask_path),
        "mask_policy_version": mask.get("policy_version"),
        "strict_task_count": len(task_ids),
        "available_trajectory_count": len(rows),
        "available_only": args.available_only,
        "context_isolation": (
            "one ephemeral worker process per trajectory; one independent user message; historical judge prompt; no shared chat history"
            if args.backend == "openai_compatible"
            else "one read-only ephemeral codex process per trajectory"
        ),
        "evidence": "all chronological trajectory screenshots plus result/action history and runtime diagnostics",
        "official_webjudge_output_modified": False,
        "proxy_configured": bool(args.proxy_url),
        "base_url": args.base_url if args.backend == "openai_compatible" else None,
        "image_detail": args.image_detail if args.backend == "openai_compatible" else None,
        "max_completion_tokens": (
            args.max_completion_tokens if args.backend == "openai_compatible" else None
        ),
    }
    write_json(args.output_dir / "run_config.json", config)
    return config


def selected_task_ids(args: argparse.Namespace) -> list[str]:
    _, task_ids = read_mask(args.mask_path)
    if args.task_id:
        if args.task_id not in task_ids:
            raise RuntimeError(f"--task-id {args.task_id!r} is outside frozen mask")
        return [args.task_id]
    pending = [
        task_id
        for task_id in task_ids
        if not (args.output_dir / "judgments" / f"{task_id}.json").is_file()
    ]
    if args.available_only:
        available: list[str] = []
        for task_id in pending:
            try:
                load_task(args.trajectories_dir, task_id)
            except (FileNotFoundError, RuntimeError):
                continue
            available.append(task_id)
        pending = available
    if args.max_attempts == 0:
        retryable = list(pending)
    else:
        retryable = [
            task_id
            for task_id in pending
            if attempts_so_far(args.output_dir, task_id) < args.max_attempts
        ]
    retryable.sort(key=lambda task_id: attempts_so_far(args.output_dir, task_id))
    pending = retryable
    if args.limit is not None:
        pending = pending[: args.limit]
    return pending


def run(args: argparse.Namespace) -> dict[str, Any]:
    _, task_ids = read_mask(args.mask_path)
    manifest_path = args.output_dir / "manifest.json"
    manifest_count = None
    if manifest_path.is_file():
        try:
            manifest_count = int(
                json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "task_count", 0
                )
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            manifest_count = None
    if args.available_only or manifest_count != len(task_ids):
        prepare(args)
    selected = selected_task_ids(args)
    stage_name = args.stage or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stage_path = args.output_dir / "stages" / f"{stage_name}.json"
    if stage_path.exists():
        raise FileExistsError(stage_path)
    stage = {
        "stage": stage_name,
        "started_at": now(),
        "concurrency": args.concurrency,
        "selected_count": len(selected),
        "task_ids": selected,
    }
    write_json(args.output_dir / "stages" / f"{stage_name}_started.json", stage)
    results: list[dict[str, Any]] = []
    if selected:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.concurrency, len(selected))
        ) as executor:
            futures = {executor.submit(run_one, args, task_id): task_id for task_id in selected}
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # keep the other isolated calls running
                    results.append(
                        {
                            "task_id": task_id,
                            "parse_valid": False,
                            "runner_exception": f"{type(exc).__name__}: {exc}",
                        }
                    )
    stage.update(
        {
            "completed_at": now(),
            "valid_results": sum(bool(row.get("parse_valid")) for row in results),
            "invalid_results": sum(not bool(row.get("parse_valid")) for row in results),
            "attempts_this_run": sum(int(row.get("attempts_this_run", 0)) for row in results),
            "results": sorted(results, key=lambda row: str(row["task_id"])),
        }
    )
    write_json(stage_path, stage)
    return {"stage": stage, "summary": summarize(args)}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    _, task_ids = read_mask(args.mask_path)
    judgments: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    for task_id in task_ids:
        path = args.output_dir / "judgments" / f"{task_id}.json"
        if not path.is_file():
            missing.append(task_id)
            continue
        try:
            task = load_task(args.trajectories_dir, task_id)
            judgment = validate_judgment(
                json.loads(path.read_text(encoding="utf-8")),
                task_id,
                task,
            )
            stored = json.loads(path.read_text(encoding="utf-8"))
            judgment.update(
                {
                    "difficulty": stored.get(
                        "difficulty", task["metadata"].get("difficulty", "unknown")
                    ),
                    "judge_attempt": stored.get("judge_attempt"),
                }
            )
            judgments.append(judgment)
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            invalid.append(f"{task_id}: {exc}")
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        by_difficulty[str(row["difficulty"]).lower()].append(row)
    attempts = list((args.output_dir / "attempts").glob("*/attempt_*/status.json"))
    attempt_statuses = [
        json.loads(path.read_text(encoding="utf-8")) for path in attempts
    ]
    success = sum(row["success"] for row in judgments)
    summary = {
        "updated_at": now(),
        "frozen_mask_count": len(task_ids),
        "completed_judgments": len(judgments),
        "missing_task_ids": missing,
        "invalid_judgments": invalid,
        "success_count": success,
        "success_rate": success / len(judgments) if judgments else None,
        "successful_task_ids": [
            task_id
            for task_id in task_ids
            if any(
                row["task_id"] == task_id and row["success"] == 1
                for row in judgments
            )
        ],
        "failed_task_ids": [
            task_id
            for task_id in task_ids
            if any(
                row["task_id"] == task_id and row["success"] == 0
                for row in judgments
            )
        ],
        "by_difficulty": {
            difficulty: {
                "count": len(rows),
                "success": sum(row["success"] for row in rows),
                "success_rate": (
                    sum(row["success"] for row in rows) / len(rows) if rows else None
                ),
            }
            for difficulty, rows in sorted(by_difficulty.items())
        },
        "technical_failures": dict(
            sorted(Counter(row["technical_failure_type"] for row in judgments).items())
        ),
        "environment_technical_failure_count": sum(
            row["environment_technical_failure"] for row in judgments
        ),
        "model_technical_failure_count": sum(
            row["model_technical_failure"] for row in judgments
        ),
        "judge_call_attempts": len(attempt_statuses),
        "judge_retries": sum(
            max(0, attempts_so_far(args.output_dir, task_id) - 1)
            for task_id in task_ids
        ),
        "judge_invalid_attempts": sum(
            not bool(status.get("parse_valid")) for status in attempt_statuses
        ),
        "judge_timeout_attempts": sum(
            bool(status.get("timed_out")) for status in attempt_statuses
        ),
        "judge_nonzero_returncode_attempts": sum(
            status.get("returncode") != 0 for status in attempt_statuses
        ),
        "strict_complete": len(judgments) == len(task_ids) and not missing and not invalid,
        "task_ids_unique": len({row["task_id"] for row in judgments})
        == len(judgments),
    }
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        indexed = {
            row["task_id"]: json.loads(
                (args.output_dir / "judgments" / f"{row['task_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            for row in judgments
        }
        for task_id in task_ids:
            if task_id in indexed:
                handle.write(json.dumps(indexed[task_id], ensure_ascii=False) + "\n")
    if summary["strict_complete"]:
        available_official_labels = sum(
            (args.official_webjudge_dir / f"{task_id}.json").is_file()
            for task_id in task_ids
        )
        if available_official_labels == len(task_ids):
            summary["official_webjudge_comparison"] = compare_official_webjudge(
                args,
                task_ids,
            )
            summary["official_webjudge_comparison_pending"] = False
        else:
            summary["official_webjudge_comparison_pending"] = True
            summary["official_webjudge_labels_available"] = available_official_labels
        write_json(args.output_dir / "summary.json", summary)
    return summary


def compare_official_webjudge(
    args: argparse.Namespace,
    task_ids: list[str],
) -> dict[str, Any]:
    """Read, but never modify, the official WebJudge labels and compare per task."""
    cell_names = {
        (1, 1): "both_success",
        (1, 0): "gpt56_success_webjudge_failure",
        (0, 1): "gpt56_failure_webjudge_success",
        (0, 0): "both_failure",
    }
    rows: list[dict[str, Any]] = []
    cells: dict[str, list[str]] = {name: [] for name in cell_names.values()}
    difficulty_cells: dict[str, Counter[str]] = defaultdict(Counter)
    for task_id in task_ids:
        judgment_path = args.output_dir / "judgments" / f"{task_id}.json"
        official_path = args.official_webjudge_dir / f"{task_id}.json"
        if not official_path.is_file():
            raise FileNotFoundError(official_path)
        judgment = json.loads(judgment_path.read_text(encoding="utf-8"))
        official = json.loads(official_path.read_text(encoding="utf-8"))
        official_task_id = str(official.get("task_id"))
        if official_task_id != task_id:
            raise RuntimeError(
                f"{official_path}: task_id={official_task_id!r}, expected {task_id!r}"
            )
        gpt_label = int(judgment["success"])
        webjudge_label = official.get("predicted_label")
        if isinstance(webjudge_label, bool):
            webjudge_label = int(webjudge_label)
        if webjudge_label not in (0, 1):
            raise RuntimeError(
                f"{official_path}: invalid predicted_label={webjudge_label!r}"
            )
        cell = cell_names[(gpt_label, int(webjudge_label))]
        difficulty = str(judgment["difficulty"]).lower()
        cells[cell].append(task_id)
        difficulty_cells[difficulty][cell] += 1
        rows.append(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "gpt56_sol_label": gpt_label,
                "official_webjudge7b_label": int(webjudge_label),
                "agreement": gpt_label == int(webjudge_label),
                "confusion_cell": cell,
                "gpt56_sol_reason": judgment["reason"],
                "gpt56_sol_key_evidence_steps": judgment[
                    "key_evidence_steps"
                ],
                "gpt56_sol_key_evidence_screenshots": judgment[
                    "key_evidence_screenshots"
                ],
                "gpt56_sol_technical_failure_type": judgment[
                    "technical_failure_type"
                ],
            }
        )
    counts = {name: len(ids) for name, ids in cells.items()}
    n = len(rows)
    gpt_success = sum(row["gpt56_sol_label"] for row in rows)
    webjudge_success = sum(row["official_webjudge7b_label"] for row in rows)
    agreement_count = counts["both_success"] + counts["both_failure"]
    expected_agreement = (
        gpt_success * webjudge_success
        + (n - gpt_success) * (n - webjudge_success)
    ) / (n * n)
    observed_agreement = agreement_count / n
    kappa = (
        (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        if expected_agreement != 1.0
        else 1.0
    )
    with (
        args.output_dir / "comparison_with_official_webjudge7b.jsonl"
    ).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    comparison = {
        "generated_at": now(),
        "task_count": n,
        "official_webjudge_dir": str(args.official_webjudge_dir),
        "official_webjudge_files_in_directory": len(
            list(args.official_webjudge_dir.glob("*.json"))
        ),
        "official_webjudge_labels_read": n,
        "official_webjudge_success": webjudge_success,
        "gpt56_sol_success": gpt_success,
        "confusion_matrix": counts,
        "confusion_task_ids": cells,
        "by_difficulty": {
            difficulty: {
                **{
                    name: int(counter.get(name, 0))
                    for name in cell_names.values()
                },
                "count": sum(counter.values()),
            }
            for difficulty, counter in sorted(difficulty_cells.items())
        },
        "agreement_count": agreement_count,
        "agreement_rate": observed_agreement,
        "cohen_kappa": kappa,
        "official_webjudge_modified": False,
    }
    write_json(
        args.output_dir / "comparison_with_official_webjudge7b.json",
        comparison,
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "run", "summarize", "openai_compatible-worker"),
    )
    parser.add_argument("--trajectories-dir", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-webjudge-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_WEBJUDGE,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        choices=("codex", "openai_compatible"),
        default="codex",
        help="codex keeps the original CLI judge; openai_compatible uses OpenAI-compatible HTTP.",
    )
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING)
    parser.add_argument("--base-url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--api-key-file", type=Path, default=Path(DEFAULT_OPENAI_API_KEY_FILE))
    parser.add_argument(
        "--image-detail",
        default="high",
        choices=("auto", "low", "high"),
        help="OpenAI-compatible/OpenAI image_url.detail. Codex backend ignores this.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--events-path", type=Path)
    parser.add_argument("--stderr-path", type=Path)
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--codex-bin", default=str(Path.home() / ".local/bin/codex"))
    parser.add_argument(
        "--proxy-url",
        default=os.environ.get("HTTPS_PROXY") or DEFAULT_PROXY,
    )
    parser.add_argument("--concurrency", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--attempts-per-invocation",
        type=int,
        help="Cap new attempts in a single run() call so a serial scheduler can rotate tasks.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id")
    parser.add_argument("--stage")
    parser.add_argument(
        "--available-only",
        action="store_true",
        help=(
            "Judge only trajectories that are complete at launch time. The frozen "
            "mask and final strict summary remain unchanged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.trajectories_dir = args.trajectories_dir.resolve()
    args.mask_path = args.mask_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.official_webjudge_dir = args.official_webjudge_dir.resolve()
    if args.concurrency < 1 or args.concurrency > 120:
        raise ValueError("--concurrency must be in [1, 120]")
    if args.backend == "openai_compatible":
        args.api_key_file = args.api_key_file.expanduser().resolve()
        if not args.api_key_file.is_file():
            raise ValueError(f"--api-key-file does not exist: {args.api_key_file}")
        if args.max_completion_tokens < 16:
            raise ValueError("--max-completion-tokens must be >= 16")
        if args.transport_retries < 1 or args.transport_retries > 8:
            raise ValueError("--transport-retries must be in [1, 8]")
        if args.model == DEFAULT_MODEL:
            args.model = DEFAULT_OPENAI_MODEL
    if args.max_attempts != 0 and (args.max_attempts < 1 or args.max_attempts > 40):
        raise ValueError("--max-attempts must be 0 (unlimited) or in [1, 40]")
    if args.attempts_per_invocation is not None and (
        args.attempts_per_invocation < 1
        or (args.max_attempts != 0 and args.attempts_per_invocation > args.max_attempts)
    ):
        raise ValueError("--attempts-per-invocation must be in [1, max-attempts] (or any >=1 if unlimited)")
    if args.timeout_seconds < 0:
        raise ValueError("--timeout-seconds must be non-negative (0 disables the watchdog)")
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run(args)
    elif args.command == "openai_compatible-worker":
        result = run_openai_compatible_worker(args)
    else:
        result = summarize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
