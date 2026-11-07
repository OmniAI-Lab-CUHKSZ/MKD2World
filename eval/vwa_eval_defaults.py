from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = Path(__file__).resolve().parent / "vwa_eval_defaults.json"
QWEN_METHOD_CONFIGS = {
    "bare_qwen": PROJECT_ROOT / "eval/examples/pipeline_vwa_bare_qwen.json",
    "mkd2world_p21": PROJECT_ROOT / "eval/examples/pipeline_vwa_mkd2world_p21.json",
}


def load_defaults() -> dict[str, Any]:
    return json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))


def apply_to_experiment_config(config: Any) -> Any:
    defaults = load_defaults()
    ga = defaults["gui_agent"]
    rt = defaults["runtime"]
    agent = config.pipeline.agent
    options = dict(agent.options or {})
    options["enable_thinking"] = ga["enable_thinking"]
    options["reasoning_effort"] = ga["reasoning_effort"]
    options["thinking_token_budget"] = ga["thinking_token_budget"]
    options["max_context_tokens"] = ga["max_context_tokens"]
    agent = replace(
        agent,
        temperature=float(ga["temperature"]),
        max_tokens=int(ga["max_tokens"]),
        timeout_seconds=float(ga["timeout_seconds"]),
        retry_attempts=int(ga["retry_attempts"]),
        options=options,
    )
    env = config.pipeline.environment
    env_options = dict(env.options or {})
    env_options["width"] = int(rt["viewport_width"])
    env_options["height"] = int(rt["viewport_height"])
    env_options["sleep_after_execution"] = float(rt["sleep_after_execution"])
    environment = replace(env, options=env_options)
    wm = config.pipeline.local_world_model
    if getattr(wm, "enabled", False):
        wd = defaults["world_model"]
        wopt = dict(wm.options or {})
        wopt["enable_thinking"] = wd["enable_thinking"]
        wopt["max_context_tokens"] = wd["max_context_tokens"]
        wm = replace(
            wm,
            temperature=float(wd["temperature"]),
            max_tokens=int(wd["max_tokens"]),
            timeout_seconds=float(wd["timeout_seconds"]),
            retry_attempts=int(wd["retry_attempts"]),
            options=wopt,
        )
    pipeline = replace(
        config.pipeline,
        agent=agent,
        environment=environment,
        local_world_model=wm,
    )
    return replace(config, pipeline=pipeline)


def validate_pipeline_json(path: Path) -> list[str]:
    defaults = load_defaults()
    ga = defaults["gui_agent"]
    rt = defaults["runtime"]
    data = json.loads(path.read_text(encoding="utf-8"))
    agent = data["pipeline"]["agent"]
    options = agent.get("options") or {}
    env = data["pipeline"]["environment"]["options"]
    errors: list[str] = []

    def check(label: str, got: Any, expected: Any) -> None:
        if got != expected:
            errors.append(f"{label}: got {got!r}, expected {expected!r}")

    check("agent.temperature", float(agent.get("temperature")), float(ga["temperature"]))
    check("agent.max_tokens", int(agent.get("max_tokens")), int(ga["max_tokens"]))
    check("agent.options.enable_thinking", bool(options.get("enable_thinking")), True)
    check("agent.options.reasoning_effort", options.get("reasoning_effort"), ga["reasoning_effort"])
    check(
        "agent.options.thinking_token_budget",
        int(options.get("thinking_token_budget") or 0),
        int(ga["thinking_token_budget"]),
    )
    check(
        "agent.options.max_context_tokens",
        int(options.get("max_context_tokens") or 0),
        int(ga["max_context_tokens"]),
    )
    check("environment.width", int(env.get("width")), int(rt["viewport_width"]))
    check("environment.height", int(env.get("height")), int(rt["viewport_height"]))
    wm = data["pipeline"].get("local_world_model") or {}
    if wm.get("enabled"):
        wd = defaults["world_model"]
        wopt = wm.get("options") or {}
        check("wm.temperature", float(wm.get("temperature")), float(wd["temperature"]))
        check("wm.max_tokens", int(wm.get("max_tokens")), int(wd["max_tokens"]))
        check("wm.options.enable_thinking", bool(wopt.get("enable_thinking")), False)
        check(
            "wm.options.max_context_tokens",
            int(wopt.get("max_context_tokens") or 0),
            int(wd["max_context_tokens"]),
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--validate-method")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    defaults = load_defaults()
    if args.print:
        print(json.dumps(defaults, ensure_ascii=False, indent=2))
        return
    path = args.config
    if args.validate_method:
        if args.validate_method not in QWEN_METHOD_CONFIGS:
            print(
                f"skip defaults validation for method={args.validate_method} "
                f"(not a Qwen GUI-agent method)",
                file=sys.stderr,
            )
            return
        path = QWEN_METHOD_CONFIGS[args.validate_method]
    if path is None:
        raise SystemExit("pass --print, --validate-method, or --config")
    errors = validate_pipeline_json(path)
    if errors:
        print(f"VWA eval defaults mismatch in {path}:", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(2)
    print(f"ok {path}")


if __name__ == "__main__":
    main()
