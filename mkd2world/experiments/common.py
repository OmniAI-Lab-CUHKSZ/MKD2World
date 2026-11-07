from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mkd2world.config import RunConfig, load_run_config, merge_run_config


def load_config_with_overrides(config_path: Path | None, overrides: dict[str, Any]) -> RunConfig:
    return merge_run_config(load_run_config(config_path), overrides)


def optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def csv_values(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def missing_path(path: str, label: str, *, must_be_file: bool = False, must_be_dir: bool = False) -> str | None:
    if not path:
        return label
    candidate = Path(path)
    if must_be_file and not candidate.is_file():
        return f"{label}:{path}"
    if must_be_dir and not candidate.is_dir():
        return f"{label}:{path}"
    return None


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def config_payload(config: RunConfig) -> dict[str, Any]:
    return asdict(config)
