from __future__ import annotations

from typing import Sequence


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


def split_local_prediction(raw_output: str) -> tuple[str, str]:
    marker = "PREDICTED_MARKDOWN:"
    if marker not in raw_output:
        return "", raw_output.strip()
    analysis = raw_output.split(marker, 1)[0].replace("ACTION_ANALYSIS:", "").strip()
    predicted_markdown = raw_output.split(marker, 1)[1].strip()
    return analysis, predicted_markdown
