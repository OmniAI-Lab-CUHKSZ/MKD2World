from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


COORD_RE = re.compile(
    r"<\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*>"
)
ACTION_ANALYSIS_MARKER_RE = re.compile(r"(?im)^\s*ACTION_ANALYSIS\s*:\s*$")
PREDICTED_MARKDOWN_MARKER_RE = re.compile(r"(?im)^\s*PREDICTED_MARKDOWN\s*:\s*$")
ELEMENT_RE = re.compile(
    r"(?P<prefix>(?:\[[ xXvV>]\]\s*)?\[[^\]]+\]\([^)]+\)|\[[^\]]+\](?:\([^)]+\))?)"
    r"\s*<\s*(?P<cx>-?\d+(?:\.\d+)?)\s*,\s*(?P<cy>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<w>-?\d+(?:\.\d+)?)\s*,\s*(?P<h>-?\d+(?:\.\d+)?)\s*>"
)
STATE_PREFIX_RE = re.compile(r"^\[(?P<state>[ xXvV>])\](?:\s+|(?=\())")
TYPE_SUFFIX_RE = re.compile(r"\]\((?P<type>[^)]+)\)\s*$")
ENVIRONMENT_LINE_RE = re.compile(r"^\[Environment:\s*[^\]]+\]$")
LAYER_LINE_RE = re.compile(r"^#{1,6}\s+\[Layer[^\]]*\]$")


@dataclass(frozen=True)
class MdElement:
    content: str
    element_type: str
    cx: float
    cy: float
    w: float
    h: float
    line: str


def clamp_grid(value: float, grid_max: int = 999) -> int:
    if value != value:
        return 0
    return max(0, min(grid_max, int(round(value * grid_max))))


def clamp_grid_coordinate(value: float, grid_max: int = 999) -> int:
    if value != value:
        return 0
    return max(0, min(grid_max, int(round(value))))


def convert_normalized_coords_to_grid(markdown: str, grid_max: int = 999) -> str:
    def repl(match: re.Match[str]) -> str:
        values = [float(match.group(i)) for i in range(1, 5)]
        if all(0.0 <= value <= 1.0 for value in values):
            converted = [clamp_grid(value, grid_max) for value in values]
        else:
            converted = [clamp_grid_coordinate(value, grid_max) for value in values]
        return "<{},{},{},{}>".format(*converted)

    return COORD_RE.sub(repl, markdown)


def bbox_to_grid_center(bbox: Iterable[float], grid_max: int = 999) -> tuple[int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return clamp_grid((x1 + x2) / 2, grid_max), clamp_grid((y1 + y2) / 2, grid_max)


def build_assistant_output(action_analysis: str, next_markdown: str) -> str:
    return f"{next_markdown.strip()}\n"


def extract_predicted_markdown(text: str) -> str:
    marker = PREDICTED_MARKDOWN_MARKER_RE.search(text)
    if marker is None:
        return text.strip()
    return text[marker.end() :].strip()


def _normalize_content(prefix: str) -> str:
    cleaned = re.sub(r"^\s*[-*]\s*", "", prefix.strip())
    cleaned = STATE_PREFIX_RE.sub("", cleaned).strip()
    match = re.match(r"\[(?P<content>[^\]]*)\](?:\([^)]+\))?", cleaned)
    return match.group("content").strip() if match else cleaned


def _stateful_element_type(prefix: str, base_type: str) -> str:
    state_match = STATE_PREFIX_RE.match(prefix.strip())
    if not state_match:
        return base_type
    state = state_match.group("state").lower()
    if state in {"x", " "}:
        return "toggle"
    if state in {">", "v"}:
        return "select"
    return base_type


def _md_line_body(line: str) -> str:
    body = line.strip()
    while body.startswith("- "):
        body = body[2:].strip()
    return body


def is_parseable_md_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if ENVIRONMENT_LINE_RE.match(stripped) or LAYER_LINE_RE.match(stripped):
        return True
    if stripped.startswith("```"):
        return False
    return bool(ELEMENT_RE.search(_md_line_body(stripped)))


def markdown_line_parse_metrics(markdown: str) -> dict[str, float]:
    lines = [line for line in markdown.splitlines() if line.strip()]
    if not lines:
        return {"md_nonempty_lines": 0.0, "md_parseable_lines": 0.0, "md_unparseable_lines": 0.0, "md_parseable_line_rate": 0.0}
    parseable = sum(1 for line in lines if is_parseable_md_line(line))
    return {
        "md_nonempty_lines": float(len(lines)),
        "md_parseable_lines": float(parseable),
        "md_unparseable_lines": float(len(lines) - parseable),
        "md_parseable_line_rate": parseable / len(lines),
    }


def parse_md_elements(markdown: str) -> list[MdElement]:
    elements = []
    for raw_line in markdown.splitlines():
        line = _md_line_body(raw_line)
        match = ELEMENT_RE.search(line)
        if not match:
            continue
        prefix = match.group("prefix")
        type_match = TYPE_SUFFIX_RE.search(prefix)
        base_type = type_match.group("type").strip().lower() if type_match else "block"
        elements.append(
            MdElement(
                content=_normalize_content(prefix),
                element_type=_stateful_element_type(prefix, base_type),
                cx=float(match.group("cx")),
                cy=float(match.group("cy")),
                w=float(match.group("w")),
                h=float(match.group("h")),
                line=line.strip(),
            )
        )
    return elements


def line_f1(pred_markdown: str, target_markdown: str) -> dict[str, float]:
    pred_lines = {line.strip() for line in pred_markdown.splitlines() if line.strip()}
    target_lines = {line.strip() for line in target_markdown.splitlines() if line.strip()}
    overlap = len(pred_lines & target_lines)
    precision = overlap / len(pred_lines) if pred_lines else 0.0
    recall = overlap / len(target_lines) if target_lines else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"line_precision": precision, "line_recall": recall, "line_f1": f1}


def element_count_metrics(pred_markdown: str, target_markdown: str) -> dict[str, float]:
    pred = parse_md_elements(pred_markdown)
    target = parse_md_elements(target_markdown)
    pred_counts = Counter(element.element_type for element in pred)
    target_counts = Counter(element.element_type for element in target)
    keys = set(pred_counts) | set(target_counts)
    if not keys:
        type_score = 1.0
    else:
        scores = []
        for key in keys:
            p_count = pred_counts.get(key, 0)
            t_count = target_counts.get(key, 0)
            scores.append(min(p_count, t_count) / max(p_count, t_count) if max(p_count, t_count) else 1.0)
        type_score = sum(scores) / len(scores)
    return {"pred_elements": float(len(pred)), "target_elements": float(len(target)), "type_score": type_score}


def output_format_metrics(model_output: str) -> dict[str, float]:
    legacy_has_action_analysis = 1.0 if ACTION_ANALYSIS_MARKER_RE.search(model_output) else 0.0
    legacy_has_predicted_markdown = 1.0 if PREDICTED_MARKDOWN_MARKER_RE.search(model_output) else 0.0
    line_metrics = markdown_line_parse_metrics(model_output.strip())
    format_score = line_metrics["md_parseable_line_rate"]
    return {
        "format_score": format_score,
        "legacy_has_action_analysis": legacy_has_action_analysis,
        "legacy_has_predicted_markdown": legacy_has_predicted_markdown,
        **line_metrics,
    }


def aggregate_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def read_jsonl(path: Path) -> Iterable[dict]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
