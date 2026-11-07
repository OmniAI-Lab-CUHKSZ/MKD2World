"""Dataset adapter boundaries for Local World Model transition samples."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mkd2world.training.markdown import build_assistant_output, clamp_grid, read_jsonl
from mkd2world.training.webchain import (
    GT_BBOX_COORDINATE_ACTION_TYPES,
    SampleSkip,
    _canonical_action_type,
    build_candidate_action,
    ensure_webchain_record_has_immediate_transition,
    validate_candidate_action_for_local_wm,
)


SUPPORTED_DATASETS = ("webchain", "weblinx", "molmo", "online_mind2web")


@dataclass(frozen=True)
class TransitionSample:
    id: str
    current_image_path: str = ""
    current_md: str = ""
    candidate_action: dict[str, Any] = field(default_factory=dict)
    target_next_md: str = ""
    action_analysis: str = "Predict the visible state after the candidate action."
    goal_text: str = ""
    subgoal_text: str = ""
    url: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_training_row(self) -> dict[str, Any]:
        row = {
            "id": self.id,
            "current_image_path": self.current_image_path,
            "current_md": self.current_md,
            "candidate_action": self.candidate_action,
            "target_next_md": self.target_next_md,
            "assistant_output": build_assistant_output(self.action_analysis, self.target_next_md),
            "meta": self.meta,
        }
        if self.goal_text:
            row["goal_text"] = self.goal_text
        if self.subgoal_text:
            row["subgoal_text"] = self.subgoal_text
        if self.url:
            row["url"] = self.url
        return row


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _first_mapping(row: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _bbox_center(bbox: object) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(item) for item in bbox]
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def _strip_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _parse_call(text: str) -> tuple[str, list[str]] | None:
    match = re.match(r"\s*(?P<name>[A-Za-z_][\w]*)\s*\((?P<args>.*)\)\s*$", text)
    if not match:
        return None
    args = []
    current = []
    quote = ""
    for char in match.group("args"):
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char == ",":
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current or match.group("args").strip():
        args.append("".join(current).strip())
    return match.group("name"), args


def _literal_to_text(value: object) -> str:
    return str(value)


def _parse_call_with_kwargs(text: str) -> tuple[str, list[str], dict[str, str]] | None:
    try:
        expr = ast.parse(text.strip(), mode="eval").body
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
            return None
        args = [_literal_to_text(ast.literal_eval(item)) for item in expr.args]
        kwargs = {}
        for item in expr.keywords:
            if item.arg is None:
                continue
            kwargs[item.arg] = _literal_to_text(ast.literal_eval(item.value))
        return expr.func.id, args, kwargs
    except (SyntaxError, ValueError):
        parsed = _parse_call(text)
        if parsed is None:
            return None
        return parsed[0], parsed[1], {}


def _float_arg(value: str) -> float:
    return float(_strip_quotes(value))


def _optional_float_arg(value: str) -> float | None:
    try:
        return _float_arg(value)
    except ValueError:
        return None


def _arg_value(args: list[str], kwargs: Mapping[str, str], index: int, *names: str, default: str = "") -> str:
    for name in names:
        if name in kwargs:
            return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def _molmo_coordinate(value: str, *, size: int | None = None) -> float | int:
    number = _float_arg(value)
    if size and size > 0 and not 0.0 <= number <= 1.0:
        return clamp_grid(number / float(size))
    return number


def parse_molmo_action_text(
    action_text: str,
    *,
    screen_width: int | None = None,
    screen_height: int | None = None,
) -> dict[str, Any]:
    parsed = _parse_call_with_kwargs(action_text)
    if parsed is None:
        lowered = action_text.strip().lower()
        return {"type": lowered} if lowered else {}
    name, args, kwargs = parsed
    action: dict[str, Any] = {"type": name}
    lowered = name.lower()
    if lowered in {"mouse_click", "hover_at"} and len(args) >= 2:
        action["coordinate"] = [
            _molmo_coordinate(args[0], size=screen_width),
            _molmo_coordinate(args[1], size=screen_height),
        ]
    elif lowered in {"mouse_click", "hover_at"} and {"x", "y"}.issubset(kwargs):
        action["coordinate"] = [
            _molmo_coordinate(kwargs["x"], size=screen_width),
            _molmo_coordinate(kwargs["y"], size=screen_height),
        ]
        if lowered == "mouse_click" and _strip_quotes(kwargs.get("button", "")).lower() == "right":
            action["type"] = "right_click"
    elif lowered == "mouse_drag_and_drop" and len(args) >= 4:
        action["from_coordinate"] = [
            _molmo_coordinate(args[0], size=screen_width),
            _molmo_coordinate(args[1], size=screen_height),
        ]
        action["to_coordinate"] = [
            _molmo_coordinate(args[2], size=screen_width),
            _molmo_coordinate(args[3], size=screen_height),
        ]
    elif lowered == "mouse_drag_and_drop" and {"x1", "y1", "x2", "y2"}.issubset(kwargs):
        action["from_coordinate"] = [
            _molmo_coordinate(kwargs["x1"], size=screen_width),
            _molmo_coordinate(kwargs["y1"], size=screen_height),
        ]
        action["to_coordinate"] = [
            _molmo_coordinate(kwargs["x2"], size=screen_width),
            _molmo_coordinate(kwargs["y2"], size=screen_height),
        ]
    elif lowered == "mouse_drag_and_drop" and {"from_x", "from_y", "to_x", "to_y"}.issubset(kwargs):
        action["from_coordinate"] = [
            _molmo_coordinate(kwargs["from_x"], size=screen_width),
            _molmo_coordinate(kwargs["from_y"], size=screen_height),
        ]
        action["to_coordinate"] = [
            _molmo_coordinate(kwargs["to_x"], size=screen_width),
            _molmo_coordinate(kwargs["to_y"], size=screen_height),
        ]
    elif lowered in {"scroll", "scroll_at"}:
        first_value = _arg_value(args, kwargs, 0, "delta_x", default="0")
        second_value = _arg_value(args, kwargs, 1, "delta_y", default="0")
        delta_x = _optional_float_arg(first_value)
        delta_y = _optional_float_arg(second_value)
        if delta_x is None or delta_y is None:
            action["direction"] = _strip_quotes(first_value or "down").lower()
            action["amount"] = "medium"
            return action
        if lowered == "scroll_at":
            x_value = _arg_value(args, kwargs, 0, "x")
            y_value = _arg_value(args, kwargs, 1, "y")
            if x_value and y_value:
                action["coordinate"] = [
                    _molmo_coordinate(x_value, size=screen_width),
                    _molmo_coordinate(y_value, size=screen_height),
                ]
            if "delta_x" not in kwargs and len(args) >= 3:
                parsed_delta_x = _optional_float_arg(args[2])
                if parsed_delta_x is None:
                    action["direction"] = _strip_quotes(args[2]).lower()
                    action["amount"] = "medium"
                    return action
                delta_x = parsed_delta_x
            if "delta_y" not in kwargs and len(args) >= 4:
                delta_y = _float_arg(args[3])
        action["delta_x"] = float(delta_x)
        action["delta_y"] = float(delta_y)
    elif lowered == "keyboard_type":
        text = _arg_value(args, kwargs, 0, "text")
        action["text"] = _strip_quotes(text)
    elif lowered == "keyboard_press":
        key = _arg_value(args, kwargs, 0, "key", default="enter")
        action["key"] = _strip_quotes(key).lower()
    elif lowered == "goto":
        url = _arg_value(args, kwargs, 0, "url")
        action["url"] = _strip_quotes(url)
    elif lowered == "send_msg_to_user":
        answer = _arg_value(args, kwargs, 0, "text", "answer")
        action["answer"] = _strip_quotes(answer)
    return action


def _candidate_from_action(raw_action: Mapping[str, Any]) -> dict[str, Any]:
    return validate_candidate_action_for_local_wm(build_candidate_action(dict(raw_action)))


def adapt_webchain_record(row: Mapping[str, Any]) -> TransitionSample:
    ensure_webchain_record_has_immediate_transition(row)
    action = _first_mapping(row, "action", "candidate_action")
    if not action and row.get("gt_action"):
        action = {"type": row["gt_action"]}
    action_type, _ = _canonical_action_type(action)
    if row.get("gt_bbox") is not None and "coordinate" not in action and action_type in GT_BBOX_COORDINATE_ACTION_TYPES:
        coordinate = _bbox_center(row["gt_bbox"])
        if coordinate is None:
            raise SampleSkip("invalid_bbox_shape")
        action["coordinate"] = coordinate
    if row.get("gt_input_text") and not any(key in action for key in ("text", "input_text", "gt_input_text")):
        action["gt_input_text"] = row["gt_input_text"]
    return TransitionSample(
        id=_first_text(row, "id", "uid") or "webchain-sample",
        current_image_path=_first_text(row, "current_image_path", "image_path"),
        current_md=_first_text(row, "current_md", "before_md"),
        candidate_action=_candidate_from_action(action),
        target_next_md=_first_text(row, "target_next_md", "next_md", "after_md"),
        action_analysis=_first_text(row, "action_analysis") or "Predict the visible state after the WebChain action.",
        goal_text=_first_text(row, "goal_text", "task", "raw_task_title"),
        subgoal_text=_first_text(row, "subgoal_text", "raw_step_title"),
        url=_first_text(row, "url") or None,
        meta={"dataset": "webchain", "raw": dict(row), "webchain_action": dict(action)},
    )


def adapt_weblinx_record(row: Mapping[str, Any]) -> TransitionSample:
    action = _first_mapping(row, "action", "candidate_action")
    raw_type = _first_text(row, "action_type", "type", "intent")
    if raw_type and "type" not in action:
        action["type"] = raw_type
    if row.get("bbox") is not None and "coordinate" not in action:
        action["coordinate"] = _bbox_center(row["bbox"]) or row["bbox"]
    if row.get("candidates") and "coordinate" not in action:
        candidates = row["candidates"]
        if isinstance(candidates, list) and candidates:
            target_uid = str(row.get("uid") or row.get("element_uid") or action.get("uid") or "")
            candidate = None
            if target_uid:
                for item in candidates:
                    if isinstance(item, Mapping) and str(item.get("uid") or item.get("id") or "") == target_uid:
                        candidate = item
                        break
            if candidate is None:
                candidate = candidates[0]
            if isinstance(candidate, Mapping) and candidate.get("bbox") is not None:
                action["coordinate"] = _bbox_center(candidate["bbox"]) or candidate["bbox"]
    if row.get("text") and "text" not in action:
        action["text"] = row["text"]
    return TransitionSample(
        id=_first_text(row, "id", "uid", "turn_id") or "weblinx-sample",
        current_image_path=_first_text(row, "current_image_path", "screenshot", "image_path"),
        current_md=_first_text(row, "current_md", "before_md"),
        candidate_action=_candidate_from_action(action),
        target_next_md=_first_text(row, "target_next_md", "next_md", "after_md"),
        action_analysis=_first_text(row, "action_analysis") or "Predict the visible state after the WebLINX action.",
        goal_text=_first_text(row, "goal_text", "utterance", "task"),
        subgoal_text=_first_text(row, "subgoal_text", "turn_text"),
        url=_first_text(row, "url") or None,
        meta={"dataset": "weblinx", "raw": dict(row)},
    )


def adapt_molmo_record(row: Mapping[str, Any]) -> TransitionSample:
    action = _first_mapping(row, "action", "candidate_action")
    action_str = _first_text(row, "action_str", "action_string")
    if action_str:
        parsed_action = parse_molmo_action_text(
            action_str,
            screen_width=int(row["screen_width"]) if row.get("screen_width") else None,
            screen_height=int(row["screen_height"]) if row.get("screen_height") else None,
        )
        action = parsed_action | action
        action["raw_action_text"] = action_str
    raw_type = _first_text(action, "type", "action_type") or _first_text(row, "type", "action_type")
    if raw_type and "type" not in action:
        action["type"] = raw_type
    if row.get("bbox") is not None and "coordinate" not in action:
        action["coordinate"] = _bbox_center(row["bbox"]) or row["bbox"]
    return TransitionSample(
        id=_first_text(row, "id", "step_id", "trajectory_id") or "molmo-sample",
        current_image_path=_first_text(row, "current_image_path", "screenshot", "image_path"),
        current_md=_first_text(row, "current_md", "before_md"),
        candidate_action=_candidate_from_action(action),
        target_next_md=_first_text(row, "target_next_md", "next_md", "after_md"),
        action_analysis=_first_text(row, "action_analysis", "think", "action_description")
        or "Predict the visible state after the Molmo action.",
        goal_text=_first_text(row, "goal_text", "instruction", "high_level", "high-level"),
        subgoal_text=_first_text(row, "subgoal_text", "step"),
        url=_first_text(row, "url") or None,
        meta={"dataset": "molmo", "raw": dict(row)},
    )


def adapt_online_mind2web_record(row: Mapping[str, Any]) -> TransitionSample:
    action = _first_mapping(row, "action", "candidate_action")
    raw_type = _first_text(action, "type", "action_type") or _first_text(row, "action_type", "operation")
    if raw_type and "type" not in action:
        action["type"] = raw_type
    if row.get("coordinate") is not None and "coordinate" not in action:
        action["coordinate"] = row["coordinate"]
    if row.get("bbox") is not None and "coordinate" not in action:
        action["coordinate"] = _bbox_center(row["bbox"]) or row["bbox"]
    if row.get("value") and "text" not in action:
        action["text"] = row["value"]
    return TransitionSample(
        id=_first_text(row, "id", "annotation_id", "action_uid") or "online-mind2web-sample",
        current_image_path=_first_text(row, "current_image_path", "screenshot", "image_path"),
        current_md=_first_text(row, "current_md", "before_md"),
        candidate_action=_candidate_from_action(action),
        target_next_md=_first_text(row, "target_next_md", "next_md", "after_md"),
        action_analysis=_first_text(row, "action_analysis") or "Predict the visible state after the Online-Mind2Web action.",
        goal_text=_first_text(row, "goal_text", "confirmed_task", "task"),
        subgoal_text=_first_text(row, "subgoal_text", "action_reprs"),
        url=_first_text(row, "url", "website") or None,
        meta={"dataset": "online_mind2web", "raw": dict(row)},
    )


DATASET_ADAPTERS = {
    "webchain": adapt_webchain_record,
    "weblinx": adapt_weblinx_record,
    "molmo": adapt_molmo_record,
    "online_mind2web": adapt_online_mind2web_record,
}


def adapt_transition_record(dataset: str, row: Mapping[str, Any]) -> TransitionSample:
    try:
        adapter = DATASET_ADAPTERS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset adapter: {dataset}") from exc
    return adapter(row)


def build_transition_samples(dataset: str, manifest: Path, output: Path) -> int:
    return int(build_transition_samples_report(dataset, manifest, output)["count"])


def build_transition_samples_report(dataset: str, manifest: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    skip_reasons: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(manifest):
            total += 1
            try:
                sample = adapt_transition_record(dataset, row).to_training_row()
            except SampleSkip as exc:
                skip_reasons[exc.reason] += 1
                continue
            except ValueError as exc:
                skip_reasons[f"invalid_action:{exc}"] += 1
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    skipped = sum(skip_reasons.values())
    return {
        "dataset": dataset,
        "output": str(output),
        "count": written,
        "written": written,
        "total": total,
        "skipped": skipped,
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }


def dry_run_report(dataset: str, manifest: Path | None, output: Path | None) -> dict[str, Any]:
    missing = []
    if dataset not in DATASET_ADAPTERS:
        missing.append(f"dataset:{dataset}")
    if manifest is None:
        missing.append("manifest")
    elif not manifest.is_file():
        missing.append(f"manifest:{manifest}")
    if output is None:
        missing.append("output")
    return {
        "status": "dry_run",
        "can_build": not missing,
        "missing_assets": missing,
        "dataset": dataset,
        "supported_datasets": list(SUPPORTED_DATASETS),
        "manifest": str(manifest) if manifest else None,
        "output": str(output) if output else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build normalized Local WM transition samples from dataset manifests.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="webchain")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_report(args.dataset, args.manifest, args.output), ensure_ascii=False, indent=2))
        return
    if args.manifest is None or args.output is None:
        raise ValueError("--manifest and --output are required unless --dry-run is set")
    report = build_transition_samples_report(args.dataset, args.manifest, args.output)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
