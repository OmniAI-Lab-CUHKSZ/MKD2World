"""WebChain transition sample builder for Local World Model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mkd2world.actions import UnifiedAction, normalize_scroll_params
from mkd2world.training.markdown import (
    bbox_to_grid_center,
    build_assistant_output,
    clamp_grid,
    clamp_grid_coordinate,
    convert_normalized_coords_to_grid,
    read_jsonl,
)
from mkd2world.training.prompts import build_local_world_model_messages
from mkd2world.training.sft import normalize_row_for_dataset


ACTION_TYPE_MAP = {
    "back": "go_back",
    "change": "select_option",
    "click": "left_click",
    "copy": "hotkey",
    "mouse_click": "left_click",
    "double_click": "double_click",
    "doubleclick": "double_click",
    "enter": "key_press",
    "impossible": "fail",
    "go_back": "go_back",
    "go_forward": "go_forward",
    "goto": "navigate",
    "hover_at": "hover",
    "launchApp": "navigate",
    "load": "navigate",
    "long_press": "left_click",
    "moveto": "move_to",
    "mouse_drag_and_drop": "drag",
    "open_app": "navigate",
    "paste": "hotkey",
    "press_back": "go_back",
    "press_key": "key_press",
    "press_down": "key_press",
    "press_enter": "key_press",
    "press_home": "key_press",
    "press_pgdn": "key_press",
    "press_space": "key_press",
    "press_tab": "key_press",
    "type": "type_text",
    "text_input": "type_text",
    "keyboard_type": "type_text",
    "right_click": "right_click",
    "rightclick": "right_click",
    "select": "select_option",
    "scroll": "scroll",
    "scroll_at": "scroll",
    "say": "finish",
    "stop": "finish",
    "press": "key_press",
    "submit": "key_press",
    "keyboard_press": "key_press",
    "hover": "hover",
    "drag": "drag",
    "wait": "wait",
    "noop": "wait",
    "complete": "finish",
    "task_complete": "finish",
    "send_msg_to_user": "finish",
    "new_tab": "new_tab",
    "tab_focus": "page_focus",
}


KEY_FOR_RAW_ACTION = {
    "enter": "enter",
    "press": "enter",
    "press_down": "down",
    "press_enter": "enter",
    "press_home": "home",
    "press_pgdn": "pagedown",
    "press_space": "space",
    "press_tab": "tab",
    "submit": "enter",
}


STRICT_TRANSITION_STATUS = "matched_next_image"
TERMINAL_LOCAL_WM_ACTION_TYPES = {"finish", "fail"}
WEBCHAIN_RENDER_DIR_NAMES = ("tsm_renderer_strict_qwen08b", "tsm_renderer_type_qwen08b")
TRAINING_JSONL_NAMES = ("sft_train.jsonl", "grpo_train.jsonl", "test.jsonl")
PARQUET_ENRICH_FIELDS = (
    "instruction",
    "gt_action",
    "gt_bbox",
    "gt_input_text",
    "steps_count",
    "history",
    "task_type",
    "id",
)
CANONICAL_CANDIDATE_ACTION_KEYS = {
    "left_click": {"type", "coordinate"},
    "double_click": {"type", "coordinate"},
    "right_click": {"type", "coordinate"},
    "type_text": {"type", "text"},
    "scroll": {"type", "delta_x", "delta_y"},
    "key_press": {"type", "key"},
    "hotkey": {"type", "keys"},
    "key_down": {"type", "key"},
    "key_up": {"type", "key"},
    "navigate": {"type", "url"},
    "go_back": {"type"},
    "go_forward": {"type"},
    "refresh": {"type"},
    "select_option": {"type", "coordinate", "option"},
    "toggle_check": {"type", "coordinate"},
    "drag": {"type", "from_coordinate", "to_coordinate"},
    "hover": {"type", "coordinate"},
    "move_to": {"type", "coordinate"},
    "mouse_down": {"type", "coordinate"},
    "mouse_up": {"type", "coordinate"},
    "wait": {"type"},
    "new_tab": {"type"},
    "page_close": {"type"},
    "page_focus": {"type", "page_index"},
    "clear": {"type", "coordinate"},
    "upload": {"type", "path"},
    "finish": {"type", "answer"},
    "fail": {"type", "reason"},
}
CANONICAL_COORDINATE_KEYS = ("coordinate", "from_coordinate", "to_coordinate")
COORDINATE_PAYLOAD_ACTION_TYPES = {
    "left_click",
    "double_click",
    "right_click",
    "select_option",
    "toggle_check",
    "hover",
    "move_to",
    "mouse_down",
    "mouse_up",
    "clear",
}
GT_BBOX_COORDINATE_ACTION_TYPES = COORDINATE_PAYLOAD_ACTION_TYPES - {"clear"}


class SampleSkip(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _first_present(action: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if action.get(key) is not None:
            return action[key]
    return None


def _first_nonempty_text(action: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = action.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _canonical_action_type(action: Mapping[str, Any], *, default: str = "left_click") -> tuple[str, str]:
    raw_type = str(action.get("type") or action.get("action_type") or default)
    normalized_raw_type = raw_type.lower()
    return ACTION_TYPE_MAP.get(raw_type, ACTION_TYPE_MAP.get(normalized_raw_type, normalized_raw_type)), normalized_raw_type


def _normalize_coordinate(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("coordinate must be a two-item coordinate")
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return [clamp_grid_coordinate(float(value[0])), clamp_grid_coordinate(float(value[1]))]
    x, y = [float(item) for item in value]
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and (0.0 < x < 1.0 or 0.0 < y < 1.0):
        return [clamp_grid(x), clamp_grid(y)]
    return [clamp_grid_coordinate(x), clamp_grid_coordinate(y)]


def build_candidate_action(action: dict, *, include_raw: bool = False) -> dict:
    action_type, normalized_raw_type = _canonical_action_type(action)
    payload = {"type": action_type}
    if include_raw:
        payload["raw"] = action
    if action_type in COORDINATE_PAYLOAD_ACTION_TYPES and action.get("coordinate") is not None:
        payload["coordinate"] = _normalize_coordinate(action["coordinate"])
    if action_type == "drag":
        from_coordinate = _first_present(action, ("from_coordinate", "start_coordinate", "start"))
        to_coordinate = _first_present(action, ("to_coordinate", "end_coordinate", "end", "coordinate"))
        if from_coordinate is None and action.get("coordinate") is not None:
            from_coordinate = action["coordinate"]
        if from_coordinate is not None:
            payload["from_coordinate"] = _normalize_coordinate(from_coordinate)
        if to_coordinate is not None:
            payload["to_coordinate"] = _normalize_coordinate(to_coordinate)
    if action_type == "type_text":
        payload["text"] = action.get("text") or action.get("input_text") or action.get("gt_input_text") or ""
    if action_type == "scroll":
        scroll_source = dict(action)
        if not any(key in scroll_source for key in ("delta_x", "delta_y", "direction", "scroll_direction")):
            scroll_source["direction"] = "down"
        if "scroll_direction" in scroll_source and "direction" not in scroll_source:
            scroll_source["direction"] = scroll_source["scroll_direction"]
        normalized_scroll = normalize_scroll_params(scroll_source)
        payload["delta_x"] = normalized_scroll["delta_x"]
        payload["delta_y"] = normalized_scroll["delta_y"]
    if action_type == "key_press":
        payload["key"] = str(action.get("key") or KEY_FOR_RAW_ACTION.get(normalized_raw_type, "enter")).lower()
    if action_type == "hotkey":
        if action.get("keys"):
            payload["keys"] = [str(key).lower() for key in action["keys"]]
        elif normalized_raw_type == "copy":
            payload["keys"] = ["ctrl", "c"]
        elif normalized_raw_type == "paste":
            payload["keys"] = ["ctrl", "v"]
        else:
            payload["keys"] = [str(action.get("key") or "ctrl"), str(action.get("key2") or "l")]
    if action_type == "navigate":
        url = _first_nonempty_text(action, ("url", "target", "text"))
        if not url:
            raise ValueError("navigate action requires url, target, or text")
        payload["url"] = url
    if action_type == "finish":
        payload["answer"] = action.get("answer") or action.get("text") or ""
    if action_type == "fail":
        payload["reason"] = action.get("reason") or "dataset action marked failure"
    return UnifiedAction.from_dict(payload).to_dict()


def ensure_webchain_record_has_immediate_transition(record: Mapping[str, Any]) -> None:
    match_status = record.get("match_status")
    if match_status != STRICT_TRANSITION_STATUS:
        reason = match_status if match_status is not None else "missing"
        raise SampleSkip(f"match_status:{reason}")
    next_status = record.get("next_status")
    if next_status is not None and next_status != STRICT_TRANSITION_STATUS:
        raise SampleSkip(f"next_status:{next_status}")


def validate_candidate_action_for_local_wm(candidate_action: Mapping[str, Any]) -> dict[str, Any]:
    action = UnifiedAction.from_dict(candidate_action).to_dict()
    if action["type"] in TERMINAL_LOCAL_WM_ACTION_TYPES:
        raise SampleSkip(f"terminal_action:{action['type']}")
    if action["type"] == "type_text" and not str(action.get("text") or "").strip():
        raise SampleSkip("empty_type_text")
    return action


def remap_md_path_to_render_root(md_path: str | Path, render_root: Path | None) -> str:
    if render_root is None:
        return str(md_path)
    path = Path(md_path)
    path_text = path.as_posix()
    render_root_text = render_root.as_posix().rstrip("/")
    if path_text == render_root_text or path_text.startswith(f"{render_root_text}/"):
        return str(path)
    parts = path.parts
    for index, part in enumerate(parts):
        if part in WEBCHAIN_RENDER_DIR_NAMES:
            return str(render_root / Path(*parts[index:]))
    raise ValueError(f"Cannot find WebChain render directory in md path: {md_path}")


def _path_index_keys(path_value: str | Path, repo_root: Path) -> set[str]:
    path = Path(str(path_value))
    keys = {str(path_value), path.as_posix()}
    if path.is_absolute():
        try:
            keys.add(path.relative_to(repo_root).as_posix())
        except ValueError:
            pass
    else:
        keys.add((repo_root / path).as_posix())
    return {key for key in keys if key}


def load_image_job_index(job_files: Sequence[Path], repo_root: Path) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for job_file in job_files:
        for job in read_jsonl(job_file):
            role = str(job.get("role") or "")
            image_path = str(job.get("image_path") or "")
            md_path = str(job.get("md_path") or "")
            if role and image_path and md_path:
                for key in _path_index_keys(image_path, repo_root):
                    index.setdefault((role, key), md_path)
    return index


def _lookup_image_job_md_path(
    image_job_index: Mapping[tuple[str, str], str],
    role: str,
    image_path: str,
    repo_root: Path,
) -> str | None:
    for key in _path_index_keys(image_path, repo_root):
        if (role, key) in image_job_index:
            return image_job_index[(role, key)]
    return None


def _attach_md_paths_from_image_jobs(
    record: dict[str, Any],
    image_job_index: Mapping[tuple[str, str], str],
    repo_root: Path,
) -> None:
    if not record.get("current_md_path"):
        current_image_path = str(record.get("current_image_path") or "")
        current_md_path = _lookup_image_job_md_path(image_job_index, "current", current_image_path, repo_root)
        if not current_md_path:
            raise SampleSkip("missing_image_job:current")
        record["current_md_path"] = current_md_path
    if not record.get("next_md_path"):
        next_image_path = str(record.get("next_image_path") or record.get("next_visible_image_path") or "")
        next_md_path = _lookup_image_job_md_path(image_job_index, "next", next_image_path, repo_root)
        if not next_md_path:
            raise SampleSkip("missing_image_job:next")
        record["next_md_path"] = next_md_path


def _read_parquet_rows(source_parquet: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment diagnostic path.
        raise RuntimeError("WebChain parquet enrichment requires pyarrow in the active conda environment.") from exc

    parquet_file = pq.ParquetFile(source_parquet)
    columns = [name for name in PARQUET_ENRICH_FIELDS if name in parquet_file.schema_arrow.names]
    return parquet_file.read(columns=columns).to_pylist()


def enrich_record_from_source_parquet(
    record: dict[str, Any],
    repo_root: Path,
    parquet_cache: dict[Path, list[dict[str, Any]]],
) -> None:
    source_parquet = record.get("source_parquet")
    row_index = record.get("row_index")
    if not source_parquet or row_index is None:
        return
    parquet_path = Path(str(source_parquet))
    if not parquet_path.is_absolute():
        parquet_path = repo_root / parquet_path
    if not parquet_path.is_file():
        raise SampleSkip("source_parquet:missing")
    if parquet_path not in parquet_cache:
        try:
            parquet_cache[parquet_path] = _read_parquet_rows(parquet_path)
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SampleSkip(f"source_parquet:read_error:{type(exc).__name__}") from exc
    rows = parquet_cache[parquet_path]
    try:
        index = int(row_index)
    except (TypeError, ValueError) as exc:
        raise SampleSkip("source_parquet:invalid_row_index") from exc
    if index < 0 or index >= len(rows):
        raise SampleSkip("source_parquet:row_index_out_of_range")
    source_row = rows[index]
    for key, value in source_row.items():
        if record.get(key) in (None, ""):
            record[key] = value


def prepare_render_record(
    record: Mapping[str, Any],
    repo_root: Path,
    *,
    render_root: Path | None = None,
    image_job_index: Mapping[tuple[str, str], str] | None = None,
    parquet_cache: dict[Path, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    prepared = dict(record)
    if image_job_index is not None:
        _attach_md_paths_from_image_jobs(prepared, image_job_index, repo_root)
    if parquet_cache is not None:
        enrich_record_from_source_parquet(prepared, repo_root, parquet_cache)
    if render_root is not None:
        try:
            if prepared.get("current_md_path"):
                prepared["current_md_path"] = remap_md_path_to_render_root(prepared["current_md_path"], render_root)
            if prepared.get("next_md_path"):
                prepared["next_md_path"] = remap_md_path_to_render_root(prepared["next_md_path"], render_root)
        except ValueError as exc:
            raise SampleSkip(f"md_path:cannot_remap:{exc}") from exc
    return prepared


def _read_markdown_file(repo_root: Path, md_path: str, *, reason_prefix: str) -> str:
    path = Path(md_path)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise SampleSkip(f"missing_{reason_prefix}_md")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SampleSkip(f"invalid_{reason_prefix}_md_encoding") from exc
    if not text.strip():
        raise SampleSkip(f"empty_{reason_prefix}_md")
    return text


def sample_from_render_record(record: dict, repo_root: Path) -> dict:
    ensure_webchain_record_has_immediate_transition(record)
    if not record.get("id"):
        raise SampleSkip("missing_id")
    if not record.get("current_image_path"):
        raise SampleSkip("missing_current_image_path")
    if not record.get("current_md_path"):
        raise SampleSkip("missing_current_md_path")
    if not record.get("next_md_path"):
        raise SampleSkip("missing_next_md_path")
    current_markdown = convert_normalized_coords_to_grid(
        _read_markdown_file(repo_root, record["current_md_path"], reason_prefix="current")
    )
    next_markdown = convert_normalized_coords_to_grid(
        _read_markdown_file(repo_root, record["next_md_path"], reason_prefix="next")
    )
    action = record.get("action") or {}
    if not action and record.get("gt_action"):
        action = {"type": record["gt_action"]}
    action_type, _ = _canonical_action_type(action)
    if "gt_bbox" in record and "coordinate" not in action and action_type in GT_BBOX_COORDINATE_ACTION_TYPES:
        gt_bbox = record["gt_bbox"]
        if not isinstance(gt_bbox, (list, tuple)) or len(gt_bbox) != 4:
            raise SampleSkip("invalid_bbox_shape")
        action = action | {"coordinate": list(bbox_to_grid_center(gt_bbox))}
    if record.get("gt_input_text") and not any(key in action for key in {"text", "input_text", "gt_input_text"}):
        action = action | {"gt_input_text": record["gt_input_text"]}
    candidate_action = validate_candidate_action_for_local_wm(build_candidate_action(action))
    analysis = record.get("action_analysis", "Predict the visible state after the candidate action.")
    current_image_path = Path(str(record["current_image_path"]))
    if not current_image_path.is_absolute():
        current_image_path = repo_root / current_image_path
    return {
        "id": record["id"],
        "current_image_path": str(current_image_path),
        "next_image_path": str(repo_root / record["next_image_path"]) if record.get("next_image_path") else "",
        "current_md_path": str(repo_root / record["current_md_path"]),
        "next_md_path": str(repo_root / record["next_md_path"]),
        "current_md": current_markdown,
        "candidate_action": candidate_action,
        "target_next_md": next_markdown,
        "assistant_output": build_assistant_output(analysis, next_markdown),
        "meta": {
            "raw": {key: value for key, value in record.items() if key not in {"current_md", "next_md"}},
            "webchain_action": action,
        },
    }


def _report_dict(total: int, written: int, skip_reasons: Counter[str], output: Path) -> dict[str, Any]:
    skipped = sum(skip_reasons.values())
    return {
        "output": str(output),
        "count": written,
        "written": written,
        "total": total,
        "skipped": skipped,
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }


def build_samples_report(
    manifest: Path,
    output: Path,
    repo_root: Path,
    *,
    render_root: Path | None = None,
    image_job_files: Sequence[Path] = (),
    enrich_parquet: bool = False,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    skip_reasons: Counter[str] = Counter()
    image_job_index = load_image_job_index(image_job_files, repo_root) if image_job_files else None
    parquet_cache: dict[Path, list[dict[str, Any]]] | None = {} if enrich_parquet else None
    with output.open("w", encoding="utf-8") as handle:
        for record in read_jsonl(manifest):
            total += 1
            try:
                prepared = prepare_render_record(
                    record,
                    repo_root,
                    render_root=render_root,
                    image_job_index=image_job_index,
                    parquet_cache=parquet_cache,
                )
                row = sample_from_render_record(prepared, repo_root)
            except SampleSkip as exc:
                skip_reasons[exc.reason] += 1
                continue
            except ValueError as exc:
                skip_reasons[f"invalid_action:{exc}"] += 1
                continue
            except (KeyError, TypeError) as exc:
                skip_reasons[f"bad_record:{type(exc).__name__}"] += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return _report_dict(total, written, skip_reasons, output)


def build_samples(manifest: Path, output: Path, repo_root: Path) -> int:
    return int(build_samples_report(manifest, output, repo_root)["count"])


def iter_group_split(rows: list[dict], ratio: float = 0.5) -> Iterable[tuple[str, dict]]:
    boundary = int(len(rows) * ratio)
    for index, row in enumerate(rows):
        yield ("sft" if index < boundary else "policy"), row


def _record_action_group(record: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> str:
    raw_action = record.get("gt_action")
    if isinstance(record.get("action"), Mapping):
        raw_action = record["action"].get("type") or raw_action
    if not raw_action and isinstance(record.get("meta"), Mapping):
        webchain_action = record["meta"].get("webchain_action")
        if isinstance(webchain_action, Mapping):
            raw_action = webchain_action.get("type")
    if not raw_action and row is not None:
        if isinstance(row.get("meta"), Mapping):
            webchain_action = row["meta"].get("webchain_action")
            if isinstance(webchain_action, Mapping):
                raw_action = webchain_action.get("type")
        action = row.get("candidate_action") if isinstance(row.get("candidate_action"), Mapping) else {}
        raw_action = raw_action or (action.get("type") if isinstance(action, Mapping) else None)
    text = str(raw_action or "unknown").lower()
    return "type" if text in {"type", "type_text", "text_input", "keyboard_type"} else text


def _safe_bucket_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text) or "unknown"


def _record_source_group(record: Mapping[str, Any]) -> str:
    for key in ("task_type", "group", "ui_type"):
        value = record.get(key)
        if value:
            return str(value).lower()
    source_parquet = record.get("source_parquet")
    if source_parquet:
        path = Path(str(source_parquet))
        parent = path.parent.name.lower()
        stem = path.stem.lower()
        if parent in {"high_parquet", "low_parquet"}:
            return parent.replace("_parquet", "")
        if stem.endswith("_test"):
            return stem
        return stem.rsplit("_part", 1)[0]
    source_manifest = record.get("source_manifest")
    if source_manifest:
        return Path(str(source_manifest)).stem.lower()
    return "unknown"


def _transition_dedup_key(row: Mapping[str, Any]) -> str:
    payload = {
        "current_md": row.get("current_md") or "",
        "candidate_action": row.get("candidate_action") or {},
        "target_next_md": row.get("target_next_md") or "",
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_is_test(record: Mapping[str, Any]) -> bool:
    for key in ("source_manifest", "source_parquet", "current_image_path"):
        value = record.get(key)
        if not value:
            continue
        stem = Path(str(value)).stem
        parent = Path(str(value)).parent.name
        if stem.endswith("_test") or parent.endswith("_test"):
            return True
    return False


def build_split_dataset_report(
    manifests: Sequence[Path],
    output_dir: Path,
    repo_root: Path,
    *,
    render_root: Path | None = None,
    image_job_files: Sequence[Path] = (),
    enrich_parquet: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / f".tmp_webchain_build_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    image_job_index = load_image_job_index(image_job_files, repo_root) if image_job_files else None
    parquet_cache: dict[Path, list[dict[str, Any]]] | None = {} if enrich_parquet else None
    skip_reasons: Counter[str] = Counter()
    rows_counter: Counter[str] = Counter()
    train_bucket_paths: dict[str, Path] = {}
    train_bucket_actions: dict[str, str] = {}
    train_bucket_sources: dict[str, str] = {}
    train_counts: Counter[str] = Counter()
    train_transition_keys: set[str] = set()
    test_candidate_path = tmp_dir / "test_candidates.jsonl"
    test_candidate_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()

    try:
        for manifest in manifests:
            rows_counter[f"manifest:{manifest.name}"] += 0
            for record in read_jsonl(manifest):
                rows_counter["total"] += 1
                rows_counter[f"manifest:{manifest.name}"] += 1
                try:
                    prepared = prepare_render_record(
                        record,
                        repo_root,
                        render_root=render_root,
                        image_job_index=image_job_index,
                        parquet_cache=parquet_cache,
                    )
                    row = sample_from_render_record(prepared, repo_root)
                except SampleSkip as exc:
                    skip_reasons[exc.reason] += 1
                    continue
                except ValueError as exc:
                    skip_reasons[f"invalid_action:{exc}"] += 1
                    continue
                except (KeyError, TypeError) as exc:
                    skip_reasons[f"bad_record:{type(exc).__name__}"] += 1
                    continue

                action_group = _record_action_group(prepared, row)
                if _record_is_test(prepared):
                    with test_candidate_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    test_candidate_counts[action_group] += 1
                    rows_counter[f"candidate:test:{action_group}"] += 1
                else:
                    source_group = _record_source_group(prepared)
                    bucket_key = f"{source_group}:{action_group}"
                    bucket_path = train_bucket_paths.setdefault(bucket_key, tmp_dir / f"train_{_safe_bucket_name(bucket_key)}.jsonl")
                    train_bucket_actions[bucket_key] = action_group
                    train_bucket_sources[bucket_key] = source_group
                    with bucket_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    train_counts[bucket_key] += 1
                    train_transition_keys.add(_transition_dedup_key(row))
                    rows_counter[f"accepted:train:{source_group}:{action_group}"] += 1

        sft_path = output_dir / "sft_train.jsonl"
        grpo_path = output_dir / "grpo_train.jsonl"
        test_path = output_dir / "test.jsonl"
        sft_count = 0
        grpo_count = 0
        action_split: dict[str, dict[str, int]] = {}
        strata_split: dict[str, dict[str, Any]] = {}
        with sft_path.open("w", encoding="utf-8") as sft_handle, grpo_path.open("w", encoding="utf-8") as grpo_handle:
            for bucket_key in sorted(train_bucket_paths):
                action_group = train_bucket_actions[bucket_key]
                source_group = train_bucket_sources[bucket_key]
                boundary = train_counts[bucket_key] // 2
                action_sft = 0
                action_grpo = 0
                with train_bucket_paths[bucket_key].open("r", encoding="utf-8") as source:
                    for index, line in enumerate(source):
                        if index < boundary:
                            sft_handle.write(line)
                            sft_count += 1
                            action_sft += 1
                        else:
                            grpo_handle.write(line)
                            grpo_count += 1
                            action_grpo += 1
                split = action_split.setdefault(action_group, {"total": 0, "sft": 0, "grpo": 0})
                split["total"] += train_counts[bucket_key]
                split["sft"] += action_sft
                split["grpo"] += action_grpo
                strata_split[bucket_key] = {
                    "source": source_group,
                    "action": action_group,
                    "total": train_counts[bucket_key],
                    "sft": action_sft,
                    "grpo": action_grpo,
                }
        test_count = 0
        with test_path.open("w", encoding="utf-8") as test_handle:
            if test_candidate_path.exists():
                with test_candidate_path.open("r", encoding="utf-8") as source:
                    for line in source:
                        row = json.loads(line)
                        action_group = _record_action_group(row, row)
                        if _transition_dedup_key(row) in train_transition_keys:
                            skip_reasons["duplicate_test_transition"] += 1
                            rows_counter[f"duplicate:test:{action_group}"] += 1
                            continue
                        test_handle.write(line)
                        test_count += 1
                        test_counts[action_group] += 1
                        rows_counter[f"accepted:test:{action_group}"] += 1

        report = {
            "output_dir": str(output_dir),
            "paths": {
                "sft_train": str(sft_path),
                "grpo_train": str(grpo_path),
                "test": str(test_path),
            },
            "counts": {
                "sft_train": sft_count,
                "grpo_train": grpo_count,
                "test": test_count,
            },
            "action_split": action_split,
            "strata_split": dict(sorted(strata_split.items())),
            "test_candidates_by_action": dict(sorted(test_candidate_counts.items())),
            "test_by_action": dict(sorted(test_counts.items())),
            "rows": dict(sorted(rows_counter.items())),
            "skipped": dict(sorted(skip_reasons.items())),
            "render_root": str(render_root) if render_root else None,
            "image_job_files": [str(path) for path in image_job_files],
            "enrich_parquet": enrich_parquet,
            "notes": [
                "candidate_action contains canonical UnifiedAction JSON only; raw WebChain fields stay in meta.",
                "Only matched_next_image immediate transitions are kept.",
                "Train rows are divided per source action group: first half SFT, second half GRPO.",
            ],
        }
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _repo_path(repo_root: Path, path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else repo_root / path


def _validate_path_exists(
    row: Mapping[str, Any],
    repo_root: Path,
    key: str,
    errors: Counter[str],
    *,
    required: bool = True,
) -> None:
    value = row.get(key)
    if value in (None, ""):
        if required:
            errors[f"missing_field:{key}"] += 1
        return
    if not _repo_path(repo_root, value).is_file():
        errors[f"missing_file:{key}"] += 1


def _validate_candidate_coordinate_types(candidate_action: Mapping[str, Any], errors: Counter[str]) -> None:
    for key in CANONICAL_COORDINATE_KEYS:
        value = candidate_action.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        if any(isinstance(item, bool) for item in value):
            errors[f"candidate_action:{key}:bool_coordinate"] += 1


def _validate_candidate_action(
    candidate_action: Mapping[str, Any],
    errors: Counter[str],
) -> dict[str, Any] | None:
    if "raw" in candidate_action:
        errors["candidate_action:raw_present"] += 1
    _validate_candidate_coordinate_types(candidate_action, errors)
    try:
        action = UnifiedAction.from_dict(candidate_action).to_dict()
    except Exception as exc:  # noqa: BLE001
        errors[f"candidate_action:invalid:{type(exc).__name__}"] += 1
        return None
    action_type = action["type"]
    if action_type in TERMINAL_LOCAL_WM_ACTION_TYPES:
        errors[f"candidate_action:terminal:{action_type}"] += 1
    if action_type == "type_text" and not str(action.get("text") or "").strip():
        errors["candidate_action:empty_type_text"] += 1
    allowed_keys = CANONICAL_CANDIDATE_ACTION_KEYS.get(action_type, {"type"})
    for key in sorted(set(candidate_action) - allowed_keys - {"raw"}):
        errors[f"candidate_action:unexpected_key:{key}"] += 1
    return action


def _candidate_action_json_has_raw_markers(candidate_action_json: str, errors: Counter[str]) -> None:
    if "gt_action" in candidate_action_json:
        errors["prompt:contains_gt_action"] += 1
    if '"raw"' in candidate_action_json:
        errors["prompt:contains_raw_key"] += 1
    if '"type": "click"' in candidate_action_json or '"type":"click"' in candidate_action_json:
        errors["prompt:contains_raw_click_type"] += 1


def _validate_training_row(
    row: Mapping[str, Any],
    repo_root: Path,
    errors: Counter[str],
    *,
    check_prompt: bool,
    require_audit_paths: bool = True,
) -> tuple[dict[str, Any] | None, bool]:
    _validate_path_exists(row, repo_root, "current_image_path", errors)
    _validate_path_exists(row, repo_root, "next_image_path", errors, required=require_audit_paths)
    _validate_path_exists(row, repo_root, "current_md_path", errors, required=require_audit_paths)
    _validate_path_exists(row, repo_root, "next_md_path", errors, required=require_audit_paths)
    if not str(row.get("current_md") or "").strip():
        errors["empty_field:current_md"] += 1
    if not str(row.get("target_next_md") or "").strip():
        errors["empty_field:target_next_md"] += 1
    if not str(row.get("assistant_output") or "").strip():
        errors["empty_field:assistant_output"] += 1

    candidate_action = row.get("candidate_action")
    if not isinstance(candidate_action, Mapping):
        errors["candidate_action:not_mapping"] += 1
        return None, False
    action = _validate_candidate_action(candidate_action, errors)

    if check_prompt:
        normalized = normalize_row_for_dataset(dict(row))
        candidate_action_json = str(normalized.get("candidate_action_json") or "")
        messages = build_local_world_model_messages(normalized, include_image=False)
        user_text = messages[1]["content"][0]["text"]
        if candidate_action_json not in user_text:
            errors["prompt:missing_candidate_action_json"] += 1
        _candidate_action_json_has_raw_markers(candidate_action_json, errors)
        return action, True
    return action, False


def validate_training_jsonl_report(
    jsonl_paths: Sequence[Path],
    repo_root: Path,
    *,
    prompt_check_samples: int = 0,
    max_error_examples: int = 20,
    require_audit_paths: bool = True,
) -> dict[str, Any]:
    total = 0
    prompt_checked = 0
    errors: Counter[str] = Counter()
    action_types: Counter[str] = Counter()
    files: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []

    for jsonl_path in jsonl_paths:
        file_total = 0
        file_errors_before = sum(errors.values())
        if not jsonl_path.is_file():
            errors["missing_jsonl"] += 1
            examples.append({"file": str(jsonl_path), "line": None, "errors": ["missing_jsonl"]})
            files[str(jsonl_path)] = {"rows": 0, "errors": 1}
            continue
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                total += 1
                file_total += 1
                row_errors_before = sum(errors.values())
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors["json:decode_error"] += 1
                    if len(examples) < max_error_examples:
                        examples.append({"file": str(jsonl_path), "line": line_number, "errors": ["json:decode_error"]})
                    continue
                if not isinstance(row, Mapping):
                    errors["json:not_object"] += 1
                    if len(examples) < max_error_examples:
                        examples.append({"file": str(jsonl_path), "line": line_number, "errors": ["json:not_object"]})
                    continue
                check_prompt = prompt_check_samples <= 0 or prompt_checked < prompt_check_samples
                action, prompt_was_checked = _validate_training_row(
                    row,
                    repo_root,
                    errors,
                    check_prompt=check_prompt,
                    require_audit_paths=require_audit_paths,
                )
                if prompt_was_checked:
                    prompt_checked += 1
                if action is not None:
                    action_types[action["type"]] += 1
                row_new_error_count = sum(errors.values()) - row_errors_before
                if row_new_error_count and len(examples) < max_error_examples:
                    examples.append(
                        {
                            "file": str(jsonl_path),
                            "line": line_number,
                            "id": row.get("id"),
                            "errors_added": row_new_error_count,
                        }
                    )
        files[str(jsonl_path)] = {
            "rows": file_total,
            "errors": sum(errors.values()) - file_errors_before,
        }

    return {
        "status": "ok" if not errors else "failed",
        "total_rows": total,
        "prompt_checked": prompt_checked,
        "files": files,
        "action_types": dict(sorted(action_types.items())),
        "error_count": sum(errors.values()),
        "errors": dict(sorted(errors.items())),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WebChain Local World Model JSONL samples.")
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--image-job-file", type=Path, action="append", default=[])
    parser.add_argument("--enrich-parquet", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validate-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--validate-output-dir", type=Path)
    parser.add_argument("--prompt-check-samples", type=int, default=0)
    parser.add_argument("--max-error-examples", type=int, default=20)
    args = parser.parse_args()
    if args.validate_jsonl or args.validate_output_dir is not None:
        jsonl_paths = list(args.validate_jsonl)
        if args.validate_output_dir is not None:
            jsonl_paths.extend(args.validate_output_dir / name for name in TRAINING_JSONL_NAMES)
        report = validate_training_jsonl_report(
            jsonl_paths,
            args.repo_root,
            prompt_check_samples=args.prompt_check_samples,
            max_error_examples=args.max_error_examples,
        )
        print(json.dumps(report, ensure_ascii=False))
        if report["status"] != "ok":
            raise SystemExit(1)
        return
    if not args.manifest:
        raise ValueError("--manifest is required unless validation mode is used")
    if args.output_dir is not None:
        report = build_split_dataset_report(
            args.manifest,
            args.output_dir,
            args.repo_root,
            render_root=args.render_root,
            image_job_files=args.image_job_file,
            enrich_parquet=True if args.enrich_parquet is None else args.enrich_parquet,
        )
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.output is None:
        raise ValueError("--output is required unless --output-dir is set")
    if len(args.manifest) != 1:
        raise ValueError("single-output mode requires exactly one --manifest")
    report = build_samples_report(
        args.manifest[0],
        args.output,
        args.repo_root,
        render_root=args.render_root,
        image_job_files=args.image_job_file,
        enrich_parquet=False if args.enrich_parquet is None else args.enrich_parquet,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
