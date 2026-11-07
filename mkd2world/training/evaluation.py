"""Offline Local World Model evaluation helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from mkd2world.actions import ActionType, UnifiedAction
from mkd2world.config import apply_config_defaults, load_run_config, provided_cli_dests
from mkd2world.training.markdown import (
    aggregate_metric_dicts,
    extract_predicted_markdown,
    output_format_metrics,
    read_jsonl,
)
from mkd2world.training.reward import SemanticEmbedder, reward_from_markdown
from mkd2world.training.webchain import build_candidate_action

DOCUMENT_METRIC_KEYS = ("EleF1",)
ACTION_BUCKETS = {
    "click": (
        "left_click",
        "double_click",
        "right_click",
        "select_option",
        "toggle_check",
        "mouse_down",
        "mouse_up",
    ),
    "text_key": ("type_text", "key_press", "hotkey", "key_down", "key_up", "clear"),
    "scroll": ("scroll",),
    "navigation": ("navigate", "go_back", "go_forward", "refresh", "new_tab", "page_focus", "page_close"),
    "other": ("drag", "hover", "move_to", "wait", "upload", "finish", "fail"),
}
ACTION_TYPE_TO_BUCKET = {
    action_type: bucket
    for bucket, action_types in ACTION_BUCKETS.items()
    for action_type in action_types
}
ACTION_FIELD_CANDIDATES = ("candidate_action_json", "candidate_action", "action")


def evaluate_prediction(model_output: str, target_markdown: str, embedder: SemanticEmbedder | None = None) -> dict[str, float]:
    if embedder is None:
        raise ValueError("EleF1 evaluation requires an embedder")
    reward_metrics = reward_from_markdown(model_output, target_markdown, embedder)
    predicted_markdown = extract_predicted_markdown(model_output)
    format_metrics = output_format_metrics(predicted_markdown)
    format_success = float(
        predicted_markdown.startswith("[Environment: Web]")
        and format_metrics["md_nonempty_lines"] > 1.0
        and format_metrics["md_unparseable_lines"] == 0.0
        and reward_metrics.get("pred_elements", 0.0) > 0.0
    )
    return {
        key: value
        for key, value in reward_metrics.items()
        if key != "reward"
    } | {**format_metrics, "format_success_rate": format_success}


def _parse_action_value(value: object) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("action value must be a JSON object")
    payload = dict(value)
    if payload.get("type") in (None, "") and payload.get("action_type") in (None, ""):
        raise ValueError("action value requires type or action_type")
    try:
        return UnifiedAction.from_dict(payload).type.value
    except Exception:
        return UnifiedAction.from_dict(build_candidate_action(payload)).type.value


def action_type_from_row(row: Mapping[str, Any], *, action_field: str = "auto") -> str:
    fields = ACTION_FIELD_CANDIDATES if action_field == "auto" else (action_field,)
    last_error: Exception | None = None
    for field in fields:
        if field not in row or row.get(field) in (None, ""):
            continue
        try:
            return _parse_action_value(row[field])
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise ValueError(str(last_error)) from last_error
    raise ValueError("missing action field")


def action_bucket_for_type(action_type: str) -> str:
    if action_type in {item.value for item in ActionType}:
        return ACTION_TYPE_TO_BUCKET.get(action_type, "other")
    return "other"


def aggregate_ele_metrics(rows: list[dict[str, float]]) -> dict[str, Any]:
    """Average per-interface metrics so every evaluated sample has equal weight."""
    metrics = aggregate_metric_dicts(rows)
    for key in ("matched_score", "pred_elements", "target_elements", "ElePrecision", "EleRecall", "EleF1"):
        metrics.setdefault(key, 0.0)
    metrics["evaluated_rows"] = float(len(rows))
    return metrics


def evaluate_jsonl(
    predictions_jsonl: Path,
    *,
    prediction_field: str = "model_output",
    target_field: str = "target_next_md",
    embedder: SemanticEmbedder | None = None,
    action_breakdown: bool = True,
    action_field: str = "auto",
) -> dict[str, Any]:
    if embedder is None:
        raise ValueError("EleF1 evaluation requires an embedder")
    if not action_breakdown:
        raise ValueError("EleF1 evaluation requires action_breakdown=True")
    rows = []
    invalid_rows = 0
    invalid_action_rows = 0
    skipped_empty_element_rows = 0
    format_success_values: list[float] = []
    format_score_values: list[float] = []
    bucket_rows: dict[str, list[dict[str, float]]] = {bucket: [] for bucket in ACTION_BUCKETS}
    for row in read_jsonl(predictions_jsonl):
        model_output = row.get(prediction_field)
        target_markdown = row.get(target_field)
        if model_output is None or target_markdown is None:
            invalid_rows += 1
            continue
        try:
            action_type = action_type_from_row(row, action_field=action_field)
        except ValueError:
            invalid_action_rows += 1
            continue
        metrics = evaluate_prediction(str(model_output), str(target_markdown), embedder=embedder)
        format_success_values.append(float(metrics.get("format_success_rate", 0.0)))
        format_score_values.append(float(metrics.get("format_score", 0.0)))
        if metrics.get("pred_elements", 0.0) == 0.0 and metrics.get("target_elements", 0.0) == 0.0:
            skipped_empty_element_rows += 1
            continue
        rows.append(metrics)
        bucket = action_bucket_for_type(action_type)
        bucket_rows.setdefault(bucket, []).append(metrics)
    metrics = aggregate_ele_metrics(rows)
    metrics["invalid_rows"] = float(invalid_rows)
    metrics["skipped_empty_element_rows"] = float(skipped_empty_element_rows)
    metrics["format_score"] = (
        sum(format_score_values) / len(format_score_values) if format_score_values else 0.0
    )
    metrics["format_success_rate"] = (
        sum(format_success_values) / len(format_success_values) if format_success_values else 0.0
    )
    if action_breakdown:
        metrics["invalid_action_rows"] = float(invalid_action_rows)
        metrics["action_bucket_config"] = {
            bucket: list(action_types)
            for bucket, action_types in ACTION_BUCKETS.items()
        }
        metrics["by_action_bucket"] = {
            bucket: aggregate_ele_metrics(bucket_rows.get(bucket, []))
            for bucket in ACTION_BUCKETS
        }
    return metrics


def build_dry_run_report(args: argparse.Namespace) -> dict:
    missing = []
    if args.predictions_jsonl is None:
        missing.append("predictions_jsonl")
    elif not args.predictions_jsonl.is_file():
        missing.append(f"predictions_jsonl:{args.predictions_jsonl}")
    if args.output is None:
        missing.append("output")
    configuration_errors = []
    if not args.use_embedding:
        configuration_errors.append("EleF1 evaluation requires --use-embedding")
    if not args.action_breakdown:
        configuration_errors.append("EleF1 evaluation requires --action-breakdown")
    return {
        "status": "dry_run",
        "can_evaluate": not missing and not configuration_errors,
        "missing_assets": missing,
        "configuration_errors": configuration_errors,
        "predictions_jsonl": str(args.predictions_jsonl) if args.predictions_jsonl else None,
        "output": str(args.output) if args.output else None,
        "prediction_field": args.prediction_field,
        "target_field": args.target_field,
        "use_embedding": args.use_embedding,
        "embedding_device": args.embedding_device,
        "action_breakdown": args.action_breakdown,
        "action_field": args.action_field,
    }


def optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def config_defaults_for_args(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    config = load_run_config(config_path)
    return {
        "predictions_jsonl": optional_path(config.evaluation.predictions_jsonl),
        "output": optional_path(config.evaluation.output),
        "prediction_field": config.evaluation.prediction_field,
        "target_field": config.evaluation.target_field,
        "use_embedding": config.evaluation.use_embedding,
        "embedding_model": config.evaluation.embedding_model or None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Local World Model JSONL predictions.")
    parser.add_argument("--config", type=optional_path)
    parser.add_argument("--predictions-jsonl", "--predictions_jsonl", type=optional_path)
    parser.add_argument("--output", type=optional_path)
    parser.add_argument("--prediction-field", default="model_output")
    parser.add_argument("--target-field", default="target_next_md")
    parser.add_argument("--use-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-device")
    parser.add_argument("--action-breakdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-field", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    provided_dests = provided_cli_dests(parser, argv)
    args = parser.parse_args(argv)
    apply_config_defaults(args, config_defaults_for_args(args.config), provided_dests)
    if args.dry_run:
        print(json.dumps(build_dry_run_report(args), ensure_ascii=False, indent=2))
        return
    if args.predictions_jsonl is None or args.output is None:
        raise ValueError("--predictions-jsonl and --output are required unless --dry-run is set")
    if not args.use_embedding:
        raise ValueError("EleF1 evaluation requires --use-embedding")
    if not args.action_breakdown:
        raise ValueError("EleF1 evaluation requires --action-breakdown")
    embedder = (
        SemanticEmbedder(args.embedding_model, device=args.embedding_device)
        if args.use_embedding else None
    )
    metrics = evaluate_jsonl(
        args.predictions_jsonl,
        prediction_field=args.prediction_field,
        target_field=args.target_field,
        embedder=embedder,
        action_breakdown=args.action_breakdown,
        action_field=args.action_field,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
