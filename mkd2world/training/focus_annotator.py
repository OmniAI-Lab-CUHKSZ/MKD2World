from __future__ import annotations

import hashlib
import json
import math
import re
from difflib import SequenceMatcher
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from mkd2world.training.focus_parser import (
    CharSpan,
    ElementMatch,
    ParsedElementSpan,
    V4FocusError,
    bbox_iou,
    canonical_assistant_target,
    center_distance,
    normalize_text,
    parse_markdown_spans,
    _has_lexical_add_grounding,
    _is_generic_role_placeholder,
    _is_malformed_rating_fragment,
    _is_oversized_low_density_text_add,
    _is_repeated_tilde_ocr_fragment,
    _is_viewport_clipped_text_add,
)

FOCUS_LABEL_VERSION = "v4-focus-v20-hover-compact-global-support"
V12_RELIABLE_SAME_REGION_RELEASE_REASON = (
    "v12_reliable_same_region_sole_shadow_release"
)
JITTER_THRESHOLD = 8.0

FOCUS_BBOX_THRESHOLD = 16.0
STABLE_ORIGIN_ACTION_DISTANCE = 90.0
STABLE_ORIGIN_EXTENT_DRIFT_REASON = "stable_origin_extent_drift"
_ACTION_POINT_KEYS = (
    "coordinate",
    "to_coordinate",
    "from_coordinate",
    "end_coordinate",
    "start_coordinate",
)
SLOT_IOU = 0.20
SLOT_DISTANCE = 56.0
HOVER_REPLACEMENT_MAX_CONTENT_SIMILARITY = 0.30
HOVER_REPLACEMENT_MIN_GROUP_SIZE = 2
HOVER_REPLACEMENT_TYPES = frozenset({"text", "link"})
DRAG_TOGGLE_REPLACEMENT_MAX_CONTENT_SIMILARITY = 0.30
DRAG_TOGGLE_REPLACEMENT_MIN_GROUP_SIZE = 2
NONCLICK_DISCLOSURE_STATE_DRIFT_REASON = "nonclick_disclosure_state_drift"
DISCLOSURE_STATES = frozenset({"", ">", "v"})
HOVER_VIEWPORT_NORMALIZATION_BBOX_DRIFT_REASON = (
    "hover_viewport_normalization_bbox_drift"
)
HOVER_VIEWPORT_MIN_ACTION_Y = 950.0
HOVER_VIEWPORT_MIN_ELEMENT_STABILITY = 0.90
HOVER_VIEWPORT_MAX_SCALE_X_DELTA = 0.01
HOVER_VIEWPORT_MIN_SCALE_Y = 1.04
HOVER_VIEWPORT_MAX_SCALE_Y = 1.08
HOVER_VIEWPORT_MAX_RESIDUAL = 8.0
HOVER_VIEWPORT_MIN_RESIDUAL_PAIRS = 20
HOVER_VIEWPORT_MIN_Q_COVERAGE = 0.95
HOVER_VIEWPORT_RELAXED_MIN_NEXT_COVERAGE = 0.30
HOVER_VIEWPORT_RELAXED_MIN_Q_COVERAGE = 0.85
HOVER_VIEWPORT_COMPACT_MIN_RESIDUAL_PAIRS = 12
HOVER_VIEWPORT_COMPACT_MIN_Q_COVERAGE = 0.80

_TIME_OR_COUNTER = re.compile(
    r"^(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?|"
    r"\d+\s*(?:seconds?|minutes?|hours?)\s+ago|"
    r"(?:views?|likes?|watching|online)\s*[:：]?\s*[\d,.]+)$", re.I
)
_AD = re.compile(
    r"^(?:ad|ads|advertisement|advertising(?: banner)?|sponsored|promoted|"
    "\u5e7f\u544a|\u63a8\u5e7f"
    r")(?:\b|\s*[:：])",
    re.I,
)
_OCR_FRAGMENT = re.compile(r"^(?:[~|¦]{2,}|[^\w\s]{1,2})$", re.UNICODE)
_GENERIC_VISUAL = re.compile(r"^(?:image|img|icon|photo|picture|graphic|logo|thumbnail)\s*\d*$", re.I)
_MONTH = re.compile(r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\.?\s+20\d{2}$", re.I)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _element_dict(element: ParsedElementSpan) -> dict[str, Any]:
    return {
        "index": element.index,
        "line_index": element.line_index,
        "raw_line": element.raw_line,
        "content": element.content,
        "normalized_content": element.normalized_content,
        "element_type": element.element_type,
        "state": element.state,
        "bbox": list(element.bbox),
        "line_span": asdict(element.line_span),
        "content_span": asdict(element.content_span),
        "type_span": asdict(element.type_span) if element.type_span else None,
        "state_span": asdict(element.state_span) if element.state_span else None,
        "bbox_span": asdict(element.bbox_span),
    }


def _bbox_delta(left: ParsedElementSpan, right: ParsedElementSpan) -> float:
    return max(abs(a - b) for a, b in zip(left.bbox, right.bbox))


def _valid_action_points(action: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    """Return finite two-dimensional action points in protocol key order."""
    points: list[tuple[float, float]] = []
    for key in _ACTION_POINT_KEYS:
        value = action.get(key)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        try:
            x, y = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return tuple(points)


def _stable_origin_extent_drift_filter(
    left: ParsedElementSpan,
    right: ParsedElementSpan,
    *,
    action: Mapping[str, Any],
    bucket: str,
) -> dict[str, Any] | None:
    origin_delta = max(
        abs(left.bbox[0] - right.bbox[0]),
        abs(left.bbox[1] - right.bbox[1]),
    )
    extent_delta = max(
        abs(left.bbox[2] - right.bbox[2]),
        abs(left.bbox[3] - right.bbox[3]),
    )
    points = _valid_action_points(action)
    min_action_distance = min(
        (
            math.hypot(right.bbox[0] - x, right.bbox[1] - y)
            for x, y in points
        ),
        default=math.inf,
    )
    action_type = normalize_text(action.get("type"))
    eligible = (
        origin_delta <= JITTER_THRESHOLD
        and extent_delta > FOCUS_BBOX_THRESHOLD
        and left.normalized_content == right.normalized_content
        and left.element_type == right.element_type
        and left.state == right.state
        and left.element_type != "block"
        and right.element_type != "block"
        and bucket != "navigation"
        and action_type != "drag"
        and min_action_distance > STABLE_ORIGIN_ACTION_DISTANCE
    )
    if not eligible:
        return None
    return {
        "field": "bbox",
        "reason": STABLE_ORIGIN_EXTENT_DRIFT_REASON,
        "origin_max_abs_delta": origin_delta,
        "extent_max_abs_delta": extent_delta,
        "min_action_distance": (
            None if math.isinf(min_action_distance) else min_action_distance
        ),
        "bucket": bucket,
        "action_type": action_type,
        "jitter_threshold": JITTER_THRESHOLD,
        "assignment_jitter_threshold": JITTER_THRESHOLD,
        "focus_bbox_threshold": FOCUS_BBOX_THRESHOLD,
        "action_distance_threshold": STABLE_ORIGIN_ACTION_DISTANCE,
    }


def _nonclick_disclosure_state_drift_filter(
    left: ParsedElementSpan,
    right: ParsedElementSpan,
    *,
    action: Mapping[str, Any],
    bucket: str,
) -> dict[str, Any] | None:

    action_type = normalize_text(action.get("type"))
    left_state = left.state.lower()
    right_state = right.state.lower()
    eligible = (
        bucket != "click"
        and action_type != "left_click"
        and left.element_type != "select"
        and right.element_type == "select"
        and left.normalized_content == right.normalized_content
        and left_state in DISCLOSURE_STATES
        and right_state in DISCLOSURE_STATES
        and left_state != right_state
    )
    if not eligible:
        return None
    return {
        "field": "state",
        "reason": NONCLICK_DISCLOSURE_STATE_DRIFT_REASON,
        "bucket": bucket,
        "action_type": action_type,
        "left_element_type": left.element_type,
        "right_element_type": right.element_type,
        "normalized_content": right.normalized_content,
        "left_state_raw": left.state,
        "right_state_raw": right.state,
        "left_state_lower": left_state,
        "right_state_lower": right_state,
        "allowed_states": sorted(DISCLOSURE_STATES),
        "excluded_bucket": "click",
        "excluded_action_type": "left_click",
        "excluded_left_element_type": "select",
    }


def _finite_hover_coordinate(
    action: Mapping[str, Any],
) -> tuple[float, float] | None:
    value = action.get("coordinate")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _valid_image_size(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    width, height = value
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        return None
    return width, height


def _read_viewport_image_sizes(
    row: Mapping[str, Any],
    *,
    project_root: Path | None,
    image_module: Any | None,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, dict[str, Any]]:
    current_raw = str(row.get("current_image_path") or "")
    next_raw = str(row.get("next_image_path") or "")
    info: dict[str, Any] = {
        "image_size_status": "unavailable",
        "current_image_path": current_raw,
        "next_image_path": next_raw,
        "current_image_resolved_path": None,
        "next_image_resolved_path": None,
    }
    if project_root is None or image_module is None:
        info["image_size_status"] = "missing_parameters"
        return None, None, info
    if not current_raw or not next_raw:
        info["image_size_status"] = "missing_paths"
        return None, None, info

    root = Path(project_root)
    resolved_paths: dict[str, Path] = {}
    for role, raw in (("current", current_raw), ("next", next_raw)):
        path = Path(raw)
        resolved = path if path.is_absolute() else root / path
        resolved_paths[role] = resolved
        info[f"{role}_image_resolved_path"] = str(resolved)

    sizes: dict[str, tuple[int, int]] = {}
    for role in ("current", "next"):
        try:
            with image_module.open(resolved_paths[role]) as image:
                size = _valid_image_size(getattr(image, "size", None))
        except Exception:
            info["image_size_status"] = f"{role}_image_read_error"
            return None, None, info
        if size is None:
            info["image_size_status"] = f"{role}_image_invalid_size"
            return None, None, info
        sizes[role] = size

    info["image_size_status"] = "available"
    return sizes["current"], sizes["next"], info


def _hover_viewport_normalization_bbox_drift_filter(
    current: Any,
    nxt: Any,
    matches: Sequence[ElementMatch],
    *,
    action: Mapping[str, Any],
    bucket: str,
    current_size: tuple[int, int] | None,
    next_size: tuple[int, int] | None,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:

    action_type = normalize_text(action.get("type"))
    coordinate = _finite_hover_coordinate(action)
    current_count = len(current.elements)
    next_count = len(nxt.elements)
    largest_count = max(current_count, next_count)
    element_stability = (
        min(current_count, next_count) / largest_count
        if largest_count > 0
        else 0.0
    )
    validated_current_size = _valid_image_size(current_size)
    validated_next_size = _valid_image_size(next_size)
    scale_x: float | None = None
    scale_y: float | None = None
    scale_x_exact: Fraction | None = None
    scale_y_exact: Fraction | None = None
    if validated_current_size is not None and validated_next_size is not None:
        scale_x_exact = Fraction(
            validated_current_size[0], validated_next_size[0]
        )
        scale_y_exact = Fraction(
            validated_current_size[1], validated_next_size[1]
        )
        scale_x = validated_current_size[0] / validated_next_size[0]
        scale_y = validated_current_size[1] / validated_next_size[1]

    qualifying: list[dict[str, Any]] = []
    residual_pairs: list[dict[str, Any]] = []
    if scale_x is not None and scale_y is not None:
        for match in matches:
            left = current.elements[match.current_index]
            right = nxt.elements[match.next_index]
            bbox_filter = _stable_origin_extent_drift_filter(
                left,
                right,
                action=action,
                bucket=bucket,
            )
            if not (
                left.normalized_content == right.normalized_content
                and left.element_type == right.element_type
                and left.state.lower() == right.state.lower()
                and left.element_type != "block"
                and right.element_type != "block"
                and _bbox_delta(left, right) > FOCUS_BBOX_THRESHOLD
                and bbox_filter is None
            ):
                continue
            expected_next_bbox = (
                scale_x * left.bbox[0],
                scale_y * left.bbox[1],
                scale_x * left.bbox[2],
                scale_y * left.bbox[3],
            )
            residuals = tuple(
                abs(right_value - expected_value)
                for right_value, expected_value in zip(
                    right.bbox,
                    expected_next_bbox,
                )
            )
            record = {
                "current_element_index": left.index,
                "next_element_index": right.index,
                "current_bbox": list(left.bbox),
                "expected_next_bbox": list(expected_next_bbox),
                "actual_next_bbox": list(right.bbox),
                "bbox_abs_residuals": list(residuals),
                "bbox_max_abs_residual": max(residuals),
            }
            qualifying.append(record)
            if all(
                residual <= HOVER_VIEWPORT_MAX_RESIDUAL
                or math.isclose(
                    residual,
                    HOVER_VIEWPORT_MAX_RESIDUAL,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for residual in residuals
            ):
                residual_pairs.append(record)

    q_count = len(qualifying)
    r_count = len(residual_pairs)
    q_coverage = r_count / q_count if q_count else 0.0
    next_coverage = r_count / next_count if next_count else 0.0
    thresholds = {
        "action_y_min": HOVER_VIEWPORT_MIN_ACTION_Y,
        "element_stability_min": HOVER_VIEWPORT_MIN_ELEMENT_STABILITY,
        "scale_x_abs_delta_max": HOVER_VIEWPORT_MAX_SCALE_X_DELTA,
        "scale_y_min": HOVER_VIEWPORT_MIN_SCALE_Y,
        "scale_y_max": HOVER_VIEWPORT_MAX_SCALE_Y,
        "focus_bbox_threshold": FOCUS_BBOX_THRESHOLD,
        "bbox_residual_max": HOVER_VIEWPORT_MAX_RESIDUAL,
        "residual_pair_count_min": HOVER_VIEWPORT_MIN_RESIDUAL_PAIRS,
        "residual_pair_q_coverage_min": HOVER_VIEWPORT_MIN_Q_COVERAGE,
        "residual_pair_next_majority_multiplier": 2,
        "residual_pair_next_majority_strict": True,
        "relaxed_residual_pair_next_coverage_min": (
            HOVER_VIEWPORT_RELAXED_MIN_NEXT_COVERAGE
        ),
        "relaxed_residual_pair_q_coverage_min": (
            HOVER_VIEWPORT_RELAXED_MIN_Q_COVERAGE
        ),
        "compact_residual_pair_count_min": (
            HOVER_VIEWPORT_COMPACT_MIN_RESIDUAL_PAIRS
        ),
        "compact_residual_pair_q_coverage_min": (
            HOVER_VIEWPORT_COMPACT_MIN_Q_COVERAGE
        ),
    }
    precondition_checks = (
        (bucket == "other", "bucket_not_other"),
        (action_type == "hover", "action_type_not_hover"),
        (coordinate is not None, "coordinate_not_finite"),
        (
            coordinate is not None
            and coordinate[1] >= HOVER_VIEWPORT_MIN_ACTION_Y,
            "coordinate_y_below_threshold",
        ),
        (
            current_count > 0 and next_count > 0,
            "empty_element_sequence",
        ),
        (
            largest_count > 0
            and Fraction(min(current_count, next_count), largest_count)
            >= Fraction(str(HOVER_VIEWPORT_MIN_ELEMENT_STABILITY)),
            "element_count_stability_below_threshold",
        ),
        (
            scale_x is not None and scale_y is not None,
            "image_sizes_unavailable",
        ),
        (
            scale_x_exact is not None
            and abs(scale_x_exact - 1)
            <= Fraction(str(HOVER_VIEWPORT_MAX_SCALE_X_DELTA)),
            "scale_x_out_of_range",
        ),
        (
            scale_y_exact is not None
            and Fraction(str(HOVER_VIEWPORT_MIN_SCALE_Y))
            <= scale_y_exact
            <= Fraction(str(HOVER_VIEWPORT_MAX_SCALE_Y)),
            "scale_y_out_of_range",
        ),
    )
    strong_checks = (
        (
            r_count >= HOVER_VIEWPORT_MIN_RESIDUAL_PAIRS,
            "residual_pair_count_below_threshold",
        ),
        (
            2 * r_count > next_count,
            "residual_pairs_not_strict_next_majority",
        ),
        (
            q_count > 0
            and Fraction(r_count, q_count)
            >= Fraction(str(HOVER_VIEWPORT_MIN_Q_COVERAGE)),
            "residual_pair_q_coverage_below_threshold",
        ),
    )
    relaxed_checks = (
        (
            r_count >= HOVER_VIEWPORT_MIN_RESIDUAL_PAIRS,
            "relaxed_residual_pair_count_below_threshold",
        ),
        (
            next_count > 0
            and Fraction(r_count, next_count)
            >= Fraction(str(HOVER_VIEWPORT_RELAXED_MIN_NEXT_COVERAGE)),
            "relaxed_residual_pair_next_coverage_below_threshold",
        ),
        (
            q_count > 0
            and Fraction(r_count, q_count)
            >= Fraction(str(HOVER_VIEWPORT_RELAXED_MIN_Q_COVERAGE)),
            "relaxed_residual_pair_q_coverage_below_threshold",
        ),
    )
    compact_checks = (
        (
            r_count >= HOVER_VIEWPORT_COMPACT_MIN_RESIDUAL_PAIRS,
            "compact_residual_pair_count_below_threshold",
        ),
        (
            q_count > 0
            and Fraction(r_count, q_count)
            >= Fraction(str(HOVER_VIEWPORT_COMPACT_MIN_Q_COVERAGE)),
            "compact_residual_pair_q_coverage_below_threshold",
        ),
    )
    precondition_failed_conditions = [
        reason for branch_passed, reason in precondition_checks
        if not branch_passed
    ]
    strong_support_failed_conditions = [
        reason for branch_passed, reason in strong_checks
        if not branch_passed
    ]
    relaxed_support_failed_conditions = [
        reason for branch_passed, reason in relaxed_checks
        if not branch_passed
    ]
    compact_support_failed_conditions = [
        reason for branch_passed, reason in compact_checks
        if not branch_passed
    ]
    preconditions_passed = not precondition_failed_conditions
    strong_passed = (
        preconditions_passed and not strong_support_failed_conditions
    )
    relaxed_passed = (
        preconditions_passed and not relaxed_support_failed_conditions
    )
    compact_passed = (
        preconditions_passed and not compact_support_failed_conditions
    )
    passed = strong_passed or relaxed_passed or compact_passed
    selected_branch = (
        "strong"
        if strong_passed
        else "relaxed"
        if relaxed_passed
        else "compact"
        if compact_passed
        else None
    )
    failed_conditions = list(precondition_failed_conditions)
    if preconditions_passed and not passed:
        failed_conditions.append("no_support_branch_passed")
    strong_branch = {
        "decision": "pass" if strong_passed else "fail",
        "passed": strong_passed,
        "support_passed": not strong_support_failed_conditions,
        "failed_conditions": [
            *precondition_failed_conditions,
            *strong_support_failed_conditions,
        ],
        "thresholds": {
            "residual_pair_count_min": HOVER_VIEWPORT_MIN_RESIDUAL_PAIRS,
            "residual_pair_next_majority_multiplier": 2,
            "residual_pair_next_majority_strict": True,
            "residual_pair_q_coverage_min": HOVER_VIEWPORT_MIN_Q_COVERAGE,
        },
    }
    relaxed_branch = {
        "decision": "pass" if relaxed_passed else "fail",
        "passed": relaxed_passed,
        "support_passed": not relaxed_support_failed_conditions,
        "failed_conditions": [
            *precondition_failed_conditions,
            *relaxed_support_failed_conditions,
        ],
        "thresholds": {
            "residual_pair_count_min": HOVER_VIEWPORT_MIN_RESIDUAL_PAIRS,
            "residual_pair_next_coverage_min": (
                HOVER_VIEWPORT_RELAXED_MIN_NEXT_COVERAGE
            ),
            "residual_pair_q_coverage_min": (
                HOVER_VIEWPORT_RELAXED_MIN_Q_COVERAGE
            ),
        },
    }
    compact_branch = {
        "decision": "pass" if compact_passed else "fail",
        "passed": compact_passed,
        "support_passed": not compact_support_failed_conditions,
        "failed_conditions": [
            *precondition_failed_conditions,
            *compact_support_failed_conditions,
        ],
        "thresholds": {
            "residual_pair_count_min": (
                HOVER_VIEWPORT_COMPACT_MIN_RESIDUAL_PAIRS
            ),
            "residual_pair_q_coverage_min": (
                HOVER_VIEWPORT_COMPACT_MIN_Q_COVERAGE
            ),
        },
    }
    diagnostic: dict[str, Any] = {
        "reason": HOVER_VIEWPORT_NORMALIZATION_BBOX_DRIFT_REASON,
        "decision": "filter_bbox" if passed else "no_op",
        "selected_branch": selected_branch,
        "strong_branch": strong_branch,
        "relaxed_branch": relaxed_branch,
        "compact_branch": compact_branch,
        "failed_conditions": failed_conditions,
        "bucket": bucket,
        "action_type": action_type,
        "action_coordinate": list(coordinate) if coordinate is not None else None,
        "current_image_size": (
            list(validated_current_size)
            if validated_current_size is not None
            else None
        ),
        "next_image_size": (
            list(validated_next_size)
            if validated_next_size is not None
            else None
        ),
        "scale_x_current_over_next": scale_x,
        "scale_y_current_over_next": scale_y,
        "transform_direction": "current_bbox_to_expected_next_bbox",
        "current_element_count": current_count,
        "next_element_count": next_count,
        "element_count_stability": element_stability,
        "qualifying_pair_count": q_count,
        "residual_pair_count": r_count,
        "residual_pair_next_coverage": next_coverage,
        "residual_pair_q_coverage": q_coverage,
        "twice_residual_pair_count": 2 * r_count,
        "thresholds": thresholds,
        "filtered_pair_indices": (
            [
                [
                    int(record["current_element_index"]),
                    int(record["next_element_index"]),
                ]
                for record in residual_pairs
            ]
            if passed
            else []
        ),
    }
    if not passed:
        return {}, diagnostic

    shared_evidence = {
        key: diagnostic[key]
        for key in (
            "current_image_size",
            "next_image_size",
            "scale_x_current_over_next",
            "scale_y_current_over_next",
            "transform_direction",
            "current_element_count",
            "next_element_count",
            "element_count_stability",
            "qualifying_pair_count",
            "residual_pair_count",
            "residual_pair_next_coverage",
            "residual_pair_q_coverage",
            "twice_residual_pair_count",
            "selected_branch",
            "strong_branch",
            "relaxed_branch",
            "compact_branch",
            "thresholds",
        )
    }
    filtered_pairs = {
        (
            int(record["current_element_index"]),
            int(record["next_element_index"]),
        ): {
            "reason": HOVER_VIEWPORT_NORMALIZATION_BBOX_DRIFT_REASON,
            **shared_evidence,
            **{
                key: value
                for key, value in record.items()
                if key not in {"current_element_index", "next_element_index"}
            },
        }
        for record in residual_pairs
    }
    return filtered_pairs, diagnostic


def _bind_viewport_image_provenance(
    filtered_pairs: Mapping[tuple[int, int], Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
    image_info: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    path_evidence = {
        key: image_info.get(key)
        for key in (
            "image_size_status",
            "current_image_path",
            "next_image_path",
            "current_image_resolved_path",
            "next_image_resolved_path",
        )
    }
    return (
        {
            pair: {**dict(evidence), **path_evidence}
            for pair, evidence in filtered_pairs.items()
        },
        {**dict(diagnostic), **path_evidence},
    )


def _size_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    lw, lh, rw, rh = left[2], left[3], right[2], right[3]
    return min(lw, rw) / max(1.0, max(lw, rw)) * min(lh, rh) / max(1.0, max(lh, rh))


def _slot_score(left: ParsedElementSpan, right: ParsedElementSpan) -> tuple[float, dict[str, float]]:
    iou = bbox_iou(left.bbox, right.bbox)
    distance = center_distance(left.bbox, right.bbox)
    size = _size_ratio(left.bbox, right.bbox)
    edge = max(abs(left.bbox[k] - right.bbox[k]) for k in (0, 1))
    type_bonus = 0.12 if left.element_type == right.element_type else 0.0
    score = 3.0 * iou + 1.5 * max(0.0, 1.0 - distance / 120.0) + 0.5 * size + type_bonus
    return score, {"iou": round(iou, 6), "center_distance": round(distance, 6), "size_ratio": round(size, 6), "center_edge_delta": round(edge, 6)}


def _strong_slot(left: ParsedElementSpan, right: ParsedElementSpan) -> bool:
    score, metrics = _slot_score(left, right)
    distance = metrics["center_distance"]
    # Adjacent calendar/list cells can be close: require overlap, or very tight
    # centers with compatible size.  Content is deliberately not required.
    return metrics["iou"] >= SLOT_IOU or (distance <= SLOT_DISTANCE and metrics["size_ratio"] >= 0.35 and score >= 1.35)


def _tier_a_slot(left: ParsedElementSpan, right: ParsedElementSpan) -> bool:
    delta = _bbox_delta(left, right)
    _, metrics = _slot_score(left, right)
    return delta <= JITTER_THRESHOLD or (metrics["iou"] >= 0.70 and metrics["size_ratio"] >= 0.55 and metrics["center_distance"] <= 24.0)


def _hover_replacement_identity_guard(
    left: ParsedElementSpan,
    right: ParsedElementSpan,
    *,
    action: Mapping[str, Any],
    bucket: str,
) -> dict[str, Any] | None:
    """Reject low-similarity text/link slot claims created by hover menus."""
    action_type = normalize_text(action.get("type"))
    content_similarity = SequenceMatcher(
        None,
        left.normalized_content,
        right.normalized_content,
    ).ratio()
    eligible = (
        bucket == "other"
        and action_type == "hover"
        and left.element_type in HOVER_REPLACEMENT_TYPES
        and right.element_type in HOVER_REPLACEMENT_TYPES
        and left.normalized_content != right.normalized_content
        and content_similarity < HOVER_REPLACEMENT_MAX_CONTENT_SIMILARITY
    )
    if not eligible:
        return None
    return {
        "reason": "hover_low_similarity_replacement",
        "bucket": bucket,
        "action_type": action_type,
        "left_type": left.element_type,
        "right_type": right.element_type,
        "content_similarity": content_similarity,
        "similarity_threshold": HOVER_REPLACEMENT_MAX_CONTENT_SIMILARITY,
    }


def _drag_toggle_replacement_identity_guard(
    left: ParsedElementSpan,
    right: ParsedElementSpan,
    *,
    action: Mapping[str, Any],
    bucket: str,
) -> dict[str, Any] | None:
    """Reject low-similarity toggle slot claims created by drag lists."""
    action_type = normalize_text(action.get("type"))
    content_similarity = SequenceMatcher(
        None,
        left.normalized_content,
        right.normalized_content,
    ).ratio()
    eligible = (
        bucket == "other"
        and action_type == "drag"
        and left.element_type == "toggle"
        and right.element_type == "toggle"
        and left.normalized_content != right.normalized_content
        and content_similarity
        < DRAG_TOGGLE_REPLACEMENT_MAX_CONTENT_SIMILARITY
    )
    if not eligible:
        return None
    return {
        "reason": "drag_low_similarity_toggle_replacement",
        "bucket": bucket,
        "action_type": action_type,
        "left_type": left.element_type,
        "right_type": right.element_type,
        "content_similarity": content_similarity,
        "similarity_threshold": (
            DRAG_TOGGLE_REPLACEMENT_MAX_CONTENT_SIMILARITY
        ),
    }


def _point_distance(element: ParsedElementSpan, point: Sequence[float] | None) -> float:
    if not point or len(point) < 2: return math.inf
    return math.hypot(element.bbox[0]-float(point[0]), element.bbox[1]-float(point[1]))


def _slot_change_supported(left: ParsedElementSpan, right: ParsedElementSpan, *, action: Mapping[str, Any], bucket: str) -> bool:
    if _tier_a_slot(left, right):
        return True
    if left.semantic_key == right.semantic_key:
        return True
    same_type = left.element_type == right.element_type
    if same_type and left.element_type in {"input", "select", "toggle"} and _size_ratio(left.bbox, right.bbox) >= 0.55:
        return True
    if same_type and _MONTH.fullmatch(left.content.strip()) and _MONTH.fullmatch(right.content.strip()):
        return True
    if same_type and SequenceMatcher(None, left.normalized_content, right.normalized_content).ratio() >= 0.78:
        return True
    if left.state.lower() != right.state.lower() and same_type:
        return True
    return False


def assign_elements(current, nxt, *, action: Mapping[str, Any] | None = None, bucket: str = "") -> tuple[tuple[ElementMatch, ...], list[dict[str, Any]]]:
    """Deterministic unique one-to-one assignment with strict slot priority."""
    action = action or {}
    used_c: set[int] = set()
    used_n: set[int] = set()
    chosen: list[ElementMatch] = []
    provenance: list[dict[str, Any]] = []

    def take(ci: int, ni: int, phase: str, score: float, metrics: Mapping[str, float]) -> None:
        used_c.add(ci); used_n.add(ni)
        left, right = current.elements[ci], nxt.elements[ni]
        similarity = 1.0 if left.normalized_content == right.normalized_content else 0.0
        chosen.append(ElementMatch(ci, ni, phase, min(0.999, max(0.5, score / 4.5)), metrics.get("iou", 0.0), metrics.get("center_distance", center_distance(left.bbox, right.bbox)), similarity))
        provenance.append({"assignment_id": len(chosen)-1, "current_element_index": ci, "next_element_index": ni, "phase": phase, "score": round(score, 6), **dict(metrics)})

    # Phase 1: physical UI slot. This deliberately wins over a remote exact text.
    current_key_counts: dict[tuple[str,str,str],int] = {}
    next_key_counts: dict[tuple[str,str,str],int] = {}
    for e in current.elements: current_key_counts[e.semantic_key]=current_key_counts.get(e.semantic_key,0)+1
    for e in nxt.elements: next_key_counts[e.semantic_key]=next_key_counts.get(e.semantic_key,0)+1
    typed = normalize_text(action.get("text")) if normalize_text(action.get("type")) == "type_text" else ""
    if typed:
        placeholder_re=re.compile(r"(?:search|query|find|where|type|enter|location|destination|what can we help)",re.I)
        typed_pairs=[]
        for right in nxt.elements:
            if right.normalized_content != typed: continue
            for left in current.elements:
                if left.semantic_key == right.semantic_key or left.normalized_content == typed: continue
                score,metrics=_slot_score(left,right)
                role=int(left.element_type in {"input","select","toggle"})
                placeholder=int(left.element_type not in {"image","icon"} and bool(placeholder_re.search(left.normalized_content)))
                tier_a=int(_tier_a_slot(left,right))
                left_echo=re.sub(r"^(?:img|image|icon)\s*:\s*","",left.normalized_content)
                visual_echo=int(left.element_type in {"image","icon"} and SequenceMatcher(None,left_echo,typed).ratio() >= 0.70)
                if tier_a or ((role or placeholder or visual_echo) and metrics["center_distance"] <= 80.0 and metrics["size_ratio"] >= 0.20):
                    typed_pairs.append((-role,-visual_echo,-placeholder,-tier_a,-score,metrics["center_distance"],left.index,right.index,metrics))
        if typed_pairs:
            _,_,_,_,neg,_,ci,ni,metrics=min(typed_pairs)
            take(ci,ni,"typed_control_reservation",-neg,metrics)
    anchor_deltas=[]
    anchor_records=[]
    for left in current.elements:
        if current_key_counts[left.semantic_key] != 1 or next_key_counts.get(left.semantic_key) != 1: continue
        right=next(e for e in nxt.elements if e.semantic_key==left.semantic_key)
        delta=(right.bbox[0]-left.bbox[0],right.bbox[1]-left.bbox[1])
        anchor_deltas.append(delta); anchor_records.append((left,delta))
    motion: tuple[float,float] | None = None
    if len(anchor_deltas) >= 2:
        bins: dict[tuple[int,int],list[tuple[float,float]]] = {}
        for dx,dy in anchor_deltas: bins.setdefault((round(dx/8),round(dy/8)),[]).append((dx,dy))
        cluster=max(bins.values(),key=len)
        if len(cluster) >= 2: motion=(sorted(x for x,_ in cluster)[len(cluster)//2],sorted(y for _,y in cluster)[len(cluster)//2])
    def local_motion_for(element: ParsedElementSpan) -> tuple[float,float] | None:
        local_bins: dict[tuple[int,int],list[tuple[float,float]]] = {}
        for anchor,(dx,dy) in anchor_records:
            if anchor.index == element.index or math.hypot(dx,dy) <= JITTER_THRESHOLD or center_distance(element.bbox,anchor.bbox) > 300.0: continue
            local_bins.setdefault((round(dx/8),round(dy/8)),[]).append((dx,dy))
        if not local_bins: return None
        local_cluster=max(local_bins.values(),key=len)
        if len(local_cluster) < 2: return None
        return (sorted(x for x,_ in local_cluster)[len(local_cluster)//2],sorted(y for _,y in local_cluster)[len(local_cluster)//2])

    for left in current.elements:
        if left.index in used_c or current_key_counts[left.semantic_key] != 1 or next_key_counts.get(left.semantic_key) != 1: continue
        right=next(e for e in nxt.elements if e.semantic_key==left.semantic_key)
        if right.index in used_n: continue
        local_motion=local_motion_for(left)
        dx,dy=right.bbox[0]-left.bbox[0],right.bbox[1]-left.bbox[1]
        if local_motion is not None and math.hypot(dx,dy) > JITTER_THRESHOLD and math.hypot(dx-local_motion[0],dy-local_motion[1]) <= 8.0:
            score,metrics=_slot_score(left,right);metrics={**metrics,"dominant_dx":local_motion[0],"dominant_dy":local_motion[1],"translation_residual":math.hypot(dx-local_motion[0],dy-local_motion[1])}
            take(left.index,right.index,"local_coherent_exact_reservation",max(2.0,score),metrics)
    slot_candidates = []
    for left in current.elements:
        if left.index in used_c: continue
        local_motion=local_motion_for(left)
        for right in nxt.elements:
            if right.index in used_n: continue
            duplicate_exact = left.semantic_key == right.semantic_key and (current_key_counts[left.semantic_key] > 1 or next_key_counts[right.semantic_key] > 1)
            preserves_moving_exact = bool(left.semantic_key != right.semantic_key and local_motion is not None and any(candidate.semantic_key == left.semantic_key and math.hypot(candidate.bbox[0]-left.bbox[0]-local_motion[0],candidate.bbox[1]-left.bbox[1]-local_motion[1]) <= 8.0 for candidate in nxt.elements))
            if preserves_moving_exact: continue
            if _strong_slot(left, right) and (not duplicate_exact or _tier_a_slot(left,right)) and _slot_change_supported(left, right, action=action, bucket=bucket):
                score, metrics = _slot_score(left, right)
                slot_candidates.append((-score, metrics["center_distance"], left.index, right.index, metrics))
    best_for_c: dict[int, tuple[float, int]] = {}
    best_for_n: dict[int, tuple[float, int]] = {}
    for neg, _, ci, ni, _ in slot_candidates:
        rank = -neg
        if ci not in best_for_c or rank > best_for_c[ci][0]: best_for_c[ci] = (rank, ni)
        if ni not in best_for_n or rank > best_for_n[ni][0]: best_for_n[ni] = (rank, ci)
    for neg, _, ci, ni, metrics in sorted(slot_candidates):
        # Changed-content slot claims require mutual geometry support. This
        # prevents adjacent compact controls from being consumed greedily.
        if best_for_c.get(ci, (None, None))[1] != ni or best_for_n.get(ni, (None, None))[1] != ci:
            continue
        if ci not in used_c and ni not in used_n:
            take(ci, ni, "slot_first", -neg, metrics)

    exact = []
    for left in current.elements:
        if left.index in used_c: continue
        for right in nxt.elements:
            if right.index in used_n: continue
            if left.semantic_key == right.semantic_key:
                distance = center_distance(left.bbox, right.bbox)
                residual = math.hypot((right.bbox[0]-left.bbox[0])-(motion[0] if motion else 0.0),(right.bbox[1]-left.bbox[1])-(motion[1] if motion else 0.0)) if motion else distance
                exact.append((residual, distance, left.index, right.index))
    for residual, distance, ci, ni in sorted(exact):
        if ci not in used_c and ni not in used_n:
            take(ci, ni, "exact_semantic_reflow", max(2.0, 4.0 - residual / 500.0), {"iou": bbox_iou(current.elements[ci].bbox, nxt.elements[ni].bbox), "center_distance": distance, "translation_residual": residual, "dominant_dx": motion[0] if motion else 0.0, "dominant_dy": motion[1] if motion else 0.0})


    hover_replacement_pairs = {
        (match.current_index, match.next_index)
        for match in chosen
        if match.method == "slot_first"
        and _hover_replacement_identity_guard(
            current.elements[match.current_index],
            nxt.elements[match.next_index],
            action=action,
            bucket=bucket,
        )
        is not None
    }
    if len(hover_replacement_pairs) >= HOVER_REPLACEMENT_MIN_GROUP_SIZE:
        chosen = [
            match
            for match in chosen
            if (match.current_index, match.next_index)
            not in hover_replacement_pairs
        ]
        provenance = [
            {
                **item,
                "assignment_id": assignment_id,
            }
            for assignment_id, item in enumerate(
                item
                for item in provenance
                if (
                    int(item["current_element_index"]),
                    int(item["next_element_index"]),
                )
                not in hover_replacement_pairs
            )
        ]

    drag_toggle_replacement_pairs = {
        (match.current_index, match.next_index)
        for match in chosen
        if match.method == "slot_first"
        and _drag_toggle_replacement_identity_guard(
            current.elements[match.current_index],
            nxt.elements[match.next_index],
            action=action,
            bucket=bucket,
        )
        is not None
    }
    if (
        len(drag_toggle_replacement_pairs)
        >= DRAG_TOGGLE_REPLACEMENT_MIN_GROUP_SIZE
    ):
        chosen = [
            match
            for match in chosen
            if (match.current_index, match.next_index)
            not in drag_toggle_replacement_pairs
        ]
        provenance = [
            {
                **item,
                "assignment_id": assignment_id,
            }
            for assignment_id, item in enumerate(
                item
                for item in provenance
                if (
                    int(item["current_element_index"]),
                    int(item["next_element_index"]),
                )
                not in drag_toggle_replacement_pairs
            )
        ]

    return tuple(chosen), provenance


def _unpredictable_content(element: ParsedElementSpan) -> tuple[bool, str | None]:
    value = normalize_text(element.content)
    if not value:
        return True, "empty_content"
    if _TIME_OR_COUNTER.fullmatch(value):
        return True, "timestamp_or_live_counter"
    if _AD.match(value):
        return True, "advertising_field"
    if _OCR_FRAGMENT.fullmatch(value):
        return True, "ocr_fragment"
    if element.element_type in {"image", "icon"} and _GENERIC_VISUAL.fullmatch(value):
        return True, "generic_visual_without_semantic_description"
    return False, None


def _span_payload(span: CharSpan, target: str, field: str) -> dict[str, Any]:
    text = target[span.start:span.end]
    return {"field": field, "start": span.start, "end": span.end, "text": text, "text_sha256": _sha(text)}


def _alnum_key(value: str) -> str:
    return "".join(char.lower() for char in normalize_text(value) if char.isalnum())


def _changed_fields(
    left: ParsedElementSpan,
    right: ParsedElementSpan,
    *,
    action: Mapping[str, Any],
    bucket: str,
    viewport_bbox_filter: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    fields: list[str] = []
    filtered: list[dict[str, Any]] = []
    noisy, reason = _unpredictable_content(right)
    if left.normalized_content != right.normalized_content:
        if noisy:
            filtered.append({"field": "content", "reason": reason})
        else:
            fields.append("content")
    if left.element_type != right.element_type:
        filtered.append({"field": "type", "reason": "type_base_only"})
    if left.state.lower() != right.state.lower():
        fields.append("state")
    state_filter = _nonclick_disclosure_state_drift_filter(
        left,
        right,
        action=action,
        bucket=bucket,
    )
    if "state" in fields and state_filter is not None:
        fields.remove("state")
        filtered.append(state_filter)
    if _bbox_delta(left, right) > FOCUS_BBOX_THRESHOLD:
        fields.append("bbox")
    bbox_filter = _stable_origin_extent_drift_filter(
        left,
        right,
        action=action,
        bucket=bucket,
    )
    if "bbox" in fields and bbox_filter is not None:
        fields.remove("bbox")
        filtered.append(bbox_filter)
    if "bbox" in fields and viewport_bbox_filter is not None:
        fields.remove("bbox")
        filtered.append({"field": "bbox", **dict(viewport_bbox_filter)})

    if left.element_type == "block" or right.element_type == "block":
        for field in ("content", "state"):
            if field in fields:
                fields.remove(field)
                filtered.append({"field": field, "reason": "block_semantic_base_only"})

    if "content" in fields:
        left_alnum = _alnum_key(left.content)
        right_alnum = _alnum_key(right.content)
        if left_alnum == right_alnum:
            fields.remove("content")
            filtered.append({"field": "content", "reason": "alnum_equivalent"})
        elif (
            not _content_action_causal(left, right, action=action, bucket=bucket)
            and _small_edit_distance(left_alnum, right_alnum, limit=2)
            and SequenceMatcher(None, left_alnum, right_alnum).ratio() >= 0.80
        ):
            fields.remove("content")
            filtered.append({"field": "content", "reason": "minor_ocr_edit"})
    return fields, filtered


def _small_edit_distance(left: str, right: str, *, limit: int) -> bool:
    if abs(len(left)-len(right)) > limit: return False
    previous = list(range(len(right)+1))
    for i, lc in enumerate(left, 1):
        current = [i]
        for j, rc in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(lc != rc)))
        if min(current) > limit: return False
        previous = current
    return previous[-1] <= limit


def _content_action_causal(left: ParsedElementSpan, right: ParsedElementSpan, *, action: Mapping[str, Any], bucket: str) -> bool:
    typed = normalize_text(action.get("text")) if normalize_text(action.get("type")) == "type_text" else ""
    if typed and (right.normalized_content == typed or typed in right.normalized_content): return True
    if _MONTH.fullmatch(left.content.strip()) and _MONTH.fullmatch(right.content.strip()): return True
    if left.state.lower() != right.state.lower() or (left.normalized_content, right.normalized_content) in {("details","close"),("close","details"),("off","on"),("on","off")}:
        return True
    point = action.get("coordinate") if isinstance(action.get("coordinate"), (list, tuple)) else None
    if bucket == "click" and _point_distance(right, point) <= 90: return True
    return False


def _reliable_add(element: ParsedElementSpan) -> tuple[bool, str | None]:
    noisy, reason = _unpredictable_content(element)
    if noisy:
        return False, reason
    if element.element_type == "block":
        semantic = re.sub(r"^block\s*:\s*", "", element.normalized_content)
        if semantic not in {"navigation", "header", "menu", "toolbar", "dialog"}:
            return False, "structural_block_without_specific_semantics"
    if _is_generic_role_placeholder(element):
        return False, "generic_role_placeholder"
    if element.element_type == "text" and not _has_lexical_add_grounding(element):
        return False, "text_without_lexical_grounding"
    if _is_oversized_low_density_text_add(element) or _is_viewport_clipped_text_add(element) or _is_malformed_rating_fragment(element) or _is_repeated_tilde_ocr_fragment(element):
        return False, "renderer_or_ocr_fragment_geometry"
    return True, None


def _dominant_motion(matches: Sequence[ElementMatch], current, nxt) -> float | None:
    deltas = [nxt.elements[m.next_index].bbox[1]-current.elements[m.current_index].bbox[1] for m in matches if current.elements[m.current_index].semantic_key == nxt.elements[m.next_index].semantic_key]
    if len(deltas) < 3: return None
    ordered=sorted(deltas); median=ordered[len(ordered)//2]
    inliers=[d for d in deltas if abs(d-median) <= 8.0]
    return float(median) if len(inliers) >= max(3, math.ceil(0.6*len(deltas))) else None


def _candidate_local_shadow_reasons(
    element: ParsedElementSpan,
    candidate: ParsedElementSpan,
    *,
    motion: float | None,
) -> tuple[str, ...]:
    iou=bbox_iou(candidate.bbox, element.bbox); distance=center_distance(candidate.bbox, element.bbox)
    diagonal=math.hypot(element.bbox[2],element.bbox[3])
    same_region=iou >= 0.15 and distance <= max(20.0,0.5*diagonal)
    similarity=SequenceMatcher(None,candidate.normalized_content,element.normalized_content).ratio()
    reasons: list[str] = []
    if same_region and candidate.element_type == element.element_type:
        reasons.append("same_region_current_shadow")
    if same_region and (similarity >= 0.55 or candidate.normalized_content in element.normalized_content or element.normalized_content in candidate.normalized_content):
        reasons.append("same_region_split_merge_shadow")
    if motion is not None:
        dx=element.bbox[0]-candidate.bbox[0]; dy=element.bbox[1]-candidate.bbox[1]
        if abs(dx) <= max(24.0,0.5*max(candidate.bbox[2],element.bbox[2])) and abs(dy-motion) <= 8.0 and (candidate.semantic_key == element.semantic_key or similarity >= 0.55):
            reasons.append("coherent_motion_current_shadow")
    return tuple(reasons)


def _local_shadow_reasons(
    element: ParsedElementSpan,
    current: Sequence[ParsedElementSpan],
    *,
    motion: float | None,
) -> tuple[str, ...]:
    return tuple(
        reason
        for candidate in current
        for reason in _candidate_local_shadow_reasons(
            element,
            candidate,
            motion=motion,
        )
    )


def _local_shadow_reason(element: ParsedElementSpan, current: Sequence[ParsedElementSpan], *, motion: float | None) -> str | None:
    reasons = _local_shadow_reasons(element, current, motion=motion)
    return reasons[0] if reasons else None


def _v12_reliable_same_region_shadow_decision(
    element: ParsedElementSpan,
    current: Sequence[ParsedElementSpan],
    *,
    reliable_add: bool,
    motion: float | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Release only when every observed shadow has the accepted same-region type."""
    shadow_reasons = _local_shadow_reasons(element, current, motion=motion)
    original_reason = shadow_reasons[0] if shadow_reasons else None
    unique_reasons = set(shadow_reasons)
    if (
        not reliable_add
        or unique_reasons != {"same_region_current_shadow"}
    ):
        return original_reason, None
    evidence = {
        "reason": V12_RELIABLE_SAME_REGION_RELEASE_REASON,
        "released_shadow_reason": "same_region_current_shadow",
        "all_shadow_reasons": list(shadow_reasons),
        "unique_shadow_reasons": sorted(unique_reasons),
    }
    return None, evidence


def _add_allowed(element: ParsedElementSpan, *, bucket: str, action: Mapping[str, Any], nxt, page_replace: bool) -> tuple[bool, str]:
    if bucket == "navigation":
        return True, "navigation_reliable_page_element"
    if bucket == "scroll":
        try: delta = float(action.get("delta_y", 0.0))
        except (TypeError, ValueError): delta = 0.0
        y = element.bbox[1]
        if (delta > 0 and y >= 680) or (delta < 0 and y <= 320):
            return True, "directional_scroll_viewport_entry"
        return False, "scroll_add_not_at_directional_entry_edge"
    if bucket == "click":
        point = action.get("coordinate") if isinstance(action.get("coordinate"), (list, tuple)) else None
        if page_replace or _point_distance(element, point) <= 350:
            return True, "click_local_or_page_replace_add"
        return False, "click_add_outside_causal_radius"
    if bucket == "text_key":
        if page_replace:
            return True, "text_key_page_replace_add"
        typed = normalize_text(action.get("text")) if normalize_text(action.get("type")) == "type_text" else ""
        anchors = [e for e in nxt.elements if typed and (typed in e.normalized_content or e.normalized_content in typed)]
        if anchors and min(center_distance(element.bbox, anchor.bbox) for anchor in anchors) <= 350:
            return True, "typed_anchor_local_add"
        return False, "text_key_add_without_local_typed_anchor"
    point = None
    for key in ("coordinate", "to_coordinate", "from_coordinate", "end_coordinate", "start_coordinate"):
        value = action.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            point = value if point is None or _point_distance(element, value) < _point_distance(element, point) else point
    action_type = normalize_text(action.get("type"))
    radius = 250 if action_type == "hover" else 350 if action_type == "drag" else 0
    if radius and _point_distance(element, point) <= radius:
        return True, f"{action_type}_endpoint_local_add"
    return False, "other_add_without_local_action_evidence"


def build_focus_annotation(row: dict[str, Any], *, project_root: Path | None = None, image_module: Any | None = None) -> dict[str, Any]:
    sample_id = str(row.get("id") or "").strip()
    if not sample_id:
        raise V4FocusError("sample is missing id")
    if row.get("transition_class") != "changed":
        raise V4FocusError(f"sample {sample_id} is not transition_class=changed")
    bucket = str(row.get("action_bucket") or "").strip()
    if bucket not in {"click", "text_key", "scroll", "navigation", "other"}:
        raise V4FocusError(f"invalid action_bucket={bucket!r}")
    current = parse_markdown_spans(row.get("current_md"))
    nxt = parse_markdown_spans(row.get("target_next_md"))
    action = row.get("candidate_action") if isinstance(row.get("candidate_action"), dict) else {}
    if not action and isinstance(row.get("candidate_action_json"), str):
        try:
            parsed_action = json.loads(row["candidate_action_json"])
            action = parsed_action if isinstance(parsed_action, dict) else {}
        except json.JSONDecodeError:
            action = {}
    matches, assignment_provenance = assign_elements(current, nxt, action=action, bucket=bucket)
    current_size, next_size, image_info = _read_viewport_image_sizes(
        row,
        project_root=project_root,
        image_module=image_module,
    )
    viewport_bbox_filters, viewport_diagnostic = (
        _hover_viewport_normalization_bbox_drift_filter(
            current,
            nxt,
            matches,
            action=action,
            bucket=bucket,
            current_size=current_size,
            next_size=next_size,
        )
    )
    viewport_bbox_filters, viewport_diagnostic = (
        _bind_viewport_image_provenance(
            viewport_bbox_filters,
            viewport_diagnostic,
            image_info,
        )
    )
    provenance_by_pair = {(p["current_element_index"], p["next_element_index"]): p for p in assignment_provenance}
    by_next = {m.next_index: m for m in matches}
    matched_current = {m.current_index for m in matches}
    focus_units: list[dict[str, Any]] = []
    next_labels: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    filtered_fields: list[dict[str, Any]] = []
    retention = len(matches) / max(1, min(len(current.elements), len(nxt.elements)))
    page_replace = retention < 0.20
    dominant_motion = _dominant_motion(matches, current, nxt)

    for right in nxt.elements:
        match = by_next.get(right.index)
        if match is None:
            reliable, exclusion = _reliable_add(right)
            shadow_reason, shadow_release = _v12_reliable_same_region_shadow_decision(
                right,
                current.elements,
                reliable_add=reliable,
                motion=dominant_motion,
            )
            action_supported, action_reason = _add_allowed(right, bucket=bucket, action=action, nxt=nxt, page_replace=page_replace)
            if reliable and shadow_reason is None:
                reasons = ["reliable_unmatched_next_element"]
                if shadow_release is not None:
                    reasons.append(V12_RELIABLE_SAME_REGION_RELEASE_REASON)
                unit = {"focus_id": len(focus_units), "kind": "ADD", "next_element_index": right.index, "current_element_index": None, "fields": ["line"], "spans": [_span_payload(right.line_span, nxt.text, "line")], "reasons": reasons, "confidence": 0.92 if action_supported else 0.82, "evidence": {"assignment": "unmatched_after_unique_assignment", "element_local_filter": "passed", "action_bucket": bucket, "action_support": action_reason if action_supported else None, "retention": retention, "page_replace": page_replace, "dominant_motion": dominant_motion, "v12_shadow_release": shadow_release}}
                focus_units.append(unit)
                next_labels.append({"next_element_index": right.index, "label": "ADD", "focus_id": unit["focus_id"]})
                relations.append({"operation": "add", "current_element_index": None, "next_element_index": right.index, "focus_decision": "focus", "confidence_score": unit["confidence"], "confidence_band": "high_recall", "reason_codes": unit["reasons"], "changed_fields": ["line"], "evidence": unit["evidence"]})
            else:
                exclusion = exclusion or shadow_reason or "unreliable_next_element"
                next_labels.append({"next_element_index": right.index, "label": "BASE", "focus_id": None, "reason": exclusion})
                relations.append({"operation": "ambiguous", "current_element_index": None, "next_element_index": right.index, "focus_decision": "base_only", "confidence_score": None, "confidence_band": "filtered", "reason_codes": [str(exclusion)], "changed_fields": [], "evidence": {"element_local_filter": exclusion}})
            continue

        left = current.elements[match.current_index]
        fields, filtered = _changed_fields(
            left,
            right,
            action=action,
            bucket=bucket,
            viewport_bbox_filter=viewport_bbox_filters.get(
                (match.current_index, match.next_index)
            ),
        )
        if left.element_type in {"image", "icon"} or right.element_type in {"image", "icon"}:
            visual_fields={"type","state"} if match.method == "typed_control_reservation" else {"content","type","state"}
            for field in [value for value in fields if value in visual_fields]:
                fields.remove(field); filtered.append({"field":field,"reason":"visual_semantic_field_unreliable"})
        if match.method == "slot_first" and right.element_type in {"text", "link"} and not _tier_a_slot(left,right):
            motion_supported = dominant_motion is not None and abs((right.bbox[1]-left.bbox[1])-dominant_motion) <= 8.0
            if not motion_supported:
                for field in [value for value in fields if value in {"content", "type", "state"}]:
                    fields.remove(field); filtered.append({"field":field,"reason":"tier_b_semantic_without_neighbor_motion"})
        for item in filtered:
            filtered_fields.append({"current_element_index": left.index, "next_element_index": right.index, **item})
        spans = []
        for field in fields:
            span = {"content": right.content_span, "type": right.type_span, "state": right.state_span, "bbox": right.bbox_span}[field]
            if span is not None:
                spans.append(_span_payload(span, nxt.text, field))
        fields = [s["field"] for s in spans]
        assignment = provenance_by_pair[(match.current_index, match.next_index)]
        if fields:
            unit = {"focus_id": len(focus_units), "kind": "UPDATE", "next_element_index": right.index, "current_element_index": left.index, "fields": fields, "spans": spans, "reasons": ["same_ui_slot_changed_fields" if match.method == "slot_first" else "exact_identity_geometry_update"], "confidence": match.confidence, "evidence": {"assignment": assignment, "bbox_max_abs_delta": _bbox_delta(left, right), "jitter_threshold": JITTER_THRESHOLD, "assignment_jitter_threshold": JITTER_THRESHOLD, "focus_bbox_threshold": FOCUS_BBOX_THRESHOLD, "filtered_fields": filtered}}
            focus_units.append(unit)
            next_labels.append({"next_element_index": right.index, "label": "UPDATE", "focus_id": unit["focus_id"]})
            decision, operation = "focus", "update"
        else:
            next_labels.append({"next_element_index": right.index, "label": "BASE", "focus_id": None})
            decision, operation = "base_only", "unchanged"
        relations.append({"operation": operation, "current_element_index": left.index, "next_element_index": right.index, "focus_decision": decision, "confidence_score": match.confidence, "confidence_band": "slot_first" if match.method == "slot_first" else "identity", "reason_codes": ["changed_fields_focused"] if fields else (["field_filtered_to_base"] if filtered else ["matched_unchanged_or_jitter"]), "changed_fields": fields, "evidence": assignment})

    for left in current.elements:
        if left.index not in matched_current:
            relations.append({"operation": "delete", "current_element_index": left.index, "next_element_index": None, "focus_decision": "base_only", "confidence_score": None, "confidence_band": "disabled", "reason_codes": ["delete_focus_disabled"], "changed_fields": [], "evidence": {}})

    return {"schema_version": FOCUS_LABEL_VERSION, "sample_id": sample_id, "transition_class": "changed", "action_bucket": bucket, "action": action, "source": str((row.get("meta") or {}).get("dataset") or "unknown"), "current_image_path": str(row.get("current_image_path") or ""), "next_image_path": str(row.get("next_image_path") or ""), "current_md": current.text, "target_next_md": nxt.text, "target_sha256": _sha(nxt.text), "base_units": [asdict(s) for s in nxt.base_units], "focus_units": focus_units, "review_candidates": [], "current_elements": [_element_dict(e) for e in current.elements], "next_elements": [_element_dict(e) for e in nxt.elements], "matches": [asdict(m) for m in matches], "assignment_provenance": assignment_provenance, "filtered_fields": filtered_fields, "hover_viewport_normalization_bbox_drift": viewport_diagnostic, "relations": relations, "next_labels": next_labels, "stats": {"base_units": len(nxt.base_units), "focus_units": len(focus_units), "add_focus": sum(u["kind"] == "ADD" for u in focus_units), "update_focus": sum(u["kind"] == "UPDATE" for u in focus_units), "matched_elements": len(matches), "unmatched_current_elements": len(current.elements)-len(matched_current), "unmatched_next_elements": len(nxt.elements)-len(by_next), "filtered_fields": len(filtered_fields), "hover_viewport_normalization_bbox_filters": len(viewport_bbox_filters)}}


def validate_focus_annotation(
    annotation: dict[str, Any],
    *,
    require_matcher_provenance: bool = False,
    source_row: dict[str, Any] | None = None,
    project_root: Path | None = None,
    image_module: Any | None = None,
) -> None:
    target = canonical_assistant_target(annotation.get("target_next_md"))
    current = parse_markdown_spans(annotation.get("current_md"))
    nxt = parse_markdown_spans(annotation.get("target_next_md"))
    if annotation.get("schema_version") != FOCUS_LABEL_VERSION:
        raise V4FocusError("wrong v20 schema_version")
    action = annotation.get("action")
    if not isinstance(action, dict):
        raise V4FocusError("action must be an object")
    bucket = str(annotation.get("action_bucket") or "")
    units = annotation.get("focus_units")
    if not isinstance(units, list):
        raise V4FocusError("focus_units must be a list")
    viewport_diagnostic = annotation.get(
        "hover_viewport_normalization_bbox_drift"
    )
    if not isinstance(viewport_diagnostic, dict):
        raise V4FocusError(
            "viewport normalization diagnostic must be an object"
        )
    if require_matcher_provenance:
        if source_row is None:
            raise V4FocusError(
                "strict matcher validation requires the source row"
            )
        if project_root is None or image_module is None:
            raise V4FocusError(
                "strict viewport validation requires project_root and image_module"
            )
        expected_source_bindings = {
            "sample_id": str(source_row.get("id") or "").strip(),
            "transition_class": source_row.get("transition_class"),
            "action_bucket": str(source_row.get("action_bucket") or "").strip(),
            "current_image_path": str(
                source_row.get("current_image_path") or ""
            ),
            "next_image_path": str(source_row.get("next_image_path") or ""),
            "source": str(
                (source_row.get("meta") or {}).get("dataset") or "unknown"
            ),
        }
        source_action = (
            source_row.get("candidate_action")
            if isinstance(source_row.get("candidate_action"), dict)
            else {}
        )
        if not source_action and isinstance(
            source_row.get("candidate_action_json"), str
        ):
            try:
                parsed_source_action = json.loads(
                    source_row["candidate_action_json"]
                )
                source_action = (
                    parsed_source_action
                    if isinstance(parsed_source_action, dict)
                    else {}
                )
            except json.JSONDecodeError:
                source_action = {}
        expected_source_bindings["action"] = source_action
        for key, expected in expected_source_bindings.items():
            if annotation.get(key) != expected:
                raise V4FocusError(
                    f"matcher annotation is not source-bound for {key}"
                )
        if current.text != parse_markdown_spans(source_row.get("current_md")).text:
            raise V4FocusError("matcher current Markdown is not source-bound")
        if nxt.text != parse_markdown_spans(
            source_row.get("target_next_md")
        ).text:
            raise V4FocusError("matcher target Markdown is not source-bound")
        validation_current_size, validation_next_size, validation_image_info = (
            _read_viewport_image_sizes(
                source_row,
                project_root=project_root,
                image_module=image_module,
            )
        )
    else:
        validation_current_size = _valid_image_size(
            viewport_diagnostic.get("current_image_size")
        )
        validation_next_size = _valid_image_size(
            viewport_diagnostic.get("next_image_size")
        )
        image_size_status = viewport_diagnostic.get("image_size_status")
        allowed_statuses = {
            "available",
            "missing_parameters",
            "missing_paths",
            "current_image_read_error",
            "next_image_read_error",
            "current_image_invalid_size",
            "next_image_invalid_size",
        }
        if image_size_status not in allowed_statuses:
            raise V4FocusError("invalid viewport image_size_status")
        if image_size_status == "available":
            if validation_current_size is None or validation_next_size is None:
                raise V4FocusError(
                    "available viewport provenance requires valid image sizes"
                )
        elif (
            validation_current_size is not None
            or validation_next_size is not None
        ):
            raise V4FocusError(
                "unavailable viewport provenance must not record image sizes"
            )
        if viewport_diagnostic.get("current_image_path") != annotation.get(
            "current_image_path"
        ) or viewport_diagnostic.get("next_image_path") != annotation.get(
            "next_image_path"
        ):
            raise V4FocusError(
                "viewport diagnostic image paths are not annotation-bound"
            )
        validation_image_info = {
            key: viewport_diagnostic.get(key)
            for key in (
                "image_size_status",
                "current_image_path",
                "next_image_path",
                "current_image_resolved_path",
                "next_image_resolved_path",
            )
        }

    expected_matches, expected_provenance = assign_elements(
        current,
        nxt,
        action=action,
        bucket=bucket,
    )
    (
        expected_viewport_bbox_filters,
        expected_viewport_diagnostic,
    ) = _hover_viewport_normalization_bbox_drift_filter(
        current,
        nxt,
        expected_matches,
        action=action,
        bucket=bucket,
        current_size=validation_current_size,
        next_size=validation_next_size,
    )
    (
        expected_viewport_bbox_filters,
        expected_viewport_diagnostic,
    ) = _bind_viewport_image_provenance(
        expected_viewport_bbox_filters,
        expected_viewport_diagnostic,
        validation_image_info,
    )
    if viewport_diagnostic != expected_viewport_diagnostic:
        raise V4FocusError(
            "viewport normalization diagnostic does not independently recompute"
        )
    if annotation.get("target_sha256") != _sha(target):
        raise V4FocusError("target hash mismatch")
    expected_base = [asdict(span) for span in nxt.base_units]
    if annotation.get("base_units") != expected_base:
        raise V4FocusError("base units do not exactly partition target")
    if not expected_base or expected_base[0]["start"] != 0 or expected_base[-1]["end"] != len(target) or any(expected_base[i]["end"] != expected_base[i+1]["start"] for i in range(len(expected_base)-1)):
        raise V4FocusError("invalid base partition")
    seen_next: set[int] = set()
    for expected_id, unit in enumerate(units):
        if unit.get("focus_id") != expected_id:
            raise V4FocusError("non-contiguous focus_id")
        ni = int(unit["next_element_index"])
        if ni < 0 or ni >= len(nxt.elements):
            raise V4FocusError("next index out of range")
        if ni in seen_next:
            raise V4FocusError("more than one focus unit for next element")
        seen_next.add(ni)
        if unit.get("kind") not in {"ADD", "UPDATE"}:
            raise V4FocusError("DELETE/unknown focus kind is forbidden")
        spans = unit.get("spans") or []
        if not spans:
            raise V4FocusError("focus unit has no spans")
        for span in spans:
            start, end = int(span["start"]), int(span["end"])
            if target[start:end] != span.get("text") or _sha(target[start:end]) != span.get("text_sha256"):
                raise V4FocusError("span provenance mismatch")
        element = nxt.elements[ni]
        release_reason_present = V12_RELIABLE_SAME_REGION_RELEASE_REASON in (unit.get("reasons") or [])
        release_evidence = (unit.get("evidence") or {}).get("v12_shadow_release")
        if release_reason_present != (release_evidence is not None):
            raise V4FocusError("V12 shadow release reason/evidence mismatch")
        if unit["kind"] == "ADD":
            if unit.get("current_element_index") is not None or unit.get("fields") != ["line"] or len(spans) != 1 or (spans[0]["start"], spans[0]["end"]) != (element.line_span.start, element.line_span.end):
                raise V4FocusError("ADD must be exactly one complete next line span")
        else:
            ci = unit.get("current_element_index")
            if not isinstance(ci, int) or ci < 0 or ci >= len(current.elements):
                raise V4FocusError("UPDATE current index out of range")
            expected_spans = {"content": element.content_span, "type": element.type_span, "state": element.state_span, "bbox": element.bbox_span}
            fields = unit.get("fields") or []
            evidence = unit.get("evidence")
            if not isinstance(evidence, dict):
                raise V4FocusError("UPDATE evidence must be an object")
            try:
                legacy_assignment_threshold = float(evidence["jitter_threshold"])
                assignment_threshold = float(evidence["assignment_jitter_threshold"])
                focus_bbox_threshold = float(evidence["focus_bbox_threshold"])
                recorded_bbox_delta = float(evidence["bbox_max_abs_delta"])
            except (KeyError, TypeError, ValueError) as exc:
                raise V4FocusError("invalid UPDATE bbox threshold provenance") from exc
            actual_bbox_delta = _bbox_delta(current.elements[ci], element)
            bbox_filter = _stable_origin_extent_drift_filter(
                current.elements[ci],
                element,
                action=action,
                bucket=bucket,
            )
            viewport_bbox_filter = expected_viewport_bbox_filters.get(
                (ci, ni)
            )
            if (
                legacy_assignment_threshold != assignment_threshold
                or assignment_threshold != JITTER_THRESHOLD
                or focus_bbox_threshold != FOCUS_BBOX_THRESHOLD
                or recorded_bbox_delta != actual_bbox_delta
            ):
                raise V4FocusError("UPDATE bbox threshold provenance mismatch")
            expected_bbox = (
                actual_bbox_delta > FOCUS_BBOX_THRESHOLD
                and bbox_filter is None
                and viewport_bbox_filter is None
            )
            if ("bbox" in fields) != expected_bbox:
                raise V4FocusError("UPDATE bbox field/threshold mismatch")
            unit_filtered = evidence.get("filtered_fields")
            if not isinstance(unit_filtered, list):
                raise V4FocusError("UPDATE filtered_fields must be a list")
            if any(not isinstance(item, dict) for item in unit_filtered):
                raise V4FocusError("UPDATE filtered_fields entries must be objects")
            actual_unit_bbox_filters = [
                item
                for item in unit_filtered
                if item.get("reason") == STABLE_ORIGIN_EXTENT_DRIFT_REASON
            ]
            expected_unit_bbox_filters = [bbox_filter] if bbox_filter else []
            if actual_unit_bbox_filters != expected_unit_bbox_filters:
                raise V4FocusError("UPDATE bbox filter evidence mismatch")
            actual_unit_viewport_filters = [
                item
                for item in unit_filtered
                if item.get("reason")
                == HOVER_VIEWPORT_NORMALIZATION_BBOX_DRIFT_REASON
            ]
            expected_unit_viewport_filters = (
                [{"field": "bbox", **viewport_bbox_filter}]
                if viewport_bbox_filter is not None
                else []
            )
            if (
                actual_unit_viewport_filters
                != expected_unit_viewport_filters
            ):
                raise V4FocusError(
                    "UPDATE viewport bbox filter evidence mismatch"
                )
            state_filter = _nonclick_disclosure_state_drift_filter(
                current.elements[ci],
                element,
                action=action,
                bucket=bucket,
            )
            actual_unit_state_filters = [
                item
                for item in unit_filtered
                if item.get("reason")
                == NONCLICK_DISCLOSURE_STATE_DRIFT_REASON
            ]
            expected_unit_state_filters = [state_filter] if state_filter else []
            if actual_unit_state_filters != expected_unit_state_filters:
                raise V4FocusError(
                    "UPDATE disclosure state filter evidence mismatch"
                )
            if state_filter is not None and "state" in fields:
                raise V4FocusError(
                    "filtered disclosure state must not remain in UPDATE"
                )
            if len(fields) != len(set(fields)) or len(fields) != len(spans):
                raise V4FocusError("UPDATE fields/spans mismatch")
            for field, span in zip(fields, spans):
                expected = expected_spans.get(field)
                if expected is None or span.get("field") != field or (span["start"], span["end"]) != (expected.start, expected.end):
                    raise V4FocusError("UPDATE does not use target field span")
    matches = annotation.get("matches") or []
    if len({m["current_index"] for m in matches}) != len(matches) or len({m["next_index"] for m in matches}) != len(matches):
        raise V4FocusError("assignment is not one-to-one")
    for match in matches:
        ci, ni = int(match["current_index"]), int(match["next_index"])
        if not (0 <= ci < len(current.elements) and 0 <= ni < len(nxt.elements)):
            raise V4FocusError("match index out of range")
        actual_iou = bbox_iou(current.elements[ci].bbox, nxt.elements[ni].bbox)
        actual_distance = center_distance(current.elements[ci].bbox, nxt.elements[ni].bbox)
        if abs(float(match["bbox_iou"])-actual_iou) > 1e-6 or abs(float(match["center_distance"])-actual_distance) > 1e-6:
            raise V4FocusError("match geometry provenance mismatch")
    try:
        parsed_matches = [ElementMatch(**match) for match in matches]
    except (TypeError, ValueError) as exc:
        raise V4FocusError("invalid serialized match") from exc
    filtered_fields = annotation.get("filtered_fields")
    if not isinstance(filtered_fields, list):
        raise V4FocusError("filtered_fields must be a list")
    if any(not isinstance(item, dict) for item in filtered_fields):
        raise V4FocusError("filtered_fields entries must be objects")
    updates_by_pair = {
        (int(unit["current_element_index"]), int(unit["next_element_index"])): unit
        for unit in units
        if unit.get("kind") == "UPDATE"
    }
    expected_bbox_filters: list[dict[str, Any]] = []
    expected_disclosure_state_filters: list[dict[str, Any]] = []
    expected_viewport_filtered_fields: list[dict[str, Any]] = []
    for match in sorted(parsed_matches, key=lambda item: item.next_index):
        left = current.elements[match.current_index]
        right = nxt.elements[match.next_index]
        bbox_filter = _stable_origin_extent_drift_filter(
            left,
            right,
            action=action,
            bucket=bucket,
        )
        expected_bbox = (
            _bbox_delta(left, right) > FOCUS_BBOX_THRESHOLD
            and bbox_filter is None
            and (match.current_index, match.next_index)
            not in expected_viewport_bbox_filters
        )
        unit = updates_by_pair.get((match.current_index, match.next_index))
        actual_bbox = bool(unit and "bbox" in (unit.get("fields") or []))
        if actual_bbox != expected_bbox:
            raise V4FocusError("matched-pair bbox Focus decision mismatch")
        if bbox_filter is not None:
            expected_bbox_filters.append(
                {
                    "current_element_index": left.index,
                    "next_element_index": right.index,
                    **bbox_filter,
                }
            )
        viewport_bbox_filter = expected_viewport_bbox_filters.get(
            (match.current_index, match.next_index)
        )
        if viewport_bbox_filter is not None:
            expected_viewport_filtered_fields.append(
                {
                    "current_element_index": left.index,
                    "next_element_index": right.index,
                    "field": "bbox",
                    **viewport_bbox_filter,
                }
            )
        state_filter = _nonclick_disclosure_state_drift_filter(
            left,
            right,
            action=action,
            bucket=bucket,
        )
        if state_filter is not None:
            expected_disclosure_state_filters.append(
                {
                    "current_element_index": left.index,
                    "next_element_index": right.index,
                    **state_filter,
                }
            )
            if unit and "state" in (unit.get("fields") or []):
                raise V4FocusError(
                    "matched disclosure state filter was not applied"
                )
    actual_bbox_filters = [
        item
        for item in filtered_fields
        if item.get("reason") == STABLE_ORIGIN_EXTENT_DRIFT_REASON
    ]
    if actual_bbox_filters != expected_bbox_filters:
        raise V4FocusError("bbox suppression filtered_fields mismatch")
    actual_disclosure_state_filters = [
        item
        for item in filtered_fields
        if item.get("reason") == NONCLICK_DISCLOSURE_STATE_DRIFT_REASON
    ]
    if actual_disclosure_state_filters != expected_disclosure_state_filters:
        raise V4FocusError(
            "disclosure state suppression filtered_fields mismatch"
        )
    actual_viewport_filtered_fields = [
        item
        for item in filtered_fields
        if item.get("reason")
        == HOVER_VIEWPORT_NORMALIZATION_BBOX_DRIFT_REASON
    ]
    if actual_viewport_filtered_fields != expected_viewport_filtered_fields:
        raise V4FocusError(
            "viewport bbox suppression filtered_fields mismatch"
        )
    dominant_motion = _dominant_motion(parsed_matches, current, nxt)
    for unit in units:
        if V12_RELIABLE_SAME_REGION_RELEASE_REASON not in (unit.get("reasons") or []):
            continue
        if unit.get("kind") != "ADD":
            raise V4FocusError("V12 shadow release is valid only for ADD")
        element = nxt.elements[int(unit["next_element_index"])]
        reliable, _ = _reliable_add(element)
        actual_reasons = _local_shadow_reasons(
            element,
            current.elements,
            motion=dominant_motion,
        )
        evidence = unit["evidence"]["v12_shadow_release"]
        if (
            not reliable
            or set(actual_reasons) != {"same_region_current_shadow"}
            or evidence.get("reason") != V12_RELIABLE_SAME_REGION_RELEASE_REASON
            or evidence.get("released_shadow_reason") != "same_region_current_shadow"
            or evidence.get("all_shadow_reasons") != list(actual_reasons)
            or evidence.get("unique_shadow_reasons") != ["same_region_current_shadow"]
        ):
            raise V4FocusError("invalid V12 reliable same-region release provenance")
    provenance = annotation.get("assignment_provenance")
    if not isinstance(provenance, list) or len(provenance) != len(matches):
        raise V4FocusError("assignment provenance cardinality mismatch")
    if matches != [asdict(match) for match in expected_matches]:
        raise V4FocusError("matches do not reproduce candidate assignment")
    if provenance != expected_provenance:
        raise V4FocusError("assignment provenance does not reproduce candidate assignment")
    match_pairs={(int(m["current_index"]),int(m["next_index"]),str(m["method"])) for m in matches}
    if [p.get("assignment_id") for p in provenance] != list(range(len(provenance))) or {(int(p["current_element_index"]),int(p["next_element_index"]),str(p["phase"])) for p in provenance} != match_pairs:
        raise V4FocusError("assignment provenance does not bind matches")
    labels=annotation.get("next_labels")
    if not isinstance(labels,list) or [x.get("next_element_index") for x in labels] != list(range(len(nxt.elements))):
        raise V4FocusError("next_labels must cover every next element in order")
    units_by_id={u["focus_id"]:u for u in units}
    for label in labels:
        fid=label.get("focus_id")
        if fid is None:
            if label.get("label") not in {"BASE"}: raise V4FocusError("unfocused next label must be BASE")
        elif fid not in units_by_id or label.get("label") != units_by_id[fid]["kind"] or label["next_element_index"] != units_by_id[fid]["next_element_index"]:
            raise V4FocusError("next label/focus unit mismatch")
    relations=annotation.get("relations")
    if not isinstance(relations,list) or {r.get("next_element_index") for r in relations if r.get("next_element_index") is not None} != set(range(len(nxt.elements))):
        raise V4FocusError("relations do not cover next elements")
    stats=annotation.get("stats") or {}
    expected_stats={"base_units":len(nxt.base_units),"focus_units":len(units),"add_focus":sum(u["kind"]=="ADD" for u in units),"update_focus":sum(u["kind"]=="UPDATE" for u in units),"matched_elements":len(matches),"hover_viewport_normalization_bbox_filters":len(expected_viewport_bbox_filters)}
    if any(stats.get(k) != v for k,v in expected_stats.items()):
        raise V4FocusError("stats do not recompute from annotation")
    if any(unit.get("kind") == "DELETE" for unit in units):
        raise V4FocusError("DELETE focus is forbidden")
    if require_matcher_provenance:
        assert source_row is not None
        rebuilt = build_focus_annotation(
            source_row,
            project_root=project_root,
            image_module=image_module,
        )
        if annotation != rebuilt:
            raise V4FocusError(
                "strict matcher annotation does not reproduce from source row"
            )


def match_elements(current, nxt, **_: Any):
    """Compatibility entry point for the final V12 matcher."""
    return assign_elements(current, nxt, action=_.get("action"), bucket=str(_.get("action_bucket") or ""))[0]
