from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from mkd2world.training.markdown import ELEMENT_RE, STATE_PREFIX_RE, TYPE_SUFFIX_RE


ENVIRONMENT_RE = re.compile(r"^\[Environment:\s*[^\]]+\]$")
LINE_LEADER_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?")
MAIN_PREFIX_RE = re.compile(r"^\[(?P<content>[^\]]*)\](?:\((?P<type>[^)]+)\))?$")
FOCUS_LABEL_VERSION = "v4-focus-v7"
DEFAULT_JITTER_THRESHOLD = 2.0
DEFAULT_GEOMETRY_THRESHOLD = 8.0
DEFAULT_ADD_REGION_DIFF = 0.08
DEFAULT_UPDATE_REGION_DIFF = 0.025
ADD_ELEMENT_TYPES = frozenset({"text", "link", "button", "input", "toggle", "select"})
MOTION_TEXTUAL_ROLE_TYPES = frozenset({"text", "link", "button"})
SCROLL_BIN_SIZE = 4.0
SCROLL_RESIDUAL_LIMIT = 6.0
MARKDOWN_GRID_EXTENT = 999.0
HOVER_EDGE_MARGIN = 3.0
HOVER_AUTOSCROLL_MIN_ANCHORS = 8
HOVER_AUTOSCROLL_MIN_FRACTION = 0.60
HOVER_AUTOSCROLL_MIN_SHIFT = 40.0
HOVER_AUTOSCROLL_CLUSTER_RADIUS = 24.0
HOVER_AUTOSCROLL_MODEL_RESIDUAL = 6.0
HOVER_AUTOSCROLL_MODEL_INLIER_FRACTION = 0.80
HOVER_AUTOSCROLL_MIN_Y_SPAN = 120.0
MONTH_SLOT_X_RESIDUAL = 24.0
MONTH_SLOT_Y_RESIDUAL = 12.0
CALENDAR_MIN_STABLE_LEAF_ANCHORS = 3
CALENDAR_MIN_DATE_GRID_ITEMS = 5
CALENDAR_LOW_ANCHOR_MIN_LOCAL_ANCHORS = 2
AUTOCOMPLETE_INPUT_IOU = 0.45
AUTOCOMPLETE_INPUT_DISTANCE = 40.0
STATE_FOCUS_ELEMENT_TYPES = frozenset({"toggle", "select"})
SPECIAL_MATCH_VISUAL_DIFF = 0.008
V6_MIN_RETENTION_RATIO = 0.20
V6_MIN_STABLE_ANCHORS = 3
V6_SEARCH_MODE_MIN_STABLE_ANCHORS = 2
V6_STICKY_MIN_FIXED_ANCHORS = 2
V6_MAX_FOCUS_UNITS = 2
V6_DIRECT_DRAG_MIN_DISPLACEMENT = 10.0
V6_DRAG_SORT_ENDPOINT_PADDING = 18.0
V6_DRAG_FORM_ENDPOINT_PADDING = 40.0
V6_BING_MIN_RETENTION_RATIO = 0.18
V6_BING_MIN_STABLE_ANCHORS = 4
V7_SEARCH_MODE_CLEAR_ALL_REGION_DIFF = 0.04
V7_SEARCH_MODE_CLEAR_ALL_TYPES = frozenset({"image", "link", "text"})
V7_SEARCH_MODE_CLEAR_ALL_MIN_VERTICAL_OVERLAP = 0.85
V7_SEARCH_MODE_CLEAR_ALL_MAX_CENTER_Y_DELTA = 12.0
V7_OVERSTOCK_QUERY_PLACEHOLDERS = frozenset(
    {
        "search crazy-good deals",
        "q search crazy-good deals",
        "qsearch crazy-good deals",
    }
)
V7_OVERSTOCK_HOST = "www.overstock.com"
V7_OVERSTOCK_INPUT_PATH = "body>div>store-header>header>div>form>input"
V7_OVERSTOCK_INPUT_SELECTOR = 'input[data-testid="search-input"]'
V7_RAW_RECT_CLOSURE_TOLERANCE = 1e-6
V7_RAW_RECT_BOUNDARY_TOLERANCE = 1.0
V7_FOX_ACTION_TARGET_X_PADDING = 96.0
V7_FOX_ACTION_TARGET_Y_PADDING = 8.0
V7_RAW_MOUSE_CLICK_PIXEL_RESIDUAL = 1.0
V7_RAW_SCROLL_GRID_RESIDUAL = 1.0
V7_RAW_RECT_OCR_PADDING = 8.0
RAW_MOUSE_CLICK_RE = re.compile(
    r"^mouse_click\(\s*x=(?P<x>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*,\s*y=(?P<y>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*,\s*button=(?P<quote>['\"])left(?P=quote)\s*\)$"
)
RAW_SCROLL_RE = re.compile(
    r"^scroll\(\s*delta_x=(?P<delta_x>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*,\s*delta_y=(?P<delta_y>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)$"
)
TYPED_DIRECT_ELEMENT_TYPES = frozenset({"text", "input"})
ROUTE_INPUT_PLACEHOLDER_RE = re.compile(
    r"^(?:(?:a\)\s*)?enter starting point(?:\s+a)?|"
    r"(?:b\)\s*)?enter destination(?:\s+b)?)$",
    flags=re.IGNORECASE,
)
CLICK_TEXT_TOGGLE_PAIRS = frozenset(
    {
        ("details", "close"),
        ("close", "details"),
    }
)
FOCUS_REASON_CONTRACTS = {
    "calendar_month_slot_content_update": {
        "fields": ("content",),
        "methods": frozenset({"calendar_month_slot"}),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click", "double_click"}),
    },
    "autocomplete_commit_input_content_update": {
        "fields": ("content",),
        "methods": frozenset({"autocomplete_commit"}),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click"}),
    },
    "clicked_exact_text_toggle_content_update": {
        "fields": ("content",),
        "methods": frozenset({"click_text_toggle"}),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click", "double_click"}),
    },
    "state_flip_at_click_target": {
        "fields": ("state",),
        "methods": frozenset({"state_identity"}),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click"}),
    },
    "typed_value_exact_spatial_update": {
        "fields": ("content",),
        "methods": frozenset({"spatial_mutual"}),
        "buckets": frozenset({"text_key"}),
        "actions": frozenset({"type_text"}),
    },
    "typed_value_exact_direct_slot_update": {
        "fields": ("content",),
        "methods": frozenset({"typed_direct_slot"}),
        "buckets": frozenset({"text_key"}),
        "actions": frozenset({"type_text"}),
    },
    "typed_value_deterministic_echo_update": {
        "fields": ("content",),
        "methods": frozenset({"typed_echo_template"}),
        "buckets": frozenset({"text_key"}),
        "actions": frozenset({"type_text"}),
    },
    "typed_value_overstock_query_marker_slot_update": {
        "fields": ("content",),
        "methods": frozenset({"typed_overstock_query_marker_slot"}),
        "buckets": frozenset({"text_key"}),
        "actions": frozenset({"type_text"}),
    },
    "search_mode_recent_header_content_update": {
        "fields": ("content",),
        "methods": frozenset({"search_mode_slot"}),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click"}),
    },
    "sticky_pos_header_content_update": {
        "fields": ("content",),
        "methods": frozenset({"sticky_pos_header"}),
        "buckets": frozenset({"scroll"}),
        "actions": frozenset({"scroll"}),
    },
    "drag_sort_content_update": {
        "fields": ("content",),
        "methods": frozenset({"drag_sort_slot"}),
        "buckets": frozenset({"other"}),
        "actions": frozenset({"drag"}),
    },
    "drag_form_control_content_update": {
        "fields": ("content",),
        "methods": frozenset({"drag_form_control_slot"}),
        "buckets": frozenset({"other"}),
        "actions": frozenset({"drag"}),
    },
    "drag_binary_selector_content_update": {
        "fields": ("content",),
        "methods": frozenset({"drag_binary_selector_slot"}),
        "buckets": frozenset({"other"}),
        "actions": frozenset({"drag"}),
    },
    "bing_search_overlay_anchor_add": {
        "fields": ("line",),
        "methods": frozenset(),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click"}),
    },
    "filter_sort_drawer_anchor_add": {
        "fields": ("line",),
        "methods": frozenset(),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click"}),
    },
    "advanced_search_row_anchor_add": {
        "fields": ("line",),
        "methods": frozenset(),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click"}),
    },
    "search_mode_recent_clear_all_anchor_add": {
        "fields": ("line",),
        "methods": frozenset(),
        "buckets": frozenset({"click"}),
        "actions": frozenset({"left_click"}),
    },
}

V7_ATOMIC_UPDATE_REASONS = frozenset(
    {
        "calendar_month_slot_content_update",
        "search_mode_recent_header_content_update",
    }
)
V7_SPECIAL_UPDATE_REASONS = frozenset(
    {
        "autocomplete_commit_input_content_update",
        "clicked_exact_text_toggle_content_update",
        "typed_value_overstock_query_marker_slot_update",
        "search_mode_recent_header_content_update",
        "sticky_pos_header_content_update",
        "drag_sort_content_update",
        "drag_form_control_content_update",
        "drag_binary_selector_content_update",
    }
)
V7_TYPED_UPDATE_REASONS = frozenset(
    {
        "typed_value_exact_spatial_update",
        "typed_value_exact_direct_slot_update",
        "typed_value_deterministic_echo_update",
    }
)
V7_GENERIC_STATE_REASON = "state_flip_at_click_target"
V7_ATOMIC_PRIORITY = 400
V7_SPECIAL_PRIORITY = 300
V7_TYPED_PRIORITY = 200
V7_GENERIC_STATE_PRIORITY = 100

V6_UPDATE_METHODS = frozenset(
    {
        "search_mode_slot",
        "sticky_pos_header",
        "drag_sort_slot",
        "drag_form_control_slot",
        "drag_binary_selector_slot",
        "typed_overstock_query_marker_slot",
    }
)
V6_ADD_REASONS = frozenset(
    {
        "bing_search_overlay_anchor_add",
        "filter_sort_drawer_anchor_add",
        "advanced_search_row_anchor_add",
        "search_mode_recent_clear_all_anchor_add",
    }
)
POS_HEADER_RE = re.compile(
    r"^(?:adjective|verb|noun|adverb|pronoun|preposition|conjunction|interjection)\s+\d(?:\s+of\s+\d)?$",
    flags=re.IGNORECASE,
)
MONTH_ORDINAL = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
MONTH_TITLE_RE = re.compile(
    r"^(?P<month>" + "|".join(sorted(MONTH_ORDINAL, key=len, reverse=True))
    + r")\.?\s+(?P<year>20\d{2})$",
    flags=re.IGNORECASE,
)


class V4FocusError(ValueError):
    """Raised when Markdown or focus annotations violate the V4 contract."""


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise V4FocusError(f"invalid character span: {self.start}:{self.end}")


@dataclass(frozen=True)
class ParsedElementSpan:
    index: int
    line_index: int
    raw_line: str
    line_span: CharSpan
    body_span: CharSpan
    content: str
    element_type: str
    state: str
    bbox: tuple[float, float, float, float]
    content_span: CharSpan
    type_span: CharSpan | None
    state_span: CharSpan | None
    bbox_span: CharSpan

    @property
    def normalized_content(self) -> str:
        return normalize_text(self.content)

    @property
    def semantic_key(self) -> tuple[str, str, str]:
        return self.normalized_content, self.element_type, self.state.lower()

    @property
    def identity_key(self) -> tuple[str, str]:
        return self.normalized_content, self.element_type


@dataclass(frozen=True)
class ParsedMarkdownSpans:
    text: str
    environment_span: CharSpan
    elements: tuple[ParsedElementSpan, ...]
    base_units: tuple[CharSpan, ...]


@dataclass(frozen=True)
class ElementMatch:
    current_index: int
    next_index: int
    method: str
    confidence: float
    bbox_iou: float
    center_distance: float
    content_similarity: float


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.lower().split())


def _strip_visual_role_prefix(value: str) -> str:
    return re.sub(r"^(?:img|icon)\s*:\s*", "", value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_assistant_target(markdown: object) -> str:
    text = str(markdown or "").strip()
    if not text:
        raise V4FocusError("empty target Markdown")
    # This is exactly markdown.build_assistant_output("", target): one trailing newline.
    return text + "\n"


def _line_segments(text: str) -> list[tuple[int, int, str]]:
    output: list[tuple[int, int, str]] = []
    cursor = 0
    for segment in text.splitlines(keepends=True):
        start = cursor
        cursor += len(segment)
        output.append((start, cursor, segment.rstrip("\r\n")))
    if cursor < len(text):
        output.append((cursor, len(text), text[cursor:]))
    return output


def _element_type(prefix: str) -> str:
    type_match = TYPE_SUFFIX_RE.search(prefix)
    base_type = type_match.group("type").strip().lower() if type_match else "block"
    state_match = STATE_PREFIX_RE.match(prefix.strip())
    if not state_match:
        return base_type
    state = state_match.group("state").lower()
    if state in {"x", " "}:
        return "toggle"
    if state in {">", "v"}:
        return "select"
    return base_type


def _parse_element_line(
    raw_line: str,
    *,
    line_index: int,
    line_start: int,
    line_end: int,
    element_index: int,
) -> ParsedElementSpan:
    leader = LINE_LEADER_RE.match(raw_line)
    body_start_in_line = leader.end() if leader else 0
    body_end_in_line = len(raw_line.rstrip())
    if body_start_in_line >= body_end_in_line:
        raise V4FocusError(f"line {line_index + 1} has no element body")
    body = raw_line[body_start_in_line:body_end_in_line]
    match = ELEMENT_RE.fullmatch(body)
    if match is None:
        raise V4FocusError(f"line {line_index + 1} is not a strict element: {raw_line!r}")

    prefix = match.group("prefix")
    prefix_offset = line_start + body_start_in_line + match.start("prefix")
    state_match = STATE_PREFIX_RE.match(prefix)
    state = state_match.group("state") if state_match else ""
    state_span = None
    remainder_start = 0
    if state_match:
        state_span = CharSpan(
            prefix_offset + state_match.start("state") - 1,
            prefix_offset + state_match.end("state") + 1,
        )
        remainder_start = state_match.end()
    while remainder_start < len(prefix) and prefix[remainder_start].isspace():
        remainder_start += 1
    remainder = prefix[remainder_start:]
    main_match = MAIN_PREFIX_RE.fullmatch(remainder)
    if main_match is None:
        raise V4FocusError(f"line {line_index + 1} has unsupported element prefix: {prefix!r}")

    content = main_match.group("content")
    content_start = prefix_offset + remainder_start + main_match.start("content")
    content_span = CharSpan(content_start, content_start + len(content))
    type_span = None
    if main_match.group("type") is not None:
        type_start = prefix_offset + remainder_start + main_match.start("type") - 1
        type_end = prefix_offset + remainder_start + main_match.end("type") + 1
        type_span = CharSpan(type_start, type_end)

    bbox_start = line_start + body_start_in_line + match.start("cx") - 1
    bbox_end = line_start + body_start_in_line + match.end("h") + 1
    return ParsedElementSpan(
        index=element_index,
        line_index=line_index,
        raw_line=raw_line,
        line_span=CharSpan(line_start, line_end),
        body_span=CharSpan(line_start + body_start_in_line, line_start + body_end_in_line),
        content=content,
        element_type=_element_type(prefix),
        state=state,
        bbox=tuple(float(match.group(name)) for name in ("cx", "cy", "w", "h")),
        content_span=content_span,
        type_span=type_span,
        state_span=state_span,
        bbox_span=CharSpan(bbox_start, bbox_end),
    )


def parse_markdown_spans(markdown: object) -> ParsedMarkdownSpans:
    text = canonical_assistant_target(markdown)
    segments = _line_segments(text)
    environment_span: CharSpan | None = None
    elements: list[ParsedElementSpan] = []
    element_line_starts: list[int] = []
    saw_nonempty = False

    for line_index, (line_start, line_end, raw_line) in enumerate(segments):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not saw_nonempty:
            saw_nonempty = True
            if not ENVIRONMENT_RE.fullmatch(stripped):
                raise V4FocusError("first non-empty line must be an Environment line")
            stripped_start = line_start + raw_line.index(stripped)
            environment_span = CharSpan(stripped_start, stripped_start + len(stripped))
            continue
        element = _parse_element_line(
            raw_line,
            line_index=line_index,
            line_start=line_start,
            line_end=line_start + len(raw_line),
            element_index=len(elements),
        )
        elements.append(element)
        element_line_starts.append(line_start)

    if environment_span is None:
        raise V4FocusError("missing Environment line")
    if not elements:
        raise V4FocusError("target Markdown contains no elements")

    base_units: list[CharSpan] = []
    first_element_start = element_line_starts[0]
    base_units.append(CharSpan(0, first_element_start))
    for index, start in enumerate(element_line_starts):
        end = element_line_starts[index + 1] if index + 1 < len(element_line_starts) else len(text)
        base_units.append(CharSpan(start, end))
    return ParsedMarkdownSpans(
        text=text,
        environment_span=environment_span,
        elements=tuple(elements),
        base_units=tuple(base_units),
    )


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    def corners(box: Sequence[float]) -> tuple[float, float, float, float]:
        cx, cy, width, height = box
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    lx1, ly1, lx2, ly2 = corners(left)
    rx1, ry1, rx2, ry2 = corners(right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def horizontal_overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    """Horizontal intersection divided by the narrower box width."""
    left_min = float(left[0]) - float(left[2]) / 2.0
    left_max = float(left[0]) + float(left[2]) / 2.0
    right_min = float(right[0]) - float(right[2]) / 2.0
    right_max = float(right[0]) + float(right[2]) / 2.0
    overlap = max(0.0, min(left_max, right_max) - max(left_min, right_min))
    return overlap / max(1.0, min(float(left[2]), float(right[2])))


def vertical_overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    left_min = float(left[1]) - float(left[3]) / 2.0
    left_max = float(left[1]) + float(left[3]) / 2.0
    right_min = float(right[1]) - float(right[3]) / 2.0
    right_max = float(right[1]) + float(right[3]) / 2.0
    overlap = max(0.0, min(left_max, right_max) - max(left_min, right_min))
    return overlap / max(1.0, min(float(left[3]), float(right[3])))


def _bbox_center_inside(
    inner: Sequence[float], outer: Sequence[float], *, padding: float = 0.0
) -> bool:
    return bool(
        float(outer[0]) - float(outer[2]) / 2.0 - padding
        <= float(inner[0])
        <= float(outer[0]) + float(outer[2]) / 2.0 + padding
        and float(outer[1]) - float(outer[3]) / 2.0 - padding
        <= float(inner[1])
        <= float(outer[1]) + float(outer[3]) / 2.0 + padding
    )


def _bbox_inside(
    inner: Sequence[float],
    outer: Sequence[float],
    *,
    padding: float = 0.0,
) -> bool:
    """Return whether every corner of ``inner`` is inside padded ``outer``."""

    try:
        inner_x, inner_y, inner_width, inner_height = [float(value) for value in inner]
        outer_x, outer_y, outer_width, outer_height = [float(value) for value in outer]
    except (TypeError, ValueError):
        return False
    values = (
        inner_x,
        inner_y,
        inner_width,
        inner_height,
        outer_x,
        outer_y,
        outer_width,
        outer_height,
        float(padding),
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return bool(
        inner_width >= 0.0
        and inner_height >= 0.0
        and outer_width >= 0.0
        and outer_height >= 0.0
        and inner_x - inner_width / 2.0
        >= outer_x - outer_width / 2.0 - padding
        and inner_x + inner_width / 2.0
        <= outer_x + outer_width / 2.0 + padding
        and inner_y - inner_height / 2.0
        >= outer_y - outer_height / 2.0 - padding
        and inner_y + inner_height / 2.0
        <= outer_y + outer_height / 2.0 + padding
    )


def center_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def content_similarity(left: ParsedElementSpan, right: ParsedElementSpan) -> float:
    return SequenceMatcher(None, left.normalized_content, right.normalized_content).ratio()


def _bbox_contains_point(
    bbox: Sequence[float],
    point: Sequence[float] | None,
    *,
    padding: float = 0.0,
) -> bool:
    if point is None or len(point) != 2:
        return False
    cx, cy, width, height = [float(value) for value in bbox]
    x, y = [float(value) for value in point]
    return (
        cx - width / 2 - padding <= x <= cx + width / 2 + padding
        and cy - height / 2 - padding <= y <= cy + height / 2 + padding
    )


def _host_matches(value: Any, registrable_domain: str) -> bool:
    host = normalize_text(value).rstrip(".")
    expected = normalize_text(registrable_domain).rstrip(".")
    return bool(host == expected or host.endswith(f".{expected}"))


def _parse_raw_mouse_click(value: Any) -> tuple[float, float, str] | None:
    """Parse the complete allowlisted raw click syntax and reject non-finite values."""

    if not isinstance(value, str):
        return None
    match = RAW_MOUSE_CLICK_RE.fullmatch(value.strip())
    if match is None:
        return None
    try:
        x, y = float(match.group("x")), float(match.group("y"))
    except (TypeError, ValueError):
        return None
    return (x, y, "left") if math.isfinite(x) and math.isfinite(y) else None


def _raw_mouse_click_contract(
    raw_action: Any,
    *,
    candidate_point: Sequence[float] | None,
    current_image: Any,
) -> dict[str, Any] | None:
    """Bind a raw pixel click to the candidate's 999-grid coordinate."""

    parsed = _parse_raw_mouse_click(raw_action)
    if parsed is None or candidate_point is None or current_image is None:
        return None
    try:
        candidate_x, candidate_y = float(candidate_point[0]), float(candidate_point[1])
        image_width, image_height = [float(value) for value in current_image.size]
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    raw_x, raw_y, button = parsed
    if not (
        all(
            math.isfinite(value)
            for value in (
                candidate_x,
                candidate_y,
                raw_x,
                raw_y,
                image_width,
                image_height,
            )
        )
        and image_width > 0.0
        and image_height > 0.0
        and 0.0 <= candidate_x <= MARKDOWN_GRID_EXTENT
        and 0.0 <= candidate_y <= MARKDOWN_GRID_EXTENT
        and 0.0 <= raw_x <= image_width
        and 0.0 <= raw_y <= image_height
    ):
        return None
    candidate_pixel = (
        candidate_x * image_width / MARKDOWN_GRID_EXTENT,
        candidate_y * image_height / MARKDOWN_GRID_EXTENT,
    )
    residual = math.hypot(candidate_pixel[0] - raw_x, candidate_pixel[1] - raw_y)
    if residual > V7_RAW_MOUSE_CLICK_PIXEL_RESIDUAL:
        return None
    raw_grid = (
        raw_x * MARKDOWN_GRID_EXTENT / image_width,
        raw_y * MARKDOWN_GRID_EXTENT / image_height,
    )
    return {
        "raw_click": [round(raw_x, 6), round(raw_y, 6)],
        "raw_click_grid": [round(value, 6) for value in raw_grid],
        "candidate_click": [candidate_x, candidate_y],
        "candidate_click_pixel": [round(value, 6) for value in candidate_pixel],
        "current_image_size": [int(image_width), int(image_height)],
        "raw_action_pixel_residual": round(residual, 6),
        "raw_action_pixel_residual_limit": V7_RAW_MOUSE_CLICK_PIXEL_RESIDUAL,
        "raw_button": button,
    }


def _parse_raw_scroll(value: Any) -> tuple[float, float] | None:
    """Parse the complete raw scroll syntax and reject non-finite values."""

    if not isinstance(value, str):
        return None
    match = RAW_SCROLL_RE.fullmatch(value.strip())
    if match is None:
        return None
    try:
        delta_x = float(match.group("delta_x"))
        delta_y = float(match.group("delta_y"))
    except (TypeError, ValueError):
        return None
    return (
        (delta_x, delta_y)
        if math.isfinite(delta_x) and math.isfinite(delta_y)
        else None
    )


def _raw_scroll_contract(
    raw_action: Any,
    *,
    action: dict[str, Any],
    current_image: Any,
) -> dict[str, Any] | None:
    """Bind raw pixel scroll deltas to candidate 999-grid deltas on both axes."""

    parsed = _parse_raw_scroll(raw_action)
    if parsed is None or current_image is None:
        return None
    try:
        candidate_dx = float(action["delta_x"])
        candidate_dy = float(action["delta_y"])
        image_width, image_height = [float(value) for value in current_image.size]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    raw_dx, raw_dy = parsed
    if not (
        all(
            math.isfinite(value)
            for value in (
                candidate_dx,
                candidate_dy,
                raw_dx,
                raw_dy,
                image_width,
                image_height,
            )
        )
        and image_width > 0.0
        and image_height > 0.0
    ):
        return None
    raw_grid = (
        raw_dx * MARKDOWN_GRID_EXTENT / image_width,
        raw_dy * MARKDOWN_GRID_EXTENT / image_height,
    )
    residuals = (
        abs(candidate_dx - raw_grid[0]),
        abs(candidate_dy - raw_grid[1]),
    )
    same_directions = all(
        candidate == raw == 0.0 or candidate * raw > 0.0
        for candidate, raw in ((candidate_dx, raw_dx), (candidate_dy, raw_dy))
    )
    if max(residuals) > V7_RAW_SCROLL_GRID_RESIDUAL or not same_directions:
        return None
    return {
        "raw_scroll": [round(raw_dx, 6), round(raw_dy, 6)],
        "raw_scroll_grid": [round(value, 6) for value in raw_grid],
        "candidate_scroll": [candidate_dx, candidate_dy],
        "current_image_size": [int(image_width), int(image_height)],
        "raw_action_grid_residuals": [round(value, 6) for value in residuals],
        "raw_action_grid_residual_limit": V7_RAW_SCROLL_GRID_RESIDUAL,
        "raw_candidate_directions_agree": same_directions,
    }


def _point_with_axis_padding(
    bbox: Sequence[float],
    point: Sequence[float] | None,
    *,
    x_padding: float,
    y_padding: float,
) -> bool:
    if point is None or len(point) < 2:
        return False
    x, y = float(point[0]), float(point[1])
    return bool(
        math.isfinite(x)
        and math.isfinite(y)
        and float(bbox[0]) - float(bbox[2]) / 2.0 - x_padding
        <= x
        <= float(bbox[0]) + float(bbox[2]) / 2.0 + x_padding
        and float(bbox[1]) - float(bbox[3]) / 2.0 - y_padding
        <= y
        <= float(bbox[1]) + float(bbox[3]) / 2.0 + y_padding
    )


def _action_point(action: dict[str, Any], key: str = "coordinate") -> tuple[float, float] | None:
    value = action.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(coordinate) for coordinate in point) else None


def _bbox_point_distance(bbox: Sequence[float], point: Sequence[float] | None) -> float:
    if point is None or len(point) != 2:
        return math.inf
    return math.hypot(float(bbox[0]) - float(point[0]), float(bbox[1]) - float(point[1]))


def _unique_semantic_vertical_anchors(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
) -> list[tuple[ParsedElementSpan, ParsedElementSpan]]:
    """Collect non-trivial y shifts from one-to-one semantic anchors."""

    current_by_key: dict[tuple[str, str, str], list[ParsedElementSpan]] = defaultdict(list)
    next_by_key: dict[tuple[str, str, str], list[ParsedElementSpan]] = defaultdict(list)
    for element in current.elements:
        if element.element_type != "block":
            current_by_key[element.semantic_key].append(element)
    for element in nxt.elements:
        if element.element_type != "block":
            next_by_key[element.semantic_key].append(element)
    anchors: list[tuple[ParsedElementSpan, ParsedElementSpan]] = []
    for key in sorted(set(current_by_key) & set(next_by_key), key=repr):
        if len(current_by_key[key]) == len(next_by_key[key]) == 1:
            left = current_by_key[key][0]
            right = next_by_key[key][0]
            delta = right.bbox[1] - left.bbox[1]
            if (
                abs(right.bbox[0] - left.bbox[0]) <= 8.0
                and abs(right.bbox[2] - left.bbox[2]) <= 16.0
                and abs(right.bbox[3] - left.bbox[3]) <= 16.0
                and abs(delta) > DEFAULT_GEOMETRY_THRESHOLD
            ):
                anchors.append((left, right))
    return anchors


def _unique_semantic_vertical_deltas(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
) -> list[float]:
    return [
        float(right.bbox[1]) - float(left.bbox[1])
        for left, right in _unique_semantic_vertical_anchors(current, nxt)
    ]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise V4FocusError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def infer_scroll_delta(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    action_delta_y: float,
) -> float | None:
    """Infer the dominant content-layer y shift from unique semantic anchors."""

    deltas = _unique_semantic_vertical_deltas(current, nxt)
    if len(deltas) < 3:
        return None
    bins: dict[int, list[float]] = defaultdict(list)
    for delta in deltas:
        bins[round(delta / SCROLL_BIN_SIZE)].append(delta)
    _, winning = max(
        sorted(bins.items()),
        key=lambda item: (
            len(item[1]),
            -abs(item[0] * SCROLL_BIN_SIZE + action_delta_y),
            -abs(item[0]),
            -item[0],
        ),
    )
    if len(winning) < 3:
        return None
    ordered = sorted(winning)
    mode = ordered[len(ordered) // 2]
    if abs(mode + action_delta_y) > max(12.0, 0.15 * abs(action_delta_y)):
        return None
    return round(mode, 4)


def infer_hover_autoscroll(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
) -> dict[str, float | int] | None:


    anchors = _unique_semantic_vertical_anchors(current, nxt)
    deltas = sorted(
        float(right.bbox[1]) - float(left.bbox[1])
        for left, right in anchors
    )
    if len(deltas) < HOVER_AUTOSCROLL_MIN_ANCHORS:
        return None

    windows: list[list[float]] = []
    right = 0
    for left, lower in enumerate(deltas):
        right = max(right, left)
        while right < len(deltas) and deltas[right] - lower <= 2.0 * HOVER_AUTOSCROLL_CLUSTER_RADIUS:
            right += 1
        windows.append(deltas[left:right])
    winning = max(
        windows,
        key=lambda values: (
            len(values),
            -(values[-1] - values[0]),
            abs(values[len(values) // 2]),
            values[len(values) // 2],
        ),
    )
    required_support = max(
        HOVER_AUTOSCROLL_MIN_ANCHORS,
        math.ceil(HOVER_AUTOSCROLL_MIN_FRACTION * len(deltas)),
    )
    if len(winning) < required_support:
        return None
    outside_support = len(deltas) - len(winning)
    if outside_support and (
        len(winning) - outside_support < 3
        or len(winning) < 1.5 * outside_support
    ):
        return None

    lower_delta, upper_delta = winning[0], winning[-1]
    coarse_anchors = [
        (left, right)
        for left, right in anchors
        if lower_delta
        <= float(right.bbox[1]) - float(left.bbox[1])
        <= upper_delta
    ]
    current_y_values = [float(left.bbox[1]) for left, _ in coarse_anchors]
    if (
        len({round(value, 4) for value in current_y_values}) < 4
        or max(current_y_values) - min(current_y_values) < HOVER_AUTOSCROLL_MIN_Y_SPAN
    ):
        return None

    slopes: list[float] = []
    for index, (left_a, right_a) in enumerate(coarse_anchors):
        x_a, y_a = float(left_a.bbox[1]), float(right_a.bbox[1])
        for left_b, right_b in coarse_anchors[index + 1 :]:
            x_b, y_b = float(left_b.bbox[1]), float(right_b.bbox[1])
            if abs(x_b - x_a) >= 20.0:
                slopes.append((y_b - y_a) / (x_b - x_a))
    if not slopes:
        return None
    scale_y = _median(slopes)
    if not 0.95 <= scale_y <= 1.10:
        return None
    offset_y = _median(
        [float(right.bbox[1]) - scale_y * float(left.bbox[1]) for left, right in coarse_anchors]
    )
    inliers = [
        (left, right)
        for left, right in coarse_anchors
        if abs(float(right.bbox[1]) - (scale_y * float(left.bbox[1]) + offset_y))
        <= HOVER_AUTOSCROLL_MODEL_RESIDUAL
    ]
    required_inliers = max(
        HOVER_AUTOSCROLL_MIN_ANCHORS,
        math.ceil(HOVER_AUTOSCROLL_MODEL_INLIER_FRACTION * len(coarse_anchors)),
        math.ceil(HOVER_AUTOSCROLL_MIN_FRACTION * len(anchors)),
    )
    if len(inliers) < required_inliers:
        return None

    mean_x = sum(float(left.bbox[1]) for left, _ in inliers) / len(inliers)
    mean_y = sum(float(right.bbox[1]) for _, right in inliers) / len(inliers)
    denominator = sum((float(left.bbox[1]) - mean_x) ** 2 for left, _ in inliers)
    if denominator <= 0.0:
        return None
    scale_y = sum(
        (float(left.bbox[1]) - mean_x) * (float(right.bbox[1]) - mean_y)
        for left, right in inliers
    ) / denominator
    if not 0.95 <= scale_y <= 1.10:
        return None
    offset_y = _median(
        [float(right.bbox[1]) - scale_y * float(left.bbox[1]) for left, right in inliers]
    )
    final_inliers = [
        (left, right)
        for left, right in coarse_anchors
        if abs(float(right.bbox[1]) - (scale_y * float(left.bbox[1]) + offset_y))
        <= HOVER_AUTOSCROLL_MODEL_RESIDUAL
    ]
    if len(final_inliers) < required_inliers:
        return None
    shifts = [float(right.bbox[1]) - float(left.bbox[1]) for left, right in final_inliers]
    mode = _median(shifts)
    if abs(mode) < HOVER_AUTOSCROLL_MIN_SHIFT:
        return None
    residuals = [
        abs(float(right.bbox[1]) - (scale_y * float(left.bbox[1]) + offset_y))
        for left, right in final_inliers
    ]
    return {
        "delta_y": round(mode, 4),
        "scale_y": round(scale_y, 8),
        "offset_y": round(offset_y, 4),
        "residual_limit": HOVER_AUTOSCROLL_MODEL_RESIDUAL,
        "max_inlier_residual": round(max(residuals), 4),
        "support": len(final_inliers),
        "coarse_support": len(coarse_anchors),
        "candidate_anchors": len(anchors),
        "support_fraction": round(len(final_inliers) / len(anchors), 6),
        "current_y_span": round(max(current_y_values) - min(current_y_values), 4),
    }


def match_hover_motion_shadows(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    unmatched_current: set[int],
    unmatched_next: set[int],
    motion: dict[str, float | int] | None,
) -> tuple[dict[str, Any], ...]:
    """Build deterministic one-to-one motion-shadow pairs for ADD vetoes."""

    if motion is None:
        return ()
    scale_y = float(motion["scale_y"])
    offset_y = float(motion["offset_y"])
    residual_limit = float(motion["residual_limit"])
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for current_index in sorted(unmatched_current):
        left = current.elements[current_index]
        if left.element_type == "block":
            continue
        for next_index in sorted(unmatched_next):
            right = nxt.elements[next_index]
            if right.element_type not in ADD_ELEMENT_TYPES:
                continue
            dx = float(right.bbox[0]) - float(left.bbox[0])
            predicted_y = scale_y * float(left.bbox[1]) + offset_y
            residual = abs(float(right.bbox[1]) - predicted_y)
            if (
                abs(dx) > 8.0
                or residual > residual_limit
                or abs(float(right.bbox[2]) - float(left.bbox[2])) > 16.0
                or abs(float(right.bbox[3]) - float(left.bbox[3])) > 16.0
            ):
                continue
            similarity = content_similarity(left, right)
            same_payload = left.normalized_content == right.normalized_content
            role_jitter_payload = (
                _strip_visual_role_prefix(left.normalized_content)
                == _strip_visual_role_prefix(right.normalized_content)
            )
            same_type = left.element_type == right.element_type
            compatible_textual_roles = (
                left.element_type in MOTION_TEXTUAL_ROLE_TYPES
                and right.element_type in MOTION_TEXTUAL_ROLE_TYPES
            )
            fuzzy_ocr = bool(
                (same_type or compatible_textual_roles)
                and min(len(left.normalized_content), len(right.normalized_content)) >= 4
                and similarity >= (0.50 if same_type else 0.70)
            )
            if same_payload:
                method, tier = "exact_payload", 0
            elif role_jitter_payload:
                method, tier = "role_jitter_payload", 1
            elif fuzzy_ocr:
                method, tier = ("fuzzy_ocr" if same_type else "fuzzy_ocr_role_jitter"), 2
            else:
                continue
            evidence = {
                "current_element_index": current_index,
                "next_element_index": next_index,
                "method": method,
                "content_similarity": round(similarity, 4),
                "dx": round(dx, 4),
                "predicted_next_y": round(predicted_y, 4),
                "actual_next_y": round(float(right.bbox[1]), 4),
                "model_residual": round(residual, 4),
            }
            key = (
                tier,
                round(residual, 8),
                round(abs(dx), 8),
                -round(similarity, 8),
                current_index,
                next_index,
            )
            candidates.append((key, evidence))

    used_current: set[int] = set()
    used_next: set[int] = set()
    selected: list[dict[str, Any]] = []
    for _, evidence in sorted(candidates, key=lambda item: item[0]):
        current_index = int(evidence["current_element_index"])
        next_index = int(evidence["next_element_index"])
        if current_index in used_current or next_index in used_next:
            continue
        used_current.add(current_index)
        used_next.add(next_index)
        selected.append(evidence)
    return tuple(sorted(selected, key=lambda item: (item["next_element_index"], item["current_element_index"])))


def hover_visual_region_shadow(
    element: ParsedElementSpan,
    current: ParsedMarkdownSpans,
    *,
    motion: dict[str, float | int] | None,
) -> dict[str, Any] | None:
    """Veto text fragments split from a large current visual/card region."""

    if motion is None or element.element_type != "text":
        return None
    scale_y = float(motion["scale_y"])
    offset_y = float(motion["offset_y"])
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for candidate in current.elements:
        if (
            candidate.element_type != "image"
            or float(candidate.bbox[2]) < 150.0
            or float(candidate.bbox[3]) < 80.0
        ):
            continue
        for method, center_y, height in (
            ("fixed_visual_region_fragment", float(candidate.bbox[1]), float(candidate.bbox[3])),
            (
                "projected_visual_region_fragment",
                scale_y * float(candidate.bbox[1]) + offset_y,
                abs(scale_y) * float(candidate.bbox[3]),
            ),
        ):
            dx = float(element.bbox[0]) - float(candidate.bbox[0])
            dy = float(element.bbox[1]) - center_y
            contained = (
                abs(dx) + 0.5 * float(element.bbox[2]) <= 0.5 * float(candidate.bbox[2]) + 16.0
                and abs(dy) + 0.5 * float(element.bbox[3]) <= 0.5 * height + 16.0
            )
            area_ratio = (
                float(element.bbox[2]) * float(element.bbox[3])
                / max(1.0, float(candidate.bbox[2]) * height)
            )
            if not contained or area_ratio > 0.35:
                continue
            evidence = {
                "next_element_index": element.index,
                "current_element_index": candidate.index,
                "method": method,
                "projected_region_center_y": round(center_y, 4),
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "area_ratio": round(area_ratio, 6),
            }
            key = (
                0 if method == "projected_visual_region_fragment" else 1,
                round(abs(dy), 8),
                round(abs(dx), 8),
                candidate.index,
            )
            candidates.append((key, evidence))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def hover_stacked_text_fragment(
    element: ParsedElementSpan,
    nxt: ParsedMarkdownSpans,
    *,
    motion: dict[str, float | int] | None,
) -> dict[str, Any] | None:
    """Veto a short OCR continuation under a much wider text line."""

    if (
        motion is None
        or element.element_type != "text"
        or len(element.normalized_content) > 10
        or any(character.isdigit() for character in element.normalized_content)
        or float(element.bbox[2]) > 100.0
        or float(element.bbox[3]) > 32.0
    ):
        return None
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for candidate in nxt.elements:
        if candidate.index == element.index or candidate.element_type != "text":
            continue
        vertical_gap = float(element.bbox[1]) - float(candidate.bbox[1])
        if not 12.0 <= vertical_gap <= 40.0:
            continue
        if float(candidate.bbox[2]) < 2.0 * float(element.bbox[2]):
            continue
        candidate_left = float(candidate.bbox[0]) - 0.5 * float(candidate.bbox[2])
        candidate_right = float(candidate.bbox[0]) + 0.5 * float(candidate.bbox[2])
        if not candidate_left <= float(element.bbox[0]) <= candidate_right:
            continue
        evidence = {
            "next_element_index": element.index,
            "parent_next_element_index": candidate.index,
            "method": "stacked_short_text_fragment",
            "vertical_gap": round(vertical_gap, 4),
            "width_ratio": round(float(candidate.bbox[2]) / max(1.0, float(element.bbox[2])), 6),
        }
        key = (round(vertical_gap, 8), -round(float(candidate.bbox[2]), 8), candidate.index)
        candidates.append((key, evidence))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _match_record(
    current: ParsedElementSpan,
    nxt: ParsedElementSpan,
    *,
    method: str,
    confidence: float,
) -> ElementMatch:
    return ElementMatch(
        current_index=current.index,
        next_index=nxt.index,
        method=method,
        confidence=round(confidence, 4),
        bbox_iou=round(bbox_iou(current.bbox, nxt.bbox), 4),
        center_distance=round(center_distance(current.bbox, nxt.bbox), 4),
        content_similarity=round(content_similarity(current, nxt), 4),
    )


def _pair_group_by_distance(
    current: Sequence[ParsedElementSpan],
    nxt: Sequence[ParsedElementSpan],
    current_indexes: Iterable[int],
    next_indexes: Iterable[int],
    *,
    method: str,
    unique_confidence: float,
    duplicate_confidence: float,
    duplicate_distance_limit: float,
    duplicate_iou_floor: float,
    scroll_delta: float | None = None,
    minimum_margin: float = 0.10,
) -> list[ElementMatch]:
    left = list(current_indexes)
    right = list(next_indexes)

    def pair_geometry(i: int, j: int) -> tuple[float, float]:
        left_box, right_box = current[i].bbox, nxt[j].bbox
        dx = float(right_box[0]) - float(left_box[0])
        dy = float(right_box[1]) - float(left_box[1])
        residuals = [math.hypot(dx, dy)]
        if scroll_delta is not None:
            residuals.append(math.hypot(dx, dy - scroll_delta))
        return min(residuals), bbox_iou(left_box, right_box)

    def geometry_is_plausible(i: int, j: int) -> bool:
        residual, overlap = pair_geometry(i, j)
        return residual <= duplicate_distance_limit or overlap >= duplicate_iou_floor

    unique = len(left) == 1 and len(right) == 1
    if unique:
        if not geometry_is_plausible(left[0], right[0]):
            return []
        return [
            _match_record(
                current[left[0]],
                nxt[right[0]],
                method=method,
                confidence=unique_confidence,
            )
        ]

    def pair_cost(i: int, j: int) -> float:
        residual, overlap = pair_geometry(i, j)
        return min(1.0, residual / max(1.0, duplicate_distance_limit)) + 0.35 * (1.0 - overlap)

    costs = {(i, j): pair_cost(i, j) for i in left for j in right}
    best_left: dict[int, tuple[float, int, float]] = {}
    best_right: dict[int, tuple[float, int, float]] = {}
    for i in left:
        ranked = sorted((costs[(i, j)], j) for j in right)
        margin = (ranked[1][0] - ranked[0][0]) if len(ranked) > 1 else 1.0
        best_left[i] = ranked[0][0], ranked[0][1], margin
    for j in right:
        ranked = sorted((costs[(i, j)], i) for i in left)
        margin = (ranked[1][0] - ranked[0][0]) if len(ranked) > 1 else 1.0
        best_right[j] = ranked[0][0], ranked[0][1], margin

    output: list[ElementMatch] = []
    for i in left:
        cost, j, left_margin = best_left[i]
        _, reverse_i, right_margin = best_right[j]
        if reverse_i != i or min(left_margin, right_margin) < minimum_margin:
            continue
        if not geometry_is_plausible(i, j):
            continue
        confidence = max(0.75, duplicate_confidence - 0.10 * min(1.0, cost))
        output.append(_match_record(current[i], nxt[j], method=method, confidence=confidence))
    return output


def _month_ordinal(value: str) -> int | None:
    match = MONTH_TITLE_RE.fullmatch(" ".join(str(value).strip().split()))
    if match is None:
        return None
    return int(match.group("year")) * 12 + MONTH_ORDINAL[match.group("month").lower()] - 1


def _action_context_strings(row: dict[str, Any], action: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            values.append(value.strip())

    for value in action.values():
        collect(value)
    provenance = (row.get("meta") or {}).get("v2_provenance") or {}
    collect(provenance.get("action_description"))
    raw_action = provenance.get("raw_action")
    if isinstance(raw_action, dict):
        for key in ("title", "value", "selector", "aria-label", "path"):
            collect(raw_action.get(key))
    else:
        collect(raw_action)
    return values


def _month_action_direction(row: dict[str, Any], action: dict[str, Any]) -> int | None:
    joined = " \n ".join(_action_context_strings(row, action)).lower()
    previous = bool(
        re.search(r"(?:previous|prev)\s*month|month\s*(?:previous|prev)|button-prev", joined)
    )
    following = bool(
        re.search(r"next\s*month|month\s*next|button-next", joined)
    )
    if previous == following:
        return None
    return -1 if previous else 1


def _stable_spatial_anchor_count(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    excluded_current: set[int] | None = None,
    excluded_next: set[int] | None = None,
    center_region: tuple[float, float, float, float] | None = None,
) -> int:
    excluded_current = excluded_current or set()
    excluded_next = excluded_next or set()
    next_by_key: dict[tuple[str, str, str], list[ParsedElementSpan]] = defaultdict(list)
    for element in nxt.elements:
        if element.index not in excluded_next and element.element_type != "block":
            next_by_key[element.semantic_key].append(element)
    used_next: set[int] = set()
    count = 0
    for left in current.elements:
        if left.index in excluded_current or left.element_type == "block":
            continue
        if center_region is not None:
            x_min, y_min, x_max, y_max = center_region
            if not (
                x_min <= float(left.bbox[0]) <= x_max
                and y_min <= float(left.bbox[1]) <= y_max
            ):
                continue
        candidates = [
            right
            for right in next_by_key.get(left.semantic_key, [])
            if right.index not in used_next
            and bbox_iou(left.bbox, right.bbox) >= 0.70
            and center_distance(left.bbox, right.bbox) <= 16.0
        ]
        if len(candidates) != 1:
            continue
        used_next.add(candidates[0].index)
        count += 1
    return count


CALENDAR_WEEKDAY_TOKENS = frozenset(
    {
        "su", "sun", "sunday",
        "mo", "mon", "monday",
        "tu", "tue", "tuesday",
        "we", "wed", "wednesday",
        "th", "thu", "thursday",
        "fr", "fri", "friday",
        "sa", "sat", "saturday",
    }
)


def _calendar_date_grid_evidence(
    parsed: ParsedMarkdownSpans,
    *,
    x_min: float,
    x_mid: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[int, int, int]:
    total_count = 0
    left_count = 0
    right_count = 0
    for element in parsed.elements:
        if element.element_type == "block":
            continue
        x, y = float(element.bbox[0]), float(element.bbox[1])
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        tokens = re.findall(r"[a-z]+|\d+", element.normalized_content)
        if not tokens or not all(
            token in CALENDAR_WEEKDAY_TOKENS
            or token.isdigit() and 1 <= int(token) <= 31
            for token in tokens
        ):
            continue
        total_count += 1
        half_width = 0.5 * float(element.bbox[2])
        if x - half_width < x_mid:
            left_count += 1
        if x + half_width >= x_mid:
            right_count += 1
    return total_count, left_count, right_count


def detect_calendar_month_slot_matches(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    """Reserve consecutive calendar headers by spatial slot before semantic matching."""

    diagnostic: dict[str, Any] = {"detector": "calendar_month_slot", "accepted": False}
    if action_bucket != "click" or normalize_text(action.get("type")) not in {
        "left_click",
        "double_click",
    }:
        diagnostic["rejection"] = "not_click"
        return (), diagnostic
    click_point = _action_point(action)
    if click_point is None:
        diagnostic["rejection"] = "missing_click_point"
        return (), diagnostic

    def month_pairs(parsed: ParsedMarkdownSpans) -> list[tuple[ParsedElementSpan, ParsedElementSpan, int, int]]:
        candidates = [
            (element, ordinal)
            for element in parsed.elements
            if element.element_type in {"text", "link", "button"}
            and (ordinal := _month_ordinal(element.content)) is not None
        ]
        pairs: list[tuple[ParsedElementSpan, ParsedElementSpan, int, int]] = []
        for (left, left_ordinal), (right, right_ordinal) in itertools.combinations(candidates, 2):
            if float(left.bbox[0]) > float(right.bbox[0]):
                left, right = right, left
                left_ordinal, right_ordinal = right_ordinal, left_ordinal
            if (
                right_ordinal == left_ordinal + 1
                and abs(float(left.bbox[1]) - float(right.bbox[1])) <= 16.0
                and float(right.bbox[0]) - float(left.bbox[0]) >= 100.0
            ):
                pairs.append((left, right, left_ordinal, right_ordinal))
        return pairs

    current_pairs = month_pairs(current)
    next_pairs = month_pairs(nxt)
    diagnostic["current_pair_candidates"] = len(current_pairs)
    diagnostic["next_pair_candidates"] = len(next_pairs)
    configurations: list[tuple[Any, ...]] = []
    metadata_direction = _month_action_direction(row, action)
    for current_left, current_right, current_start, _ in current_pairs:
        for next_left, next_right, next_start, _ in next_pairs:
            delta = next_start - current_start
            if delta not in {-1, 1}:
                continue
            residuals = [
                (
                    abs(float(right.bbox[0]) - float(left.bbox[0])),
                    abs(float(right.bbox[1]) - float(left.bbox[1])),
                )
                for left, right in ((current_left, next_left), (current_right, next_right))
            ]
            if any(
                dx > MONTH_SLOT_X_RESIDUAL or dy > MONTH_SLOT_Y_RESIDUAL
                for dx, dy in residuals
            ):
                continue
            inferred_direction = None
            header_y = 0.5 * (float(current_left.bbox[1]) + float(current_right.bbox[1]))
            if abs(click_point[1] - header_y) <= 90.0:
                left_edge = float(current_left.bbox[0]) - 0.5 * float(current_left.bbox[2])
                right_edge = float(current_right.bbox[0]) + 0.5 * float(current_right.bbox[2])
                if click_point[0] < left_edge:
                    inferred_direction = -1
                elif click_point[0] > right_edge:
                    inferred_direction = 1
            if metadata_direction is not None and inferred_direction is not None:
                if metadata_direction != inferred_direction:
                    continue
            direction = metadata_direction if metadata_direction is not None else inferred_direction
            if direction != delta:
                continue
            stable_anchors = _stable_spatial_anchor_count(
                current,
                nxt,
                excluded_current={current_left.index, current_right.index},
                excluded_next={next_left.index, next_right.index},
            )
            if stable_anchors < CALENDAR_MIN_STABLE_LEAF_ANCHORS:
                continue
            header_y = 0.25 * sum(
                float(element.bbox[1])
                for element in (current_left, current_right, next_left, next_right)
            )
            x_min = min(float(current_left.bbox[0]), float(next_left.bbox[0])) - 180.0
            x_max = max(float(current_right.bbox[0]), float(next_right.bbox[0])) + 180.0
            x_mid = 0.25 * sum(
                float(element.bbox[0])
                for element in (current_left, current_right, next_left, next_right)
            )
            date_grid_evidence = (
                _calendar_date_grid_evidence(
                    current,
                    x_min=x_min,
                    x_mid=x_mid,
                    x_max=x_max,
                    y_min=header_y + 35.0,
                    y_max=header_y + 650.0,
                ),
                _calendar_date_grid_evidence(
                    nxt,
                    x_min=x_min,
                    x_mid=x_mid,
                    x_max=x_max,
                    y_min=header_y + 35.0,
                    y_max=header_y + 650.0,
                ),
            )
            if any(
                total < CALENDAR_MIN_DATE_GRID_ITEMS
                or left_count < 2
                or right_count < 2
                for total, left_count, right_count in date_grid_evidence
            ):
                continue
            local_anchor_count = _stable_spatial_anchor_count(
                current,
                nxt,
                excluded_current={current_left.index, current_right.index},
                excluded_next={next_left.index, next_right.index},
                center_region=(
                    x_min,
                    header_y - 160.0,
                    x_max,
                    header_y + 160.0,
                ),
            )
            if (
                stable_anchors == CALENDAR_MIN_STABLE_LEAF_ANCHORS
                and local_anchor_count < CALENDAR_LOW_ANCHOR_MIN_LOCAL_ANCHORS
            ):
                continue
            configurations.append(
                (
                    current_left,
                    current_right,
                    next_left,
                    next_right,
                    delta,
                    inferred_direction,
                    metadata_direction,
                    stable_anchors,
                    local_anchor_count,
                    date_grid_evidence,
                    residuals,
                )
            )
    if len(configurations) != 1:
        diagnostic["rejection"] = "configuration_not_unique"
        diagnostic["valid_configurations"] = len(configurations)
        return (), diagnostic
    (
        current_left,
        current_right,
        next_left,
        next_right,
        delta,
        inferred_direction,
        metadata_direction,
        stable_anchors,
        local_anchor_count,
        date_grid_evidence,
        residuals,
    ) = configurations[0]
    matches = tuple(
        _match_record(left, right, method="calendar_month_slot", confidence=0.99)
        for left, right in ((current_left, next_left), (current_right, next_right))
    )
    diagnostic.update(
        {
            "accepted": True,
            "direction": delta,
            "metadata_direction": metadata_direction,
            "geometry_direction": inferred_direction,
            "stable_anchor_count": stable_anchors,
            "local_anchor_count": local_anchor_count,
            "date_grid_evidence": [
                {"total": total, "left": left_count, "right": right_count}
                for total, left_count, right_count in date_grid_evidence
            ],
            "slot_residuals": residuals,
            "pairs": [
                {"current_element_index": match.current_index, "next_element_index": match.next_index}
                for match in matches
            ],
        }
    )
    return matches, diagnostic


def _action_described_value(row: dict[str, Any]) -> str:
    provenance = (row.get("meta") or {}).get("v2_provenance") or {}
    description = str(provenance.get("action_description") or "")
    match = re.search(r"\bvalue\s+['\"](?P<value>[^'\"]+)['\"]", description, re.IGNORECASE)
    if match:
        return normalize_text(match.group("value"))
    raw_action = provenance.get("raw_action")
    if isinstance(raw_action, dict):
        for key in ("value", "title"):
            value = normalize_text(raw_action.get(key))
            if value:
                return value
    return ""


def _has_strong_autocomplete_option_provenance(row: dict[str, Any]) -> bool:
    raw_action = ((row.get("meta") or {}).get("v2_provenance") or {}).get("raw_action")
    if not isinstance(raw_action, dict):
        return False
    attributes = raw_action.get("attributes")
    data = attributes.get("data") if isinstance(attributes, dict) else None
    role = normalize_text(data.get("role")) if isinstance(data, dict) else ""
    node = data.get("node") if isinstance(data, dict) else None
    node_name = normalize_text(node.get("name")) if isinstance(node, dict) else ""
    selector = normalize_text(raw_action.get("selector"))
    return bool(
        role == "option"
        or "downshift-" in selector and "-item-" in selector
        or node_name == "option"
    )


def _autocomplete_query_related(query: str, candidate: str) -> bool:
    left = "".join(re.findall(r"[a-z0-9]+", normalize_text(query)))
    right = "".join(re.findall(r"[a-z0-9]+", normalize_text(candidate)))
    if len(left) < 5 or not right:
        return False
    common = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        common += 1
    return common >= max(4, math.ceil(0.80 * len(left)))


def detect_autocomplete_commit_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    """Reserve input-slot identity when a clicked autocomplete option is committed."""

    diagnostic: dict[str, Any] = {"detector": "autocomplete_commit", "accepted": False}
    if action_bucket != "click" or normalize_text(action.get("type")) != "left_click":
        diagnostic["rejection"] = "not_left_click"
        return (), diagnostic
    click_point = _action_point(action)
    if click_point is None:
        diagnostic["rejection"] = "missing_click_point"
        return (), diagnostic
    strong_option_provenance = _has_strong_autocomplete_option_provenance(row)
    suggestion_types = {"text", "button"}
    input_types = {"text", "input"}
    if strong_option_provenance:
        input_types.add("button")
    clicked = [
        element
        for element in current.elements
        if element.element_type in suggestion_types
        and _bbox_contains_point(element.bbox, click_point, padding=2.0)
        and element.normalized_content
    ]
    if not clicked:
        diagnostic["rejection"] = "no_clicked_leaf"
        return (), diagnostic
    minimum_area = min(float(element.bbox[2]) * float(element.bbox[3]) for element in clicked)
    clicked = [
        element
        for element in clicked
        if float(element.bbox[2]) * float(element.bbox[3]) <= 1.10 * minimum_area
    ]
    if len(clicked) != 1:
        diagnostic["rejection"] = "clicked_leaf_not_unique"
        return (), diagnostic
    suggestion = clicked[0]
    described_value = _action_described_value(row)
    if not described_value:
        diagnostic["rejection"] = "missing_action_value"
        return (), diagnostic
    if described_value and described_value != suggestion.normalized_content:
        diagnostic["rejection"] = "action_value_disagrees"
        return (), diagnostic

    next_values = [
        element
        for element in nxt.elements
        if element.element_type in input_types
        and element.normalized_content == suggestion.normalized_content
    ]
    triplets: list[tuple[ParsedElementSpan, ParsedElementSpan]] = []
    for next_value in next_values:
        if (
            bbox_iou(suggestion.bbox, next_value.bbox) >= 0.15
            or center_distance(suggestion.bbox, next_value.bbox) <= 40.0
        ):
            continue
        input_candidates = []
        for candidate in current.elements:
            if candidate.index == suggestion.index or candidate.element_type not in input_types:
                continue
            value = candidate.normalized_content
            if (
                not value
                or value == suggestion.normalized_content
                or not suggestion.normalized_content.startswith(value)
                or len(value) < 2
            ):
                continue
            overlap = bbox_iou(candidate.bbox, next_value.bbox)
            distance = center_distance(candidate.bbox, next_value.bbox)
            if overlap < AUTOCOMPLETE_INPUT_IOU or distance > AUTOCOMPLETE_INPUT_DISTANCE:
                continue
            if float(suggestion.bbox[1]) - float(candidate.bbox[1]) < 40.0:
                continue
            if abs(float(suggestion.bbox[0]) - float(candidate.bbox[0])) > 120.0:
                continue
            input_candidates.append(candidate)
        if len(input_candidates) == 1:
            triplets.append((input_candidates[0], next_value))
    if len(triplets) != 1:
        diagnostic["rejection"] = "input_triplet_not_unique"
        diagnostic["valid_triplets"] = len(triplets)
        return (), diagnostic
    input_element, next_value = triplets[0]
    if input_element.element_type != next_value.element_type:
        diagnostic["rejection"] = "input_slot_type_changed"
        return (), diagnostic

    popup_siblings = [
        element
        for element in current.elements
        if element.index not in {suggestion.index, input_element.index}
        and element.element_type in suggestion_types
        and element.normalized_content
        and float(element.bbox[1]) >= float(input_element.bbox[1]) + 20.0
        and abs(float(element.bbox[1]) - float(suggestion.bbox[1])) <= 300.0
        and abs(float(element.bbox[0]) - float(suggestion.bbox[0])) <= 120.0
    ]

    def popup_slot_still_present(element: ParsedElementSpan) -> bool:
        return any(
            right.element_type in suggestion_types
            and (
                bbox_iou(right.bbox, element.bbox) >= 0.30
                or center_distance(right.bbox, element.bbox) <= 25.0
            )
            and (
                right.normalized_content == element.normalized_content
                or content_similarity(element, right) >= 0.75
            )
            for right in nxt.elements
        )

    if popup_slot_still_present(suggestion):
        diagnostic["rejection"] = "clicked_suggestion_slot_remains"
        return (), diagnostic
    vanished_siblings = [
        element
        for element in popup_siblings
        if not popup_slot_still_present(element)
    ]
    if not popup_siblings or len(vanished_siblings) != len(popup_siblings):
        diagnostic["rejection"] = "popup_did_not_close"
        diagnostic["popup_sibling_count"] = len(popup_siblings)
        diagnostic["vanished_sibling_count"] = len(vanished_siblings)
        return (), diagnostic
    query_sibling_count = sum(
        _autocomplete_query_related(input_element.normalized_content, element.normalized_content)
        for element in vanished_siblings
    )
    if query_sibling_count < 2 and not strong_option_provenance:
        diagnostic["rejection"] = "popup_not_query_related"
        diagnostic["query_sibling_count"] = query_sibling_count
        return (), diagnostic
    stable_anchors = _stable_spatial_anchor_count(
        current,
        nxt,
        excluded_current={
            input_element.index,
            suggestion.index,
            *(item.index for item in popup_siblings),
        },
        excluded_next={next_value.index},
    )
    if stable_anchors < 4:
        diagnostic["rejection"] = "insufficient_stable_background"
        diagnostic["stable_anchor_count"] = stable_anchors
        return (), diagnostic
    match = _match_record(
        input_element,
        next_value,
        method="autocomplete_commit",
        confidence=0.99,
    )
    diagnostic.update(
        {
            "accepted": True,
            "action_described_value": described_value,
            "clicked_suggestion_index": suggestion.index,
            "input_current_index": input_element.index,
            "input_next_index": next_value.index,
            "popup_sibling_count": len(popup_siblings),
            "vanished_sibling_count": len(vanished_siblings),
            "query_sibling_count": query_sibling_count,
            "strong_option_provenance": strong_option_provenance,
            "stable_anchor_count": stable_anchors,
        }
    )
    return (match,), diagnostic


def _strip_typed_slot_caret(value: str) -> str:
    return _strip_ui_query_marker(value).rstrip("|_").strip()


def _raw_input_target_rect(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    provenance = (row.get("meta") or {}).get("v2_provenance") or {}
    raw = provenance.get("raw_action")
    if not isinstance(raw, dict):
        return None
    attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    data = attributes.get("data") if isinstance(attributes.get("data"), dict) else {}
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    node_name = normalize_text(node.get("name"))
    role = normalize_text(data.get("role") or attributes.get("role"))
    if node_name not in {"input", "textarea"} and role not in {"textbox", "searchbox"}:
        return None
    rect = raw.get("rect")
    viewport = raw.get("viewport")
    if not isinstance(rect, dict) or not isinstance(viewport, dict):
        return None
    try:
        viewport_width = float(viewport["width"])
        viewport_height = float(viewport["height"])
        raw_left = float(rect.get("left", rect.get("x")))
        raw_top = float(rect.get("top", rect.get("y")))
        raw_width = float(rect["width"])
        raw_height = float(rect["height"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    numeric = (viewport_width, viewport_height, raw_left, raw_top, raw_width, raw_height)
    if not all(math.isfinite(value) for value in numeric):
        return None
    if not (
        viewport_width > 0.0
        and viewport_height > 0.0
        and 1.0 <= raw_width <= viewport_width * 1.05
        and 1.0 <= raw_height <= viewport_height * 1.05
        and -0.05 * viewport_width <= raw_left <= 1.05 * viewport_width
        and -0.05 * viewport_height <= raw_top <= 1.05 * viewport_height
        and raw_left + raw_width >= -0.05 * viewport_width
        and raw_top + raw_height >= -0.05 * viewport_height
    ):
        return None
    left = raw_left * MARKDOWN_GRID_EXTENT / viewport_width
    top = raw_top * MARKDOWN_GRID_EXTENT / viewport_height
    width = raw_width * MARKDOWN_GRID_EXTENT / viewport_width
    height = raw_height * MARKDOWN_GRID_EXTENT / viewport_height
    return left + width / 2.0, top + height / 2.0, width, height


def _strict_overstock_input_target_rect(
    row: dict[str, Any],
    *,
    current_image: Any,
) -> tuple[tuple[float, float, float, float] | None, dict[str, Any]]:
    provenance = (row.get("meta") or {}).get("v2_provenance") or {}
    raw = provenance.get("raw_action")
    rect = raw.get("rect") if isinstance(raw, dict) else None
    viewport = raw.get("viewport") if isinstance(raw, dict) else None
    if not isinstance(rect, dict) or not isinstance(viewport, dict) or current_image is None:
        return None, {"rejection": "missing_rect_viewport_or_image"}
    rect_keys = ("x", "y", "left", "top", "right", "bottom", "width", "height")
    viewport_keys = ("width", "height")
    if any(
        isinstance(rect.get(key), bool)
        or not isinstance(rect.get(key), (int, float))
        for key in rect_keys
    ) or any(
        isinstance(viewport.get(key), bool)
        or not isinstance(viewport.get(key), (int, float))
        for key in viewport_keys
    ):
        return None, {"rejection": "rect_fields_not_numeric"}
    try:
        values = {key: float(rect[key]) for key in rect_keys}
        viewport_width = float(viewport["width"])
        viewport_height = float(viewport["height"])
        image_width, image_height = [float(value) for value in current_image.size]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None, {"rejection": "rect_fields_not_numeric"}
    if not all(
        math.isfinite(value)
        for value in (*values.values(), viewport_width, viewport_height, image_width, image_height)
    ):
        return None, {"rejection": "rect_fields_not_finite"}
    closure_residuals = {
        "x_left": abs(values["x"] - values["left"]),
        "y_top": abs(values["y"] - values["top"]),
        "right": abs(values["right"] - (values["left"] + values["width"])),
        "bottom": abs(values["bottom"] - (values["top"] + values["height"])),
    }
    if not (
        values["width"] > 0.0
        and values["height"] > 0.0
        and viewport_width > 0.0
        and viewport_height > 0.0
        and image_width > 0.0
        and image_height > 0.0
        and max(closure_residuals.values()) <= V7_RAW_RECT_CLOSURE_TOLERANCE
    ):
        return None, {
            "rejection": "rect_not_positive_or_closed",
            "closure_residuals": closure_residuals,
        }
    tolerance = V7_RAW_RECT_BOUNDARY_TOLERANCE
    if not (
        -tolerance <= values["left"]
        and -tolerance <= values["top"]
        and values["right"] <= viewport_width + tolerance
        and values["bottom"] <= viewport_height + tolerance
        and values["right"] <= image_width + tolerance
        and values["bottom"] <= image_height + tolerance
    ):
        return None, {"rejection": "rect_outside_viewport_or_screenshot"}
    target = (
        (values["left"] + values["width"] / 2.0)
        * MARKDOWN_GRID_EXTENT
        / viewport_width,
        (values["top"] + values["height"] / 2.0)
        * MARKDOWN_GRID_EXTENT
        / viewport_height,
        values["width"] * MARKDOWN_GRID_EXTENT / viewport_width,
        values["height"] * MARKDOWN_GRID_EXTENT / viewport_height,
    )
    return target, {
        "rect": values,
        "viewport": [viewport_width, viewport_height],
        "current_image_size": [image_width, image_height],
        "closure_residuals": closure_residuals,
        "closure_tolerance": V7_RAW_RECT_CLOSURE_TOLERANCE,
        "boundary_tolerance": V7_RAW_RECT_BOUNDARY_TOLERANCE,
    }


def _strict_local_retention_ratio(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    action_bucket: str,
) -> float:
    baseline_matches = match_elements(current, nxt, action_bucket=action_bucket)
    next_leaf_count = sum(element.element_type != "block" for element in nxt.elements)
    retained_leaf_count = sum(
        nxt.elements[match.next_index].element_type != "block"
        and (
            match.method in {"exact", "semantic", "state_identity"}
            or (match.method == "spatial_mutual" and match.bbox_iou >= 0.75)
        )
        for match in baseline_matches
    )
    return retained_leaf_count / max(1, next_leaf_count)


def detect_typed_direct_slot_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    """Reserve the input slot when its next value is explicitly supplied by type_text."""

    diagnostic: dict[str, Any] = {"detector": "typed_direct_slot", "accepted": False}
    if action_bucket != "text_key" or normalize_text(action.get("type")) != "type_text":
        diagnostic["rejection"] = "not_type_text"
        return (), diagnostic
    typed = _typed_text(action)
    if not 2 <= len(typed) <= 128:
        diagnostic["rejection"] = "typed_value_length"
        return (), diagnostic
    next_values = [
        element
        for element in nxt.elements
        if element.element_type in TYPED_DIRECT_ELEMENT_TYPES
        and _ui_value_matches_typed(element.normalized_content, typed)
    ]
    if len(next_values) != 1:
        diagnostic["rejection"] = "next_value_not_unique"
        diagnostic["next_value_count"] = len(next_values)
        return (), diagnostic
    right = next_values[0]
    raw_target = _raw_input_target_rect(row)
    retention_ratio = _strict_local_retention_ratio(
        current,
        nxt,
        action_bucket=action_bucket,
    )
    if retention_ratio < 0.20:
        diagnostic["rejection"] = "page_replacement"
        diagnostic["retention_ratio"] = retention_ratio
        return (), diagnostic
    candidates: list[tuple[ParsedElementSpan, tuple[str, ...]]] = []
    unstable_query_marker_pairs = 0
    for left in current.elements:
        if (
            left.element_type not in TYPED_DIRECT_ELEMENT_TYPES
            or not left.normalized_content
            or _ui_value_matches_typed(left.normalized_content, typed)
        ):
            continue
        rendered = _strip_typed_slot_caret(left.normalized_content)
        if not rendered or len(rendered) > 64:
            continue
        if vertical_overlap_ratio(left.bbox, right.bbox) < 0.75:
            continue
        if center_distance(left.bbox, right.bbox) > 80.0:
            continue
        if not _bbox_center_inside(right.bbox, left.bbox, padding=12.0):
            continue
        height_ratio = float(right.bbox[3]) / max(1.0, float(left.bbox[3]))
        if not 0.50 <= height_ratio <= 2.0:
            continue
        if not _typed_pair_has_predictable_target(
            left.normalized_content,
            right.normalized_content,
            typed,
        ):
            unstable_query_marker_pairs += 1
            continue
        prefix_branch = len(rendered) >= 2 and typed.startswith(rendered) and typed != rendered
        raw_target_branch = bool(
            raw_target is not None
            and _bbox_center_inside(left.bbox, raw_target, padding=8.0)
            and _bbox_center_inside(right.bbox, raw_target, padding=8.0)
        )
        route_placeholder_branch = bool(
            ROUTE_INPUT_PLACEHOLDER_RE.fullmatch(rendered)
            and bbox_iou(left.bbox, right.bbox) >= 0.10
        )
        branches = tuple(
            name
            for name, accepted in (
                ("prefix_extension", prefix_branch),
                ("raw_input_target", raw_target_branch),
                ("route_input_placeholder", route_placeholder_branch),
            )
            if accepted
        )
        if branches:
            candidates.append((left, branches))
    if len(candidates) != 1:
        diagnostic["rejection"] = (
            "renderer_query_marker_introduced_or_changed"
            if not candidates and unstable_query_marker_pairs
            else "current_slot_not_unique"
        )
        diagnostic["current_slot_count"] = len(candidates)
        diagnostic["unstable_query_marker_pairs"] = unstable_query_marker_pairs
        return (), diagnostic
    left, branches = candidates[0]
    stable_anchors = _stable_spatial_anchor_count(
        current,
        nxt,
        excluded_current={left.index},
        excluded_next={right.index},
    )
    if stable_anchors < 4:
        diagnostic["rejection"] = "insufficient_stable_background"
        diagnostic["stable_anchor_count"] = stable_anchors
        return (), diagnostic
    match = _match_record(left, right, method="typed_direct_slot", confidence=0.98)
    diagnostic.update(
        {
            "accepted": True,
            "typed_value": typed,
            "slot_branches": list(branches),
            "current_element_index": left.index,
            "next_element_index": right.index,
            "bbox_iou": bbox_iou(left.bbox, right.bbox),
            "center_distance": center_distance(left.bbox, right.bbox),
            "vertical_overlap_ratio": vertical_overlap_ratio(left.bbox, right.bbox),
            "stable_anchor_count": stable_anchors,
            "retention_ratio": retention_ratio,
            "raw_input_target_rect": list(raw_target) if raw_target is not None else None,
        }
    )
    return (match,), diagnostic


def detect_typed_overstock_query_marker_slot_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
    existing_matches: Sequence[ElementMatch] = (),
    current_image: Any,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    """Recover only Overstock's proven, space-separated ``Q `` query marker."""

    diagnostic: dict[str, Any] = {
        "detector": "typed_overstock_query_marker_slot",
        "accepted": False,
    }
    if action_bucket != "text_key" or normalize_text(action.get("type")) != "type_text":
        diagnostic["rejection"] = "not_type_text"
        return (), diagnostic
    raw_candidate_text = action.get("text")
    if (
        not isinstance(raw_candidate_text, str)
        or raw_candidate_text != raw_candidate_text.strip()
    ):
        diagnostic["rejection"] = "candidate_text_not_canonical"
        return (), diagnostic
    typed = _typed_text(action)
    if not 2 <= len(typed) <= 128:
        diagnostic["rejection"] = "typed_value_length"
        return (), diagnostic

    provenance = (row.get("meta") or {}).get("v2_provenance") or {}
    raw = provenance.get("raw_action")
    if not isinstance(raw, dict):
        diagnostic["rejection"] = "missing_raw_action"
        return (), diagnostic
    attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    data = attributes.get("data") if isinstance(attributes.get("data"), dict) else {}
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    action_rule = raw.get("actionRuleSetting")
    action_rule_type = (
        action_rule.get("type") if isinstance(action_rule, dict) else None
    )
    action_rule_type = action_rule_type if isinstance(action_rule_type, dict) else {}
    raw_contract = {
        "current_domain": normalize_text(provenance.get("current_domain")),
        "next_domain": normalize_text(provenance.get("next_domain")),
        "host": normalize_text(raw.get("host")),
        "path": str(raw.get("path") or "").strip(),
        "selector": str(raw.get("selector") or "").strip(),
        "raw_type": normalize_text(raw.get("type")),
        "node_name": normalize_text(node.get("name")),
        "role": normalize_text(data.get("role")),
        "operation_type": normalize_text(action_rule_type.get("operationType")),
        "raw_value": normalize_text(raw.get("value")),
    }
    if not (
        raw_contract["current_domain"] == V7_OVERSTOCK_HOST
        and raw_contract["next_domain"] == V7_OVERSTOCK_HOST
        and raw_contract["host"] == V7_OVERSTOCK_HOST
        and raw_contract["path"] == V7_OVERSTOCK_INPUT_PATH
        and raw_contract["selector"] == V7_OVERSTOCK_INPUT_SELECTOR
        and raw_contract["raw_type"] == "type"
        and raw_contract["node_name"] == "input"
        and raw_contract["role"] == "textbox"
        and raw_contract["operation_type"] == "replace"
        and raw_contract["raw_value"] == typed
    ):
        diagnostic.update(
            {
                "rejection": "overstock_raw_provenance_mismatch",
                "raw_contract": raw_contract,
            }
        )
        return (), diagnostic
    raw_target, raw_rect_contract = _strict_overstock_input_target_rect(
        row,
        current_image=current_image,
    )
    if raw_target is None:
        diagnostic["rejection"] = "invalid_raw_input_rect"
        diagnostic["raw_rect_contract"] = raw_rect_contract
        return (), diagnostic

    current_slots = [
        element
        for element in current.elements
        if element.element_type in TYPED_DIRECT_ELEMENT_TYPES
        and _bbox_inside(element.bbox, raw_target, padding=V7_RAW_RECT_OCR_PADDING)
    ]
    next_slots = [
        element
        for element in nxt.elements
        if element.element_type in TYPED_DIRECT_ELEMENT_TYPES
        and _bbox_inside(element.bbox, raw_target, padding=V7_RAW_RECT_OCR_PADDING)
    ]
    diagnostic.update(
        {
            "current_raw_rect_slot_count": len(current_slots),
            "next_raw_rect_slot_count": len(next_slots),
            "raw_input_target_rect": list(raw_target),
            "raw_rect_contract": raw_rect_contract,
            "raw_rect_ocr_padding": V7_RAW_RECT_OCR_PADDING,
        }
    )
    if len(current_slots) != 1 or len(next_slots) != 1:
        diagnostic["rejection"] = "raw_rect_slot_not_unique"
        return (), diagnostic
    left, right = current_slots[0], next_slots[0]
    raw_placeholders = {
        "Search Crazy-Good Deals",
        "Q Search Crazy-Good Deals",
        "QSearch Crazy-Good Deals",
    }
    if (
        left.content.strip() not in raw_placeholders
        or left.normalized_content not in V7_OVERSTOCK_QUERY_PLACEHOLDERS
    ):
        diagnostic["rejection"] = "current_placeholder_not_allowlisted"
        return (), diagnostic
    expected_next = f"q {typed}"
    expected_next_raw = f"Q {raw_candidate_text}"
    if (
        right.content.strip() != expected_next_raw
        or right.normalized_content != expected_next
    ):
        diagnostic["rejection"] = "next_query_marker_not_exact"
        diagnostic["expected_next_value"] = expected_next
        diagnostic["expected_next_raw"] = expected_next_raw
        return (), diagnostic

    height_ratio = float(right.bbox[3]) / max(1.0, float(left.bbox[3]))
    geometry_valid = bool(
        vertical_overlap_ratio(left.bbox, right.bbox) >= 0.75
        and center_distance(left.bbox, right.bbox) <= 80.0
        and _bbox_center_inside(right.bbox, left.bbox, padding=12.0)
        and 0.50 <= height_ratio <= 2.0
    )
    if not geometry_valid:
        diagnostic.update(
            {
                "rejection": "strict_slot_geometry_failed",
                "vertical_overlap_ratio": round(
                    vertical_overlap_ratio(left.bbox, right.bbox), 6
                ),
                "center_distance": round(center_distance(left.bbox, right.bbox), 6),
                "height_ratio": round(height_ratio, 6),
            }
        )
        return (), diagnostic
    retention_ratio = _strict_local_retention_ratio(
        current,
        nxt,
        action_bucket=action_bucket,
    )
    stable_anchors = _stable_spatial_anchor_count(
        current,
        nxt,
        excluded_current={left.index},
        excluded_next={right.index},
    )
    if retention_ratio < V6_MIN_RETENTION_RATIO or stable_anchors < 4:
        diagnostic.update(
            {
                "rejection": "insufficient_retained_page",
                "retention_ratio": round(retention_ratio, 6),
                "stable_anchor_count": stable_anchors,
            }
        )
        return (), diagnostic
    if any(
        match.current_index == left.index or match.next_index == right.index
        for match in existing_matches
    ):
        diagnostic.update(
            {
                "rejection": "existing_typed_direct_match",
                "current_element_index": left.index,
                "next_element_index": right.index,
            }
        )
        return (), diagnostic

    match = _match_record(
        left,
        right,
        method="typed_overstock_query_marker_slot",
        confidence=0.99,
    )
    diagnostic.update(
        {
            "accepted": True,
            "typed_value": typed,
            "expected_next_value": expected_next,
            "expected_next_raw": expected_next_raw,
            "current_element_index": left.index,
            "next_element_index": right.index,
            "bbox_iou": bbox_iou(left.bbox, right.bbox),
            "center_distance": center_distance(left.bbox, right.bbox),
            "vertical_overlap_ratio": vertical_overlap_ratio(left.bbox, right.bbox),
            "height_ratio": height_ratio,
            "stable_anchor_count": stable_anchors,
            "retention_ratio": retention_ratio,
            "raw_contract": raw_contract,
        }
    )
    return (match,), diagnostic


def detect_click_text_toggle_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    action: dict[str, Any],
    action_bucket: str,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    """Reserve an explicitly clicked, exact lexical Details/Close state toggle."""

    diagnostic: dict[str, Any] = {"detector": "click_text_toggle", "accepted": False}
    if action_bucket != "click" or normalize_text(action.get("type")) not in {
        "left_click",
        "double_click",
    }:
        diagnostic["rejection"] = "not_supported_click"
        return (), diagnostic
    point = _action_point(action)
    if point is None:
        diagnostic["rejection"] = "missing_click_point"
        return (), diagnostic
    clicked = [
        element
        for element in current.elements
        if element.element_type in {"text", "link", "button"}
        and _bbox_contains_point(element.bbox, point, padding=2.0)
        and element.normalized_content in {pair[0] for pair in CLICK_TEXT_TOGGLE_PAIRS}
    ]
    if not clicked:
        diagnostic["rejection"] = "no_clicked_toggle"
        return (), diagnostic
    minimum_area = min(float(element.bbox[2]) * float(element.bbox[3]) for element in clicked)
    clicked = [
        element
        for element in clicked
        if float(element.bbox[2]) * float(element.bbox[3]) <= 1.10 * minimum_area
    ]
    if len(clicked) != 1:
        diagnostic["rejection"] = "clicked_toggle_not_unique"
        diagnostic["clicked_toggle_count"] = len(clicked)
        return (), diagnostic
    left = clicked[0]
    candidates = [
        right
        for right in nxt.elements
        if right.element_type == left.element_type
        and (left.normalized_content, right.normalized_content) in CLICK_TEXT_TOGGLE_PAIRS
        and (
            bbox_iou(left.bbox, right.bbox) >= 0.35
            or center_distance(left.bbox, right.bbox) <= 30.0
        )
    ]
    if len(candidates) != 1:
        diagnostic["rejection"] = "next_toggle_not_unique"
        diagnostic["next_toggle_count"] = len(candidates)
        return (), diagnostic
    right = candidates[0]
    retention_ratio = _strict_local_retention_ratio(
        current,
        nxt,
        action_bucket=action_bucket,
    )
    if retention_ratio < 0.20:
        diagnostic["rejection"] = "page_replacement"
        diagnostic["retention_ratio"] = retention_ratio
        return (), diagnostic
    stable_anchors = _stable_spatial_anchor_count(
        current,
        nxt,
        excluded_current={left.index},
        excluded_next={right.index},
    )
    if stable_anchors < 4:
        diagnostic["rejection"] = "insufficient_stable_background"
        diagnostic["stable_anchor_count"] = stable_anchors
        return (), diagnostic
    match = _match_record(left, right, method="click_text_toggle", confidence=0.99)
    diagnostic.update(
        {
            "accepted": True,
            "current_element_index": left.index,
            "next_element_index": right.index,
            "pair": [left.normalized_content, right.normalized_content],
            "stable_anchor_count": stable_anchors,
            "retention_ratio": retention_ratio,
        }
    )
    return (match,), diagnostic


def _v6_local_evidence(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    action_bucket: str,
    excluded_current: set[int] | None = None,
    excluded_next: set[int] | None = None,
) -> tuple[float, int]:
    retention_ratio = _strict_local_retention_ratio(
        current,
        nxt,
        action_bucket=action_bucket,
    )
    stable_anchors = _stable_spatial_anchor_count(
        current,
        nxt,
        excluded_current=excluded_current,
        excluded_next=excluded_next,
    )
    return retention_ratio, stable_anchors


def _v6_high_retention(
    retention_ratio: float,
    stable_anchors: int,
) -> bool:
    return bool(
        retention_ratio >= V6_MIN_RETENTION_RATIO
        and stable_anchors >= V6_MIN_STABLE_ANCHORS
    )


def _drag_action_points(
    action: dict[str, Any],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    return (
        _action_point(action, "from_coordinate"),
        _action_point(action, "to_coordinate"),
    )


def _dominant_page_translation(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
) -> tuple[float, float, int] | None:
    current_by_key: dict[tuple[str, str, str], list[ParsedElementSpan]] = defaultdict(list)
    next_by_key: dict[tuple[str, str, str], list[ParsedElementSpan]] = defaultdict(list)
    for element in current.elements:
        if element.element_type != "block":
            current_by_key[element.semantic_key].append(element)
    for element in nxt.elements:
        if element.element_type != "block":
            next_by_key[element.semantic_key].append(element)
    translations: list[tuple[float, float]] = []
    for key in set(current_by_key) & set(next_by_key):
        if len(current_by_key[key]) != 1 or len(next_by_key[key]) != 1:
            continue
        left, right = current_by_key[key][0], next_by_key[key][0]
        if (
            abs(float(left.bbox[2]) - float(right.bbox[2])) <= 24.0
            and abs(float(left.bbox[3]) - float(right.bbox[3])) <= 24.0
        ):
            translations.append(
                (
                    float(right.bbox[0]) - float(left.bbox[0]),
                    float(right.bbox[1]) - float(left.bbox[1]),
                )
            )
    bins: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for dx, dy in translations:
        bins[(round(dx / 4.0), round(dy / 4.0))].append((dx, dy))
    if not bins:
        return None
    _, winning = max(sorted(bins.items()), key=lambda item: (len(item[1]), -abs(item[0][0]), -abs(item[0][1])))
    if len(winning) < 3:
        return None
    dx = _median([item[0] for item in winning])
    dy = _median([item[1] for item in winning])
    inliers = [
        item
        for item in translations
        if math.hypot(item[0] - dx, item[1] - dy) <= 6.0
    ]
    if len(inliers) < 3:
        return None
    return round(dx, 4), round(dy, 4), len(inliers)


def _translated_bbox(
    bbox: Sequence[float],
    translation: tuple[float, float, int] | None,
) -> tuple[float, float, float, float]:
    dx, dy = (translation[0], translation[1]) if translation is not None else (0.0, 0.0)
    return (
        float(bbox[0]) + dx,
        float(bbox[1]) + dy,
        float(bbox[2]),
        float(bbox[3]),
    )


def _v6_pair_diagnostic(
    detector: str,
    left: ParsedElementSpan,
    right: ParsedElementSpan,
    *,
    retention_ratio: float,
    stable_anchors: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "detector": detector,
        "accepted": True,
        "current_element_index": left.index,
        "next_element_index": right.index,
        "retention_ratio": round(retention_ratio, 6),
        "stable_anchor_count": stable_anchors,
        "center_distance": round(center_distance(left.bbox, right.bbox), 4),
        "horizontal_overlap": round(horizontal_overlap_ratio(left.bbox, right.bbox), 4),
        **(extra or {}),
    }


def detect_search_mode_slot_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
    current_image: Any,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    diagnostic: dict[str, Any] = {"detector": "search_mode_slot", "accepted": False}
    metadata = row.get("meta") or {}
    provenance = metadata.get("v2_provenance") if isinstance(metadata, dict) else None
    raw_action = provenance.get("raw_action") if isinstance(provenance, dict) else None
    action_description = normalize_text(
        provenance.get("action_description") if isinstance(provenance, dict) else ""
    )
    current_domain = normalize_text(
        provenance.get("current_domain") if isinstance(provenance, dict) else ""
    )
    next_domain = normalize_text(
        provenance.get("next_domain") if isinstance(provenance, dict) else ""
    )
    if not (
        _host_matches(current_domain, "foxsports.com")
        and _host_matches(next_domain, "foxsports.com")
        and action_description in {"click on input element", "click on button element"}
    ):
        diagnostic["rejection"] = "missing_foxsports_click_provenance"
        return (), diagnostic
    if action_bucket != "click" or normalize_text(action.get("type")) != "left_click":
        diagnostic["rejection"] = "not_left_click"
        return (), diagnostic
    click_point = _action_point(action)
    if click_point is None:
        diagnostic["rejection"] = "missing_click_point"
        return (), diagnostic
    raw_click_contract = _raw_mouse_click_contract(
        raw_action,
        candidate_point=click_point,
        current_image=current_image,
    )
    if raw_click_contract is None:
        diagnostic["rejection"] = "invalid_raw_mouse_click"
        return (), diagnostic
    raw_grid_point = raw_click_contract["raw_click_grid"]
    lefts = [
        element
        for element in current.elements
        if element.normalized_content == "popular searches"
    ]
    rights = [
        element
        for element in nxt.elements
        if _strip_visual_role_prefix(element.normalized_content) == "recent searches"
    ]
    search_targets = [
        element
        for element in current.elements
        if (
            element.element_type == "input"
            or _strip_visual_role_prefix(element.normalized_content).startswith("q search ")
        )
        and element not in lefts
        and _point_with_axis_padding(
            element.bbox,
            click_point,
            x_padding=V7_FOX_ACTION_TARGET_X_PADDING,
            y_padding=V7_FOX_ACTION_TARGET_Y_PADDING,
        )
        and _point_with_axis_padding(
            element.bbox,
            raw_grid_point,
            x_padding=V7_FOX_ACTION_TARGET_X_PADDING,
            y_padding=V7_FOX_ACTION_TARGET_Y_PADDING,
        )
    ]
    pairs = [
        (left, right)
        for left in lefts
        for right in rights
        if center_distance(left.bbox, right.bbox) <= 80.0
        and horizontal_overlap_ratio(left.bbox, right.bbox) >= 0.60
    ]
    diagnostic.update(
        {
            "pair_candidates": len(pairs),
            "search_target_count": len(search_targets),
        }
    )
    if len(pairs) != 1 or len(search_targets) != 1:
        diagnostic["rejection"] = "slot_or_action_target_not_unique"
        return (), diagnostic
    left, right = pairs[0]
    retention_ratio, stable_anchors = _v6_local_evidence(
        current,
        nxt,
        action_bucket=action_bucket,
        excluded_current={left.index},
        excluded_next={right.index},
    )
    if not (
        retention_ratio >= V6_MIN_RETENTION_RATIO
        and stable_anchors >= V6_SEARCH_MODE_MIN_STABLE_ANCHORS
    ):
        diagnostic.update(
            {
                "rejection": "insufficient_retained_page",
                "retention_ratio": round(retention_ratio, 6),
                "stable_anchor_count": stable_anchors,
            }
        )
        return (), diagnostic
    match = _match_record(left, right, method="search_mode_slot", confidence=0.99)
    diagnostic = _v6_pair_diagnostic(
        "search_mode_slot",
        left,
        right,
        retention_ratio=retention_ratio,
        stable_anchors=stable_anchors,
        extra={
            "action_target_index": search_targets[0].index,
            "fox_current_host": current_domain,
            "fox_next_host": next_domain,
            **raw_click_contract,
            "action_target_x_padding": V7_FOX_ACTION_TARGET_X_PADDING,
            "action_target_y_padding": V7_FOX_ACTION_TARGET_Y_PADDING,
        },
    )
    return (match,), diagnostic


def detect_sticky_pos_header_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
    current_image: Any,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    diagnostic: dict[str, Any] = {"detector": "sticky_pos_header", "accepted": False}
    metadata = row.get("meta") or {}
    provenance = metadata.get("v2_provenance") if isinstance(metadata, dict) else None
    raw_action = provenance.get("raw_action") if isinstance(provenance, dict) else None
    action_description = normalize_text(
        provenance.get("action_description") if isinstance(provenance, dict) else ""
    )
    current_domain = normalize_text(
        provenance.get("current_domain") if isinstance(provenance, dict) else ""
    )
    next_domain = normalize_text(
        provenance.get("next_domain") if isinstance(provenance, dict) else ""
    )
    if not (
        _host_matches(current_domain, "merriam-webster.com")
        and _host_matches(next_domain, "merriam-webster.com")
    ):
        diagnostic["rejection"] = "missing_merriam_scroll_provenance"
        return (), diagnostic
    raw_scroll_contract = _raw_scroll_contract(
        raw_action,
        action=action,
        current_image=current_image,
    )
    if raw_scroll_contract is None:
        diagnostic["rejection"] = "invalid_raw_scroll"
        return (), diagnostic
    try:
        delta_y = float(action.get("delta_y", 0.0))
    except (TypeError, ValueError):
        delta_y = 0.0
    if (
        action_bucket != "scroll"
        or normalize_text(action.get("type")) != "scroll"
        or abs(delta_y) < 300.0
    ):
        diagnostic["rejection"] = "not_large_scroll"
        return (), diagnostic
    lefts = [
        element
        for element in current.elements
        if element.element_type == "text"
        and POS_HEADER_RE.fullmatch(element.normalized_content)
        and float(element.bbox[1]) <= 150.0
    ]
    rights = [
        element
        for element in nxt.elements
        if element.element_type == "text"
        and POS_HEADER_RE.fullmatch(element.normalized_content)
        and float(element.bbox[1]) <= 150.0
    ]
    pairs = [
        (left, right)
        for left in lefts
        for right in rights
        if left.normalized_content != right.normalized_content
        and center_distance(left.bbox, right.bbox) <= 22.0
        and horizontal_overlap_ratio(left.bbox, right.bbox) >= 0.95
    ]
    if len(pairs) != 1:
        diagnostic.update({"rejection": "sticky_slot_not_unique", "pair_candidates": len(pairs)})
        return (), diagnostic
    left, right = pairs[0]
    pos_rank = {"adjective": 0, "verb": 1, "noun": 2}
    left_pos = left.normalized_content.split()[0]
    right_pos = right.normalized_content.split()[0]
    rank_delta = pos_rank.get(right_pos, -99) - pos_rank.get(left_pos, -99)
    expected_direction = 1 if rank_delta > 0 else -1 if rank_delta < 0 else 0
    action_direction = 1 if delta_y > 0 else -1
    expected_min_scroll = 250.0 * abs(rank_delta)
    expected_max_scroll = 350.0 * abs(rank_delta)
    target_body = [
        element
        for element in nxt.elements
        if element.index != right.index and float(element.bbox[1]) > 150.0
    ]
    target_structure_indexes: list[int] = []
    if right_pos == "verb":
        required = {"transitive verb", "intransitive verb"}
        found = {
            element.normalized_content
            for element in target_body
            if element.normalized_content in required
        }
        if found == required:
            target_structure_indexes = [
                element.index
                for element in target_body
                if element.normalized_content in required
            ]
    elif right_pos == "noun":
        target_structure_indexes = [
            element.index
            for element in target_body
            if element.normalized_content == "plural"
        ]
    elif right_pos == "adjective":
        adverb = [
            element.index
            for element in target_body
            if re.fullmatch(r"\S+\s+adverb", element.normalized_content)
        ]
        derived_noun = [
            element.index
            for element in target_body
            if re.fullmatch(r"\S+\s+noun", element.normalized_content)
        ]
        if adverb and derived_noun:
            target_structure_indexes = [adverb[0], derived_noun[0]]
    expected_description = "scroll down by" if delta_y > 0 else "scroll up by"
    if not (
        left_pos in pos_rank
        and right_pos in pos_rank
        and 1 <= abs(rank_delta) <= 2
        and action_direction == expected_direction
        and expected_min_scroll <= abs(delta_y) <= expected_max_scroll
        and action_description.startswith(expected_description)
        and target_structure_indexes
    ):
        diagnostic.update(
            {
                "rejection": "pos_sequence_or_scroll_direction_invalid",
                "left_pos": left_pos,
                "right_pos": right_pos,
                "rank_delta": rank_delta,
                "target_structure_count": len(target_structure_indexes),
            }
        )
        return (), diagnostic
    current_by_key: dict[tuple[str, str, str], list[ParsedElementSpan]] = defaultdict(list)
    next_by_key: dict[tuple[str, str, str], list[ParsedElementSpan]] = defaultdict(list)
    for element in current.elements:
        if element.index != left.index and element.element_type != "block":
            current_by_key[element.semantic_key].append(element)
    for element in nxt.elements:
        if element.index != right.index and element.element_type != "block":
            next_by_key[element.semantic_key].append(element)
    fixed_anchors = [
        (current_by_key[key][0], next_by_key[key][0])
        for key in set(current_by_key) & set(next_by_key)
        if len(current_by_key[key]) == len(next_by_key[key]) == 1
        and max(float(current_by_key[key][0].bbox[1]), float(next_by_key[key][0].bbox[1])) <= 180.0
        and center_distance(current_by_key[key][0].bbox, next_by_key[key][0].bbox) <= 40.0
    ]
    if len(fixed_anchors) < V6_STICKY_MIN_FIXED_ANCHORS:
        diagnostic.update(
            {
                "rejection": "insufficient_fixed_scroll_anchors",
                "fixed_scroll_anchor_count": len(fixed_anchors),
            }
        )
        return (), diagnostic
    match = _match_record(left, right, method="sticky_pos_header", confidence=0.99)
    diagnostic = _v6_pair_diagnostic(
        "sticky_pos_header",
        left,
        right,
        retention_ratio=_strict_local_retention_ratio(current, nxt, action_bucket=action_bucket),
        stable_anchors=len(fixed_anchors),
        extra={
            "fixed_scroll_anchor_count": len(fixed_anchors),
            "action_delta_y": delta_y,
            "left_pos": left_pos,
            "right_pos": right_pos,
            "rank_delta": rank_delta,
            "target_structure_indexes": target_structure_indexes,
            "target_structure_count": len(target_structure_indexes),
            "sticky_exception": True,
            "merriam_current_host": current_domain,
            "merriam_next_host": next_domain,
            **raw_scroll_contract,
        },
    )
    return (match,), diagnostic


def _detect_drag_direct_slot_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    action: dict[str, Any],
    action_bucket: str,
    detector: str,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    diagnostic: dict[str, Any] = {"detector": detector, "accepted": False}
    if action_bucket != "other" or normalize_text(action.get("type")) != "drag":
        diagnostic["rejection"] = "not_drag"
        return (), diagnostic
    start, _ = _drag_action_points(action)
    if start is None:
        diagnostic["rejection"] = "missing_drag_start"
        return (), diagnostic
    _, end = _drag_action_points(action)
    if (
        end is None
        or math.hypot(end[0] - start[0], end[1] - start[1])
        < V6_DIRECT_DRAG_MIN_DISPLACEMENT
    ):
        diagnostic["rejection"] = "drag_displacement_too_small"
        return (), diagnostic
    translation = _dominant_page_translation(current, nxt)
    endpoint_padding = (
        V6_DRAG_SORT_ENDPOINT_PADDING
        if detector == "drag_sort_slot"
        else V6_DRAG_FORM_ENDPOINT_PADDING
    )
    pairs: list[tuple[ParsedElementSpan, ParsedElementSpan]] = []
    for left in current.elements:
        if not _bbox_contains_point(left.bbox, start, padding=18.0):
            continue
        if detector == "drag_sort_slot":
            if (
                left.element_type not in MOTION_TEXTUAL_ROLE_TYPES
                or "sort" not in left.normalized_content
                or ":" not in left.normalized_content
                or len(left.normalized_content) > 80
            ):
                continue
        else:
            if left.element_type not in {"input", "select"}:
                continue
        for right in nxt.elements:
            if right.normalized_content == left.normalized_content:
                continue
            if detector == "drag_sort_slot":
                if (
                    right.element_type not in MOTION_TEXTUAL_ROLE_TYPES
                    or "sort" not in right.normalized_content
                    or ":" not in right.normalized_content
                    or len(right.normalized_content) > 80
                ):
                    continue
            elif right.element_type != left.element_type:
                continue
            projected_left = _translated_bbox(left.bbox, translation)
            if (
                _bbox_contains_point(right.bbox, end, padding=endpoint_padding)
                and
                min(
                    center_distance(left.bbox, right.bbox),
                    center_distance(projected_left, right.bbox),
                )
                <= 28.0
                and horizontal_overlap_ratio(projected_left, right.bbox) >= 0.50
                and vertical_overlap_ratio(projected_left, right.bbox) >= 0.50
            ):
                pairs.append((left, right))
    diagnostic["pair_candidates"] = len(pairs)
    if len(pairs) != 1:
        diagnostic["rejection"] = "direct_slot_not_unique"
        return (), diagnostic
    left, right = pairs[0]
    retention_ratio, stable_anchors = _v6_local_evidence(
        current,
        nxt,
        action_bucket=action_bucket,
        excluded_current={left.index},
        excluded_next={right.index},
    )
    if translation is not None:
        stable_anchors = max(stable_anchors, int(translation[2]))
    coherent_translation_retention = bool(
        translation is not None and int(translation[2]) >= V6_MIN_STABLE_ANCHORS
    )
    if not (
        _v6_high_retention(retention_ratio, stable_anchors)
        or coherent_translation_retention
    ):
        diagnostic.update(
            {
                "rejection": "insufficient_retained_page",
                "retention_ratio": round(retention_ratio, 6),
                "stable_anchor_count": stable_anchors,
            }
        )
        return (), diagnostic
    method = detector
    match = _match_record(left, right, method=method, confidence=0.99)
    diagnostic = _v6_pair_diagnostic(
        detector,
        left,
        right,
        retention_ratio=retention_ratio,
        stable_anchors=stable_anchors,
        extra={
            "drag_start": list(start),
            "coherent_translation_retention": coherent_translation_retention,
        },
    )
    diagnostic["dominant_translation"] = (
        list(translation) if translation is not None else None
    )
    return (match,), diagnostic


def detect_drag_binary_selector_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    action: dict[str, Any],
    action_bucket: str,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    diagnostic: dict[str, Any] = {"detector": "drag_binary_selector_slot", "accepted": False}
    if action_bucket != "other" or normalize_text(action.get("type")) != "drag":
        diagnostic["rejection"] = "not_drag"
        return (), diagnostic
    start, end = _drag_action_points(action)
    if (
        start is None
        or end is None
        or math.hypot(end[0] - start[0], end[1] - start[1]) < 20.0
    ):
        diagnostic["rejection"] = "drag_displacement_too_small"
        return (), diagnostic
    labels_current = [element for element in current.elements if element.normalized_content == "new/used"]
    labels_next = [element for element in nxt.elements if element.normalized_content == "new/used"]
    values_current = [element for element in current.elements if element.normalized_content == "new"]
    values_next = [element for element in nxt.elements if element.normalized_content == "used"]
    configurations = [
        (label_left, label_right, left, right)
        for label_left in labels_current
        for label_right in labels_next
        for left in values_current
        for right in values_next
        if center_distance(label_left.bbox, label_right.bbox) <= 10.0
        and center_distance(left.bbox, right.bbox) <= 10.0
        and horizontal_overlap_ratio(left.bbox, right.bbox) >= 0.80
        and any(
            _bbox_contains_point(label_left.bbox, point, padding=20.0)
            for point in (start, end)
            if point is not None
        )
    ]
    diagnostic["valid_configurations"] = len(configurations)
    if len(configurations) != 1:
        diagnostic["rejection"] = "binary_labeled_slot_not_unique"
        return (), diagnostic
    label_left, _, left, right = configurations[0]
    retention_ratio, stable_anchors = _v6_local_evidence(
        current,
        nxt,
        action_bucket=action_bucket,
        excluded_current={left.index},
        excluded_next={right.index},
    )
    if not _v6_high_retention(retention_ratio, stable_anchors):
        diagnostic.update(
            {
                "rejection": "insufficient_retained_page",
                "retention_ratio": round(retention_ratio, 6),
                "stable_anchor_count": stable_anchors,
            }
        )
        return (), diagnostic
    match = _match_record(left, right, method="drag_binary_selector_slot", confidence=0.99)
    diagnostic = _v6_pair_diagnostic(
        "drag_binary_selector_slot",
        left,
        right,
        retention_ratio=retention_ratio,
        stable_anchors=stable_anchors,
        extra={"selector_label_index": label_left.index},
    )
    return (match,), diagnostic


def detect_v6_update_matches(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
    current_image: Any,
) -> tuple[tuple[ElementMatch, ...], dict[str, dict[str, Any]]]:
    results = {
        "search_mode_slot": detect_search_mode_slot_match(
            current,
            nxt,
            row=row,
            action=action,
            action_bucket=action_bucket,
            current_image=current_image,
        ),
        "sticky_pos_header": detect_sticky_pos_header_match(
            current,
            nxt,
            row=row,
            action=action,
            action_bucket=action_bucket,
            current_image=current_image,
        ),
        "drag_sort_slot": _detect_drag_direct_slot_match(
            current,
            nxt,
            action=action,
            action_bucket=action_bucket,
            detector="drag_sort_slot",
        ),
        "drag_form_control_slot": _detect_drag_direct_slot_match(
            current,
            nxt,
            action=action,
            action_bucket=action_bucket,
            detector="drag_form_control_slot",
        ),
        "drag_binary_selector_slot": detect_drag_binary_selector_match(
            current, nxt, action=action, action_bucket=action_bucket
        ),
    }
    matches: list[ElementMatch] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    used_current: set[int] = set()
    used_next: set[int] = set()
    for name, (candidate_matches, diagnostic) in results.items():
        diagnostics[name] = diagnostic
        for match in candidate_matches:
            if match.current_index in used_current or match.next_index in used_next:
                diagnostics[name] = {
                    **diagnostic,
                    "accepted": False,
                    "rejection": "reserved_match_conflict",
                }
                continue
            used_current.add(match.current_index)
            used_next.add(match.next_index)
            matches.append(match)
    return tuple(matches), diagnostics


def match_elements(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    action_bucket: str,
    scroll_delta: float | None = None,
    reserved_matches: Sequence[ElementMatch] = (),
) -> tuple[ElementMatch, ...]:
    matched_current: set[int] = set()
    matched_next: set[int] = set()
    matches: list[ElementMatch] = []
    for match in reserved_matches:
        if match.current_index in matched_current or match.next_index in matched_next:
            raise V4FocusError("reserved action matches are not one-to-one")
        matched_current.add(match.current_index)
        matched_next.add(match.next_index)
        matches.append(match)

    exact_current: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    exact_next: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for element in current.elements:
        if element.index not in matched_current:
            exact_current[element.semantic_key + element.bbox].append(element.index)
    for element in nxt.elements:
        if element.index not in matched_next:
            exact_next[element.semantic_key + element.bbox].append(element.index)
    for key in sorted(set(exact_current) & set(exact_next), key=repr):
        pairs = _pair_group_by_distance(
            current.elements,
            nxt.elements,
            exact_current[key],
            exact_next[key],
            method="exact",
            unique_confidence=1.0,
            duplicate_confidence=0.99,
            duplicate_distance_limit=0.0,
            duplicate_iou_floor=1.0,
            scroll_delta=None,
        )
        for pair in pairs:
            matched_current.add(pair.current_index)
            matched_next.add(pair.next_index)
            matches.append(pair)

    semantic_current: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    semantic_next: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for element in current.elements:
        if element.index not in matched_current:
            semantic_current[element.semantic_key].append(element.index)
    for element in nxt.elements:
        if element.index not in matched_next:
            semantic_next[element.semantic_key].append(element.index)
    for key in sorted(set(semantic_current) & set(semantic_next), key=repr):
        if key[1] == "block" and key[0].startswith("block:"):
            distance_limit, iou_floor = 60.0, 0.45
        else:
            distance_limit, iou_floor = 180.0, 0.10
        pairs = _pair_group_by_distance(
            current.elements,
            nxt.elements,
            semantic_current[key],
            semantic_next[key],
            method="semantic",
            unique_confidence=0.97,
            duplicate_confidence=0.88,
            duplicate_distance_limit=distance_limit,
            duplicate_iou_floor=iou_floor,
            scroll_delta=scroll_delta,
        )
        for pair in pairs:
            matched_current.add(pair.current_index)
            matched_next.add(pair.next_index)
            matches.append(pair)

    identity_current: dict[tuple[str, str], list[int]] = defaultdict(list)
    identity_next: dict[tuple[str, str], list[int]] = defaultdict(list)
    for element in current.elements:
        if element.index not in matched_current:
            identity_current[element.identity_key].append(element.index)
    for element in nxt.elements:
        if element.index not in matched_next:
            identity_next[element.identity_key].append(element.index)
    for key in sorted(set(identity_current) & set(identity_next), key=repr):
        pairs = _pair_group_by_distance(
            current.elements,
            nxt.elements,
            identity_current[key],
            identity_next[key],
            method="state_identity",
            unique_confidence=0.95,
            duplicate_confidence=0.86,
            duplicate_distance_limit=50.0,
            duplicate_iou_floor=0.45,
            scroll_delta=scroll_delta,
        )
        for pair in pairs:
            if pair.bbox_iou < 0.45 and pair.center_distance > 50.0:
                continue
            matched_current.add(pair.current_index)
            matched_next.add(pair.next_index)
            matches.append(pair)

    if action_bucket == "text_key":
        candidates: list[tuple[float, int, int]] = []
        for left in current.elements:
            if left.index in matched_current or left.element_type == "block":
                continue
            for right in nxt.elements:
                if right.index in matched_next or right.element_type == "block":
                    continue
                overlap = bbox_iou(left.bbox, right.bbox)
                if overlap < 0.75 or left.element_type != right.element_type:
                    continue
                similarity = content_similarity(left, right)
                score = 0.70 * overlap + 0.30 * similarity
                if score >= 0.65:
                    candidates.append((score, left.index, right.index))
        best_left: dict[int, tuple[float, int]] = {}
        best_right: dict[int, tuple[float, int]] = {}
        for score, left, right in candidates:
            if score > best_left.get(left, (-1.0, -1))[0]:
                best_left[left] = (score, right)
            if score > best_right.get(right, (-1.0, -1))[0]:
                best_right[right] = (score, left)
        for score, left, right in sorted(candidates, reverse=True):
            if left in matched_current or right in matched_next:
                continue
            if best_left.get(left) != (score, right) or best_right.get(right) != (score, left):
                continue
            pair = _match_record(
                current.elements[left],
                nxt.elements[right],
                method="spatial_mutual",
                confidence=min(0.92, 0.75 + 0.20 * score),
            )
            matched_current.add(left)
            matched_next.add(right)
            matches.append(pair)

    return tuple(sorted(matches, key=lambda item: (item.next_index, item.current_index)))


def _grid_crop_box(bbox: Sequence[float], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    cx, cy, width, height = [float(value) for value in bbox]
    image_width, image_height = size
    x1 = max(0, min(image_width, round((cx - width / 2) / 999.0 * image_width)))
    y1 = max(0, min(image_height, round((cy - height / 2) / 999.0 * image_height)))
    x2 = max(0, min(image_width, round((cx + width / 2) / 999.0 * image_width)))
    y2 = max(0, min(image_height, round((cy + height / 2) / 999.0 * image_height)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def patch_difference(
    left_image: Any,
    right_image: Any,
    left_bbox: Sequence[float],
    right_bbox: Sequence[float] | None = None,
) -> float | None:
    if left_image is None or right_image is None:
        return None
    left_box = _grid_crop_box(left_bbox, left_image.size)
    right_box = _grid_crop_box(right_bbox or left_bbox, right_image.size)
    if left_box is None or right_box is None:
        return None
    try:
        from PIL import ImageChops, ImageStat

        left = left_image.crop(left_box).convert("RGB").resize((64, 64))
        right = right_image.crop(right_box).convert("RGB").resize((64, 64))
        statistics = ImageStat.Stat(ImageChops.difference(left, right))
        return round(sum(statistics.mean) / (3.0 * 255.0), 6)
    except Exception:
        return None


def _action_payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("candidate_action")
    if isinstance(value, dict):
        return value
    value = row.get("candidate_action_json")
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _typed_text(action: dict[str, Any]) -> str:
    for key in ("text", "value", "input_text"):
        if action.get(key) not in (None, ""):
            return normalize_text(action[key])
    return ""


def _strip_ui_query_marker(value: str) -> str:
    """Remove renderer-only query-role markers from Markdown, never from actions."""
    normalized = normalize_text(value)
    for prefix in ("q ", "q: "):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip()
    return normalized


def _ui_query_marker(value: str) -> str | None:
    """Return the exact renderer query marker, if one prefixes the UI value."""

    normalized = normalize_text(value)
    if normalized.startswith("q: "):
        return "q:"
    if normalized.startswith("q "):
        return "q"
    return None


def _ui_value_matches_typed(value: str, typed: str) -> bool:
    normalized_value = normalize_text(value)
    normalized_typed = normalize_text(typed)
    return bool(
        normalized_typed
        and (
            normalized_value == normalized_typed
            or _strip_ui_query_marker(normalized_value) == normalized_typed
        )
    )


def _typed_pair_has_predictable_target(
    current_value: str,
    next_value: str,
    typed: str,
) -> bool:
    """Accept an action-exact target without amplifying a newly fused UI marker.

    A literal Next value is clean supervision.  A renderer-only ``Q`` marker in
    Next is also predictable when the same marker already existed in Current;
    otherwise the extra target token is OCR/segmentation churn and remains base.
    """

    normalized_typed = normalize_text(typed)
    normalized_next = normalize_text(next_value)
    if not normalized_typed:
        return False
    if normalized_next == normalized_typed:
        return True
    next_marker = _ui_query_marker(normalized_next)
    return bool(
        next_marker is not None
        and _strip_ui_query_marker(normalized_next) == normalized_typed
        and _ui_query_marker(current_value) == next_marker
    )


def _element_dict(element: ParsedElementSpan) -> dict[str, Any]:
    payload = asdict(element)
    payload["bbox"] = list(element.bbox)
    payload["normalized_content"] = element.normalized_content
    return payload


def _focus_update(
    current: ParsedElementSpan,
    nxt: ParsedElementSpan,
    match: ElementMatch,
    *,
    action_bucket: str,
    action: dict[str, Any],
    current_image: Any,
    next_image: Any,
    scroll_delta: float | None,
    typed_echo_candidate: bool = False,
) -> dict[str, Any] | None:
    changed_fields: list[str] = []
    spans: list[CharSpan] = []
    reasons: list[str] = []
    action_type = normalize_text(action.get("type"))
    click_point = _action_point(action)
    special_candidate = bool(
        match.method in {
            "calendar_month_slot",
            "autocomplete_commit",
            "click_text_toggle",
            *V6_UPDATE_METHODS,
        }
        and current.normalized_content != nxt.normalized_content
    )
    click_state_candidate = bool(
        action_bucket == "click"
        and action_type == "left_click"
        and current.state != nxt.state
        and current.content == nxt.content
        and current.element_type == nxt.element_type
        and current.element_type in STATE_FOCUS_ELEMENT_TYPES
        and (
            _bbox_contains_point(current.bbox, click_point, padding=10.0)
            or _bbox_contains_point(nxt.bbox, click_point, padding=10.0)
        )
    )
    typed_candidate = bool(
        action_bucket == "text_key"
        and action_type == "type_text"
        and current.normalized_content != nxt.normalized_content
    )
    if not (special_candidate or click_state_candidate or typed_candidate):
        return None
    region_diff = patch_difference(current_image, next_image, nxt.bbox)
    identity_diff = patch_difference(current_image, next_image, current.bbox, nxt.bbox)

    if (
        special_candidate
        and (
            (region_diff is not None and region_diff >= SPECIAL_MATCH_VISUAL_DIFF)
            or (identity_diff is not None and identity_diff >= SPECIAL_MATCH_VISUAL_DIFF)
        )
    ):
        changed_fields.append("content")
        spans.append(nxt.content_span)
        reason_by_method = {
            "calendar_month_slot": "calendar_month_slot_content_update",
            "autocomplete_commit": "autocomplete_commit_input_content_update",
            "click_text_toggle": "clicked_exact_text_toggle_content_update",
            "typed_overstock_query_marker_slot": "typed_value_overstock_query_marker_slot_update",
            "search_mode_slot": "search_mode_recent_header_content_update",
            "sticky_pos_header": "sticky_pos_header_content_update",
            "drag_sort_slot": "drag_sort_content_update",
            "drag_form_control_slot": "drag_form_control_content_update",
            "drag_binary_selector_slot": "drag_binary_selector_content_update",
        }
        reasons.append(reason_by_method[match.method])

    if (
        click_state_candidate
    ):
        if nxt.state_span is not None and region_diff is not None and region_diff >= 0.01:
            changed_fields.append("state")
            spans.append(nxt.state_span)
            reasons.append("state_flip_at_click_target")

    if typed_candidate:
        typed = _typed_text(action)
        typed_evidence = bool(
            typed
            and _typed_pair_has_predictable_target(
                current.normalized_content,
                nxt.normalized_content,
                typed,
            )
            and not _ui_value_matches_typed(current.normalized_content, typed)
        )
        visually_changed = bool(
            (region_diff is not None and region_diff >= 0.008)
            or (identity_diff is not None and identity_diff >= 0.02)
        )
        spatial_identity = (
            match.method == "typed_direct_slot"
            or (
                match.method == "spatial_mutual"
                and match.bbox_iou >= 0.75
                and match.confidence >= 0.85
            )
        )
        if typed_evidence and spatial_identity and visually_changed:
            changed_fields.append("content")
            spans.append(nxt.content_span)
            reasons.append(
                "typed_value_exact_direct_slot_update"
                if match.method == "typed_direct_slot"
                else "typed_value_exact_spatial_update"
            )
        elif typed_echo_candidate and visually_changed:
            changed_fields.append("content")
            spans.append(nxt.content_span)
            reasons.append("typed_value_deterministic_echo_update")

    if not spans:
        return None
    unique_spans = sorted({(span.start, span.end) for span in spans})
    return {
        "kind": "UPDATE",
        "next_element_index": nxt.index,
        "current_element_index": current.index,
        "fields": sorted(set(changed_fields)),
        "spans": [{"start": start, "end": end} for start, end in unique_spans],
        "confidence": round(min(0.99, match.confidence), 4),
        "reasons": reasons,
        "evidence": {
            "match_method": match.method,
            "bbox_iou": match.bbox_iou,
            "content_similarity": match.content_similarity,
            "same_region_pixel_diff": region_diff,
            "old_to_new_crop_pixel_diff": identity_diff,
        },
    }


def _v7_update_priority(candidate: dict[str, Any]) -> int:
    reasons = {str(reason) for reason in candidate.get("reasons") or ()}
    if reasons & V7_SPECIAL_UPDATE_REASONS:
        return V7_SPECIAL_PRIORITY
    if reasons & V7_TYPED_UPDATE_REASONS:
        return V7_TYPED_PRIORITY
    if V7_GENERIC_STATE_REASON in reasons:
        return V7_GENERIC_STATE_PRIORITY
    return 0


def _select_v7_focus_units(
    update_candidates: dict[int, dict[str, Any]],
    add_candidate: dict[str, Any] | None,
) -> tuple[set[int], int | None, dict[str, Any]]:
    """Apply the two-unit cap to explicit atomic groups and ranked mechanisms."""

    groups: list[dict[str, Any]] = []
    consumed_updates: set[int] = set()
    add_consumed = False

    add_index = (
        int(add_candidate["next_element_index"])
        if add_candidate is not None
        else None
    )
    add_reason = (
        str((add_candidate.get("reasons") or [""])[0])
        if add_candidate is not None
        else ""
    )

    def update_has_reason(index: int, reason: str) -> bool:
        return reason in {
            str(value)
            for value in (update_candidates.get(index) or {}).get("reasons") or ()
        }

    def append_group(
        *,
        name: str,
        priority: int,
        update_indices: Sequence[int] = (),
        grouped_add_index: int | None = None,
        atomic: bool,
    ) -> None:
        members = [
            {
                "kind": "UPDATE",
                "next_element_index": int(index),
                "reason": str(
                    ((update_candidates[index].get("reasons") or [""])[0])
                ),
            }
            for index in sorted(update_indices)
        ]
        if grouped_add_index is not None:
            members.append(
                {
                    "kind": "ADD",
                    "next_element_index": grouped_add_index,
                    "reason": add_reason,
                }
            )
        groups.append(
            {
                "name": name,
                "priority": priority,
                "atomic": atomic,
                "members": members,
                "stable_next_index": min(
                    member["next_element_index"] for member in members
                ),
            }
        )

    # Fox search activation is one semantic event: the header UPDATE and the
    # newly exposed clear-all ADD are selected together or not at all.
    if (
        add_candidate is not None
        and add_reason == "search_mode_recent_clear_all_anchor_add"
    ):
        add_consumed = True
        linked_header = (add_candidate.get("evidence") or {}).get(
            "search_mode_next_index"
        )
        if (
            isinstance(linked_header, int)
            and not isinstance(linked_header, bool)
            and linked_header in update_candidates
            and update_has_reason(
                linked_header, "search_mode_recent_header_content_update"
            )
        ):
            consumed_updates.add(linked_header)
            append_group(
                name="fox_search_mode_bundle",
                priority=V7_ATOMIC_PRIORITY,
                update_indices=(linked_header,),
                grouped_add_index=add_index,
                atomic=True,
            )
        else:
            groups.append(
                {
                    "name": "fox_search_mode_bundle",
                    "priority": V7_ATOMIC_PRIORITY,
                    "atomic": True,
                    "members": [],
                    "stable_next_index": add_index,
                    "selected": False,
                    "rejection": "missing_header_focus_partner",
                }
            )

    # The calendar detector establishes an ordered two-month rerender.  A
    # partial calendar label would teach an arbitrary half of that mechanism.
    calendar_indices = sorted(
        index
        for index, candidate in update_candidates.items()
        if "calendar_month_slot_content_update"
        in {str(reason) for reason in candidate.get("reasons") or ()}
    )
    if calendar_indices:
        consumed_updates.update(calendar_indices)
        if len(calendar_indices) == 2:
            append_group(
                name="calendar_two_month_bundle",
                priority=V7_ATOMIC_PRIORITY,
                update_indices=calendar_indices,
                atomic=True,
            )
        else:
            groups.append(
                {
                    "name": "calendar_two_month_bundle",
                    "priority": V7_ATOMIC_PRIORITY,
                    "atomic": True,
                    "members": [
                        {
                            "kind": "UPDATE",
                            "next_element_index": index,
                            "reason": "calendar_month_slot_content_update",
                        }
                        for index in calendar_indices
                    ],
                    "stable_next_index": calendar_indices[0],
                    "selected": False,
                    "rejection": "requires_exactly_two_updates",
                }
            )

    for index in sorted(set(update_candidates) - consumed_updates):
        candidate = update_candidates[index]
        reason = str((candidate.get("reasons") or [""])[0])
        append_group(
            name=f"update:{reason}:{index}",
            priority=_v7_update_priority(candidate),
            update_indices=(index,),
            atomic=False,
        )

    if add_candidate is not None and not add_consumed:
        append_group(
            name=f"add:{add_reason}:{add_index}",
            priority=V7_SPECIAL_PRIORITY,
            grouped_add_index=add_index,
            atomic=False,
        )

    selectable = [group for group in groups if "selected" not in group]
    selectable.sort(
        key=lambda group: (
            -int(group["priority"]),
            int(group["stable_next_index"]),
            str(group["name"]),
        )
    )
    selected_updates: set[int] = set()
    selected_add: int | None = None
    selected_count = 0
    for group in selectable:
        members = group["members"]
        group_adds = sum(member["kind"] == "ADD" for member in members)
        if selected_count + len(members) > V6_MAX_FOCUS_UNITS:
            group["selected"] = False
            group["rejection"] = "global_focus_cap"
            continue
        if group_adds and selected_add is not None:
            group["selected"] = False
            group["rejection"] = "global_add_cap"
            continue
        group["selected"] = True
        selected_count += len(members)
        for member in members:
            if member["kind"] == "ADD":
                selected_add = int(member["next_element_index"])
            else:
                selected_updates.add(int(member["next_element_index"]))

    diagnostics = {
        "schema": "v7-global-focus-priority-v1",
        "focus_cap": V6_MAX_FOCUS_UNITS,
        "add_cap": 1,
        "detected_update_candidates": len(update_candidates),
        "detected_add_candidates": int(add_candidate is not None),
        "groups": groups,
        "selected_updates": sorted(selected_updates),
        "selected_add": selected_add,
        "emitted_focus_units": selected_count,
    }
    return selected_updates, selected_add, diagnostics


def _compact_ocr_token(value: str) -> str:
    return "".join(character for character in normalize_text(value) if character.isalnum())


def _normalize_echo_text(value: str) -> str:
    return normalize_text(value).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))


def detect_typed_echo_match(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    matches: Sequence[ElementMatch],
    *,
    action: dict[str, Any],
    action_bucket: str,
    typed_direct_diagnostic: dict[str, Any],
    current_image: Any,
    next_image: Any,
) -> tuple[tuple[ElementMatch, ...], dict[str, Any]]:
    """Find one stable text label that deterministically echoes an appended typed value."""

    diagnostic: dict[str, Any] = {"detector": "typed_echo", "accepted": False}
    if action_bucket != "text_key" or normalize_text(action.get("type")) != "type_text":
        diagnostic["rejection"] = "not_type_text"
        return (), diagnostic
    typed = _typed_text(action)
    typed_value = _normalize_echo_text(typed)
    if not 2 <= len(typed_value) <= 128:
        diagnostic["rejection"] = "typed_value_too_short"
        return (), diagnostic
    retention_ratio = _strict_local_retention_ratio(
        current,
        nxt,
        action_bucket=action_bucket,
    )
    if retention_ratio < 0.20:
        diagnostic["rejection"] = "page_replacement"
        diagnostic["retention_ratio"] = retention_ratio
        return (), diagnostic
    anchors: list[tuple[ElementMatch, str, str, float | None, float | None]] = []
    for match in matches:
        left = current.elements[match.current_index]
        right = nxt.elements[match.next_index]
        old_value = _normalize_echo_text(_strip_typed_slot_caret(left.normalized_content))
        direct_identity = match.method == "typed_direct_slot"
        strict_spatial_fallback = bool(
            match.method == "spatial_mutual"
            and match.bbox_iou >= 0.75
            and match.confidence >= 0.85
            and typed_direct_diagnostic.get("rejection") == "next_value_not_unique"
            and int(typed_direct_diagnostic.get("next_value_count") or 0) >= 2
        )
        region_diff = patch_difference(current_image, next_image, right.bbox)
        identity_diff = patch_difference(current_image, next_image, left.bbox, right.bbox)
        visually_changed = bool(
            (region_diff is not None and region_diff >= 0.008)
            or (identity_diff is not None and identity_diff >= 0.02)
        )
        if (
            (direct_identity or strict_spatial_fallback)
            and left.element_type in TYPED_DIRECT_ELEMENT_TYPES
            and right.element_type in TYPED_DIRECT_ELEMENT_TYPES
            and left.element_type == right.element_type
            and _typed_pair_has_predictable_target(
                left.normalized_content,
                right.normalized_content,
                typed,
            )
            and not _ui_value_matches_typed(left.normalized_content, typed)
            and old_value
            and old_value != typed_value
            and visually_changed
        ):
            anchors.append(
                (
                    match,
                    old_value,
                    "direct" if direct_identity else "strict_spatial_multiplicity",
                    region_diff,
                    identity_diff,
                )
            )
    if len(anchors) != 1:
        diagnostic["rejection"] = "typed_anchor_not_unique"
        diagnostic["typed_anchor_count"] = len(anchors)
        return (), diagnostic
    anchor, old_value, anchor_path, anchor_region_diff, anchor_identity_diff = anchors[0]
    candidates: list[tuple[ParsedElementSpan, ParsedElementSpan]] = []
    for left in current.elements:
        if left.index == anchor.current_index or left.element_type != "text":
            continue
        left_text = _normalize_echo_text(left.content)
        if left_text.count(old_value) != 1:
            continue
        prefix, suffix = left_text.split(old_value, 1)
        if len(prefix) + len(suffix) < 6:
            continue
        expected = left_text.replace(old_value, typed_value, 1)
        for right in nxt.elements:
            if right.index == anchor.next_index or right.element_type != "text":
                continue
            right_text = _normalize_echo_text(right.content)
            if right_text != expected or right_text.count(typed_value) != 1:
                continue
            if (
                bbox_iou(left.bbox, right.bbox) < 0.75
                or center_distance(left.bbox, right.bbox) > 24.0
                or vertical_overlap_ratio(left.bbox, right.bbox) < 0.75
            ):
                continue
            stable_anchors = _stable_spatial_anchor_count(
                current,
                nxt,
                excluded_current={anchor.current_index, left.index},
                excluded_next={anchor.next_index, right.index},
            )
            if stable_anchors >= 4:
                candidates.append((left, right))
    if len(candidates) != 1:
        diagnostic["rejection"] = "echo_not_unique"
        diagnostic["echo_count"] = len(candidates)
        return (), diagnostic
    left, right = candidates[0]
    echo = _match_record(left, right, method="typed_echo_template", confidence=0.98)
    diagnostic.update(
        {
            "accepted": True,
            "typed_anchor_path": anchor_path,
            "typed_anchor_current_index": anchor.current_index,
            "typed_anchor_next_index": anchor.next_index,
            "current_element_index": left.index,
            "next_element_index": right.index,
            "retention_ratio": retention_ratio,
            "typed_anchor_region_diff": anchor_region_diff,
            "typed_anchor_identity_diff": anchor_identity_diff,
        }
    )
    return (echo,), diagnostic


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    edits = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
        else:
            edits += 1
            long_index += 1
            if edits > 1:
                return False
    return True


def _is_short_token_ocr_shadow(
    candidate: ParsedElementSpan,
    element: ParsedElementSpan,
    *,
    distance: float,
) -> bool:
    left = _compact_ocr_token(candidate.normalized_content)
    right = _compact_ocr_token(element.normalized_content)
    return bool(
        1 <= len(left) <= 4
        and 1 <= len(right) <= 4
        and not left.isdigit()
        and not right.isdigit()
        and distance <= 24.0
        and _edit_distance_at_most_one(left, right)
    )


def _is_right_edge_tall_ocr_shadow(
    candidate: ParsedElementSpan,
    element: ParsedElementSpan,
) -> bool:
    if candidate.element_type != element.element_type:
        return False
    left_content = _compact_ocr_token(candidate.normalized_content)
    right_content = _compact_ocr_token(element.normalized_content)
    if not (1 <= len(left_content) <= 8 and 1 <= len(right_content) <= 8):
        return False
    left_x, left_y, left_width, left_height = [float(value) for value in candidate.bbox]
    right_x, right_y, right_width, right_height = [float(value) for value in element.bbox]
    if min(left_width, left_height, right_width, right_height) <= 0.0:
        return False
    size_stable = (
        abs(left_width - right_width) / max(left_width, right_width) <= 0.25
        and abs(left_height - right_height) / max(left_height, right_height) <= 0.25
    )
    return bool(
        left_height / left_width >= 6.0
        and right_height / right_width >= 6.0
        and left_x >= 0.94 * MARKDOWN_GRID_EXTENT
        and right_x >= 0.94 * MARKDOWN_GRID_EXTENT
        and abs(left_x - right_x) <= 24.0
        and abs(left_y - right_y) <= 8.0
        and size_stable
    )


def _is_generic_role_placeholder(element: ParsedElementSpan) -> bool:
    return element.normalized_content in {
        normalize_text(element.element_type),
        "element",
        "item",
        "control",
    }


def _has_lexical_add_grounding(element: ParsedElementSpan) -> bool:
    """Require text ADDs to contain a word, multi-letter script, or sane number."""

    if element.element_type != "text":
        return True
    content = normalize_text(element.content)
    if re.search(r"[a-z]{2,}", content, flags=re.IGNORECASE):
        return True
    if sum(character.isalpha() and not character.isascii() for character in content) >= 2:
        return True
    numeric_candidates = {
        content.strip().strip("()[]{}"),
        " ".join(str(element.content).strip().split()).strip("()[]{}"),
    }
    for numeric in numeric_candidates:
        if re.fullmatch(
            r"[+\-]?(?:[$€£¥]\s*)?\d[\d,]*(?:\.\d+)?(?:%|[°º])?",
            numeric,
            flags=re.IGNORECASE,
        ):
            return True
        if re.fullmatch(r"\d+\s*/\s*\d+", numeric):
            return True
        if re.fullmatch(r"\d+(?:st|nd|rd|th)", numeric, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"\d{1,4}([./-])\d{1,2}\1\d{1,4}", numeric):
            return True
    return False


def _is_oversized_low_density_text_add(element: ParsedElementSpan) -> bool:
    """Reject coarse OCR regions that span image tiles rather than a text line."""

    if element.element_type != "text":
        return False
    _, _, width, height = [float(value) for value in element.bbox]
    alphanumeric_count = sum(character.isalnum() for character in element.content)
    return bool(width >= 240.0 and height >= 80.0 and alphanumeric_count <= 24)


def _is_viewport_clipped_text_add(element: ParsedElementSpan) -> bool:
    """Treat text-like boxes touching the Markdown viewport edge as partial OCR."""

    if element.element_type not in {"text", "link"}:
        return False
    x, y, width, height = [float(value) for value in element.bbox]
    left = x - width / 2.0
    right = x + width / 2.0
    top = y - height / 2.0
    bottom = y + height / 2.0
    return bool(left <= 3.0 or right >= 996.0 or top <= 3.0 or bottom >= 996.0)


def _is_malformed_rating_fragment(element: ParsedElementSpan) -> bool:
    """Reject star-rating OCR fragments whose lexical evidence is only a suffix."""

    if element.element_type != "text":
        return False
    content = " ".join(str(element.content).strip().split())
    if re.fullmatch(r"[★☆]{1,8}\s*\d+\s*&\s*up", content, flags=re.IGNORECASE):
        return False
    rating_symbols = sum(character in {"★", "☆"} for character in element.content)
    alphanumeric_count = sum(character.isalnum() for character in element.content)
    return bool(rating_symbols >= 3 and alphanumeric_count <= 4)


def _is_repeated_tilde_ocr_fragment(element: ParsedElementSpan) -> bool:
    """Reject a narrow short OCR-gibberish pattern without filtering normal math."""

    if element.element_type != "text":
        return False
    content = "".join(str(element.content).split())
    if len(content) > 32 or content.count("~") < 3:
        return False
    punctuation_count = sum(not character.isalnum() for character in content)
    return bool(
        "..." in content
        and ":" in content
        and content.endswith("+")
        and punctuation_count / max(1, len(content)) >= 0.40
    )


def _has_shadow_current(
    element: ParsedElementSpan,
    current: Sequence[ParsedElementSpan],
    *,
    scroll_delta: float | None,
    scroll_shadow_delta: float | None,
) -> bool:
    for candidate in current:
        overlap = bbox_iou(candidate.bbox, element.bbox)
        similarity = content_similarity(candidate, element)
        diagonal = math.hypot(float(element.bbox[2]), float(element.bbox[3]))
        distance = center_distance(candidate.bbox, element.bbox)
        if (
            candidate.element_type == element.element_type
            and overlap >= 0.15
            and distance <= max(20.0, 0.5 * diagonal)
        ):
            return True
        if similarity >= 0.60 and distance <= 150.0:
            return True
        if _is_short_token_ocr_shadow(candidate, element, distance=distance):
            return True
        if _is_right_edge_tall_ocr_shadow(candidate, element):
            return True
        if scroll_delta is not None and candidate.semantic_key == element.semantic_key:
            dx = float(element.bbox[0]) - float(candidate.bbox[0])
            dy = float(element.bbox[1]) - float(candidate.bbox[1])
            if abs(dx) <= SCROLL_RESIDUAL_LIMIT and abs(dy - scroll_delta) <= SCROLL_RESIDUAL_LIMIT:
                return True
        # Suppression may use the commanded delta even when there are too few
        # exact anchors to promote geometry to focus.  It can only reduce ADD
        # recall; it never creates an UPDATE.
        if scroll_shadow_delta is not None:
            dx = float(element.bbox[0]) - float(candidate.bbox[0])
            dy = float(element.bbox[1]) - float(candidate.bbox[1])
            vertical_residual = abs(dy - scroll_shadow_delta)
            same_payload = candidate.normalized_content == element.normalized_content
            role_jitter_payload = (
                _strip_visual_role_prefix(candidate.normalized_content)
                == _strip_visual_role_prefix(element.normalized_content)
            )
            horizontally_local = abs(dx) <= max(
                24.0,
                0.5 * max(float(candidate.bbox[2]), float(element.bbox[2])),
            )
            fuzzy_ocr_shadow = similarity >= 0.55 and horizontally_local
            if vertical_residual <= 8.0 and (
                same_payload or (role_jitter_payload and horizontally_local) or fuzzy_ocr_shadow
            ):
                return True
    return False


def _scroll_entry_evidence(
    element: ParsedElementSpan,
    *,
    scroll_shadow_delta: float | None,
    viewport_height: float | None,
) -> tuple[bool, dict[str, Any]]:
    """Prove that an unmatched element was fully outside before the scroll."""

    if (
        scroll_shadow_delta is None
        or scroll_shadow_delta == 0.0
        or viewport_height is None
        or viewport_height <= 0.0
    ):
        return False, {
            "scroll_entry_proven": False,
            "reason": "missing_scroll_delta_or_viewport_height",
        }
    reverse_center_y = float(element.bbox[1]) - scroll_shadow_delta
    half_height = float(element.bbox[3]) / 2.0
    if scroll_shadow_delta < 0.0:
        reverse_edge = reverse_center_y - half_height
        enters = reverse_edge >= viewport_height - 2.0
        edge = "bottom"
    else:
        reverse_edge = reverse_center_y + half_height
        enters = reverse_edge <= 2.0
        edge = "top"
    return enters, {
        "scroll_entry_proven": enters,
        "entry_edge": edge,
        "reverse_projected_center_y": round(reverse_center_y, 4),
        "reverse_projected_edge": round(reverse_edge, 4),
        "viewport_height": round(viewport_height, 4),
        "scroll_shadow_delta": round(scroll_shadow_delta, 4),
    }


def detect_v6_add_anchor(
    current: ParsedMarkdownSpans,
    nxt: ParsedMarkdownSpans,
    *,
    row: dict[str, Any],
    action: dict[str, Any],
    action_bucket: str,
    unmatched_next: set[int],
    search_mode_matches: Sequence[ElementMatch],
    search_mode_diagnostic: dict[str, Any],
    current_image: Any,
    next_image: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return one allowlisted ADD anchor, never a generic unmatched element."""

    diagnostic: dict[str, Any] = {"detector": "v6_add_anchor", "accepted": False}
    action_type = normalize_text(action.get("type"))
    if action_bucket in {"scroll", "navigation"} or action_type in {"navigate", "hover"}:
        diagnostic["rejection"] = "action_class_forbidden"
        return None, diagnostic
    retention_ratio, stable_anchors = _v6_local_evidence(
        current,
        nxt,
        action_bucket=action_bucket,
    )
    diagnostic.update(
        {
            "retention_ratio": round(retention_ratio, 6),
            "stable_anchor_count": stable_anchors,
        }
    )
    standard_retention = _v6_high_retention(retention_ratio, stable_anchors)

    current_texts = {element.normalized_content for element in current.elements}
    click_point = _action_point(action)
    candidates: list[tuple[int, str, ParsedElementSpan, dict[str, Any]]] = []

    # Fox Sports search activation changes the unique POPULAR header into the
    # unique RECENT header and exposes one row-local "clear all" control.  The
    # existing search-mode match carries the strict Fox provenance, click target,
    # slot geometry and retained-page proof; this branch only binds its sibling.
    if len(search_mode_matches) == 1 and search_mode_diagnostic.get("accepted"):
        search_match = search_mode_matches[0]
        search_left = current.elements[search_match.current_index]
        search_right = nxt.elements[search_match.next_index]
        diagnostic_pair_agrees = bool(
            search_match.method == "search_mode_slot"
            and search_mode_diagnostic.get("current_element_index")
            == search_match.current_index
            and search_mode_diagnostic.get("next_element_index")
            == search_match.next_index
        )
        current_clear_all = [
            element
            for element in current.elements
            if _strip_visual_role_prefix(element.normalized_content) == "clear all"
        ]
        next_clear_all = [
            element
            for element in nxt.elements
            if element.index in unmatched_next
            and _strip_visual_role_prefix(element.normalized_content) == "clear all"
        ]
        row_pairs = [
            element
            for element in next_clear_all
            if vertical_overlap_ratio(search_right.bbox, element.bbox)
            >= V7_SEARCH_MODE_CLEAR_ALL_MIN_VERTICAL_OVERLAP
            and abs(float(search_right.bbox[1]) - float(element.bbox[1]))
            <= V7_SEARCH_MODE_CLEAR_ALL_MAX_CENTER_Y_DELTA
            and float(element.bbox[0]) > float(search_right.bbox[0])
        ]
        if (
            diagnostic_pair_agrees
            and search_left.normalized_content == "popular searches"
            and _strip_visual_role_prefix(search_right.normalized_content)
            == "recent searches"
            and not current_clear_all
            and len(next_clear_all) == 1
            and next_clear_all[0].element_type in V7_SEARCH_MODE_CLEAR_ALL_TYPES
            and row_pairs == next_clear_all
        ):
            clear_all = next_clear_all[0]
            candidates.append(
                (
                    150,
                    "search_mode_recent_clear_all_anchor_add",
                    clear_all,
                    {
                        "search_mode_current_index": search_match.current_index,
                        "search_mode_next_index": search_match.next_index,
                        "row_vertical_overlap": round(
                            vertical_overlap_ratio(search_right.bbox, clear_all.bbox),
                            4,
                        ),
                        "row_center_y_delta": round(
                            abs(
                                float(search_right.bbox[1])
                                - float(clear_all.bbox[1])
                            ),
                            4,
                        ),
                        "clear_all_element_type": clear_all.element_type,
                        "retention_ratio": float(
                            search_mode_diagnostic.get("retention_ratio") or 0.0
                        ),
                        "stable_anchor_count": int(
                            search_mode_diagnostic.get("stable_anchor_count") or 0
                        ),
                    },
                )
            )


    if action_bucket == "click" and action_type == "left_click" and click_point is not None:
        bing_current = [element for element in current.elements if element.normalized_content == "microsoft bing"]
        bing_next = [element for element in nxt.elements if element.normalized_content == "microsoft bing"]
        bing_brand_stable = bool(
            len(bing_current) == len(bing_next) == 1
            and bbox_iou(bing_current[0].bbox, bing_next[0].bbox) >= 0.70
            and center_distance(bing_current[0].bbox, bing_next[0].bbox) <= 16.0
        )
        trending = [
            element
            for element in nxt.elements
            if element.index in unmatched_next
            and element.normalized_content == "trending now"
            and element.normalized_content not in current_texts
        ]
        metadata = row.get("meta") or {}
        provenance = metadata.get("v2_provenance") if isinstance(metadata, dict) else None
        action_description = normalize_text(
            provenance.get("action_description") if isinstance(provenance, dict) else ""
        )
        current_domain = normalize_text(
            provenance.get("current_domain") if isinstance(provenance, dict) else ""
        )
        next_domain = normalize_text(
            provenance.get("next_domain") if isinstance(provenance, dict) else ""
        )
        raw_action = provenance.get("raw_action") if isinstance(provenance, dict) else None
        raw_click_contract = _raw_mouse_click_contract(
            raw_action,
            candidate_point=click_point,
            current_image=current_image,
        )
        search_action_target = bool(
            re.fullmatch(r"click on (?:textarea|form) element", action_description)
            and _host_matches(current_domain, "bing.com")
            and _host_matches(next_domain, "bing.com")
            and raw_click_contract is not None
        )
        if (
            bing_brand_stable
            and len(trending) == 1
            and search_action_target
            and 80.0 <= click_point[0] <= 900.0
            and 150.0 <= click_point[1] <= 320.0
        ):
            candidates.append(
                (
                    200,
                    "bing_search_overlay_anchor_add",
                    trending[0],
                    {
                        "brand_current_index": bing_current[0].index,
                        "brand_next_index": bing_next[0].index,
                        "brand_bbox_iou": round(
                            bbox_iou(bing_current[0].bbox, bing_next[0].bbox), 4
                        ),
                        "brand_center_distance": round(
                            center_distance(
                                bing_current[0].bbox, bing_next[0].bbox
                            ),
                            4,
                        ),
                        "action_description": action_description,
                        "click_point": list(click_point),
                        "bing_current_host": current_domain,
                        "bing_next_host": next_domain,
                        **(raw_click_contract or {}),
                    },
                )
            )

        # Filter drawer: click must hit the unique trigger and the next page must
        # expose a complete multi-facet drawer.  Only Clear all is supervised.
        triggers = [
            element
            for element in current.elements
            if element.normalized_content == "filter & sort"
            and _bbox_contains_point(element.bbox, click_point, padding=2.0)
        ]
        clear_all = [
            element
            for element in nxt.elements
            if element.index in unmatched_next
            and element.normalized_content == "clear all"
            and element.normalized_content not in current_texts
        ]
        facet_names = {
            "topic",
            "duration",
            "learning product",
            "language",
            "level",
            "skills",
            "subtitles",
            "sort by",
        }
        visible_facets = {
            name
            for name in facet_names
            if any(name == element.normalized_content for element in nxt.elements)
        }
        if len(triggers) == len(clear_all) == 1 and len(visible_facets) >= 5:
            candidates.append(
                (
                    100,
                    "filter_sort_drawer_anchor_add",
                    clear_all[0],
                    {
                        "trigger_current_index": triggers[0].index,
                        "visible_facets": sorted(visible_facets),
                    },
                )
            )

        # Explicit row insertion: the action hits Add another term and a unique
        # blank advanced-search row appears at that same local vertical slot.
        add_triggers = [
            element
            for element in current.elements
            if element.normalized_content in {"add another term", "add another term+"}
            and _bbox_contains_point(element.bbox, click_point, padding=2.0)
        ]
        inserted_rows = [
            element
            for element in nxt.elements
            if element.index in unmatched_next
            and element.normalized_content == "title and search term..."
            and element.normalized_content not in current_texts
        ]
        configurations = [
            (trigger, inserted)
            for trigger in add_triggers
            for inserted in inserted_rows
            if center_distance(trigger.bbox, inserted.bbox) <= 40.0
            and horizontal_overlap_ratio(trigger.bbox, inserted.bbox) >= 0.90
        ]
        if len(configurations) == 1:
            trigger, inserted = configurations[0]
            candidates.append(
                (
                    100,
                    "advanced_search_row_anchor_add",
                    inserted,
                    {"trigger_current_index": trigger.index},
                )
            )

    if not candidates:
        diagnostic.update(
            {
                "candidate_count": 0,
                "visually_qualified_count": 0,
                "rejection": "no_allowlisted_mechanism",
            }
        )
        return None, diagnostic
    retained_candidates: list[
        tuple[int, str, ParsedElementSpan, dict[str, Any]]
    ] = []
    for priority, reason, element, evidence in candidates:
        bing_overlay_retention = bool(
            reason == "bing_search_overlay_anchor_add"
            and retention_ratio >= V6_BING_MIN_RETENTION_RATIO
            and stable_anchors >= V6_BING_MIN_STABLE_ANCHORS
        )
        search_mode_retention = bool(
            reason == "search_mode_recent_clear_all_anchor_add"
            and float(evidence.get("retention_ratio") or 0.0)
            >= V6_MIN_RETENTION_RATIO
            and int(evidence.get("stable_anchor_count") or 0)
            >= V6_SEARCH_MODE_MIN_STABLE_ANCHORS
        )
        retained_page = (
            search_mode_retention
            if reason == "search_mode_recent_clear_all_anchor_add"
            else standard_retention or bing_overlay_retention
        )
        if retained_page:
            retained_candidates.append(
                (
                    priority,
                    reason,
                    element,
                    {
                        **evidence,
                        "retention_policy": (
                            "search_mode_slot"
                            if reason == "search_mode_recent_clear_all_anchor_add"
                            else "standard"
                            if standard_retention
                            else "bing_overlay_four_anchor"
                        ),
                    },
                )
            )
    if not retained_candidates:
        diagnostic.update(
            {
                "candidate_count": len(candidates),
                "visually_qualified_count": 0,
                "rejection": "insufficient_retained_page",
            }
        )
        return None, diagnostic

    qualified: list[tuple[int, str, ParsedElementSpan, dict[str, Any], float]] = []
    for priority, reason, element, evidence in retained_candidates:
        region_diff = patch_difference(current_image, next_image, element.bbox)
        minimum_region_diff = (
            V7_SEARCH_MODE_CLEAR_ALL_REGION_DIFF
            if reason == "search_mode_recent_clear_all_anchor_add"
            else DEFAULT_ADD_REGION_DIFF
        )
        allowed_element_type = (
            element.element_type in V7_SEARCH_MODE_CLEAR_ALL_TYPES
            if reason == "search_mode_recent_clear_all_anchor_add"
            else element.element_type in ADD_ELEMENT_TYPES
        )
        if (
            allowed_element_type
            and region_diff is not None
            and region_diff >= minimum_region_diff
        ):
            qualified.append((priority, reason, element, evidence, region_diff))
    diagnostic["candidate_count"] = len(candidates)
    diagnostic["visually_qualified_count"] = len(qualified)
    if not qualified:
        diagnostic["rejection"] = (
            "missing_visual_evidence"
        )
        return None, diagnostic
    qualified.sort(key=lambda item: (-item[0], item[2].index, item[1]))
    priority, reason, element, evidence, region_diff = qualified[0]
    diagnostic.update(
        {
            "accepted": True,
            "selected_reason": reason,
            "selected_next_element_index": element.index,
            "selected_priority": priority,
            "selected_retention_policy": evidence.get("retention_policy"),
        }
    )
    return (
        {
            "kind": "ADD",
            "next_element_index": element.index,
            "current_element_index": None,
            "fields": ["line"],
            "spans": [asdict(element.line_span)],
            "confidence": 0.99,
            "visual_change_score": round(region_diff, 6),
            "qualification": "rule_qualified",
            "reasons": [reason],
            "evidence": {
                "same_region_pixel_diff": region_diff,
                "retention_ratio": round(retention_ratio, 6),
                "stable_anchor_count": stable_anchors,
                **evidence,
            },
        },
        diagnostic,
    )


def build_focus_annotation(
    row: dict[str, Any],
    *,
    project_root: Path | None = None,
    image_module: Any | None = None,
) -> dict[str, Any]:
    identifier = str(row.get("id") or "").strip()
    if not identifier:
        raise V4FocusError("sample is missing id")
    if row.get("transition_class") != "changed":
        raise V4FocusError(f"sample {identifier} is not transition_class=changed")
    action_bucket = str(row.get("action_bucket") or "").strip()
    if action_bucket not in {"click", "text_key", "scroll", "navigation", "other"}:
        raise V4FocusError(f"sample {identifier} has invalid action_bucket={action_bucket!r}")

    current = parse_markdown_spans(row.get("current_md"))
    nxt = parse_markdown_spans(row.get("target_next_md"))
    action = _action_payload(row)
    action_type = normalize_text(action.get("type"))
    primary_action_point = _action_point(action)
    current_image = None
    next_image = None
    image_errors: list[str] = []
    if image_module is not None:
        for field, destination in (("current_image_path", "current"), ("next_image_path", "next")):
            raw_path = Path(str(row.get(field) or ""))
            resolved = raw_path if raw_path.is_absolute() else (project_root or Path.cwd()) / raw_path
            try:
                image = image_module.open(resolved).convert("RGB")
            except Exception as exc:
                image_errors.append(f"{field}: {type(exc).__name__}: {exc}")
                image = None
            if destination == "current":
                current_image = image
            else:
                next_image = image

    try:
        action_delta_y = float(action.get("delta_y", 0.0))
    except (TypeError, ValueError):
        action_delta_y = 0.0
    scroll_delta = (
        infer_scroll_delta(current, nxt, action_delta_y=action_delta_y)
        if action_bucket == "scroll" and action_delta_y != 0.0
        else None
    )
    edge_hover = bool(
        action_bucket == "other"
        and action_type == "hover"
        and primary_action_point is not None
        and (
            primary_action_point[1] <= HOVER_EDGE_MARGIN
            or primary_action_point[1] >= MARKDOWN_GRID_EXTENT - HOVER_EDGE_MARGIN
        )
    )
    hover_autoscroll = infer_hover_autoscroll(current, nxt) if edge_hover else None
    scroll_shadow_delta = (
        scroll_delta
        if scroll_delta is not None
        else (-action_delta_y if action_bucket == "scroll" and action_delta_y != 0.0 else None)
    )
    calendar_matches, calendar_diagnostic = detect_calendar_month_slot_matches(
        current,
        nxt,
        row=row,
        action=action,
        action_bucket=action_bucket,
    )
    autocomplete_matches, autocomplete_diagnostic = detect_autocomplete_commit_match(
        current,
        nxt,
        row=row,
        action=action,
        action_bucket=action_bucket,
    )
    typed_direct_matches, typed_direct_diagnostic = detect_typed_direct_slot_match(
        current,
        nxt,
        row=row,
        action=action,
        action_bucket=action_bucket,
    )
    (
        typed_overstock_matches,
        typed_overstock_diagnostic,
    ) = detect_typed_overstock_query_marker_slot_match(
        current,
        nxt,
        row=row,
        action=action,
        action_bucket=action_bucket,
        existing_matches=typed_direct_matches,
        current_image=current_image,
    )
    click_toggle_matches, click_toggle_diagnostic = detect_click_text_toggle_match(
        current,
        nxt,
        action=action,
        action_bucket=action_bucket,
    )
    v6_update_matches, v6_update_diagnostics = detect_v6_update_matches(
        current,
        nxt,
        row=row,
        action=action,
        action_bucket=action_bucket,
        current_image=current_image,
    )
    preliminary_reserved_matches = [
        *calendar_matches,
        *autocomplete_matches,
        *typed_direct_matches,
        *typed_overstock_matches,
        *click_toggle_matches,
        *v6_update_matches,
    ]
    coherent_motion_delta = scroll_delta
    if coherent_motion_delta is None and hover_autoscroll is not None:
        coherent_motion_delta = float(hover_autoscroll["delta_y"])
    preliminary_matches = match_elements(
        current,
        nxt,
        action_bucket=action_bucket,
        scroll_delta=coherent_motion_delta,
        reserved_matches=preliminary_reserved_matches,
    )
    typed_echo_matches, typed_echo_diagnostic = detect_typed_echo_match(
        current,
        nxt,
        preliminary_matches,
        action=action,
        action_bucket=action_bucket,
        typed_direct_diagnostic=typed_direct_diagnostic,
        current_image=current_image,
        next_image=next_image,
    )
    reserved_matches = [*preliminary_reserved_matches, *typed_echo_matches]
    matches = (
        match_elements(
            current,
            nxt,
            action_bucket=action_bucket,
            scroll_delta=coherent_motion_delta,
            reserved_matches=reserved_matches,
        )
        if typed_echo_matches
        else preliminary_matches
    )
    match_by_next = {match.next_index: match for match in matches}
    matched_current = {match.current_index for match in matches}
    matched_next = set(match_by_next)
    unmatched_current = set(range(len(current.elements))) - matched_current
    unmatched_next = set(range(len(nxt.elements))) - matched_next
    hover_motion_shadows = match_hover_motion_shadows(
        current,
        nxt,
        unmatched_current=unmatched_current,
        unmatched_next=unmatched_next,
        motion=hover_autoscroll,
    )
    hover_shadow_by_next = {
        int(pair["next_element_index"]): pair for pair in hover_motion_shadows
    }
    hover_shadow_current = {
        int(pair["current_element_index"]) for pair in hover_motion_shadows
    }
    hover_shadow_next = set(hover_shadow_by_next)
    next_leaf_count = sum(element.element_type != "block" for element in nxt.elements)
    retained_leaf_count = sum(
        nxt.elements[match.next_index].element_type != "block"
        and (
            match.method in {
                "exact",
                "semantic",
                "state_identity",
                "calendar_month_slot",
                "autocomplete_commit",
                "typed_direct_slot",
                "typed_echo_template",
                "click_text_toggle",
                *V6_UPDATE_METHODS,
            }
            or (match.method == "spatial_mutual" and match.bbox_iou >= 0.75)
        )
        for match in matches
    )
    retention_ratio = retained_leaf_count / max(1, next_leaf_count)
    # A strict action-specific local detector with four or more stable anchors is
    # stronger evidence than the coarse leaf-retention heuristic.  Calendar
    # rerenders can otherwise look like page replacement because most date-cell
    # tokens change at once.
    coherent_drag_retention = any(
        bool((v6_update_diagnostics.get(name) or {}).get("accepted"))
        and bool(
            (v6_update_diagnostics.get(name) or {}).get(
                "coherent_translation_retention"
            )
        )
        for name in ("drag_sort_slot", "drag_form_control_slot")
    )
    general_page_replace = (
        retention_ratio < 0.20
        and not calendar_matches
        and not coherent_drag_retention
    )
    navigation_page_replace = (
        action_bucket == "navigation"
        and normalize_text(action.get("type")) == "navigate"
        and retention_ratio < 0.20
    )
    typed = _typed_text(action)
    typed_echo_pairs = {
        (match.current_index, match.next_index) for match in typed_echo_matches
    }
    typed_anchor_boxes = [
        nxt.elements[match.next_index].bbox
        for match in matches
        if (
            match.method == "typed_direct_slot"
            or (match.method == "spatial_mutual" and match.bbox_iou >= 0.75)
        )
        and bool(typed)
        and _typed_pair_has_predictable_target(
            current.elements[match.current_index].normalized_content,
            nxt.elements[match.next_index].normalized_content,
            typed,
        )
        and current.elements[match.current_index].normalized_content != nxt.elements[match.next_index].normalized_content
    ]

    focus_units: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    next_labels: list[dict[str, Any]] = []
    page_replace_update_vetoes = 0
    hover_visual_region_shadows: list[dict[str, Any]] = []
    hover_stacked_fragments: list[dict[str, Any]] = []
    hover_aux_shadow_by_next: dict[int, dict[str, Any]] = {}
    v6_add_anchor, v6_add_diagnostic = detect_v6_add_anchor(
        current,
        nxt,
        row=row,
        action=action,
        action_bucket=action_bucket,
        unmatched_next=unmatched_next,
        search_mode_matches=tuple(
            match for match in matches if match.method == "search_mode_slot"
        ),
        search_mode_diagnostic=v6_update_diagnostics.get("search_mode_slot") or {},
        current_image=current_image,
        next_image=next_image,
    )
    bing_overlay_retention = bool(
        v6_add_anchor is not None
        and v6_add_anchor.get("reasons") == ["bing_search_overlay_anchor_add"]
        and (v6_add_anchor.get("evidence") or {}).get("retention_policy")
        == "bing_overlay_four_anchor"
    )

    update_page_replace_veto = general_page_replace
    if bing_overlay_retention:
        general_page_replace = False

    update_candidates: dict[int, dict[str, Any]] = {}
    for candidate_match in matches:
        left = current.elements[candidate_match.current_index]
        right = nxt.elements[candidate_match.next_index]
        candidate = _focus_update(
            left,
            right,
            candidate_match,
            action_bucket=action_bucket,
            action=action,
            current_image=current_image,
            next_image=next_image,
            scroll_delta=scroll_delta,
            typed_echo_candidate=(candidate_match.current_index, candidate_match.next_index)
            in typed_echo_pairs,
        )
        sticky_exception = candidate_match.method == "sticky_pos_header"
        if candidate is not None and update_page_replace_veto and not sticky_exception:
            candidate = None
            page_replace_update_vetoes += 1
        if candidate is not None:
            update_candidates[candidate_match.next_index] = candidate
    (
        selected_update_indices,
        selected_add_index,
        focus_selection_diagnostic,
    ) = _select_v7_focus_units(update_candidates, v6_add_anchor)
    for element in nxt.elements:
        match = match_by_next.get(element.index)
        if match is not None:
            current_element = current.elements[match.current_index]
            update = (
                update_candidates.get(element.index)
                if element.index in selected_update_indices
                else None
            )
            if update is not None:
                update["focus_id"] = len(focus_units)
                focus_units.append(update)
                next_labels.append({"next_element_index": element.index, "label": "UPDATE", "focus_id": update["focus_id"]})
            else:
                changed = (
                    current_element.semantic_key != element.semantic_key
                    or max(abs(a - b) for a, b in zip(current_element.bbox, element.bbox)) > DEFAULT_JITTER_THRESHOLD
                )
                next_labels.append(
                    {
                        "next_element_index": element.index,
                        "label": "UNKNOWN" if changed else "BASE",
                        "focus_id": None,
                    }
                )
            continue

        region_diff = patch_difference(current_image, next_image, element.bbox)
        hover_motion_shadow = hover_shadow_by_next.get(element.index)
        hover_region_shadow = hover_visual_region_shadow(
            element,
            current,
            motion=hover_autoscroll,
        )
        hover_stacked_fragment = hover_stacked_text_fragment(
            element,
            nxt,
            motion=hover_autoscroll,
        )
        if hover_region_shadow is not None:
            hover_visual_region_shadows.append(hover_region_shadow)
            hover_aux_shadow_by_next[element.index] = hover_region_shadow
        elif hover_stacked_fragment is not None:
            hover_stacked_fragments.append(hover_stacked_fragment)
            hover_aux_shadow_by_next[element.index] = hover_stacked_fragment
        plausible = (
            hover_motion_shadow is not None
            or hover_region_shadow is not None
            or hover_stacked_fragment is not None
            or _has_shadow_current(
            element,
            current.elements,
            scroll_delta=scroll_delta,
            scroll_shadow_delta=scroll_shadow_delta,
            )
        )
        generic_role_placeholder = _is_generic_role_placeholder(element)
        lexical_add_grounded = _has_lexical_add_grounding(element)
        oversized_low_density_text = (
            action_bucket == "scroll"
            and _is_oversized_low_density_text_add(element)
        )
        viewport_clipped_text = (
            action_bucket == "scroll"
            and _is_viewport_clipped_text_add(element)
        )
        malformed_rating_fragment = (
            action_bucket == "scroll"
            and _is_malformed_rating_fragment(element)
        )
        repeated_tilde_ocr_fragment = (
            action_bucket == "scroll"
            and _is_repeated_tilde_ocr_fragment(element)
        )
        viewport_height = MARKDOWN_GRID_EXTENT if current_image is not None else None
        scroll_entry, scroll_entry_evidence = _scroll_entry_evidence(
            element,
            scroll_shadow_delta=scroll_shadow_delta,
            viewport_height=viewport_height,
        )
        # Precision baseline: generic unmatched-next ADDs are not action-identifiable.
        # They include viewport entry, unocclusion, dynamic suggestions, page
        # replacement and OCR resegmentation.  Every line still receives base
        # loss; later schema versions may re-enable individually proven ADD
        # mechanisms, never this generic gate.
        selected_v6_add = bool(
            v6_add_anchor is not None
            and selected_add_index == element.index
        )
        action_allows_add = selected_v6_add
        is_add = (
            action_allows_add
        )
        if is_add:
            assert v6_add_anchor is not None
            add_unit = dict(v6_add_anchor)
            add_unit["focus_id"] = len(focus_units)
            focus_units.append(add_unit)
            next_labels.append(
                {
                    "next_element_index": element.index,
                    "label": "ADD",
                    "focus_id": add_unit["focus_id"],
                }
            )
        else:
            next_labels.append(
                {
                    "next_element_index": element.index,
                    "label": "UNKNOWN",
                    "focus_id": None,
                    "reason": (
                        "generic role placeholder is excluded from ADD focus"
                        if generic_role_placeholder
                        else "text ADD lacks lexical grounding and is excluded from focus"
                        if not lexical_add_grounded
                        else "oversized low-density text region is excluded from ADD focus"
                        if oversized_low_density_text
                        else "viewport-clipped text is excluded from ADD focus"
                        if viewport_clipped_text
                        else "malformed rating fragment is excluded from ADD focus"
                        if malformed_rating_fragment
                        else "repeated-tilde OCR fragment is excluded from ADD focus"
                        if repeated_tilde_ocr_fragment
                        else "edge-hover coherent-motion shadow is excluded from ADD focus"
                        if hover_motion_shadow is not None
                        else "edge-hover visual-region OCR split is excluded from ADD focus"
                        if hover_region_shadow is not None
                        else "edge-hover stacked OCR fragment is excluded from ADD focus"
                        if hover_stacked_fragment is not None
                        else "generic ADD focus is disabled in the precision baseline"
                    ),
                }
            )

    for unit in [*focus_units, *review_candidates]:
        for span in unit["spans"]:
            start, end = int(span["start"]), int(span["end"])
            span_text = nxt.text[start:end]
            span["text"] = span_text
            span["text_sha256"] = sha256_text(span_text)

    focus_by_next = {int(unit["next_element_index"]): unit for unit in focus_units}
    candidate_by_next = {
        int(unit["next_element_index"]): unit for unit in review_candidates
    }
    relations: list[dict[str, Any]] = []
    for match in matches:
        left = current.elements[match.current_index]
        right = nxt.elements[match.next_index]
        focus = focus_by_next.get(right.index)
        changed = left.semantic_key != right.semantic_key or left.bbox != right.bbox
        relations.append(
            {
                "operation": "update" if focus and focus["kind"] == "UPDATE" else ("ambiguous" if changed else "unchanged"),
                "current_element_index": left.index,
                "next_element_index": right.index,
                "focus_decision": "focus" if focus else "base_only",
                "confidence_score": match.confidence,
                "confidence_band": "high" if match.confidence >= 0.90 else "medium",
                "reason_codes": list(focus["reasons"]) if focus else (["changed_but_not_action_supported"] if changed else ["matched_unchanged"]),
                "changed_fields": list(focus["fields"]) if focus else [],
                "evidence": asdict(match),
            }
        )
    for shadow in hover_motion_shadows:
        relations.append(
            {
                "operation": "ambiguous",
                "current_element_index": shadow["current_element_index"],
                "next_element_index": shadow["next_element_index"],
                "focus_decision": "base_only",
                "confidence_score": None,
                "confidence_band": "motion_shadow",
                "reason_codes": ["edge_hover_coherent_motion_shadow"],
                "changed_fields": [],
                "evidence": dict(shadow),
            }
        )
    for element in current.elements:
        if element.index not in matched_current and element.index not in hover_shadow_current:
            relations.append(
                {
                    "operation": "delete",
                    "current_element_index": element.index,
                    "next_element_index": None,
                    "focus_decision": "base_only",
                    "confidence_score": None,
                    "confidence_band": "unscored",
                    "reason_codes": ["delete_focus_disabled"],
                    "changed_fields": [],
                    "evidence": {},
                }
            )
    for element in nxt.elements:
        if element.index not in matched_next and element.index not in hover_shadow_next:
            focus = focus_by_next.get(element.index)
            candidate = candidate_by_next.get(element.index)
            selected = focus or candidate
            hover_aux_shadow = hover_aux_shadow_by_next.get(element.index)
            relations.append(
                {
                    "operation": "add" if selected else "ambiguous",
                    "current_element_index": None,
                    "next_element_index": element.index,
                    "focus_decision": "focus" if focus else ("review_only" if candidate else "base_only"),
                    "confidence_score": selected["confidence"] if selected else None,
                    "confidence_band": "rule_qualified" if focus else ("review" if candidate else "unscored"),
                    "reason_codes": (
                        list(selected["reasons"])
                        if selected
                        else [str(hover_aux_shadow["method"])]
                        if hover_aux_shadow
                        else ["unmatched_without_high_confidence_add_evidence"]
                    ),
                    "changed_fields": list(selected["fields"]) if selected else [],
                    "evidence": (
                        dict(selected["evidence"])
                        if selected
                        else dict(hover_aux_shadow)
                        if hover_aux_shadow
                        else {}
                    ),
                }
            )

    source = str((row.get("meta") or {}).get("dataset") or "unknown")
    return {
        "schema_version": FOCUS_LABEL_VERSION,
        "sample_id": identifier,
        "transition_class": "changed",
        "action_bucket": action_bucket,
        "action": action,
        "source": source,
        "current_image_path": str(row.get("current_image_path") or ""),
        "next_image_path": str(row.get("next_image_path") or ""),
        "current_md": current.text,
        "target_next_md": nxt.text,
        "target_sha256": sha256_text(nxt.text),
        "base_units": [asdict(span) for span in nxt.base_units],
        "focus_units": focus_units,
        "review_candidates": review_candidates,
        "current_elements": [_element_dict(element) for element in current.elements],
        "next_elements": [_element_dict(element) for element in nxt.elements],
        "matches": [asdict(match) for match in matches],
        "hover_motion_shadows": list(hover_motion_shadows),
        "hover_visual_region_shadows": hover_visual_region_shadows,
        "hover_stacked_fragments": hover_stacked_fragments,
            "special_match_diagnostics": {
            "calendar_month_slot": calendar_diagnostic,
            "autocomplete_commit": autocomplete_diagnostic,
            "typed_direct_slot": typed_direct_diagnostic,
            "typed_overstock_query_marker_slot": typed_overstock_diagnostic,
            "click_text_toggle": click_toggle_diagnostic,
            "typed_echo": typed_echo_diagnostic,
            **v6_update_diagnostics,
            "v6_add_anchor": v6_add_diagnostic,
            "focus_unit_selection": focus_selection_diagnostic,
        },
        "relations": relations,
        "next_labels": next_labels,
        "stats": {
            "base_units": len(nxt.base_units),
            "focus_units": len(focus_units),
            "add_focus": sum(unit["kind"] == "ADD" for unit in focus_units),
            "update_focus": sum(unit["kind"] == "UPDATE" for unit in focus_units),
            "review_candidates": len(review_candidates),
            "matched_elements": len(matches),
            "unmatched_current_elements": len(current.elements) - len(matched_current),
            "unmatched_next_elements": len(nxt.elements) - len(matched_next),
            "image_errors": image_errors,
            "inferred_scroll_delta": scroll_delta,
            "scroll_shadow_delta": scroll_shadow_delta,
            "hover_autoscroll": hover_autoscroll,
            "hover_motion_shadow_pairs": len(hover_motion_shadows),
            "hover_visual_region_shadows": len(hover_visual_region_shadows),
            "hover_stacked_fragments": len(hover_stacked_fragments),
            "retention_ratio": round(retention_ratio, 6),
            "navigation_page_replace": navigation_page_replace,
            "general_page_replace": general_page_replace,
            "typed_anchor_count": len(typed_anchor_boxes),
            "calendar_month_slot_matches": len(calendar_matches),
            "autocomplete_commit_matches": len(autocomplete_matches),
            "typed_direct_slot_matches": len(typed_direct_matches),
            "typed_overstock_query_marker_slot_matches": len(
                typed_overstock_matches
            ),
            "click_text_toggle_matches": len(click_toggle_matches),
            "typed_echo_matches": len(typed_echo_matches),
            "search_mode_slot_matches": sum(
                match.method == "search_mode_slot" for match in matches
            ),
            "sticky_pos_header_matches": sum(
                match.method == "sticky_pos_header" for match in matches
            ),
            "drag_sort_slot_matches": sum(
                match.method == "drag_sort_slot" for match in matches
            ),
            "drag_form_control_slot_matches": sum(
                match.method == "drag_form_control_slot" for match in matches
            ),
            "drag_binary_selector_slot_matches": sum(
                match.method == "drag_binary_selector_slot" for match in matches
            ),
            "v6_add_anchor_detected": int(v6_add_anchor is not None),
            "v6_add_anchor_matches": int(selected_add_index is not None),
            "page_replace_update_vetoes": page_replace_update_vetoes,
        },
    }


def _validate_v6_focus_rule_contract(
    annotation: dict[str, Any],
    unit: dict[str, Any],
    *,
    source_row: dict[str, Any] | None,
    current_image: Any,
) -> None:
    current_elements = annotation.get("current_elements") or []
    next_elements = annotation.get("next_elements") or []
    current_index = unit.get("current_element_index")
    next_index = unit.get("next_element_index")
    reasons = unit.get("reasons")
    if not isinstance(reasons, list) or len(reasons) != 1:
        raise V4FocusError("v6 focus must have exactly one rule reason")
    reason = str(reasons[0])
    contract = FOCUS_REASON_CONTRACTS.get(reason)
    if contract is None:
        raise V4FocusError(f"v6 focus has unknown rule reason {reason!r}")
    if tuple(unit.get("fields") or ()) != contract["fields"]:
        raise V4FocusError(f"v6 rule {reason} has invalid fields")
    action_bucket = str(annotation.get("action_bucket") or "")
    action_type = normalize_text((annotation.get("action") or {}).get("type"))
    if action_bucket not in contract["buckets"] or action_type not in contract["actions"]:
        raise V4FocusError(f"v6 rule {reason} conflicts with action contract")
    source_provenance = (
        ((source_row or {}).get("meta") or {}).get("v2_provenance") or {}
    )
    if unit.get("kind") == "ADD":
        if reason not in V6_ADD_REASONS or current_index is not None:
            raise V4FocusError("v6 ADD focus is not an allowlisted unmatched anchor")
        if (
            not isinstance(next_index, int)
            or isinstance(next_index, bool)
            or not 0 <= next_index < len(next_elements)
        ):
            raise V4FocusError("ADD focus references invalid next element")
        evidence = unit.get("evidence") or {}
        region_diff = evidence.get("same_region_pixel_diff")
        evidence_retention = float(evidence.get("retention_ratio") or 0.0)
        evidence_anchors = int(evidence.get("stable_anchor_count") or 0)
        standard_retention = bool(
            reason != "search_mode_recent_clear_all_anchor_add"
            and
            evidence_retention >= V6_MIN_RETENTION_RATIO
            and evidence_anchors >= V6_MIN_STABLE_ANCHORS
            and evidence.get("retention_policy") == "standard"
        )
        bing_overlay_retention = bool(
            reason == "bing_search_overlay_anchor_add"
            and evidence_retention >= V6_BING_MIN_RETENTION_RATIO
            and evidence_anchors >= V6_BING_MIN_STABLE_ANCHORS
            and evidence.get("retention_policy") == "bing_overlay_four_anchor"
        )
        search_mode_retention = bool(
            reason == "search_mode_recent_clear_all_anchor_add"
            and evidence_retention >= V6_MIN_RETENTION_RATIO
            and evidence_anchors >= V6_SEARCH_MODE_MIN_STABLE_ANCHORS
            and evidence.get("retention_policy") == "search_mode_slot"
        )
        minimum_region_diff = (
            V7_SEARCH_MODE_CLEAR_ALL_REGION_DIFF
            if reason == "search_mode_recent_clear_all_anchor_add"
            else DEFAULT_ADD_REGION_DIFF
        )
        if (
            isinstance(region_diff, bool)
            or not isinstance(region_diff, (int, float))
            or not math.isfinite(float(region_diff))
            or float(region_diff) < minimum_region_diff
            or not (
                standard_retention
                or bing_overlay_retention
                or search_mode_retention
            )
        ):
            raise V4FocusError(f"v6 ADD rule {reason} lacks visual/retention evidence")
        add_diagnostic = (
            (annotation.get("special_match_diagnostics") or {}).get("v6_add_anchor")
            or {}
        )
        if (
            not add_diagnostic.get("accepted")
            or add_diagnostic.get("selected_reason") != reason
            or add_diagnostic.get("selected_next_element_index") != next_index
        ):
            raise V4FocusError(f"v6 ADD rule {reason} lacks detector provenance")
        right = next_elements[next_index]
        normalized = normalize_text(right.get("content"))
        expected_content = {
            "bing_search_overlay_anchor_add": "trending now",
            "filter_sort_drawer_anchor_add": "clear all",
            "advanced_search_row_anchor_add": "title and search term...",
            "search_mode_recent_clear_all_anchor_add": "clear all",
        }[reason]
        normalized_payload = (
            _strip_visual_role_prefix(normalized)
            if reason == "search_mode_recent_clear_all_anchor_add"
            else normalized
        )
        if normalized_payload != expected_content:
            raise V4FocusError(f"v6 ADD rule {reason} has an invalid anchor payload")
        if reason == "bing_search_overlay_anchor_add":
            brand_current_index = evidence.get("brand_current_index")
            brand_next_index = evidence.get("brand_next_index")
            if (
                not isinstance(brand_current_index, int)
                or isinstance(brand_current_index, bool)
                or not 0 <= brand_current_index < len(current_elements)
                or not isinstance(brand_next_index, int)
                or isinstance(brand_next_index, bool)
                or not 0 <= brand_next_index < len(next_elements)
            ):
                raise V4FocusError("v6 Bing ADD lacks valid brand anchors")
            brand_left = current_elements[brand_current_index]
            brand_right = next_elements[brand_next_index]
            if not (
                normalize_text(brand_left.get("content")) == "microsoft bing"
                and normalize_text(brand_right.get("content")) == "microsoft bing"
                and bbox_iou(brand_left["bbox"], brand_right["bbox"]) >= 0.70
                and center_distance(brand_left["bbox"], brand_right["bbox"])
                <= 16.0
            ):
                raise V4FocusError("v6 Bing ADD lacks stable brand identity")
            raw_click_contract = _raw_mouse_click_contract(
                source_provenance.get("raw_action"),
                candidate_point=_action_point(annotation.get("action") or {}),
                current_image=current_image,
            )
            action_description = normalize_text(
                source_provenance.get("action_description")
            )
            if not (
                _host_matches(source_provenance.get("current_domain"), "bing.com")
                and _host_matches(source_provenance.get("next_domain"), "bing.com")
                and re.fullmatch(
                    r"click on (?:textarea|form) element", action_description
                )
                and raw_click_contract is not None
                and all(
                    evidence.get(key) == value
                    for key, value in raw_click_contract.items()
                )
                and evidence.get("bing_current_host")
                == normalize_text(source_provenance.get("current_domain"))
                and evidence.get("bing_next_host")
                == normalize_text(source_provenance.get("next_domain"))
            ):
                raise V4FocusError("v7 Bing ADD lacks exact raw click provenance")
        if reason == "search_mode_recent_clear_all_anchor_add":
            if right.get("element_type") not in V7_SEARCH_MODE_CLEAR_ALL_TYPES:
                raise V4FocusError("v7 search-mode ADD has invalid clear-all type")
            if any(
                _strip_visual_role_prefix(normalize_text(element.get("content")))
                == "clear all"
                for element in current_elements
            ):
                raise V4FocusError("v7 search-mode ADD was not newly introduced")
            if any(
                match.get("next_index") == next_index
                for match in annotation.get("matches") or []
            ):
                raise V4FocusError("v7 search-mode ADD is not unmatched")
            search_matches = [
                match
                for match in annotation.get("matches") or []
                if match.get("method") == "search_mode_slot"
            ]
            if len(search_matches) != 1:
                raise V4FocusError("v7 search-mode ADD lacks unique slot match")
            search_match = search_matches[0]
            search_current_index = search_match.get("current_index")
            search_next_index = search_match.get("next_index")
            if (
                evidence.get("search_mode_current_index") != search_current_index
                or evidence.get("search_mode_next_index") != search_next_index
                or not isinstance(search_current_index, int)
                or isinstance(search_current_index, bool)
                or not 0 <= search_current_index < len(current_elements)
                or not isinstance(search_next_index, int)
                or isinstance(search_next_index, bool)
                or not 0 <= search_next_index < len(next_elements)
            ):
                raise V4FocusError("v7 search-mode ADD slot evidence mismatch")
            search_left = current_elements[search_current_index]
            search_right = next_elements[search_next_index]
            search_diagnostic = (
                (annotation.get("special_match_diagnostics") or {}).get(
                    "search_mode_slot"
                )
                or {}
            )
            if not (
                search_diagnostic.get("accepted")
                and search_diagnostic.get("current_element_index")
                == search_current_index
                and search_diagnostic.get("next_element_index") == search_next_index
                and normalize_text(search_left.get("content")) == "popular searches"
                and _strip_visual_role_prefix(
                    normalize_text(search_right.get("content"))
                )
                == "recent searches"
                and vertical_overlap_ratio(search_right["bbox"], right["bbox"])
                >= V7_SEARCH_MODE_CLEAR_ALL_MIN_VERTICAL_OVERLAP
                and abs(
                    float(search_right["bbox"][1]) - float(right["bbox"][1])
                )
                <= V7_SEARCH_MODE_CLEAR_ALL_MAX_CENTER_Y_DELTA
                and float(right["bbox"][0]) > float(search_right["bbox"][0])
            ):
                raise V4FocusError("v7 search-mode ADD lacks row-local slot proof")
        return
    if (
        not isinstance(current_index, int)
        or isinstance(current_index, bool)
        or not 0 <= current_index < len(current_elements)
    ):
        raise V4FocusError("UPDATE focus references invalid current element")
    if (
        not isinstance(next_index, int)
        or isinstance(next_index, bool)
        or not 0 <= next_index < len(next_elements)
    ):
        raise V4FocusError("UPDATE focus references invalid next element")
    pair_matches = [
        match
        for match in annotation.get("matches") or []
        if match.get("current_index") == current_index
        and match.get("next_index") == next_index
    ]
    if len(pair_matches) != 1:
        raise V4FocusError(f"v6 rule {reason} lacks a unique element match")
    method = str(pair_matches[0].get("method") or "")
    if method not in contract["methods"]:
        raise V4FocusError(f"v6 rule {reason} conflicts with match method {method!r}")
    evidence_method = str((unit.get("evidence") or {}).get("match_method") or "")
    if evidence_method != method:
        raise V4FocusError(f"v6 rule {reason} evidence method mismatch")
    evidence = unit.get("evidence") or {}
    for key in ("bbox_iou", "content_similarity"):
        if evidence.get(key) != pair_matches[0].get(key):
            raise V4FocusError(f"v6 rule {reason} has inconsistent {key} evidence")
    region_diff = evidence.get("same_region_pixel_diff")
    identity_diff = evidence.get("old_to_new_crop_pixel_diff")
    region_visible = bool(
        isinstance(region_diff, (int, float))
        and not isinstance(region_diff, bool)
        and math.isfinite(float(region_diff))
        and float(region_diff) >= 0.008
    )
    identity_visible = bool(
        isinstance(identity_diff, (int, float))
        and not isinstance(identity_diff, bool)
        and math.isfinite(float(identity_diff))
        and float(identity_diff) >= 0.02
    )
    special_identity_visible = bool(
        isinstance(identity_diff, (int, float))
        and not isinstance(identity_diff, bool)
        and math.isfinite(float(identity_diff))
        and float(identity_diff) >= SPECIAL_MATCH_VISUAL_DIFF
    )
    special_reasons = {
        "calendar_month_slot_content_update",
        "autocomplete_commit_input_content_update",
        "clicked_exact_text_toggle_content_update",
        "typed_value_overstock_query_marker_slot_update",
        "search_mode_recent_header_content_update",
        "sticky_pos_header_content_update",
        "drag_sort_content_update",
        "drag_form_control_content_update",
        "drag_binary_selector_content_update",
    }
    if reason == "state_flip_at_click_target":
        if not region_visible or float(region_diff) < 0.01:
            raise V4FocusError("v5 state focus lacks visual evidence")
    elif reason in special_reasons:
        if not (region_visible or special_identity_visible):
            raise V4FocusError(f"v6 rule {reason} lacks visual evidence")
    elif not (region_visible or identity_visible):
        raise V4FocusError(f"v6 rule {reason} lacks visual evidence")
    left_element = current_elements[current_index]
    right_element = next_elements[next_index]
    if reason != "state_flip_at_click_target" and normalize_text(
        left_element.get("content")
    ) == normalize_text(right_element.get("content")):
        raise V4FocusError(f"v6 content rule {reason} does not change content")
    if reason == "typed_value_exact_spatial_update" and (
        float(pair_matches[0].get("bbox_iou") or 0.0) < 0.75
        or float(pair_matches[0].get("confidence") or 0.0) < 0.85
    ):
        raise V4FocusError("v5 spatial typed focus lacks strict local identity")
    if reason == "typed_value_exact_direct_slot_update" and (
        vertical_overlap_ratio(left_element["bbox"], right_element["bbox"]) < 0.75
        or center_distance(left_element["bbox"], right_element["bbox"]) > 80.0
        or not _bbox_center_inside(right_element["bbox"], left_element["bbox"], padding=12.0)
    ):
        raise V4FocusError("v5 direct typed focus lacks slot geometry")
    if reason == "typed_value_deterministic_echo_update" and (
        float(pair_matches[0].get("bbox_iou") or 0.0) < 0.75
        or float(pair_matches[0].get("center_distance") or 0.0) > 24.0
        or vertical_overlap_ratio(left_element["bbox"], right_element["bbox"]) < 0.75
    ):
        raise V4FocusError("v5 typed echo lacks strict local identity")
    if reason.startswith("typed_value_exact_"):
        typed = _typed_text(annotation.get("action") or {})
        left_value = str(current_elements[current_index].get("normalized_content") or "")
        right_value = str(next_elements[next_index].get("normalized_content") or "")
        if not (
            typed
            and _typed_pair_has_predictable_target(left_value, right_value, typed)
            and not _ui_value_matches_typed(left_value, typed)
        ):
            raise V4FocusError(f"v5 rule {reason} lacks exact typed-value evidence")
    if reason == "typed_value_overstock_query_marker_slot_update":
        typed = _typed_text(annotation.get("action") or {})
        raw_candidate_text = (annotation.get("action") or {}).get("text")
        diagnostic = (
            (annotation.get("special_match_diagnostics") or {}).get(
                "typed_overstock_query_marker_slot"
            )
            or {}
        )
        raw_target = diagnostic.get("raw_input_target_rect")
        raw_contract = diagnostic.get("raw_contract")
        raw = source_provenance.get("raw_action")
        raw = raw if isinstance(raw, dict) else {}
        attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
        data = attributes.get("data") if isinstance(attributes.get("data"), dict) else {}
        node = data.get("node") if isinstance(data.get("node"), dict) else {}
        action_rule = raw.get("actionRuleSetting")
        action_rule_type = (
            action_rule.get("type") if isinstance(action_rule, dict) else None
        )
        action_rule_type = (
            action_rule_type if isinstance(action_rule_type, dict) else {}
        )
        expected_raw_contract = {
            "current_domain": normalize_text(source_provenance.get("current_domain")),
            "next_domain": normalize_text(source_provenance.get("next_domain")),
            "host": normalize_text(raw.get("host")),
            "path": str(raw.get("path") or "").strip(),
            "selector": str(raw.get("selector") or "").strip(),
            "raw_type": normalize_text(raw.get("type")),
            "node_name": normalize_text(node.get("name")),
            "role": normalize_text(data.get("role")),
            "operation_type": normalize_text(action_rule_type.get("operationType")),
            "raw_value": normalize_text(raw.get("value")),
        }
        replay_target, replay_rect_contract = _strict_overstock_input_target_rect(
            source_row or {},
            current_image=current_image,
        )
        if not (
            2 <= len(typed) <= 128
            and isinstance(raw_candidate_text, str)
            and raw_candidate_text == raw_candidate_text.strip()
            and normalize_text(raw_candidate_text) == typed
            and left_element.get("element_type") in TYPED_DIRECT_ELEMENT_TYPES
            and right_element.get("element_type") in TYPED_DIRECT_ELEMENT_TYPES
            and str(left_element.get("content") or "").strip()
            in {
                "Search Crazy-Good Deals",
                "Q Search Crazy-Good Deals",
                "QSearch Crazy-Good Deals",
            }
            and str(right_element.get("content") or "").strip()
            == f"Q {raw_candidate_text}"
            and vertical_overlap_ratio(
                left_element["bbox"], right_element["bbox"]
            )
            >= 0.75
            and center_distance(left_element["bbox"], right_element["bbox"])
            <= 80.0
            and _bbox_center_inside(
                right_element["bbox"], left_element["bbox"], padding=12.0
            )
            and 0.50
            <= float(right_element["bbox"][3])
            / max(1.0, float(left_element["bbox"][3]))
            <= 2.0
            and isinstance(raw_target, list)
            and len(raw_target) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in raw_target
            )
            and replay_target is not None
            and raw_target == list(replay_target)
            and diagnostic.get("raw_rect_contract") == replay_rect_contract
            and diagnostic.get("raw_rect_ocr_padding")
            == V7_RAW_RECT_OCR_PADDING
            and _bbox_inside(
                left_element["bbox"],
                raw_target,
                padding=V7_RAW_RECT_OCR_PADDING,
            )
            and _bbox_inside(
                right_element["bbox"],
                raw_target,
                padding=V7_RAW_RECT_OCR_PADDING,
            )
            and raw_contract == expected_raw_contract
            and expected_raw_contract
            == {
                "current_domain": V7_OVERSTOCK_HOST,
                "next_domain": V7_OVERSTOCK_HOST,
                "host": V7_OVERSTOCK_HOST,
                "path": V7_OVERSTOCK_INPUT_PATH,
                "selector": V7_OVERSTOCK_INPUT_SELECTOR,
                "raw_type": "type",
                "node_name": "input",
                "role": "textbox",
                "operation_type": "replace",
                "raw_value": typed,
            }
        ):
            raise V4FocusError("v7 Overstock typed marker lacks exact slot/provenance evidence")
        current_raw_slots = [
            index
            for index, element in enumerate(current_elements)
            if element.get("element_type") in TYPED_DIRECT_ELEMENT_TYPES
            and _bbox_inside(
                element["bbox"], raw_target, padding=V7_RAW_RECT_OCR_PADDING
            )
        ]
        next_raw_slots = [
            index
            for index, element in enumerate(next_elements)
            if element.get("element_type") in TYPED_DIRECT_ELEMENT_TYPES
            and _bbox_inside(
                element["bbox"], raw_target, padding=V7_RAW_RECT_OCR_PADDING
            )
        ]
        if current_raw_slots != [current_index] or next_raw_slots != [next_index]:
            raise V4FocusError("v7 Overstock typed marker raw slot is not unique")
    if reason == "typed_value_deterministic_echo_update":
        typed = _typed_text(annotation.get("action") or {})
        diagnostic = (
            (annotation.get("special_match_diagnostics") or {}).get("typed_echo")
            or {}
        )
        anchor_current = diagnostic.get("typed_anchor_current_index")
        anchor_next = diagnostic.get("typed_anchor_next_index")
        if (
            not diagnostic.get("accepted")
            or not isinstance(anchor_current, int)
            or not 0 <= anchor_current < len(current_elements)
            or not isinstance(anchor_next, int)
            or not 0 <= anchor_next < len(next_elements)
        ):
            raise V4FocusError("v5 typed echo lacks a valid primary anchor")
        anchor_matches = [
            match
            for match in annotation.get("matches") or []
            if match.get("current_index") == anchor_current
            and match.get("next_index") == anchor_next
        ]
        if len(anchor_matches) != 1 or not _typed_pair_has_predictable_target(
            current_elements[anchor_current]["normalized_content"],
            next_elements[anchor_next]["normalized_content"],
            typed,
        ):
            raise V4FocusError("v5 typed echo primary is not action-exact")
        anchor_path = diagnostic.get("typed_anchor_path")
        direct_diagnostic = (
            (annotation.get("special_match_diagnostics") or {}).get(
                "typed_direct_slot"
            )
            or {}
        )
        if anchor_path == "direct":
            if anchor_matches[0].get("method") != "typed_direct_slot":
                raise V4FocusError("v5 typed echo direct anchor method mismatch")
        elif anchor_path == "strict_spatial_multiplicity":
            if (
                anchor_matches[0].get("method") != "spatial_mutual"
                or direct_diagnostic.get("rejection") != "next_value_not_unique"
                or int(direct_diagnostic.get("next_value_count") or 0) < 2
            ):
                raise V4FocusError("v5 typed echo spatial fallback is not justified")
        else:
            raise V4FocusError("v5 typed echo has an unknown anchor path")
        old_value = _normalize_echo_text(
            _strip_typed_slot_caret(
                current_elements[anchor_current]["normalized_content"]
            )
        )
        typed_value = _normalize_echo_text(typed)
        left_text = _normalize_echo_text(left_element.get("content"))
        right_text = _normalize_echo_text(right_element.get("content"))
        if (
            not old_value
            or old_value == typed_value
            or left_text.count(old_value) != 1
            or len(left_text) - len(old_value) < 6
            or left_text.replace(old_value, typed_value, 1) != right_text
            or right_text.count(typed_value) != 1
        ):
            raise V4FocusError("v5 typed echo is not an exact unique wrapper substitution")
    if reason == "clicked_exact_text_toggle_content_update":
        left = current_elements[current_index]
        right = next_elements[next_index]
        left_bbox = left.get("bbox")
        pair = (
            normalize_text(left.get("content")),
            normalize_text(right.get("content")),
        )
        if (
            pair not in CLICK_TEXT_TOGGLE_PAIRS
            or not isinstance(left_bbox, (list, tuple))
            or len(left_bbox) != 4
            or not _bbox_contains_point(
                left_bbox,
                _action_point(annotation.get("action") or {}),
                padding=2.0,
            )
        ):
            raise V4FocusError("v5 click toggle lacks exact clicked lexical evidence")
    if reason == "state_flip_at_click_target":
        click_point = _action_point(annotation.get("action") or {})
        if (
            left_element.get("content") != right_element.get("content")
            or left_element.get("element_type") != right_element.get("element_type")
            or left_element.get("state") == right_element.get("state")
            or not (
                _bbox_contains_point(left_element["bbox"], click_point, padding=10.0)
                or _bbox_contains_point(right_element["bbox"], click_point, padding=10.0)
            )
        ):
            raise V4FocusError("v5 state focus lacks clicked state-transition evidence")

    v6_diagnostic_by_reason = {
        "search_mode_recent_header_content_update": "search_mode_slot",
        "sticky_pos_header_content_update": "sticky_pos_header",
        "drag_sort_content_update": "drag_sort_slot",
        "drag_form_control_content_update": "drag_form_control_slot",
        "drag_binary_selector_content_update": "drag_binary_selector_slot",
        "typed_value_overstock_query_marker_slot_update": "typed_overstock_query_marker_slot",
    }
    if reason in v6_diagnostic_by_reason:
        diagnostic_name = v6_diagnostic_by_reason[reason]
        diagnostic = (
            (annotation.get("special_match_diagnostics") or {}).get(diagnostic_name)
            or {}
        )
        if (
            not diagnostic.get("accepted")
            or diagnostic.get("current_element_index") != current_index
            or diagnostic.get("next_element_index") != next_index
        ):
            raise V4FocusError(f"v6 rule {reason} lacks canonical detector provenance")
        minimum_stable_anchors = (
            V6_SEARCH_MODE_MIN_STABLE_ANCHORS
            if reason == "search_mode_recent_header_content_update"
            else 4
            if reason == "typed_value_overstock_query_marker_slot_update"
            else V6_MIN_STABLE_ANCHORS
        )
        coherent_drag_retention = bool(
            reason in {"drag_sort_content_update", "drag_form_control_content_update"}
            and diagnostic.get("coherent_translation_retention")
            and isinstance(diagnostic.get("dominant_translation"), list)
            and len(diagnostic["dominant_translation"]) == 3
            and int(diagnostic["dominant_translation"][2]) >= V6_MIN_STABLE_ANCHORS
        )
        if reason != "sticky_pos_header_content_update" and (
            (
                float(diagnostic.get("retention_ratio") or 0.0)
                < V6_MIN_RETENTION_RATIO
                and not coherent_drag_retention
            )
            or int(diagnostic.get("stable_anchor_count") or 0) < minimum_stable_anchors
        ):
            raise V4FocusError(f"v6 rule {reason} lacks retained-page evidence")
    if reason == "search_mode_recent_header_content_update" and not (
        normalize_text(left_element.get("content")) == "popular searches"
        and _strip_visual_role_prefix(normalize_text(right_element.get("content")))
        == "recent searches"
        and center_distance(left_element["bbox"], right_element["bbox"]) <= 80.0
        and horizontal_overlap_ratio(left_element["bbox"], right_element["bbox"]) >= 0.60
    ):
        raise V4FocusError("v6 search-mode update lacks exact slot semantics")
    if reason == "search_mode_recent_header_content_update":
        diagnostic = (
            (annotation.get("special_match_diagnostics") or {}).get(
                "search_mode_slot"
            )
            or {}
        )
        click_point = _action_point(annotation.get("action") or {})
        raw_click_contract = _raw_mouse_click_contract(
            source_provenance.get("raw_action"),
            candidate_point=click_point,
            current_image=current_image,
        )
        action_description = normalize_text(
            source_provenance.get("action_description")
        )
        action_target_index = diagnostic.get("action_target_index")
        search_targets = []
        if raw_click_contract is not None:
            raw_grid_point = raw_click_contract["raw_click_grid"]
            search_targets = [
                index
                for index, element in enumerate(current_elements)
                if (
                    element.get("element_type") == "input"
                    or _strip_visual_role_prefix(
                        normalize_text(element.get("content"))
                    ).startswith("q search ")
                )
                and _point_with_axis_padding(
                    element["bbox"],
                    click_point,
                    x_padding=V7_FOX_ACTION_TARGET_X_PADDING,
                    y_padding=V7_FOX_ACTION_TARGET_Y_PADDING,
                )
                and _point_with_axis_padding(
                    element["bbox"],
                    raw_grid_point,
                    x_padding=V7_FOX_ACTION_TARGET_X_PADDING,
                    y_padding=V7_FOX_ACTION_TARGET_Y_PADDING,
                )
            ]
        if not (
            _host_matches(source_provenance.get("current_domain"), "foxsports.com")
            and _host_matches(source_provenance.get("next_domain"), "foxsports.com")
            and action_description
            in {"click on input element", "click on button element"}
            and raw_click_contract is not None
            and search_targets == [action_target_index]
            and diagnostic.get("fox_current_host")
            == normalize_text(source_provenance.get("current_domain"))
            and diagnostic.get("fox_next_host")
            == normalize_text(source_provenance.get("next_domain"))
            and all(
                diagnostic.get(key) == value
                for key, value in raw_click_contract.items()
            )
            and diagnostic.get("action_target_x_padding")
            == V7_FOX_ACTION_TARGET_X_PADDING
            and diagnostic.get("action_target_y_padding")
            == V7_FOX_ACTION_TARGET_Y_PADDING
        ):
            raise V4FocusError("v7 search-mode update lacks exact raw click provenance")
    if reason == "sticky_pos_header_content_update":
        try:
            action_delta_y = float((annotation.get("action") or {}).get("delta_y", 0.0))
        except (TypeError, ValueError):
            action_delta_y = 0.0
        diagnostic = (
            (annotation.get("special_match_diagnostics") or {}).get("sticky_pos_header")
            or {}
        )
        raw_scroll_contract = _raw_scroll_contract(
            source_provenance.get("raw_action"),
            action=annotation.get("action") or {},
            current_image=current_image,
        )
        if not (
            POS_HEADER_RE.fullmatch(normalize_text(left_element.get("content")))
            and POS_HEADER_RE.fullmatch(normalize_text(right_element.get("content")))
            and abs(action_delta_y) >= 300.0
            and max(float(left_element["bbox"][1]), float(right_element["bbox"][1])) <= 150.0
            and center_distance(left_element["bbox"], right_element["bbox"]) <= 22.0
            and horizontal_overlap_ratio(left_element["bbox"], right_element["bbox"]) >= 0.95
            and int(diagnostic.get("fixed_scroll_anchor_count") or 0)
            >= V6_STICKY_MIN_FIXED_ANCHORS
            and int(diagnostic.get("target_structure_count") or 0) > 0
            and _host_matches(
                source_provenance.get("current_domain"), "merriam-webster.com"
            )
            and _host_matches(
                source_provenance.get("next_domain"), "merriam-webster.com"
            )
            and raw_scroll_contract is not None
            and diagnostic.get("merriam_current_host")
            == normalize_text(source_provenance.get("current_domain"))
            and diagnostic.get("merriam_next_host")
            == normalize_text(source_provenance.get("next_domain"))
            and all(
                diagnostic.get(key) == value
                for key, value in raw_scroll_contract.items()
            )
        ):
            raise V4FocusError("v6 sticky POS update lacks scroll-anchor evidence")
    if reason in {"drag_sort_content_update", "drag_form_control_content_update"}:
        start = _action_point(annotation.get("action") or {}, "from_coordinate")
        end = _action_point(annotation.get("action") or {}, "to_coordinate")
        translation = _dominant_page_translation(
            parse_markdown_spans(annotation.get("current_md")),
            parse_markdown_spans(annotation.get("target_next_md")),
        )
        projected_left = _translated_bbox(left_element["bbox"], translation)
        endpoint_padding = (
            V6_DRAG_SORT_ENDPOINT_PADDING
            if reason == "drag_sort_content_update"
            else V6_DRAG_FORM_ENDPOINT_PADDING
        )
        if not (
            start is not None
            and end is not None
            and math.hypot(end[0] - start[0], end[1] - start[1])
            >= V6_DIRECT_DRAG_MIN_DISPLACEMENT
            and
            _bbox_contains_point(left_element["bbox"], start, padding=18.0)
            and _bbox_contains_point(
                right_element["bbox"], end, padding=endpoint_padding
            )
            and min(
                center_distance(left_element["bbox"], right_element["bbox"]),
                center_distance(projected_left, right_element["bbox"]),
            )
            <= 28.0
        ):
            raise V4FocusError("v6 drag direct update lacks action/slot evidence")
        if reason == "drag_sort_content_update" and not (
            "sort" in normalize_text(left_element.get("content"))
            and "sort" in normalize_text(right_element.get("content"))
            and ":" in normalize_text(left_element.get("content"))
            and ":" in normalize_text(right_element.get("content"))
        ):
            raise V4FocusError("v6 drag sort update lacks sort semantics")
        if reason == "drag_form_control_content_update" and not (
            left_element.get("element_type") == right_element.get("element_type")
            and left_element.get("element_type") in {"input", "select"}
        ):
            raise V4FocusError("v6 drag form update changes control type")
    if reason == "drag_binary_selector_content_update":
        points = _drag_action_points(annotation.get("action") or {})
        labels = [
            element
            for element in current_elements
            if normalize_text(element.get("content")) == "new/used"
            and any(
                _bbox_contains_point(element["bbox"], point, padding=20.0)
                for point in points
                if point is not None
            )
        ]
        if not (
            len(labels) == 1
            and points[0] is not None
            and points[1] is not None
            and math.hypot(
                points[1][0] - points[0][0], points[1][1] - points[0][1]
            )
            >= 20.0
            and normalize_text(left_element.get("content")) == "new"
            and normalize_text(right_element.get("content")) == "used"
            and center_distance(left_element["bbox"], right_element["bbox"]) <= 10.0
        ):
            raise V4FocusError("v6 drag binary selector lacks labeled endpoint evidence")


def _validate_v7_focus_selection(annotation: dict[str, Any]) -> None:
    focus_units = annotation.get("focus_units") or []
    diagnostics = annotation.get("special_match_diagnostics") or {}
    selection = diagnostics.get("focus_unit_selection") or {}
    if not (
        selection.get("schema") == "v7-global-focus-priority-v1"
        and selection.get("focus_cap") == V6_MAX_FOCUS_UNITS
        and selection.get("add_cap") == 1
    ):
        raise V4FocusError("v7 focus selection lacks the global priority contract")

    actual_updates = sorted(
        int(unit["next_element_index"])
        for unit in focus_units
        if unit.get("kind") == "UPDATE"
    )
    actual_adds = [
        int(unit["next_element_index"])
        for unit in focus_units
        if unit.get("kind") == "ADD"
    ]
    if (
        selection.get("selected_updates") != actual_updates
        or selection.get("selected_add")
        != (actual_adds[0] if actual_adds else None)
        or selection.get("emitted_focus_units") != len(focus_units)
    ):
        raise V4FocusError("v7 focus selection does not match emitted units")

    groups = selection.get("groups")
    if not isinstance(groups, list):
        raise V4FocusError("v7 focus selection groups are missing")
    selected_members: list[tuple[str, int, str]] = []
    selectable: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("members"), list):
            raise V4FocusError("v7 focus selection contains a malformed group")
        name = str(group.get("name") or "")
        members = group["members"]
        if name in {"fox_search_mode_bundle", "calendar_two_month_bundle"}:
            if group.get("priority") != V7_ATOMIC_PRIORITY or not group.get("atomic"):
                raise V4FocusError("v7 atomic focus group has invalid priority")
            if "rejection" not in group and len(members) != 2:
                raise V4FocusError("v7 atomic focus group is not a complete pair")
        elif members:
            member = members[0]
            expected_priority = (
                V7_SPECIAL_PRIORITY
                if member.get("kind") == "ADD"
                else _v7_update_priority({"reasons": [member.get("reason")]})
            )
            if group.get("priority") != expected_priority:
                raise V4FocusError("v7 focus group has invalid mechanism priority")
        for member in members:
            if not (
                isinstance(member, dict)
                and member.get("kind") in {"ADD", "UPDATE"}
                and isinstance(member.get("next_element_index"), int)
                and not isinstance(member.get("next_element_index"), bool)
                and isinstance(member.get("reason"), str)
            ):
                raise V4FocusError("v7 focus selection has a malformed member")
            if group.get("selected"):
                selected_members.append(
                    (
                        str(member["kind"]),
                        int(member["next_element_index"]),
                        str(member["reason"]),
                    )
                )
        if group.get("rejection") not in {
            "missing_header_focus_partner",
            "requires_exactly_two_updates",
        }:
            selectable.append(group)

    actual_members = sorted(
        (
            str(unit.get("kind")),
            int(unit["next_element_index"]),
            str((unit.get("reasons") or [""])[0]),
        )
        for unit in focus_units
    )
    if sorted(selected_members) != actual_members:
        raise V4FocusError("v7 selected groups do not exactly cover emitted focus")


    selected_count = 0
    selected_add = False
    for group in sorted(
        selectable,
        key=lambda value: (
            -int(value["priority"]),
            int(value["stable_next_index"]),
            str(value["name"]),
        ),
    ):
        members = group["members"]
        group_has_add = any(member["kind"] == "ADD" for member in members)
        expected_selected = bool(
            selected_count + len(members) <= V6_MAX_FOCUS_UNITS
            and not (group_has_add and selected_add)
        )
        if bool(group.get("selected")) != expected_selected:
            raise V4FocusError("v7 focus selection violates stable greedy priority")
        if expected_selected:
            selected_count += len(members)
            selected_add = selected_add or group_has_add

    reasons = [
        str((unit.get("reasons") or [""])[0]) for unit in focus_units
    ]
    clear_count = reasons.count("search_mode_recent_clear_all_anchor_add")
    header_count = reasons.count("search_mode_recent_header_content_update")
    if clear_count and (clear_count != 1 or header_count != 1):
        raise V4FocusError("v7 Fox search-mode bundle was emitted non-atomically")
    calendar_count = reasons.count("calendar_month_slot_content_update")
    if calendar_count not in {0, 2}:
        raise V4FocusError("v7 calendar bundle was emitted non-atomically")


def _validate_v6_stats(annotation: dict[str, Any]) -> None:
    focus_units = annotation.get("focus_units") or []
    review_candidates = annotation.get("review_candidates") or []
    matches = annotation.get("matches") or []
    stats = annotation.get("stats") or {}
    current = parse_markdown_spans(annotation.get("current_md"))
    nxt = parse_markdown_spans(annotation.get("target_next_md"))
    matched_current = {int(match["current_index"]) for match in matches}
    matched_next = {int(match["next_index"]) for match in matches}
    next_leaf_count = sum(element.element_type != "block" for element in nxt.elements)
    retained_leaf_count = sum(
        nxt.elements[int(match["next_index"])].element_type != "block"
        and (
            match.get("method")
            in {
                "exact",
                "semantic",
                "state_identity",
                "calendar_month_slot",
                "autocomplete_commit",
                "typed_direct_slot",
                "typed_echo_template",
                "click_text_toggle",
                *V6_UPDATE_METHODS,
            }
            or (
                match.get("method") == "spatial_mutual"
                and float(match.get("bbox_iou") or 0.0) >= 0.75
            )
        )
        for match in matches
    )
    retention_ratio = retained_leaf_count / max(1, next_leaf_count)
    calendar_matches = sum(
        match.get("method") == "calendar_month_slot" for match in matches
    )
    sticky_matches = sum(
        match.get("method") == "sticky_pos_header" for match in matches
    )
    special_diagnostics = annotation.get("special_match_diagnostics") or {}
    add_diagnostic = special_diagnostics.get("v6_add_anchor") or {}
    selection_diagnostic = special_diagnostics.get("focus_unit_selection") or {}
    coherent_drag_retention = any(
        bool((special_diagnostics.get(name) or {}).get("accepted"))
        and bool(
            (special_diagnostics.get(name) or {}).get(
                "coherent_translation_retention"
            )
        )
        for name in ("drag_sort_slot", "drag_form_control_slot")
    )
    bing_overlay_retention = bool(
        add_diagnostic.get("accepted")
        and add_diagnostic.get("selected_reason")
        == "bing_search_overlay_anchor_add"
        and add_diagnostic.get("selected_retention_policy")
        == "bing_overlay_four_anchor"
    )
    typed = _typed_text(annotation.get("action") or {})
    typed_anchor_count = sum(
        (
            match.get("method") == "typed_direct_slot"
            or (
                match.get("method") == "spatial_mutual"
                and float(match.get("bbox_iou") or 0.0) >= 0.75
            )
        )
        and bool(typed)
        and _typed_pair_has_predictable_target(
            current.elements[int(match["current_index"])].normalized_content,
            nxt.elements[int(match["next_index"])].normalized_content,
            typed,
        )
        and current.elements[int(match["current_index"])].normalized_content
        != nxt.elements[int(match["next_index"])].normalized_content
        for match in matches
    )
    expected = {
        "base_units": len(annotation.get("base_units") or []),
        "focus_units": len(focus_units),
        "add_focus": sum(unit.get("kind") == "ADD" for unit in focus_units),
        "update_focus": sum(unit.get("kind") == "UPDATE" for unit in focus_units),
        "review_candidates": len(review_candidates),
        "matched_elements": len(matches),
        "unmatched_current_elements": len(current.elements) - len(matched_current),
        "unmatched_next_elements": len(nxt.elements) - len(matched_next),
        "retention_ratio": round(retention_ratio, 6),
        "general_page_replace": (
            retention_ratio < 0.20
            and calendar_matches == 0
            and not coherent_drag_retention
            and not bing_overlay_retention
        ),
        "navigation_page_replace": bool(
            annotation.get("action_bucket") == "navigation"
            and normalize_text((annotation.get("action") or {}).get("type"))
            == "navigate"
            and retention_ratio < 0.20
        ),
        "typed_anchor_count": typed_anchor_count,
        "typed_direct_slot_matches": sum(
            match.get("method") == "typed_direct_slot" for match in matches
        ),
        "typed_overstock_query_marker_slot_matches": sum(
            match.get("method") == "typed_overstock_query_marker_slot"
            for match in matches
        ),
        "click_text_toggle_matches": sum(
            match.get("method") == "click_text_toggle" for match in matches
        ),
        "typed_echo_matches": sum(
            match.get("method") == "typed_echo_template" for match in matches
        ),
        "calendar_month_slot_matches": calendar_matches,
        "autocomplete_commit_matches": sum(
            match.get("method") == "autocomplete_commit" for match in matches
        ),
        "search_mode_slot_matches": sum(
            match.get("method") == "search_mode_slot" for match in matches
        ),
        "sticky_pos_header_matches": sticky_matches,
        "drag_sort_slot_matches": sum(
            match.get("method") == "drag_sort_slot" for match in matches
        ),
        "drag_form_control_slot_matches": sum(
            match.get("method") == "drag_form_control_slot" for match in matches
        ),
        "drag_binary_selector_slot_matches": sum(
            match.get("method") == "drag_binary_selector_slot" for match in matches
        ),
        "v6_add_anchor_detected": int(bool(add_diagnostic.get("accepted"))),
        "v6_add_anchor_matches": int(
            any(unit.get("kind") == "ADD" for unit in focus_units)
        ),
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise V4FocusError(f"v6 stats mismatch for {key}: {stats.get(key)!r} != {value!r}")
    if len(focus_units) > V6_MAX_FOCUS_UNITS:
        raise V4FocusError("v6 forbids more than two focus groups per sample")
    if sum(unit.get("kind") == "ADD" for unit in focus_units) > 1:
        raise V4FocusError("v6 permits only the top-ranked ADD anchor per sample")
    if selection_diagnostic.get("detected_add_candidates") != int(
        bool(add_diagnostic.get("accepted"))
    ):
        raise V4FocusError("v7 ADD detected/emitted accounting mismatch")
    _validate_v7_focus_selection(annotation)


def _validate_v6_matches(annotation: dict[str, Any]) -> None:
    """Bind every serialized match to the Markdown-derived pair and geometry."""

    current = parse_markdown_spans(annotation.get("current_md"))
    nxt = parse_markdown_spans(annotation.get("target_next_md"))
    allowed_methods = {
        "exact",
        "semantic",
        "state_identity",
        "spatial_mutual",
        "calendar_month_slot",
        "autocomplete_commit",
        "typed_direct_slot",
        "typed_echo_template",
        "click_text_toggle",
        *V6_UPDATE_METHODS,
    }
    seen_current: set[int] = set()
    seen_next: set[int] = set()
    for match in annotation.get("matches") or []:
        current_index = match.get("current_index")
        next_index = match.get("next_index")
        if (
            not isinstance(current_index, int)
            or isinstance(current_index, bool)
            or not 0 <= current_index < len(current.elements)
            or not isinstance(next_index, int)
            or isinstance(next_index, bool)
            or not 0 <= next_index < len(nxt.elements)
        ):
            raise V4FocusError("match references an invalid Markdown element")
        if current_index in seen_current or next_index in seen_next:
            raise V4FocusError("matches are not globally one-to-one")
        seen_current.add(current_index)
        seen_next.add(next_index)
        method = str(match.get("method") or "")
        if method not in allowed_methods:
            raise V4FocusError(f"unsupported match method: {method!r}")
        confidence = match.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise V4FocusError("match confidence is not finite in [0, 1]")
        left = current.elements[current_index]
        right = nxt.elements[next_index]
        expected_geometry = {
            "bbox_iou": round(bbox_iou(left.bbox, right.bbox), 4),
            "center_distance": round(center_distance(left.bbox, right.bbox), 4),
            "content_similarity": round(content_similarity(left, right), 4),
        }
        for key, expected in expected_geometry.items():
            if match.get(key) != expected:
                raise V4FocusError(
                    f"match {current_index}->{next_index} has forged {key}: "
                    f"{match.get(key)!r} != {expected!r}"
                )


def validate_focus_annotation(
    annotation: dict[str, Any],
    *,
    require_matcher_provenance: bool = False,
    source_row: dict[str, Any] | None = None,
    project_root: Path | None = None,
    image_module: Any | None = None,
) -> None:
    if annotation.get("schema_version") != FOCUS_LABEL_VERSION:
        raise V4FocusError(
            f"unsupported focus annotation schema: {annotation.get('schema_version')!r}"
        )
    if require_matcher_provenance:
        if source_row is None:
            raise V4FocusError("strict matcher validation requires the source row")
        expected_source_bindings = {
            "sample_id": str(source_row.get("id") or "").strip(),
            "transition_class": source_row.get("transition_class"),
            "action_bucket": str(source_row.get("action_bucket") or "").strip(),
            "action": _action_payload(source_row),
            "current_image_path": str(source_row.get("current_image_path") or ""),
            "next_image_path": str(source_row.get("next_image_path") or ""),
            "source": str((source_row.get("meta") or {}).get("dataset") or "unknown"),
        }
        for key, expected in expected_source_bindings.items():
            if annotation.get(key) != expected:
                raise V4FocusError(f"matcher annotation is not source-bound for {key}")
        if annotation.get("current_md") != parse_markdown_spans(
            source_row.get("current_md")
        ).text:
            raise V4FocusError("matcher current Markdown is not source-bound")
        if annotation.get("target_next_md") != parse_markdown_spans(
            source_row.get("target_next_md")
        ).text:
            raise V4FocusError("matcher target Markdown is not source-bound")
    reparsed_next = parse_markdown_spans(annotation.get("target_next_md"))
    expected_next_elements = [
        _element_dict(element) for element in reparsed_next.elements
    ]
    if require_matcher_provenance:
        reparsed_current = parse_markdown_spans(annotation.get("current_md"))
        expected_current_elements = [
            _element_dict(element) for element in reparsed_current.elements
        ]
        if annotation.get("current_elements") != expected_current_elements:
            raise V4FocusError("current_elements do not match reparsed current Markdown")
    if (
        require_matcher_provenance
        and annotation.get("next_elements") != expected_next_elements
    ):
        raise V4FocusError("next_elements do not match reparsed target Markdown")
    target = str(annotation.get("target_next_md") or "")
    if annotation.get("target_sha256") != sha256_text(target):
        raise V4FocusError("focus target SHA256 mismatch")
    target_length = len(target)
    base_units = annotation.get("base_units") or []
    expected_base_units = [asdict(span) for span in reparsed_next.base_units]
    if base_units != expected_base_units:
        raise V4FocusError(
            "canonical target requires Markdown-derived base_units"
        )
    if not base_units:
        raise V4FocusError("focus annotation has no base units")
    cursor = 0
    for span in base_units:
        start, end = int(span["start"]), int(span["end"])
        if start != cursor or end <= start or end > target_length:
            raise V4FocusError(f"base units do not partition target at {start}:{end}/{target_length}")
        cursor = end
    if cursor != target_length:
        raise V4FocusError(f"base units end at {cursor}, expected {target_length}")
    if require_matcher_provenance:
        _validate_v6_matches(annotation)

    seen_focus_elements: set[int] = set()
    next_elements = annotation.get("next_elements") or []
    focus_units = annotation.get("focus_units") or []
    review_candidates = annotation.get("review_candidates") or []
    source_current_image = None
    raw_bound_reasons = {
        "typed_value_overstock_query_marker_slot_update",
        "search_mode_recent_header_content_update",
        "search_mode_recent_clear_all_anchor_add",
        "sticky_pos_header_content_update",
        "bing_search_overlay_anchor_add",
    }
    needs_source_current_image = bool(
        require_matcher_provenance
        and any(
            set(str(reason) for reason in unit.get("reasons") or ())
            & raw_bound_reasons
            for unit in focus_units
        )
    )
    if needs_source_current_image:
        if source_row is None or image_module is None:
            raise V4FocusError("strict raw provenance validation requires image access")
        raw_path = Path(str(source_row.get("current_image_path") or ""))
        resolved = (
            raw_path
            if raw_path.is_absolute()
            else (project_root or Path.cwd()) / raw_path
        )
        try:
            source_current_image = image_module.open(resolved).convert("RGB")
        except Exception as exc:
            raise V4FocusError(
                f"strict raw provenance cannot load current image: {exc}"
            ) from exc
    if annotation.get("schema_version") == FOCUS_LABEL_VERSION:
        if review_candidates:
            raise V4FocusError(f"{FOCUS_LABEL_VERSION} forbids review candidates")
        if require_matcher_provenance:
            stats = annotation.get("stats") or {}
            if focus_units and annotation.get("action_bucket") == "navigation":
                raise V4FocusError(
                    f"{FOCUS_LABEL_VERSION} forbids navigation focus"
                )
            if annotation.get("action_bucket") == "scroll" and any(
                (unit.get("reasons") or [None])[0]
                != "sticky_pos_header_content_update"
                for unit in focus_units
            ):
                raise V4FocusError(
                    f"{FOCUS_LABEL_VERSION} permits only sticky POS scroll focus"
                )
            if bool(stats.get("general_page_replace")) and any(
                (unit.get("reasons") or [None])[0]
                != "sticky_pos_header_content_update"
                for unit in focus_units
            ):
                raise V4FocusError(
                    f"{FOCUS_LABEL_VERSION} forbids page-replacement focus"
                )
    for expected_focus_id, unit in enumerate(focus_units):
        if unit.get("kind") not in {"ADD", "UPDATE"}:
            raise V4FocusError(f"unsupported focus kind: {unit.get('kind')!r}")
        next_index = int(unit["next_element_index"])
        if next_index < 0 or next_index >= len(next_elements):
            raise V4FocusError(f"focus references invalid next element {next_index}")
        if int(unit.get("focus_id", -1)) != expected_focus_id:
            raise V4FocusError(f"focus_id must be contiguous at {expected_focus_id}")
        if next_index in seen_focus_elements:
            raise V4FocusError(f"next element {next_index} has multiple focus groups")
        seen_focus_elements.add(next_index)
        if next_index + 1 >= len(base_units):
            raise V4FocusError(f"focus references absent next element {next_index}")
        owner = base_units[next_index + 1]
        spans = unit.get("spans") or []
        if not spans:
            raise V4FocusError(f"focus {expected_focus_id} has no spans")
        if unit.get("kind") == "ADD":
            expected_line = next_elements[next_index].get("line_span")
            if unit.get("fields") != ["line"] or len(spans) != 1:
                raise V4FocusError("ADD focus must contain exactly the full line span")
            if int(spans[0]["start"]) != int(expected_line["start"]) or int(spans[0]["end"]) != int(expected_line["end"]):
                raise V4FocusError("ADD focus span is not the exact next element line")
        else:
            fields = unit.get("fields") or []
            if (
                not fields
                or fields != sorted(set(fields))
                or not set(fields).issubset({"content", "type", "state", "bbox", "value"})
            ):
                raise V4FocusError("UPDATE focus contains invalid fields")
            element = next_elements[next_index]
            field_spans = {
                "content": element.get("content_span"),
                "value": element.get("content_span"),
                "type": element.get("type_span"),
                "state": element.get("state_span"),
                "bbox": element.get("bbox_span"),
            }
            if any(field_spans[field] is None for field in fields):
                raise V4FocusError("UPDATE focus references an absent target field")
            expected_spans = {
                (int(field_spans[field]["start"]), int(field_spans[field]["end"]))
                for field in fields
            }
            actual_spans = {
                (int(span["start"]), int(span["end"])) for span in spans
            }
            if actual_spans != expected_spans:
                raise V4FocusError(
                    "UPDATE focus spans do not exactly match declared target fields"
                )
        for span in spans:
            start, end = int(span["start"]), int(span["end"])
            if start < 0 or end <= start or end > target_length:
                raise V4FocusError(f"focus span outside target: {start}:{end}/{target_length}")
            if start < int(owner["start"]) or end > int(owner["end"]):
                raise V4FocusError(f"focus span {start}:{end} is outside element base unit")
            span_text = target[start:end]
            if span.get("text") != span_text or span.get("text_sha256") != sha256_text(span_text):
                raise V4FocusError(f"focus span text/hash mismatch at {start}:{end}")
        if require_matcher_provenance:
            _validate_v6_focus_rule_contract(
                annotation,
                unit,
                source_row=source_row,
                current_image=source_current_image,
            )

    if require_matcher_provenance:
        _validate_v6_stats(annotation)

    seen_candidates: set[int] = set()
    for expected_candidate_id, unit in enumerate(review_candidates):
        if unit.get("kind") != "ADD":
            raise V4FocusError(f"unsupported review candidate kind: {unit.get('kind')!r}")
        next_index = int(unit["next_element_index"])
        if next_index < 0 or next_index >= len(next_elements):
            raise V4FocusError(f"review candidate references invalid next element {next_index}")
        if int(unit.get("candidate_id", -1)) != expected_candidate_id:
            raise V4FocusError(f"candidate_id must be contiguous at {expected_candidate_id}")
        if next_index in seen_focus_elements or next_index in seen_candidates:
            raise V4FocusError(f"next element {next_index} has conflicting focus/candidate groups")
        seen_candidates.add(next_index)
        if next_index + 1 >= len(base_units):
            raise V4FocusError(f"review candidate references absent next element {next_index}")
        owner = base_units[next_index + 1]
        spans = unit.get("spans") or []
        expected_line = next_elements[next_index].get("line_span")
        if unit.get("fields") != ["line"] or len(spans) != 1:
            raise V4FocusError("ADD review candidate must contain exactly the full line span")
        if int(spans[0]["start"]) != int(expected_line["start"]) or int(spans[0]["end"]) != int(expected_line["end"]):
            raise V4FocusError("ADD review span is not the exact next element line")
        for span in spans:
            start, end = int(span["start"]), int(span["end"])
            if start < int(owner["start"]) or end > int(owner["end"]):
                raise V4FocusError(f"review span {start}:{end} is outside element base unit")
            span_text = target[start:end]
            if span.get("text") != span_text or span.get("text_sha256") != sha256_text(span_text):
                raise V4FocusError(f"review span text/hash mismatch at {start}:{end}")

    if require_matcher_provenance:
        assert source_row is not None
        rebuilt = build_focus_annotation(
            source_row,
            project_root=project_root,
            image_module=image_module,
        )
        critical_keys = (
            "schema_version",
            "sample_id",
            "transition_class",
            "action_bucket",
            "action",
            "current_md",
            "target_next_md",
            "target_sha256",
            "base_units",
            "focus_units",
            "review_candidates",
            "current_elements",
            "next_elements",
            "matches",
            "special_match_diagnostics",
            "relations",
            "next_labels",
            "stats",
        )
        for key in critical_keys:
            if annotation.get(key) != rebuilt.get(key):
                raise V4FocusError(
                    f"matcher annotation differs from canonical detector replay for {key}"
                )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V4FocusError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise V4FocusError(f"non-object JSON row at {path}:{line_number}")
            rows.append(row)
    return rows
