from __future__ import annotations
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence
from mkd2world.training.markdown import ELEMENT_RE, STATE_PREFIX_RE, TYPE_SUFFIX_RE

ENVIRONMENT_RE = re.compile(r"^\[Environment:\s*[^\]]+\]$")
LINE_LEADER_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?")
MAIN_PREFIX_RE = re.compile(r"^\[(?P<content>[^\]]*)\](?:\((?P<type>[^)]+)\))?$")


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


def center_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def content_similarity(left: ParsedElementSpan, right: ParsedElementSpan) -> float:
    return SequenceMatcher(None, left.normalized_content, right.normalized_content).ratio()


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
