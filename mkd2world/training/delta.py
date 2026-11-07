"""Deterministic Local WM delta construction, parsing, and flat replay."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from mkd2world.training.markdown import ELEMENT_RE, MdElement, parse_md_elements


POSITION_JITTER_THRESHOLD = 2
UPDATE_CENTER_LIMIT = 120.0
UPDATE_DISTANCE_LIMIT = 0.42
DELTA_HEADER = "[State Change]"
NO_VISIBLE_CHANGE = "NO_VISIBLE_CHANGE"
OPERATION_ORDER = {"REMOVE": 0, "ADD": 1, "UPDATE": 2}
LEAF_TYPES = frozenset({"text", "link", "button", "input", "icon", "image", "toggle", "select"})
BLOCK_LABELS = frozenset({"Navigation", "Sidebar", "Footer", "Form", "Table", "Section"})
BLOCK_PAYLOAD_RE = re.compile(r"^\[Block:\s*(?P<label>[^\]]+)\]$")


class DeltaFormatError(ValueError):
    """Raised when model output is not a strict delta document."""


class DeltaReplayError(ValueError):
    """Raised when a valid delta cannot be applied to its current state."""


@dataclass(frozen=True)
class DeltaOperation:
    kind: str
    old: MdElement | None = None
    new: MdElement | None = None

    def serialize(self) -> str:
        if self.kind == "REMOVE" and self.old is not None:
            return f"REMOVE: {self.old.line}"
        if self.kind == "ADD" and self.new is not None:
            return f"ADD: {self.new.line}"
        if self.kind == "UPDATE" and self.old is not None and self.new is not None:
            return f"UPDATE: {self.old.line} => {self.new.line}"
        raise DeltaFormatError(f"invalid {self.kind} operation payload")


@dataclass(frozen=True)
class ParsedDelta:
    operations: tuple[DeltaOperation, ...]
    no_visible_change: bool = False

    def serialize(self) -> str:
        body = [NO_VISIBLE_CHANGE] if self.no_visible_change else [operation.serialize() for operation in self.operations]
        return "\n".join([DELTA_HEADER, *body]) + "\n"


@dataclass(frozen=True)
class PositionJitterRemoval:
    current_index: int
    next_index: int
    current_line: str
    next_line: str
    dx: float
    dy: float
    distance: float


@dataclass(frozen=True)
class DeltaBuildResult:
    delta: str
    target_next_md_clean: str
    position_jitter_removed: tuple[PositionJitterRemoval, ...]


def element_payload(element: MdElement) -> str:
    """Return the complete element syntax without its final coordinate box."""
    payload, separator, _ = element.line.rpartition("<")
    if not separator:
        raise ValueError(f"element line lacks coordinates: {element.line!r}")
    return unicodedata.normalize("NFKC", payload.strip())


def position_distance(left: MdElement, right: MdElement) -> float:
    return max(abs(left.cx - right.cx), abs(left.cy - right.cy))


def is_position_jitter(
    current: MdElement,
    nxt: MdElement,
    threshold: float = POSITION_JITTER_THRESHOLD,
) -> bool:
    """Return true only for identical payload/size with a small center displacement."""
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("position jitter threshold must be finite and non-negative")
    return (
        element_payload(current) == element_payload(nxt)
        and current.w == nxt.w
        and current.h == nxt.h
        and 0.0 < position_distance(current, nxt) <= threshold
    )


def element_distance(left: MdElement, right: MdElement) -> float:
    center = math.hypot(left.cx - right.cx, left.cy - right.cy) / 1_413.0
    size = (abs(left.w - right.w) + abs(left.h - right.h)) / 1_998.0
    text = 1.0 - SequenceMatcher(None, left.content.lower(), right.content.lower()).ratio()
    return 0.45 * center + 0.15 * size + 0.40 * text


def _exact_pairs(current: list[MdElement], nxt: list[MdElement]) -> list[tuple[int, int]]:
    next_by_line: dict[str, deque[int]] = defaultdict(deque)
    for index, element in enumerate(nxt):
        next_by_line[element.line].append(index)
    pairs = []
    for index, element in enumerate(current):
        candidates = next_by_line.get(element.line)
        if candidates:
            pairs.append((index, candidates.popleft()))
    return pairs


def _jitter_key(element: MdElement) -> tuple[str, float, float]:
    return element_payload(element), element.w, element.h


def _jitter_pairs(
    current: list[MdElement],
    nxt: list[MdElement],
    current_indexes: Iterable[int],
    next_indexes: Iterable[int],
    *,
    threshold: float,
) -> list[tuple[int, int]]:
    current_by_key: dict[tuple[str, float, float], list[int]] = defaultdict(list)
    next_by_key: dict[tuple[str, float, float], list[int]] = defaultdict(list)
    for index in current_indexes:
        current_by_key[_jitter_key(current[index])].append(index)
    for index in next_indexes:
        next_by_key[_jitter_key(nxt[index])].append(index)
    output = []
    for key in sorted(current_by_key, key=repr):
        edges = sorted(
            (
                position_distance(current[left], nxt[right]),
                left,
                right,
            )
            for left in current_by_key[key]
            for right in next_by_key.get(key, ())
        )
        used_current: set[int] = set()
        used_next: set[int] = set()
        for distance, left, right in edges:
            if distance > threshold:
                break
            if left in used_current or right in used_next:
                continue
            if not is_position_jitter(current[left], nxt[right], threshold):
                continue
            used_current.add(left)
            used_next.add(right)
            output.append((left, right))
    return sorted(output)


def _replace_target_elements(markdown: str, replacements: dict[int, str]) -> str:
    if not replacements:
        return markdown
    output = []
    element_index = 0
    for raw_line in markdown.splitlines():
        match = ELEMENT_RE.search(raw_line)
        if match is None:
            output.append(raw_line)
            continue
        replacement = replacements.get(element_index)
        if replacement is None:
            output.append(raw_line)
        else:
            output.append(raw_line[: match.start()] + replacement + raw_line[match.end() :])
        element_index += 1
    suffix = "\n" if markdown.endswith("\n") else ""
    return "\n".join(output) + suffix


def build_delta_result(
    current_md: str,
    next_md: str,
    position_jitter_threshold: float = POSITION_JITTER_THRESHOLD,
) -> DeltaBuildResult:
    """Build a deterministic delta and a next target with position-only jitter removed."""
    if position_jitter_threshold < 0 or not math.isfinite(position_jitter_threshold):
        raise ValueError("position jitter threshold must be finite and non-negative")
    current = parse_md_elements(current_md)
    nxt = parse_md_elements(next_md)
    matched_current: set[int] = set()
    matched_next: set[int] = set()
    for left, right in _exact_pairs(current, nxt):
        matched_current.add(left)
        matched_next.add(right)

    jitter_pairs = _jitter_pairs(
        current,
        nxt,
        (index for index in range(len(current)) if index not in matched_current),
        (index for index in range(len(nxt)) if index not in matched_next),
        threshold=position_jitter_threshold,
    )
    removals = []
    replacements: dict[int, str] = {}
    for left, right in jitter_pairs:
        matched_current.add(left)
        matched_next.add(right)
        dx = abs(current[left].cx - nxt[right].cx)
        dy = abs(current[left].cy - nxt[right].cy)
        removals.append(
            PositionJitterRemoval(
                current_index=left,
                next_index=right,
                current_line=current[left].line,
                next_line=nxt[right].line,
                dx=dx,
                dy=dy,
                distance=max(dx, dy),
            )
        )
        replacements[right] = current[left].line

    updates: list[tuple[int, int]] = []
    removes: list[int] = []
    for left in range(len(current)):
        if left in matched_current:
            continue
        candidates = [
            (element_distance(current[left], nxt[right]), right)
            for right in range(len(nxt))
            if right not in matched_next
            and current[left].element_type == nxt[right].element_type
            and math.hypot(current[left].cx - nxt[right].cx, current[left].cy - nxt[right].cy)
            <= UPDATE_CENTER_LIMIT
        ]
        if candidates:
            distance, right = min(candidates, key=lambda item: (item[0], item[1]))
            if distance <= UPDATE_DISTANCE_LIMIT:
                updates.append((left, right))
                matched_current.add(left)
                matched_next.add(right)
                continue
        removes.append(left)
    adds = [index for index in range(len(nxt)) if index not in matched_next]

    operations = [DeltaOperation("REMOVE", old=current[index]) for index in removes]
    operations.extend(DeltaOperation("ADD", new=nxt[index]) for index in adds)
    operations.extend(DeltaOperation("UPDATE", old=current[left], new=nxt[right]) for left, right in updates)
    parsed = ParsedDelta(tuple(operations), no_visible_change=not operations)
    return DeltaBuildResult(
        delta=parsed.serialize(),
        target_next_md_clean=_replace_target_elements(next_md, replacements),
        position_jitter_removed=tuple(removals),
    )


def build_delta(
    current_md: str,
    next_md: str,
    position_jitter_threshold: float = POSITION_JITTER_THRESHOLD,
) -> str:
    return build_delta_result(current_md, next_md, position_jitter_threshold).delta


def _parse_element(text: str) -> MdElement:
    stripped = text.strip()
    elements = parse_md_elements(stripped)
    if len(elements) != 1 or elements[0].line != stripped:
        raise DeltaFormatError(f"invalid element syntax: {text!r}")
    element = elements[0]
    if element.element_type == "block":
        match = BLOCK_PAYLOAD_RE.fullmatch(element_payload(element))
        if match is None or match.group("label") not in BLOCK_LABELS:
            raise DeltaFormatError(f"invalid block element: {text!r}")
    elif element.element_type not in LEAF_TYPES:
        raise DeltaFormatError(f"unsupported element type: {element.element_type!r}")
    if not (0 <= element.cx <= 999 and 0 <= element.cy <= 999):
        raise DeltaFormatError(f"element center is outside 0..999: {text!r}")
    if not (0 <= element.w <= 999 and 0 <= element.h <= 999):
        raise DeltaFormatError(f"element size is outside 0..999: {text!r}")
    return element


def _split_update(text: str) -> tuple[MdElement, MdElement]:
    candidates = []
    marker = " => "
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        try:
            left = _parse_element(text[:index])
            right = _parse_element(text[index + len(marker) :])
        except DeltaFormatError:
            pass
        else:
            candidates.append((left, right))
        start = index + len(marker)
    if len(candidates) != 1:
        raise DeltaFormatError(f"UPDATE must contain one unambiguous old => new pair: {text!r}")
    return candidates[0]


def parse_delta(text: object) -> ParsedDelta:
    """Strictly parse a delta document without discarding order or duplicates."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines or lines[0] != DELTA_HEADER:
        raise DeltaFormatError(f"delta must start with {DELTA_HEADER}")
    body = lines[1:]
    if not body:
        raise DeltaFormatError("delta body is empty")
    if NO_VISIBLE_CHANGE in body:
        if body != [NO_VISIBLE_CHANGE]:
            raise DeltaFormatError("NO_VISIBLE_CHANGE must be the only body line")
        return ParsedDelta((), no_visible_change=True)

    operations = []
    previous_rank = -1
    for line in body:
        if ": " not in line:
            raise DeltaFormatError(f"unknown delta operation: {line!r}")
        kind, payload = line.split(": ", 1)
        if kind not in OPERATION_ORDER:
            raise DeltaFormatError(f"unknown delta operation: {kind!r}")
        rank = OPERATION_ORDER[kind]
        if rank < previous_rank:
            raise DeltaFormatError("delta operations must be ordered REMOVE, ADD, UPDATE")
        previous_rank = rank
        if kind == "REMOVE":
            operations.append(DeltaOperation(kind, old=_parse_element(payload)))
        elif kind == "ADD":
            operations.append(DeltaOperation(kind, new=_parse_element(payload)))
        else:
            old, new = _split_update(payload)
            operations.append(DeltaOperation(kind, old=old, new=new))
    return ParsedDelta(tuple(operations))


def _environment_line(markdown: str) -> str:
    return next(
        (line.strip() for line in markdown.splitlines() if line.strip().startswith("[Environment:")),
        "[Environment: Web]",
    )


def canonicalize_flat_md(markdown: str) -> str:
    """Serialize a Markdown state as a deterministic top-left ordered flat element list."""
    elements = parse_md_elements(markdown)
    elements.sort(
        key=lambda element: (
            element.cy - element.h / 2,
            element.cx - element.w / 2,
            0 if element.element_type == "block" else 1,
            element.cy + element.h / 2,
            element_payload(element),
            element.line,
        )
    )
    lines = [_environment_line(markdown), ""]
    lines.extend(f"- {element.line}" for element in elements)
    return "\n".join(lines).rstrip() + "\n"


def element_line_counter(markdown: str) -> Counter[str]:
    return Counter(element.line for element in parse_md_elements(markdown))


def apply_delta_flat(current_md: str, delta: str | ParsedDelta) -> str:
    """Apply a strict delta as a flat multiset patch; invalid patches raise instead of copying current."""
    if not any(
        line.strip().startswith("[Environment:")
        for line in str(current_md or "").splitlines()
    ):
        raise DeltaReplayError("current_md must contain an [Environment: ...] header")
    parsed = delta if isinstance(delta, ParsedDelta) else parse_delta(delta)
    if parsed.no_visible_change:
        return canonicalize_flat_md(current_md)
    current = element_line_counter(current_md)
    consumed: Counter[str] = Counter()
    additions: Counter[str] = Counter()
    for operation in parsed.operations:
        if operation.kind == "REMOVE" and operation.old is not None:
            consumed[operation.old.line] += 1
        elif operation.kind == "ADD" and operation.new is not None:
            additions[operation.new.line] += 1
        elif operation.kind == "UPDATE" and operation.old is not None and operation.new is not None:
            consumed[operation.old.line] += 1
            additions[operation.new.line] += 1
    missing = {line: count - current.get(line, 0) for line, count in consumed.items() if count > current.get(line, 0)}
    if missing:
        raise DeltaReplayError(f"delta consumes unavailable current elements: {missing}")
    result = current - consumed
    result.update(additions)
    lines = [_environment_line(current_md), ""]
    for line, count in result.items():
        lines.extend(f"- {line}" for _ in range(count))
    return canonicalize_flat_md("\n".join(lines))
