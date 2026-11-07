"""AtomBlock-WebUI Markdown Renderer"""

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Set, Tuple
from common import GUINode, normalize_bbox


BLOCK_NAME_MAP = {
    'block-nav': 'Navigation',
    'block-sidebar': 'Sidebar',
    'block-footer': 'Footer',
    'block-form': 'Form',
    'block-table': 'Table',
    'block': 'Section',
}


@dataclass(frozen=True)
class _RenderEntry:
    node: GUINode
    tag: str
    text: str
    base_str: str
    loc_str: str
    bbox: Optional[Tuple[float, float, float, float]]
    indent: int
    order: int


def _escape_brackets(text: str) -> str:
    return text.replace('[', ' ').replace(']', ' ').replace('\n', ' ').replace('\r', ' ').strip()


def _is_noise_text(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    if not re.search(r'[a-zA-Z0-9\u4e00-\u9fff]', text):
        return True
    if len(text) == 1:
        return True
    return False


def _render_fields(node: GUINode, w: int, h: int) -> Optional[Tuple[str, str, str]]:
    """Return (base markdown, normalized visible text, loc) or None if hidden."""
    tag = node.tag
    raw_text = node.text or ""
    if "[INVALID]" in raw_text or "<INVALID>" in raw_text:
        if tag not in ('icon', 'button', 'image'):
            return None
        raw_text = ""
    lower = raw_text.lower().strip(". \t")
    if lower in ("no text visible", "none visible"):
        if tag not in ('icon', 'button', 'image'):
            return None
        raw_text = ""

    text = _escape_brackets(raw_text)
    if text == "INVALID":
        if tag not in ('icon', 'button', 'image'):
            return None
        text = ""
    if not node.is_structural_block and _is_noise_text(text):
        if tag not in ('icon', 'button', 'image'):
            return None
        text = ""

    loc_str = normalize_bbox(node.bbox, w, h)
    if node.is_structural_block:
        semantic = BLOCK_NAME_MAP.get(tag, tag.replace('block-', '').capitalize())
        return f"[Block: {semantic}]", "", loc_str
    if tag == 'button':
        return (f"[{text}](button)" if text else "[button](button)", text, loc_str)
    if tag == 'link':
        if not text:
            return None
        return f"[{text}](link)", text, loc_str
    if tag == 'input':
        if not text:
            return None
        return f"[{text}](input)", text, loc_str
    if tag == 'image':
        return f"[img: {text or ''}](image)", text, loc_str
    if tag == 'icon':
        return f"[icon: {text or ''}](icon)", text, loc_str
    if tag in ('checkbox', 'radio'):
        checked_mark = "[x]" if node.checked else "[ ]"
        if text:
            return f"{checked_mark} [{text}](toggle)", text, loc_str
        return f"{checked_mark}(toggle)", text, loc_str
    if tag == 'select':
        if not text:
            return None
        arrow = "[v]" if node.expanded else "[>]"
        return f"{arrow} [{text}](select)", text, loc_str
    if tag == 'text':
        if not text:
            return None
        return f"[{text}](text)", text, loc_str
    if not text:
        return None
    return f"[{text}](text)", text, loc_str


def _has_renderable_child(node: GUINode, w: int, h: int) -> bool:
    for child in node.children:
        if child.is_structural_block:
            if _has_renderable_child(child, w, h):
                return True
            continue
        if _render_fields(child, w, h) is not None:
            return True
    return False


def _collect_entries(
    node: GUINode,
    w: int,
    h: int,
    indent: int,
    entries: List[_RenderEntry],
) -> None:
    fields = _render_fields(node, w, h)
    if fields is not None:
        base_str, text, loc_str = fields
        entries.append(
            _RenderEntry(
                node=node,
                tag=node.tag,
                text=text,
                base_str=base_str,
                loc_str=loc_str,
                bbox=node.bbox,
                indent=indent,
                order=len(entries),
            )
        )
    for child in node.children:
        _collect_entries(child, w, h, indent + 1, entries)


def _claim_priority(tag: str) -> int:
    if tag in ('input', 'select', 'button', 'link', 'checkbox', 'radio'):
        return 0
    if tag in ('image', 'icon'):
        return 2
    if tag == 'text':
        return 4
    return 3


def _bbox_area(bbox: Optional[Tuple[float, float, float, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(
    a: Optional[Tuple[float, float, float, float]],
    b: Optional[Tuple[float, float, float, float]],
) -> float:
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _horizontal_gap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0.0


def _same_visual_row(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ah = max(1.0, a[3] - a[1])
    bh = max(1.0, b[3] - b[1])
    y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    min_h = min(ah, bh)
    acy = (a[1] + a[3]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return y_overlap / min_h >= 0.35 or abs(acy - bcy) <= max(ah, bh) * 0.75


def _same_visual_row_relaxed(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ah = max(1.0, a[3] - a[1])
    bh = max(1.0, b[3] - b[1])
    y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    min_h = min(ah, bh)
    acy = (a[1] + a[3]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return y_overlap / min_h >= 0.15 or abs(acy - bcy) <= max(ah, bh) * 1.45


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _vertical_gap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def _text_inside_claim(text_entry: _RenderEntry, claim: _RenderEntry) -> bool:
    """Return True when a plain OCR text is visually inside an owning control."""
    if text_entry.tag != 'text' or not text_entry.bbox or not claim.bbox:
        return False
    if claim.tag not in ('input', 'select', 'button', 'link', 'checkbox', 'radio'):
        return False

    tb = text_entry.bbox
    cb = claim.bbox
    text_area = _bbox_area(tb)
    if text_area <= 0:
        return False

    inside_with_tolerance = (
        tb[0] >= cb[0] - 2
        and tb[1] >= cb[1] - 2
        and tb[2] <= cb[2] + 2
        and tb[3] <= cb[3] + 2
    )
    inter = max(0.0, min(tb[2], cb[2]) - max(tb[0], cb[0])) * max(0.0, min(tb[3], cb[3]) - max(tb[1], cb[1]))
    mostly_inside = inter / text_area >= 0.85
    center_inside = _same_visual_row_relaxed(tb, cb) and (
        cb[0] - 2 <= (tb[0] + tb[2]) / 2.0 <= cb[2] + 2
        and cb[1] - 2 <= (tb[1] + tb[3]) / 2.0 <= cb[3] + 2
    )
    return inside_with_tolerance or (mostly_inside and center_inside)


def _nearby_label_claim_related(a: _RenderEntry, b: _RenderEntry) -> bool:
    """Relate a plain OCR text to the control/image that visually owns it.

    Detector boxes for checkbox/radio/icon/image often cover only the control or
    crop, while the OCR text covers the adjacent label. Exact-IoU matching misses
    these duplicates, so use a conservative nearby-label relation only when one
    side is plain text and the other side is a higher-priority visual/control.
    """
    if not a.bbox or not b.bbox:
        return False
    if a.tag == b.tag:
        return False
    if a.tag == 'text':
        text_entry, claim = a, b
    elif b.tag == 'text':
        text_entry, claim = b, a
    else:
        return False
    if claim.tag not in ('input', 'select', 'button', 'link', 'checkbox', 'radio', 'image', 'icon'):
        return False

    tb = text_entry.bbox
    cb = claim.bbox
    tw = max(1.0, tb[2] - tb[0])
    th = max(1.0, tb[3] - tb[1])
    cw = max(1.0, cb[2] - cb[0])
    ch = max(1.0, cb[3] - cb[1])

    if _same_visual_row_relaxed(tb, cb):
        gap = _horizontal_gap(tb, cb)
        # Allow a small control box to own a label on the same row.
        limit = max(48.0, min(220.0, tw * 1.8), ((th + ch) / 2.0) * 8.0)
        if gap <= limit:
            return True

    x_overlap = _overlap_1d(tb[0], tb[2], cb[0], cb[2])
    min_w = min(tw, cw)
    txc = (tb[0] + tb[2]) / 2.0
    cxc = (cb[0] + cb[2]) / 2.0
    y_gap = _vertical_gap(tb, cb)
    # Some image-like controls carry text immediately above/below the crop.
    if (
        claim.tag in ('image', 'button', 'icon')
        and y_gap <= max(36.0, min(th, ch) * 2.5)
        and (x_overlap >= min_w * 0.35 or abs(txc - cxc) <= max(tw, cw) * 0.35)
    ):
        return True

    return False


def _claim_boxes_related(a: _RenderEntry, b: _RenderEntry, relaxed: bool = False) -> bool:
    if not a.bbox or not b.bbox:
        return False
    if _bbox_iou(a.bbox, b.bbox) >= (0.20 if relaxed else 0.35):
        return True
    if _nearby_label_claim_related(a, b):
        return True
    if not _same_visual_row(a.bbox, b.bbox):
        return False
    aw = max(1.0, a.bbox[2] - a.bbox[0])
    bw = max(1.0, b.bbox[2] - b.bbox[0])
    ah = max(1.0, a.bbox[3] - a.bbox[1])
    bh = max(1.0, b.bbox[3] - b.bbox[1])
    gap = _horizontal_gap(a.bbox, b.bbox)
    limit = max(24.0, min(aw, bw) * (1.2 if relaxed else 0.8), ((ah + bh) / 2.0) * 3.0)
    return gap <= limit


def _normalize_claim_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _compact_claim_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', '', text)
    return text


def _better_claim(a: _RenderEntry, b: _RenderEntry) -> _RenderEntry:
    pa = _claim_priority(a.tag)
    pb = _claim_priority(b.tag)
    if pa != pb:
        return a if pa < pb else b
    # Same semantic class: duplicate detector boxes should usually keep the
    # fuller clickable region, while duplicate plain text keeps the earliest box.
    if a.tag != 'text':
        aa = _bbox_area(a.bbox)
        ba = _bbox_area(b.bbox)
        if abs(aa - ba) > 1e-6:
            return a if aa > ba else b
    return a if a.order <= b.order else b


def _suppressed_node_ids(entries: Iterable[_RenderEntry]) -> Set[int]:
    non_blocks = [e for e in entries if not e.node.is_structural_block]
    suppressed: Set[int] = set()

    for i, left in enumerate(non_blocks):
        for right in non_blocks[i + 1:]:
            if id(left.node) in suppressed or id(right.node) in suppressed:
                continue
            left_norm = _normalize_claim_text(left.text)
            right_norm = _normalize_claim_text(right.text)
            same_text = left_norm and left_norm == right_norm
            same_empty_kind = not left_norm and not right_norm and left.tag == right.tag
            if not same_text and not same_empty_kind:
                continue
            if same_empty_kind and _bbox_iou(left.bbox, right.bbox) < 0.60:
                continue
            if same_text and not _claim_boxes_related(left, right):
                continue
            winner = _better_claim(left, right)
            loser = right if winner is left else left
            suppressed.add(id(loser.node))

    for entry in non_blocks:
        if id(entry.node) in suppressed or entry.tag != 'text':
            continue
        compact = _compact_claim_text(entry.text)
        if len(compact) >= 2:
            for other in non_blocks:
                if other is entry or id(other.node) in suppressed:
                    continue
                if _claim_priority(other.tag) >= _claim_priority(entry.tag) or not other.text:
                    continue
                if other.tag not in ('input', 'select', 'button', 'link', 'checkbox', 'radio'):
                    continue
                other_compact = _compact_claim_text(other.text)
                if len(other_compact) < len(compact) or compact not in other_compact:
                    continue
                if _text_inside_claim(entry, other) or (
                    len(compact) >= 4 and _claim_boxes_related(entry, other, relaxed=True)
                ):
                    suppressed.add(id(entry.node))
                    break
        if id(entry.node) in suppressed:
            continue
        if len(compact) < 4:
            continue
        matching_claims = []
        for other in non_blocks:
            if other is entry or id(other.node) in suppressed:
                continue
            if _claim_priority(other.tag) >= _claim_priority(entry.tag) or not other.text:
                continue
            other_compact = _compact_claim_text(other.text)
            if len(other_compact) < 2 or other_compact not in compact:
                continue
            if _claim_boxes_related(entry, other, relaxed=True):
                matching_claims.append(other_compact)
        unique_claims = set(matching_claims)
        claimed_chars = sum(len(text) for text in unique_claims)
        if len(unique_claims) >= 2 and claimed_chars >= max(4, int(len(compact) * 0.50)):
            suppressed.add(id(entry.node))

    return suppressed


def _render_node_pruned(
    node: GUINode,
    w: int,
    h: int,
    indent: int,
    suppressed_ids: Set[int],
    seen_set: Set[str],
) -> List[str]:
    fields = _render_fields(node, w, h)
    if fields is None:
        return []

    if node.is_structural_block:
        child_lines: List[str] = []
        for child in node.children:
            child_lines.extend(_render_node_pruned(child, w, h, indent + 1, suppressed_ids, seen_set))
        if not child_lines:
            return []
        base_str, _, loc_str = fields
        prefix = "  " * indent + "- "
        hash_key = f"{base_str}{loc_str}{len(child_lines)}"
        result = []
        if hash_key not in seen_set:
            result.append(f"{prefix}{base_str}{loc_str}")
            seen_set.add(hash_key)
        result.extend(child_lines)
        return result

    if id(node) in suppressed_ids:
        return []
    base_str, _, loc_str = fields
    prefix = "  " * indent + "- "
    hash_key = f"{base_str}{loc_str}"
    if hash_key in seen_set:
        return []
    seen_set.add(hash_key)
    return [f"{prefix}{base_str}{loc_str}"]


def render_nodes(nodes: Iterable[GUINode], w: int, h: int, indent: int = 0, seen_set: Set[str] = None) -> List[str]:
    """Render a whole sibling list with global text-claim de-duplication."""
    if seen_set is None:
        seen_set = set()
    node_list = list(nodes)
    entries: List[_RenderEntry] = []
    for node in node_list:
        _collect_entries(node, w, h, indent, entries)
    suppressed_ids = _suppressed_node_ids(entries)

    result: List[str] = []
    for node in node_list:
        result.extend(_render_node_pruned(node, w, h, indent, suppressed_ids, seen_set))
    return result


def render_node(node: GUINode, w: int, h: int, indent: int = 0, seen_set: Set[str] = None) -> List[str]:
    return render_nodes([node], w, h, indent=indent, seen_set=seen_set)
