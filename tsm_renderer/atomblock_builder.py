"""AtomBlock-WebUI Builder: YOLO v4 + OCR + VL → GUINode tree (Remote Service Version)"""

import os
import sys
import json
import base64
import concurrent.futures
import re
import threading
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from common import GUINode

import requests
from PIL import Image

DEFAULT_YOLO_URL = "http://localhost:8001"
DEFAULT_OCR_URL = "http://localhost:8002"
DEFAULT_QWEN_URL = "http://localhost:8000"
DEFAULT_QWEN_MODEL = os.environ.get(
    "MD_RENDER_QWEN_MODEL",
    "qwen08b",
)
_HTTP_LOCAL = threading.local()


def _http_session() -> requests.Session:
    """Return one proxy-free requests session per calling thread."""
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _HTTP_LOCAL.session = session
    return session


def _service_root(url: str) -> str:
    root = url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_pil_to_base64(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pad_extreme_vl_crop(image_crop: Image.Image, max_ratio: float = 190.0) -> Image.Image:
    """Pad very thin UI crops below the Qwen-VL processor aspect-ratio limit."""
    width, height = image_crop.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid VL crop size: {image_crop.size}")
    if max(width / height, height / width) <= max_ratio:
        return image_crop
    if width > height:
        target_width = width
        target_height = max(height, int(np.ceil(width / max_ratio)))
    else:
        target_height = height
        target_width = max(width, int(np.ceil(height / max_ratio)))
    padded = Image.new("RGB", (target_width, target_height), "white")
    rgb_crop = image_crop.convert("RGB")
    try:
        padded.paste(
            rgb_crop,
            ((target_width - width) // 2, (target_height - height) // 2),
        )
    finally:
        if rgb_crop is not image_crop:
            rgb_crop.close()
    return padded


YOLO_V4_CLASS_NAMES = [
    'button', 'link', 'input', 'image', 'icon', 'checkbox', 'radio', 'select',
    'block-nav', 'block-sidebar', 'block-footer', 'block-form', 'block-table', 'block'
]

BLOCK_CLASSES = {'block-nav', 'block-sidebar', 'block-footer', 'block-form', 'block-table', 'block'}
ATOMIC_CLASSES = {'button', 'link', 'input', 'image', 'icon', 'checkbox', 'radio', 'select'}

SEMANTIC_PRIORITY = {
    'block-nav': 0, 'block-sidebar': 0, 'block-footer': 0,
    'block-form': 1, 'block-table': 1, 'block': 2,
}


def _filter_code_false_positives(detections: List[Dict], img_h: int) -> List[Dict]:
    ""
    if img_h <= 0:
        return detections
    filtered = []
    for d in detections:
        tag = d['class_name']
        x1, y1, x2, y2 = d['bbox']
        box_w = x2 - x1
        box_h = y2 - y1
        if tag in ('link', 'button') and box_h < 25 and box_w > 300 and y1 > img_h * 0.5:
            continue
        if tag == 'button' and box_h < 20 and box_w < 50 and y1 > img_h * 0.5:
            continue
        filtered.append(d)
    return filtered


def run_yolo_v4(
    image_path: str,
    model_path: str,
    conf: float = 0.1,
    iou: float = 0.1,
    imgsz: int = 1920,
    agnostic_nms: bool = True,
    yolo_url: str = DEFAULT_YOLO_URL,
) -> List[Dict]:
    ""
    payload = {
        "image_path": image_path,
        "model_path": model_path,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "agnostic_nms": agnostic_nms,
    }
    r = _http_session().post(f"{yolo_url.rstrip('/')}/detect", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    detections = data.get("detections", [])
    for d in detections:
        d["bbox"] = tuple(d["bbox"])
    return detections


def run_ocr(image_path: str, ocr_url: str = DEFAULT_OCR_URL) -> List[Dict]:
    ""
    payload = {"image_path": image_path}
    r = _http_session().post(f"{ocr_url.rstrip('/')}/ocr", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    texts = data.get("texts", [])
    for t in texts:
        if t.get("bbox"):
            t["bbox"] = tuple(t["bbox"])
    return texts


def merge_ocr_texts(ocr_results: List[Dict]) -> List[Dict]:
    ""
    if not ocr_results:
        return []

    items = [
        {"text": r["text"], "bbox": r["bbox"], "confidence": r.get("confidence", 0.0)}
        for r in ocr_results
    ]
    n = len(items)

    def _should_merge(a, b):
        ""
        x1a, y1a, x2a, y2a = a["bbox"]
        x1b, y1b, x2b, y2b = b["bbox"]
        wa, ha = x2a - x1a, y2a - y1a
        wb, hb = x2b - x1b, y2b - y1b
        cya, cyb = (y1a + y2a) / 2, (y1b + y2b) / 2
        cxa, cxb = (x1a + x2a) / 2, (x1b + x2b) / 2

        h_avg = (ha + hb) / 2
        y_diff = abs(cya - cyb)
        x_gap = max(0, x1b - x2a) if x1b > x2a else max(0, x1a - x2b)

        if y_diff < h_avg * 0.4 and x_gap < h_avg * 0.5 and x_gap < min(wa, wb) * 0.15:
            return True

        y_gap = max(0, y1b - y2a) if y1b > y2a else max(0, y1a - y2b)
        x_overlap = max(0.0, min(x2a, x2b) - max(x1a, x1b))
        min_w = min(wa, wb)
        max_w = max(wa, wb)
        left_aligned = abs(x1a - x1b) <= h_avg * 0.35
        width_similar = max_w > 0 and min_w / max_w >= 0.65
        both_sentence_like = (
            len(a["text"].strip()) >= 12
            and len(b["text"].strip()) >= 12
            and bool(re.search(r'[，。,.!?！？]$', a["text"].strip()))
        )
        if (
            y_gap < h_avg * 0.45
            and x_overlap > min_w * 0.82
            and min_w > 24
            and left_aligned
            and width_similar
            and both_sentence_like
        ):
            return True

        return False

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge(items[i], items[j]):
                union(i, j)

    groups: Dict[int, List[Dict]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(items[i])

    def _try_merge_subgroup(subgroup: List[Dict]) -> Optional[Dict]:
        ""
        if len(subgroup) == 1:
            return subgroup[0]

        total_chars = sum(len(r["text"]) for r in subgroup)
        avg_chars = total_chars / len(subgroup) if subgroup else 0
        char_limit = 150 if avg_chars >= 15 else 80
        if len(subgroup) > 6 or total_chars > char_limit:
            return None

        all_x = [r["bbox"][0] for r in subgroup] + [r["bbox"][2] for r in subgroup]
        all_y = [r["bbox"][1] for r in subgroup] + [r["bbox"][3] for r in subgroup]
        merged_bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        merged_w = merged_bbox[2] - merged_bbox[0]
        merged_h = merged_bbox[3] - merged_bbox[1]
        if len(subgroup) > 3 and merged_h > merged_w * 1.2:
            return None

        subgroup.sort(key=lambda r: (bbox_center(r["bbox"])[1], bbox_center(r["bbox"])[0]))
        texts = [r["text"] for r in subgroup]
        merged_text = texts[0]
        for i in range(1, len(texts)):
            prev, curr = texts[i - 1], texts[i]
            sep = "" if (bool(re.search(r"[\u4e00-\u9fff]", prev)) or bool(re.search(r"[\u4e00-\u9fff]", curr))) else " "
            merged_text = merged_text + sep + curr

        return {
            "text": merged_text,
            "bbox": merged_bbox,
            "confidence": max(r.get("confidence", 0.0) for r in subgroup),
        }

    merged: List[Dict] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        group.sort(key=lambda r: bbox_center(r["bbox"])[1])

        subgroups: List[List[Dict]] = [[group[0]]]
        for i in range(1, len(group)):
            prev = group[i - 1]
            curr = group[i]
            _, y1a, _, y2a = prev["bbox"]
            _, y1b, _, y2b = curr["bbox"]
            y_gap = max(0, y1b - y2a) if y1b > y2a else max(0, y1a - y2b)
            h_avg = ((y2a - y1a) + (y2b - y1b)) / 2
            if y_gap > h_avg * 0.55:
                subgroups.append([curr])
            else:
                subgroups[-1].append(curr)

        for sg in subgroups:
            result = _try_merge_subgroup(sg)
            if result:
                merged.append(result)
            else:
                for r in sg:
                    merged.append(r)

    return sorted(merged, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def _clean_vl_response(content: str) -> str:
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    if "</thinking>" in content:
        content = content.split("</thinking>")[-1].strip()
    content = re.sub(r"</?(?:think|thinking)>", "", content).strip()
    content = content.strip().strip("`")
    lines = [line.strip(" -*\t") for line in content.splitlines() if line.strip()]
    if not lines:
        return ""
    content = " ".join(lines)
    content = re.sub(r"\s+", " ", content).strip()
    content = re.sub(r"^The user wants[^.]*\.\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^\d+\.\s*", "", content)
    content = re.sub(r"^.*?\*\*[^*]+:\*\*\s*", "", content)
    content = re.split(r"\s+\d+\.\s+", content, maxsplit=1)[0].strip()
    content = re.sub(
        r"^(the image|this image|the icon|this icon|it)\s+(shows|contains|is|appears to be|looks like)\s+",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"^it'?s\s+", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s+\d+\.$", "", content).strip()
    for prefix in (
        "final answer:",
        "answer:",
        "label:",
        "description:",
    ):
        if content.lower().startswith(prefix):
            content = content[len(prefix):].strip()
    return content[:160].strip()


def describe_image_crop(image_crop, prompt: str = "Describe this UI element in one short sentence.", qwen_url: str = DEFAULT_QWEN_URL) -> str:
    ""
    safe_crop = _pad_extreme_vl_crop(image_crop)
    try:
        b64_img = _encode_pil_to_base64(safe_crop)
    finally:
        if safe_crop is not image_crop:
            safe_crop.close()
    instruction = (
        f"{prompt}\n"
        "Return only the requested label/value. Do not explain. Do not mention the user, "
        "the image, or your reasoning. Do not use bullets or numbering."
    )
    payload = {
        "model": DEFAULT_QWEN_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You label small UI crops for a web automation agent. Output concise final labels only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        "max_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error = None
    for attempt in range(1, 4):
        try:
            r = _http_session().post(
                f"{_service_root(qwen_url)}/v1/chat/completions",
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Qwen VL returned empty content")

            return _clean_vl_response(content)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"Qwen VL request failed after 3 attempts: {last_error}") from last_error


def get_area(bbox: Tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def get_intersection(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    return ix * iy


def center_in_bbox(center: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    return bbox[0] <= center[0] <= bbox[2] and bbox[1] <= center[1] <= bbox[3]


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def sort_nodes_reading_order(nodes: List[GUINode]) -> List[GUINode]:
    ""
    if not nodes:
        return nodes
    sorted_nodes = sorted(
        nodes,
        key=lambda n: (n.bbox[1], (n.bbox[0] + n.bbox[2]) / 2) if n.bbox else (float('inf'), float('inf'))
    )
    rows: List[List[GUINode]] = []
    current_row: List[GUINode] = [sorted_nodes[0]]
    for node in sorted_nodes[1:]:
        prev = current_row[-1]
        if node.bbox and prev.bbox:
            y_diff = abs(node.bbox[1] - prev.bbox[1])
            cy_diff = abs(((node.bbox[1] + node.bbox[3]) / 2) - ((prev.bbox[1] + prev.bbox[3]) / 2))
            h_avg = ((prev.bbox[3] - prev.bbox[1]) + (node.bbox[3] - node.bbox[1])) / 2.0
            h_avg = max(h_avg, 1e-6)
            if y_diff < h_avg * 0.6 and cy_diff < h_avg * 1.0:
                current_row.append(node)
                continue
        rows.append(current_row)
        current_row = [node]
    if current_row:
        rows.append(current_row)
    result: List[GUINode] = []
    for row in rows:
        row.sort(key=lambda n: ((n.bbox[0] + n.bbox[2]) / 2) if n.bbox else float('inf'))
        result.extend(row)
    return result


def _filter_garbage_ocr(ocr_results: List[Dict], img_w: int, img_h: int) -> List[Dict]:
    ""
    filtered = []
    for r in ocr_results:
        x1, y1, x2, y2 = r['bbox']
        w, h = x2 - x1, y2 - y1
        text = r.get('text', '')
        if w > img_w * 0.85 and h < img_h * 0.05:
            if len(text) > 30 and ' ' not in text:
                continue
        filtered.append(r)
    return filtered


def _is_low_quality_text(text: str, target_tag: str) -> bool:
    ""
    text = text.strip()
    if not text:
        return True
    if target_tag in ('image', 'icon'):
        if len(text) <= 1 and text.isascii() and text.isdigit():
            return True
        return False
    if text.isdigit() and len(text) <= 3:
        return True
    if target_tag in ('link', 'button', 'input', 'select'):
        if len(text) <= 1 and not ('\u4e00' <= text <= '\u9fff'):
            return True
    return False


def secondary_nms(detections: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    ""
    by_class: Dict[str, List[Dict]] = {}
    for d in detections:
        by_class.setdefault(d['class_name'], []).append(d)

    kept = []
    for cls_name, dets in by_class.items():
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        while dets:
            base = dets.pop(0)
            kept.append(base)
            filtered = []
            b1 = base['bbox']
            a1 = get_area(b1)
            for d in dets:
                b2 = d['bbox']
                inter = get_intersection(b1, b2)
                a2 = get_area(b2)
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0
                if iou < iou_thresh:
                    filtered.append(d)
            dets = filtered
    return kept


def _collect_vl_nodes(nodes: List[GUINode]) -> List[GUINode]:
    ""
    targets = []
    for n in nodes:
        if n.tag in ("image", "icon", "checkbox", "radio", "select"):
            targets.append(n)
        elif n.tag in ATOMIC_CLASSES and not n.text:
            targets.append(n)
        targets.extend(_collect_vl_nodes(n.children))
    return targets


VL_PROMPTS = {
    "image": (
        "Identify the main visible object, product, or scene in this UI crop. "
        "For e-commerce or fashion crops, output the product type plus visible color, pattern, or material when clear, "
        "for example: grey striped t-shirt, red floral dress, black leather shoes. "
        "For landscape, background, or travel photos, output the visible scene or setting, "
        "for example: tropical beach with boats, city skyline, forest trail, airplane cabin. "
        "Do not read surrounding page text, prices, ratings, or navigation unless text is the main content of the crop. "
        "Return a concise noun phrase of 2-8 words. "
        "Use [INVALID] only when the crop is blank, a placeholder, or truly unrecognizable."
    ),
    "icon": (
        "Identify the visible UI icon or action in 1-4 words, such as search, favorite heart, cart, menu, close, "
        "arrow, plus, minus, account, filter, or dropdown. "
        "If the symbol is visible but generic, output the most likely UI action. "
        "Use [INVALID] only when no symbol is visible or the crop is truly unrecognizable."
    ),
    "button": (
        "Read visible text on this button. If there is no text, identify the icon/action in 1-4 words. "
        "Use [INVALID] only when the button is blank or truly unrecognizable."
    ),
    "link": (
        "Read visible text on this link. If there is no text, identify the linked object or action in 1-6 words. "
        "Use [INVALID] only when the link is blank or truly unrecognizable."
    ),
    "input": (
        "Read the visible placeholder, label, or current value of this input. "
        "If there is no readable text, identify the input purpose in 1-5 words when clear. "
        "Use [INVALID] only when the input is blank or truly unrecognizable."
    ),
    "select": (
        "This crop contains a select/dropdown field and the area below it. "
        "Decide whether the dropdown is visibly open by inspecting the menu/options that belong to this field. "
        "A dropdown menu is often the main UI region below the field: if the text below the field belongs to this menu "
        "and forms a list of options for this select, "
        "mark expanded: yes even when the menu is not drawn as a floating overlay. "
        "Use expanded: yes when options/menu items are visible below, aligned with, or visually grouped under this field. "
        "Use expanded: yes only when the visible below-text is a grouped option list for this exact select, not a single isolated value. "
        "The option labels must be plausible values for this exact field; options for a neighboring select do not count. "
        "A menu option must be an item inside this dropdown menu, not text inside another input, button, tab, date field, "
        "location field, traveler popup, or form row below. For example, if the select text is Economy, locations like "
        "New York or London are not valid options for it. "
        "If the only below text belongs to another form control or page content, use expanded: no and options: none. "
        "Do not mark expanded just because normal page content, product cards, filters, checkboxes, or form fields appear below. "
        "If only the closed field and its down arrow are visible, if the below text is not this field's menu, or if uncertain, use expanded: no. "
        "Read the current selected value or placeholder from the field. "
        "If expanded: yes, list 2-3 visible option labels for this field whenever possible. "
        "If no plausible option labels are visible for this field, use expanded: no and options: none. "
        "Output exactly: expanded: yes/no | text: <value or none> | options: <option labels or none>"
    ),
    "checkbox": (
        "Decide the checkbox/toggle state by inspecting only the small square control near the left side of the crop. "
        "Use checked: yes only when that control itself contains a visible check mark, tick, solid fill, or selected dot. "
        "Use checked: no when the control is an empty outlined square/circle, off, unselected, or uncertain. "
        "Do not infer checked from the label text, row hover color, nearby product image, count, or page context. "
        "Read only the label on the same horizontal row as the control; ignore labels above or below. "
        "Output exactly: checked: yes/no | label: <label or none>"
    ),
    "radio": (
        "Decide the radio button state by inspecting only the small circle control near the left side of the crop. "
        "Use checked: yes only when that control itself contains a filled dot or selected mark. "
        "Use checked: no when the control is an empty outlined circle, unselected, or uncertain. "
        "Do not infer checked from the label text, row hover color, nearby product image, count, or page context. "
        "Read only the label on the same horizontal row as the control; ignore labels above or below. "
        "Output exactly: checked: yes/no | label: <label or none>"
    ),
}


_ABSENT_VL_LABELS = {
    "",
    "none",
    "none visible",
    "no text",
    "no visible text",
    "not visible",
    "nothing visible",
    "n/a",
    "na",
    "null",
    "invalid",
    "[invalid]",
}


def _clean_vl_label(text: str) -> str:
    """Normalize Qwen placeholder answers that mean no visible label."""
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text).strip(" .,:;|`'\"")
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if normalized in _ABSENT_VL_LABELS:
        return ""
    if normalized.startswith(("no visible ", "not visible ", "nothing visible")):
        return ""
    return text


def _parse_checkbox_vl(desc: str) -> Tuple[bool, str]:
    ""
    checked = False
    label = ""
    desc = desc.strip()
    m_checked = re.search(r'checked:\s*(yes|no|true|false|1|0)', desc, re.IGNORECASE)
    if m_checked:
        checked = m_checked.group(1).lower() in ('yes', 'true', '1')
    m_label = re.search(r'label:\s*(.+)', desc, re.IGNORECASE)
    if m_label:
        label = m_label.group(1).strip()
        label = re.split(r'\||\n', label)[0].strip()
        label = _clean_vl_label(label)
    return checked, label


def _parse_select_vl(desc: str) -> Tuple[bool, str, str]:
    ""
    expanded = False
    text = ""
    options = ""
    desc = desc.strip()
    m_expanded = re.search(r'expanded:\s*(yes|no|true|false|1|0)', desc, re.IGNORECASE)
    if m_expanded:
        expanded = m_expanded.group(1).lower() in ('yes', 'true', '1')
    m_text = re.search(r'text:\s*(.+)', desc, re.IGNORECASE)
    if m_text:
        text = m_text.group(1).strip()
        text = re.split(r'\||\n', text)[0].strip()
        text = _clean_vl_label(text)
    m_options = re.search(r'options:\s*(.+)', desc, re.IGNORECASE)
    if m_options:
        options = m_options.group(1).strip()
        options = re.split(r'\||\n', options)[0].strip()
        options = _clean_vl_label(options)
    normalized_options = re.sub(r'[\s,;.-]+', ' ', options.lower()).strip()
    if expanded and (
        not normalized_options
        or normalized_options in {'none', 'n a', 'na', 'no', 'not visible', 'no visible options'}
        or not _select_options_plausible(text, options)
    ):
        expanded = False
    return expanded, text, options


def _split_option_labels(options: str) -> List[str]:
    ""
    options = options.strip()
    if not options:
        return []
    parts = re.split(r'\s*(?:,|;|/|、|\||\band\b)\s*', options, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _normalize_option_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\u4e00-\u9fff¥$€£]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _option_label_matches_ocr(label: str, ocr_text: str) -> bool:
    label_norm = _normalize_option_text(label)
    ocr_norm = _normalize_option_text(ocr_text)
    if not label_norm or not ocr_norm:
        return False
    if len(label_norm) >= 3 and (label_norm in ocr_norm or ocr_norm in label_norm):
        return True
    label_tokens = {t for t in label_norm.split() if len(t) >= 3 or re.search(r'\d', t)}
    ocr_tokens = {t for t in ocr_norm.split() if len(t) >= 3 or re.search(r'\d', t)}
    if not label_tokens or not ocr_tokens:
        return False
    return bool(label_tokens & ocr_tokens)


def _select_options_aligned_with_node(
    node: GUINode,
    ocr_results: List[Dict],
    options: str,
) -> bool:
    """Check that reported options are visibly below and aligned with this select.

    The 4h crop can include neighboring menus or lower form controls. Qwen may
    correctly read their text but attach it to the wrong select, so geometry is
    used as a high-precision guard for `[v]`.
    """
    if not node.bbox:
        return False
    labels = _split_option_labels(options)
    if not labels:
        return False

    x1, y1, x2, y2 = map(float, node.bbox)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    allowed_x1 = x1 - max(8.0, box_w * 0.25)
    allowed_x2 = x2 + max(8.0, box_w * 0.50)
    min_y = y2 + max(2.0, box_h * 0.05)
    max_y = y1 + box_h * 4.05

    for ocr in ocr_results:
        text = (ocr.get("text") or "").strip()
        bbox = ocr.get("bbox")
        if not text or not bbox:
            continue
        ox1, oy1, ox2, oy2 = map(float, bbox)
        ocx = (ox1 + ox2) / 2.0
        ocy = (oy1 + oy2) / 2.0
        if x1 <= ocx <= x2 and y1 <= ocy <= y2:
            continue
        if ocy < min_y or ocy > max_y:
            continue
        if ocx < allowed_x1 or ocx > allowed_x2:
            continue
        if any(_option_label_matches_ocr(label, text) for label in labels):
            return True

    return False


def _select_options_plausible(field_text: str, options: str) -> bool:
    ""
    labels = _split_option_labels(options)
    if len(labels) < 2:
        return False

    field = field_text.lower()
    field_compact = re.sub(r'\s+', ' ', field).strip()
    joined = " ".join(label.lower() for label in labels)
    field_tokens = {
        token
        for token in re.split(r'[^a-z0-9\u4e00-\u9fff]+', field)
        if len(token) >= 2
    }
    option_tokens = {
        token
        for label in labels
        for token in re.split(r'[^a-z0-9\u4e00-\u9fff]+', label.lower())
        if len(token) >= 2
    }

    if all(re.fullmatch(r'\d{1,2}', label) for label in labels):
        return False
    if all(re.fullmatch(r'\d+', label) for label in labels):
        return False

    generic_numeric_options = field_compact in {"no min", "no max", "any"} and (
        bool(re.search(r'[\d¥$€£]', options.lower()))
        or any(marker in options.lower() for marker in ("sqft", "acre", "mo", "month", "+"))
    )
    if option_tokens and field_tokens and option_tokens.issubset(field_tokens) and not generic_numeric_options:
        return False

    if re.fullmatch(r'\d+\s*(?:-|to)\s*\d+', field_compact) or re.fullmatch(r'\d+\+', field_compact):
        return False
    if "/" in field_compact or "\u4e3b\u9875" in field_compact or "\u9996\u9875" in field_compact:
        return False
    if "age" in field and re.search(r'\b(?:ord|mdw|rfd|jfk|lax|sfo|mad|syd)\b', joined):
        return False
    if field_compact in {"no min", "no max", "any"}:
        options_lower = options.lower()
        has_numeric_option = bool(re.search(r'[\d¥$€£]', options_lower)) or any(
            marker in options_lower for marker in ("sqft", "acre", "mo", "month", "+")
        )
        if not has_numeric_option:
            return False

    if any(marker in field for marker in ("room", "guest", "traveler", "traveller", "adult", "child", "people", "person", "\u4eba")):
        room_terms = (
            "room", "guest", "adult", "child", "infant", "traveler", "traveller",
            "people", "person", "persons", "\u4eba", "\u6210\u4eba", "\u513f\u7ae5",
        )
        if not any(term in joined for term in room_terms):
            return False

    if any(marker in field for marker in ("return to", "same location", "different location", "palaa samaan")):
        return_location_terms = ("same", "different", "location", "return", "palaa", "sijainti")
        if not any(term in joined for term in return_location_terms):
            return False

    price_field_markers = ("price", "budget", "cost", "fare", "rate")
    if any(marker in field for marker in price_field_markers):
        price_terms = (
            "$", "¥", "€", "£", "usd", "eur", "gbp", "rmb",
            "min", "max", "minimum", "maximum", "under", "over", "any price",
            "price", "budget", "rate", "fare", "cheap", "expensive",
        )
        options_lower = options.lower()
        if not any(term in joined or term in options_lower for term in price_terms):
            return False

    if any(marker in field for marker in ("bed", "bath")):
        beds_baths_terms = (
            "bed", "beds", "bedroom", "bedrooms", "bath", "baths", "bathroom",
            "bathrooms", "studio", "any", "1+", "2+", "3+", "4+", "5+",
        )
        if not any(term in joined for term in beds_baths_terms):
            return False

    if any(marker in field for marker in ("home type", "property type", "type")):
        home_type_terms = (
            "house", "home", "apartment", "condo", "townhouse", "villa",
            "land", "lot", "manufactured", "multi-family", "multifamily",
            "single family", "property",
        )
        if not any(term in joined for term in home_type_terms):
            return False

    if any(marker in field for marker in ("amenit", "facility", "\u8bbe\u65bd")):
        amenity_field_terms = (
            "pool", "wifi", "wi-fi", "breakfast", "parking", "gym", "spa",
            "restaurant", "bar", "kitchen", "washer", "dryer", "air conditioning",
            "pet", "accessible", "elevator", "outdoor", "waterfront", "fireplace",
        )
        if not any(term in joined for term in amenity_field_terms):
            return False

    if any(marker in field for marker in ("date", "month", "year")):
        date_terms = (
            "date", "day", "month", "year", "jan", "feb", "mar", "apr",
            "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            "january", "february", "march", "april", "june", "july",
            "august", "september", "october", "november", "december",
        )
        if not (re.search(r'\d', joined) or any(term in joined for term in date_terms)):
            return False

    if any(marker in field for marker in ("multi-city", "multicity", "one-way", "one way", "round trip", "return")):
        trip_terms = ("multi-city", "multicity", "one-way", "one way", "round trip", "return")
        if not any(term in joined for term in trip_terms):
            return False

    job_card_terms = (
        "specialist", "agent", "summary", "responsible", "relations",
        "operations", "full-time", "part-time", "learn more", "handling",
    )
    if any(term in joined for term in job_card_terms):
        return False

    source_chip_terms = (
        "kayak", "booking", "booking.com", "expedia", "hotels", "agoda",
        "priceline", "ihg", "super.com",
    )
    if any(term in field for term in source_chip_terms):
        return False

    filter_control_terms = ("all filters", "filters", "price", "rating", "category")
    if any(term in joined for term in filter_control_terms) and not any(
        marker in field for marker in ("sort", "order", "price", "budget", "cost", "fare", "rate")
    ):
        return False

    if (
        re.search(r'[¥$€£]\s*\d', options)
        and not any(marker in field for marker in ("sort", "order", "price", "budget", "cost", "fare", "rate"))
        and field_compact not in {"no min", "no max", "any"}
    ):
        return False

    city_terms = {
        "new york", "london", "paris", "tokyo", "beijing", "shanghai",
        "los angeles", "san francisco", "chicago", "dublin", "madrid",
        "salem", "boise", "idaho falls", "rapid city", "medford",
        "twin falls", "salt lake", "cheyenne", "carson city", "denver",
        "las vegas", "bakersfield", "flagstaff", "santa fe", "amarillo",
        "phoenix", "tucson", "lubbock", "san antonio",
        "sydney", "melbourne", "brisbane", "perth", "guangzhou",
        "tel aviv", "dubai",
    }
    state_terms = {
        "washington", "oregon", "idaho", "montana", "wyoming", "nevada",
        "utah", "colorado", "california", "arizona", "new mexico",
        "texas", "south dakota", "north dakota", "nebraska",
    }
    city_like = any(term in joined for term in city_terms)
    map_like = city_like or any(term in joined for term in state_terms)
    city_field_markers = (
        "destination", "city", "airport", "station", "leaving", "going",
        "pickup", "dropoff", "origin", "location", "address",
    )
    if map_like and not any(marker in field for marker in city_field_markers):
        return False

    if city_like and any(marker in field for marker in ("airport", "station")):
        return False

    address_or_card_terms = (
        "apartment", "apartments", "rental", "rentals", "lincoln", "street",
        "avenue", "road", "fuller", "results",
        "dreamcatcher", "stone", "framed", "canvas", "modern",
        "mi from", "booking.com", "expedia", "hotels.com", "agoda.com",
        "bank owned", "listing price", "sqft", "images coming soon",
        "kurta", "dress", "shirt", "dupatta", "embroidered", "cotton",
        "printed", "sleeve", "self design", "limited stock",
        "drone", "camera", "remote control", "obstacle", "gps", "quadcopter",
        "folding", "beginner",
    )
    if any(term in joined for term in address_or_card_terms):
        return False

    if "fabric elasticity" in field:
        elasticity_terms = ("stretch", "elastic", "slight", "medium", "high", "micro", "non-stretch", "no stretch")
        if not any(term in joined for term in elasticity_terms):
            return False

    # Temu-like product filter pills such as "Collar Style" often sit above a side filter panel.
    # The 4h crop below the pill contains product cards, so Qwen may hallucinate neck/collar options.
    if "collar style" in field:
        return False
    if "collar" in field:
        collar_terms = ("collar", "neck", "crew", "v-neck", "round", "polo", "hood", "lapel")
        if not any(term in joined for term in collar_terms):
            return False

    amenity_terms = (
        "basement", "washer", "dryer", "hookup", "pets allowed", "guest house",
        "fireplace", "waterfront", "air conditioning", "elevator", "garage",
        "parking", "accessible", "gym", "restaurant", "outdoor",
    )
    if any(term in joined for term in amenity_terms) and not any(
        marker in field for marker in ("amenit", "feature", "facility", "\u8bbe\u65bd")
    ):
        return False

    if any(marker in field for marker in ("economy", "cabin", "class")):
        cabin_terms = ("economy", "premium", "business", "first", "coach", "standard")
        if not any(term in joined for term in cabin_terms):
            return False

    return True


def _clamp_crop_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    ""
    x1, y1, x2, y2 = box
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))
    return x1, y1, x2, y2


def _make_vl_crop(img: Image.Image, node: GUINode) -> Image.Image:
    ""
    img_w, img_h = img.size
    x1, y1, x2, y2 = map(int, node.bbox)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    if node.tag == "select":
        pad_x = max(6, int(box_w * 0.08))
        pad_top = max(2, int(box_h * 0.10))
        crop_box = (
            x1 - pad_x,
            y1 - pad_top,
            x2 + pad_x,
            y1 + box_h * 4,
        )
    elif node.tag in ("checkbox", "radio"):
        pad_left = max(10, int(box_w * 0.8))
        pad_right = max(180, int(box_w * 16.0), int(box_h * 12.0))
        pad_y = max(5, int(box_h * 0.45))
        crop_box = (
            x1 - pad_left,
            y1 - pad_y,
            x2 + pad_right,
            y2 + pad_y,
        )
    else:
        crop_box = (x1, y1, x2, y2)

    return img.crop(_clamp_crop_box(crop_box, img_w, img_h))


def _infer_control_checked_from_pixels(img: Image.Image, node: GUINode) -> Optional[bool]:
    ""
    if not node.bbox:
        return None
    img_w, img_h = img.size
    x1, y1, x2, y2 = _clamp_crop_box(tuple(map(int, node.bbox)), img_w, img_h)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 6 or box_h < 6:
        return False

    crop = img.crop((x1, y1, x2, y2)).convert("RGB")
    arr = np.asarray(crop).astype(np.float32)
    if arr.size == 0:
        return None

    gray = arr.mean(axis=2)
    h, w = gray.shape
    ix1 = max(0, int(w * 0.25))
    ix2 = max(ix1 + 1, int(w * 0.75))
    iy1 = max(0, int(h * 0.25))
    iy2 = max(iy1 + 1, int(h * 0.75))
    inner = gray[iy1:iy2, ix1:ix2]
    if inner.size == 0:
        return None

    max_rgb = arr.max(axis=2)
    min_rgb = arr.min(axis=2)
    saturation = (max_rgb - min_rgb)[iy1:iy2, ix1:ix2]

    inner_mean = float(inner.mean())
    inner_std = float(inner.std())
    dark_ratio = float((inner < 120).mean())
    mid_ratio = float((inner < 220).mean())
    saturated_ratio = float((saturation > 40).mean())

    has_inner_mark = (
        dark_ratio >= 0.08
        or (mid_ratio >= 0.22 and inner_std >= 30)
        or (saturated_ratio >= 0.12 and inner_mean < 245)
    )
    if has_inner_mark:
        return True

    clearly_empty = (
        inner_mean >= 245
        and dark_ratio < 0.02
        and saturated_ratio < 0.05
        and mid_ratio < 0.08
    )
    if clearly_empty:
        return False

    return None


def build_atomblock_tree(
    image_path: str,
    model_path: str,
    conf: float = 0.1,
    iou: float = 0.1,
    imgsz: int = 1920,
    agnostic_nms: bool = True,
    use_ocr: bool = True,
    use_vl: bool = False,
    device: int = 0,
    secondary_nms_thresh: float = 0.3,
    yolo_url: str = DEFAULT_YOLO_URL,
    ocr_url: str = DEFAULT_OCR_URL,
    qwen_url: str = DEFAULT_QWEN_URL,
) -> GUINode:
    ""
    # Detach pixel data from the source file immediately so failures later in the
    # remote pipeline cannot leak an open image file descriptor.
    with Image.open(image_path) as source_img:
        img = source_img.convert("RGB")
    w, h = img.size

    detections = run_yolo_v4(image_path, model_path, conf, iou, imgsz, agnostic_nms, yolo_url=yolo_url)
    detections = [d for d in detections if d['confidence'] >= conf]
    detections = _filter_code_false_positives(detections, h)
    detections = secondary_nms(detections, iou_thresh=secondary_nms_thresh)

    blocks = [d for d in detections if d['class_name'] in BLOCK_CLASSES]
    atoms = [d for d in detections if d['class_name'] in ATOMIC_CLASSES]

    block_nodes: List[GUINode] = []
    for b in blocks:
        node = GUINode(
            node_id=f"block_{b['class_id']}_{id(b)}",
            tag=b['class_name'],
            bbox=b['bbox'],
            attributes={'confidence': b['confidence']},
            is_structural_block=True,
        )
        block_nodes.append(node)

    block_nodes.sort(key=lambda n: get_area(n.bbox), reverse=True)

    root_blocks: List[GUINode] = []
    for node in block_nodes:
        placed = False
        candidates = []
        for parent in root_blocks:
            if center_in_bbox(bbox_center(node.bbox), parent.bbox):
                inter = get_intersection(node.bbox, parent.bbox)
                node_area = get_area(node.bbox)
                if node_area > 0 and inter / node_area > 0.85:
                    parent_prio = SEMANTIC_PRIORITY.get(parent.tag, 2)
                    node_prio = SEMANTIC_PRIORITY.get(node.tag, 2)
                    if parent_prio <= node_prio:
                        candidates.append(parent)
        if candidates:
            parent = min(candidates, key=lambda p: get_area(p.bbox))
            parent.children.append(node)
            placed = True
        if not placed:
            root_blocks.append(node)

    unassigned_atoms: List[GUINode] = []
    for a in atoms:
        node = GUINode(
            node_id=f"atom_{a['class_id']}_{id(a)}",
            tag=a['class_name'],
            bbox=a['bbox'],
            attributes={'confidence': a['confidence']},
        )
        center = bbox_center(a['bbox'])
        def find_best_block(blocks_list: List[GUINode], atom_bbox) -> Optional[GUINode]:
            candidates = []
            for blk in blocks_list:
                if center_in_bbox(bbox_center(atom_bbox), blk.bbox):
                    candidates.append(blk)
                    child = find_best_block(blk.children, atom_bbox)
                    if child:
                        candidates.append(child)
            if not candidates:
                for blk in blocks_list:
                    inter = get_intersection(atom_bbox, blk.bbox)
                    atom_area = get_area(atom_bbox)
                    if atom_area > 0 and inter / atom_area > 0.5:
                        candidates.append(blk)
                        child = find_best_block(blk.children, atom_bbox)
                        if child:
                            candidates.append(child)
            if not candidates:
                return None
            def priority(b: GUINode) -> Tuple[int, float]:
                semantic_rank = SEMANTIC_PRIORITY.get(b.tag, 2)
                return (semantic_rank, get_area(b.bbox))
            return min(candidates, key=priority)

        best = find_best_block(root_blocks, a['bbox'])
        if best:
            best.children.append(node)
        else:
            unassigned_atoms.append(node)

    if use_ocr:
        ocr_results = run_ocr(image_path, ocr_url=ocr_url)
        ocr_results = merge_ocr_texts(ocr_results)
        ocr_results = _filter_garbage_ocr(ocr_results, w, h)
    else:
        ocr_results = []

    def assign_ocr_to_tree(ocr_text: str, ocr_bbox: Optional[Tuple[float, float, float, float]]):
        center = bbox_center(ocr_bbox) if ocr_bbox else None

        def _collect_all_atoms(blocks_list: List[GUINode]) -> List[GUINode]:
            atoms = []
            for blk in blocks_list:
                for child in blk.children:
                    if not child.is_structural_block:
                        atoms.append(child)
                atoms.extend(_collect_all_atoms(blk.children))
            return atoms

        all_atoms = _collect_all_atoms(root_blocks) + [
            a for a in unassigned_atoms if not a.is_structural_block
        ]
        ocr_claim_atoms = [
            a for a in all_atoms
            if a.tag not in ('image', 'icon')
        ]

        def _atom_priority(a: GUINode) -> int:
            if a.tag in ('input', 'select', 'button', 'link', 'checkbox', 'radio'):
                return 0
            if a.tag == 'text':
                return 1
            return 2

        atom = None
        if center is not None:
            if ocr_bbox:
                link_candidates = []
                ocr_area = get_area(ocr_bbox)
                ox1, oy1, ox2, oy2 = ocr_bbox
                for a in ocr_claim_atoms:
                    if a.tag != 'link' or not a.bbox:
                        continue
                    ax1, ay1, ax2, ay2 = a.bbox
                    inside_link = (
                        ox1 >= ax1 - 2
                        and oy1 >= ay1 - 2
                        and ox2 <= ax2 + 2
                        and oy2 <= ay2 + 2
                    )
                    inter = get_intersection(ocr_bbox, a.bbox)
                    mostly_inside_link = (
                        center_in_bbox(center, a.bbox)
                        and ocr_area > 0
                        and inter / ocr_area >= 0.90
                    )
                    if inside_link or mostly_inside_link:
                        link_candidates.append(a)
                if link_candidates:
                    atom = min(link_candidates, key=lambda a: get_area(a.bbox))

            candidates = [a for a in ocr_claim_atoms if center_in_bbox(center, a.bbox)]
            if atom is None and candidates:
                atom = min(candidates, key=lambda a: (_atom_priority(a), get_area(a.bbox)))
            if atom is None:
                best_atom = None
                best_inter = 0.0
                for a in ocr_claim_atoms:
                    inter = get_intersection(ocr_bbox, a.bbox) if a.bbox else 0.0
                    if inter > best_inter:
                        best_inter = inter
                        best_atom = a
                if best_atom and best_inter > 0:
                    atom_area = get_area(best_atom.bbox)
                    ocr_area = get_area(ocr_bbox) if ocr_bbox else 0.0
                    atom_ratio = best_inter / atom_area if atom_area > 0 else 0.0
                    ocr_ratio = best_inter / ocr_area if ocr_area > 0 else 0.0
                    if atom_ratio >= 0.3 and ocr_ratio >= 0.3:
                        atom = best_atom

            if atom is None and ocr_bbox:
                best_score = 0.0
                best_atom2 = None
                for a in ocr_claim_atoms:
                    if a.tag not in ('button', 'link', 'input', 'select', 'checkbox', 'radio'):
                        continue
                    if not a.bbox:
                        continue
                    ax1, ay1, ax2, ay2 = a.bbox
                    ox1, oy1, ox2, oy2 = ocr_bbox
                    x_overlap = max(0.0, min(ax2, ox2) - max(ax1, ox1))
                    y_overlap = max(0.0, min(ay2, oy2) - max(ay1, oy1))
                    if x_overlap > 0 and y_overlap > 0:
                        a_area = get_area(a.bbox)
                        o_area = get_area(ocr_bbox)
                        if o_area > a_area * 3:
                            continue
                    min_w = min(ax2 - ax1, ox2 - ox1)
                    y_gap = max(0, oy1 - ay2) if oy1 > ay2 else max(0, ay1 - oy2)
                    if min_w <= 0:
                        continue
                    atom_h = ay2 - ay1
                    ocr_h = oy2 - oy1
                    max_h = max(atom_h, ocr_h)
                    atom_w = ax2 - ax1
                    ocr_w = ox2 - ox1
                    min_h = min(atom_h, ocr_h)
                    y_overlap_ratio = y_overlap / min_h if min_h > 0 else 0.0
                    if a.tag in ('button', 'link', 'input', 'select') and y_overlap_ratio < 0.35:
                        compact_text = re.sub(r'[\s\W_]+', '', ocr_text, flags=re.UNICODE)
                        short_centered_text = (
                            0 < len(compact_text) <= 12
                            and y_gap <= max(12.0, max_h * 0.40)
                            and x_overlap / min_w > 0.75
                            and ocr_w <= atom_w * 1.25
                        )
                        if not short_centered_text:
                            continue
                    if x_overlap / min_w > 0.3 and y_gap < max_h * 1.5 + 5 and ocr_w < atom_w * 4:
                        score = x_overlap / min_w
                        if score > best_score:
                            best_score = score
                            best_atom2 = a
                if best_atom2:
                    atom = best_atom2

        if atom:
            if _is_low_quality_text(ocr_text, atom.tag):
                if atom.tag in ('link', 'button', 'input', 'select'):
                    return
                if atom.tag in ('image', 'icon'):
                    return

            if atom.tag in ('image', 'icon') and len(ocr_text.strip()) > 30:
                return

            if not atom.text and ocr_bbox:
                covered_same = []
                for a in all_atoms:
                    if a.tag == atom.tag and not a.text and a.bbox:
                        if center_in_bbox(bbox_center(a.bbox), ocr_bbox):
                            covered_same.append(a)
                        else:
                            inter = get_intersection(a.bbox, ocr_bbox)
                            a_area = get_area(a.bbox)
                            if a_area > 0 and inter / a_area > 0.3:
                                covered_same.append(a)
                seen_ids = set()
                uniq = []
                for a in covered_same:
                    if a.node_id not in seen_ids:
                        seen_ids.add(a.node_id)
                        uniq.append(a)
                covered_same = uniq

                if len(covered_same) > 1:
                    text_clean = ocr_text.strip()
                    parts = [p for p in re.split(r'[\s　]+', text_clean) if p]
                    if len(parts) < len(covered_same):
                        extended = []
                        for p in parts:
                            m = re.match(r'^(\d+)(.+)$', p)
                            if m and len(m.group(2)) > 1:
                                extended.append(m.group(1))
                                extended.append(m.group(2))
                            else:
                                extended.append(p)
                        parts = extended

                    if len(parts) >= len(covered_same):
                        covered_same.sort(key=lambda a: a.bbox[0])
                        for i, a in enumerate(covered_same):
                            part = parts[i] if i < len(parts) else ""
                            if _is_low_quality_text(part, a.tag):
                                continue
                            a.text = part
                        return
            if atom.text and len(atom.text) >= 5:
                if _is_low_quality_text(ocr_text, atom.tag):
                    return
            if atom.text:
                has_cjk = bool(re.search(r'[\u4e00-\u9fff]', atom.text + ocr_text))
                sep = "" if has_cjk else " "
                merged_len = len(atom.text) + len(sep) + len(ocr_text)
                if atom.tag != 'text' and merged_len > 60:
                    return
                if atom.bbox and ocr_bbox:
                    atom_y_center = (atom.bbox[1] + atom.bbox[3]) / 2
                    ocr_y_center = (ocr_bbox[1] + ocr_bbox[3]) / 2
                    y_diff = abs(atom_y_center - ocr_y_center)
                    atom_h = atom.bbox[3] - atom.bbox[1]
                    if y_diff > atom_h * 0.8:
                        return
                atom.text = (atom.text + sep + ocr_text).strip()
            else:
                atom.text = ocr_text
            return

        all_blocks = []
        def _collect_blocks(blocks_list: List[GUINode]):
            for blk in blocks_list:
                all_blocks.append(blk)
                _collect_blocks(blk.children)
        _collect_blocks(root_blocks)

        blk = None
        if center is not None:
            candidates = [b for b in all_blocks if center_in_bbox(center, b.bbox)]
            if candidates:
                blk = min(candidates, key=lambda b: get_area(b.bbox))
            else:
                best_blk = None
                best_inter = 0.0
                for b in all_blocks:
                    inter = get_intersection(ocr_bbox, b.bbox) if b.bbox else 0.0
                    if inter > best_inter:
                        best_inter = inter
                        best_blk = b
                if best_blk and best_inter > 0:
                    blk = best_blk

                if blk is None:
                    y_candidates = []
                    for b in all_blocks:
                        if b.bbox and b.bbox[1] <= center[1] <= b.bbox[3]:
                            y_candidates.append(b)
                    if y_candidates:
                        blk = min(y_candidates, key=lambda b: abs((b.bbox[0] + b.bbox[2]) / 2 - center[0]))

        if blk:
            text_node = GUINode(
                node_id=f"text_{id(ocr_text)}",
                tag='text',
                text=ocr_text,
                bbox=ocr_bbox,
            )
            blk.children.append(text_node)
            return

        text_node = GUINode(
            node_id=f"text_top_{id(ocr_text)}",
            tag='text',
            text=ocr_text,
            bbox=ocr_bbox,
        )
        unassigned_atoms.append(text_node)

    for ocr in ocr_results:
        assign_ocr_to_tree(ocr['text'], ocr.get('bbox'))

    if use_vl:
        vl_targets = _collect_vl_nodes(root_blocks) + _collect_vl_nodes(unassigned_atoms)
        print(f"   [VL] \u53d1\u73b0 {len(vl_targets)} \u4e2a\u5f85\u5206\u6790\u5143\u7d20")
        vl_inputs = []
        for idx_vl, node in enumerate(vl_targets, 1):
            try:
                crop = _make_vl_crop(img, node)
                prompt = VL_PROMPTS.get(node.tag, "Describe this UI element in one short sentence.")
                vl_inputs.append((idx_vl, node, crop, prompt))
            except Exception as exc:
                vl_inputs.append((idx_vl, node, None, exc))

        def run_vl_request(item):
            idx_vl, node, crop, prompt_or_error = item
            if crop is None:
                return idx_vl, node, "", prompt_or_error
            try:
                desc = describe_image_crop(crop, prompt=prompt_or_error, qwen_url=qwen_url)
                return idx_vl, node, desc, None
            except Exception as exc:
                return idx_vl, node, "", exc
            finally:
                crop.close()

        try:
            max_vl_workers = int(os.environ.get("MD_RENDER_VL_WORKERS", "8"))
        except ValueError as exc:
            raise ValueError("MD_RENDER_VL_WORKERS must be an integer in [1, 16]") from exc
        if not 1 <= max_vl_workers <= 16:
            raise ValueError("MD_RENDER_VL_WORKERS must be in [1, 16]")
        if len(vl_inputs) <= 1 or max_vl_workers == 1:
            vl_results = [run_vl_request(item) for item in vl_inputs]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(max_vl_workers, len(vl_inputs))
            ) as executor:
                vl_results = list(executor.map(run_vl_request, vl_inputs))

        vl_failures = []
        for idx_vl, node, desc, vl_error in vl_results:
            try:
                if vl_error is not None:
                    raise vl_error
                if desc:
                    if node.tag in ("checkbox", "radio"):
                        checked, label = _parse_checkbox_vl(desc)
                        visual_checked = _infer_control_checked_from_pixels(img, node)
                        node.checked = visual_checked if visual_checked is not None else checked
                        if label:
                            node.text = label
                        print(
                            f"      [{idx_vl}/{len(vl_targets)}] {node.tag}: "
                            f"checked={node.checked} vl_checked={checked} visual_checked={visual_checked} label='{label}'"
                        )
                    elif node.tag == "select":
                        expanded, text, options = _parse_select_vl(desc)
                        if expanded and not _select_options_aligned_with_node(node, ocr_results, options):
                            expanded = False
                        node.expanded = expanded
                        if text:
                            node.text = text
                        print(
                            f"      [{idx_vl}/{len(vl_targets)}] {node.tag}: "
                            f"expanded={expanded} text='{text}' options='{options}'"
                        )
                    else:
                        desc = _clean_vl_label(desc)
                        if desc:
                            node.text = desc
                        print(f"      [{idx_vl}/{len(vl_targets)}] {node.tag}: {desc[:60] if desc else '(empty)'}...")
                else:
                    print(f"      [{idx_vl}/{len(vl_targets)}] {node.tag}: (empty)")
            except Exception as e:
                print(f"      [{idx_vl}/{len(vl_targets)}] {node.tag} VL error: {e}")
                vl_failures.append((idx_vl, node.tag, str(e)))

        # Never publish a partially annotated image. Individual calls already
        # retry; any remaining request failure makes this image retryable.
        if vl_failures:
            img.close()
            sample = "; ".join(
                f"#{idx}:{tag}:{message}" for idx, tag, message in vl_failures[:3]
            )
            raise RuntimeError(
                f"VL annotation failed for {len(vl_failures)}/{len(vl_targets)} elements: {sample}"
            )

    top_nodes = root_blocks + unassigned_atoms
    top_nodes = sort_nodes_reading_order(top_nodes)

    def sort_children(node: GUINode):
        node.children = sort_nodes_reading_order(node.children)
        for c in node.children:
            sort_children(c)

    for node in top_nodes:
        sort_children(node)

    dummy_root = GUINode(node_id='root', tag='root', is_structural_block=True)
    dummy_root.children = top_nodes
    dummy_root.attributes['width'] = w
    dummy_root.attributes['height'] = h
    img.close()
    return dummy_root
