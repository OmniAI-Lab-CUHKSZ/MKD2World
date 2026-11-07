"""Runtime chat message builders."""

from __future__ import annotations

import base64
import json


def _balanced_json_value(text: str, start: int) -> tuple[object, int] | None:
    """Decode one JSON value beginning at start."""
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None


def _repair_split_selected_action(text: str) -> dict | None:
    """Recover GUI-Owl responses that split selected_action across fragments."""
    if any(text.count(key) != 1 for key in ('"selected_action"', '"unified_action"', '"action_intent"')):
        return None
    marker = '"unified_action"'
    marker_pos = text.find(marker)
    if marker_pos < 0:
        return None
    colon = text.find(":", marker_pos + len(marker))
    if colon < 0:
        return None
    action_start = colon + 1
    while action_start < len(text) and text[action_start].isspace():
        action_start += 1
    decoded = _balanced_json_value(text, action_start)
    if decoded is None or not isinstance(decoded[0], dict):
        return None
    unified_action = decoded[0]
    intent_marker = '"action_intent"'
    intent_pos = text.find(intent_marker, action_start + decoded[1])
    if intent_pos < 0:
        return None
    intent_colon = text.find(":", intent_pos + len(intent_marker))
    if intent_colon < 0:
        return None
    intent_start = intent_colon + 1
    while intent_start < len(text) and text[intent_start].isspace():
        intent_start += 1
    intent = _balanced_json_value(text, intent_start)
    if intent is None or not isinstance(intent[0], str):
        return None
    return {"selected_action": {"unified_action": unified_action, "action_intent": intent[0]}}


def parse_json_object(raw: str, *, stage: str) -> dict:
    """Parse JSON, with narrow repairs for observed model wrappers."""
    text = raw.strip()
    for suffix in (("<|im_end|>", "<|endoftext|>") if stage in {"direct_action", "best_action"} else ()):
        while text.endswith(suffix):
            text = text[:-len(suffix)].rstrip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        data = _repair_split_selected_action(text) if stage == "direct_action" else None
        if data is None:
            snippet = raw[:300].replace("\n", "\\n")
            raise ValueError(f"{stage} returned invalid JSON: {exc}. raw_prefix={snippet!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{stage} must return a JSON object, got {type(data).__name__}")
    return data

def image_block_from_screenshot(screenshot: bytes | None) -> dict | None:
    if not screenshot:
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(screenshot).decode("ascii"),
        },
    }


def json_messages(system_prompt: str, payload: dict) -> list[dict]:
    payload = dict(payload)
    screenshot = payload.pop("_screenshot", None)
    image_items = payload.pop("_images", None)
    labeled_images: list[tuple[str, bytes]] = []
    if screenshot:
        labeled_images.append(("current_screenshot", screenshot))
    if isinstance(image_items, list):
        for index, item in enumerate(image_items):
            if isinstance(item, dict) and item.get("data"):
                labeled_images.append((str(item.get("label") or f"image_{index}"), item["data"]))
    if labeled_images and image_items:
        payload["current_screenshot"] = {
            "type": "attached_images",
            "labels": [label for label, _ in labeled_images],
        }
    elif labeled_images:
        payload["current_screenshot"] = {"type": "attached_image", "mime_type": "image/png"}
    user_text = json.dumps(payload, ensure_ascii=False, indent=2)
    user_content: str | list[dict] = user_text
    if labeled_images:
        user_content = [{"type": "text", "text": user_text}]
        for label, data in labeled_images:
            block = image_block_from_screenshot(data)
            if block is not None:
                if image_items:
                    user_content.append({"type": "text", "text": f"Attached image: {label}"})
                user_content.append(block)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
