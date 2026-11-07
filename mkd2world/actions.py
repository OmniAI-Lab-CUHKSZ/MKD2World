"""Unified GUI action contract.

Coordinates use the project-wide 0..999 relative grid defined by prompt templates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence


GRID_MIN = 0
GRID_MAX = 999
SCROLL_GRID_MIN = -999
SCROLL_GRID_MAX = 999
LEGACY_SCROLL_AMOUNT_TO_DELTA = {
    "small": 300.0,
    "medium": 700.0,
    "large": 999.0,
}


class ActionType(StrEnum):
    LEFT_CLICK = "left_click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    SCROLL = "scroll"
    KEY_PRESS = "key_press"
    HOTKEY = "hotkey"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    NAVIGATE = "navigate"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    REFRESH = "refresh"
    SELECT_OPTION = "select_option"
    TOGGLE_CHECK = "toggle_check"
    DRAG = "drag"
    HOVER = "hover"
    MOVE_TO = "move_to"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    WAIT = "wait"
    NEW_TAB = "new_tab"
    PAGE_CLOSE = "page_close"
    PAGE_FOCUS = "page_focus"
    CLEAR = "clear"
    UPLOAD = "upload"
    FINISH = "finish"
    FAIL = "fail"


COORDINATE_KEYS = {
    "coordinate",
    "from_coordinate",
    "to_coordinate",
}


def _validate_point(value: object, *, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item coordinate")
    x, y = value
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError(f"{field_name} must contain integers")
    if not (GRID_MIN <= x <= GRID_MAX and GRID_MIN <= y <= GRID_MAX):
        raise ValueError(f"{field_name} must be within {GRID_MIN}..{GRID_MAX}")
    return x, y


def _validate_text(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _validate_key_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{field_name} must be a non-empty sequence of keys")
    keys = tuple(str(item) for item in value)
    if any(not key for key in keys):
        raise ValueError(f"{field_name} must not contain empty keys")
    return keys


def _numeric_scroll_delta(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    if not SCROLL_GRID_MIN <= number <= SCROLL_GRID_MAX:
        raise ValueError(f"{field_name} must be within {SCROLL_GRID_MIN}..{SCROLL_GRID_MAX}")
    return 0.0 if number == 0.0 else number


def normalize_scroll_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical numeric scroll deltas; accept the legacy enum schema for migration."""

    payload = dict(params)
    has_delta = "delta_x" in payload or "delta_y" in payload
    if has_delta:
        if "delta_x" not in payload or "delta_y" not in payload:
            raise ValueError("scroll requires both delta_x and delta_y")
        delta_x = _numeric_scroll_delta(payload["delta_x"], field_name="delta_x")
        delta_y = _numeric_scroll_delta(payload["delta_y"], field_name="delta_y")
    else:
        direction = str(payload.get("direction") or "").lower()
        amount = str(payload.get("amount") or "medium").lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("scroll requires numeric delta_x/delta_y or a legacy direction")
        if amount not in LEGACY_SCROLL_AMOUNT_TO_DELTA:
            raise ValueError("legacy scroll amount must be small, medium, or large")
        magnitude = LEGACY_SCROLL_AMOUNT_TO_DELTA[amount]
        delta_x = magnitude if direction == "right" else -magnitude if direction == "left" else 0.0
        delta_y = magnitude if direction == "down" else -magnitude if direction == "up" else 0.0
    payload.pop("direction", None)
    payload.pop("amount", None)
    payload["delta_x"] = delta_x
    payload["delta_y"] = delta_y
    return payload


def scroll_direction(delta_x: float, delta_y: float) -> str:
    """Map numeric deltas to a direction for backends that only support direction strings."""

    if abs(delta_y) >= abs(delta_x):
        return "down" if delta_y >= 0 else "up"
    return "right" if delta_x >= 0 else "left"


@dataclass(frozen=True)
class UnifiedAction:
    type: ActionType
    params: Mapping[str, Any] = field(default_factory=dict)
    raw: Any | None = None

    def __post_init__(self) -> None:
        if self.type is ActionType.SCROLL:
            object.__setattr__(self, "params", normalize_scroll_params(self.params))
        self.validate()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnifiedAction":
        if "type" not in payload:
            raise ValueError("UnifiedAction payload requires a type")
        action_type = ActionType(str(payload["type"]))
        params = {key: value for key, value in payload.items() if key not in {"type", "raw"}}
        if action_type is ActionType.UPLOAD:
            if "path" in params and "file_path" in params and params["path"] != params["file_path"]:
                raise ValueError("upload path and file_path disagree")
            if "file_path" in params and "path" not in params:
                params["path"] = params["file_path"]
            params.pop("file_path", None)
        action = cls(type=action_type, params=params, raw=payload.get("raw"))
        action.validate()
        return action

    def to_dict(self) -> dict[str, Any]:
        payload = {"type": self.type.value, **dict(self.params)}
        if self.raw is not None:
            payload["raw"] = self.raw
        return payload

    def validate(self) -> None:
        for key in COORDINATE_KEYS:
            if key in self.params:
                _validate_point(self.params[key], field_name=key)
        if self.type is ActionType.DRAG:
            _validate_point(self.params.get("from_coordinate"), field_name="from_coordinate")
            _validate_point(self.params.get("to_coordinate"), field_name="to_coordinate")
        if self.type in {
            ActionType.LEFT_CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.SELECT_OPTION,
            ActionType.TOGGLE_CHECK,
            ActionType.HOVER,
            ActionType.MOVE_TO,
            ActionType.MOUSE_DOWN,
            ActionType.MOUSE_UP,
        }:
            _validate_point(self.params.get("coordinate"), field_name="coordinate")
        if self.type is ActionType.TYPE_TEXT:
            _validate_text(self.params.get("text"), field_name="text", allow_empty=True)
        if self.type is ActionType.SCROLL:
            _numeric_scroll_delta(self.params.get("delta_x"), field_name="delta_x")
            _numeric_scroll_delta(self.params.get("delta_y"), field_name="delta_y")
        if self.type is ActionType.KEY_PRESS:
            _validate_text(self.params.get("key"), field_name="key")
        if self.type is ActionType.HOTKEY:
            _validate_key_sequence(self.params.get("keys"), field_name="keys")
        if self.type in {ActionType.KEY_DOWN, ActionType.KEY_UP}:
            _validate_text(self.params.get("key"), field_name="key")
        if self.type is ActionType.NAVIGATE:
            _validate_text(self.params.get("url"), field_name="url")
        if self.type is ActionType.UPLOAD:
            _validate_text(self.params.get("path"), field_name="path")
        if self.type is ActionType.FAIL:
            _validate_text(self.params.get("reason"), field_name="reason")


def action_to_prompt_dict(
    action: UnifiedAction | Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize an action for model-visible traces on the project 0..999 grid.

    Prompt traces must be built from the pre-adapter UnifiedAction. Backend
    actions may contain environment-specific pixel coordinates and must never
    be passed here. The optional raw field is excluded because it can preserve
    backend-native payloads outside the unified coordinate contract.
    """

    payload = (
        action.to_dict()
        if isinstance(action, UnifiedAction)
        else dict(action)
    )
    payload.pop("raw", None)
    if "type" not in payload:
        raise ValueError("prompt trace action requires a type")
    action_type = payload["type"]
    payload["type"] = (
        action_type.value
        if isinstance(action_type, ActionType)
        else str(action_type)
    )
    for key in COORDINATE_KEYS:
        if key in payload:
            x, y = _validate_point(payload[key], field_name=key)
            payload[key] = [x, y]
    if payload["type"] == ActionType.SCROLL.value:
        normalized = normalize_scroll_params(payload)
        payload["delta_x"] = normalized["delta_x"]
        payload["delta_y"] = normalized["delta_y"]
        payload.pop("direction", None)
        payload.pop("amount", None)
    return payload


SUPPORTED_ACTION_TYPES = tuple(action_type.value for action_type in ActionType)
