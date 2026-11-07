"""Environment adapter interfaces and action conversion boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from mkd2world.actions import ActionType, UnifiedAction, scroll_direction
from mkd2world.contracts import GuiState


class EnvironmentAdapter(Protocol):
    supported_actions: tuple[str, ...]

    def observe(self) -> GuiState:
        ...

    def execute(self, action: UnifiedAction) -> GuiState:
        ...


@dataclass
class StaticConversionEnvironment:
    state: GuiState
    supported_actions: tuple[str, ...]
    converter: object | None = None
    after_markdown: str | None = None
    executed: list[dict] = field(default_factory=list)

    def observe(self) -> GuiState:
        return self.state

    def execute(self, action: UnifiedAction) -> GuiState:
        backend_action: object
        if self.converter is None:
            backend_action = action.to_dict()
        elif hasattr(self.converter, "to_pyautogui"):
            backend_action = self.converter.to_pyautogui(action)
        elif hasattr(self.converter, "to_action_string"):
            backend_action = self.converter.to_action_string(action)
        elif hasattr(self.converter, "to_action"):
            backend_action = self.converter.to_action(action)
        else:
            raise TypeError(f"Unsupported environment converter: {type(self.converter).__name__}")
        self.executed.append({"unified_action": action.to_dict(), "backend_action": backend_action})
        return GuiState(
            screenshot=self.state.screenshot,
            markdown=self.after_markdown if self.after_markdown is not None else self.state.markdown,
            url=self.state.url,
            environment=self.state.environment,
        )

def grid_to_pixel(coordinate: tuple[int, int] | list[int], width: int, height: int) -> tuple[int, int]:
    x, y = coordinate
    return round(int(x) * max(width - 1, 0) / 999), round(int(y) * max(height - 1, 0) / 999)


def _py_string(value: object) -> str:
    return repr(str(value))


def _py_hotkey(keys: tuple[str, ...] | list[str]) -> str:
    return ", ".join(_py_string(key) for key in keys)


OSWORLD_SUPPORTED_ACTIONS = tuple(
    item.value
    for item in (
        ActionType.LEFT_CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.TYPE_TEXT,
        ActionType.SCROLL,
        ActionType.KEY_PRESS,
        ActionType.HOTKEY,
        ActionType.KEY_DOWN,
        ActionType.KEY_UP,
        ActionType.NAVIGATE,
        ActionType.GO_BACK,
        ActionType.GO_FORWARD,
        ActionType.REFRESH,
        ActionType.DRAG,
        ActionType.HOVER,
        ActionType.MOVE_TO,
        ActionType.MOUSE_DOWN,
        ActionType.MOUSE_UP,
        ActionType.WAIT,
        ActionType.NEW_TAB,
        ActionType.PAGE_CLOSE,
        ActionType.CLEAR,
        ActionType.FINISH,
        ActionType.FAIL,
    )
)


VWA_SUPPORTED_ACTIONS = tuple(
    item.value
    for item in (
        ActionType.LEFT_CLICK,
        ActionType.TYPE_TEXT,
        ActionType.SCROLL,
        ActionType.KEY_PRESS,
        ActionType.NAVIGATE,
        ActionType.GO_BACK,
        ActionType.GO_FORWARD,
        ActionType.REFRESH,
        ActionType.HOVER,
        ActionType.WAIT,
        ActionType.NEW_TAB,
        ActionType.PAGE_CLOSE,
        ActionType.CLEAR,
        ActionType.UPLOAD,
        ActionType.FINISH,
    )
)


ONLINE_MIND2WEB_SUPPORTED_ACTIONS = tuple(
    item.value
    for item in (
        ActionType.LEFT_CLICK,
        ActionType.TYPE_TEXT,
        ActionType.SCROLL,
        ActionType.KEY_PRESS,
        ActionType.NAVIGATE,
        ActionType.GO_BACK,
        ActionType.GO_FORWARD,
        ActionType.REFRESH,
        ActionType.SELECT_OPTION,
        ActionType.HOVER,
        ActionType.WAIT,
        ActionType.FINISH,
    )
)


@dataclass(frozen=True)
class OSWorldActionAdapter:
    width: int
    height: int

    def to_pyautogui(self, action: UnifiedAction) -> str:
        action.validate()
        if action.type in {ActionType.LEFT_CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK, ActionType.MOVE_TO}:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            name = {
                ActionType.LEFT_CLICK: "click",
                ActionType.DOUBLE_CLICK: "doubleClick",
                ActionType.RIGHT_CLICK: "rightClick",
                ActionType.MOVE_TO: "moveTo",
            }[action.type]
            return f"pyautogui.{name}({x}, {y})"
        if action.type in {ActionType.HOVER, ActionType.MOVE_TO}:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return f"pyautogui.moveTo({x}, {y})"
        if action.type is ActionType.MOUSE_DOWN:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return f"pyautogui.moveTo({x}, {y})\npyautogui.mouseDown()"
        if action.type is ActionType.MOUSE_UP:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return f"pyautogui.moveTo({x}, {y})\npyautogui.mouseUp()"
        if action.type is ActionType.DRAG:
            x1, y1 = grid_to_pixel(action.params["from_coordinate"], self.width, self.height)
            x2, y2 = grid_to_pixel(action.params["to_coordinate"], self.width, self.height)
            return f"pyautogui.moveTo({x1}, {y1})\npyautogui.dragTo({x2}, {y2}, duration=0.2, button='left')"
        if action.type is ActionType.TYPE_TEXT:
            return f"pyautogui.write({_py_string(action.params['text'])})"
        if action.type is ActionType.SCROLL:
            delta_x = int(round(action.params["delta_x"]))
            delta_y = int(round(action.params["delta_y"]))
            commands = []
            if delta_x:
                commands.append(f"pyautogui.hscroll({delta_x})")
            if delta_y:
                commands.append(f"pyautogui.scroll({-delta_y})")
            return "\n".join(commands) if commands else "pyautogui.scroll(0)"
        if action.type is ActionType.KEY_PRESS:
            return f"pyautogui.press({_py_string(action.params['key'])})"
        if action.type is ActionType.HOTKEY:
            return f"pyautogui.hotkey({_py_hotkey(action.params['keys'])})"
        if action.type is ActionType.KEY_DOWN:
            return f"pyautogui.keyDown({_py_string(action.params['key'])})"
        if action.type is ActionType.KEY_UP:
            return f"pyautogui.keyUp({_py_string(action.params['key'])})"
        if action.type is ActionType.NAVIGATE:
            return f"pyautogui.hotkey('ctrl', 'l')\npyautogui.write({_py_string(action.params['url'])})\npyautogui.press('enter')"
        if action.type is ActionType.GO_BACK:
            return "pyautogui.hotkey('alt', 'left')"
        if action.type is ActionType.GO_FORWARD:
            return "pyautogui.hotkey('alt', 'right')"
        if action.type is ActionType.REFRESH:
            return "pyautogui.press('f5')"
        if action.type is ActionType.NEW_TAB:
            return "pyautogui.hotkey('ctrl', 't')"
        if action.type is ActionType.PAGE_CLOSE:
            return "pyautogui.hotkey('ctrl', 'w')"
        if action.type is ActionType.CLEAR:
            return "pyautogui.hotkey('ctrl', 'a')\npyautogui.press('backspace')"
        if action.type is ActionType.WAIT:
            return "pyautogui.sleep(1)"
        if action.type is ActionType.FINISH:
            return "DONE"
        if action.type is ActionType.FAIL:
            return f"FAIL: {action.params['reason']}"
        raise NotImplementedError(f"OSWorld adapter does not yet support {action.type.value}")


@dataclass(frozen=True)
class VisualWebArenaActionAdapter:
    width: int
    height: int

    def to_action_string(self, action: UnifiedAction) -> str:
        action.validate()
        if action.type is ActionType.LEFT_CLICK:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return f"click [{x}] [{y}]"
        if action.type is ActionType.HOVER:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return f"hover [{x}] [{y}]"
        if action.type is ActionType.TYPE_TEXT:
            return f"type [focused] [{action.params['text']}]"
        if action.type is ActionType.SCROLL:
            return f"scroll [{scroll_direction(action.params['delta_x'], action.params['delta_y'])}]"
        if action.type is ActionType.KEY_PRESS:
            return f"key_press [{action.params['key']}]"
        if action.type is ActionType.NAVIGATE:
            return f"goto [{action.params['url']}]"
        if action.type is ActionType.GO_BACK:
            return "go_back"
        if action.type is ActionType.GO_FORWARD:
            return "go_forward"
        if action.type is ActionType.REFRESH:
            return "refresh"
        if action.type is ActionType.WAIT:
            return "wait"
        if action.type is ActionType.NEW_TAB:
            return "new_tab"
        if action.type is ActionType.PAGE_CLOSE:
            return "page_close"
        if action.type is ActionType.CLEAR:
            return "clear [focused]"
        if action.type is ActionType.UPLOAD:
            return f"upload [{action.params['path']}]"
        if action.type is ActionType.FINISH:
            return f"stop [{action.params.get('answer', '')}]"
        raise NotImplementedError(f"VisualWebArena adapter does not yet support {action.type.value}")


@dataclass(frozen=True)
class OnlineMind2WebActionAdapter:
    width: int
    height: int

    def to_action(self, action: UnifiedAction) -> dict:
        action.validate()
        if action.type is ActionType.LEFT_CLICK:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return {"action_type": "CLICK", "coordinate": [x, y]}
        if action.type is ActionType.TYPE_TEXT:
            return {"action_type": "TYPE", "text": action.params["text"]}
        if action.type is ActionType.SCROLL:
            return {
                "action_type": "SCROLL",
                "delta_x": action.params["delta_x"],
                "delta_y": action.params["delta_y"],
            }
        if action.type is ActionType.KEY_PRESS:
            return {"action_type": "PRESS_KEY", "key": action.params["key"]}
        if action.type is ActionType.NAVIGATE:
            return {"action_type": "NAVIGATE", "url": action.params["url"]}
        if action.type is ActionType.GO_BACK:
            return {"action_type": "GO_BACK"}
        if action.type is ActionType.GO_FORWARD:
            return {"action_type": "GO_FORWARD"}
        if action.type is ActionType.REFRESH:
            return {"action_type": "REFRESH"}
        if action.type is ActionType.SELECT_OPTION:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return {"action_type": "SELECT", "coordinate": [x, y], "option": action.params.get("option", "")}
        if action.type is ActionType.HOVER:
            x, y = grid_to_pixel(action.params["coordinate"], self.width, self.height)
            return {"action_type": "HOVER", "coordinate": [x, y]}
        if action.type is ActionType.WAIT:
            return {"action_type": "WAIT"}
        if action.type is ActionType.FINISH:
            return {"action_type": "TASK_COMPLETE", "answer": action.params.get("answer", "")}
        raise NotImplementedError(f"Online-Mind2Web adapter does not yet support {action.type.value}")
