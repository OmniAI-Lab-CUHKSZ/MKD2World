"""VisualWebArena runtime adapter for the MKD2World pipeline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from mkd2world.actions import ActionType, UnifiedAction, scroll_direction
from mkd2world.contracts import GuiState
from mkd2world.rendering import render_markdown_from_screenshot


VWA_RUNTIME_SUPPORTED_ACTIONS = tuple(
    item.value
    for item in (
        ActionType.LEFT_CLICK,
        ActionType.TYPE_TEXT,
        ActionType.SCROLL,
        ActionType.KEY_PRESS,
        ActionType.HOTKEY,
        ActionType.NAVIGATE,
        ActionType.GO_BACK,
        ActionType.GO_FORWARD,
        ActionType.REFRESH,
        ActionType.HOVER,
        ActionType.WAIT,
        ActionType.NEW_TAB,
        ActionType.PAGE_CLOSE,
        ActionType.CLEAR,
        ActionType.FINISH,
        ActionType.FAIL,
    )
)


def map_vwa_url_to_real(url: str) -> str:
    """Expose a local VWA URL as the public domain used by official prompts."""

    from browser_env.env_config import URL_MAPPINGS

    for local, real in sorted(URL_MAPPINGS.items(), key=lambda item: len(item[0]), reverse=True):
        if local and url.startswith(local):
            suffix = url[len(local) :]
            return f"{real.rstrip('/')}/{suffix.lstrip('/')}"
    return url


def map_vwa_url_to_local(url: str) -> str:
    """Map official/public domains back to the local VWA services."""

    from browser_env.env_config import URL_MAPPINGS

    for local, real in URL_MAPPINGS.items():
        real = real.rstrip("/")
        domain = real.split("://", 1)[-1]
        aliases = [
            f"http://{domain}",
            f"https://{domain}",
            f"http://www.{domain}",
            f"https://www.{domain}",
        ]
        if real == "http://wikipedia.org":
            aliases.extend(["http://en.wikipedia.org", "https://en.wikipedia.org"])
        for alias in aliases:
            if not url.startswith(alias):
                continue
            suffix = url[len(alias) :]
            if alias.endswith("en.wikipedia.org") and suffix.startswith("/wiki/"):
                suffix = f"/A/{suffix[len('/wiki/') :]}"
            return f"{local.rstrip('/')}/{suffix.lstrip('/')}"
    return url


@dataclass
class VisualWebArenaEnvironment:
    vwa_root: str | Path = "envs/visualwebarena"
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    observation_type: str = "accessibility_tree"
    sleep_after_execution: float = 1.0
    render_config: dict[str, Any] = field(default_factory=dict)
    site_urls: dict[str, str] = field(default_factory=dict)
    eval_captioning_model: str = "Salesforce/blip2-flan-t5-xl"
    eval_captioning_model_device: str = "cpu"
    supported_actions: tuple[str, ...] = VWA_RUNTIME_SUPPORTED_ACTIONS
    _env: Any = field(default=None, init=False, repr=False)
    _task_config_path: Path | None = field(default=None, init=False)
    _temporary_task: tempfile.TemporaryDirectory | None = field(default=None, init=False, repr=False)
    _trajectory: list[Any] = field(default_factory=list, init=False, repr=False)
    _state: GuiState | None = field(default=None, init=False, repr=False)
    _last_info: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _terminated: bool = field(default=False, init=False, repr=False)
    _truncated: bool = field(default=False, init=False, repr=False)
    _eval_captioning_fn: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vwa_root = Path(self.vwa_root).resolve()
        if self.observation_type not in {
            "html",
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image",
            "image_som",
        }:
            raise ValueError(f"Unsupported VisualWebArena observation_type: {self.observation_type}")
        if self.eval_captioning_model_device not in {"cpu", "cuda"}:
            raise ValueError("eval_captioning_model_device must be 'cpu' or 'cuda'")

    def _load_runtime(self) -> None:
        if self._env is not None:
            return
        if not self.vwa_root.is_dir():
            raise FileNotFoundError(self.vwa_root)
        defaults = {
            "HOMEPAGE": "http://127.0.0.1:4399",
            "CLASSIFIEDS": "http://127.0.0.1:9980",
            "CLASSIFIEDS_RESET_TOKEN": "4b61655535e7ed388f0d40a93600254c",
            "SHOPPING": "http://127.0.0.1:7770",
            "REDDIT": "http://127.0.0.1:9999",
            "WIKIPEDIA": "http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/",
        }
        for key, value in {**defaults, **self.site_urls}.items():
            os.environ.setdefault(key, value)
        os.environ.setdefault("DATASET", "visualwebarena")
        os.environ.setdefault("OPENAI_API_KEY", "none")
        project_root = self.vwa_root.parents[1]
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(project_root / "envs/ms-playwright"))
        native_lib = project_root / "envs/vwa_eval/lib"
        current_lib = os.environ.get("LD_LIBRARY_PATH", "")
        if str(native_lib) not in current_lib.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{native_lib}:{current_lib}".rstrip(":")
        if str(self.vwa_root) not in sys.path:
            sys.path.insert(0, str(self.vwa_root))
        from browser_env import ScriptBrowserEnv

        self._env = ScriptBrowserEnv(
            headless=self.headless,
            observation_type=self.observation_type,
            current_viewport_only=True,
            viewport_size={"width": self.viewport_width, "height": self.viewport_height},
            sleep_after_execution=self.sleep_after_execution,
        )

    def reset(self, task: Any = None) -> GuiState:
        self._load_runtime()
        self._task_config_path = self._resolve_task(task)
        observation, info = self._env.reset(options={"config_file": str(self._task_config_path)})
        self._last_info = dict(info)
        self._terminated = False
        self._truncated = False
        self._trajectory = [{"observation": observation, "info": info}]
        self._state = self._state_from_observation(observation, info)
        return self._state

    def observe(self) -> GuiState:
        if self._state is None:
            raise RuntimeError("VisualWebArena environment must be reset before observe")
        return self._state

    def execute(self, action: UnifiedAction) -> GuiState:
        if self._env is None or self._state is None:
            raise RuntimeError("VisualWebArena environment must be reset before execute")
        for native in self._to_native_actions(action):
            self._trajectory.append(native)
            if action.type in {ActionType.FINISH, ActionType.FAIL}:
                return self._state
            observation, _reward, self._terminated, self._truncated, info = self._env.step(native)
            self._last_info = dict(info)
            self._trajectory.append({"observation": observation, "info": info})
            self._state = self._state_from_observation(observation, info)
        return self._state

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    def evaluate(self) -> float:
        if self._env is None or self._task_config_path is None:
            return 0.0
        task_payload = json.loads(self._task_config_path.read_text(encoding="utf-8"))
        eval_types = task_payload.get("eval", {}).get("eval_types") or []
        if "string_match" in eval_types:
            # String-match tasks without a final STOP answer score 0.
            from browser_env.actions import ActionTypes

            last_action = self._trajectory[-1] if self._trajectory else None
            if (
                not isinstance(last_action, dict)
                or last_action.get("action_type") != ActionTypes.STOP
            ):
                return 0.0
        from evaluation_harness import evaluator_router

        captioning_fn = (
            self._get_eval_captioning_fn() if self._task_requires_captioning() else None
        )
        evaluator = evaluator_router(str(self._task_config_path), captioning_fn=captioning_fn)
        try:
            return float(
                evaluator(
                    trajectory=self._trajectory,
                    config_file=str(self._task_config_path),
                    page=self._env.page,
                )
            )
        except KeyError as exc:
            if exc.args == ("answer",):
                return 0.0
            raise

    def close(self) -> None:
        if self._env is not None:
            env = self._env
            try:
                env.close()
            except Exception:
                pass
            if getattr(env, "reset_finished", False) is False:
                context_manager = getattr(env, "context_manager", None)
                if context_manager is not None:
                    try:
                        context_manager.__exit__(None, None, None)
                    except Exception:
                        pass
            self._env = None
        if self._temporary_task is not None:
            self._temporary_task.cleanup()
            self._temporary_task = None

    def _resolve_task(self, task: Any) -> Path:
        if isinstance(task, (str, Path)):
            path = Path(task).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif isinstance(task, dict):
            payload = dict(task)
        else:
            raise ValueError("VisualWebArena reset requires a task config path or mapping")
        if not isinstance(payload, dict):
            raise ValueError("VisualWebArena task config must be a JSON object")
        for key in ("storage_state", "image"):
            value = payload.get(key)
            if isinstance(value, str) and value and not value.startswith(("http://", "https://")):
                candidate = Path(value)
                if not candidate.is_absolute():
                    payload[key] = str((self.vwa_root / candidate).resolve())
        self._normalize_page_image_references(payload)
        if self._temporary_task is not None:
            self._temporary_task.cleanup()
        self._temporary_task = tempfile.TemporaryDirectory(prefix="mkd2world_vwa_task_")
        normalized_path = Path(self._temporary_task.name) / "task.json"
        normalized_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return normalized_path

    def _task_requires_captioning(self) -> bool:
        if self._task_config_path is None:
            return False
        payload = json.loads(self._task_config_path.read_text(encoding="utf-8"))
        return any(
            bool(query.get("eval_vqa"))
            for query in payload.get("eval", {}).get("page_image_query", [])
        )

    def _get_eval_captioning_fn(self):
        if self._eval_captioning_fn is not None:
            return self._eval_captioning_fn
        import torch
        from evaluation_harness.image_utils import get_captioning_fn

        device = self.eval_captioning_model_device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VisualWebArena eval captioning requested CUDA, but CUDA is unavailable")
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = self.eval_captioning_model
        local_official_model = (
            self.vwa_root.parents[1]
            / "models/visualwebarena_eval/blip2-flan-t5-xl"
        )
        if (
            model == "Salesforce/blip2-flan-t5-xl"
            and (local_official_model / "config.json").is_file()
        ):
            model = str(local_official_model)
        self._eval_captioning_fn = get_captioning_fn(
            device,
            dtype,
            model,
        )
        return self._eval_captioning_fn

    def _normalize_page_image_references(self, payload: dict[str, Any]) -> None:
        shopping_url = (
            os.environ.get("SHOPPING")
            or self.site_urls.get("SHOPPING")
            or "http://127.0.0.1:7770"
        )
        for query in payload.get("eval", {}).get("page_image_query", []):
            fuzzy = query.get("eval_fuzzy_image_match")
            if not isinstance(fuzzy, str):
                continue
            normalized = []
            for value in fuzzy.split(" |OR| "):
                if value.startswith(("http://", "https://")):
                    normalized.append(value)
                elif value.startswith("media/"):
                    normalized.append(f"{shopping_url.rstrip('/')}/{value}")
                else:
                    normalized.append(str((self.vwa_root / value).resolve()))
            query["eval_fuzzy_image_match"] = " |OR| ".join(normalized)

    def _state_from_observation(self, observation: dict, info: dict) -> GuiState:
        image_array = observation.get("image")
        if image_array is None:
            raise ValueError("VisualWebArena observation has no image")
        image = Image.fromarray(image_array).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            image.save(handle.name)
            screenshot = Path(handle.name).read_bytes()
        markdown = render_markdown_from_screenshot(
            screenshot,
            {"environment": "Web", **self.render_config},
        )
        page = info.get("page") or getattr(self._env, "page", None)
        url = str(getattr(page, "url", "") or "") or None
        return GuiState(screenshot=screenshot, markdown=markdown, url=url, environment="Web")

    @staticmethod
    def _json_safe_native(native: Any):
        coords = native.get("coords") if isinstance(native, dict) else None
        if coords is not None:
            native["coords"] = [float(coords[0]), float(coords[1])]
        return native

    def _to_native_action(self, action: UnifiedAction):
        action.validate()
        from browser_env.actions import (
            create_go_back_action,
            create_go_forward_action,
            create_goto_url_action,
            create_key_press_action,
            create_keyboard_type_action,
            create_mouse_click_action,
            create_mouse_hover_action,
            create_new_tab_action,
            create_none_action,
            create_page_close_action,
            create_scroll_action,
            create_stop_action,
        )

        if action.type is ActionType.LEFT_CLICK:
            # VWA mouse actions take normalized coordinates, not pixels.
            x = max(0.001, min(1.0, int(action.params["coordinate"][0]) / 999))
            y = max(0.001, min(1.0, int(action.params["coordinate"][1]) / 999))
            return self._json_safe_native(create_mouse_click_action(x, y))
        if action.type is ActionType.HOVER:
            x = max(0.001, min(1.0, int(action.params["coordinate"][0]) / 999))
            y = max(0.001, min(1.0, int(action.params["coordinate"][1]) / 999))
            return self._json_safe_native(create_mouse_hover_action(x, y))
        if action.type is ActionType.TYPE_TEXT:
            return create_keyboard_type_action(action.params["text"])
        if action.type is ActionType.SCROLL:
            return create_scroll_action(scroll_direction(action.params["delta_x"], action.params["delta_y"]))
        if action.type is ActionType.KEY_PRESS:
            return create_key_press_action(action.params["key"])
        if action.type is ActionType.HOTKEY:
            key_aliases = {
                "ctrl": "Control",
                "control": "Control",
                "cmd": "Meta",
                "command": "Meta",
                "meta": "Meta",
                "alt": "Alt",
                "shift": "Shift",
            }
            combo = "+".join(
                key_aliases.get(str(key).lower(), str(key))
                for key in action.params["keys"]
            )
            return create_key_press_action(combo)
        if action.type is ActionType.NAVIGATE:
            return create_goto_url_action(map_vwa_url_to_local(action.params["url"]))
        if action.type is ActionType.GO_BACK:
            return create_go_back_action()
        if action.type is ActionType.GO_FORWARD:
            return create_go_forward_action()
        if action.type is ActionType.REFRESH:
            return create_key_press_action("Control+R")
        if action.type is ActionType.NEW_TAB:
            return create_new_tab_action()
        if action.type is ActionType.PAGE_CLOSE:
            return create_page_close_action()
        if action.type is ActionType.WAIT:
            return create_none_action()
        if action.type is ActionType.FINISH:
            return create_stop_action(str(action.params.get("answer") or ""))
        if action.type is ActionType.FAIL:
            reason = str(action.params.get("reason") or "").strip()
            if reason.startswith("Invalid agent action:"):
                return create_none_action()

            return create_stop_action("N/A")
        raise NotImplementedError(f"VisualWebArena runtime does not support {action.type.value}")

    def _to_native_actions(self, action: UnifiedAction) -> list[Any]:
        """Expand one unified action into the smallest equivalent VWA sequence."""

        action.validate()
        from browser_env.actions import (
            create_key_press_action,
            create_keyboard_type_action,
            create_mouse_click_action,
        )

        if action.type not in {ActionType.CLEAR, ActionType.TYPE_TEXT}:
            return [self._to_native_action(action)]

        native_actions = []
        coordinate = action.params.get("coordinate")
        if coordinate is not None:
            x = max(0.001, min(1.0, int(coordinate[0]) / 999))
            y = max(0.001, min(1.0, int(coordinate[1]) / 999))
            native_actions.append(self._json_safe_native(create_mouse_click_action(x, y)))

        if action.type is ActionType.CLEAR or bool(action.params.get("clear", False)):
            native_actions.extend(
                [
                    create_key_press_action("Control+A"),
                    create_key_press_action("Backspace"),
                ]
            )
        if action.type is ActionType.TYPE_TEXT:
            native_actions.append(create_keyboard_type_action(action.params["text"]))
            if bool(action.params.get("enter", False)):
                native_actions.append(create_key_press_action("Enter"))
        return native_actions
