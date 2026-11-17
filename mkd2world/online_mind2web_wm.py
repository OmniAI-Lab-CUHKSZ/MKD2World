

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from openai import AsyncOpenAI
from playwright.async_api import CDPSession, Page


TEXT_WORLD_MODELS = {"agentworld", "webworld", "wma", "webworld32b"}
WEBWORLD_KINDS = {"webworld", "webworld32b"}
WORLD_MODEL_KINDS = {"none", "agentworld", "dreamer", "webworld", "wma", "webworld32b"}
WMA_INPUT_CONTRACTS = {"official_contextual", "axtree_action_only"}
TERMINAL_ACTIONS = {"finish", "fail"}
NATIVE_CONTEXT_PROTOCOL_MARGIN = 16
ACTIONABLE_AX_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "slider",
    "switch",
    "tab",
    "textbox",
}


class NativeWorldModelError(RuntimeError):
    def __init__(self, message: str, *, audit_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit_payload = audit_payload


def _ax_value(value: Any, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get("value", default)
    return default


def _clean_inline(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


@dataclass(frozen=True)
class TargetResolution:
    status: str
    normalized_coordinate: list[int] | None
    pixel_coordinate: list[int] | None
    backend_node_id: int | None = None
    bid: str | None = None
    role: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeWorldModelPrediction:
    candidate_index: int
    unified_action: dict[str, Any]
    serialized_action: str
    observation_format: str
    predicted_observation: str
    raw_output: str
    reasoning_content: str
    target_resolution: TargetResolution
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    input_state_tokens: int | None
    finish_reason: str | None
    latency_seconds: float
    format_retry_count: int
    output_limit_hit: bool
    context_window_tokens: int | None = None
    context_utilization: float | None = None
    wm_called: bool = True
    skip_reason: str | None = None
    content_validation_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_resolution"] = self.target_resolution.to_dict()
        return payload


@dataclass(frozen=True)
class _AXNode:
    node_id: str
    parent_id: str | None
    child_ids: tuple[str, ...]
    backend_node_id: int | None
    role: str
    name: str
    value: str
    properties: tuple[tuple[str, Any], ...]
    focused: bool
    bounds: tuple[float, float, float, float] | None

    @property
    def bid(self) -> str | None:
        return str(self.backend_node_id) if self.backend_node_id is not None else None


class A11yDOMSnapshot:
    """A viewport-oriented AX tree whose bids are Chromium backend node ids."""

    def __init__(
        self,
        *,
        nodes: Sequence[_AXNode],
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        self.nodes = tuple(nodes)
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.by_id = {node.node_id: node for node in self.nodes}
        self.by_backend = {
            node.backend_node_id: node for node in self.nodes if node.backend_node_id is not None
        }
        self.order = {node.node_id: index for index, node in enumerate(self.nodes)}

    def _is_viewport_visible(self, node: _AXNode) -> bool:
        if node.role == "RootWebArea":
            return True
        if node.bounds is None:
            return False
        x, y, width, height = node.bounds
        return (
            width > 0
            and height > 0
            and x < self.viewport_width
            and y < self.viewport_height
            and x + width > 0
            and y + height > 0
        )

    def _ancestor_ids(self, node_id: str | None) -> set[str]:
        result: set[str] = set()
        current = self.by_id.get(node_id or "")
        while current is not None and current.node_id not in result:
            result.add(current.node_id)
            current = self.by_id.get(current.parent_id or "")
        return result

    @staticmethod
    def _line(node: _AXNode, depth: int) -> str:
        prefix = "\t" * max(0, depth)
        identity = (
            f"[{node.bid}] "
            if node.bid is not None and node.role != "RootWebArea"
            else ""
        )
        name = f" '{_clean_inline(node.name)}'" if node.name else ""
        value = f" value='{_clean_inline(node.value)}'" if node.value else ""
        flags = []
        for key, item in node.properties:
            if key in {"visible", "clickable", "editable"}:
                if bool(item):
                    flags.append(key)
                continue
            if key in {
                "autocomplete",
                "checked",
                "disabled",
                "expanded",
                "focusable",
                "focused",
                "hasPopup",
                "invalid",
                "level",
                "multiline",
                "multiselectable",
                "orientation",
                "pressed",
                "readonly",
                "required",
                "selected",
            }:
                flags.append(f"{key}={_clean_inline(item)}")
        suffix = f", {', '.join(flags)}" if flags else ""
        return f"{prefix}{identity}{node.role or 'generic'}{name}{value}{suffix}".rstrip()

    def _render_ids(self, included: set[str]) -> str:
        roots = [
            node
            for node in self.nodes
            if node.parent_id not in self.by_id or node.role == "RootWebArea"
        ]
        roots.sort(key=lambda node: (node.role != "RootWebArea", self.order[node.node_id]))
        lines: list[str] = []
        visited: set[str] = set()

        def visit(node: _AXNode, depth: int) -> None:
            if node.node_id in visited:
                return
            visited.add(node.node_id)
            keep = node.node_id in included
            if keep:
                lines.append(self._line(node, depth))
            next_depth = depth + 1 if keep else depth
            for child_id in node.child_ids:
                child = self.by_id.get(child_id)
                if child is not None:
                    visit(child, next_depth)

        for root in roots:
            visit(root, 0)
        for node in self.nodes:
            if node.node_id not in visited:
                visit(node, 0)
        return "\n".join(lines).strip()

    def render(
        self,
        *,
        target_bid: str | None,
        max_tokens: int,
        count_tokens: Callable[[str], int],
    ) -> tuple[str, int]:
        visible = {
            node.node_id
            for node in self.nodes
            if self._is_viewport_visible(node)
        }
        mandatory = {
            node.node_id
            for node in self.nodes
            if node.role == "RootWebArea" or node.focused
        }
        target_node = None
        if target_bid is not None:
            try:
                target_node = self.by_backend.get(int(target_bid))
            except ValueError:
                target_node = None
        if target_node is not None:
            mandatory.update(self._ancestor_ids(target_node.node_id))
        for node_id in tuple(visible):
            mandatory.update(self._ancestor_ids(node_id))

        ordered_visible = sorted(visible, key=lambda node_id: self.order[node_id])
        included = set(mandatory)
        for node_id in ordered_visible:
            included.add(node_id)
        text = self._render_ids(included)
        tokens = count_tokens(text)
        if tokens <= max_tokens:
            return text, tokens

        # Preserve roots, focus, and the resolved target path. Add remaining
        # viewport nodes in DOM order only while the real tokenizer budget fits.
        required = {
            node.node_id
            for node in self.nodes
            if node.role == "RootWebArea" or node.focused
        }
        if target_node is not None:
            required.update(self._ancestor_ids(target_node.node_id))
        included = set(required)
        text = self._render_ids(included)
        tokens = count_tokens(text)
        for node_id in ordered_visible:
            trial = included | self._ancestor_ids(node_id) | {node_id}
            trial_text = self._render_ids(trial)
            trial_tokens = count_tokens(trial_text)
            if trial_tokens <= max_tokens:
                included = trial
                text = trial_text
                tokens = trial_tokens
        if tokens > max_tokens:
            raise ValueError(
                f"mandatory A11y target path uses {tokens} tokens, exceeding budget {max_tokens}"
            )
        return text, tokens

    def render_full(
        self,
        *,
        target_bid: str | None,
        count_tokens: Callable[[str], int],
    ) -> tuple[str, int]:
        """Render the complete current viewport without budget-driven pruning."""

        included = {
            node.node_id
            for node in self.nodes
            if self._is_viewport_visible(node)
        }
        included.update(
            node.node_id
            for node in self.nodes
            if node.role == "RootWebArea" or node.focused
        )
        if target_bid is not None:
            try:
                target = self.by_backend.get(int(target_bid))
            except ValueError:
                target = None
            if target is not None:
                included.update(self._ancestor_ids(target.node_id))
        for node_id in tuple(included):
            included.update(self._ancestor_ids(node_id))
        text = self._render_ids(included)
        return text, count_tokens(text)


def _layout_bounds_by_backend(
    snapshot: dict[str, Any],
    *,
    scroll_x: float = 0,
    scroll_y: float = 0,
) -> dict[int, tuple[float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float]] = {}
    for document in snapshot.get("documents", []):
        nodes = document.get("nodes", {})
        backend_ids = nodes.get("backendNodeId", [])
        layout = document.get("layout", {})
        node_indices = layout.get("nodeIndex", [])
        bounds = layout.get("bounds", [])
        for node_index, rect in zip(node_indices, bounds):
            if (
                isinstance(node_index, int)
                and 0 <= node_index < len(backend_ids)
                and isinstance(rect, list)
                and len(rect) == 4
            ):
                result[int(backend_ids[node_index])] = (
                    float(rect[0]) - scroll_x,
                    float(rect[1]) - scroll_y,
                    float(rect[2]),
                    float(rect[3]),
                )
    return result



class _AwaitableCdpSession:
    """Expose one async CDP API for Playwright sync and async sessions."""

    def __init__(self, session: Any, *, threaded: bool) -> None:
        self._session = session
        self._threaded = threaded

    async def send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._threaded:
            # Playwright sync CDP is thread-affine; run inline on the caller thread.
            if params is None:
                return self._session.send(method)
            return self._session.send(method, params)
        if params is None:
            return await self._session.send(method)
        return await self._session.send(method, params)

    async def detach(self) -> None:
        detach = getattr(self._session, "detach", None)
        if detach is None:
            return
        if self._threaded:
            detach()
            return
        result = detach()
        if asyncio.iscoroutine(result):
            await result


def _page_uses_sync_playwright(page: Any) -> bool:
    context = getattr(page, "context", None)
    new_cdp = getattr(context, "new_cdp_session", None)
    if not callable(new_cdp):
        raise TypeError("page.context.new_cdp_session is required for native WM capture")
    return not asyncio.iscoroutinefunction(new_cdp)


async def _open_cdp_session(page: Any) -> Any:
    if _page_uses_sync_playwright(page):
        raw = page.context.new_cdp_session(page)
        return _AwaitableCdpSession(raw, threaded=True)
    return await page.context.new_cdp_session(page)


async def _wait_domcontentloaded(page: Any) -> None:
    wait = getattr(page, "wait_for_load_state", None)
    if wait is None:
        return
    if asyncio.iscoroutinefunction(wait):
        await wait("domcontentloaded", timeout=10_000)
        return
    wait("domcontentloaded", timeout=10_000)


async def _capture_a11y_dom_once(
    page: Page,
    *,
    viewport_width: int,
    viewport_height: int,
) -> tuple[CDPSession, A11yDOMSnapshot]:
    session = await _open_cdp_session(page)
    try:
        await session.send("DOM.enable")
        await session.send("Accessibility.enable")
        # Read scroll offset via CDP; some sites disable ``window.eval``.
        ax_result, dom_snapshot, layout_metrics = await asyncio.gather(
            session.send("Accessibility.getFullAXTree"),
            session.send(
                "DOMSnapshot.captureSnapshot",
                {
                    "computedStyles": [],
                    "includeDOMRects": True,
                    "includePaintOrder": True,
                },
            ),
            session.send("Page.getLayoutMetrics"),
        )
        viewport = (
            layout_metrics.get("cssVisualViewport")
            or layout_metrics.get("visualViewport")
            or layout_metrics.get("cssLayoutViewport")
            or layout_metrics.get("layoutViewport")
            or {}
        )
        bounds_by_backend = _layout_bounds_by_backend(
            dom_snapshot,
            scroll_x=float(viewport.get("pageX", 0)),
            scroll_y=float(viewport.get("pageY", 0)),
        )
        nodes: list[_AXNode] = []
        for raw in ax_result.get("nodes", []):
            if raw.get("ignored"):
                continue
            node_id = str(raw.get("nodeId", ""))
            if not node_id:
                continue
            backend = raw.get("backendDOMNodeId")
            backend_id = int(backend) if isinstance(backend, int) else None
            properties = tuple(
                (str(item.get("name", "")), _ax_value(item.get("value")))
                for item in raw.get("properties", [])
                if item.get("name")
            )
            property_map = dict(properties)
            role = _clean_inline(_ax_value(raw.get("role"), "generic"))
            bounds = bounds_by_backend.get(backend_id) if backend_id is not None else None
            if bounds is not None:
                x, y, width, height = bounds
                if (
                    width > 0
                    and height > 0
                    and x < viewport_width
                    and y < viewport_height
                    and x + width > 0
                    and y + height > 0
                ):
                    properties += (("visible", True),)
            if role.lower() in {
                "button",
                "checkbox",
                "combobox",
                "link",
                "menuitem",
                "option",
                "radio",
                "slider",
                "switch",
                "tab",
                "textbox",
            }:
                properties += (("clickable", True),)
            if role.lower() in {"combobox", "searchbox", "textbox"}:
                properties += (("editable", True),)
            nodes.append(
                _AXNode(
                    node_id=node_id,
                    parent_id=str(raw["parentId"]) if raw.get("parentId") is not None else None,
                    child_ids=tuple(str(item) for item in raw.get("childIds", [])),
                    backend_node_id=backend_id,
                    role=role,
                    name=_clean_inline(_ax_value(raw.get("name"))),
                    value=_clean_inline(_ax_value(raw.get("value"))),
                    properties=properties,
                    focused=bool(property_map.get("focused", False)),
                    bounds=bounds,
                )
            )
        return session, A11yDOMSnapshot(
            nodes=nodes,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
    except Exception:
        await session.detach()
        raise


def _is_transient_navigation_snapshot_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "execution context was destroyed",
            "cannot find context with specified id",
            "inspected target navigated or closed",
            "target closed",
        )
    )


async def capture_a11y_dom(
    page: Page,
    *,
    viewport_width: int,
    viewport_height: int,
) -> tuple[CDPSession, A11yDOMSnapshot]:
    """Capture a stable native web observation across redirect races.

    A click can start an asynchronous navigation after the action wait has
    returned.  Retrying the read-only CDP snapshot is preferable to converting
    that browser race into a model failure and restarting the entire task.
    """

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return await _capture_a11y_dom_once(
                page,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
            )
        except Exception as exc:
            last_error = exc
            if not _is_transient_navigation_snapshot_error(exc) or attempt == 2:
                raise
            try:
                await _wait_domcontentloaded(page)
            except Exception:
                pass
            await asyncio.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _action_coordinate(action: dict[str, Any]) -> list[int] | None:
    for key in ("coordinate", "from_coordinate"):
        value = action.get(key)
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, int) for item in value)
        ):
            return [int(value[0]), int(value[1])]
    return None


def _to_pixel(coordinate: Sequence[int], width: int, height: int) -> list[int]:
    return [
        max(0, min(width - 1, round(int(coordinate[0]) * (width - 1) / 999))),
        max(0, min(height - 1, round(int(coordinate[1]) * (height - 1) / 999))),
    ]


def _prefer_actionable_ancestor(
    snapshot: A11yDOMSnapshot,
    node: _AXNode,
) -> _AXNode:
    """Map a hit-test leaf to the nearest interactive AX ancestor.

    Chromium commonly returns the anonymous ``generic`` or ``StaticText``
    child inside a button. Native web-world models need the actionable
    element's bid, not that presentational leaf's bid.
    """

    current = node
    visited: set[str] = set()
    while current.node_id not in visited:
        visited.add(current.node_id)
        if current.role.lower() in ACTIONABLE_AX_ROLES:
            return current
        parent = snapshot.by_id.get(current.parent_id or "")
        if parent is None:
            break
        current = parent
    return node



def _capture_a11y_dom_sync(
    page: Any,
    *,
    viewport_width: int,
    viewport_height: int,
) -> tuple[Any, A11yDOMSnapshot]:
    last_error: Exception | None = None
    for attempt in range(3):
        session = None
        try:
            session = page.context.new_cdp_session(page)
            session.send("DOM.enable")
            session.send("Accessibility.enable")
            ax_result = session.send("Accessibility.getFullAXTree")
            dom_snapshot = session.send(
                "DOMSnapshot.captureSnapshot",
                {
                    "computedStyles": [],
                    "includeDOMRects": True,
                    "includePaintOrder": True,
                },
            )
            layout_metrics = session.send("Page.getLayoutMetrics")
            viewport = (
                layout_metrics.get("cssVisualViewport")
                or layout_metrics.get("visualViewport")
                or layout_metrics.get("cssLayoutViewport")
                or layout_metrics.get("layoutViewport")
                or {}
            )
            bounds_by_backend = _layout_bounds_by_backend(
                dom_snapshot,
                scroll_x=float(viewport.get("pageX", 0)),
                scroll_y=float(viewport.get("pageY", 0)),
            )
            nodes: list[_AXNode] = []
            for raw in ax_result.get("nodes", []):
                if raw.get("ignored"):
                    continue
                node_id = str(raw.get("nodeId", ""))
                if not node_id:
                    continue
                backend = raw.get("backendDOMNodeId")
                backend_id = int(backend) if isinstance(backend, int) else None
                properties = tuple(
                    (str(item.get("name", "")), _ax_value(item.get("value")))
                    for item in raw.get("properties", [])
                    if item.get("name")
                )
                property_map = dict(properties)
                role = _clean_inline(_ax_value(raw.get("role"), "generic"))
                bounds = bounds_by_backend.get(backend_id) if backend_id is not None else None
                if bounds is not None:
                    x, y, width, height = bounds
                    if (
                        width > 0
                        and height > 0
                        and x < viewport_width
                        and y < viewport_height
                        and x + width > 0
                        and y + height > 0
                    ):
                        properties += (("visible", True),)
                if role.lower() in ACTIONABLE_AX_ROLES:
                    properties += (("clickable", True),)
                if role.lower() in {"combobox", "searchbox", "textbox"}:
                    properties += (("editable", True),)
                nodes.append(
                    _AXNode(
                        node_id=node_id,
                        parent_id=str(raw["parentId"]) if raw.get("parentId") is not None else None,
                        child_ids=tuple(str(item) for item in raw.get("childIds", [])),
                        backend_node_id=backend_id,
                        role=role,
                        name=_clean_inline(_ax_value(raw.get("name"))),
                        value=_clean_inline(_ax_value(raw.get("value"))),
                        properties=properties,
                        focused=bool(property_map.get("focused", False)),
                        bounds=bounds,
                    )
                )
            return session, A11yDOMSnapshot(
                nodes=nodes,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
            )
        except Exception as exc:
            last_error = exc
            if session is not None:
                try:
                    session.detach()
                except Exception:
                    pass
            if not _is_transient_navigation_snapshot_error(exc) or attempt == 2:
                raise
            wait = getattr(page, "wait_for_load_state", None)
            if callable(wait):
                try:
                    wait("domcontentloaded", timeout=10_000)
                except Exception:
                    pass
            time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def resolve_action_target_sync(
    session: Any,
    snapshot: A11yDOMSnapshot,
    action: dict[str, Any],
    *,
    width: int,
    height: int,
) -> TargetResolution:
    coordinate = _action_coordinate(action)
    if coordinate is None:
        return TargetResolution("not_applicable", None, None)
    pixel = _to_pixel(coordinate, width, height)
    try:
        located = session.send(
            "DOM.getNodeForLocation",
            {
                "x": pixel[0],
                "y": pixel[1],
                "includeUserAgentShadowDOM": True,
                "ignorePointerEventsNone": True,
            },
        )
        backend_id = located.get("backendNodeId")
        if not isinstance(backend_id, int):
            raise ValueError("DOM.getNodeForLocation returned no backend node id")
        node = snapshot.by_backend.get(backend_id)
        status = "exact"
        if node is None:
            partial = session.send(
                "Accessibility.getPartialAXTree",
                {"backendNodeId": backend_id, "fetchRelatives": True},
            )
            for raw in partial.get("nodes", []):
                candidate_backend = raw.get("backendDOMNodeId")
                if isinstance(candidate_backend, int) and candidate_backend in snapshot.by_backend:
                    node = snapshot.by_backend[candidate_backend]
                    status = "ancestor"
                    break
        if node is None:
            return TargetResolution(
                "coordinate_fallback",
                coordinate,
                pixel,
                backend_node_id=backend_id,
            )
        actionable = _prefer_actionable_ancestor(snapshot, node)
        if actionable is not node:
            node = actionable
            status = "actionable_ancestor"
        return TargetResolution(
            status,
            coordinate,
            pixel,
            backend_node_id=node.backend_node_id,
            bid=node.bid,
            role=node.role,
            name=node.name,
        )
    except Exception:
        return TargetResolution("coordinate_fallback", coordinate, pixel)


async def resolve_action_target(
    session: CDPSession,
    snapshot: A11yDOMSnapshot,
    action: dict[str, Any],
    *,
    width: int,
    height: int,
) -> TargetResolution:
    coordinate = _action_coordinate(action)
    if coordinate is None:
        return TargetResolution("not_applicable", None, None)
    pixel = _to_pixel(coordinate, width, height)
    try:
        located = await session.send(
            "DOM.getNodeForLocation",
            {
                "x": pixel[0],
                "y": pixel[1],
                "includeUserAgentShadowDOM": True,
                "ignorePointerEventsNone": True,
            },
        )
        backend_id = located.get("backendNodeId")
        if not isinstance(backend_id, int):
            raise ValueError("DOM.getNodeForLocation returned no backend node id")
        node = snapshot.by_backend.get(backend_id)
        status = "exact"
        if node is None:
            partial = await session.send(
                "Accessibility.getPartialAXTree",
                {"backendNodeId": backend_id, "fetchRelatives": True},
            )
            for raw in partial.get("nodes", []):
                candidate_backend = raw.get("backendDOMNodeId")
                if isinstance(candidate_backend, int) and candidate_backend in snapshot.by_backend:
                    node = snapshot.by_backend[candidate_backend]
                    status = "ancestor"
                    break
        if node is None:
            return TargetResolution(
                "coordinate_fallback",
                coordinate,
                pixel,
                backend_node_id=backend_id,
            )
        actionable = _prefer_actionable_ancestor(snapshot, node)
        if actionable is not node:
            node = actionable
            status = "actionable_ancestor"
        return TargetResolution(
            status,
            coordinate,
            pixel,
            backend_node_id=node.backend_node_id,
            bid=node.bid,
            role=node.role,
            name=node.name,
        )
    except Exception:
        return TargetResolution("coordinate_fallback", coordinate, pixel)


def serialize_native_action(
    action: dict[str, Any],
    resolution: TargetResolution,
    *,
    width: int,
    height: int,
    model_kind: str = "webworld",
) -> str:
    kind = str(action["type"])
    pixel = resolution.pixel_coordinate
    bid = f"[{resolution.bid}]" if resolution.bid else None
    if kind in {"left_click", "toggle_check"}:
        return f"click({bid})" if bid else f"mouse_click({pixel[0]}, {pixel[1]})"
    if kind == "double_click":
        return f"mouse_dblclick({pixel[0]}, {pixel[1]})"
    if kind == "middle_click":
        return f"mouse_click({pixel[0]}, {pixel[1]}, button='middle')"
    if kind == "triple_click":
        return (
            f"[mouse_dblclick({pixel[0]}, {pixel[1]}), "
            f"mouse_click({pixel[0]}, {pixel[1]})]"
        )
    if kind == "right_click":
        return (
            f"click({bid}, button='right')"
            if bid
            else f"mouse_click({pixel[0]}, {pixel[1]}, button='right')"
        )
    if kind == "type_text":
        text = json.dumps(str(action.get("text", "")), ensure_ascii=False)
        clear = bool(action.get("clear", False))
        enter = bool(action.get("enter", False))
        steps: list[str] = []
        if resolution.normalized_coordinate is not None:
            if bid and clear:
                if model_kind == "webworld":
                    return f"fill({bid}, {text}, press_enter={enter})"
                steps.append(f"fill({bid}, {text})")
            else:
                steps.append(
                    f"click({bid})"
                    if bid
                    else f"mouse_click({pixel[0]}, {pixel[1]})"
                )
        if not (bid and clear):
            if clear:
                steps.extend(
                    ['keyboard_press("CTRL+A")', 'keyboard_press("BACKSPACE")']
                )
            steps.append(f"keyboard_type({text})")
        if enter:
            steps.append('keyboard_press("ENTER")')
        return steps[0] if len(steps) == 1 else f"[{', '.join(steps)}]"
    if kind == "scroll":
        dx = round(int(action.get("delta_x", 0)) * width / 999)
        dy = round(int(action.get("delta_y", 0)) * height / 999)
        return f"scroll({dx}, {dy})"
    if kind == "key_press":
        return f"keyboard_press({json.dumps(str(action.get('key', '')).upper())})"
    if kind == "key_down":
        return f"keyboard_down({json.dumps(str(action.get('key', '')).upper())})"
    if kind == "key_up":
        return f"keyboard_up({json.dumps(str(action.get('key', '')).upper())})"
    if kind == "hotkey":
        keys = "+".join(str(item).upper() for item in action.get("keys", []))
        return f"keyboard_press({json.dumps(keys)})"
    if kind == "navigate":
        return f"goto({json.dumps(str(action.get('url', '')), ensure_ascii=False)})"
    if kind == "go_back":
        return "go_back()"
    if kind == "go_forward":
        return "go_forward()"
    if kind == "refresh":
        return 'keyboard_press("CTRL+R")'
    if kind == "wait":
        return "noop(1500)"
    if kind == "select_option":
        option = json.dumps(str(action.get("option_text", "")), ensure_ascii=False)
        return (
            f"select_option({bid}, {option})"
            if bid
            else (
                f"[mouse_click({pixel[0]}, {pixel[1]}), keyboard_type({option}), "
                'keyboard_press("ENTER")]'
            )
        )
    if kind == "hover":
        return f"hover({bid})" if bid else f"mouse_move({pixel[0]}, {pixel[1]})"
    if kind == "move_to":
        return f"mouse_move({pixel[0]}, {pixel[1]})"
    if kind == "mouse_down":
        return f"mouse_down({pixel[0]}, {pixel[1]})"
    if kind == "mouse_up":
        return f"mouse_up({pixel[0]}, {pixel[1]})"
    if kind == "clear":
        if bid:
            return f"fill({bid}, \"\")"
        return '[keyboard_press("CTRL+A"), keyboard_press("BACKSPACE")]'
    if kind == "drag":
        start = _to_pixel(action["from_coordinate"], width, height)
        end = _to_pixel(action["to_coordinate"], width, height)
        return (
            f"[mouse_move({start[0]}, {start[1]}), mouse_down({start[0]}, {start[1]}), "
            f"mouse_move({end[0]}, {end[1]}), mouse_up({end[0]}, {end[1]})]"
        )
    if kind == "new_tab":
        return "tab_new()" if model_kind == "webworld" else "new_tab()"
    if kind == "page_close":
        return "tab_close()"
    if kind == "page_focus":
        index = action.get("page_index", action.get("page_id", 0))
        return f"tab_focus({json.dumps(index, ensure_ascii=False)})"
    if kind == "upload":
        return f"upload({json.dumps(str(action.get('path', '')), ensure_ascii=False)})"
    if kind in TERMINAL_ACTIONS:
        if kind == "finish":
            return f"send_msg_to_user({json.dumps(str(action.get('answer', '')), ensure_ascii=False)})"
        infeasible = "infeasible" if model_kind == "webworld" else "report_infeasible"
        return f"{infeasible}({json.dumps(str(action.get('reason', '')), ensure_ascii=False)})"
    raise ValueError(f"unsupported native WM action: {kind}")


def _wma_value(value: Any) -> str:
    """Render one WMA bracket argument without inventing a new action grammar."""

    return _clean_inline(value).replace("]", "\\]")


def serialize_wma_action(
    action: dict[str, Any],
    resolution: TargetResolution,
) -> str:
    """Translate a UnifiedAction into the official WMA action vocabulary.

    WMA has no coordinate action fallback. Callers must preserve unsupported
    candidates for selection/audit, but must not send a made-up instruction to
    the model.
    """

    kind = str(action["type"])
    bid = resolution.bid
    if kind in {"left_click", "toggle_check"}:
        if not bid:
            raise ValueError("WMA click requires a resolved bid")
        return f"click [{bid}]"
    if kind == "type_text":
        if not bid:
            raise ValueError("WMA type requires a resolved bid")
        text = _wma_value(action.get("text", ""))
        press_enter = 1 if bool(action.get("enter", False)) else 0
        return f"type [{bid}] [{text}] [{press_enter}]"
    if kind == "hover":
        if not bid:
            raise ValueError("WMA hover requires a resolved bid")
        return f"hover [{bid}]"
    if kind == "key_press":
        return f"press [{_wma_value(str(action.get('key', '')).upper())}]"
    if kind == "hotkey":
        keys = "+".join(str(item).upper() for item in action.get("keys", []))
        if not keys:
            raise ValueError("WMA press requires a non-empty key combination")
        return f"press [{_wma_value(keys)}]"
    if kind == "refresh":
        return "press [CTRL+R]"
    if kind == "scroll":
        dx = int(action.get("delta_x", 0))
        dy = int(action.get("delta_y", 0))
        if dx != 0 or dy == 0:
            raise ValueError("WMA only supports non-zero vertical scrolling")
        return "scroll [down]" if dy > 0 else "scroll [up]"
    if kind == "new_tab":
        return "new_tab"
    if kind == "page_focus":
        index = action.get("page_index", action.get("page_id"))
        if index is None:
            raise ValueError("WMA tab_focus requires a tab index")
        return f"tab_focus [{_wma_value(index)}]"
    if kind == "page_close":
        return "close_tab"
    if kind == "navigate":
        url = _wma_value(action.get("url", ""))
        if not url:
            raise ValueError("WMA goto requires a URL")
        return f"goto [{url}]"
    if kind == "go_back":
        return "go_back"
    if kind == "go_forward":
        return "go_forward"
    if kind == "finish":
        return f"stop [{_wma_value(action.get('answer', ''))}]"
    raise ValueError(f"unsupported WMA action: {kind}")


def serialize_dreamer_action(
    action: dict[str, Any],
    resolution: TargetResolution,
) -> str:
    kind = str(action["type"]).replace("_", " ")
    target = ""
    if resolution.name or resolution.role:
        role = resolution.role or "element"
        name = f' "{resolution.name}"' if resolution.name else ""
        target = f" the {role}{name}"
    coordinate = (
        f" at normalized coordinate {resolution.normalized_coordinate}"
        if resolution.normalized_coordinate is not None
        else ""
    )
    if action["type"] == "type_text":
        text = json.dumps(str(action.get("text", "")), ensure_ascii=False)
        steps = []
        if resolution.normalized_coordinate is not None:
            steps.append(f"click{target}{coordinate}")
        if action.get("clear", False):
            steps.append("clear the existing text")
        steps.append(f"type {text}")
        if action.get("enter", False):
            steps.append("press Enter")
        if resolution.normalized_coordinate is None:
            steps[-1 if not action.get("enter", False) else -2] += (
                " in the focused field"
            )
        return ", then ".join(steps)
    if action["type"] == "scroll":
        return f"scroll by ({action.get('delta_x', 0)}, {action.get('delta_y', 0)}) on the 0..999 screen grid"
    if action["type"] == "navigate":
        return f"navigate to {action.get('url', '')}"
    if action["type"] == "select_option":
        return f"select {json.dumps(str(action.get('option_text', '')), ensure_ascii=False)} from{target}{coordinate}"
    if action["type"] == "hotkey":
        return "press " + "+".join(str(item).upper() for item in action.get("keys", []))
    if action["type"] == "key_press":
        return f"press {str(action.get('key', '')).upper()}"
    if action["type"] == "drag":
        return f"drag from {action.get('from_coordinate')} to {action.get('to_coordinate')} on the 0..999 screen grid"
    return f"{kind}{target}{coordinate}".strip()


def parse_agentworld_output(raw: str) -> str:
    matches = re.findall(
        r"<predicted_observation>\s*(.*?)\s*</predicted_observation>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(matches) != 1:
        raise ValueError("AgentWorld must return exactly one closed predicted_observation block")
    outside = re.sub(
        r"<predicted_observation>\s*.*?\s*</predicted_observation>",
        "",
        raw,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if outside:
        raise ValueError("AgentWorld returned text outside predicted_observation")
    prediction = matches[0].strip()
    if not prediction:
        raise ValueError("AgentWorld predicted_observation is empty")
    return prediction


def parse_webworld_output(raw: str) -> str:
    prediction = raw.strip()
    if prediction.startswith("```") or not prediction:
        raise ValueError("WebWorld must return a non-empty unfenced page state")
    reason_match = re.match(
        r"<(?:reason|think)>\s*(.*?)\s*</(?:reason|think)>\s*(.*)",
        prediction,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if reason_match:
        prediction = reason_match.group(2).strip()
    first = prediction.splitlines()[0].strip()
    if re.match(r"^RootWebArea(?:\s|$)", first) is None:
        raise ValueError("WebWorld page state must begin with RootWebArea")
    return prediction


def extract_webworld_reason(raw: str) -> str:
    match = re.match(
        r"\s*<(?:reason|think)>\s*(.*?)\s*</(?:reason|think)>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def parse_wma_output(raw: str) -> str:
    text = raw.strip()
    if not text or text.startswith("```"):
        raise ValueError("WMA must return a non-empty unfenced transition description")
    match = re.search(r"\[Next State\]\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if match is None or not match.group(1).strip():
        raise ValueError("WMA output must contain a non-empty [Next State] section")
    return match.group(1).strip()


def extract_wma_rationale(raw: str) -> str:
    text = raw.strip()
    next_state = re.search(r"\[Next State\]", text, flags=re.IGNORECASE)
    if next_state is None:
        return ""
    rationale = text[: next_state.start()].strip()
    rationale = re.sub(r"^\[Rationale\]\s*", "", rationale, flags=re.IGNORECASE)
    return rationale


def parse_dreamer_output(raw: str) -> str:
    prediction = raw.strip()
    if not prediction or prediction.startswith("```"):
        raise ValueError("Dreamer must return a non-empty unfenced state prediction")
    return prediction


OUTPUT_PARSERS: dict[str, Callable[[str], str]] = {
    "agentworld": parse_agentworld_output,
    "webworld": parse_webworld_output,
    "webworld32b": parse_webworld_output,
    "wma": parse_wma_output,
    "dreamer": parse_dreamer_output,
}


class NativeWorldModelClient:
    """One-step native world-model predictor with aligned candidate outputs."""

    def __init__(
        self,
        *,
        kind: str,
        endpoint: str,
        model: str,
        system_prompt: str,
        max_tokens: int,
        context_tokens: int,
        timeout_seconds: int = 300,
        tokenizer_path: str | Path | None = None,
        max_concurrency: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking_policy: str = "disabled",
        wma_input_contract: str = "official_contextual",
    ) -> None:
        if kind not in WORLD_MODEL_KINDS - {"none"}:
            raise ValueError(f"unsupported world-model kind: {kind}")
        if max_tokens < 0:
            raise ValueError("world-model max tokens must be non-negative")
        if max_tokens > 0 and context_tokens <= max_tokens + 2048:
            raise ValueError("world-model context must leave at least 2048 prompt tokens")
        if wma_input_contract not in WMA_INPUT_CONTRACTS:
            raise ValueError(
                f"unsupported WMA input contract: {wma_input_contract}"
            )
        if kind != "wma" and wma_input_contract != "official_contextual":
            raise ValueError("WMA input contracts can only be used with kind=wma")
        self.kind = kind
        self.wma_input_contract = wma_input_contract
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.context_tokens = context_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        if thinking_policy not in {"disabled", "enabled"}:
            raise ValueError(f"unsupported thinking policy: {thinking_policy}")
        self.thinking_policy = thinking_policy
        endpoint_list = [e.strip().rstrip("/") for e in endpoint.split(",") if e.strip()]
        self.clients = []
        for ep in endpoint_list:
            base_url = ep
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            self.clients.append(AsyncOpenAI(
                base_url=base_url,
                api_key="none",
                timeout=timeout_seconds,
            ))
        self._client_round_robin = 0
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.tokenizer = None
        self.processor = None
        self.reasoning_logit_bias: dict[str, float] | None = None
        if tokenizer_path is not None:
            from transformers import AutoProcessor, AutoTokenizer

            if kind == "dreamer":
                self.processor = AutoProcessor.from_pretrained(
                    str(tokenizer_path),
                    trust_remote_code=True,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path),
                    trust_remote_code=True,
                )
            if kind == "webworld":
                # Ban the first token of <think>/<reason> so WebWorld outputs only the page state.
                banned: dict[str, float] = {}
                for marker in ("<reason>", "<think>"):
                    marker_ids = self.tokenizer.encode(
                        marker,
                        add_special_tokens=False,
                    )
                    if marker_ids:
                        banned[str(marker_ids[0])] = -100.0
                self.reasoning_logit_bias = banned or None

    async def close(self) -> None:
        for client in self.clients:
            await client.close()

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            # Tests and mock endpoints may omit a tokenizer. Production CLI
            # requires tokenizer_path for text world models.
            return max(1, len(text) // 4)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_message_tokens(
        self,
        *,
        state: str,
        serialized_action: str,
        url: str = "",
        objective: str = "",
        previous_action: str = "",
    ) -> int:
        messages = self._messages(
            state=state,
            screenshot=b"",
            serialized_action=serialized_action,
            url=url,
            objective=objective,
            previous_action=previous_action,
        )
        return self.count_text_messages(messages)

    def count_text_messages(self, messages: list[dict[str, Any]]) -> int:
        if self.tokenizer is None:
            return self.count_tokens(json.dumps(messages, ensure_ascii=False))
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=self.thinking_policy == "enabled",
        )
        return self.count_tokens(rendered)

    def count_dreamer_message_tokens(
        self,
        *,
        screenshot: bytes,
        serialized_action: str,
        url: str = "",
        objective: str = "",
        previous_action: str = "",
    ) -> int:
        if self.processor is None:
            raise RuntimeError("Dreamer native-window policy requires its processor")
        from PIL import Image

        with Image.open(io.BytesIO(screenshot)) as source:
            image = source.convert("RGB")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Below is the current screenshot. Predict the visible webpage state "
                            f"after this action: {serialized_action}"
                        ),
                    },
                ],
            },
        ]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.processor(
            text=[rendered],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        return int(encoded.input_ids.shape[-1])

    def output_budget(self, input_tokens: int) -> int:
        """Use a configured cap or every token left in the native context."""

        available = (
            self.context_tokens
            - input_tokens
            - NATIVE_CONTEXT_PROTOCOL_MARGIN
        )
        if available <= 0:
            raise ValueError(
                f"input uses {input_tokens} tokens, exceeding native window "
                f"{self.context_tokens}"
            )
        return min(self.max_tokens, available) if self.max_tokens > 0 else available

    def adjust_output_budget_from_context_error(
        self,
        error: Exception | str,
        current_budget: int,
    ) -> int:
        match = re.search(
            r"maximum context length is\s+(\d+).*?"
            r"requested\s+(\d+)\s+output tokens.*?"
            r"prompt contains at least\s+(\d+)\s+input tokens",
            str(error),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return current_budget
        native_context = int(match.group(1))
        server_prompt = int(match.group(3))
        adjusted = native_context - server_prompt - NATIVE_CONTEXT_PROTOCOL_MARGIN
        return min(current_budget, adjusted) if adjusted > 0 else current_budget

    def _messages(
        self,
        *,
        state: str,
        screenshot: bytes,
        serialized_action: str,
        url: str = "",
        objective: str = "",
        previous_action: str = "",
    ) -> list[dict[str, Any]]:
        if self.kind == "agentworld":
            user = (
                "### Turn 1\n"
                "**Current State:**\n"
                f"{state}\n\n"
                "**Action:**\n"
                f"```text\n{serialized_action}\n```"
            )
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ]
        if self.kind in WEBWORLD_KINDS:
            user = (
                f"Initial Page State:\n{state}\n\n"
                f"First Action: '{serialized_action}'\n\n"
                "Next Page State:"
            )
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ]
        if self.kind == "wma":
            input_contract = getattr(
                self, "wma_input_contract", "official_contextual"
            )
            if input_contract == "axtree_action_only":
                user = (
                    f"CURRENT OBSERVATION: {state}\n"
                    f"CURRENT ACTION: {serialized_action}"
                )
            elif input_contract == "official_contextual":
                user = (
                    f"URL: {url}\n"
                    f"OBJECTIVE: {objective}\n"
                    f"PREVIOUS ACTION: {previous_action or 'None'}\n"
                    f"CURRENT OBSERVATION: {state}\n"
                    f"CURRENT ACTION: {serialized_action}"
                )
            else:
                raise ValueError(
                    f"unsupported WMA input contract: {input_contract}"
                )
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ]
        encoded = base64.b64encode(screenshot).decode("ascii")
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Below is the current screenshot. Predict the visible webpage state "
                            f"after this action: {serialized_action}"
                        ),
                    },
                ],
            },
        ]

    def _format_retry_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        current_messages: list[dict[str, Any]],
        raw_output: str,
        validation_error: Exception,
    ) -> tuple[list[dict[str, Any]], str, int | None]:
        """Build a bounded format-correction turn without truncating state.

        Replaying a malformed full-state prediction can itself consume the
        remaining native window.  Keep the normal assistant-replay correction
        when it fits.  Otherwise retry from the unchanged full state with a
        compact format instruction; the model output is still retained in the
        audit record, but is not duplicated into the next request.
        """

        correction_text = (
            "Correct only the output format and return the complete predicted "
            f"next state. Validation error: {validation_error}"
        )
        replay_messages = [
            *current_messages,
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": correction_text},
        ]
        if self.kind not in TEXT_WORLD_MODELS:
            return replay_messages, "assistant_replay", None

        replay_tokens = self.count_text_messages(replay_messages)
        if replay_tokens + NATIVE_CONTEXT_PROTOCOL_MARGIN < self.context_tokens:
            return replay_messages, "assistant_replay", replay_tokens

        compact_messages = [dict(message) for message in base_messages]
        original_user = str(compact_messages[-1]["content"])
        compact_messages[-1] = {
            **compact_messages[-1],
            "content": (
                f"{original_user}\n\n"
                f"FORMAT CORRECTION: {correction_text} "
                "Regenerate the prediction from the unchanged full state."
            ),
        }
        compact_tokens = self.count_text_messages(compact_messages)
        return compact_messages, "compact_regeneration", compact_tokens

    async def _predict_one(
        self,
        *,
        candidate_index: int,
        action: dict[str, Any],
        state: str,
        state_tokens: int | None,
        screenshot: bytes,
        serialized_action: str,
        resolution: TargetResolution,
        request_max_tokens: int,
        url: str = "",
        objective: str = "",
        previous_action: str = "",
    ) -> NativeWorldModelPrediction:
        parser = OUTPUT_PARSERS[self.kind]
        base_messages = self._messages(
            state=state,
            screenshot=screenshot,
            serialized_action=serialized_action,
            url=url,
            objective=objective,
            previous_action=previous_action,
        )
        messages = list(base_messages)
        started = time.monotonic()
        last_error: Exception | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(2):
            try:
                client = self.clients[self._client_round_robin % len(self.clients)]
                self._client_round_robin += 1
                async with self.semaphore:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=request_max_tokens,
                        logit_bias=(
                            getattr(self, "reasoning_logit_bias", None)
                            if self.kind == "webworld"
                            else None
                        ),
                        extra_body={
                            "chat_template_kwargs": {
                                "enable_thinking": self.thinking_policy == "enabled"
                            },
                            **(
                                {"bad_words": ["<think>", "<reason>"]}
                                if self.kind == "webworld"
                                else {}
                            ),
                            "add_generation_prompt": True,
                            "continue_final_message": False,
                        },
                    )
                choice = response.choices[0]
                raw = choice.message.content or ""
                server_reasoning = (
                    getattr(choice.message, "reasoning_content", None) or ""
                )
                embedded_reasoning = extract_webworld_reason(raw)
                content_validation_warning: str | None = None
                if self.kind == "webworld":
                    warnings: list[str] = []
                    if choice.finish_reason == "length":
                        warnings.append(
                            f"finish_reason=length at {request_max_tokens} tokens"
                        )
                    try:
                        parser(raw)
                    except Exception as exc:
                        warnings.append(f"{type(exc).__name__}: {exc}")
                    if self.thinking_policy == "disabled" and (
                        server_reasoning.strip()
                        or embedded_reasoning
                        or re.search(r"<(?:think|reason)(?:ing)?>", raw, re.IGNORECASE)
                    ):
                        warnings.append("reasoning content emitted while disabled")
                    prediction = raw
                    content_validation_warning = "; ".join(warnings) or None
                else:
                    try:
                        if self.thinking_policy == "disabled" and self.kind != "wma" and (
                            server_reasoning.strip()
                            or embedded_reasoning
                            or re.search(r"<(?:think|reason)(?:ing)?>", raw, re.IGNORECASE)
                        ):
                            raise ValueError(
                                f"{self.kind} emitted reasoning although thinking is disabled"
                            )
                        if choice.finish_reason == "length":
                            warnings = [
                                f"finish_reason=length at {request_max_tokens} tokens"
                            ]
                            try:
                                prediction = parser(raw)
                            except Exception as exc:
                                warnings.append(f"{type(exc).__name__}: {exc}")
                                prediction = raw.strip()
                            content_validation_warning = "; ".join(warnings)
                        else:
                            prediction = parser(raw)
                    except Exception as exc:
                        last_error = exc
                        attempts.append(
                            {
                                "attempt": attempt,
                                "raw_output": raw,
                                "finish_reason": choice.finish_reason,
                                "validation_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if attempt == 0:
                            messages, retry_mode, retry_input_tokens = (
                                self._format_retry_messages(
                                    base_messages=base_messages,
                                    current_messages=messages,
                                    raw_output=raw,
                                    validation_error=exc,
                                )
                            )
                            attempts[-1]["retry_context_policy"] = retry_mode
                            if retry_input_tokens is not None:
                                attempts[-1]["retry_input_tokens"] = retry_input_tokens
                                request_max_tokens = self.output_budget(retry_input_tokens)
                            continue
                        raise
                usage = response.usage
                reasoning = getattr(choice.message, "reasoning_content", None) or ""
                if not reasoning and self.kind in WEBWORLD_KINDS:
                    reasoning = extract_webworld_reason(raw)
                if not reasoning and self.kind == "wma":
                    reasoning = extract_wma_rationale(raw)
                return NativeWorldModelPrediction(
                    candidate_index=candidate_index,
                    unified_action=dict(action),
                    serialized_action=serialized_action,
                    observation_format=(
                        "transition_description"
                        if self.kind == "wma"
                        else (
                            "a11y_dom"
                            if self.kind in TEXT_WORLD_MODELS
                            else "visual_state_change"
                        )
                    ),
                    predicted_observation=prediction,
                    raw_output=raw,
                    reasoning_content=reasoning,
                    target_resolution=resolution,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                    input_state_tokens=state_tokens,
                    finish_reason=choice.finish_reason,
                    latency_seconds=time.monotonic() - started,
                    format_retry_count=attempt,
                    output_limit_hit=choice.finish_reason == "length",
                    context_window_tokens=self.context_tokens,
                    context_utilization=(
                        float(getattr(usage, "prompt_tokens", 0) or 0)
                        / float(self.context_tokens)
                    ),
                    content_validation_warning=content_validation_warning,
                )
            except Exception as exc:
                last_error = exc
                if not attempts or attempts[-1].get("attempt") != attempt:
                    attempts.append(
                        {
                            "attempt": attempt,
                            "request_max_tokens": request_max_tokens,
                            "request_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                adjusted_budget = self.adjust_output_budget_from_context_error(
                    exc,
                    request_max_tokens,
                )
                if adjusted_budget < request_max_tokens:
                    request_max_tokens = adjusted_budget
                if attempt == 0:
                    await asyncio.sleep(1)
        raise NativeWorldModelError(
            f"{self.kind} prediction failed for candidate {candidate_index}: {last_error}",
            audit_payload={
                "component": self.kind,
                "candidate_index": candidate_index,
                "unified_action": action,
                "serialized_action": serialized_action,
                "target_resolution": resolution.to_dict(),
                "attempts": attempts,
            },
        )


    def serialize_candidates_sync(
        self,
        *,
        page: Any,
        screenshot: bytes,
        candidates: Sequence[dict[str, Any]],
        candidate_indices: Sequence[int] | None = None,
        width: int,
        height: int,
        url: str = "",
        objective: str = "",
        previous_action: str = "",
    ) -> tuple[list[dict[str, Any]], dict[int, NativeWorldModelPrediction]]:
        """Capture AXTree and serialize actions on the Playwright owner thread."""

        indices = list(candidate_indices or range(len(candidates)))
        if len(indices) != len(candidates) or len(set(indices)) != len(indices):
            raise ValueError("native WM candidate indices must be unique and aligned")
        session, snapshot = _capture_a11y_dom_sync(
            page,
            viewport_width=width,
            viewport_height=height,
        )
        prepared: list[dict[str, Any]] = []
        skipped: dict[int, NativeWorldModelPrediction] = {}
        try:
            for index, candidate in zip(indices, candidates):
                action = candidate["unified_action"]
                resolution = resolve_action_target_sync(
                    session,
                    snapshot,
                    action,
                    width=width,
                    height=height,
                )
                try:
                    serialized = serialize_wma_action(action, resolution) if self.kind == "wma" else serialize_native_action(
                        action,
                        resolution,
                        width=width,
                        height=height,
                        model_kind=self.kind,
                    )
                except ValueError:
                    if self.kind != "wma":
                        raise
                    skipped[index] = NativeWorldModelPrediction(
                        candidate_index=index,
                        unified_action=dict(action),
                        serialized_action="",
                        observation_format="transition_description",
                        predicted_observation="",
                        raw_output="",
                        reasoning_content="",
                        target_resolution=resolution,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        input_state_tokens=None,
                        finish_reason=None,
                        latency_seconds=0.0,
                        format_retry_count=0,
                        output_limit_hit=False,
                        context_window_tokens=self.context_tokens,
                        context_utilization=None,
                        wm_called=False,
                        skip_reason="unsupported_native_action",
                    )
                    continue
                state, state_tokens = snapshot.render_full(
                    target_bid=resolution.bid,
                    count_tokens=self.count_tokens,
                )
                input_tokens = self.count_message_tokens(
                    state=state,
                    serialized_action=serialized,
                    url=url,
                    objective=objective,
                    previous_action=previous_action,
                )
                request_max_tokens = self.output_budget(input_tokens)
                prepared.append(
                    {
                        "candidate_index": index,
                        "action": action,
                        "state": state,
                        "state_tokens": state_tokens,
                        "screenshot": screenshot,
                        "serialized_action": serialized,
                        "resolution": resolution,
                        "request_max_tokens": request_max_tokens,
                        "url": url,
                        "objective": objective,
                        "previous_action": previous_action,
                    }
                )
        finally:
            detach = getattr(session, "detach", None)
            if callable(detach):
                detach()
        return prepared, skipped

    async def predict_serialized(
        self,
        prepared: Sequence[dict[str, Any]],
        skipped: dict[int, NativeWorldModelPrediction],
        indices: Sequence[int],
    ) -> list[NativeWorldModelPrediction]:
        called = await asyncio.gather(
            *[
                self._predict_one(
                    candidate_index=item["candidate_index"],
                    action=item["action"],
                    state=item["state"],
                    state_tokens=item["state_tokens"],
                    screenshot=item["screenshot"],
                    serialized_action=item["serialized_action"],
                    resolution=item["resolution"],
                    request_max_tokens=item["request_max_tokens"],
                    url=item["url"],
                    objective=item["objective"],
                    previous_action=item["previous_action"],
                )
                for item in prepared
            ]
        )
        by_index = {item.candidate_index: item for item in called}
        by_index.update(skipped)
        return [by_index[index] for index in indices]

    async def predict_candidates(
        self,
        *,
        page: Page,
        screenshot: bytes,
        candidates: Sequence[dict[str, Any]],
        candidate_indices: Sequence[int] | None = None,
        width: int,
        height: int,
        url: str = "",
        objective: str = "",
        previous_action: str = "",
    ) -> list[NativeWorldModelPrediction]:
        indices = list(candidate_indices or range(len(candidates)))
        if len(indices) != len(candidates) or len(set(indices)) != len(indices):
            raise ValueError("native WM candidate indices must be unique and aligned")
        session, snapshot = await capture_a11y_dom(
            page,
            viewport_width=width,
            viewport_height=height,
        )
        try:
            prepared: list[Any] = []
            skipped: dict[int, NativeWorldModelPrediction] = {}
            for index, candidate in zip(indices, candidates):
                action = candidate["unified_action"]
                resolution = await resolve_action_target(
                    session,
                    snapshot,
                    action,
                    width=width,
                    height=height,
                )
                if self.kind == "dreamer":
                    serialized = serialize_dreamer_action(action, resolution)
                    state = ""
                    input_tokens = self.count_dreamer_message_tokens(
                        screenshot=screenshot,
                        serialized_action=serialized,
                    )
                    state_tokens = input_tokens
                else:
                    try:
                        serialized = (
                            serialize_wma_action(action, resolution)
                            if self.kind == "wma"
                            else serialize_native_action(
                                action,
                                resolution,
                                width=width,
                                height=height,
                                model_kind=self.kind,
                            )
                        )
                    except ValueError:
                        if self.kind != "wma":
                            raise
                        skipped[index] = NativeWorldModelPrediction(
                            candidate_index=index,
                            unified_action=dict(action),
                            serialized_action="",
                            observation_format="transition_description",
                            predicted_observation="",
                            raw_output="",
                            reasoning_content="",
                            target_resolution=resolution,
                            prompt_tokens=None,
                            completion_tokens=None,
                            total_tokens=None,
                            input_state_tokens=None,
                            finish_reason=None,
                            latency_seconds=0.0,
                            format_retry_count=0,
                            output_limit_hit=False,
                            context_window_tokens=self.context_tokens,
                            context_utilization=None,
                            wm_called=False,
                            skip_reason="unsupported_native_action",
                        )
                        continue
                    state, state_tokens = snapshot.render_full(
                        target_bid=resolution.bid,
                        count_tokens=self.count_tokens,
                    )
                    input_tokens = self.count_message_tokens(
                        state=state,
                        serialized_action=serialized,
                        url=url,
                        objective=objective,
                        previous_action=previous_action,
                    )
                try:
                    request_max_tokens = self.output_budget(input_tokens)
                except ValueError as exc:
                    raise NativeWorldModelError(
                        str(exc),
                        audit_payload={
                            "component": self.kind,
                            "error_kind": "context_overflow",
                            "candidate_index": index,
                            "input_tokens": input_tokens,
                            "configured_output_tokens": self.max_tokens,
                            "native_context_tokens": self.context_tokens,
                            "context_utilization": input_tokens / self.context_tokens,
                            "input_policy": "full_current_state_no_truncation",
                        },
                    ) from exc
                if self.max_tokens > 0 and input_tokens + self.max_tokens > self.context_tokens:
                        raise NativeWorldModelError(
                            (
                                f"{self.kind} full input requires {input_tokens} input + "
                                f"{self.max_tokens} output tokens, exceeding native "
                                f"window {self.context_tokens}"
                            ),
                            audit_payload={
                                "component": self.kind,
                                "error_kind": "context_overflow",
                                "candidate_index": index,
                                "input_tokens": input_tokens,
                                "reserved_output_tokens": self.max_tokens,
                                "native_context_tokens": self.context_tokens,
                                "context_utilization": input_tokens / self.context_tokens,
                                "input_policy": "full_current_viewport_no_truncation",
                            },
                        )
                prepared.append(
                    self._predict_one(
                        candidate_index=index,
                        action=action,
                        state=state,
                        state_tokens=state_tokens,
                        screenshot=screenshot,
                        serialized_action=serialized,
                        resolution=resolution,
                        request_max_tokens=request_max_tokens,
                        url=url,
                        objective=objective,
                        previous_action=previous_action,
                    )
                )
            called_predictions = await asyncio.gather(*prepared)
        finally:
            await session.detach()
        by_index = {item.candidate_index: item for item in called_predictions}
        by_index.update(skipped)
        predictions = [by_index[index] for index in indices]
        if [item.candidate_index for item in predictions] != indices:
            raise RuntimeError("native WM candidate prediction order diverged")
        return predictions


def profile_defaults(kind: str) -> tuple[int, int]:
    # Returns (context window, output cap); evaluation runs set the output cap to 6000.
    if kind in {"agentworld", "webworld", "dreamer"}:
        return 60_000, 0
    if kind == "wma":
        return 16_384, 0
    if kind == "webworld32b":
        return 40_960, 4_096
    raise ValueError(f"unsupported world-model kind: {kind}")
