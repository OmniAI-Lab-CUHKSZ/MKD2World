"""Read-only prompt registry backed by source-controlled English templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROMPT_FILENAMES = {
    "gui_agent_action_best": "gui_agent_action_best.md",
    "gui_agent_action_candidates": "gui_agent_action_candidates.md",
    "gui_agent_vanilla": "gui_agent_vanilla.md",
    "gui_agent_vanilla_md": "gui_agent_vanilla_md.md",
    "gui_agent_subgoal_best": "gui_agent_subgoal_best.md",
    "gui_agent_subgoal_candidates": "gui_agent_subgoal_candidates.md",
    "memory_trajectory": "memory_trajectory.md",
    "subgoal_completion_manager": "subgoal_completion_manager.md",
    "world_global": "world_global.md",
    "world_global_with_memory": "world_global_with_memory.md",
    "world_local": "world_local.md",
    "world_local_example": "world_local_example.md",
    "world_local_with_memory": "world_local_with_memory.md",
    "localwm_cur": "localwm_cur.md",
    "localwm_next": "localwm_next.md",
    "localwm_delta": "localwm_delta.md",
    "molmo_trajectory_annotation": "molmo_trajectory_annotation.md",
}


def default_prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class PromptRegistry:
    prompt_dir: Path

    @classmethod
    def from_project(cls, root: Path | None = None) -> "PromptRegistry":
        if root is not None:
            source_prompt_dir = Path(root) / "mkd2world" / "prompts" / "templates"
            if source_prompt_dir.exists():
                return cls(source_prompt_dir)
        return cls(default_prompt_dir())

    def keys(self) -> Iterable[str]:
        return PROMPT_FILENAMES.keys()

    def path_for(self, key: str) -> Path:
        try:
            filename = PROMPT_FILENAMES[key]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt key: {key}") from exc
        if key in {"localwm_cur", "localwm_next", "localwm_delta"}:
            training_path = self.prompt_dir.parent / "templatesV3" / "train_localwm" / PROMPT_FILENAMES[key]
            if training_path.is_file():
                return training_path
        path = self.prompt_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def load(self, key: str) -> str:
        return self.path_for(key).read_text(encoding="utf-8")

    def load_pair(self, base_key: str, *, with_memory: bool) -> str:
        key = f"{base_key}_with_memory" if with_memory else base_key
        return self.load(key)
