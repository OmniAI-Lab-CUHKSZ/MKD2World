"""Training and evaluation helpers for the Local World Model."""

from __future__ import annotations


__all__ = [
    "MdElement",
    "TransitionSample",
    "adapt_transition_record",
    "evaluate_jsonl",
    "evaluate_prediction",
    "parse_md_elements",
    "reward_from_markdown",
]


def __getattr__(name: str):
    if name in {"TransitionSample", "adapt_transition_record"}:
        from mkd2world.training.datasets import TransitionSample, adapt_transition_record

        return {"TransitionSample": TransitionSample, "adapt_transition_record": adapt_transition_record}[name]
    if name in {"evaluate_jsonl", "evaluate_prediction"}:
        from mkd2world.training.evaluation import evaluate_jsonl, evaluate_prediction

        return {"evaluate_jsonl": evaluate_jsonl, "evaluate_prediction": evaluate_prediction}[name]
    if name in {"MdElement", "parse_md_elements"}:
        from mkd2world.training.markdown import MdElement, parse_md_elements

        return {"MdElement": MdElement, "parse_md_elements": parse_md_elements}[name]
    if name == "reward_from_markdown":
        from mkd2world.training.reward import reward_from_markdown

        return reward_from_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
