from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "eval"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

import run_online_mind2web_mkd2world_vllm as base
from mkd2world.contracts import LocalPrediction
from mkd2world.experiments.tsm_component_mask import LEVELS, mask_tsm

BEST_FIELDS = ["candidate_index", "unified_action", "action_intent"]
BEST_RESPONSE_FORMAT = copy.deepcopy(base.core.BEST_ACTION_RESPONSE_FORMAT)
_best = BEST_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]["best_action"]
_best["required"] = BEST_FIELDS
_best["properties"] = {key: _best["properties"][key] for key in BEST_FIELDS}


def _fill_best_placeholders(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    best = payload.get("best_action") if isinstance(payload, dict) else None
    if not isinstance(best, dict):
        return raw
    best.setdefault("evidence", "")
    best.setdefault("confidence", 0)
    return json.dumps(payload, ensure_ascii=False)


LEVEL = os.environ.get("MKD2WORLD_TSM_MASK_LEVEL", "")
if LEVEL not in LEVELS:
    raise SystemExit(f"MKD2WORLD_TSM_MASK_LEVEL must be one of {LEVELS}, got {LEVEL!r}")


class MaskedVllmLocalWMAgent(base.VllmLocalWMAgent):
    async def _http_predict_one(self, replica_index: int, current_state: Any, candidate: Any) -> LocalPrediction:
        prediction = await super()._http_predict_one(replica_index, current_state, candidate)
        return dataclasses.replace(
            prediction,
            predicted_markdown=mask_tsm(prediction.predicted_markdown, LEVEL),
        )

    async def _chat_json(self, *, system_prompt: str, **kwargs: Any) -> tuple[Any, str, dict[str, Any]]:
        if system_prompt == self.best_prompt:
            parser = kwargs["parser"]
            kwargs["response_format"] = BEST_RESPONSE_FORMAT
            kwargs["parser"] = lambda raw: parser(_fill_best_placeholders(raw))
        try:
            return await super()._chat_json(system_prompt=system_prompt, **kwargs)
        except RuntimeError as exc:
            if system_prompt != self.best_prompt:
                raise
            selection = {
                "selected_index": 0,
                "reported_index": None,
                "index_action_consistent": None,
                "matching_indices": [],
            }
            return selection, "", {"best_action_fallback": "first_candidate", "error": str(exc)[:2000]}


if __name__ == "__main__":
    print(f"MKD2WORLD_TSM_MASK_LEVEL={LEVEL}", flush=True)
    base.VllmLocalWMAgent = MaskedVllmLocalWMAgent
    base.main()
