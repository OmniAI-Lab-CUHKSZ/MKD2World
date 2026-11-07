"""LoRA SFT entrypoint for the Local World Model."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from mkd2world.actions import UnifiedAction
from mkd2world.config import apply_config_defaults, load_run_config, provided_cli_dests
from mkd2world.prompts import PromptRegistry
from mkd2world.training.delta import (
    DeltaFormatError,
    DeltaReplayError,
    apply_delta_flat,
    parse_delta,
)
from mkd2world.training.markdown import build_assistant_output, write_jsonl
from mkd2world.training.prompts import (
    apply_chat_template,
    build_local_world_model_messages,
    local_world_model_target,
    resolve_local_world_model_prompt_key,
)
from mkd2world.training.reward import SemanticEmbedder
from mkd2world.world_model import split_local_prediction


DATASET_TEXT_FIELDS = (
    "id",
    "task_type",
    "current_image_path",
    "current_md",
    "target_next_md",
    "target_next_md_clean",
    "target_delta_md_clean",
    "assistant_output",
    "action_analysis",
)


def import_training_dependencies():
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments
    except Exception as exc:  # pragma: no cover - environment diagnostic path.
        raise RuntimeError(
            "SFT requires CUDA torch, datasets, peft, Pillow, transformers, and a Qwen-VL compatible processor "
            "in the training environment."
        ) from exc
    return torch, load_dataset, LoraConfig, PeftModel, get_peft_model, Image, AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments


def optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _json_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _strip_action_raw(value: object) -> object:
    if isinstance(value, dict):
        stripped = dict(value)
        stripped.pop("raw", None)
        return stripped
    return value


def _parse_action_json(value: str, *, field_name: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc


def _canonical_action_payload(value: object) -> dict[str, Any]:
    stripped = _strip_action_raw(value)
    if not isinstance(stripped, dict):
        raise ValueError("candidate action must be a JSON object")
    return UnifiedAction.from_dict(stripped).to_dict()


def _candidate_action_payload(row: dict[str, Any]) -> object:
    action_json = row.get("candidate_action_json")
    if action_json is not None and action_json != "":
        if isinstance(action_json, str):
            action_json = _parse_action_json(action_json, field_name="candidate_action_json")
        return _canonical_action_payload(action_json)

    action = row.get("candidate_action")
    if action is None:
        action = row.get("action")
    if action is not None:
        return _canonical_action_payload(action)
    raise ValueError("training row requires candidate_action_json or candidate_action")


def normalize_row_for_dataset(row: dict[str, Any]) -> dict[str, str]:
    normalized = {field: _as_text(row.get(field)) for field in DATASET_TEXT_FIELDS}
    task_type = normalized["task_type"]
    if task_type == "describe" and not any(
        row.get(field) not in (None, "")
        for field in ("candidate_action_json", "candidate_action", "action")
    ):
        normalized["candidate_action_json"] = _json_text({"type": "wait"})
    else:
        normalized["candidate_action_json"] = _json_text(_candidate_action_payload(row))
    normalized["meta_json"] = _json_text(row.get("meta"))
    normalized["assistant_output"] = ""
    if normalized["task_type"] == "forward_delta" and normalized["target_delta_md_clean"]:
        assistant_target = normalized["target_delta_md_clean"]
    else:
        assistant_target = normalized["target_next_md"]
    if assistant_target:
        normalized["assistant_output"] = build_assistant_output(
            normalized["action_analysis"],
            assistant_target,
        )
    return normalized


def expanded_qwen35_lora_target_modules(model: Any) -> list[str]:
    """Return exact Qwen3.5 language/vision linear module names for Expanded LoRA."""
    language_suffixes = (
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".o_proj",
        ".in_proj_qkv",
        ".in_proj_z",
        ".in_proj_a",
        ".in_proj_b",
        ".out_proj",
        ".gate_proj",
        ".up_proj",
        ".down_proj",
    )
    vision_suffixes = (
        ".attn.qkv",
        ".attn.proj",
        ".mlp.linear_fc1",
        ".mlp.linear_fc2",
        ".merger.linear_fc1",
        ".merger.linear_fc2",
    )
    selected: list[str] = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2:
            continue
        is_language = "language_model.layers." in name and name.endswith(language_suffixes)
        is_vision = ".visual." in f".{name}." and name.endswith(vision_suffixes)
        if is_language or is_vision:
            selected.append(name)
    selected = sorted(set(selected))
    if len(selected) != 358:
        raise ValueError(
            "qwen35_expanded LoRA profile expected 358 exact linear modules, "
            f"found {len(selected)}. Refusing to train against an unknown model structure."
        )
    return selected


def resolve_lora_target_modules(model: Any, specification: str) -> list[str]:
    if specification.strip() == "qwen35_expanded":
        return expanded_qwen35_lora_target_modules(model)
    modules = [item.strip() for item in specification.split(",") if item.strip()]
    if not modules:
        raise ValueError("--target-modules resolved to an empty set")
    return modules


def build_length_filter_decision(
    normalized: dict[str, str],
    *,
    processor: Any | None,
    max_training_tokens: int | None,
    max_input_tokens: int | None,
    max_output_tokens: int | None,
    prompt_key: str = "world_local",
) -> dict[str, Any] | None:
    if processor is None:
        return None
    prompt_messages = build_local_world_model_messages(normalized, include_output=False, prompt_key=prompt_key)
    training_messages = build_local_world_model_messages(normalized, include_output=True, prompt_key=prompt_key)
    prompt_text = apply_chat_template(processor, prompt_messages, tokenize=False, add_generation_prompt=True)
    training_text = apply_chat_template(processor, training_messages, tokenize=False, add_generation_prompt=False)
    prompt_tokens = token_count(processor.tokenizer, prompt_text)
    training_tokens = token_count(processor.tokenizer, training_text)
    assistant_target = build_assistant_output(
        normalized.get("action_analysis", ""),
        local_world_model_target(normalized, prompt_key=prompt_key),
    )
    assistant_tokens = token_count(processor.tokenizer, assistant_target)
    reasons = []
    if max_input_tokens is not None and max_input_tokens > 0 and prompt_tokens > max_input_tokens:
        reasons.append("input_tokens_exceed_limit")
    if max_output_tokens is not None and max_output_tokens > 0 and assistant_tokens > max_output_tokens:
        reasons.append("output_tokens_exceed_limit")
    if max_training_tokens is not None and max_training_tokens > 0 and training_tokens > max_training_tokens:
        reasons.append("training_tokens_exceed_limit")
    if not reasons:
        return None
    return {
        "reason": "+".join(reasons),
        "id": normalized.get("id", ""),
        "prompt_tokens": prompt_tokens,
        "assistant_tokens": assistant_tokens,
        "training_tokens": training_tokens,
        "max_training_tokens": max_training_tokens,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "current_md_chars": len(normalized.get("current_md", "")),
        "assistant_output_chars": len(assistant_target),
    }


class LocalWMJsonlDataset:
    """Random-access JSONL dataset that normalizes rows lazily."""

    def __init__(
        self,
        path: Path,
        *,
        processor: Any | None = None,
        image_module: Any | None = None,
        skip_overlong_samples: bool = False,
        max_training_tokens: int | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_length: int | None = None,
        min_supervised_tokens: int | None = None,
        max_current_md_chars: int | None = None,
        max_target_next_md_chars: int | None = None,
        max_assistant_output_chars: int | None = None,
        skip_missing_images: bool = True,
        max_samples: int | None = None,
        prompt_key: str = "world_local",
    ) -> None:
        self.path = path
        self.processor = processor
        self.image_module = image_module
        self.skip_overlong_samples = skip_overlong_samples
        self.max_training_tokens = max_training_tokens
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_length = max_length
        self.min_supervised_tokens = min_supervised_tokens
        self.max_current_md_chars = max_current_md_chars
        self.max_target_next_md_chars = max_target_next_md_chars
        self.max_assistant_output_chars = max_assistant_output_chars
        self.skip_missing_images = skip_missing_images
        self.prompt_key = prompt_key
        self.offsets = self._build_offsets(path, max_samples=max_samples)
        self._handle = None
        self._invalid_indexes: set[int] = set()
        if not self.offsets:
            raise ValueError(f"{path} produced zero JSONL samples")

    @staticmethod
    def _build_offsets(path: Path, *, max_samples: int | None) -> list[int]:
        offsets: list[int] = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
                    if max_samples is not None and len(offsets) >= max_samples:
                        break
        return offsets

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_raw_row(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8")
        self._handle.seek(self.offsets[index])
        line = self._handle.readline()
        return json.loads(line)

    def _actual_supervised_tokens(self, normalized: dict[str, str]) -> int | None:
        if (
            self.processor is None
            or self.image_module is None
            or self.max_length is None
            or self.min_supervised_tokens is None
            or self.min_supervised_tokens <= 0
        ):
            return None
        image_path = normalized.get("current_image_path") or ""
        if not image_path:
            return None
        image = None
        try:
            image = self.image_module.open(image_path).convert("RGB")
            prompt_messages = build_local_world_model_messages(
                normalized,
                include_output=False,
                prompt_key=self.prompt_key,
            )
            training_messages = build_local_world_model_messages(
                normalized,
                include_output=True,
                prompt_key=self.prompt_key,
            )
            prompt_text = apply_chat_template(self.processor, prompt_messages, tokenize=False, add_generation_prompt=True)
            training_text = apply_chat_template(self.processor, training_messages, tokenize=False, add_generation_prompt=False)
            training_batch = self.processor(
                text=[training_text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_batch = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            full_len = int(training_batch["attention_mask"][0].sum().item())
            prompt_len = int(prompt_batch["attention_mask"][0].sum().item())
            return max(0, full_len - min(prompt_len, full_len))
        finally:
            if image is not None:
                image.close()

    def _skip_reason(self, normalized: dict[str, str], *, candidate_index: int) -> str | None:
        if candidate_index in self._invalid_indexes:
            return "cached_invalid"
        if self.skip_missing_images:
            image_path = normalized.get("current_image_path") or ""
            if not image_path or not Path(str(image_path)).is_file():
                self._invalid_indexes.add(candidate_index)
                return "missing_image"
        if self.max_current_md_chars and len(normalized.get("current_md") or "") > self.max_current_md_chars:
            self._invalid_indexes.add(candidate_index)
            return "current_md_chars_exceed_limit"
        if self.max_target_next_md_chars and len(normalized.get("target_next_md") or "") > self.max_target_next_md_chars:
            self._invalid_indexes.add(candidate_index)
            return "target_next_md_chars_exceed_limit"
        if self.max_assistant_output_chars and len(normalized.get("assistant_output") or "") > self.max_assistant_output_chars:
            self._invalid_indexes.add(candidate_index)
            return "assistant_output_chars_exceed_limit"
        if self.skip_overlong_samples:
            decision = build_length_filter_decision(
                normalized,
                processor=self.processor,
                max_training_tokens=self.max_training_tokens,
                max_input_tokens=self.max_input_tokens,
                max_output_tokens=self.max_output_tokens,
                prompt_key=self.prompt_key,
            )
            if decision is not None:
                self._invalid_indexes.add(candidate_index)
                return str(decision["reason"])
        supervised_tokens = self._actual_supervised_tokens(normalized)
        if supervised_tokens is not None and self.min_supervised_tokens is not None and supervised_tokens < self.min_supervised_tokens:
            self._invalid_indexes.add(candidate_index)
            return "supervised_tokens_after_truncation_below_limit"
        return None

    def __getitem__(self, index: int) -> dict[str, str]:
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        first_error: Exception | None = None
        for attempt in range(len(self.offsets)):
            candidate_index = (index + attempt) % len(self.offsets)
            try:
                normalized = normalize_row_for_dataset(self._read_raw_row(candidate_index))
                if self._skip_reason(normalized, candidate_index=candidate_index) is None:
                    return normalized
            except (json.JSONDecodeError, ValueError) as exc:
                if first_error is None:
                    first_error = exc
                continue
        if first_error is not None:
            raise ValueError(f"No valid samples in {self.path}; first error: {first_error}") from first_error
        raise ValueError(f"No valid samples in {self.path}; all samples were filtered")


def prepare_jsonl_for_dataset_loader(
    source_path: Path,
    output_path: Path,
    *,
    processor: Any | None = None,
    skip_overlong_samples: bool = False,
    max_training_tokens: int | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    skipped_output_path: Path | None = None,
    skipped_report_limit: int = 1000,
    prompt_key: str = "world_local",
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    skipped_tmp_path = None
    skipped_handle = None
    if skipped_output_path is not None:
        skipped_output_path.parent.mkdir(parents=True, exist_ok=True)
        skipped_tmp_path = skipped_output_path.with_name(f"{skipped_output_path.name}.tmp.{os.getpid()}")
        skipped_handle = skipped_tmp_path.open("w", encoding="utf-8")
    row_count = 0
    kept_count = 0
    skipped_count = 0
    skipped_by_reason: dict[str, int] = {}
    skipped_examples = []
    try:
        with source_path.open("r", encoding="utf-8") as source, tmp_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {source_path}:{line_number}: {exc}") from exc
                try:
                    normalized = normalize_row_for_dataset(row)
                except ValueError as exc:
                    raise ValueError(f"Invalid training row in {source_path}:{line_number}: {exc}") from exc

                skip_info = None
                if skip_overlong_samples:
                    skip_info = build_length_filter_decision(
                        normalized,
                        processor=processor,
                        max_training_tokens=max_training_tokens,
                        max_input_tokens=max_input_tokens,
                        max_output_tokens=max_output_tokens,
                        prompt_key=prompt_key,
                    )
                if skip_info is not None:
                    skipped_count += 1
                    skip_info["source_path"] = str(source_path)
                    skip_info["line_number"] = line_number
                    reason = str(skip_info["reason"])
                    skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
                    if len(skipped_examples) < skipped_report_limit:
                        skipped_examples.append(skip_info)
                    if skipped_handle is not None:
                        skipped_handle.write(json.dumps(skip_info, ensure_ascii=False) + "\n")
                    continue

                target.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                kept_count += 1
    finally:
        if skipped_handle is not None:
            skipped_handle.close()
    tmp_path.replace(output_path)
    if skipped_tmp_path is not None and skipped_output_path is not None:
        skipped_tmp_path.replace(skipped_output_path)
    return {
        "source_path": str(source_path),
        "prepared_path": str(output_path),
        "input_rows": row_count,
        "kept_rows": kept_count,
        "skipped_rows": skipped_count,
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "skipped_report": str(skipped_output_path) if skipped_output_path is not None else None,
        "skipped_examples": skipped_examples,
    }


def wait_for_file(path: Path, *, timeout_seconds: int = 3600, poll_seconds: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_seconds)


def distributed_barrier(torch_module) -> None:
    distributed_module = getattr(torch_module, "distributed", None)
    if distributed_module is None:
        return
    if distributed_module.is_available() and distributed_module.is_initialized():
        distributed_module.barrier()


def destroy_distributed_process_group() -> None:
    try:
        import torch
    except ImportError:
        return
    distributed_module = getattr(torch, "distributed", None)
    if (
        distributed_module is not None
        and distributed_module.is_available()
        and distributed_module.is_initialized()
    ):
        distributed_module.destroy_process_group()


def run_with_distributed_cleanup(operation):
    """Clean up successful DDP runs without hiding a rank-local failure.

    Destroying a process group while peer ranks are still inside a collective can
    block before Python reports the original exception.  On failure, report the
    exception first and let process exit release CUDA/NCCL resources; torchrun
    then terminates the peer ranks.
    """
    try:
        result = operation()
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "event": "sft_rank_failure",
                    "rank": int(os.environ.get("RANK", "0")),
                    "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    destroy_distributed_process_group()
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_adapter_safetensors(path: Path) -> int:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise RuntimeError("Checkpoint integrity validation requires safetensors.") from exc
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor_count = len(list(handle.keys()))
    if tensor_count <= 0:
        raise ValueError(f"Adapter checkpoint contains no tensors: {path}")
    return tensor_count


def write_checkpoint_integrity_manifest(
    checkpoint_dir: Path,
    *,
    global_step: int,
    world_size: int,
) -> Path:
    rng_state_names = (
        ["rng_state.pth"]
        if world_size == 1
        else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    required_names = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        *rng_state_names,
    ]
    for name in required_names:
        wait_for_file(checkpoint_dir / name, timeout_seconds=300, poll_seconds=0.25)
    missing = [name for name in required_names if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {checkpoint_dir}: missing {missing}")
    trainer_state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    saved_step = int(trainer_state.get("global_step", -1))
    if saved_step != global_step:
        raise ValueError(
            f"Checkpoint global_step mismatch for {checkpoint_dir}: "
            f"trainer_state={saved_step}, expected={global_step}"
        )
    tensor_count = _validate_adapter_safetensors(checkpoint_dir / "adapter_model.safetensors")
    files = {
        name: {
            "size_bytes": (checkpoint_dir / name).stat().st_size,
            "sha256": _sha256_file(checkpoint_dir / name),
        }
        for name in required_names
    }
    manifest = {
        "status": "passed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "global_step": global_step,
        "world_size": world_size,
        "adapter_tensor_count": tensor_count,
        "required_files": files,
    }
    manifest_path = checkpoint_dir / "checkpoint_integrity.json"
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(manifest_path)
    return manifest_path


def write_final_adapter_integrity_manifest(adapter_dir: Path, *, global_step: int) -> Path:
    required_names = ["adapter_model.safetensors", "adapter_config.json"]
    missing = [name for name in required_names if not (adapter_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete final adapter {adapter_dir}: missing {missing}")
    tensor_count = _validate_adapter_safetensors(adapter_dir / "adapter_model.safetensors")
    manifest = {
        "status": "passed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "global_step": global_step,
        "adapter_tensor_count": tensor_count,
        "required_files": {
            name: {
                "size_bytes": (adapter_dir / name).stat().st_size,
                "sha256": _sha256_file(adapter_dir / name),
            }
            for name in required_names
        },
    }
    manifest_path = adapter_dir / "adapter_integrity.json"
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(manifest_path)
    return manifest_path


def token_count(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def load_wm_eval_rows(path: Path, *, max_samples: int | None) -> list[dict[str, str]]:
    if max_samples is not None and max_samples <= 0:
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw_row = json.loads(line)
                from mkd2world.training.evaluation import action_type_from_row

                action_type_from_row(raw_row)
                row = raw_row
                normalized = normalize_row_for_dataset(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid WM eval row in {path}:{line_number}: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"Invalid WM eval row in {path}:{line_number}: {exc}") from exc
            rows.append(normalized)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_wm_eval_length_validation(
    rows: Sequence[dict[str, str]],
    *,
    processor: Any,
    max_length: int,
    max_output_tokens: int | None,
    prompt_key: str,
) -> dict[str, Any]:
    violations = []
    for index, row in enumerate(rows):
        decision = build_length_filter_decision(
            row,
            processor=processor,
            max_training_tokens=max_length,
            max_input_tokens=None,
            max_output_tokens=max_output_tokens,
            prompt_key=prompt_key,
        )
        if decision is not None:
            violations.append({"source_eval_index": index, **decision})
    return {
        "status": "passed" if not violations else "failed",
        "source_rows": len(rows),
        "max_length": max_length,
        "max_input_tokens": None,
        "max_output_tokens": max_output_tokens,
        "violations": violations,
    }


def _model_device(model) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return "cpu"


def _model_context_window(model) -> int | None:
    """Return the model's finite context window when it is declared."""
    candidates = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(get_base_model())
        except Exception:
            pass
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested_model = getattr(base_model, "model", None)
        if nested_model is not None:
            candidates.append(nested_model)
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        config_candidates = (config, getattr(config, "text_config", None))
        for config_candidate in config_candidates:
            for field_name in ("max_position_embeddings", "max_sequence_length", "seq_length"):
                value = getattr(config_candidate, field_name, None)
                if isinstance(value, int) and value > 0:
                    return value
    return None


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def generation_token_limit(
    model,
    *,
    prompt_width: int,
    max_new_tokens: int | None,
    max_length: int,
) -> int:
    context_window = _model_context_window(model)
    effective_context = min(max_length, context_window) if context_window is not None else max_length
    remaining = effective_context - prompt_width
    if remaining <= 0:
        raise ValueError(
            f"prompt uses {prompt_width} tokens, leaving no output capacity under max_length={effective_context}"
        )
    return remaining if max_new_tokens is None else min(max_new_tokens, remaining)


def _prediction_row(
    sample: dict[str, str],
    model_output: str,
    *,
    index: int,
    rank: int,
    global_step: int,
    prompt_key: str,
) -> dict[str, Any]:
    target_next_md = sample.get("target_next_md_clean") or sample.get("target_next_md", "")
    row = {
        "id": sample.get("id", ""),
        "source_eval_index": index,
        "rank": rank,
        "global_step": global_step,
        "model_output": model_output,
        "model_output_next_md": model_output,
        "prediction_replay_valid": True,
        "target_next_md": target_next_md,
        "candidate_action_json": sample.get("candidate_action_json", ""),
        "meta_json": sample.get("meta_json", ""),
    }
    concrete_prompt_key = resolve_local_world_model_prompt_key(sample, prompt_key)
    if concrete_prompt_key == "localwm_delta":
        _, delta_text = split_local_prediction(model_output)
        row["model_output_delta"] = delta_text
        try:
            row["model_output_next_md"] = apply_delta_flat(
                sample.get("current_md", ""),
                parse_delta(delta_text),
            )
        except (DeltaFormatError, DeltaReplayError) as exc:
            row["model_output_next_md"] = ""
            row["prediction_replay_valid"] = False
            row["prediction_replay_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _progress_rows(rows: Sequence[dict[str, str]], description: str | None):
    if not description:
        return rows
    try:
        from tqdm.auto import tqdm

        return tqdm(rows, total=len(rows), desc=description, dynamic_ncols=True)
    except Exception:
        print(json.dumps({"stage": description, "total": len(rows), "progress": "started"}), flush=True)
        return rows


def generate_wm_eval_prediction_rows(
    torch_module,
    model,
    processor,
    image_module,
    rows: Sequence[dict[str, str]],
    *,
    rank: int,
    world_size: int,
    max_new_tokens: int | None,
    max_length: int,
    global_step: int,
    prompt_key: str = "world_local",
    progress_description: str | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    generation_model = model.module if hasattr(model, "module") else model
    device = _model_device(generation_model)
    was_training = bool(getattr(generation_model, "training", False))
    model_config = getattr(generation_model, "config", None)
    previous_use_cache = getattr(model_config, "use_cache", None)
    if model_config is not None and previous_use_cache is not None:
        model_config.use_cache = True
    generation_model.eval()
    predictions: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(_progress_rows(rows, progress_description)):
            if index % world_size != rank:
                continue
            messages = build_local_world_model_messages(sample, include_output=False, prompt_key=prompt_key)
            text = apply_chat_template(processor, messages, tokenize=False, add_generation_prompt=True)
            image = image_module.open(sample["current_image_path"]).convert("RGB") if sample.get("current_image_path") else None
            processor_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
            if image is not None:
                processor_kwargs["images"] = [image]
            batch = _to_device(processor(**processor_kwargs), device)
            prompt_width = batch["input_ids"].shape[1]
            generate_kwargs: dict[str, Any] = {
                "do_sample": False,
                "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None),
                "eos_token_id": getattr(processor.tokenizer, "eos_token_id", None),
            }
            generate_kwargs["max_new_tokens"] = generation_token_limit(
                generation_model,
                prompt_width=prompt_width,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
            with torch_module.no_grad():
                sequences = generation_model.generate(**batch, **generate_kwargs)
            output = processor.batch_decode(sequences[:, prompt_width:], skip_special_tokens=True)[0]
            predictions.append(
                _prediction_row(
                    sample,
                    output,
                    index=index,
                    rank=rank,
                    global_step=global_step,
                    prompt_key=prompt_key,
                )
            )
            del batch, sequences
            if image is not None:
                image.close()
    finally:
        if model_config is not None and previous_use_cache is not None:
            model_config.use_cache = previous_use_cache
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        if was_training:
            generation_model.train()
    return predictions


def wm_eval_worker_ranks(
    *,
    training_rank: int,
    training_world_size: int,
    workers_per_gpu: int,
) -> tuple[int, tuple[int, ...]]:
    """Map one training rank to contiguous evaluation worker ranks on the same GPU."""
    if training_world_size < 1:
        raise ValueError("training_world_size must be >= 1")
    if training_rank < 0 or training_rank >= training_world_size:
        raise ValueError(f"training_rank must be in [0, {training_world_size})")
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be >= 1")
    eval_world_size = training_world_size * workers_per_gpu
    first_worker = training_rank * workers_per_gpu
    return eval_world_size, tuple(range(first_worker, first_worker + workers_per_gpu))


def _physical_gpu_for_local_rank(local_rank: int) -> str:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible:
        if local_rank >= len(visible):
            raise ValueError(
                f"LOCAL_RANK={local_rank} is outside CUDA_VISIBLE_DEVICES={visible}"
            )
        return visible[local_rank]
    return str(local_rank)


def _start_external_wm_eval_workers(
    *,
    worker_ranks: Sequence[int],
    eval_world_size: int,
    local_rank: int,
    dataset_jsonl: Path,
    worker_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> list[tuple[subprocess.Popen[Any], Any, Path]]:
    """Start the extra inference workers that share the current training rank's GPU."""
    if len(worker_ranks) <= 1:
        return []
    project_root = Path(__file__).resolve().parents[3]
    evaluator = project_root / "scripts/evaluation/datasetV3_eval_step/datasetV3_eval_step.py"
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    physical_gpu = _physical_gpu_for_local_rank(local_rank)
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    child_env["PYTHONPATH"] = str(project_root / "src")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "GROUP_RANK", "ROLE_RANK"):
        child_env.pop(key, None)
    dtype_name = "bf16" if args.bf16 else ("fp16" if args.fp16 else "auto")
    workers: list[tuple[subprocess.Popen[Any], Any, Path]] = []
    for worker_rank in worker_ranks[1:]:
        log_path = worker_dir / f"worker_rank{worker_rank}.log"
        command = [
            sys.executable,
            str(evaluator),
            "--dataset-jsonl",
            str(dataset_jsonl),
            "--output-dir",
            str(worker_dir),
            "--base-model",
            str(args.base_model),
            "--adapter-path",
            str(checkpoint_path),
            "--max-new-tokens",
            str(args.wm_eval_max_new_tokens),
            "--max-length",
            str(args.max_length),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--image-min-pixels",
            str(args.image_min_pixels),
            "--image-max-pixels",
            str(args.image_max_pixels),
            "--torch-dtype",
            dtype_name,
            "--prompt-key",
            str(args.prompt_key),
            "--device-map",
            "cuda:0",
            "--rank",
            str(worker_rank),
            "--world-size",
            str(eval_world_size),
            "--shard-only",
        ]
        if args.wm_eval_max_samples is not None:
            command.extend(["--max-samples", str(args.wm_eval_max_samples)])
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=child_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        workers.append((process, log_handle, log_path))
    return workers


def _wait_external_wm_eval_workers(
    workers: Sequence[tuple[subprocess.Popen[Any], Any, Path]],
) -> None:
    failures = []
    for process, log_handle, log_path in workers:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append({"pid": process.pid, "return_code": return_code, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"External WM evaluation workers failed: {failures}")


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wm_evaluation_split_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if args.wm_eval_in_domain_jsonl is not None:
        paths["in_domain"] = args.wm_eval_in_domain_jsonl
    if args.wm_eval_ood_jsonl is not None:
        paths["ood"] = args.wm_eval_ood_jsonl
    if not paths and args.wm_eval_jsonl is not None:
        paths["validation"] = args.wm_eval_jsonl
    return paths


def _write_wm_eval_summary(summary_path: Path, *, split: str, step: int, metrics: dict[str, Any], metrics_path: Path, predictions_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_scores = {
        bucket: values.get("EleF1", 0.0)
        for bucket, values in metrics.get("by_action_bucket", {}).items()
        if isinstance(values, dict)
    }
    row = {
        "split": split,
        "global_step": step,
        "EleF1": metrics.get("EleF1", 0.0),
        "format_score": metrics.get("format_score", 0.0),
        "format_success_rate": metrics.get("format_success_rate", 0.0),
        "evaluated_rows": metrics.get("evaluated_rows", 0.0),
        "invalid_action_rows": metrics.get("invalid_action_rows", 0.0),
        "by_action_bucket_EleF1": bucket_scores,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
    with summary_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _wm_eval_log_payload(split: str, step: int, metrics: dict[str, Any], metrics_path: Path, predictions_path: Path | None = None) -> dict[str, Any]:
    bucket_scores = {
        bucket: round(values.get("EleF1", 0.0), 4)
        for bucket, values in metrics.get("by_action_bucket", {}).items()
        if isinstance(values, dict)
    }
    payload: dict[str, Any] = {
        "wm_eval_split": split,
        "wm_eval_step": step,
        "EleF1": metrics.get("EleF1", 0.0),
        "format_score": metrics.get("format_score", 0.0),
        "format_success_rate": metrics.get("format_success_rate", 0.0),
        "evaluated_rows": metrics.get("evaluated_rows", 0.0),
        "invalid_action_rows": metrics.get("invalid_action_rows", 0.0),
        "by_action_bucket_EleF1": bucket_scores,
        "metrics_path": str(metrics_path),
    }
    if predictions_path is not None:
        payload["predictions_path"] = str(predictions_path)
    return payload


def update_wm_eval_best_checkpoint(
    output_dir: Path,
    *,
    split: str,
    step: int,
    adapter_path: Path,
    metrics: dict[str, Any],
    metrics_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], bool]:
    """Atomically maintain the best inline Validation checkpoint for one split."""
    best_path = output_dir / "wm_eval" / f"best_{split}.json"
    pointer_path = output_dir / "wm_eval" / f"best_{split}.adapter.txt"
    candidate = {
        "status": "passed",
        "selection_metric": "Validation EleF1",
        "split": split,
        "global_step": int(step),
        "EleF1": float(metrics.get("EleF1", 0.0)),
        "format_score": float(metrics.get("format_score", 0.0)),
        "format_success_rate": float(metrics.get("format_success_rate", 0.0)),
        "adapter": str(adapter_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "by_action_bucket_EleF1": {
            bucket: float(values.get("EleF1", 0.0))
            for bucket, values in metrics.get("by_action_bucket", {}).items()
            if isinstance(values, dict)
        },
    }
    existing = None
    if best_path.is_file():
        existing = json.loads(best_path.read_text(encoding="utf-8"))
    updated = existing is None or (
        candidate["EleF1"], -candidate["global_step"]
    ) > (
        float(existing.get("EleF1", 0.0)), -int(existing.get("global_step", 0))
    )
    report = candidate if updated else existing
    if updated:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = best_path.with_name(f"{best_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(best_path)
        pointer_temporary = pointer_path.with_name(f"{pointer_path.name}.tmp.{os.getpid()}")
        pointer_temporary.write_text(str(adapter_path) + "\n", encoding="utf-8")
        pointer_temporary.replace(pointer_path)
    return report, updated


def build_wm_score_eval_callback(
    TrainerCallback,
    *,
    torch_module,
    processor,
    image_module,
    eval_rows: Sequence[dict[str, str]],
    eval_jsonl_path: Path,
    length_validation: dict[str, Any],
    split_name: str,
    embedder_cache: dict[str, Any],
    output_dir: Path,
    checkpoint_root: Path,
    final_adapter_dir: Path,
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    allowed_steps: Sequence[int] | None = None,
):
    allowed_step_set = (
        None if allowed_steps is None else frozenset(int(step) for step in allowed_steps)
    )

    class EleF1EvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self._best_elef1: float | None = None
            self._consecutive_bad_checkpoints = 0

        def on_save(self, training_args, state, control, **kwargs):
            if not eval_rows:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            step = int(state.global_step)
            if allowed_step_set is not None and step not in allowed_step_set:
                return control
            step_dir = output_dir / "wm_eval" / f"step_{step:06d}" / split_name
            if rank == 0 and step_dir.exists():
                shutil.rmtree(step_dir)
            distributed_barrier(torch_module)
            step_dir.mkdir(parents=True, exist_ok=True)
            worker_dir = step_dir / "workers"
            worker_dir.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                (step_dir / "length_validation.json").write_text(
                    json.dumps(length_validation, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            eval_world_size, worker_ranks = wm_eval_worker_ranks(
                training_rank=rank,
                training_world_size=world_size,
                workers_per_gpu=args.wm_eval_workers_per_gpu,
            )
            checkpoint_path = checkpoint_root / f"checkpoint-{step}"
            if not checkpoint_path.is_dir():
                checkpoint_path = final_adapter_dir
            if not checkpoint_path.is_dir():
                raise FileNotFoundError(
                    f"No adapter found for WM evaluation at step {step}: "
                    f"{checkpoint_root / f'checkpoint-{step}'} or {final_adapter_dir}"
                )
            torch_module.cuda.empty_cache()
            external_workers = _start_external_wm_eval_workers(
                worker_ranks=worker_ranks,
                eval_world_size=eval_world_size,
                local_rank=local_rank,
                dataset_jsonl=eval_jsonl_path,
                worker_dir=worker_dir,
                checkpoint_path=checkpoint_path,
                args=args,
            )
            local_worker_rank = worker_ranks[0]
            local_worker_path = worker_dir / f"predictions_rank{local_worker_rank}.jsonl"
            try:
                predictions = generate_wm_eval_prediction_rows(
                    torch_module,
                    model,
                    processor,
                    image_module,
                    eval_rows,
                    rank=local_worker_rank,
                    world_size=eval_world_size,
                    max_new_tokens=args.wm_eval_max_new_tokens,
                    max_length=args.max_length,
                    global_step=step,
                    prompt_key=args.prompt_key,
                )
                write_jsonl(local_worker_path, predictions)
                _wait_external_wm_eval_workers(external_workers)
            except Exception:
                for process, log_handle, _log_path in external_workers:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)
                    if not log_handle.closed:
                        log_handle.close()
                raise
            shard_path = step_dir / f"predictions_rank{rank}.jsonl"
            physical_rank_rows: list[dict[str, Any]] = []
            for worker_rank in worker_ranks:
                worker_path = worker_dir / f"predictions_rank{worker_rank}.jsonl"
                worker_rows = _read_jsonl_rows(worker_path)
                expected_indexes = {
                    index for index in range(len(eval_rows)) if index % eval_world_size == worker_rank
                }
                actual_indexes = {int(row.get("source_eval_index", -1)) for row in worker_rows}
                if actual_indexes != expected_indexes:
                    raise RuntimeError(
                        f"Invalid worker shard at step {step}, rank {worker_rank}: "
                        f"expected={len(expected_indexes)} actual={len(actual_indexes)}"
                    )
                for row in worker_rows:
                    row["rank"] = rank
                    row["global_step"] = step
                physical_rank_rows.extend(worker_rows)
            physical_rank_rows.sort(key=lambda row: int(row.get("source_eval_index", 0)))
            write_jsonl(shard_path, physical_rank_rows)
            distributed_barrier(torch_module)
            if rank == 0:
                merged_rows: list[dict[str, Any]] = []
                expected_shards = [step_dir / f"predictions_rank{rank_id}.jsonl" for rank_id in range(world_size)]
                missing_shards = [str(path) for path in expected_shards if not path.is_file()]
                if missing_shards:
                    raise RuntimeError(f"Missing WM eval prediction shards: {missing_shards}")
                for shard in expected_shards:
                    merged_rows.extend(_read_jsonl_rows(shard))
                merged_rows.sort(key=lambda row: int(row.get("source_eval_index", 0)))
                source_indexes = [int(row.get("source_eval_index", -1)) for row in merged_rows]
                if len(merged_rows) != len(eval_rows) or len(set(source_indexes)) != len(eval_rows):
                    raise RuntimeError(
                        f"Invalid WM eval shard merge at step {step}: "
                        f"rows={len(merged_rows)} expected={len(eval_rows)} unique_indexes={len(set(source_indexes))}"
                    )
                predictions_path = step_dir / "predictions.jsonl"
                metrics_path = step_dir / "metrics.json"
                write_jsonl(predictions_path, merged_rows)
                from mkd2world.training.evaluation import evaluate_jsonl

                if "embedder" not in embedder_cache:
                    embedder_cache["embedder"] = SemanticEmbedder(args.wm_eval_embedding_model, device=args.wm_eval_embedding_device)
                metrics = evaluate_jsonl(
                    predictions_path,
                    prediction_field="model_output_next_md",
                    embedder=embedder_cache["embedder"],
                    action_breakdown=True,
                )
                metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                _write_wm_eval_summary(
                    output_dir / "wm_eval" / "summary.jsonl",
                    split=split_name,
                    step=step,
                    metrics=metrics,
                    metrics_path=metrics_path,
                    predictions_path=predictions_path,
                )
                print(
                    json.dumps(
                        _wm_eval_log_payload(split_name, step, metrics, metrics_path, predictions_path),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            distributed_barrier(torch_module)
            metrics_path = step_dir / "metrics.json"
            predictions_path = step_dir / "predictions.jsonl"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            elef1 = float(metrics.get("EleF1", 0.0))
            format_success_rate = float(metrics.get("format_success_rate", 0.0))
            previous_best = self._best_elef1
            bad_reasons = []
            if (
                args.wm_eval_min_format_success_rate is not None
                and format_success_rate < args.wm_eval_min_format_success_rate
            ):
                bad_reasons.append(
                    f"format_success_rate={format_success_rate:.6f}"
                    f"<{args.wm_eval_min_format_success_rate:.6f}"
                )
            if (
                previous_best is not None
                and args.wm_eval_max_elef1_drop_from_best is not None
                and previous_best - elef1 >= args.wm_eval_max_elef1_drop_from_best
            ):
                bad_reasons.append(
                    f"EleF1_drop={previous_best - elef1:.6f}"
                    f">={args.wm_eval_max_elef1_drop_from_best:.6f}"
                )
            self._best_elef1 = elef1 if previous_best is None else max(previous_best, elef1)
            self._consecutive_bad_checkpoints = (
                self._consecutive_bad_checkpoints + 1 if bad_reasons else 0
            )
            stop_training = (
                args.wm_eval_stop_after_bad_checkpoints > 0
                and self._consecutive_bad_checkpoints >= args.wm_eval_stop_after_bad_checkpoints
            )
            if rank == 0:
                best, best_updated = update_wm_eval_best_checkpoint(
                    output_dir,
                    split=split_name,
                    step=step,
                    adapter_path=checkpoint_path,
                    metrics=metrics,
                    metrics_path=metrics_path,
                    predictions_path=predictions_path,
                )
                health = {
                    "event": "wm_eval_checkpoint_health",
                    "split": split_name,
                    "global_step": step,
                    "EleF1": elef1,
                    "format_success_rate": format_success_rate,
                    "best_EleF1": self._best_elef1,
                    "best_step": int(best["global_step"]),
                    "best_updated": best_updated,
                    "bad_reasons": bad_reasons,
                    "consecutive_bad_checkpoints": self._consecutive_bad_checkpoints,
                    "stop_training": stop_training,
                }
                health_path = output_dir / "wm_eval" / "health.jsonl"
                with health_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(health, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(json.dumps(health, ensure_ascii=False), flush=True)
            distributed_barrier(torch_module)
            if stop_training:
                control.should_training_stop = True
            return control

    return EleF1EvalCallback()


def _checkpoint_step_from_path(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def _wm_eval_output_dir_for_args(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.wm_eval_output_dir is not None:
        return args.wm_eval_output_dir
    if checkpoint_path.parent.name in {"checkpoints", "rolling_checkpoints"}:
        return checkpoint_path.parent.parent
    if args.output_dir is None:
        raise ValueError("--output-dir is required when --wm-eval-output-dir cannot be derived from checkpoint path")
    return args.output_dir / (args.run_id or timestamp_run_id())


def missing_wm_checkpoint_eval_assets(args: argparse.Namespace) -> list[str]:
    missing = []
    checkpoint_path = args.wm_eval_checkpoint or args.resume_from_checkpoint or args.init_adapter
    if checkpoint_path is None:
        missing.append("wm_eval_checkpoint")
    elif not checkpoint_path.is_dir():
        missing.append(f"wm_eval_checkpoint:{checkpoint_path}")
    split_paths = wm_evaluation_split_paths(args)
    if (args.wm_eval_in_domain_jsonl is None) != (args.wm_eval_ood_jsonl is None):
        missing.append("both wm_eval_in_domain_jsonl and wm_eval_ood_jsonl are required")
    if not split_paths:
        missing.extend(["wm_eval_in_domain_jsonl", "wm_eval_ood_jsonl"])
    for split, path in split_paths.items():
        if args.wm_eval_max_samples != 0 and not path.is_file():
            missing.append(f"wm_eval_{split}_jsonl:{path}")
    if not args.base_model:
        missing.append("base_model")
    if checkpoint_path is not None and checkpoint_path.is_dir():
        try:
            _wm_eval_output_dir_for_args(args, checkpoint_path)
        except ValueError:
            missing.append("wm_eval_output_dir")
    return missing


def build_wm_checkpoint_eval_dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.wm_eval_checkpoint or args.resume_from_checkpoint or args.init_adapter
    output_dir = None
    step = None
    if checkpoint_path is not None:
        step = args.wm_eval_step if args.wm_eval_step is not None else _checkpoint_step_from_path(checkpoint_path)
        try:
            output_dir = _wm_eval_output_dir_for_args(args, checkpoint_path)
        except ValueError:
            output_dir = None
    return {
        "status": "dry_run",
        "can_start_wm_eval": not missing_wm_checkpoint_eval_assets(args),
        "missing_assets": missing_wm_checkpoint_eval_assets(args),
        "base_model": args.base_model,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "wm_eval_jsonl": str(args.wm_eval_jsonl) if args.wm_eval_jsonl else None,
        "wm_eval_splits": {split: str(path) for split, path in wm_evaluation_split_paths(args).items()},
        "wm_eval_output_dir": str(output_dir) if output_dir else None,
        "wm_eval_step": step,
        "wm_eval_max_samples": args.wm_eval_max_samples,
        "wm_eval_max_new_tokens": args.wm_eval_max_new_tokens,
        "wm_eval_workers_per_gpu": args.wm_eval_workers_per_gpu,
        "wm_eval_embedding_model": args.wm_eval_embedding_model,
        "wm_eval_embedding_device": args.wm_eval_embedding_device,
    }


def _init_eval_distributed(
    torch_module,
    *,
    local_rank: int,
    world_size: int,
    timeout_seconds: int,
) -> bool:
    if world_size <= 1:
        return False
    if not torch_module.cuda.is_available():
        raise RuntimeError("Distributed WM eval requires CUDA.")
    torch_module.cuda.set_device(local_rank)
    distributed_module = torch_module.distributed
    initialized_here = False
    if not distributed_module.is_initialized():
        if timeout_seconds <= 0:
            raise ValueError("WM eval DDP timeout must be positive.")
        distributed_module.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout_seconds),
        )
        initialized_here = True
    return initialized_here


def run_wm_checkpoint_eval(args: argparse.Namespace) -> dict[str, Path]:
    checkpoint_path = args.wm_eval_checkpoint or args.resume_from_checkpoint or args.init_adapter
    if checkpoint_path is None:
        raise ValueError("--wm-eval-checkpoint, --resume-from-checkpoint, or --init-adapter is required with --wm-eval-only")
    if not checkpoint_path.is_dir():
        raise ValueError(f"WM eval checkpoint must be an existing directory: {checkpoint_path}")
    split_paths = wm_evaluation_split_paths(args)
    if not split_paths:
        raise ValueError("in-domain and OOD WM eval JSONL files are required")
    missing_splits = {split: path for split, path in split_paths.items() if not path.is_file()}
    if missing_splits:
        raise ValueError(f"WM eval JSONL files must exist: {missing_splits}")
    if not args.base_model:
        raise ValueError("--base-model is required with --wm-eval-only")

    (
        torch,
        _load_dataset,
        _LoraConfig,
        PeftModel,
        _get_peft_model,
        Image,
        AutoModelForImageTextToText,
        AutoProcessor,
        _Trainer,
        _TrainerCallback,
        _TrainingArguments,
    ) = import_training_dependencies()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        args.device_mode = "ddp"
    initialized_dist = _init_eval_distributed(
        torch,
        local_rank=local_rank,
        world_size=world_size,
        timeout_seconds=args.wm_eval_ddp_timeout_seconds,
    )
    if args.device_mode == "ddp":
        if not torch.cuda.is_available():
            raise RuntimeError("--device-mode ddp requires CUDA.")
        torch.cuda.set_device(local_rank)
        device_map: Any = {"": local_rank}
    else:
        device_map = "auto"

    step = args.wm_eval_step if args.wm_eval_step is not None else _checkpoint_step_from_path(checkpoint_path)
    output_dir = _wm_eval_output_dir_for_args(args, checkpoint_path)
    step_dir = output_dir / "wm_eval" / f"step_{step:06d}"
    if rank == 0:
        if step_dir.exists():
            shutil.rmtree(step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(torch)

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
        trust_remote_code=True,
    )
    eval_rows_by_split = {
        split: load_wm_eval_rows(path, max_samples=args.wm_eval_max_samples)
        for split, path in split_paths.items()
    }
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(checkpoint_path), is_trainable=False)
    model.eval()

    metrics_paths: dict[str, Path] = {}
    embedder = None
    for split, eval_rows in eval_rows_by_split.items():
        split_dir = step_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        worker_dir = split_dir / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)
        eval_world_size, worker_ranks = wm_eval_worker_ranks(
            training_rank=rank,
            training_world_size=world_size,
            workers_per_gpu=args.wm_eval_workers_per_gpu,
        )
        external_workers = _start_external_wm_eval_workers(
            worker_ranks=worker_ranks,
            eval_world_size=eval_world_size,
            local_rank=local_rank,
            dataset_jsonl=split_paths[split],
            worker_dir=worker_dir,
            checkpoint_path=checkpoint_path,
            args=args,
        )
        local_worker_rank = worker_ranks[0]
        local_worker_path = worker_dir / f"predictions_rank{local_worker_rank}.jsonl"
        try:
            predictions = generate_wm_eval_prediction_rows(
                torch,
                model,
                processor,
                Image,
                eval_rows,
                rank=local_worker_rank,
                world_size=eval_world_size,
                max_new_tokens=args.wm_eval_max_new_tokens,
                max_length=args.max_length,
                global_step=step,
                prompt_key=args.prompt_key,
                progress_description=f"wm_eval_{split}_step_{step}_rank_{local_worker_rank}",
            )
            write_jsonl(local_worker_path, predictions)
            _wait_external_wm_eval_workers(external_workers)
        except Exception:
            for process, log_handle, _log_path in external_workers:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                if not log_handle.closed:
                    log_handle.close()
            raise

        shard_path = split_dir / f"predictions_rank{rank}.jsonl"
        physical_rank_rows: list[dict[str, Any]] = []
        for worker_rank in worker_ranks:
            worker_path = worker_dir / f"predictions_rank{worker_rank}.jsonl"
            worker_rows = _read_jsonl_rows(worker_path)
            expected_indexes = {
                index for index in range(len(eval_rows)) if index % eval_world_size == worker_rank
            }
            actual_indexes = {int(row.get("source_eval_index", -1)) for row in worker_rows}
            if actual_indexes != expected_indexes:
                raise RuntimeError(
                    f"Invalid posthoc WM eval worker shard for {split}, rank {worker_rank}: "
                    f"expected={len(expected_indexes)} actual={len(actual_indexes)}"
                )
            for row in worker_rows:
                row["rank"] = rank
                row["global_step"] = step
            physical_rank_rows.extend(worker_rows)
        physical_rank_rows.sort(key=lambda row: int(row.get("source_eval_index", 0)))
        write_jsonl(shard_path, physical_rank_rows)
        distributed_barrier(torch)

        metrics_path = split_dir / "metrics.json"
        metrics_paths[split] = metrics_path
        if rank == 0:
            merged_rows: list[dict[str, Any]] = []
            expected_shards = [split_dir / f"predictions_rank{rank_id}.jsonl" for rank_id in range(world_size)]
            missing_shards = [str(path) for path in expected_shards if not path.is_file()]
            if missing_shards:
                raise RuntimeError(f"Missing WM eval prediction shards for {split}: {missing_shards}")
            for shard in expected_shards:
                merged_rows.extend(_read_jsonl_rows(shard))
            merged_rows.sort(key=lambda row: int(row.get("source_eval_index", 0)))
            source_indexes = [int(row.get("source_eval_index", -1)) for row in merged_rows]
            if len(merged_rows) != len(eval_rows) or len(set(source_indexes)) != len(eval_rows):
                raise RuntimeError(
                    f"Invalid WM eval shard merge for {split} at step {step}: "
                    f"rows={len(merged_rows)} expected={len(eval_rows)} unique_indexes={len(set(source_indexes))}"
                )
            predictions_path = split_dir / "predictions.jsonl"
            write_jsonl(predictions_path, merged_rows)
            from mkd2world.training.evaluation import evaluate_jsonl

            if embedder is None:
                embedder = SemanticEmbedder(args.wm_eval_embedding_model, device=args.wm_eval_embedding_device)
            metrics = evaluate_jsonl(
                predictions_path,
                prediction_field="model_output_next_md",
                embedder=embedder,
                action_breakdown=True,
            )
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_wm_eval_summary(
                output_dir / "wm_eval" / "summary.jsonl",
                split=split,
                step=step,
                metrics=metrics,
                metrics_path=metrics_path,
                predictions_path=predictions_path,
            )
            print(json.dumps(_wm_eval_log_payload(split, step, metrics, metrics_path, predictions_path), ensure_ascii=False), flush=True)
        distributed_barrier(torch)
    if initialized_dist:
        torch.distributed.destroy_process_group()
    return metrics_paths


def find_action_analysis_body_token_span(target_text: str, tokenizer) -> tuple[int, int] | None:
    action_match = re.search(r"(?is)\AACTION_ANALYSIS\s*:?", target_text or "")
    predicted_match = re.search(r"(?is)\n\s*\nPREDICTED_MARKDOWN\s*:?", target_text or "")
    if action_match is None or predicted_match is None or predicted_match.start() <= action_match.end():
        return None

    body_start_char = action_match.end()
    if body_start_char < len(target_text) and target_text[body_start_char] == "\n":
        body_start_char += 1
    body_end_char = predicted_match.start()
    if body_end_char <= body_start_char:
        return None

    return (
        token_count(tokenizer, target_text[:body_start_char]),
        token_count(tokenizer, target_text[:body_end_char]),
    )


def mask_prompt_tokens(
    labels,
    full_attention_mask,
    prompt_attention_mask,
    *,
    tokenizer,
    assistant_targets: list[str],
    assistant_loss_scope: str,
    padding_side: str,
    min_supervised_tokens: int,
    sample_ids: list[str],
) -> None:
    labels[full_attention_mask == 0] = -100
    seq_len = labels.shape[1]
    for row_idx, sample_id in enumerate(sample_ids):
        full_len = int(full_attention_mask[row_idx].sum().item())
        prompt_len = int(prompt_attention_mask[row_idx].sum().item())
        prompt_start = seq_len - full_len if padding_side == "left" else 0
        prompt_end = min(prompt_start + prompt_len, seq_len)
        labels[row_idx, prompt_start:prompt_end] = -100

        if assistant_loss_scope not in {"full", "markdown_only"}:
            raise ValueError(f"Unsupported assistant_loss_scope: {assistant_loss_scope}")

        supervised_tokens = int((labels[row_idx] != -100).sum().item())
        if supervised_tokens < min_supervised_tokens:
            raise ValueError(
                f"Too few supervised assistant tokens for sample {sample_id}: "
                f"{supervised_tokens} < {min_supervised_tokens}. Increase --max-length, "
                "lower --image-max-pixels, or filter very long Markdown samples."
            )


@dataclass
class LocalWMSftCollator:
    processor: Any
    image_module: Any
    max_length: int
    max_output_tokens: int | None
    assistant_loss_scope: str
    min_supervised_tokens: int
    prompt_key: str = "world_local"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        prompt_texts = []
        assistant_targets = []
        images = []
        sample_ids = []
        for sample in features:
            prompt_messages = build_local_world_model_messages(
                sample,
                include_output=False,
                prompt_key=self.prompt_key,
            )
            training_messages = build_local_world_model_messages(
                sample,
                include_output=True,
                prompt_key=self.prompt_key,
            )
            prompt_texts.append(
                apply_chat_template(self.processor, prompt_messages, tokenize=False, add_generation_prompt=True)
            )
            texts.append(
                apply_chat_template(self.processor, training_messages, tokenize=False, add_generation_prompt=False)
            )
            assistant_targets.append(
                build_assistant_output(
                    str(sample.get("action_analysis") or ""),
                    local_world_model_target(sample, prompt_key=self.prompt_key),
                )
            )
            images.append(self.image_module.open(sample["current_image_path"]).convert("RGB"))
            sample_ids.append(str(sample.get("id") or "<unknown>"))

        if self.max_output_tokens is not None:
            output_lengths = [token_count(self.processor.tokenizer, target) for target in assistant_targets]
            overlong_outputs = [
                (sample_id, length)
                for sample_id, length in zip(sample_ids, output_lengths)
                if length > self.max_output_tokens
            ]
            if overlong_outputs:
                raise ValueError(
                    f"assistant output exceeds max_output_tokens={self.max_output_tokens}; "
                    f"samples={overlong_outputs}"
                )

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        full_lengths = [int(mask.sum().item()) for mask in batch["attention_mask"]]
        overlong_sequences = [
            (sample_id, length)
            for sample_id, length in zip(sample_ids, full_lengths)
            if length > self.max_length
        ]
        if overlong_sequences:
            raise ValueError(
                f"training sequence exceeds max_length={self.max_length}; samples={overlong_sequences}. "
                "The source dataset must be corrected; training does not truncate or skip rows."
            )
        labels = batch["input_ids"].clone()
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        mask_prompt_tokens(
            labels,
            batch["attention_mask"],
            prompt_batch["attention_mask"],
            tokenizer=self.processor.tokenizer,
            assistant_targets=assistant_targets,
            assistant_loss_scope=self.assistant_loss_scope,
            padding_side=getattr(self.processor.tokenizer, "padding_side", "right"),
            min_supervised_tokens=self.min_supervised_tokens,
            sample_ids=sample_ids,
        )
        batch["labels"] = labels
        return batch


def missing_training_assets(args: argparse.Namespace) -> list[str]:
    missing = []
    if not args.base_model:
        missing.append("base_model")
    if args.train_jsonl is None:
        missing.append("train_jsonl")
    elif not args.train_jsonl.is_file():
        missing.append(f"train_jsonl:{args.train_jsonl}")
    if args.eval_jsonl is not None and not args.eval_jsonl.is_file():
        missing.append(f"eval_jsonl:{args.eval_jsonl}")
    if (args.wm_eval_in_domain_jsonl is None) != (args.wm_eval_ood_jsonl is None):
        missing.append("both wm_eval_in_domain_jsonl and wm_eval_ood_jsonl are required")
    for split, path in wm_evaluation_split_paths(args).items():
        if args.wm_eval_max_samples != 0 and not path.is_file():
            missing.append(f"wm_eval_{split}_jsonl:{path}")
    if args.wm_eval_workers_per_gpu < 1:
        missing.append("wm_eval_workers_per_gpu must be >= 1")
    if (
        args.wm_eval_min_format_success_rate is not None
        and not 0.0 <= args.wm_eval_min_format_success_rate <= 1.0
    ):
        missing.append("wm_eval_min_format_success_rate must be in [0, 1]")
    if (
        args.wm_eval_max_elef1_drop_from_best is not None
        and args.wm_eval_max_elef1_drop_from_best <= 0.0
    ):
        missing.append("wm_eval_max_elef1_drop_from_best must be > 0")
    if args.wm_eval_stop_after_bad_checkpoints < 0:
        missing.append("wm_eval_stop_after_bad_checkpoints must be >= 0")
    if args.output_dir is None:
        missing.append("output_dir")
    try:
        PromptRegistry.from_project().path_for(args.prompt_key)
    except (KeyError, FileNotFoundError) as exc:
        missing.append(f"prompt_key:{exc}")
    return missing


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_local_wm_sft")


def git_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def training_arguments_kwargs(args: argparse.Namespace, checkpoint_dir: Path, *, has_eval: bool, distributed: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "seed": args.seed,
        "data_seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "report_to": [],
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "dataloader_persistent_workers": args.dataloader_persistent_workers,
        "disable_tqdm": args.disable_tqdm,
    }
    if has_eval:
        kwargs["per_device_eval_batch_size"] = args.per_device_eval_batch_size
        kwargs["eval_steps"] = args.eval_steps
    if args.dataloader_prefetch_factor is not None:
        kwargs["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor
    if distributed:
        kwargs["ddp_find_unused_parameters"] = False
    return kwargs


def build_training_arguments(TrainingArguments, args: argparse.Namespace, checkpoint_dir: Path, *, has_eval: bool, distributed: bool):
    kwargs = training_arguments_kwargs(args, checkpoint_dir, has_eval=has_eval, distributed=distributed)
    signature = inspect.signature(TrainingArguments)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"
    return TrainingArguments(**kwargs)


def build_interval_override_callback(TrainerCallback, args: argparse.Namespace, *, has_eval: bool):
    class IntervalOverrideCallback(TrainerCallback):
        def on_train_begin(self, training_args, state, control, **kwargs):
            state.logging_steps = args.logging_steps
            state.save_steps = args.save_steps
            if has_eval:
                state.eval_steps = args.eval_steps
            return control

    return IntervalOverrideCallback()


def build_checkpoint_integrity_callback(
    TrainerCallback,
    *,
    torch_module,
    checkpoint_root: Path,
    rank: int,
    world_size: int,
):
    class CheckpointIntegrityCallback(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            step = int(state.global_step)
            if rank == 0:
                checkpoint_path = checkpoint_root / f"checkpoint-{step}"
                manifest_path = write_checkpoint_integrity_manifest(
                    checkpoint_path,
                    global_step=step,
                    world_size=world_size,
                )
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_integrity_passed",
                            "global_step": step,
                            "manifest": str(manifest_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            distributed_barrier(torch_module)
            return control

    return CheckpointIntegrityCallback()


def _duration_text(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def build_training_timing_callback(TrainerCallback, args: argparse.Namespace):
    class TrainingTimingCallback(TrainerCallback):
        def __init__(self) -> None:
            self._start_wall: float | None = None
            self._last_log_wall: float | None = None
            self._start_step: int = 0
            self._last_log_step: int = 0

        def _is_rank_zero(self) -> bool:
            return int(os.environ.get("RANK", "0")) == 0

        def on_step_begin(self, training_args, state, control, **kwargs):
            if self._start_wall is None:
                now = time.monotonic()
                self._start_wall = now
                self._last_log_wall = now
                self._start_step = int(state.global_step)
                self._last_log_step = int(state.global_step)
                if self._is_rank_zero():
                    print(
                        json.dumps(
                            {
                                "event": "sft_progress_start",
                                "global_step": int(state.global_step),
                                "max_steps": int(state.max_steps),
                                "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            return control

        def on_step_end(self, training_args, state, control, **kwargs):
            if not self._is_rank_zero() or self._start_wall is None or self._last_log_wall is None:
                return control
            step = int(state.global_step)
            if step <= self._last_log_step:
                return control
            if args.logging_steps > 0 and step % args.logging_steps != 0 and step < int(state.max_steps):
                return control
            now = time.monotonic()
            total_steps = max(0, step - self._start_step)
            recent_steps = max(0, step - self._last_log_step)
            elapsed = max(0.0, now - self._start_wall)
            recent_elapsed = max(1e-9, now - self._last_log_wall)
            recent_steps_per_sec = recent_steps / recent_elapsed if recent_steps else 0.0
            average_steps_per_sec = total_steps / elapsed if elapsed > 0 and total_steps else 0.0
            remaining_steps = max(0, int(state.max_steps) - step)
            eta_seconds = remaining_steps / recent_steps_per_sec if recent_steps_per_sec > 0 else None
            print(
                json.dumps(
                    {
                        "event": "sft_progress",
                        "global_step": step,
                        "max_steps": int(state.max_steps),
                        "steps_since_launch": total_steps,
                        "remaining_steps": remaining_steps,
                        "recent_steps_per_sec": recent_steps_per_sec,
                        "average_steps_per_sec": average_steps_per_sec,
                        "elapsed_since_first_train_step": _duration_text(elapsed),
                        "eta_by_recent_rate": _duration_text(eta_seconds),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            self._last_log_step = step
            self._last_log_wall = now
            return control

    return TrainingTimingCallback()


def load_and_prepare_dataset(
    load_dataset,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    processor: Any | None,
    image_module: Any | None = None,
    is_main_process: bool,
) -> tuple[Any, dict[str, Any]]:
    if args.dataset_loading_mode == "direct":
        dataset: dict[str, Any] = {
            "train": LocalWMJsonlDataset(
                args.train_jsonl,
                processor=processor,
                image_module=image_module,
                skip_overlong_samples=args.skip_overlong_samples,
                max_training_tokens=args.max_length,
                max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens,
                max_length=args.max_length,
                min_supervised_tokens=None,
                max_current_md_chars=args.max_current_md_chars,
                max_target_next_md_chars=args.max_target_next_md_chars,
                max_assistant_output_chars=args.max_assistant_output_chars,
                skip_missing_images=args.skip_missing_images,
                max_samples=args.max_train_samples,
                prompt_key=args.prompt_key,
            )
        }
        if args.eval_jsonl is not None:
            dataset["eval"] = LocalWMJsonlDataset(
                args.eval_jsonl,
                processor=processor,
                image_module=image_module,
                skip_overlong_samples=args.skip_overlong_samples,
                max_training_tokens=args.max_length,
                max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens,
                max_length=args.max_length,
                min_supervised_tokens=None,
                max_current_md_chars=args.max_current_md_chars,
                max_target_next_md_chars=args.max_target_next_md_chars,
                max_assistant_output_chars=args.max_assistant_output_chars,
                skip_missing_images=args.skip_missing_images,
                max_samples=args.max_eval_samples,
                prompt_key=args.prompt_key,
            )
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "loading_mode": "direct",
            "source_train_jsonl": str(args.train_jsonl),
            "source_eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
            "train_rows": len(dataset["train"]),
            "eval_rows": len(dataset["eval"]) if "eval" in dataset else 0,
            "overlong_filter": {
                "enabled": args.skip_overlong_samples,
                "mode": "lazy_getitem_skip",
                "max_input_tokens": args.max_input_tokens,
                "max_output_tokens": args.max_output_tokens,
                "max_training_tokens": args.max_length,
            },
            "length_filter_stats": {},
            "image_filter_stats": {"mode": "lazy_getitem_skip", "enabled": args.skip_missing_images},
            "dataset_sizes": {split_name: len(split) for split_name, split in dataset.items()},
            "notes": [
                "Source JSONL files are read directly; no prepared_jsonl copy is generated.",
                "Rows are normalized lazily, and assistant_output is rebuilt from target_next_md.",
                "Length filtering is disabled by default; the collator fails instead of truncating an invalid sequence.",
                "Missing-image handling remains an independent data-integrity guard.",
            ],
        }
        return dataset, manifest

    prepared_data_dir = output_dir / "prepared_jsonl"
    prepared_train_jsonl = prepared_data_dir / "train.jsonl"
    prepared_eval_jsonl = prepared_data_dir / "eval.jsonl"
    skipped_train_jsonl = prepared_data_dir / "skipped_overlong_train.jsonl"
    skipped_eval_jsonl = prepared_data_dir / "skipped_overlong_eval.jsonl"
    prepared_manifest_path = prepared_data_dir / "manifest.json"
    if is_main_process:
        train_prepare_report = prepare_jsonl_for_dataset_loader(
            args.train_jsonl,
            prepared_train_jsonl,
            processor=processor,
            skip_overlong_samples=args.skip_overlong_samples,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            max_training_tokens=args.max_length,
            skipped_output_path=skipped_train_jsonl,
            skipped_report_limit=args.length_filter_report_limit,
            prompt_key=args.prompt_key,
        )
        eval_prepare_report = None
        if args.eval_jsonl is not None:
            eval_prepare_report = prepare_jsonl_for_dataset_loader(
                args.eval_jsonl,
                prepared_eval_jsonl,
                processor=processor,
                skip_overlong_samples=args.skip_overlong_samples,
                max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens,
                max_training_tokens=args.max_length,
                skipped_output_path=skipped_eval_jsonl,
                skipped_report_limit=args.length_filter_report_limit,
                prompt_key=args.prompt_key,
            )
        prepared_manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_train_jsonl": str(args.train_jsonl),
            "source_eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
            "prepared_train_jsonl": str(prepared_train_jsonl),
            "prepared_eval_jsonl": str(prepared_eval_jsonl) if args.eval_jsonl else None,
            "train_rows": train_prepare_report["kept_rows"],
            "eval_rows": eval_prepare_report["kept_rows"] if eval_prepare_report else 0,
            "overlong_filter": {
                "enabled": args.skip_overlong_samples,
                "max_input_tokens": args.max_input_tokens,
                "max_output_tokens": args.max_output_tokens,
                "max_training_tokens": args.max_length,
                "train": train_prepare_report,
                "eval": eval_prepare_report,
            },
            "notes": [
                "Original source JSONL files are not modified.",
                "candidate_action is serialized to candidate_action_json for stable datasets schema inference.",
                "Overlong samples are skipped before datasets loading so DDP ranks see identical rows.",
            ],
        }
        prepared_manifest_path.write_text(
            json.dumps(prepared_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        wait_for_file(prepared_manifest_path)
        prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))

    data_files = {"train": str(prepared_train_jsonl)}
    if args.eval_jsonl is not None:
        data_files["eval"] = str(prepared_eval_jsonl)
    dataset = load_dataset("json", data_files=data_files)

    filter_stats = {}
    if args.max_current_md_chars or args.max_target_next_md_chars or args.max_assistant_output_chars:
        for split_name in list(dataset.keys()):
            before_count = len(dataset[split_name])

            def keep_reasonable_length(sample):
                if args.max_current_md_chars and len(sample.get("current_md") or "") > args.max_current_md_chars:
                    return False
                if args.max_target_next_md_chars and len(sample.get("target_next_md") or "") > args.max_target_next_md_chars:
                    return False
                if args.max_assistant_output_chars and len(sample.get("assistant_output") or "") > args.max_assistant_output_chars:
                    return False
                return True

            dataset[split_name] = dataset[split_name].filter(keep_reasonable_length)
            filter_stats[split_name] = {
                "before": before_count,
                "after": len(dataset[split_name]),
                "removed": before_count - len(dataset[split_name]),
            }
    image_filter_stats = {}
    if args.skip_missing_images:
        for split_name in list(dataset.keys()):
            before_count = len(dataset[split_name])

            def keep_existing_image(sample):
                image_path = sample.get("current_image_path") or ""
                return bool(image_path) and Path(str(image_path)).is_file()

            dataset[split_name] = dataset[split_name].filter(keep_existing_image)
            image_filter_stats[split_name] = {
                "before": before_count,
                "after": len(dataset[split_name]),
                "removed": before_count - len(dataset[split_name]),
            }
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.eval_jsonl is not None and args.max_eval_samples is not None:
        dataset["eval"] = dataset["eval"].select(range(min(args.max_eval_samples, len(dataset["eval"]))))
    prepared_manifest["length_filter_stats"] = filter_stats
    prepared_manifest["image_filter_stats"] = image_filter_stats
    prepared_manifest["dataset_sizes"] = {split_name: len(dataset[split_name]) for split_name in dataset.keys()}
    return dataset, prepared_manifest


def run_training(args: argparse.Namespace) -> Path:
    missing = missing_training_assets(args)
    if missing:
        raise ValueError("Missing SFT training assets: " + ", ".join(missing))

    (
        torch,
        load_dataset,
        LoraConfig,
        PeftModel,
        get_peft_model,
        Image,
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    ) = import_training_dependencies()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        args.device_mode = "ddp"
    if args.device_mode == "ddp":
        if not torch.cuda.is_available():
            raise RuntimeError("--device-mode ddp requires CUDA.")
        torch.cuda.set_device(local_rank)
        device_map: Any = {"": local_rank}
    else:
        device_map = "auto"
    is_main_process = rank == 0

    run_id = args.run_id or timestamp_run_id()
    output_dir = args.output_dir / run_id
    checkpoint_dir = output_dir / "checkpoints"
    final_dir = output_dir / "final_adapter"
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
        trust_remote_code=True,
    )
    dataset, prepared_manifest = load_and_prepare_dataset(
        load_dataset,
        args,
        output_dir,
        processor=processor,
        image_module=Image,
        is_main_process=is_main_process,
    )
    wm_eval_paths = wm_evaluation_split_paths(args)
    wm_eval_rows_by_split = {
        split: load_wm_eval_rows(path, max_samples=args.wm_eval_max_samples)
        for split, path in wm_eval_paths.items()
    }
    wm_eval_length_validation_by_split = {
        split: build_wm_eval_length_validation(
            rows,
            processor=processor,
            max_length=args.max_length,
            max_output_tokens=args.max_output_tokens,
            prompt_key=args.prompt_key,
        )
        for split, rows in wm_eval_rows_by_split.items()
    }
    length_failures = {
        split: report["violations"]
        for split, report in wm_eval_length_validation_by_split.items()
        if report["status"] != "passed"
    }
    if length_failures:
        raise ValueError(
            "WM evaluation rows violate the configured length policy: "
            + json.dumps({split: len(rows) for split, rows in length_failures.items()}, ensure_ascii=False)
        )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    for param in model.parameters():
        param.requires_grad = False

    target_modules = resolve_lora_target_modules(model, args.target_modules)
    if args.init_adapter is not None:
        if not args.init_adapter.is_dir():
            raise ValueError(f"--init-adapter must be an existing adapter directory: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    if is_main_process and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    prompt_path = PromptRegistry.from_project().path_for(args.prompt_key)
    prompt_sha256 = _sha256_file(prompt_path)
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": {
            "key": args.prompt_key,
            "path": str(prompt_path),
            "sha256": prompt_sha256,
        },
        "model": {
            "base_model": args.base_model,
            "target_modules": target_modules,
            "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        },
        "data": prepared_manifest,
        "wm_eval": {
            "jsonl": {split: str(path) for split, path in wm_eval_paths.items()},
            "rows": {split: len(rows) for split, rows in wm_eval_rows_by_split.items()},
            "workers_per_gpu": args.wm_eval_workers_per_gpu,
            "effective_worker_world_size": world_size * args.wm_eval_workers_per_gpu,
            "max_samples": args.wm_eval_max_samples,
            "max_new_tokens": args.wm_eval_max_new_tokens,
            "max_length": args.max_length,
            "max_input_tokens": None,
            "max_output_tokens": args.max_output_tokens,
            "embedding_model": args.wm_eval_embedding_model,
            "embedding_device": args.wm_eval_embedding_device,
            "min_format_success_rate": args.wm_eval_min_format_success_rate,
            "max_elef1_drop_from_best": args.wm_eval_max_elef1_drop_from_best,
            "stop_after_bad_checkpoints": args.wm_eval_stop_after_bad_checkpoints,
        },
        "output_dir": str(output_dir),
        "final_adapter": str(final_dir),
        "distributed": {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device_mode": args.device_mode,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "args": vars(args) | {"train_jsonl": str(args.train_jsonl), "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None, "output_dir": str(args.output_dir)},
        "git_status_short": git_status_short(),
        "notes": [
            "Base model parameters are frozen; PEFT LoRA parameters are trainable.",
            "System/user prompt tokens and padding are masked from loss.",
            "Local WM assistant targets are pure Markdown; assistant_loss_scope is retained for CLI compatibility and is a no-op for Markdown-only completions.",
            "Original base model weights are not written by this script.",
        ],
    }
    if is_main_process:
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        shutil.copyfile(prompt_path, output_dir / "prompt_snapshot.md")

    collator = LocalWMSftCollator(
        processor=processor,
        image_module=Image,
        max_length=args.max_length,
        max_output_tokens=args.max_output_tokens,
        assistant_loss_scope=args.assistant_loss_scope,
        min_supervised_tokens=args.min_supervised_tokens,
        prompt_key=args.prompt_key,
    )
    has_eval = "eval" in dataset
    training_args = build_training_arguments(TrainingArguments, args, checkpoint_dir, has_eval=has_eval, distributed=distributed)
    callbacks = [
        build_interval_override_callback(TrainerCallback, args, has_eval=has_eval),
        build_training_timing_callback(TrainerCallback, args),
        build_checkpoint_integrity_callback(
            TrainerCallback,
            torch_module=torch,
            checkpoint_root=checkpoint_dir,
            rank=rank,
            world_size=world_size,
        ),
    ]
    wm_eval_callbacks = []
    wm_eval_embedder_cache: dict[str, Any] = {}
    for split_name, wm_eval_rows in wm_eval_rows_by_split.items():
        if wm_eval_rows:
            wm_callback = build_wm_score_eval_callback(
                    TrainerCallback,
                    torch_module=torch,
                    processor=processor,
                    image_module=Image,
                    eval_rows=wm_eval_rows,
                    eval_jsonl_path=wm_eval_paths[split_name],
                    length_validation=wm_eval_length_validation_by_split[split_name],
                    split_name=split_name,
                    embedder_cache=wm_eval_embedder_cache,
                    output_dir=output_dir,
                    checkpoint_root=checkpoint_dir,
                    final_adapter_dir=final_dir,
                    args=args,
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                )
            wm_eval_callbacks.append(wm_callback)
            callbacks.append(wm_callback)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if has_eval else None,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(final_dir))
    distributed_barrier(torch)
    final_step = int(trainer.state.global_step)
    if trainer.is_world_process_zero():
        write_final_adapter_integrity_manifest(final_dir, global_step=final_step)
    distributed_barrier(torch)
    if wm_eval_callbacks and int(trainer.state.global_step) % max(1, args.save_steps) != 0:
        for wm_callback in wm_eval_callbacks:
            wm_callback.on_save(
                training_args,
                trainer.state,
                trainer.control,
                model=trainer.model,
            )
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(final_dir))
        train_summary = {
            "run_id": run_id,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "final_adapter": str(final_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "final_global_step": final_step,
            "world_size": world_size,
            "wm_eval_workers_per_gpu": args.wm_eval_workers_per_gpu,
            "wm_eval_effective_world_size": world_size * args.wm_eval_workers_per_gpu,
            "device_mode": args.device_mode,
        }
        (output_dir / "train_summary.json").write_text(
            json.dumps(train_summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (output_dir / "latest_successful_checkpoint.txt").write_text(str(final_dir) + "\n", encoding="utf-8")
    return final_dir


def build_dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or timestamp_run_id()
    missing = missing_training_assets(args)
    output_dir = args.output_dir / run_id if args.output_dir is not None else None
    return {
        "status": "dry_run",
        "can_start_training": not missing,
        "missing_assets": missing,
        "run_id": run_id,
        "train_jsonl": str(args.train_jsonl) if args.train_jsonl else None,
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "base_model": args.base_model,
        "prompt_key": args.prompt_key,
        "output_dir": str(output_dir) if output_dir else None,
        "training_flow": [
            "read source JSONL directly and normalize rows lazily",
            "load Qwen-VL processor and base model",
            "attach LoRA adapters",
            (
                "build multimodal batches from screenshot + candidate_action"
                if args.prompt_key in {"localwm_next", "localwm_delta"}
                else "build multimodal batches from screenshot + current_md + candidate_action"
            ),
            "mask prompt/padding tokens",
            "run transformers.Trainer",
            "after each saved checkpoint, generate EleF1 predictions for the configured evaluation split with five action buckets",
            "save final_adapter and processor",
        ],
        "wm_eval_jsonl": str(args.wm_eval_jsonl) if args.wm_eval_jsonl else None,
        "wm_eval_splits": {split: str(path) for split, path in wm_evaluation_split_paths(args).items()},
        "wm_eval_max_samples": args.wm_eval_max_samples,
        "wm_eval_max_new_tokens": args.wm_eval_max_new_tokens,
        "wm_eval_workers_per_gpu": args.wm_eval_workers_per_gpu,
        "wm_eval_embedding_model": args.wm_eval_embedding_model,
        "wm_eval_embedding_device": args.wm_eval_embedding_device,
        "skip_overlong_samples": args.skip_overlong_samples,
        "dataset_loading_mode": args.dataset_loading_mode,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "length_filter_report_limit": args.length_filter_report_limit,
    }


def config_defaults_for_args(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    config = load_run_config(config_path)
    defaults: dict[str, Any] = {
        "train_jsonl": optional_path(config.data.train_jsonl),
        "eval_jsonl": optional_path(config.data.eval_jsonl),
        "wm_eval_in_domain_jsonl": optional_path(config.data.eval_in_domain_jsonl),
        "wm_eval_ood_jsonl": optional_path(config.data.eval_ood_jsonl),
        "output_dir": optional_path(config.training.output_dir or config.data.output),
        "base_model": config.training.base_model or None,
        "run_id": config.training.run_id or None,
        "max_train_samples": config.data.max_train_samples,
        "max_eval_samples": config.data.max_eval_samples,
    }
    if config.training.stage == "sft":
        defaults |= {
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "num_train_epochs": config.training.num_train_epochs,
            "save_steps": config.training.save_steps,
            "eval_steps": config.training.eval_steps,
            "logging_steps": config.training.logging_steps,
            "save_total_limit": config.training.save_total_limit,
            "lora_r": config.training.lora_r,
            "lora_alpha": config.training.lora_alpha,
            "lora_dropout": config.training.lora_dropout,
            "target_modules": config.training.target_modules,
            "max_length": config.training.max_length,
            "max_input_tokens": getattr(config.training, "max_input_tokens", None),
            "max_output_tokens": getattr(config.training, "max_output_tokens", None),
            "image_min_pixels": config.training.image_min_pixels,
            "image_max_pixels": config.training.image_max_pixels,
            "min_supervised_tokens": config.training.min_supervised_tokens,
            "assistant_loss_scope": config.training.assistant_loss_scope,
            "max_current_md_chars": config.training.max_current_md_chars,
            "max_target_next_md_chars": getattr(config.training, "max_target_next_md_chars", None),
            "max_assistant_output_chars": config.training.max_assistant_output_chars,
            "device_mode": config.training.device_mode,
            "gradient_checkpointing": config.training.gradient_checkpointing,
            "bf16": config.training.bf16,
            "fp16": config.training.fp16,
            "dataloader_num_workers": config.training.dataloader_num_workers,
            "dataloader_prefetch_factor": config.training.dataloader_prefetch_factor,
            "dataloader_pin_memory": config.training.dataloader_pin_memory,
            "dataloader_persistent_workers": config.training.dataloader_persistent_workers,
        }
    return defaults


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Local World Model LoRA SFT.")
    parser.add_argument("--config", type=optional_path)
    parser.add_argument("--train-jsonl", "--train_jsonl", type=optional_path)
    parser.add_argument("--eval-jsonl", "--eval_jsonl", type=optional_path)
    parser.add_argument("--output-dir", "--output_dir", type=optional_path)
    parser.add_argument("--base-model", "--base_model")
    parser.add_argument("--prompt-key", default="world_local")
    parser.add_argument("--run-id", "--run_id")
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", type=optional_path)
    parser.add_argument("--init-adapter", type=optional_path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--max-length", type=int, default=20000)
    parser.add_argument("--image-min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--image-max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--min-supervised-tokens", type=int, default=16)
    parser.add_argument("--assistant-loss-scope", choices=["markdown_only", "full"], default="full")
    parser.add_argument("--max-current-md-chars", type=int)
    parser.add_argument("--max-target-next-md-chars", type=int)
    parser.add_argument("--max-assistant-output-chars", type=int)
    parser.add_argument("--dataset-loading-mode", choices=["direct", "prepared"], default="direct")
    parser.add_argument("--skip-missing-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-overlong-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--length-filter-report-limit", type=int, default=1000)
    parser.add_argument("--device-mode", choices=["auto", "ddp"], default="auto")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int)
    parser.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wm-eval-jsonl", type=optional_path)
    parser.add_argument("--wm-eval-in-domain-jsonl", type=optional_path)
    parser.add_argument("--wm-eval-ood-jsonl", type=optional_path)
    parser.add_argument("--wm-eval-only", action="store_true")
    parser.add_argument("--wm-eval-checkpoint", type=optional_path)
    parser.add_argument("--wm-eval-output-dir", type=optional_path)
    parser.add_argument("--wm-eval-step", type=int)
    parser.add_argument("--wm-eval-max-samples", type=int)
    parser.add_argument("--wm-eval-max-new-tokens", type=int, default=6000)
    parser.add_argument("--wm-eval-workers-per-gpu", type=int, default=1)
    parser.add_argument("--wm-eval-ddp-timeout-seconds", type=int, default=21600)
    parser.add_argument("--wm-eval-embedding-model")
    parser.add_argument("--wm-eval-embedding-device", default="cpu")
    parser.add_argument("--wm-eval-min-format-success-rate", type=float)
    parser.add_argument("--wm-eval-max-elef1-drop-from-best", type=float)
    parser.add_argument("--wm-eval-stop-after-bad-checkpoints", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    provided_dests = provided_cli_dests(parser, argv)
    args = parser.parse_args(argv)
    apply_config_defaults(args, config_defaults_for_args(args.config), provided_dests)
    if args.wm_eval_only and args.dry_run:
        print(json.dumps(build_wm_checkpoint_eval_dry_run_report(args), ensure_ascii=False, indent=2, default=str))
        return
    if args.dry_run:
        print(json.dumps(build_dry_run_report(args), ensure_ascii=False, indent=2, default=str))
        return

    def execute():
        if args.wm_eval_only:
            metrics_paths = run_wm_checkpoint_eval(args)
            if int(os.environ.get("RANK", "0")) == 0:
                print(json.dumps({"metrics_paths": {split: str(path) for split, path in metrics_paths.items()}}, ensure_ascii=False))
            return
        final_dir = run_training(args)
        print(json.dumps({"final_adapter": str(final_dir)}, ensure_ascii=False))

    run_with_distributed_cleanup(execute)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
