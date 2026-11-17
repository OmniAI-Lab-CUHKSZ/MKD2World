from __future__ import annotations

import gc
import json
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mkd2world.contracts import CandidateAction, GuiState, LocalPrediction
from mkd2world.training.delta import apply_delta_flat, parse_delta
from mkd2world.training.prompts import apply_chat_template, build_local_world_model_messages
from mkd2world.world_model import local_guidance_mode, local_memory_reference, split_local_prediction


def shared_context_output_limit(
    *,
    max_length: int,
    model_context_window: int,
    prompt_width: int,
    max_new_tokens: int,
) -> int:
    """Return the generation cap under one shared input+output context window."""

    effective_context = min(int(max_length), int(model_context_window))
    remaining = effective_context - int(prompt_width)
    if remaining <= 0:
        raise ValueError(
            f"prompt uses {prompt_width} tokens and exceeds max_length={effective_context}"
        )
    return min(int(max_new_tokens), remaining)


def decode_local_prediction(
    raw_output: str,
    *,
    current_md: str,
    prompt_key: str,
    delta_state_mode: str = "replayed_next",
) -> tuple[str, str]:
    """Decode a Local WM output and materialize Delta predictions to next MD.

    Delta parsing and replay are deliberately strict: malformed or stale
    operations raise instead of silently returning the current state.
    """

    if delta_state_mode not in {"raw_delta", "replayed_next"}:
        raise ValueError(f"Unsupported delta_state_mode={delta_state_mode!r}")
    analysis, predicted = split_local_prediction(raw_output)
    if prompt_key == "localwm_delta":
        delta = parse_delta(predicted)
        if delta_state_mode == "raw_delta":
            # Replay anyway so invalid deltas fail as in full mode.
            apply_delta_flat(current_md, delta)
            return analysis, predicted
        predicted = apply_delta_flat(current_md, delta)
    elif delta_state_mode != "replayed_next":
        raise ValueError("raw_delta state mode requires prompt_key='localwm_delta'")
    if not predicted.startswith("[Environment:"):
        raise ValueError(f"Local WM returned invalid Markdown prefix: {predicted[:120]!r}")
    return analysis, predicted


@dataclass
class TransformersLocalWorldModel:
    base_model: str | Path
    adapter_path: str | Path | None
    base_only: bool = False
    device_map: str | dict[str, str] = "auto"
    torch_dtype: str = "bfloat16"
    max_length: int = 60000
    # Evaluation runs set 6000; generation is clamped to ``max_length - prompt_width``.
    max_new_tokens: int = 60000
    image_min_pixels: int = 256 * 28 * 28
    image_max_pixels: int = 1024 * 28 * 28
    batch_size: int = 1
    prompt_key: str = "world_local"
    delta_state_mode: str = "replayed_next"
    _torch: Any = field(default=None, init=False, repr=False)
    _image_module: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _lifecycle_lock: Any = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _inference_lock: Any = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _telemetry: dict[str, int] = field(
        default_factory=lambda: {
            "calls": 0,
            "candidates": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_prompt_tokens": 0,
            "max_completion_tokens": 0,
            "output_limit_hits": 0,
            "context_limited_calls": 0,
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.prompt_key not in {"world_local", "localwm_next", "localwm_delta"}:
            raise ValueError(
                "Local WM prompt_key must be 'world_local', 'localwm_next', or 'localwm_delta'"
            )
        if self.delta_state_mode not in {"raw_delta", "replayed_next"}:
            raise ValueError(
                "delta_state_mode must be 'raw_delta' or 'replayed_next'"
            )
        if self.prompt_key != "localwm_delta" and self.delta_state_mode != "replayed_next":
            raise ValueError("raw_delta state mode requires prompt_key='localwm_delta'")

    def _resolve_prompt_key(self, reference: object) -> str:
        if self.prompt_key in {"localwm_next", "localwm_delta"}:
            if reference:
                raise ValueError(
                    f"{self.prompt_key} does not accept retrieved Markdown memory"
                )
            return self.prompt_key
        return "world_local_with_memory" if reference else "world_local"

    @staticmethod
    def _prompt_sample(
        *,
        current_state: GuiState,
        image_path: str,
        candidate_action: CandidateAction,
        guidance_mode: str,
        reference: object,
        prompt_key: str,
    ) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "current_image_path": image_path,
            "candidate_action_json": json.dumps(
                candidate_action.unified_action.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "guidance_mode": guidance_mode if reference else "",
            "retrieved_memory": reference,
        }
        if prompt_key not in {"localwm_next", "localwm_delta"}:
            sample["current_md"] = current_state.markdown
        return sample

    def telemetry_snapshot(self) -> dict[str, int]:
        return {**self._telemetry, "configured_max_new_tokens": int(self.max_new_tokens)}

    def _record_generation(self, *, prompt_width: int, generated_width: int, output_limit: int, candidates: int) -> None:
        self._telemetry["calls"] += 1
        self._telemetry["candidates"] += int(candidates)
        self._telemetry["prompt_tokens"] += int(prompt_width) * int(candidates)
        self._telemetry["completion_tokens"] += int(generated_width) * int(candidates)
        self._telemetry["max_prompt_tokens"] = max(self._telemetry["max_prompt_tokens"], int(prompt_width))
        self._telemetry["max_completion_tokens"] = max(
            self._telemetry["max_completion_tokens"], int(generated_width)
        )
        if generated_width >= output_limit:
            self._telemetry["output_limit_hits"] += int(candidates)
        if output_limit < self.max_new_tokens:
            self._telemetry["context_limited_calls"] += 1

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lifecycle_lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if self.loaded:
            return
        base_model = Path(self.base_model)
        if not base_model.is_dir():
            raise FileNotFoundError(base_model)
        if self.base_only:
            if self.adapter_path is not None:
                raise ValueError("base-only Local WM must not configure an adapter")
            adapter_path = None
        else:
            if self.adapter_path is None:
                raise ValueError("adapter-backed Local WM requires an adapter path")
            adapter_path = Path(self.adapter_path)
            if not adapter_path.is_dir() or not (adapter_path / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(f"Local WM adapter is incomplete: {adapter_path}")
        from mkd2world.training.sft import import_training_dependencies

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
        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(self.torch_dtype.lower())
        if dtype is None:
            raise ValueError(f"Unsupported Local WM torch dtype: {self.torch_dtype}")
        processor = AutoProcessor.from_pretrained(
            str(base_model),
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
            trust_remote_code=True,
        )
        # Batched decoder-only generation needs left padding.
        processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(
            str(base_model),
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()
        self._torch = torch
        self._image_module = Image
        self._processor = processor
        self._model = model

    def close(self) -> None:
        # Wait for any in-flight inference before shutdown.
        with self._inference_lock:
            with self._lifecycle_lock:
                self._close_unlocked()

    def _close_unlocked(self) -> None:
        """Release model references and the CUDA allocator owned by this runtime."""

        model = self._model
        torch = self._torch
        self._model = None
        self._processor = None
        self._image_module = None
        if model is not None:
            # Drop references instead of moving the model to CPU.
            del model
        gc.collect()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass
        self._torch = None

    def predict_next_state(
        self,
        current_state: GuiState,
        candidate_action: CandidateAction,
        *,
        memory: Sequence[dict] = (),
        guidance_mode: str | None = None,
    ) -> LocalPrediction:
        with self._inference_lock:
            return self._predict_next_state_unlocked(
                current_state,
                candidate_action,
                memory=memory,
                guidance_mode=guidance_mode,
            )

    def _predict_next_state_unlocked(
        self,
        current_state: GuiState,
        candidate_action: CandidateAction,
        *,
        memory: Sequence[dict] = (),
        guidance_mode: str | None = None,
    ) -> LocalPrediction:
        self.load()
        assert self._torch is not None and self._processor is not None and self._model is not None
        mode = guidance_mode or local_guidance_mode(memory)
        reference = local_memory_reference(memory)
        prompt_key = self._resolve_prompt_key(reference)
        with tempfile.TemporaryDirectory(prefix="mkd2world_local_wm_") as tmp:
            image_path = ""
            if current_state.screenshot:
                image_path = str(Path(tmp) / "current.png")
                Path(image_path).write_bytes(current_state.screenshot)
            sample = self._prompt_sample(
                current_state=current_state,
                image_path=image_path,
                candidate_action=candidate_action,
                guidance_mode=mode,
                reference=reference,
                prompt_key=prompt_key,
            )
            messages = build_local_world_model_messages(sample, include_output=False, prompt_key=prompt_key)
            text = apply_chat_template(self._processor, messages, tokenize=False, add_generation_prompt=True)
            processor_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
            image = None
            try:
                if image_path:
                    image = self._image_module.open(image_path).convert("RGB")
                    processor_kwargs["images"] = [image]
                batch = self._processor(**processor_kwargs)
                device = next(self._model.parameters()).device
                batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                prompt_width = int(batch["input_ids"].shape[1])
                context_window = int(getattr(self._model.config, "max_position_embeddings", self.max_length))
                output_limit = shared_context_output_limit(
                    max_length=self.max_length,
                    model_context_window=context_window,
                    prompt_width=prompt_width,
                    max_new_tokens=self.max_new_tokens,
                )
                with self._torch.no_grad():
                    sequences = self._model.generate(
                        **batch,
                        do_sample=False,
                        max_new_tokens=output_limit,
                        pad_token_id=getattr(self._processor.tokenizer, "pad_token_id", None),
                        eos_token_id=getattr(self._processor.tokenizer, "eos_token_id", None),
                    )
                self._record_generation(
                    prompt_width=prompt_width,
                    generated_width=int(sequences.shape[1]) - prompt_width,
                    output_limit=output_limit,
                    candidates=1,
                )
                raw = self._processor.batch_decode(sequences[:, prompt_width:], skip_special_tokens=True)[0].strip()
            finally:
                if image is not None:
                    image.close()
        analysis, predicted_markdown = decode_local_prediction(
            raw,
            current_md=current_state.markdown,
            prompt_key=prompt_key,
            delta_state_mode=self.delta_state_mode,
        )
        predicted_delta = (
            split_local_prediction(raw)[1]
            if prompt_key == "localwm_delta"
            else None
        )
        return LocalPrediction(
            candidate_action=candidate_action,
            action_analysis=analysis,
            predicted_markdown=predicted_markdown,
            raw_output=raw,
            predicted_delta=predicted_delta,
            output_state_mode=self.delta_state_mode,
        )

    def predict_next_states(
        self,
        current_state: GuiState,
        candidate_actions: Sequence[CandidateAction],
        *,
        memories: Sequence[Sequence[dict]] | None = None,
        guidance_modes: Sequence[str | None] | None = None,
    ) -> list[LocalPrediction]:
        items = [(current_state, action) for action in candidate_actions]
        with self._inference_lock:
            return self._predict_items_unlocked(
                items,
                memories=memories,
                guidance_modes=guidance_modes,
            )

    def predict_mixed_states(
        self,
        items: Sequence[tuple[GuiState, CandidateAction]],
        *,
        memories: Sequence[Sequence[dict]] | None = None,
        guidance_modes: Sequence[str | None] | None = None,
    ) -> list[LocalPrediction]:
        """Predict one candidate per possibly-distinct GUI state in a single batch."""
        with self._inference_lock:
            return self._predict_items_unlocked(
                items,
                memories=memories,
                guidance_modes=guidance_modes,
            )

    def _predict_items_unlocked(
        self,
        items: Sequence[tuple[GuiState, CandidateAction]],
        *,
        memories: Sequence[Sequence[dict]] | None = None,
        guidance_modes: Sequence[str | None] | None = None,
    ) -> list[LocalPrediction]:
        """Predict mixed or same-state candidates without reordering outputs."""
        rows = list(items)
        if not rows:
            return []
        actions = [action for _, action in rows]
        states = [state for state, _ in rows]
        memory_rows = list(memories or [()] * len(rows))
        modes = list(guidance_modes or [None] * len(rows))
        if len(memory_rows) != len(rows) or len(modes) != len(rows):
            raise ValueError("Local WM batched actions, memories, and guidance modes must align")
        if self.batch_size < 1:
            raise ValueError("Local WM batch_size must be at least 1")
        self.load()
        assert self._torch is not None and self._processor is not None and self._model is not None
        predictions: list[LocalPrediction] = []
        with tempfile.TemporaryDirectory(prefix="mkd2world_local_wm_batch_") as tmp:
            for offset in range(0, len(rows), self.batch_size):
                chunk_states = states[offset : offset + self.batch_size]
                chunk_actions = actions[offset : offset + self.batch_size]
                chunk_memories = memory_rows[offset : offset + self.batch_size]
                chunk_modes = modes[offset : offset + self.batch_size]
                have_images = [bool(state.screenshot) for state in chunk_states]
                if any(have_images) and not all(have_images):
                    raise ValueError("Mixed WM batch cannot mix screenshot and screenshot-less items")
                texts = []
                prompt_keys: list[str] = []
                image_paths: list[str] = []
                for index, (state, action, memory, requested_mode) in enumerate(
                    zip(chunk_states, chunk_actions, chunk_memories, chunk_modes)
                ):
                    mode = requested_mode or local_guidance_mode(memory)
                    reference = local_memory_reference(memory)
                    prompt_key = self._resolve_prompt_key(reference)
                    prompt_keys.append(prompt_key)
                    image_path = ""
                    if state.screenshot:
                        image_path = str(Path(tmp) / f"current_{offset}_{index}.png")
                        Path(image_path).write_bytes(state.screenshot)
                    image_paths.append(image_path)
                    sample = self._prompt_sample(
                        current_state=state,
                        image_path=image_path,
                        candidate_action=action,
                        guidance_mode=mode,
                        reference=reference,
                        prompt_key=prompt_key,
                    )
                    messages = build_local_world_model_messages(sample, include_output=False, prompt_key=prompt_key)
                    texts.append(
                        apply_chat_template(self._processor, messages, tokenize=False, add_generation_prompt=True)
                    )
                processor_kwargs: dict[str, Any] = {
                    "text": texts,
                    "return_tensors": "pt",
                    "padding": True,
                }
                images = []
                try:
                    if any(have_images):
                        images = [
                            self._image_module.open(path).convert("RGB")
                            for path in image_paths
                        ]
                        processor_kwargs["images"] = images
                    batch = self._processor(**processor_kwargs)
                    device = next(self._model.parameters()).device
                    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                    prompt_width = int(batch["input_ids"].shape[1])
                    context_window = int(getattr(self._model.config, "max_position_embeddings", self.max_length))
                    output_limit = shared_context_output_limit(
                        max_length=self.max_length,
                        model_context_window=context_window,
                        prompt_width=prompt_width,
                        max_new_tokens=self.max_new_tokens,
                    )
                    with self._torch.no_grad():
                        sequences = self._model.generate(
                            **batch,
                            do_sample=False,
                            max_new_tokens=output_limit,
                            pad_token_id=getattr(self._processor.tokenizer, "pad_token_id", None),
                            eos_token_id=getattr(self._processor.tokenizer, "eos_token_id", None),
                        )
                    self._record_generation(
                        prompt_width=prompt_width,
                        generated_width=int(sequences.shape[1]) - prompt_width,
                        output_limit=output_limit,
                        candidates=len(chunk_actions),
                    )
                    outputs = self._processor.batch_decode(sequences[:, prompt_width:], skip_special_tokens=True)
                    for state, action, raw_output, item_prompt_key in zip(
                        chunk_states, chunk_actions, outputs, prompt_keys
                    ):
                        raw_output = raw_output.strip()
                        analysis, predicted_markdown = decode_local_prediction(
                            raw_output,
                            current_md=state.markdown,
                            prompt_key=item_prompt_key,
                            delta_state_mode=self.delta_state_mode,
                        )
                        predictions.append(
                            LocalPrediction(
                                candidate_action=action,
                                action_analysis=analysis,
                                predicted_markdown=predicted_markdown,
                                raw_output=raw_output,
                                predicted_delta=(
                                    split_local_prediction(raw_output)[1]
                                    if item_prompt_key == "localwm_delta"
                                    else None
                                ),
                                output_state_mode=self.delta_state_mode,
                            )
                        )
                finally:
                    for image in images:
                        image.close()
        if [item.candidate_action for item in predictions] != actions:
            raise RuntimeError("Local WM batched output order diverged from candidate action order")
        return predictions
