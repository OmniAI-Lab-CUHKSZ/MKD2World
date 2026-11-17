"""Stage-2 MKD2World training: next-TSM element loss plus a changed-only page margin.

The loss on the next TSM is element-averaged base CE plus element-averaged focus CE.
For changed transitions whose next and current targets differ, a hinge pushes the
page CE of the next target below the page CE of the current target by ``MARGIN``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from mkd2world.training import focus_contract
from mkd2world.training import sft as sft_module
from mkd2world.training.focus_contract import CharacterSpan, ValidatedTrainingSidecar
from mkd2world.training.markdown import build_assistant_output
from mkd2world.training.prompts import apply_chat_template, build_local_world_model_messages

try:
    from transformers import Trainer as _TransformersTrainer
except ImportError:

    class _TransformersTrainer:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError('NextTsmTrainer requires transformers')

BASE_UNIT_IDS_KEY = 'base_unit_ids'
FOCUS_UNIT_IDS_KEY = 'focus_unit_ids'
CURRENT_INPUTS_KEY = 'current_inputs'
RANK_ELIGIBLE_KEY = 'rank_eligible'
IDENTITY_EXCLUDED_KEY = 'identity_excluded'
SAMPLE_IDS_KEY = 'sample_ids'
TARGET_TOKEN_IDS_KEY = '_target_token_ids'
IGNORE_UNIT_ID = -1
MAX_FOCUS_LAMBDA = 3.0
MARGIN = 0.1
BETA_MAX = 0.1
BETA_WARMUP_STEPS = 100
REPLAY_ATOL = 1e-05


class NextTsmSftError(ValueError):
    """Raised when labels, token alignment, or loss inputs are invalid."""


def canonical_next_target(sample: Mapping[str, Any]) -> str:
    target_markdown = sample.get('target_next_md')
    if not isinstance(target_markdown, str) or not target_markdown.strip():
        raise NextTsmSftError(f"sample {sample.get('id') or '<unknown>'} has an empty target_next_md")
    return build_assistant_output('', target_markdown)


def canonical_current_target(sample: Mapping[str, Any]) -> str:
    current_markdown = sample.get('current_md')
    if not isinstance(current_markdown, str) or not current_markdown.strip():
        raise NextTsmSftError(f"sample {sample.get('id') or '<unknown>'} has an empty current_md")
    return build_assistant_output('', current_markdown)


def build_current_sidecar(sample: Mapping[str, Any]) -> ValidatedTrainingSidecar:
    sample_id = str(sample.get('id') or '').strip()
    if not sample_id:
        raise NextTsmSftError('training sample is missing id')
    target_text = canonical_current_target(sample)
    try:
        parsed = focus_contract.focus_labeler.parse_markdown_spans(target_text)
    except focus_contract.focus_labeler.V4FocusError as exc:
        raise NextTsmSftError(f'sample {sample_id} current_md violates the TSM parser: {exc}') from exc
    if parsed.text != target_text or not parsed.base_units:
        raise NextTsmSftError(f'sample {sample_id} current target parsing drifted')
    return ValidatedTrainingSidecar(sample_id=sample_id, target_text=target_text, base_spans=tuple((CharacterSpan(span.start, span.end) for span in parsed.base_units)), focus_spans=())


def validate_sidecar(sample: Mapping[str, Any], raw_sidecar: Mapping[str, Any]) -> ValidatedTrainingSidecar:
    try:
        validated = focus_contract.validate_training_sidecar(sample, raw_sidecar)
    except focus_contract.V4TrainingSidecarError as exc:
        raise NextTsmSftError(f"sample {sample.get('id') or '<unknown>'} violates the focus sidecar contract: {exc}") from exc
    if validated.target_text != canonical_next_target(sample):
        raise NextTsmSftError(f'sample {validated.sample_id} sidecar target is not raw target_next_md')
    return validated


def load_focus_sidecar(path: str | Path) -> dict[str, dict[str, Any]]:
    sidecar_path = Path(path)
    rows: dict[str, dict[str, Any]] = {}
    with sidecar_path.open(encoding='utf-8') as handle:
        for (line_number, line) in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get('sample_id') or '').strip() if isinstance(row, dict) else ''
            if not sample_id:
                raise NextTsmSftError(f'missing sidecar sample_id at {sidecar_path}:{line_number}')
            if sample_id in rows:
                raise NextTsmSftError(f'duplicate sidecar sample_id {sample_id!r}')
            rows[sample_id] = row
    if not rows:
        raise NextTsmSftError(f'sidecar has no rows: {sidecar_path}')
    return rows


def build_focus_sidecar(train_jsonl: str | Path, output: str | Path) -> dict[str, int]:
    """Write one compact focus sidecar row per training row."""
    counts = {'rows': 0, 'focus_units': 0}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(train_jsonl).open(encoding='utf-8') as source, output_path.open('w', encoding='utf-8') as sink:
        for line in source:
            if not line.strip():
                continue
            compact = focus_contract.build_training_sidecar(_normalize_training_row(json.loads(line)))
            sink.write(json.dumps(compact, ensure_ascii=False) + '\n')
            counts['rows'] += 1
            counts['focus_units'] += len(compact['focus_units'])
    return counts


def _single_tokenizer_output(value: object, *, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, list):
        raise NextTsmSftError(f'tokenizer {name} must be a list')
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise NextTsmSftError(f'tokenizer {name} unexpectedly contains a batch')
        value = value[0]
    return value


def map_target_tokens_to_units(tokenizer: Any, sidecar: ValidatedTrainingSidecar) -> tuple[list[int], list[int], list[int]]:
    """Tokenize one target and map fast-tokenizer offsets to base and focus unit ids."""
    if not getattr(tokenizer, 'is_fast', False):
        raise NextTsmSftError('unit alignment requires a fast tokenizer with offset_mapping')
    encoded = tokenizer(sidecar.target_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = _single_tokenizer_output(encoded['input_ids'], name='input_ids')
    raw_offsets = _single_tokenizer_output(encoded['offset_mapping'], name='offset_mapping')
    if not token_ids or len(token_ids) != len(raw_offsets):
        raise NextTsmSftError('target token ids and offsets have inconsistent lengths')
    base_unit_ids: list[int] = []
    focus_unit_ids: list[int] = []
    seen_base: set[int] = set()
    seen_focus: set[int] = set()
    previous_start = 0
    covered_end = 0
    for (token_index, (start, end)) in enumerate(raw_offsets):
        start = int(start)
        end = int(end)
        if start < previous_start or start > covered_end or start < 0 or (end <= start) or (end > len(sidecar.target_text)):
            raise NextTsmSftError(f'invalid or uncovered token offset at index {token_index}: {start}:{end}/{len(sidecar.target_text)}')
        previous_start = start
        covered_end = max(covered_end, end)
        base_overlaps = [(min(end, span.end) - max(start, span.start), unit_index) for (unit_index, span) in enumerate(sidecar.base_spans) if start < span.end and end > span.start]
        if not base_overlaps:
            raise NextTsmSftError(f'target token {token_index} at {start}:{end} has no base unit')
        (_, base_id) = max(base_overlaps)
        base_unit_ids.append(base_id)
        seen_base.add(base_id)
        focus_owners = {focus_index for (focus_index, spans) in enumerate(sidecar.focus_spans) for span in spans if start < span.end and end > span.start}
        if len(focus_owners) > 1:
            raise NextTsmSftError(f'target token {token_index} overlaps multiple focus groups')
        focus_id = next(iter(focus_owners), IGNORE_UNIT_ID)
        focus_unit_ids.append(focus_id)
        if focus_id >= 0:
            seen_focus.add(focus_id)
    if covered_end != len(sidecar.target_text):
        raise NextTsmSftError(f'token offsets cover through {covered_end}, expected {len(sidecar.target_text)}')
    if seen_base != set(range(len(sidecar.base_spans))):
        raise NextTsmSftError('some base units have no tokenizer coverage')
    if seen_focus != set(range(len(sidecar.focus_spans))):
        raise NextTsmSftError('some focus units have no tokenizer coverage')
    return ([int(token_id) for token_id in token_ids], base_unit_ids, focus_unit_ids)


def _attach_unit_ids(batch: dict[str, Any], features: Sequence[Mapping[str, Any]], sidecars: Sequence[ValidatedTrainingSidecar], tokenizer: Any, *, with_focus: bool) -> None:
    """Write per-token base/focus unit ids for the supervised assistant target of each row."""
    labels = batch['labels']
    input_ids = batch['input_ids']
    if labels.ndim != 2 or labels.shape != input_ids.shape or labels.shape[0] != len(features):
        raise NextTsmSftError('input_ids and labels must have equal [batch, tokens] shapes')
    base_ids = torch.full_like(labels, IGNORE_UNIT_ID)
    focus_ids = torch.full_like(labels, IGNORE_UNIT_ID)
    aligned_token_ids: list[list[int]] = []
    for (row_index, sidecar) in enumerate(sidecars):
        sample_id = sidecar.sample_id
        (token_ids, token_base_ids, token_focus_ids) = map_target_tokens_to_units(tokenizer, sidecar)
        supervised = torch.nonzero(labels[row_index].ne(-100), as_tuple=False).flatten()
        if supervised.numel() == 0:
            raise NextTsmSftError(f'sample {sample_id} has no supervised labels')
        target_start = int(supervised[0].item())

        def target_matches(candidate_ids: Sequence[int]) -> bool:
            target_end = target_start + len(candidate_ids)
            if target_end > labels.shape[1]:
                return False
            expected = torch.as_tensor(candidate_ids, dtype=input_ids.dtype, device=input_ids.device)
            return torch.equal(input_ids[row_index, target_start:target_end], expected) and torch.equal(labels[row_index, target_start:target_end].to(input_ids.dtype), expected)
        if not target_matches(token_ids):
            if not sidecar.target_text.endswith('\n'):
                raise NextTsmSftError(f'sample {sample_id} target token ids do not match the processor')
            trimmed = ValidatedTrainingSidecar(sample_id=sample_id, target_text=sidecar.target_text[:-1], base_spans=sidecar.base_spans, focus_spans=sidecar.focus_spans)
            (token_ids, token_base_ids, token_focus_ids) = map_target_tokens_to_units(tokenizer, trimmed)
            if not target_matches(token_ids):
                raise NextTsmSftError(f'sample {sample_id} target token ids do not match the processor')
        aligned_token_ids.append(list(token_ids))
        target_end = target_start + len(token_ids)
        base_ids[row_index, target_start:target_end] = torch.as_tensor(token_base_ids, dtype=base_ids.dtype, device=base_ids.device)
        if with_focus:
            focus_ids[row_index, target_start:target_end] = torch.as_tensor(token_focus_ids, dtype=focus_ids.dtype, device=focus_ids.device)
        suffix = torch.nonzero(labels[row_index, target_end:].ne(-100), as_tuple=False).flatten()
        if suffix.numel() > 0:
            suffix = suffix + target_end
            expected_suffix = torch.arange(target_end, int(suffix[-1].item()) + 1, device=suffix.device, dtype=suffix.dtype)
            if not torch.equal(suffix, expected_suffix):
                raise NextTsmSftError(f'sample {sample_id} has a supervision gap in the chat suffix')
            if not torch.equal(labels[row_index, suffix].to(input_ids.dtype), input_ids[row_index, suffix]):
                raise NextTsmSftError(f'sample {sample_id} chat suffix labels do not match input_ids')
            base_ids[row_index, suffix] = 0
    batch[BASE_UNIT_IDS_KEY] = base_ids
    if with_focus:
        batch[FOCUS_UNIT_IDS_KEY] = focus_ids
    batch[TARGET_TOKEN_IDS_KEY] = aligned_token_ids


class NextCollator:
    """``LocalWMSftCollator`` for ``localwm_next`` plus base/focus unit ids from the sidecar."""

    def __init__(self, base_collator: Any, sidecars: Mapping[str, Mapping[str, Any]]) -> None:
        if getattr(base_collator, 'prompt_key', None) != 'localwm_next':
            raise NextTsmSftError("NextCollator requires LocalWMSftCollator(prompt_key='localwm_next')")
        self.base_collator = base_collator
        self.sidecars = sidecars
        self.tokenizer = base_collator.processor.tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(features)
        sidecars = []
        for sample in features:
            sample_id = str(sample.get('id') or '').strip()
            raw_sidecar = self.sidecars.get(sample_id)
            if raw_sidecar is None:
                raise NextTsmSftError(f'missing sidecar row for sample {sample_id}')
            sidecars.append(validate_sidecar(sample, raw_sidecar))
        _attach_unit_ids(batch, features, sidecars, self.tokenizer, with_focus=True)
        return batch


@dataclass
class CurrentCollator:
    """Encode ``current_md`` as the assistant target under the same ``localwm_next`` prompt."""
    processor: Any
    image_module: Any
    max_length: int
    max_output_tokens: int | None
    min_supervised_tokens: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        assistant_targets: list[str] = []
        images: list[Any] = []
        sample_ids: list[str] = []
        for sample in features:
            prompt_messages = build_local_world_model_messages(sample, include_output=False, prompt_key='localwm_next')
            current_target = canonical_current_target(sample)
            training_messages = [*prompt_messages, {'role': 'assistant', 'content': [{'type': 'text', 'text': current_target}]}]
            prompt_texts.append(apply_chat_template(self.processor, prompt_messages, tokenize=False, add_generation_prompt=True))
            texts.append(apply_chat_template(self.processor, training_messages, tokenize=False, add_generation_prompt=False))
            assistant_targets.append(current_target)
            images.append(self.image_module.open(sample['current_image_path']).convert('RGB'))
            sample_ids.append(str(sample.get('id') or '<unknown>'))
        if self.max_output_tokens is not None:
            overlong = [(sample_id, length) for (sample_id, length) in zip(sample_ids, (sft_module.token_count(self.processor.tokenizer, target) for target in assistant_targets)) if length > self.max_output_tokens]
            if overlong:
                raise NextTsmSftError(f'current target exceeds max_output_tokens={self.max_output_tokens}: {overlong}')
        batch = self.processor(text=texts, images=images, return_tensors='pt', padding=True, truncation=False)
        prompt_batch = self.processor(text=prompt_texts, images=images, return_tensors='pt', padding=True, truncation=False)
        overlong_sequences = [(sample_id, int(mask.sum().item())) for (sample_id, mask) in zip(sample_ids, batch['attention_mask']) if int(mask.sum().item()) > self.max_length]
        if overlong_sequences:
            raise NextTsmSftError(f'current training sequence exceeds max_length={self.max_length}: {overlong_sequences}')
        labels = batch['input_ids'].clone()
        pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        sft_module.mask_prompt_tokens(labels, batch['attention_mask'], prompt_batch['attention_mask'], tokenizer=self.processor.tokenizer, assistant_targets=assistant_targets, assistant_loss_scope='full', padding_side=getattr(self.processor.tokenizer, 'padding_side', 'right'), min_supervised_tokens=self.min_supervised_tokens, sample_ids=sample_ids)
        batch['labels'] = labels
        _attach_unit_ids(batch, features, [build_current_sidecar(sample) for sample in features], self.processor.tokenizer, with_focus=False)
        return batch


def _prompt_prefixes(batch: Mapping[str, Any]) -> list[list[int]]:
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch.get('attention_mask')
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    prefixes: list[list[int]] = []
    for (row_inputs, row_labels, row_mask) in zip(input_ids, labels, attention_mask):
        valid = row_mask.to(dtype=torch.bool)
        compact_inputs = row_inputs[valid]
        supervised = row_labels[valid].ne(-100).nonzero(as_tuple=False)
        if supervised.numel() == 0:
            raise NextTsmSftError('paired target contains no supervised token')
        prefixes.append(compact_inputs[:int(supervised[0].item())].tolist())
    return prefixes


class PairedCollator:
    """Batch the next target and the current target that share one prompt."""

    def __init__(self, next_collator: NextCollator, current_collator: CurrentCollator) -> None:
        self.next_collator = next_collator
        self.current_collator = current_collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        next_batch = self.next_collator(features)
        current_batch = self.current_collator(features)
        next_target_ids = next_batch.pop(TARGET_TOKEN_IDS_KEY)
        current_target_ids = current_batch.pop(TARGET_TOKEN_IDS_KEY)
        if _prompt_prefixes(next_batch) != _prompt_prefixes(current_batch):
            raise NextTsmSftError('next/current prompt token prefixes differ')
        for key in ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'):
            next_value = next_batch.get(key)
            current_value = current_batch.get(key)
            if isinstance(next_value, torch.Tensor) or isinstance(current_value, torch.Tensor):
                if not (isinstance(next_value, torch.Tensor) and isinstance(current_value, torch.Tensor) and torch.equal(next_value, current_value)):
                    raise NextTsmSftError(f'next/current source tensor {key!r} differs')
        eligible: list[bool] = []
        identity_excluded: list[bool] = []
        for (sample, next_ids, current_ids) in zip(features, next_target_ids, current_target_ids):
            changed = str(sample.get('transition_class') or '') == 'changed'
            identical = list(next_ids) == list(current_ids)
            eligible.append(changed and (not identical))
            identity_excluded.append(changed and identical)
        next_batch[CURRENT_INPUTS_KEY] = current_batch
        next_batch[RANK_ELIGIBLE_KEY] = torch.tensor(eligible, dtype=torch.bool)
        next_batch[IDENTITY_EXCLUDED_KEY] = torch.tensor(identity_excluded, dtype=torch.bool)
        next_batch[SAMPLE_IDS_KEY] = [str(sample.get('id') or '').strip() for sample in features]
        return next_batch


def _validate_loss_inputs(token_ce: torch.Tensor, base_unit_ids: torch.Tensor, focus_unit_ids: torch.Tensor) -> None:
    if token_ce.ndim != 2 or base_unit_ids.shape != token_ce.shape or focus_unit_ids.shape != token_ce.shape:
        raise NextTsmSftError('token CE and unit id tensors must have equal [batch, tokens] shapes')
    if bool(base_unit_ids.lt(IGNORE_UNIT_ID).any()) or bool(focus_unit_ids.lt(IGNORE_UNIT_ID).any()):
        raise NextTsmSftError('unit ids may only use -1 as the ignore value')
    if bool(focus_unit_ids.ge(0).logical_and(base_unit_ids.lt(0)).any()):
        raise NextTsmSftError('every focus token must also belong to a base unit')


def _scatter_unit_means(token_ce: torch.Tensor, unit_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each sample's sum of unit means and number of non-empty units."""
    (batch_size, token_count) = token_ce.shape
    stride = token_count + 1
    valid = unit_ids.ge(0)
    if bool(unit_ids.masked_fill(~valid, 0).ge(token_count).any()):
        raise NextTsmSftError('unit id exceeds the number of available token positions')
    working_ce = token_ce.float() if token_ce.dtype in {torch.float16, torch.bfloat16} else token_ce
    batch_offsets = torch.arange(batch_size, device=unit_ids.device, dtype=unit_ids.dtype)
    flat_indexes = (unit_ids + batch_offsets[:, None] * stride)[valid].to(torch.long)
    flat_values = working_ce[valid]
    sums = working_ce.new_zeros(batch_size * stride)
    counts = working_ce.new_zeros(batch_size * stride)
    sums.scatter_add_(0, flat_indexes, flat_values)
    counts.scatter_add_(0, flat_indexes, torch.ones_like(flat_values))
    sums = sums.view(batch_size, stride)
    counts = counts.view(batch_size, stride)
    active = counts.gt(0)
    means = sums / counts.clamp_min(1)
    return (means.masked_fill(~active, 0).sum(dim=1), active.sum(dim=1))


def element_loss(token_ce: torch.Tensor, base_unit_ids: torch.Tensor, focus_unit_ids: torch.Tensor) -> torch.Tensor:
    """Element-averaged base CE plus element-averaged focus CE weighted by sqrt(N/C), capped at 3."""
    _validate_loss_inputs(token_ce, base_unit_ids, focus_unit_ids)
    (base_sum, base_count) = _scatter_unit_means(token_ce, base_unit_ids)
    (focus_sum, focus_count) = _scatter_unit_means(token_ce, focus_unit_ids)
    if bool(base_count.eq(0).any()):
        raise NextTsmSftError('every sample must contain at least one base unit')
    n_units = base_count.to(base_sum.dtype)
    c_units = focus_count.to(base_sum.dtype)
    focus_lambda = torch.where(focus_count.gt(0), torch.sqrt(n_units / c_units.clamp_min(1)).clamp(max=MAX_FOCUS_LAMBDA), torch.zeros_like(n_units))
    per_sample = (base_sum + focus_lambda * focus_sum) / (n_units + focus_lambda * c_units)
    return per_sample.mean()


def page_element_ce(token_ce: torch.Tensor, base_unit_ids: torch.Tensor) -> torch.Tensor:
    """Average tokens inside each base element, then elements inside each page."""
    _validate_loss_inputs(token_ce, base_unit_ids, torch.full_like(base_unit_ids, IGNORE_UNIT_ID))
    (base_sum, base_count) = _scatter_unit_means(token_ce, base_unit_ids)
    if bool(base_count.eq(0).any()):
        raise NextTsmSftError('every page score must contain at least one base element')
    return base_sum / base_count.to(base_sum.dtype)


def margin_beta(global_step: int) -> float:
    return BETA_MAX * min(1.0, (int(global_step) + 1) / BETA_WARMUP_STEPS)


def ranking_terms(next_page_ce: torch.Tensor, current_page_ce: torch.Tensor, eligible: torch.Tensor, *, margin: float=MARGIN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample hinge, active mask, and current-minus-next gap."""
    gap = current_page_ce - next_page_ce
    violation = margin - gap
    active = eligible & violation.gt(0)
    hinge = violation.clamp_min(0) * eligible.to(violation.dtype)
    return (hinge, active, gap)


def causal_components(logits: torch.Tensor, labels: torch.Tensor, base_unit_ids: torch.Tensor, focus_unit_ids: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Compute causal token CE once and return (token CE, element loss, page CE)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_base = base_unit_ids[:, 1:].contiguous()
    if bool(shift_base.ge(0).ne(shift_labels.ne(-100)).any()):
        raise NextTsmSftError('supervised labels and base-unit membership differ')
    shift_labels = shift_labels.masked_fill(shift_base.lt(0), -100)
    token_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), reduction='none', ignore_index=-100).view_as(shift_labels)
    page_ce = page_element_ce(token_ce, shift_base)
    loss = None
    if focus_unit_ids is not None:
        loss = element_loss(token_ce, shift_base, focus_unit_ids[:, 1:].contiguous())
    return (token_ce, loss, page_ce)


class NextTsmTrainer(_TransformersTrainer):
    """Three forwards and two backwards per step: current score, next loss, current replay."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._raw_metric_sums: dict[str, float] = {}
        self._ready_metrics: dict[str, float] | None = None

    @staticmethod
    def _model_logits(outputs: Any) -> torch.Tensor:
        logits = outputs.get('logits') if isinstance(outputs, Mapping) else getattr(outputs, 'logits', None)
        if logits is None and isinstance(outputs, (tuple, list)) and outputs:
            logits = outputs[0]
        if not isinstance(logits, torch.Tensor):
            raise NextTsmSftError('model output does not contain tensor logits')
        return logits

    @staticmethod
    def _capture_rng(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())

    @staticmethod
    def _restore_rng(state: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> None:
        (cpu_state, cuda_state) = state
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)

    def _accumulate_raw_metrics(self, metrics: Mapping[str, float]) -> None:
        for (key, value) in metrics.items():
            if key == 'current_replay_max_error':
                self._raw_metric_sums[key] = max(self._raw_metric_sums.get(key, 0.0), float(value))
            else:
                self._raw_metric_sums[key] = self._raw_metric_sums.get(key, 0.0) + float(value)

    def _finalize_metrics(self, *, device: torch.device) -> None:
        """All-reduce one optimizer update's raw numerators over all ranks."""
        sum_keys = ('original_sum', 'hinge_sum', 'next_eligible_sum', 'current_eligible_sum', 'gap_eligible_sum', 'batch_count', 'eligible_count', 'active_count', 'identity_excluded_count', 'beta_batch_sum')
        values = torch.tensor([self._raw_metric_sums.get(key, 0.0) for key in sum_keys], dtype=torch.float64, device=device)
        replay_max = torch.tensor([self._raw_metric_sums.get('current_replay_max_error', 0.0)], dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(replay_max, op=torch.distributed.ReduceOp.MAX)
        totals = {key: float(value) for (key, value) in zip(sum_keys, values.tolist())}
        batch_count = max(totals['batch_count'], 1.0)
        eligible_denominator = max(totals['eligible_count'], 1.0)
        beta = totals['beta_batch_sum'] / batch_count
        ranking_loss = totals['hinge_sum'] / batch_count
        ce_gap = totals['gap_eligible_sum'] / eligible_denominator
        self._ready_metrics = {'element_loss': totals['original_sum'] / batch_count, 'ranking_loss': ranking_loss, 'weighted_ranking_loss': beta * ranking_loss, 'next_ce': totals['next_eligible_sum'] / eligible_denominator, 'current_ce': totals['current_eligible_sum'] / eligible_denominator, 'ce_gap': ce_gap, 'margin_violation': MARGIN - ce_gap, 'eligible_rate': totals['eligible_count'] / batch_count, 'ranking_active_rate': totals['active_count'] / eligible_denominator, 'beta': beta, 'identity_excluded_count': totals['identity_excluded_count'], 'current_replay_max_error': float(replay_max.item())}
        self._raw_metric_sums.clear()

    def training_step(self, model: Any, inputs: dict[str, Any], num_items_in_batch: torch.Tensor | int | None=None) -> torch.Tensor:
        del num_items_in_batch
        (cp_context, inputs) = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
                self.optimizer.train()
            next_inputs = dict(self._prepare_inputs(inputs))
            current_inputs = dict(next_inputs.pop(CURRENT_INPUTS_KEY))
            eligible = next_inputs.pop(RANK_ELIGIBLE_KEY)
            identity_excluded = next_inputs.pop(IDENTITY_EXCLUDED_KEY)
            next_inputs.pop(SAMPLE_IDS_KEY)
            device = next(model.parameters()).device
            rng0 = self._capture_rng(device)
            current_labels = current_inputs.pop('labels')
            current_base_ids = current_inputs.pop(BASE_UNIT_IDS_KEY)
            unwrapped = self.accelerator.unwrap_model(model)
            with self.compute_loss_context_manager(), torch.no_grad():
                current_outputs = unwrapped(**current_inputs)
                (_token_ce, _unused, current_score) = causal_components(self._model_logits(current_outputs), current_labels, current_base_ids, None)
                current_score = current_score.detach()
            del current_outputs, _token_ce
            torch.cuda.empty_cache()
            self._restore_rng(rng0, device)
            next_labels = next_inputs.pop('labels')
            next_base_ids = next_inputs.pop(BASE_UNIT_IDS_KEY)
            next_focus_ids = next_inputs.pop(FOCUS_UNIT_IDS_KEY)
            with self.compute_loss_context_manager():
                next_outputs = model(**next_inputs)
                (_next_token_ce, original, next_page_ce) = causal_components(self._model_logits(next_outputs), next_labels, next_base_ids, next_focus_ids)
                (hinge, active, gap) = ranking_terms(next_page_ce, current_score, eligible)
                beta = margin_beta(int(self.state.global_step))
                positive_loss = original + beta * (active.to(next_page_ce.dtype) * next_page_ce).mean()
            ga = int(self.current_gradient_accumulation_steps)
            self.accelerator.backward(positive_loss / ga)
            original_detached = original.detach()
            next_detached = next_page_ce.detach()
            hinge_detached = hinge.detach()
            active_detached = active.detach()
            gap_detached = gap.detach()
            del next_outputs, _next_token_ce, original, next_page_ce, positive_loss, hinge, active, gap
            torch.cuda.empty_cache()
            with torch.random.fork_rng(devices=[device.index]):
                self._restore_rng(rng0, device)
                with self.compute_loss_context_manager():
                    replay_outputs = model(**current_inputs)
                    (_replay_token_ce, _unused, current_replay) = causal_components(self._model_logits(replay_outputs), current_labels, current_base_ids, None)
                    max_replay_error = float((current_replay.detach() - current_score).abs().max().item())
                    if max_replay_error > REPLAY_ATOL:
                        raise NextTsmSftError(f'current score/replay mismatch: max_abs={max_replay_error:.8g}>{REPLAY_ATOL}')
                    (_replay_hinge, replay_active, _replay_gap) = ranking_terms(next_detached, current_replay.detach(), eligible)
                    if bool(replay_active.ne(active_detached).any()):
                        raise NextTsmSftError('current replay changed ranking gates')
                    negative_loss = -beta * (active_detached.to(current_replay.dtype) * current_replay).mean()
                self.accelerator.backward(negative_loss / ga)
                del replay_outputs, _replay_token_ce, current_replay, negative_loss
            eligible_float = eligible.to(next_detached.dtype)
            local_batch = int(eligible.numel())
            self._accumulate_raw_metrics({'original_sum': float(original_detached.item()) * local_batch, 'hinge_sum': float(hinge_detached.sum().item()), 'next_eligible_sum': float((next_detached * eligible_float).sum().item()), 'current_eligible_sum': float((current_score * eligible_float).sum().item()), 'gap_eligible_sum': float((gap_detached * eligible_float).sum().item()), 'batch_count': float(local_batch), 'eligible_count': float(eligible.sum().item()), 'active_count': float(active_detached.sum().item()), 'identity_excluded_count': float(identity_excluded.sum().item()), 'beta_batch_sum': beta * local_batch, 'current_replay_max_error': max_replay_error})
            if bool(self.accelerator.sync_gradients):
                self._finalize_metrics(device=device)
            return ((original_detached + beta * hinge_detached.mean()) / ga).detach()

    def log(self, logs: dict[str, float], start_time: float | None=None) -> None:
        if self._ready_metrics is not None:
            logs.update(self._ready_metrics)
            self._ready_metrics = None
        super().log(logs, start_time=start_time)


def _normalize_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = sft_module.normalize_row_for_dataset(dict(row))
    normalized['transition_class'] = str(row.get('transition_class') or '')
    normalized['action_bucket'] = str(row.get('action_bucket') or '')
    return normalized


class JsonlDataset:
    """Random access over a JSONL file by byte offset."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offsets: list[int] = []
        with self.path.open('rb') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise NextTsmSftError(f'training dataset is empty: {self.path}')
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__) | {'_handle': None}

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open('r', encoding='utf-8')
        self._handle.seek(self.offsets[index])
        return _normalize_training_row(json.loads(self._handle.readline()))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = sft_module.build_arg_parser()
    parser.description = 'Stage-2 MKD2World training on the next TSM.'
    parser.add_argument('--focus-sidecar', type=sft_module.optional_path, required=True)
    parser.set_defaults(prompt_key='localwm_next', per_device_train_batch_size=1, gradient_accumulation_steps=10, num_train_epochs=2.0, learning_rate=2e-05, lr_scheduler_type='cosine', warmup_ratio=0.03, weight_decay=0.01, seed=0, lora_r=32, lora_alpha=64, lora_dropout=0.05, target_modules='qwen35_expanded', max_length=20000, max_output_tokens=6000, image_min_pixels=200704, image_max_pixels=802816, save_steps=100, logging_steps=1, skip_missing_images=False, device_mode='ddp')
    return parser


def run_training(args: argparse.Namespace) -> Path:
    if args.prompt_key != 'localwm_next':
        raise NextTsmSftError('stage-2 training requires --prompt-key localwm_next')
    if args.init_adapter is None or not Path(args.init_adapter).is_dir():
        raise NextTsmSftError('--init-adapter must be the stage-1 adapter directory')
    (torch_module, _load_dataset, _LoraConfig, PeftModel, _get_peft_model, Image, AutoModelForImageTextToText, AutoProcessor, _Trainer, TrainerCallback, TrainingArguments) = sft_module.import_training_dependencies()
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if not torch_module.cuda.is_available():
        raise NextTsmSftError('stage-2 training requires CUDA')
    torch_module.cuda.set_device(local_rank)
    is_main_process = rank == 0
    run_id = args.run_id or sft_module.timestamp_run_id()
    output_dir = Path(args.output_dir) / run_id
    checkpoint_dir = output_dir / 'checkpoints'
    final_dir = output_dir / 'final_adapter'
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.base_model, min_pixels=args.image_min_pixels, max_pixels=args.image_max_pixels, trust_remote_code=True)
    sidecars = load_focus_sidecar(args.focus_sidecar)
    train_dataset = JsonlDataset(args.train_jsonl)
    dtype = torch_module.bfloat16 if args.bf16 else torch_module.float16 if args.fp16 else 'auto'
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, torch_dtype=dtype, device_map={'': local_rank}, trust_remote_code=True)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    if is_main_process and hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    base_collator = sft_module.LocalWMSftCollator(processor=processor, image_module=Image, max_length=args.max_length, max_output_tokens=args.max_output_tokens, assistant_loss_scope='full', min_supervised_tokens=args.min_supervised_tokens, prompt_key='localwm_next')
    current_collator = CurrentCollator(processor=processor, image_module=Image, max_length=args.max_length, max_output_tokens=args.max_output_tokens, min_supervised_tokens=args.min_supervised_tokens)
    collator = PairedCollator(NextCollator(base_collator, sidecars), current_collator)
    training_args = sft_module.build_training_arguments(TrainingArguments, args, checkpoint_dir, has_eval=False, distributed=world_size > 1)
    callbacks = [sft_module.build_interval_override_callback(TrainerCallback, args, has_eval=False), sft_module.build_training_timing_callback(TrainerCallback, args)]
    if is_main_process:
        prompt_path = sft_module.PromptRegistry.from_project().path_for(args.prompt_key)
        run_config = {'run_id': run_id, 'created_at': datetime.now().isoformat(timespec='seconds'), 'loss': {'focus_lambda_max': MAX_FOCUS_LAMBDA, 'margin': MARGIN, 'beta_max': BETA_MAX, 'beta_warmup_steps': BETA_WARMUP_STEPS}, 'args': {key: str(value) if isinstance(value, Path) else value for (key, value) in vars(args).items()}}
        (output_dir / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
        shutil.copyfile(prompt_path, output_dir / 'prompt_snapshot.md')
    trainer = NextTsmTrainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator, callbacks=callbacks)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(final_dir))
    sft_module.distributed_barrier(torch_module)
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(final_dir))
        summary = {'run_id': run_id, 'finished_at': datetime.now().isoformat(timespec='seconds'), 'final_global_step': int(trainer.state.global_step), 'final_adapter': str(final_dir)}
        (output_dir / 'train_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    sft_module.distributed_barrier(torch_module)
    return final_dir


def main(argv: list[str] | None=None) -> None:
    import sys
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ['build-sidecar']:
        parser = argparse.ArgumentParser(description='Build the focus sidecar for stage-2 training.')
        parser.add_argument('--train-jsonl', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        sidecar_args = parser.parse_args(raw_argv[1:])
        print(json.dumps(build_focus_sidecar(sidecar_args.train_jsonl, sidecar_args.output), ensure_ascii=False))
        return
    args = build_arg_parser().parse_args(raw_argv)
    final_dir = sft_module.run_with_distributed_cleanup(lambda : run_training(args))
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps({'final_adapter': str(final_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
