from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import torch.nn.functional as F
from mkd2world.training import focus_parser as v4_focus_module
from mkd2world.training import element_sft as frozen_v4
from mkd2world.training import focus_contract as focus_contract
from mkd2world.training import sft as sft_module
from mkd2world.training.markdown import build_assistant_output
from mkd2world.training.prompts import apply_chat_template, build_local_world_model_messages, local_world_model_target
try:
    from transformers import Trainer as _TransformersTrainer
except ImportError:

    class _TransformersTrainer:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError('V6Trainer requires transformers')
BASE_UNIT_IDS_KEY = 'base_unit_ids'
FOCUS_UNIT_IDS_KEY = 'focus_unit_ids'
CURRENT_INPUTS_KEY = 'current_inputs'
RANK_ELIGIBLE_KEY = 'rank_eligible'
IDENTITY_EXCLUDED_KEY = 'identity_excluded'
SAMPLE_IDS_KEY = 'sample_ids'
TARGET_TOKEN_IDS_KEY = '_v6_target_token_ids'
IGNORE_UNIT_ID = -1
MAX_FOCUS_LAMBDA = 3.0
V6_MARGIN = 0.1
V6_BETA_MAX = 0.1
V6_BETA_WARMUP_STEPS = 100
V6_REPLAY_ATOL = 1e-05
V6_CERTIFY_VALIDATION_STEPS: tuple[int, ...] = ()
V6_VALIDATION_WORKERS_PER_GPU = 3
V6_LOSS_GRANULARITIES = ('element', 'token')
V6_LOSS_PROFILES: dict[str, tuple[str, str]] = {'ee': ('element', 'element')}
V6_LOSS_PROFILE = 'ee'
V6_BASE_LOSS_GRANULARITY = 'element'
V6_FOCUS_LOSS_GRANULARITY = 'element'
DISPLAY_NAME_PATTERN = 'V6-Cur-FullSFT-{step}'
CANONICAL_TRAIN_JSONL = Path('dataset/LocalWM/webchain_molmo_v4/stage1_sft/train.jsonl')
CANONICAL_VALIDATION_JSONL = Path('dataset/LocalWM/webchain_molmo_v4/validation/test.jsonl')
CANONICAL_EMBEDDING_MODEL = Path('models/embeddings/paraphrase-multilingual-MiniLM-L12-v2')
CANONICAL_TRANSITION_CACHE_ROOT = Path('output/local_wm_v4_transition_metrics_v2/validation500_cache_20260812')
CANONICAL_FOCUS_SIDECAR = Path('output/local_wm_v4_focus_labels/v20-f37247cf-training-v4/focus_sidecar.jsonl')
CANONICAL_FOCUS_MANIFEST_SHA256 = '35c2f38c10f75d8529640d20f3b693907b6d9b23a5d14e597c8e1aad7c5175c2'
CANONICAL_BASE_MODEL = 'models/Qwen3.5-9B'
CANONICAL_INIT_ADAPTER = Path('output/local_wm_cpt_qwen35_9b/p1_description_100k_cur_20260801/checkpoints/checkpoint-1200')
CANONICAL_V5_RUN_CONFIG = Path('output/local_wm_v5_full_sft/v5_full_sft_ee_page_margin_m010_bare_qwen_v20_20260813/run_config.json')
EXPECTED_V5_RUN_CONFIG_SHA256 = '3a05e66c2982fd6a57835a9f4ae98fb6019b96aa53b65a57f043a9cd7530e79a'
CANONICAL_V5_PREFLIGHT = Path('logs/training/v5_full_sft_ee_page_margin_m010_bare_qwen_v20_20260813/preflight.json')
EXPECTED_V5_PREFLIGHT_SHA256 = '7ec4375e07a7ee4751129cca87b801289180a2c18289dee9e8b96435aaee71b4'
CANONICAL_V6_MANIFEST = Path('configs/experiments/v6_cur_full_sft_v3_cur_1200_20260815.json')
EXPECTED_INIT_ADAPTER_CONFIG_SHA256 = '4fcc5813378c127033cfd356be551c99253f32ca714c02f939153700baddaecf'
EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256 = 'fd47b57800bb628b6afe435e9095ea930fba683808a736ca5f3b82a694e762e1'
EXPECTED_INIT_ADAPTER_INTEGRITY_SHA256 = '06de326953671956476b229a0dc7651e7811c35ca4de98f2686264f09023d738'
EXPECTED_INIT_SOURCE_RUN_CONFIG_SHA256 = '306a43a93b78e635ea7a61c19935f9813efe3e43d264e89abbedb60220743206'
EXPECTED_INIT_SELECTION_SHA256 = '55d23e249f095ef9366a67aafdb5d071c755acc410a6c61f38409bb7c772cb36'
EXPECTED_INIT_SELECTION_POINTER_SHA256 = '666bb8c8e3545bc3876a600bcbdde39124b190aeac36f9c18fb62a418359f644'
CANONICAL_OUTPUT_ROOT = Path('output/local_wm_v6_cur_full_sft')
CANONICAL_CERTIFY_OUTPUT_ROOT = Path('output/local_wm_v6_cur_full_sft_certify')
V6_RUN_CONFIG_SCHEMA = 'local-wm-v6-cur-full-sft-run-v1'
V6_CHECKPOINT_BINDING_SCHEMA = 'local-wm-v6-checkpoint-binding-v2'
V6_CERTIFY_PLAN_SCHEMA = 'local-wm-v6-certify-sample-plan-v1'
V6_CERTIFY_RECEIPT_SCHEMA = 'local-wm-v6-six-gpu-certify-receipt-v1'
V6_FORMAL_RECEIPT_SCHEMA = 'local-wm-v6-formal-monitor-receipt-v1'
V6_ADAPTER_INIT_MODE = 'continue_train_selected_v3_cur_lora'
V6_FORMAL_RUN_ID = 'v6_cur_full_sft_v3_cur_1200_20260815'
V6_CERTIFY_RUN_ID = f'{V6_FORMAL_RUN_ID}_certify'
V6_ALLOWED_V5_DIFFERENCES: tuple[str, ...] = ('identity', 'initialization', 'artifacts', 'implementation.trainer_module', 'implementation.trainer_path', 'implementation.runner_path', 'implementation.certification_optimizer_steps', 'implementation.certification_requires_save_finalize', 'implementation.certification_runs_validation500')
CANONICAL_V6_LOG_ROOT = Path('logs/training') / V6_FORMAL_RUN_ID
V6_CUDA_VISIBLE_DEVICES = '0,1,2,3,4,5'
V6_WORLD_SIZE = 6
V6_VALIDATION_WORKERS = V6_WORLD_SIZE * V6_VALIDATION_WORKERS_PER_GPU
V6_PER_DEVICE_TRAIN_BATCH_SIZE = 1
V6_GRADIENT_ACCUMULATION_STEPS = 10
V6_GLOBAL_BATCH_SIZE = 60
V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB = 1024
V6_SAVE_STEPS = 100
V6_SAVE_INTERVAL_EXAMPLES = 6000
V6_EXTRA_CHECKPOINT_STEPS: tuple[int, ...] = (1042,)
V6_VALIDATION_STEPS: tuple[int, ...] = (*range(100, 1001, 100), 1042)
V6_CERTIFY_ROWS = 120
V6_CERTIFY_EXPECTED_STEPS = 2
V6_ACTION_BUCKETS = ('click', 'text_key', 'scroll', 'navigation', 'other')
V6_CERTIFY_CONTRACT_CASES = ('transition:changed', 'transition:near_noop', 'transition:exact_noop', 'kind:ADD', 'kind:UPDATE', 'changed:base_only', 'update:multi_field')
EXPECTED_TRAINABLE_PARAMS = 102530048
EXPECTED_TRAINING_ROWS = 31245
EXPECTED_FINAL_STEP = 1042
EXPECTED_BASE_CONFIG_SHA256 = 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05'
EXPECTED_BASE_INDEX_SHA256 = '26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5'
EXPECTED_BASE_RESOLVED_PATH = Path('/testessfs10/models/huggingface/Qwen3.5-9B')
EXPECTED_BASE_SHARDS: dict[str, tuple[int, str]] = {'model.safetensors-00001-of-00004.safetensors': (5276436216, 'db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e'), 'model.safetensors-00002-of-00004.safetensors': (5335161512, '31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330'), 'model.safetensors-00003-of-00004.safetensors': (5368717440, '7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec'), 'model.safetensors-00004-of-00004.safetensors': (3325995712, 'b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6')}
EXPECTED_EXPANDED_MODULES = 358
EXPECTED_ADAPTER_TENSORS = 716
EXPECTED_TARGET_MODULES_SHA256 = '187f94f7be79b17409b9b4aa521d61537589833fd9f5380500a5f09caa2c0be1'
EXPECTED_TOTAL_TARGET_TOKENS = 32339723
EXPECTED_MAX_TARGET_TOKENS = 3536
EXPECTED_MAX_TARGET_SAMPLE_ID = 'b03a815b19a56b488031f7eb8e70291158019eeb9b59975cb2ceb611a76a6e98'
EXPECTED_TOKEN_CONTRACT_SHA256 = '4d049c638adbaaa50482d7d8902c61a95cd0ad1d13c6fccda5539042f51cfae8'
EXPECTED_TOTAL_CURRENT_TARGET_TOKENS = 31025778
EXPECTED_MAX_CURRENT_TARGET_TOKENS = 3551
EXPECTED_MAX_CURRENT_TARGET_SAMPLE_ID = '17d03a9af35b198d11efb49b04f1984d8344012e1a47d39d1305d4cfe6259e2d'
EXPECTED_NONCHANGED_TOKEN_EQUAL_ROWS = 1966
EXPECTED_MAX_CURRENT_BASE_UNIT_SAMPLE_ID = '6b2c7c91439820f7834ed19f8971cbd7a83e81e9096f98e353595db4f05178bb'

def v6_runner_relative_path() -> Path:
    return Path('scripts/training/run_v6_cur_full_sft.sh')

def configure_v6_loss_profile(profile: str) -> dict[str, Any]:
    normalized = str(profile).strip().lower()
    if getattr(frozen_v4, 'V4_LOSS_PROFILE', None) != 'ee':
        raise V6SftError('V6 requires the imported frozen V4 reducer profile to be ee; export V4_LOSS_PROFILE=ee before Python starts')
    if normalized not in V6_LOSS_PROFILES:
        raise V6SftError(f'unsupported V6 loss profile {profile!r}; expected one of {sorted(V6_LOSS_PROFILES)}')
    (base_granularity, focus_granularity) = V6_LOSS_PROFILES[normalized]
    global V6_LOSS_PROFILE
    global V6_BASE_LOSS_GRANULARITY
    global V6_FOCUS_LOSS_GRANULARITY
    global DISPLAY_NAME_PATTERN
    global CANONICAL_OUTPUT_ROOT
    global CANONICAL_CERTIFY_OUTPUT_ROOT
    global V6_FORMAL_RUN_ID
    global V6_CERTIFY_RUN_ID
    global CANONICAL_V6_LOG_ROOT
    global V6_SAVE_STEPS
    global V6_SAVE_INTERVAL_EXAMPLES
    global V6_EXTRA_CHECKPOINT_STEPS
    global V6_VALIDATION_STEPS
    V6_LOSS_PROFILE = normalized
    V6_BASE_LOSS_GRANULARITY = base_granularity
    V6_FOCUS_LOSS_GRANULARITY = focus_granularity
    DISPLAY_NAME_PATTERN = 'V6-Cur-FullSFT-{step}'
    CANONICAL_OUTPUT_ROOT = Path('output/local_wm_v6_cur_full_sft')
    CANONICAL_CERTIFY_OUTPUT_ROOT = Path('output/local_wm_v6_cur_full_sft_certify')
    V6_SAVE_STEPS = 100
    V6_SAVE_INTERVAL_EXAMPLES = 6000
    V6_EXTRA_CHECKPOINT_STEPS = (1042,)
    V6_VALIDATION_STEPS = (*range(100, 1001, 100), 1042)
    V6_FORMAL_RUN_ID = 'v6_cur_full_sft_v3_cur_1200_20260815'
    V6_CERTIFY_RUN_ID = f'{V6_FORMAL_RUN_ID}_certify'
    CANONICAL_V6_LOG_ROOT = Path('logs/training') / V6_FORMAL_RUN_ID
    return {'profile': normalized, 'base_loss_granularity': base_granularity, 'focus_loss_granularity': focus_granularity, 'display_name_pattern': DISPLAY_NAME_PATTERN, 'ranking_margin': V6_MARGIN, 'ranking_beta_max': V6_BETA_MAX, 'ranking_beta_warmup_steps': V6_BETA_WARMUP_STEPS, 'formal_run_id': V6_FORMAL_RUN_ID, 'output_root': str(CANONICAL_OUTPUT_ROOT), 'save_steps': V6_SAVE_STEPS, 'save_total_limit': None, 'extra_checkpoint_steps': list(V6_EXTRA_CHECKPOINT_STEPS), 'validation_steps': list(V6_VALIDATION_STEPS)}

class V6SftError(ValueError):
    """Raised when V6 labels, token alignment, or loss inputs are invalid."""
configure_v6_loss_profile(os.environ.get('V6_LOSS_PROFILE', 'ee'))

@dataclass(frozen=True)
class CharacterSpan:
    start: int
    end: int

@dataclass(frozen=True)
class ValidatedSidecar:
    """Validated character-level supervision for one canonical target."""
    sample_id: str
    target_text: str
    base_spans: tuple[CharacterSpan, ...]
    focus_spans: tuple[tuple[CharacterSpan, ...], ...]

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def expected_qwen35_expanded_target_modules() -> tuple[str, ...]:
    modules: list[str] = []
    for layer in range(32):
        prefix = f'model.language_model.layers.{layer}'
        if layer % 4 == 3:
            modules.extend((f'{prefix}.self_attn.{name}' for name in ('k_proj', 'o_proj', 'q_proj', 'v_proj')))
        else:
            modules.extend((f'{prefix}.linear_attn.{name}' for name in ('in_proj_a', 'in_proj_b', 'in_proj_qkv', 'in_proj_z', 'out_proj')))
        modules.extend((f'{prefix}.mlp.{name}' for name in ('down_proj', 'gate_proj', 'up_proj')))
    for block in range(27):
        prefix = f'model.visual.blocks.{block}'
        modules.extend((f'{prefix}.attn.proj', f'{prefix}.attn.qkv', f'{prefix}.mlp.linear_fc1', f'{prefix}.mlp.linear_fc2'))
    modules.extend(('model.visual.merger.linear_fc1', 'model.visual.merger.linear_fc2'))
    canonical = tuple(sorted(modules))
    if len(canonical) != EXPECTED_EXPANDED_MODULES or len(set(canonical)) != EXPECTED_EXPANDED_MODULES:
        raise V6SftError('internal qwen35_expanded target-module contract drifted')
    return canonical

def _validate_target_modules(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_EXPANDED_MODULES or any((not isinstance(name, str) or not name for name in value)) or (len(set(value)) != EXPECTED_EXPANDED_MODULES):
        raise V6SftError(f'{context} must contain 358 unique module names')
    canonical = tuple(sorted(value))
    if canonical != expected_qwen35_expanded_target_modules():
        raise V6SftError(f'{context} differs from frozen qwen35_expanded modules')
    return canonical

def _require_lexical_path(value: str | Path, expected: str | Path, *, context: str) -> Path:
    """Reject aliases even when they resolve to the same canonical artifact."""
    raw = os.fspath(value)
    expected_raw = os.fspath(expected)
    candidate = Path(raw)
    if raw != expected_raw or '..' in candidate.parts or candidate.is_absolute():
        raise V6SftError(f'{context} must be lexical path {expected_raw!r}')
    return candidate

def canonical_v6_token_contract() -> dict[str, Any]:
    contract = {'rows': EXPECTED_TRAINING_ROWS, 'total_target_tokens': EXPECTED_TOTAL_TARGET_TOKENS, 'max_target_tokens': EXPECTED_MAX_TARGET_TOKENS, 'max_target_sample_id': EXPECTED_MAX_TARGET_SAMPLE_ID, 'max_output_tokens': 6000, 'tokenizer_base_config_sha256': EXPECTED_BASE_CONFIG_SHA256}
    actual_sha = _sha256_text(json.dumps(contract, sort_keys=True, separators=(',', ':')))
    return contract

def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V6SftError(f'{context} must be an integer')
    return value

def _read_span(raw_span: object, *, target_text: str, context: str, require_text_hash: bool) -> CharacterSpan:
    if not isinstance(raw_span, Mapping):
        raise V6SftError(f'{context} must be an object')
    start = _require_int(raw_span.get('start'), context=f'{context}.start')
    end = _require_int(raw_span.get('end'), context=f'{context}.end')
    if start < 0 or end <= start or end > len(target_text):
        raise V6SftError(f'{context} is outside target: {start}:{end}/{len(target_text)}')
    expected_text = target_text[start:end]
    has_text = 'text' in raw_span
    has_hash = 'text_sha256' in raw_span
    if require_text_hash and (not has_text or not has_hash):
        raise V6SftError(f'{context} must include text and text_sha256')
    if has_text != has_hash:
        raise V6SftError(f'{context} must provide text and text_sha256 together')
    if has_text:
        if raw_span['text'] != expected_text:
            raise V6SftError(f'{context}.text does not match target span')
        if raw_span['text_sha256'] != _sha256_text(expected_text):
            raise V6SftError(f'{context}.text_sha256 does not match target span')
    return CharacterSpan(start=start, end=end)

def canonical_v6_target(sample: Mapping[str, Any]) -> str:
    """Build the Next target directly from raw ``target_next_md``."""
    target_markdown = sample.get('target_next_md')
    if not isinstance(target_markdown, str) or not target_markdown.strip():
        sample_id = str(sample.get('id') or '<unknown>')
        raise V6SftError(f'sample {sample_id} has an empty target_next_md')
    return build_assistant_output('', target_markdown)

def canonical_v6_current_target(sample: Mapping[str, Any]) -> str:
    current_markdown = sample.get('current_md')
    if not isinstance(current_markdown, str) or not current_markdown.strip():
        sample_id = str(sample.get('id') or '<unknown>')
        raise V6SftError(f'sample {sample_id} has an empty current_md')
    return build_assistant_output('', current_markdown)

def build_v6_current_sidecar(sample: Mapping[str, Any]) -> ValidatedSidecar:
    sample_id = str(sample.get('id') or '').strip()
    if not sample_id:
        raise V6SftError('training sample is missing id')
    target_text = canonical_v6_current_target(sample)
    try:
        parsed = focus_contract.focus_labeler.parse_markdown_spans(target_text)
    except focus_contract.focus_labeler.V4FocusError as exc:
        raise V6SftError(f'sample {sample_id} current_md violates the V4 Markdown parser: {exc}') from exc
    if parsed.text != target_text or not parsed.base_units:
        raise V6SftError(f'sample {sample_id} current target parsing drifted')
    return ValidatedSidecar(sample_id=sample_id, target_text=target_text, base_spans=tuple((CharacterSpan(span.start, span.end) for span in parsed.base_units)), focus_spans=())

def _validate_legacy_v7_sidecar(sample: Mapping[str, Any], raw_sidecar: Mapping[str, Any]) -> ValidatedSidecar:
    sample_id = str(sample.get('id') or '').strip()
    if not sample_id:
        raise V6SftError('training sample is missing id')
    sidecar_id = str(raw_sidecar.get('sample_id') or '').strip()
    if sidecar_id != sample_id:
        raise V6SftError(f'sidecar sample_id mismatch: expected {sample_id!r}, got {sidecar_id!r}')
    target_text = raw_sidecar.get('target_next_md')
    target_alias = raw_sidecar.get('target_text')
    if target_text is None:
        target_text = target_alias
    elif target_alias is not None and target_alias != target_text:
        raise V6SftError(f'sample {sample_id} has conflicting target aliases')
    if not isinstance(target_text, str):
        raise V6SftError(f'sample {sample_id} sidecar target_next_md must be text')
    expected_target = canonical_v6_target(sample)
    if target_text != expected_target:
        raise V6SftError(f'sample {sample_id} sidecar target is not the canonical localwm_next target')
    if raw_sidecar.get('target_sha256') != _sha256_text(expected_target):
        raise V6SftError(f'sample {sample_id} target_sha256 mismatch')
    if raw_sidecar.get('schema_version') != v4_focus_module.FOCUS_LABEL_VERSION:
        raise V6SftError(f'sample {sample_id} requires schema_version {v4_focus_module.FOCUS_LABEL_VERSION!r}')
    for field_name in ('base_units', 'focus_units', 'review_candidates', 'next_elements'):
        if field_name not in raw_sidecar:
            raise V6SftError(f'sample {sample_id} sidecar is missing {field_name}')
        value = raw_sidecar[field_name]
        if not isinstance(value, list):
            raise V6SftError(f'sample {sample_id} {field_name} must be a list')
    try:
        v4_focus_module.validate_focus_annotation(dict(raw_sidecar))
        parsed_target = v4_focus_module.parse_markdown_spans(expected_target)
    except v4_focus_module.V4FocusError as exc:
        raise V6SftError(f'sample {sample_id} violates the frozen V4 focus schema: {exc}') from exc
    if parsed_target.text != expected_target:
        raise V6SftError(f'sample {sample_id} target is not canonical V6 Markdown')
    raw_next_elements = raw_sidecar['next_elements']
    if len(raw_next_elements) != len(parsed_target.elements):
        raise V6SftError(f'sample {sample_id} next_elements length does not match target elements')
    for (element_index, (raw_element, parsed_element)) in enumerate(zip(raw_next_elements, parsed_target.elements)):
        if not isinstance(raw_element, Mapping):
            raise V6SftError(f'sample {sample_id} next_elements[{element_index}] must be an object')
        raw_line_span = raw_element.get('line_span')
        if not isinstance(raw_line_span, Mapping):
            raise V6SftError(f'sample {sample_id} next_elements[{element_index}].line_span is required')
        expected_line_span = (parsed_element.line_span.start, parsed_element.line_span.end)
        actual_line_span = (_require_int(raw_line_span.get('start'), context=f'sample {sample_id} next_elements[{element_index}].line_span.start'), _require_int(raw_line_span.get('end'), context=f'sample {sample_id} next_elements[{element_index}].line_span.end'))
        if actual_line_span != expected_line_span:
            raise V6SftError(f'sample {sample_id} next_elements[{element_index}] line span mismatch')
        expected_element = asdict(parsed_element)
        expected_element['bbox'] = list(parsed_element.bbox)
        expected_element['normalized_content'] = parsed_element.normalized_content
        if dict(raw_element) != expected_element:
            raise V6SftError(f'sample {sample_id} next_elements[{element_index}] metadata mismatch')
    raw_base_units = raw_sidecar.get('base_units')
    if not isinstance(raw_base_units, Sequence) or isinstance(raw_base_units, (str, bytes)):
        raise V6SftError(f'sample {sample_id} base_units must be a list')
    if not raw_base_units:
        raise V6SftError(f'sample {sample_id} has no base units')
    expected_base_spans = tuple((CharacterSpan(span.start, span.end) for span in parsed_target.base_units))
    if len(raw_base_units) != len(expected_base_spans):
        raise V6SftError(f'sample {sample_id} has {len(raw_base_units)} base units; canonical target requires {len(expected_base_spans)}')
    base_spans: list[CharacterSpan] = []
    cursor = 0
    for (unit_index, raw_span) in enumerate(raw_base_units):
        span = _read_span(raw_span, target_text=target_text, context=f'sample {sample_id} base_units[{unit_index}]', require_text_hash=False)
        if span.start != cursor:
            raise V6SftError(f'sample {sample_id} base units do not partition target at {cursor}:{span.start}')
        base_spans.append(span)
        cursor = span.end
    if cursor != len(target_text):
        raise V6SftError(f'sample {sample_id} base units end at {cursor}, expected {len(target_text)}')
    if tuple(base_spans) != expected_base_spans:
        raise V6SftError(f'sample {sample_id} base units must be exactly Environment/EOS neutral group followed by one complete group per target element line')
    raw_focus_units = raw_sidecar['focus_units']
    if not isinstance(raw_focus_units, Sequence) or isinstance(raw_focus_units, (str, bytes)):
        raise V6SftError(f'sample {sample_id} focus_units must be a list')
    focus_spans: list[tuple[CharacterSpan, ...]] = []
    occupied_focus_spans: list[tuple[CharacterSpan, int]] = []
    seen_focus_elements: set[int] = set()
    for (focus_index, raw_unit) in enumerate(raw_focus_units):
        context = f'sample {sample_id} focus_units[{focus_index}]'
        if not isinstance(raw_unit, Mapping):
            raise V6SftError(f'{context} must be an object')
        if raw_unit.get('kind') not in {'ADD', 'UPDATE'}:
            raise V6SftError(f"{context} has unsupported kind {raw_unit.get('kind')!r}")
        if 'focus_id' not in raw_unit or _require_int(raw_unit.get('focus_id'), context=f'{context}.focus_id') != focus_index:
            raise V6SftError(f'{context}.focus_id must equal {focus_index}')
        next_element_index = _require_int(raw_unit.get('next_element_index'), context=f'{context}.next_element_index')
        if next_element_index < 0 or next_element_index >= len(parsed_target.elements):
            raise V6SftError(f'{context} references an invalid next element')
        if next_element_index in seen_focus_elements:
            raise V6SftError(f'{context} duplicates next element {next_element_index}')
        seen_focus_elements.add(next_element_index)
        raw_fields = raw_unit.get('fields')
        if not isinstance(raw_fields, list) or not raw_fields or (not all((isinstance(field, str) and field for field in raw_fields))) or (len(set(raw_fields)) != len(raw_fields)) or (raw_fields != sorted(raw_fields)):
            raise V6SftError(f'{context}.fields must be a non-empty sorted unique string list')
        raw_spans = raw_unit.get('spans')
        if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
            raise V6SftError(f'{context}.spans must be a list')
        if not raw_spans:
            raise V6SftError(f'{context} has no spans')
        unit_spans: list[CharacterSpan] = []
        owner_indexes: set[int] = set()
        for (span_index, raw_span) in enumerate(raw_spans):
            span = _read_span(raw_span, target_text=target_text, context=f'{context}.spans[{span_index}]', require_text_hash=True)
            owners = [index for (index, base_span) in enumerate(base_spans) if span.start >= base_span.start and span.end <= base_span.end]
            if len(owners) != 1:
                raise V6SftError(f'{context}.spans[{span_index}] crosses a base unit')
            owner_indexes.add(owners[0])
            for (occupied, occupied_group) in occupied_focus_spans:
                if span.start < occupied.end and span.end > occupied.start:
                    raise V6SftError(f'{context}.spans[{span_index}] overlaps focus group {occupied_group}')
            occupied_focus_spans.append((span, focus_index))
            unit_spans.append(span)
        if len(owner_indexes) != 1:
            raise V6SftError(f'{context} spans more than one base unit')
        element = parsed_target.elements[next_element_index]
        if raw_unit.get('kind') == 'ADD':
            expected_fields = ['line']
            expected_spans = {(element.line_span.start, element.line_span.end)}
        else:
            field_spans = {'content': element.content_span, 'value': element.content_span, 'type': element.type_span, 'state': element.state_span, 'bbox': element.bbox_span}
            if any((field not in field_spans for field in raw_fields)):
                raise V6SftError(f'{context} has unsupported UPDATE fields')
            if any((field_spans[field] is None for field in raw_fields)):
                raise V6SftError(f'{context} references a field absent from the target element')
            expected_fields = raw_fields
            expected_spans = {(field_spans[field].start, field_spans[field].end) for field in raw_fields}
        actual_spans = {(span.start, span.end) for span in unit_spans}
        if raw_fields != expected_fields or actual_spans != expected_spans:
            raise V6SftError(f'{context} spans must exactly match its declared target element fields')
        focus_spans.append(tuple(unit_spans))
    return ValidatedSidecar(sample_id=sample_id, target_text=target_text, base_spans=tuple(base_spans), focus_spans=tuple(focus_spans))

def validate_sidecar(sample: Mapping[str, Any], raw_sidecar: Mapping[str, Any]) -> ValidatedSidecar:
    """Validate one final matcher-bound compact sidecar without replacing the full target."""
    try:
        validated = focus_contract.validate_training_sidecar(sample, raw_sidecar)
    except focus_contract.V4TrainingSidecarError as exc:
        sample_id = str(sample.get('id') or '<unknown>')
        raise V6SftError(f'sample {sample_id} violates the final training sidecar contract: {exc}') from exc
    raw_next_target = canonical_v6_target(sample)
    if validated.target_text != raw_next_target:
        raise V6SftError(f'sample {validated.sample_id} sidecar target is not raw target_next_md')
    return ValidatedSidecar(sample_id=validated.sample_id, target_text=validated.target_text, base_spans=tuple((CharacterSpan(span.start, span.end) for span in validated.base_spans)), focus_spans=tuple((tuple((CharacterSpan(span.start, span.end) for span in group)) for group in validated.focus_spans)))

def load_focus_sidecar(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load sidecar JSONL and reject malformed or duplicate sample ids."""
    sidecar_path = Path(path)
    rows: dict[str, dict[str, Any]] = {}
    with sidecar_path.open(encoding='utf-8') as handle:
        for (line_number, line) in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V6SftError(f'invalid sidecar JSON at {sidecar_path}:{line_number}') from exc
            if not isinstance(row, dict):
                raise V6SftError(f'non-object sidecar row at {sidecar_path}:{line_number}')
            sample_id = str(row.get('sample_id') or '').strip()
            if not sample_id:
                raise V6SftError(f'missing sidecar sample_id at {sidecar_path}:{line_number}')
            if sample_id in rows:
                raise V6SftError(f'duplicate sidecar sample_id {sample_id!r}')
            rows[sample_id] = row
    if not rows:
        raise V6SftError(f'sidecar has no rows: {sidecar_path}')
    return rows

def _single_tokenizer_output(value: object, *, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, list):
        raise V6SftError(f'tokenizer {name} must be a list')
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise V6SftError(f'tokenizer {name} unexpectedly contains a batch')
        value = value[0]
    return value

def map_target_tokens_to_units(tokenizer: Any, sidecar: ValidatedSidecar) -> tuple[list[int], list[int], list[int]]:
    """Tokenize one target and map fast-tokenizer offsets to V6 unit ids."""
    if not getattr(tokenizer, 'is_fast', False):
        raise V6SftError('V6Collator requires a fast tokenizer with offset_mapping')
    try:
        encoded = tokenizer(sidecar.target_text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception as exc:
        raise V6SftError('fast tokenizer failed to produce target offsets') from exc
    if not isinstance(encoded, Mapping) or 'input_ids' not in encoded:
        raise V6SftError('tokenizer output is missing input_ids')
    if 'offset_mapping' not in encoded:
        raise V6SftError('fast tokenizer output is missing offset_mapping')
    token_ids = _single_tokenizer_output(encoded['input_ids'], name='input_ids')
    raw_offsets = _single_tokenizer_output(encoded['offset_mapping'], name='offset_mapping')
    if not token_ids or len(token_ids) != len(raw_offsets):
        raise V6SftError('target token ids and offsets have inconsistent lengths')
    base_unit_ids: list[int] = []
    focus_unit_ids: list[int] = []
    seen_base: set[int] = set()
    seen_focus: set[int] = set()
    previous_start = 0
    covered_end = 0
    for (token_index, raw_offset) in enumerate(raw_offsets):
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise V6SftError(f'invalid token offset at index {token_index}')
        start = _require_int(raw_offset[0], context=f'token offset {token_index}.start')
        end = _require_int(raw_offset[1], context=f'token offset {token_index}.end')
        if start < previous_start or start > covered_end or start < 0 or (end <= start) or (end > len(sidecar.target_text)):
            raise V6SftError(f'invalid or uncovered token offset at index {token_index}: {start}:{end}/{len(sidecar.target_text)}')
        previous_start = start
        covered_end = max(covered_end, end)
        base_overlaps = [(min(end, span.end) - max(start, span.start), unit_index) for (unit_index, span) in enumerate(sidecar.base_spans) if start < span.end and end > span.start]
        if not base_overlaps:
            raise V6SftError(f'target token {token_index} at {start}:{end} has no base unit')
        (_, base_id) = max(base_overlaps)
        base_unit_ids.append(base_id)
        seen_base.add(base_id)
        focus_owners = {focus_index for (focus_index, spans) in enumerate(sidecar.focus_spans) for span in spans if start < span.end and end > span.start}
        if len(focus_owners) > 1:
            raise V6SftError(f'target token {token_index} overlaps multiple focus groups')
        focus_id = next(iter(focus_owners), IGNORE_UNIT_ID)
        focus_unit_ids.append(focus_id)
        if focus_id >= 0:
            seen_focus.add(focus_id)
    if covered_end != len(sidecar.target_text):
        raise V6SftError(f'token offsets cover through {covered_end}, expected {len(sidecar.target_text)}')
    expected_base = set(range(len(sidecar.base_spans)))
    expected_focus = set(range(len(sidecar.focus_spans)))
    if seen_base != expected_base:
        raise V6SftError(f'base units without tokenizer coverage: {sorted(expected_base - seen_base)}')
    if seen_focus != expected_focus:
        raise V6SftError(f'focus units without tokenizer coverage: {sorted(expected_focus - seen_focus)}')
    return ([int(token_id) for token_id in token_ids], base_unit_ids, focus_unit_ids)

def _index_sidecar(sidecar: str | Path | Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if isinstance(sidecar, (str, Path)):
        return load_focus_sidecar(sidecar)
    indexed: dict[str, dict[str, Any]] = {}
    for (key, value) in sidecar.items():
        if not isinstance(value, Mapping):
            raise V6SftError(f'sidecar entry {key!r} must be an object')
        sample_id = str(value.get('sample_id') or key).strip()
        if sample_id != str(key):
            raise V6SftError(f'sidecar mapping key {key!r} does not match sample_id {sample_id!r}')
        indexed[sample_id] = dict(value)
    if not indexed:
        raise V6SftError('sidecar mapping is empty')
    return indexed

class V6Collator:
    """Wrap ``LocalWMSftCollator`` with strict V6 token-unit alignment."""

    def __init__(self, base_collator: Any, sidecar: str | Path | Mapping[str, Mapping[str, Any]], *, tokenizer: Any | None=None) -> None:
        self.base_collator = base_collator
        self.sidecar_by_id = _index_sidecar(sidecar)
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            processor = getattr(base_collator, 'processor', None)
            self.tokenizer = getattr(processor, 'tokenizer', None)
        if self.tokenizer is None:
            raise V6SftError('V6Collator requires a tokenizer')
        base_prompt_key = getattr(base_collator, 'prompt_key', 'localwm_next')
        if base_prompt_key != 'localwm_next':
            raise V6SftError("V6Collator requires LocalWMSftCollator(prompt_key='localwm_next')")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(features)
        if 'labels' not in batch or 'input_ids' not in batch:
            raise V6SftError('base collator must return input_ids and labels')
        if BASE_UNIT_IDS_KEY in batch or FOCUS_UNIT_IDS_KEY in batch:
            raise V6SftError('base collator unexpectedly returned V6 unit ids')
        labels = batch['labels']
        input_ids = batch['input_ids']
        if not isinstance(labels, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            raise V6SftError('base collator input_ids and labels must be tensors')
        if labels.ndim != 2 or labels.shape != input_ids.shape:
            raise V6SftError('base collator input_ids and labels must have equal 2-D shapes')
        if labels.shape[0] != len(features):
            raise V6SftError('base collator batch size does not match features')
        base_ids = torch.full_like(labels, IGNORE_UNIT_ID)
        focus_ids = torch.full_like(labels, IGNORE_UNIT_ID)
        aligned_target_token_ids: list[list[int]] = []
        for (row_index, sample) in enumerate(features):
            sample_id = str(sample.get('id') or '').strip()
            if not sample_id:
                raise V6SftError(f'training feature {row_index} is missing id')
            raw_sidecar = self.sidecar_by_id.get(sample_id)
            if raw_sidecar is None:
                raise V6SftError(f'missing sidecar row for sample {sample_id}')
            validated = validate_sidecar(sample, raw_sidecar)
            (target_token_ids, target_base_ids, target_focus_ids) = map_target_tokens_to_units(self.tokenizer, validated)
            supervised_positions = torch.nonzero(labels[row_index].ne(-100), as_tuple=False).flatten()
            if supervised_positions.numel() == 0:
                raise V6SftError(f'sample {sample_id} has no supervised labels')
            target_start = int(supervised_positions[0].item())

            def target_ids_match(candidate_ids: list[int]) -> bool:
                candidate_end = target_start + len(candidate_ids)
                if candidate_end > labels.shape[1]:
                    return False
                expected = torch.as_tensor(candidate_ids, dtype=input_ids.dtype, device=input_ids.device)
                return torch.equal(input_ids[row_index, target_start:candidate_end], expected) and torch.equal(labels[row_index, target_start:candidate_end].to(input_ids.dtype), expected)
            if not target_ids_match(target_token_ids):
                trimmed = ValidatedSidecar(sample_id=validated.sample_id, target_text=validated.target_text[:-1], base_spans=validated.base_spans, focus_spans=validated.focus_spans)
                if not validated.target_text.endswith('\n'):
                    raise V6SftError(f'sample {sample_id} target token IDs do not match processor')
                (trimmed_ids, trimmed_base_ids, trimmed_focus_ids) = map_target_tokens_to_units(self.tokenizer, trimmed)
                if not target_ids_match(trimmed_ids):
                    raise V6SftError(f'sample {sample_id} target token IDs do not match processor')
                target_token_ids = trimmed_ids
                target_base_ids = trimmed_base_ids
                target_focus_ids = trimmed_focus_ids
            aligned_target_token_ids.append(target_token_ids)
            target_end = target_start + len(target_token_ids)
            base_ids[row_index, target_start:target_end] = torch.as_tensor(target_base_ids, dtype=base_ids.dtype, device=base_ids.device)
            focus_ids[row_index, target_start:target_end] = torch.as_tensor(target_focus_ids, dtype=focus_ids.dtype, device=focus_ids.device)
            suffix_positions = torch.nonzero(labels[row_index, target_end:].ne(-100), as_tuple=False).flatten()
            if suffix_positions.numel() > 0:
                suffix_positions = suffix_positions + target_end
                expected_suffix = torch.arange(target_end, int(suffix_positions[-1].item()) + 1, device=suffix_positions.device, dtype=suffix_positions.dtype)
                if not torch.equal(suffix_positions, expected_suffix):
                    raise V6SftError(f'sample {sample_id} has a supervision gap in chat suffix')
                if not torch.equal(labels[row_index, suffix_positions].to(input_ids.dtype), input_ids[row_index, suffix_positions]):
                    raise V6SftError(f'sample {sample_id} chat suffix labels do not match input_ids')
                base_ids[row_index, suffix_positions] = 0
        batch[BASE_UNIT_IDS_KEY] = base_ids
        batch[FOCUS_UNIT_IDS_KEY] = focus_ids
        batch[TARGET_TOKEN_IDS_KEY] = aligned_target_token_ids
        return batch
V6SftCollator = V6Collator

def _align_explicit_target_units(batch: dict[str, Any], features: Sequence[Mapping[str, Any]], sidecars_by_id: Mapping[str, ValidatedSidecar], tokenizer: Any) -> list[list[int]]:
    """Attach base-unit ids to an explicitly constructed assistant target."""
    labels = batch.get('labels')
    input_ids = batch.get('input_ids')
    if not isinstance(labels, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
        raise V6SftError('explicit target collator requires tensor input_ids/labels')
    if labels.ndim != 2 or labels.shape != input_ids.shape:
        raise V6SftError('explicit target input_ids/labels must have equal 2-D shapes')
    if labels.shape[0] != len(features):
        raise V6SftError('explicit target batch size does not match features')
    base_ids = torch.full_like(labels, IGNORE_UNIT_ID)
    aligned_token_ids: list[list[int]] = []
    for (row_index, sample) in enumerate(features):
        sample_id = str(sample.get('id') or '').strip()
        sidecar = sidecars_by_id.get(sample_id)
        if sidecar is None:
            sidecar = build_v6_current_sidecar(sample)
        (token_ids, token_base_ids, token_focus_ids) = map_target_tokens_to_units(tokenizer, sidecar)
        if any((unit_id != IGNORE_UNIT_ID for unit_id in token_focus_ids)):
            raise V6SftError(f'Current target {sample_id} unexpectedly has focus ids')
        supervised = torch.nonzero(labels[row_index].ne(-100), as_tuple=False).flatten()
        if supervised.numel() == 0:
            raise V6SftError(f'Current target {sample_id} has no supervised labels')
        target_start = int(supervised[0].item())

        def target_matches(candidate_ids: Sequence[int]) -> bool:
            target_end = target_start + len(candidate_ids)
            if target_end > labels.shape[1]:
                return False
            expected = torch.as_tensor(candidate_ids, dtype=input_ids.dtype, device=input_ids.device)
            return torch.equal(input_ids[row_index, target_start:target_end], expected) and torch.equal(labels[row_index, target_start:target_end].to(input_ids.dtype), expected)
        if not target_matches(token_ids):
            if not sidecar.target_text.endswith('\n'):
                raise V6SftError(f'Current target {sample_id} token IDs do not match processor')
            trimmed = ValidatedSidecar(sample_id=sidecar.sample_id, target_text=sidecar.target_text[:-1], base_spans=sidecar.base_spans, focus_spans=())
            (token_ids, token_base_ids, token_focus_ids) = map_target_tokens_to_units(tokenizer, trimmed)
            if not target_matches(token_ids):
                raise V6SftError(f'Current target {sample_id} token IDs do not match processor')
        aligned_token_ids.append(list(token_ids))
        target_end = target_start + len(token_ids)
        base_ids[row_index, target_start:target_end] = torch.as_tensor(token_base_ids, dtype=base_ids.dtype, device=base_ids.device)
        suffix = torch.nonzero(labels[row_index, target_end:].ne(-100), as_tuple=False).flatten()
        if suffix.numel() > 0:
            suffix = suffix + target_end
            expected_suffix = torch.arange(target_end, int(suffix[-1].item()) + 1, device=suffix.device, dtype=suffix.dtype)
            if not torch.equal(suffix, expected_suffix):
                raise V6SftError(f'Current target {sample_id} has a supervision gap')
            if not torch.equal(labels[row_index, suffix].to(input_ids.dtype), input_ids[row_index, suffix]):
                raise V6SftError(f'Current target {sample_id} suffix labels differ from input ids')
            base_ids[row_index, suffix] = 0
    batch[BASE_UNIT_IDS_KEY] = base_ids
    batch[TARGET_TOKEN_IDS_KEY] = aligned_token_ids
    return aligned_token_ids

@dataclass
class V6ExplicitCurrentCollator:
    """Encode raw ``current_md`` as a teacher-forcing target without row mutation."""
    processor: Any
    image_module: Any
    max_length: int
    max_output_tokens: int | None
    min_supervised_tokens: int
    sidecars_by_id: Mapping[str, ValidatedSidecar]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        assistant_targets: list[str] = []
        images: list[Any] = []
        sample_ids: list[str] = []
        for sample in features:
            prompt_messages = build_local_world_model_messages(sample, include_output=False, prompt_key='localwm_next')
            current_target = canonical_v6_current_target(sample)
            training_messages = [*prompt_messages, {'role': 'assistant', 'content': [{'type': 'text', 'text': current_target}]}]
            prompt_texts.append(apply_chat_template(self.processor, prompt_messages, tokenize=False, add_generation_prompt=True))
            texts.append(apply_chat_template(self.processor, training_messages, tokenize=False, add_generation_prompt=False))
            assistant_targets.append(current_target)
            images.append(self.image_module.open(sample['current_image_path']).convert('RGB'))
            sample_ids.append(str(sample.get('id') or '<unknown>'))
        if self.max_output_tokens is not None:
            output_lengths = [sft_module.token_count(self.processor.tokenizer, target) for target in assistant_targets]
            overlong = [(sample_id, length) for (sample_id, length) in zip(sample_ids, output_lengths) if length > self.max_output_tokens]
            if overlong:
                raise V6SftError(f'Current target exceeds max_output_tokens={self.max_output_tokens}: {overlong}')
        batch = self.processor(text=texts, images=images, return_tensors='pt', padding=True, truncation=False)
        prompt_batch = self.processor(text=prompt_texts, images=images, return_tensors='pt', padding=True, truncation=False)
        full_lengths = [int(mask.sum().item()) for mask in batch['attention_mask']]
        overlong_sequences = [(sample_id, length) for (sample_id, length) in zip(sample_ids, full_lengths) if length > self.max_length]
        if overlong_sequences:
            raise V6SftError(f'Current training sequence exceeds max_length={self.max_length}: {overlong_sequences}')
        labels = batch['input_ids'].clone()
        pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        sft_module.mask_prompt_tokens(labels, batch['attention_mask'], prompt_batch['attention_mask'], tokenizer=self.processor.tokenizer, assistant_targets=assistant_targets, assistant_loss_scope='full', padding_side=getattr(self.processor.tokenizer, 'padding_side', 'right'), min_supervised_tokens=self.min_supervised_tokens, sample_ids=sample_ids)
        batch['labels'] = labels
        _align_explicit_target_units(batch, features, self.sidecars_by_id, self.processor.tokenizer)
        return batch

class V6PairedCollator:

    def __init__(self, next_collator: V6Collator, current_collator: V6ExplicitCurrentCollator) -> None:
        self.next_collator = next_collator
        self.current_collator = current_collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        next_batch = self.next_collator(features)
        current_batch = self.current_collator(features)
        next_target_ids = next_batch.pop(TARGET_TOKEN_IDS_KEY)
        current_target_ids = current_batch.pop(TARGET_TOKEN_IDS_KEY)
        if len(next_target_ids) != len(features) or len(current_target_ids) != len(features):
            raise V6SftError('paired target-token metadata has the wrong batch size')

        def prompt_prefixes(batch: Mapping[str, Any]) -> list[list[int]]:
            input_ids = batch.get('input_ids')
            labels = batch.get('labels')
            attention_mask = batch.get('attention_mask')
            if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise V6SftError('paired batch is missing tensor input_ids/labels')
            if input_ids.shape != labels.shape or input_ids.ndim != 2:
                raise V6SftError('paired input_ids/labels must have equal [B,T] shapes')
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
                raise V6SftError('paired attention_mask has the wrong shape')
            prefixes: list[list[int]] = []
            for (row_inputs, row_labels, row_mask) in zip(input_ids, labels, attention_mask):
                valid = row_mask.to(dtype=torch.bool)
                compact_inputs = row_inputs[valid]
                compact_labels = row_labels[valid]
                supervised = compact_labels.ne(-100).nonzero(as_tuple=False)
                if supervised.numel() == 0:
                    raise V6SftError('paired target contains no supervised token')
                first_target = int(supervised[0].item())
                prefixes.append(compact_inputs[:first_target].tolist())
            return prefixes
        if prompt_prefixes(next_batch) != prompt_prefixes(current_batch):
            raise V6SftError('Next/Current source prompt token prefixes differ')
        eligible: list[bool] = []
        identity_excluded: list[bool] = []
        sample_ids: list[str] = []
        for (sample, next_ids, current_ids) in zip(features, next_target_ids, current_target_ids):
            sample_id = str(sample.get('id') or '').strip()
            changed = str(sample.get('transition_class') or '') == 'changed'
            identical = list(next_ids) == list(current_ids)
            eligible.append(changed and (not identical))
            identity_excluded.append(changed and identical)
            sample_ids.append(sample_id)
        for key in ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'):
            next_value = next_batch.get(key)
            current_value = current_batch.get(key)
            if isinstance(next_value, torch.Tensor) or isinstance(current_value, torch.Tensor):
                if not (isinstance(next_value, torch.Tensor) and isinstance(current_value, torch.Tensor) and torch.equal(next_value, current_value)):
                    raise V6SftError(f'Next/Current source tensor {key!r} differs')
        next_batch[CURRENT_INPUTS_KEY] = current_batch
        next_batch[RANK_ELIGIBLE_KEY] = torch.tensor(eligible, dtype=torch.bool)
        next_batch[IDENTITY_EXCLUDED_KEY] = torch.tensor(identity_excluded, dtype=torch.bool)
        next_batch[SAMPLE_IDS_KEY] = sample_ids
        return next_batch

def _validate_loss_inputs(token_ce: torch.Tensor, base_unit_ids: torch.Tensor, focus_unit_ids: torch.Tensor) -> None:
    if token_ce.ndim != 2:
        raise V6SftError('token_ce must have shape [batch, tokens]')
    if base_unit_ids.shape != token_ce.shape or focus_unit_ids.shape != token_ce.shape:
        raise V6SftError('token CE and V6 unit id tensors must have equal shapes')
    if not token_ce.is_floating_point():
        raise V6SftError('token_ce must be floating point')
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if base_unit_ids.dtype not in integer_dtypes or focus_unit_ids.dtype not in integer_dtypes:
        raise V6SftError('V6 unit ids must use an integer dtype')
    if token_ce.device != base_unit_ids.device or token_ce.device != focus_unit_ids.device:
        raise V6SftError('token CE and V6 unit ids must be on the same device')
    if bool(base_unit_ids.lt(IGNORE_UNIT_ID).any()) or bool(focus_unit_ids.lt(IGNORE_UNIT_ID).any()):
        raise V6SftError('V6 unit ids may only use -1 as the ignore value')
    if bool(focus_unit_ids.ge(0).logical_and(base_unit_ids.lt(0)).any()):
        raise V6SftError('every focus token must also belong to a base unit')

def _scatter_unit_means(token_ce: torch.Tensor, unit_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each sample's sum of unit means and number of non-empty units."""
    (batch_size, token_count) = token_ce.shape
    stride = token_count + 1
    valid = unit_ids.ge(0)
    if bool(unit_ids.masked_fill(~valid, 0).ge(token_count).any()):
        raise V6SftError('unit id exceeds the number of available token positions')
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

def compute_v6_loss(token_ce: torch.Tensor, base_unit_ids: torch.Tensor, focus_unit_ids: torch.Tensor, *, base_loss_granularity: str='element', focus_loss_granularity: str='element') -> torch.Tensor:
    if base_loss_granularity != 'element' or focus_loss_granularity != 'element':
        raise V6SftError('V6 original loss is frozen to the V4-EE profile')
    try:
        return frozen_v4.compute_v4_loss(token_ce, base_unit_ids, focus_unit_ids, base_loss_granularity='element', focus_loss_granularity='element')
    except frozen_v4.V4SftError as exc:
        raise V6SftError(f'frozen V4-EE reducer rejected V6 tensors: {exc}') from exc

def page_element_ce(token_ce: torch.Tensor, base_unit_ids: torch.Tensor) -> torch.Tensor:
    """Average tokens inside each base element, then elements inside each page."""
    if token_ce.ndim != 2 or base_unit_ids.shape != token_ce.shape:
        raise V6SftError('page Element CE requires equal [batch,tokens] tensors')
    dummy_focus = torch.full_like(base_unit_ids, IGNORE_UNIT_ID)
    _validate_loss_inputs(token_ce, base_unit_ids, dummy_focus)
    (base_sum, base_count) = _scatter_unit_means(token_ce, base_unit_ids)
    if bool(base_count.eq(0).any()):
        raise V6SftError('every page score must contain at least one base element')
    return base_sum / base_count.to(base_sum.dtype)

def v6_beta(global_step: int) -> float:
    if isinstance(global_step, bool) or int(global_step) < 0:
        raise V6SftError('global_step must be a non-negative optimizer-step index')
    update = int(global_step) + 1
    return V6_BETA_MAX * min(1.0, update / V6_BETA_WARMUP_STEPS)

def v6_ranking_terms(next_page_ce: torch.Tensor, current_page_ce: torch.Tensor, eligible: torch.Tensor, *, margin: float=V6_MARGIN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample hinge, active mask, and Current-minus-Next gap."""
    if next_page_ce.ndim != 1 or current_page_ce.shape != next_page_ce.shape:
        raise V6SftError('ranking page scores must have equal [batch] shapes')
    if eligible.shape != next_page_ce.shape or eligible.dtype != torch.bool:
        raise V6SftError('ranking eligibility must be a bool [batch] tensor')
    gap = current_page_ce - next_page_ce
    violation = margin - gap
    active = eligible & violation.gt(0)
    hinge = violation.clamp_min(0) * eligible.to(violation.dtype)
    return (hinge, active, gap)

def _causal_v6_components(logits: torch.Tensor, labels: torch.Tensor, base_unit_ids: torch.Tensor, focus_unit_ids: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Compute causal token CE once and expose V4-EE/page-element reductions."""
    if logits.ndim != 3 or logits.shape[1] < 2:
        raise V6SftError('causal V6 components require [batch,seq,vocab] logits')
    if labels.shape != logits.shape[:2] or base_unit_ids.shape != labels.shape:
        raise V6SftError('labels/base units must match logits batch/sequence shape')
    if focus_unit_ids is not None and focus_unit_ids.shape != labels.shape:
        raise V6SftError('focus units must match logits batch/sequence shape')
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_base = base_unit_ids[:, 1:].contiguous()
    if bool(shift_base.ge(0).ne(shift_labels.ne(-100)).any()):
        raise V6SftError('supervised labels and base-unit membership differ')
    shift_labels = shift_labels.masked_fill(shift_base.lt(0), -100)
    token_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), reduction='none', ignore_index=-100).view_as(shift_labels)
    page_ce = page_element_ce(token_ce, shift_base)
    original = None
    if focus_unit_ids is not None:
        shift_focus = focus_unit_ids[:, 1:].contiguous()
        original = compute_v6_loss(token_ce, shift_base, shift_focus, base_loss_granularity='element', focus_loss_granularity='element')
    return (token_ce, original, page_ce)

def compute_causal_v6_loss(logits: torch.Tensor, labels: torch.Tensor, base_unit_ids: torch.Tensor, focus_unit_ids: torch.Tensor, *, base_loss_granularity: str='element', focus_loss_granularity: str='element') -> torch.Tensor:
    """Compute causal token CE once, shift unit ids once, then apply V6 loss."""
    if base_loss_granularity != 'element' or focus_loss_granularity != 'element':
        raise V6SftError('causal V6 original loss is frozen to V4-EE')
    (_token_ce, original, _page) = _causal_v6_components(logits, labels, base_unit_ids, focus_unit_ids)
    if original is None:
        raise V6SftError('causal V6 original loss requires focus-unit ids')
    return original

class V6Trainer(_TransformersTrainer):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.base_loss_granularity = kwargs.pop('base_loss_granularity', 'element')
        self.focus_loss_granularity = kwargs.pop('focus_loss_granularity', 'element')
        expected_profile = kwargs.pop('loss_profile', None)
        if expected_profile is not None:
            expected_pair = V6_LOSS_PROFILES.get(str(expected_profile))
            actual_pair = (self.base_loss_granularity, self.focus_loss_granularity)
            if expected_pair != actual_pair:
                raise V6SftError(f'V6 Trainer loss profile/granularity mismatch: profile={expected_profile!r}, actual={actual_pair!r}')
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._v6_raw_metric_sums: dict[str, float] = {}
        self._v6_ready_metrics: dict[str, float] | None = None

    @staticmethod
    def _model_logits(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, Mapping):
            logits = outputs.get('logits')
        else:
            logits = getattr(outputs, 'logits', None)
            if logits is None and isinstance(outputs, (tuple, list)) and outputs:
                logits = outputs[0]
        if not isinstance(logits, torch.Tensor):
            raise V6SftError('model output does not contain tensor logits')
        return logits

    @staticmethod
    def _capture_rng(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())

    @staticmethod
    def _restore_rng(state: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> None:
        (cpu_state, cuda_state) = state
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)

    @staticmethod
    def _split_batch(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, list[str]]:
        work = dict(inputs)
        try:
            current = work.pop(CURRENT_INPUTS_KEY)
            eligible = work.pop(RANK_ELIGIBLE_KEY)
            identity_excluded = work.pop(IDENTITY_EXCLUDED_KEY)
            sample_ids = work.pop(SAMPLE_IDS_KEY)
        except KeyError as exc:
            raise V6SftError(f'V6 paired batch is missing {exc.args[0]!r}') from exc
        if not isinstance(current, Mapping):
            raise V6SftError('current_inputs must be a nested tensor mapping')
        current = dict(current)
        if not isinstance(eligible, torch.Tensor) or eligible.dtype != torch.bool:
            raise V6SftError('rank_eligible must be a bool tensor')
        if not isinstance(identity_excluded, torch.Tensor) or identity_excluded.dtype != torch.bool:
            raise V6SftError('identity_excluded must be a bool tensor')
        if not isinstance(sample_ids, list):
            raise V6SftError('sample_ids must be a list')
        return (work, current, eligible, identity_excluded, sample_ids)

    def _accumulate_v6_raw_metrics(self, metrics: Mapping[str, float]) -> None:
        for (key, value) in metrics.items():
            if key == 'current_replay_max_error':
                self._v6_raw_metric_sums[key] = max(self._v6_raw_metric_sums.get(key, 0.0), float(value))
            else:
                self._v6_raw_metric_sums[key] = self._v6_raw_metric_sums.get(key, 0.0) + float(value)

    def _finalize_v6_metrics(self, *, device: torch.device) -> None:
        """All-reduce one optimizer update's raw numerators over all six ranks."""
        sum_keys = ('original_sum', 'hinge_sum', 'next_eligible_sum', 'current_eligible_sum', 'gap_eligible_sum', 'batch_count', 'eligible_count', 'active_count', 'identity_excluded_count', 'beta_batch_sum', 'gate_flip_count')
        values = torch.tensor([self._v6_raw_metric_sums.get(key, 0.0) for key in sum_keys], dtype=torch.float64, device=device)
        replay_max = torch.tensor([self._v6_raw_metric_sums.get('current_replay_max_error', 0.0)], dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(replay_max, op=torch.distributed.ReduceOp.MAX)
        totals = {key: float(value) for (key, value) in zip(sum_keys, values.tolist())}
        batch_count = totals['batch_count']
        eligible_count = totals['eligible_count']
        if batch_count <= 0:
            raise V6SftError('V6 metric reduction has an empty optimizer batch')
        beta = totals['beta_batch_sum'] / batch_count
        ranking_loss = totals['hinge_sum'] / batch_count
        eligible_denominator = max(eligible_count, 1.0)
        next_ce = totals['next_eligible_sum'] / eligible_denominator
        current_ce = totals['current_eligible_sum'] / eligible_denominator
        ce_gap = totals['gap_eligible_sum'] / eligible_denominator
        self._v6_ready_metrics = {'original_v4_loss': totals['original_sum'] / batch_count, 'ranking_loss': ranking_loss, 'weighted_ranking_loss': beta * ranking_loss, 'next_ce': next_ce, 'current_ce': current_ce, 'ce_gap': ce_gap, 'margin_violation': V6_MARGIN - ce_gap, 'batch_count': batch_count, 'eligible_count': eligible_count, 'eligible_rate': eligible_count / batch_count, 'active_count': totals['active_count'], 'ranking_active_rate': totals['active_count'] / eligible_denominator, 'beta': beta, 'identity_excluded_count': totals['identity_excluded_count'], 'current_replay_max_error': float(replay_max.item()), 'gate_flip_count': totals['gate_flip_count']}
        self._v6_raw_metric_sums.clear()

    def training_step(self, model: Any, inputs: dict[str, Any], num_items_in_batch: torch.Tensor | int | None=None) -> torch.Tensor:
        del num_items_in_batch
        (cp_context, inputs) = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            (next_inputs, current_inputs, eligible, identity_excluded, sample_ids) = self._split_batch(inputs)
            if eligible.ndim != 1 or len(sample_ids) != eligible.numel():
                raise V6SftError('V6 eligibility/sample metadata has wrong shape')
            if not torch.cuda.is_available():
                raise V6SftError('V6 formal 3F+2B requires CUDA')
            device = next(model.parameters()).device
            if device.type != 'cuda':
                raise V6SftError('V6 formal model must reside on CUDA')
            rng0 = self._capture_rng(device)
            current_labels = current_inputs.pop('labels')
            current_base_ids = current_inputs.pop(BASE_UNIT_IDS_KEY)
            current_model_inputs = dict(current_inputs)
            unwrapped = self.accelerator.unwrap_model(model)
            with self.compute_loss_context_manager(), torch.no_grad():
                current_outputs = unwrapped(**current_model_inputs)
                current_logits = self._model_logits(current_outputs)
                (_token_ce, _unused_original, current_score) = _causal_v6_components(current_logits, current_labels, current_base_ids, None)
                current_score = current_score.detach()
            del current_outputs, current_logits, _token_ce
            torch.cuda.empty_cache()
            self._restore_rng(rng0, device)
            next_labels = next_inputs.pop('labels')
            next_base_ids = next_inputs.pop(BASE_UNIT_IDS_KEY)
            next_focus_ids = next_inputs.pop(FOCUS_UNIT_IDS_KEY)
            with self.compute_loss_context_manager():
                next_outputs = model(**next_inputs)
                next_logits = self._model_logits(next_outputs)
                (_next_token_ce, original_v4, next_page_ce) = _causal_v6_components(next_logits, next_labels, next_base_ids, next_focus_ids)
                if original_v4 is None:
                    raise V6SftError('Next forward did not produce V4-EE loss')
                (hinge, active, gap) = v6_ranking_terms(next_page_ce, current_score, eligible)
                beta = v6_beta(int(self.state.global_step))
                positive_loss = original_v4 + beta * (active.to(next_page_ce.dtype) * next_page_ce).mean()
            ga = int(self.current_gradient_accumulation_steps)
            if ga <= 0:
                raise V6SftError('current_gradient_accumulation_steps must be positive')
            self.accelerator.backward(positive_loss / ga)
            original_detached = original_v4.detach()
            next_detached = next_page_ce.detach()
            hinge_detached = hinge.detach()
            active_detached = active.detach()
            gap_detached = gap.detach()
            del next_outputs, next_logits, _next_token_ce, original_v4, next_page_ce
            del positive_loss, hinge, active, gap
            torch.cuda.empty_cache()
            with torch.random.fork_rng(devices=[device.index]):
                self._restore_rng(rng0, device)
                with self.compute_loss_context_manager():
                    replay_outputs = model(**current_model_inputs)
                    replay_logits = self._model_logits(replay_outputs)
                    (_replay_token_ce, _unused_original, current_replay) = _causal_v6_components(replay_logits, current_labels, current_base_ids, None)
                    replay_error = (current_replay.detach() - current_score).abs()
                    max_replay_error = float(replay_error.max().item())
                    if max_replay_error > V6_REPLAY_ATOL:
                        raise V6SftError(f'Current score/replay mismatch: max_abs={max_replay_error:.8g}>{V6_REPLAY_ATOL}')
                    (replay_hinge, replay_active, _replay_gap) = v6_ranking_terms(next_detached, current_replay.detach(), eligible)
                    gate_flips = int(replay_active.ne(active_detached).sum().item())
                    if gate_flips:
                        raise V6SftError(f'Current replay changed {gate_flips} ranking gates')
                    negative_loss = -beta * (active_detached.to(current_replay.dtype) * current_replay).mean()
                self.accelerator.backward(negative_loss / ga)
                del replay_outputs, replay_logits, _replay_token_ce
                del current_replay, replay_hinge, replay_active, _replay_gap
                del negative_loss
            eligible_count = int(eligible.sum().item())
            active_count = int(active_detached.sum().item())
            eligible_float = eligible.to(next_detached.dtype)
            local_batch = int(eligible.numel())
            raw_metrics = {'original_sum': float(original_detached.item()) * local_batch, 'hinge_sum': float(hinge_detached.sum().item()), 'next_eligible_sum': float((next_detached * eligible_float).sum().item()), 'current_eligible_sum': float((current_score * eligible_float).sum().item()), 'gap_eligible_sum': float((gap_detached * eligible_float).sum().item()), 'batch_count': float(local_batch), 'eligible_count': float(eligible_count), 'active_count': float(active_count), 'identity_excluded_count': float(identity_excluded.sum().item()), 'beta_batch_sum': beta * local_batch, 'current_replay_max_error': max_replay_error, 'gate_flip_count': float(gate_flips)}
            self._accumulate_v6_raw_metrics(raw_metrics)
            if bool(self.accelerator.sync_gradients):
                self._finalize_v6_metrics(device=device)
            report_loss = (original_detached + beta * hinge_detached.mean()) / ga
            return report_loss.detach()

    def log(self, logs: dict[str, float], start_time: float | None=None) -> None:
        if self._v6_ready_metrics is not None:
            logs.update(self._v6_ready_metrics)
            self._v6_ready_metrics = None
        super().log(logs, start_time=start_time)

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool=False, num_items_in_batch: torch.Tensor | int | None=None) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        if CURRENT_INPUTS_KEY in inputs:
            raise V6SftError('V6 paired training must use V6Trainer.training_step')
        model_inputs = dict(inputs)
        try:
            labels = model_inputs.pop('labels')
            base_unit_ids = model_inputs.pop(BASE_UNIT_IDS_KEY)
            focus_unit_ids = model_inputs.pop(FOCUS_UNIT_IDS_KEY)
        except KeyError as exc:
            raise V6SftError(f'V6 Trainer batch is missing {exc.args[0]!r}') from exc
        outputs = model(**model_inputs)
        logits = self._model_logits(outputs)
        loss = compute_causal_v6_loss(logits, labels, base_unit_ids, focus_unit_ids, base_loss_granularity=getattr(self, 'base_loss_granularity', 'element'), focus_loss_granularity=getattr(self, 'focus_loss_granularity', 'element'))
        return (loss, outputs) if return_outputs else loss

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _normalized_adapter_module_name(name: str) -> str:
    prefix = 'base_model.model.'
    return name[len(prefix):] if name.startswith(prefix) else name

def validate_bare_base_model(base_model: str | Path) -> dict[str, Any]:
    """Pin the local bare Qwen3.5-9B identity without loading model weights."""
    requested = _require_lexical_path(base_model, CANONICAL_BASE_MODEL, context='V6-FullSFT base model')
    if not requested.is_dir() or requested.resolve(strict=True) != EXPECTED_BASE_RESOLVED_PATH:
        raise V6SftError('canonical bare Qwen3.5-9B directory no longer resolves to its pinned target')
    config_path = requested / 'config.json'
    index_path = requested / 'model.safetensors.index.json'
    for path in (config_path, index_path):
        if path.is_symlink() or not path.is_file():
            raise V6SftError(f'bare base-model asset is missing or a symlink: {path}')
    for forbidden in ('adapter_config.json', 'adapter_model.safetensors'):
        if (requested / forbidden).exists():
            raise V6SftError(f'bare base-model directory unexpectedly contains {forbidden}')
    config_sha256 = _sha256_file(config_path)
    index_sha256 = _sha256_file(index_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('architectures') != ['Qwen3_5ForConditionalGeneration']:
        raise V6SftError('bare base-model architecture drifted')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or len(weight_map) != 775:
        raise V6SftError('bare base-model weight index must contain 775 tensors')
    shards = sorted(set(weight_map.values()))
    if shards != sorted(EXPECTED_BASE_SHARDS):
        raise V6SftError('bare base-model weight index references unexpected shards')
    shard_reports: dict[str, dict[str, Any]] = {}
    for name in shards:
        shard_path = requested / name
        (expected_size, expected_sha256) = EXPECTED_BASE_SHARDS[name]
        if shard_path.is_symlink() or not shard_path.is_file() or shard_path.resolve(strict=True).parent != EXPECTED_BASE_RESOLVED_PATH or (shard_path.stat().st_size != expected_size):
            raise V6SftError(f'bare base-model shard identity drifted: {name}')
        actual_sha256 = _sha256_file(shard_path)
        shard_reports[name] = {'size_bytes': expected_size, 'sha256': expected_sha256, 'resolved_path': str(shard_path.resolve(strict=True))}
    return {'path': str(requested), 'resolved_path': str(requested.resolve()), 'config_sha256': config_sha256, 'weight_index_sha256': index_sha256, 'weight_tensors': len(weight_map), 'weight_shards': shards, 'weight_shard_contract': shard_reports, 'contains_adapter': False}

def validate_v3_cur_1200_init_adapter(init_adapter: str | Path) -> dict[str, Any]:
    """Pin V6 initialization to the complete V3-Cur-1200 LoRA adapter."""
    adapter_path = _require_lexical_path(init_adapter, CANONICAL_INIT_ADAPTER, context='V6 --init-adapter')
    if adapter_path.is_symlink() or not adapter_path.is_dir():
        raise V6SftError('V3-Cur-1200 init adapter must be a regular non-symlink directory')
    integrity_path = adapter_path / 'checkpoint_integrity.json'
    integrity = _load_json_object(integrity_path, context='V3-Cur-1200 checkpoint integrity')
    if integrity.get('status') != 'passed' or int(integrity.get('global_step', -1)) != 1200 or int(integrity.get('world_size', -1)) != V6_WORLD_SIZE or (int(integrity.get('adapter_tensor_count', -1)) != EXPECTED_ADAPTER_TENSORS):
        raise V6SftError('V3-Cur-1200 checkpoint integrity contract failed')
    adapter = validate_expanded_adapter_files(adapter_path)
    if int(adapter.get('tensor_count', -1)) != EXPECTED_ADAPTER_TENSORS or int(adapter.get('tensor_parameters', -1)) != EXPECTED_TRAINABLE_PARAMS or int(adapter.get('module_count', -1)) != EXPECTED_EXPANDED_MODULES:
        raise V6SftError('V3-Cur-1200 Expanded-LoRA contract failed')
    source_run_dir = adapter_path.parent.parent
    source_run_config = source_run_dir / 'run_config.json'
    selection_path = source_run_dir / 'validation_selection.json'
    selection_pointer = source_run_dir / 'validation_selection.best_adapter.txt'
    selection = _load_json_object(selection_path, context='V3-Cur checkpoint selection manifest')
    if selection.get('status') != 'passed' or selection.get('best', {}).get('step') != 1200 or selection.get('best', {}).get('adapter') != str(CANONICAL_INIT_ADAPTER) or (selection_pointer.read_text(encoding='utf-8').strip() != str(CANONICAL_INIT_ADAPTER)):
        raise V6SftError('V3-Cur-1200 is no longer the selected source adapter')
    return {**adapter, 'path': str(adapter_path), 'display_name': 'V3-Cur-1200', 'source_global_step': 1200, 'checkpoint_integrity_path': str(integrity_path), 'checkpoint_integrity_sha256': EXPECTED_INIT_ADAPTER_INTEGRITY_SHA256, 'source_run_config': str(source_run_config), 'source_run_config_sha256': EXPECTED_INIT_SOURCE_RUN_CONFIG_SHA256, 'selection_manifest': str(selection_path), 'selection_manifest_sha256': EXPECTED_INIT_SELECTION_SHA256, 'selection_pointer': str(selection_pointer), 'selection_pointer_sha256': EXPECTED_INIT_SELECTION_POINTER_SHA256}

def v6_adapter_initialization_contract() -> dict[str, Any]:
    return {'mode': V6_ADAPTER_INIT_MODE, 'base_model': CANONICAL_BASE_MODEL, 'init_adapter': str(CANONICAL_INIT_ADAPTER), 'source_display_name': 'V3-Cur-1200', 'source_global_step': 1200, 'loads_v3_adapter': True, 'is_trainable': True, 'fresh_optimizer': True, 'fresh_scheduler': True, 'initial_global_step': 0, 'adapter_config_sha256': EXPECTED_INIT_ADAPTER_CONFIG_SHA256, 'adapter_weights_sha256': EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256, 'checkpoint_integrity_sha256': EXPECTED_INIT_ADAPTER_INTEGRITY_SHA256, 'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none', 'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'target_modules': 'qwen35_expanded', 'expected_module_count': EXPECTED_EXPANDED_MODULES, 'expected_tensor_count': EXPECTED_ADAPTER_TENSORS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'expected_trainable_parameters': EXPECTED_TRAINABLE_PARAMS}

def v6_formal_completion_contract() -> dict[str, Any]:
    steps_per_epoch = math.ceil(EXPECTED_TRAINING_ROWS / V6_GLOBAL_BATCH_SIZE)
    if steps_per_epoch * 2 != EXPECTED_FINAL_STEP:
        raise V6SftError('internal V6 final-step contract is inconsistent')
    return {'canonical_training_rows': EXPECTED_TRAINING_ROWS, 'epochs': 2.0, 'world_size': V6_WORLD_SIZE, 'per_device_train_batch_size': V6_PER_DEVICE_TRAIN_BATCH_SIZE, 'gradient_accumulation_steps': V6_GRADIENT_ACCUMULATION_STEPS, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'steps_per_epoch': steps_per_epoch, 'expected_final_step': EXPECTED_FINAL_STEP, 'expected_trainable_parameters': EXPECTED_TRAINABLE_PARAMS}
V5_PARITY_ARG_FIELDS: tuple[str, ...] = ('adam_beta1', 'adam_beta2', 'adam_epsilon', 'assistant_loss_scope', 'base_loss_granularity', 'base_model', 'bf16', 'dataloader_num_workers', 'dataloader_persistent_workers', 'dataloader_pin_memory', 'dataloader_prefetch_factor', 'dataset_loading_mode', 'device_mode', 'disable_tqdm', 'eval_jsonl', 'eval_steps', 'focus_loss_granularity', 'focus_sidecar', 'fp16', 'gradient_accumulation_steps', 'gradient_checkpointing', 'image_max_pixels', 'image_min_pixels', 'learning_rate', 'length_filter_report_limit', 'logging_steps', 'lora_alpha', 'lora_dropout', 'lora_r', 'loss_profile', 'lr_scheduler_type', 'max_assistant_output_chars', 'max_current_md_chars', 'max_eval_samples', 'max_grad_norm', 'max_input_tokens', 'max_length', 'max_output_tokens', 'max_target_next_md_chars', 'max_train_samples', 'min_supervised_tokens', 'per_device_eval_batch_size', 'per_device_train_batch_size', 'prompt_key', 'save_steps', 'save_total_limit', 'seed', 'skip_missing_images', 'skip_overlong_samples', 'target_modules', 'train_jsonl', 'transition_cache_root', 'warmup_ratio', 'weight_decay', 'wm_eval_checkpoint', 'wm_eval_ddp_timeout_seconds', 'wm_eval_embedding_device', 'wm_eval_embedding_model', 'wm_eval_in_domain_jsonl', 'wm_eval_jsonl', 'wm_eval_max_elef1_drop_from_best', 'wm_eval_max_new_tokens', 'wm_eval_max_samples', 'wm_eval_min_format_success_rate', 'wm_eval_only', 'wm_eval_ood_jsonl', 'wm_eval_output_dir', 'wm_eval_step', 'wm_eval_stop_after_bad_checkpoints', 'wm_eval_workers_per_gpu')

def validate_v6_v5_parity_contract(args: argparse.Namespace) -> dict[str, Any]:
    v5 = _load_json_object(CANONICAL_V5_RUN_CONFIG, context='frozen V5 formal run_config')
    v5_args = v5.get('args')
    if not isinstance(v5_args, Mapping):
        raise V6SftError('frozen V5 run_config lacks args')

    def comparable(value: Any) -> Any:
        return os.fspath(value) if isinstance(value, Path) else value
    mismatches = {name: {'v5': v5_args.get(name), 'v6': comparable(getattr(args, name, None))} for name in V5_PARITY_ARG_FIELDS if comparable(getattr(args, name, None)) != v5_args.get(name)}
    if not args.certify_run and args.num_train_epochs != v5_args.get('num_train_epochs'):
        mismatches['num_train_epochs'] = {'v5': v5_args.get('num_train_epochs'), 'v6': args.num_train_epochs}
    if mismatches:
        raise V6SftError(f'V6 differs from frozen V5 parameters: {mismatches}')
    v5_loss = v5.get('loss')
    v6_loss = v6_loss_contract(args.base_loss_granularity, args.focus_loss_granularity)
    if v6_loss != v5_loss:
        raise V6SftError('V6 loss contract differs from frozen V5')
    return {'status': 'passed', 'v5_run_config': str(CANONICAL_V5_RUN_CONFIG), 'v5_run_config_sha256': EXPECTED_V5_RUN_CONFIG_SHA256, 'equal_argument_fields': list(V5_PARITY_ARG_FIELDS), 'formal_epochs_equal': not args.certify_run, 'loss_equal': True, 'allowed_differences': ['run_id/display/output/log/schema identity', 'init_adapter: null -> V3-Cur-1200', 'certify-only rows/steps/artifacts']}

def validate_expanded_adapter_files(adapter_dir: str | Path, *, expected_modules: Sequence[str] | None=None) -> dict[str, Any]:
    """Validate one V6-produced Expanded-LoRA adapter on CPU."""
    adapter_path = Path(adapter_dir)
    if adapter_path.is_symlink() or not adapter_path.is_dir():
        raise V6SftError(f'adapter directory is missing or a symlink: {adapter_path}')
    config_path = adapter_path / 'adapter_config.json'
    weights_path = adapter_path / 'adapter_model.safetensors'
    if config_path.is_symlink() or weights_path.is_symlink() or (not config_path.is_file()) or (not weights_path.is_file()):
        raise V6SftError(f'incomplete Expanded LoRA adapter: {adapter_path}')
    config_sha256 = _sha256_file(config_path)
    weights_sha256 = _sha256_file(weights_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected_config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none', 'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05}
    mismatches = {key: {'expected': expected, 'actual': config.get(key)} for (key, expected) in expected_config.items() if config.get(key) != expected}
    if mismatches:
        raise V6SftError(f'Expanded LoRA config mismatch: {mismatches}')
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise V6SftError('adapter validation requires safetensors') from exc
    pattern = re.compile('^(?P<module>.+)\\.lora_(?P<side>[AB])\\.weight$')
    module_sides: dict[str, dict[str, tuple[int, ...]]] = {}
    tensor_parameters = 0
    accepted_dtypes = {'BF16', 'F16', 'F32'}
    with safe_open(str(weights_path), framework='pt', device='cpu') as handle:
        tensor_keys = list(handle.keys())
        for key in tensor_keys:
            match = pattern.fullmatch(key)
            if match is None:
                raise V6SftError(f'unexpected adapter tensor key: {key}')
            module = _normalized_adapter_module_name(match.group('module'))
            side = match.group('side')
            tensor_slice = handle.get_slice(key)
            shape = tuple((int(size) for size in tensor_slice.get_shape()))
            dtype = str(tensor_slice.get_dtype())
            if len(shape) != 2 or dtype not in accepted_dtypes:
                raise V6SftError(f'adapter tensor {key} has invalid shape/dtype: {shape}/{dtype}')
            if side in module_sides.setdefault(module, {}):
                raise V6SftError(f'duplicate adapter tensor side: {key}')
            module_sides[module][side] = shape
            tensor_parameters += math.prod(shape)
    if len(tensor_keys) != EXPECTED_ADAPTER_TENSORS:
        raise V6SftError(f'Expanded LoRA requires {EXPECTED_ADAPTER_TENSORS} tensors, found {len(tensor_keys)}')
    if len(module_sides) != EXPECTED_EXPANDED_MODULES or any((set(sides) != {'A', 'B'} for sides in module_sides.values())):
        raise V6SftError('Expanded LoRA requires exactly 358 modules with one A/B tensor pair')
    invalid_shapes = {module: sides for (module, sides) in module_sides.items() if sides['A'][0] != 32 or sides['B'][1] != 32 or sides['A'][1] <= 0 or (sides['B'][0] <= 0)}
    if invalid_shapes:
        raise V6SftError(f'Expanded LoRA A/B rank or paired dimensions drifted: {list(invalid_shapes.items())[:3]}')
    canonical_modules = tuple(sorted(module_sides))
    if canonical_modules != expected_qwen35_expanded_target_modules():
        raise V6SftError('adapter tensor modules differ from frozen qwen35_expanded')
    if expected_modules is not None and tuple(sorted(expected_modules)) != canonical_modules:
        raise V6SftError('adapter tensor modules differ from this V6 run')
    if tensor_parameters != EXPECTED_TRAINABLE_PARAMS:
        raise V6SftError(f'Expanded LoRA tensor parameter count drifted: expected={EXPECTED_TRAINABLE_PARAMS}, actual={tensor_parameters}')
    return {'path': str(adapter_path), 'config_sha256': config_sha256, 'weights_sha256': weights_sha256, 'tensor_count': len(tensor_keys), 'tensor_parameters': tensor_parameters, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'module_count': len(module_sides), 'r': config['r'], 'lora_alpha': config['lora_alpha'], 'lora_dropout': config['lora_dropout']}

def validate_loaded_expanded_modules(model: Any, expected_modules: Sequence[str]) -> dict[str, Any]:
    """Require the newly created trainable PEFT graph to match all 358 modules."""
    canonical_expected = _validate_target_modules(list(expected_modules), context='loaded adapter target_modules')
    peft_config = getattr(model, 'peft_config', None)
    if not isinstance(peft_config, Mapping) or len(peft_config) != 1:
        raise V6SftError('V6 requires exactly one loaded PEFT adapter')
    adapter_names = tuple(sorted((str(name) for name in peft_config)))
    active_adapters = getattr(model, 'active_adapters', None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]
    if not isinstance(active_adapters, (list, tuple)) or tuple(sorted((str(name) for name in active_adapters))) != adapter_names:
        raise V6SftError('V6 single PEFT adapter is not the sole active adapter')
    expected = set(canonical_expected)
    actual = {_normalized_adapter_module_name(name) for (name, module) in model.named_modules() if hasattr(module, 'lora_A') and hasattr(module, 'lora_B')}
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise V6SftError(f'loaded Expanded LoRA module mismatch: missing={missing}, extra={extra}')
    invalid_trainable = [name for (name, parameter) in model.named_parameters() if parameter.requires_grad and '.lora_A.' not in name and ('.lora_B.' not in name)]
    if invalid_trainable:
        raise V6SftError(f'non-LoRA parameters are trainable: {invalid_trainable[:10]}')
    trainable = [(name, parameter) for (name, parameter) in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != EXPECTED_ADAPTER_TENSORS:
        raise V6SftError(f'Expanded LoRA trainable tensor count drifted: expected={EXPECTED_ADAPTER_TENSORS}, actual={len(trainable)}')
    trainable_parameters = sum((int(parameter.numel()) for (_, parameter) in trainable))
    if trainable_parameters != EXPECTED_TRAINABLE_PARAMS:
        raise V6SftError(f'Expanded LoRA trainable-parameter count drifted: expected={EXPECTED_TRAINABLE_PARAMS}, actual={trainable_parameters}')
    return {'module_count': len(actual), 'trainable_tensor_count': len(trainable), 'trainable_parameters': trainable_parameters, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'only_lora_parameters_trainable': True}

def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch('checkpoint-([1-9][0-9]*)', path.name)
    if match is None:
        raise V6SftError('--resume-from-checkpoint must end in checkpoint-<positive-step>')
    return int(match.group(1))

def _required_checkpoint_files(world_size: int) -> tuple[str, ...]:
    return ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'training_args.bin', *(f'rng_state_{rank}.pth' for rank in range(world_size)))

def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise V6SftError(f'{context} is missing or is a symlink: {path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise V6SftError(f'{context} must be a JSON object: {path}')
    return value

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one small JSON control artifact without partial reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}.{threading.get_ident()}')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def validate_v6_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _require_lexical_path(path, CANONICAL_V6_MANIFEST, context='V6 --manifest')
    manifest = _load_json_object(manifest_path, context='V6 immutable manifest')

    def section(name: str) -> Mapping[str, Any]:
        value = manifest.get(name)
        if not isinstance(value, Mapping):
            raise V6SftError(f'V6 manifest section {name!r} is missing')
        return value
    identity = section('identity')
    model = section('model')
    initialization = section('initialization')
    data = section('data')
    training = section('training')
    validation = section('validation')
    checkpoint = section('checkpoint')
    loss = section('loss')
    implementation = section('implementation')
    artifacts = section('artifacts')
    lora = model.get('lora')
    ranking = loss.get('ranking')
    if not isinstance(lora, Mapping) or not isinstance(ranking, Mapping):
        raise V6SftError('V6 manifest LoRA/ranking sections are missing')
    expected: dict[str, tuple[Any, Any]] = {'schema_version': (manifest.get('schema_version'), 'mkd2world-v6-cur-full-sft-manifest-v1'), 'immutable': (manifest.get('immutable'), True), 'identity.run_id': (identity.get('run_id'), V6_FORMAL_RUN_ID), 'identity.display_name': (identity.get('display_name'), 'V6-Cur-FullSFT'), 'identity.v5_formal_run_config': (identity.get('v5_formal_run_config'), str(CANONICAL_V5_RUN_CONFIG)), 'identity.v5_formal_run_config_sha256': (identity.get('v5_formal_run_config_sha256'), EXPECTED_V5_RUN_CONFIG_SHA256), 'identity.gate_tier': (identity.get('gate_tier'), 2), 'identity.allowed_v5_differences': (identity.get('allowed_v5_differences'), list(V6_ALLOWED_V5_DIFFERENCES)), 'model.base_model': (model.get('base_model'), CANONICAL_BASE_MODEL), 'model.lora.target_modules': (lora.get('target_modules'), 'qwen35_expanded'), 'model.lora.target_modules_sha256': (lora.get('target_modules_sha256'), EXPECTED_TARGET_MODULES_SHA256), 'model.lora.r': (lora.get('r'), 32), 'model.lora.alpha': (lora.get('alpha'), 64), 'model.lora.dropout': (lora.get('dropout'), 0.05), 'model.lora.expected_modules': (lora.get('expected_modules'), EXPECTED_EXPANDED_MODULES), 'model.lora.expected_trainable_tensors': (lora.get('expected_trainable_tensors'), EXPECTED_ADAPTER_TENSORS), 'model.lora.expected_trainable_parameters': (lora.get('expected_trainable_parameters'), EXPECTED_TRAINABLE_PARAMS), 'initialization.mode': (initialization.get('mode'), V6_ADAPTER_INIT_MODE), 'initialization.init_adapter': (initialization.get('init_adapter'), str(CANONICAL_INIT_ADAPTER)), 'initialization.load_api': (initialization.get('load_api'), 'PeftModel.from_pretrained'), 'initialization.is_trainable': (initialization.get('is_trainable'), True), 'initialization.adapter_model_sha256': (initialization.get('adapter_model_sha256'), EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256), 'initialization.adapter_config_sha256': (initialization.get('adapter_config_sha256'), EXPECTED_INIT_ADAPTER_CONFIG_SHA256), 'initialization.checkpoint_integrity_sha256': (initialization.get('checkpoint_integrity_sha256'), EXPECTED_INIT_ADAPTER_INTEGRITY_SHA256), 'initialization.source_run_config_sha256': (initialization.get('source_run_config_sha256'), EXPECTED_INIT_SOURCE_RUN_CONFIG_SHA256), 'initialization.selection_manifest_sha256': (initialization.get('selection_manifest_sha256'), EXPECTED_INIT_SELECTION_SHA256), 'initialization.selection_pointer_sha256': (initialization.get('selection_pointer_sha256'), EXPECTED_INIT_SELECTION_POINTER_SHA256), 'initialization.source_checkpoint_step': (initialization.get('source_checkpoint_step'), 1200), 'initialization.reset_optimizer': (initialization.get('reset_optimizer'), True), 'initialization.reset_scheduler': (initialization.get('reset_scheduler'), True), 'initialization.reset_rng_from_seed': (initialization.get('reset_rng_from_seed'), True), 'initialization.reset_global_step': (initialization.get('reset_global_step'), True), 'initialization.load_source_optimizer_state': (initialization.get('load_source_optimizer_state'), False), 'initialization.load_source_scheduler_state': (initialization.get('load_source_scheduler_state'), False), 'initialization.load_source_rng_state': (initialization.get('load_source_rng_state'), False), 'data.train_jsonl': (data.get('train_jsonl'), str(CANONICAL_TRAIN_JSONL)), 'data.train_rows': (data.get('train_rows'), EXPECTED_TRAINING_ROWS), 'data.focus_sidecar': (data.get('focus_sidecar'), str(CANONICAL_FOCUS_SIDECAR)), 'data.prompt_key': (data.get('prompt_key'), 'localwm_next'), 'data.validation_jsonl': (data.get('validation_jsonl'), str(CANONICAL_VALIDATION_JSONL)), 'data.validation_rows': (data.get('validation_rows'), 500), 'data.transition_cache_root': (data.get('transition_cache_root'), str(CANONICAL_TRANSITION_CACHE_ROOT)), 'data.strict_no_skip': (data.get('strict_no_skip'), True), 'training.physical_gpus': (training.get('physical_gpus'), list(range(6))), 'training.cuda_visible_devices': (training.get('cuda_visible_devices'), V6_CUDA_VISIBLE_DEVICES), 'training.world_size': (training.get('world_size'), V6_WORLD_SIZE), 'training.per_device_train_batch_size': (training.get('per_device_train_batch_size'), V6_PER_DEVICE_TRAIN_BATCH_SIZE), 'training.gradient_accumulation_steps': (training.get('gradient_accumulation_steps'), V6_GRADIENT_ACCUMULATION_STEPS), 'training.global_batch_size': (training.get('global_batch_size'), V6_GLOBAL_BATCH_SIZE), 'training.epochs': (training.get('epochs'), 2), 'training.steps_per_epoch': (training.get('steps_per_epoch'), 521), 'training.final_step': (training.get('final_step'), EXPECTED_FINAL_STEP), 'training.learning_rate': (training.get('learning_rate'), 2e-05), 'training.lr_scheduler': (training.get('lr_scheduler'), 'cosine'), 'training.warmup_ratio': (training.get('warmup_ratio'), 0.03), 'training.weight_decay': (training.get('weight_decay'), 0.01), 'training.seed': (training.get('seed'), 0), 'training.data_seed': (training.get('data_seed'), 0), 'training.precision': (training.get('precision'), 'bf16'), 'training.gradient_checkpointing': (training.get('gradient_checkpointing'), True), 'training.max_length': (training.get('max_length'), 20000), 'training.max_output_tokens': (training.get('max_output_tokens'), 6000), 'training.image_min_pixels': (training.get('image_min_pixels'), 200704), 'training.image_max_pixels': (training.get('image_max_pixels'), 802816), 'training.logging_steps': (training.get('logging_steps'), 1), 'validation.physical_gpus': (validation.get('physical_gpus'), list(range(6))), 'validation.generation_workers_per_gpu': (validation.get('generation_workers_per_gpu'), V6_VALIDATION_WORKERS_PER_GPU), 'validation.total_generation_workers': (validation.get('total_generation_workers'), V6_VALIDATION_WORKERS), 'validation.max_samples': (validation.get('max_samples'), 500), 'validation.elef1_embedding_device': (validation.get('elef1_embedding_device'), 'cuda:3'), 'validation.cpu_fallback': (validation.get('cpu_fallback'), False), 'validation.strict_complete': (validation.get('strict_complete'), True), 'checkpoint.save_steps': (checkpoint.get('save_steps'), V6_SAVE_STEPS), 'checkpoint.steps': (checkpoint.get('steps'), list(v6_checkpoint_steps())), 'checkpoint.forbidden_steps': (checkpoint.get('forbidden_steps'), [1040]), 'checkpoint.blocking_inline_validation': (checkpoint.get('blocking_inline_validation'), True), 'loss.profile': (loss.get('profile'), 'v4_ee_plus_page_element_margin'), 'loss.ranking.margin': (ranking.get('margin'), V6_MARGIN), 'loss.ranking.beta_max': (ranking.get('beta_max'), V6_BETA_MAX), 'loss.ranking.beta_warmup_optimizer_steps': (ranking.get('beta_warmup_optimizer_steps'), V6_BETA_WARMUP_STEPS), 'loss.ranking.recompute': (ranking.get('recompute'), '3F+2B exact Current gate/replay'), 'implementation.trainer_module': (implementation.get('trainer_module'), 'mkd2world.training.next_tsm_sft'), 'implementation.certification_optimizer_steps': (implementation.get('certification_optimizer_steps'), V6_CERTIFY_EXPECTED_STEPS), 'implementation.certification_requires_save_finalize': (implementation.get('certification_requires_save_finalize'), True), 'implementation.certification_runs_validation500': (implementation.get('certification_runs_validation500'), False), 'artifacts.output_root': (artifacts.get('output_root'), str(CANONICAL_OUTPUT_ROOT)), 'artifacts.run_dir': (artifacts.get('run_dir'), str(CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID)), 'artifacts.certify_root': (artifacts.get('certify_root'), str(CANONICAL_CERTIFY_OUTPUT_ROOT)), 'artifacts.log_root': (artifacts.get('log_root'), str(CANONICAL_V6_LOG_ROOT)), 'artifacts.formal_receipt': (artifacts.get('formal_receipt'), str(CANONICAL_V6_LOG_ROOT / 'formal_receipt.json')), 'artifacts.run_state': (artifacts.get('run_state'), str(CANONICAL_V6_LOG_ROOT / 'run_state.json'))}
    mismatches = {name: {'actual': actual, 'expected': wanted} for (name, (actual, wanted)) in expected.items() if actual != wanted}
    if mismatches:
        raise V6SftError(f'V6 immutable manifest drifted: {mismatches}')
    return manifest

def write_v6_run_state(*, status: str, phase: str, global_step: int, max_steps: int, last_step_at: str | None=None, last_error: str | None=None, retryable: bool=False) -> dict[str, Any]:
    """Atomically publish watchdog state under the one canonical V6 log root."""
    state_path = CANONICAL_V6_LOG_ROOT / 'run_state.json'
    previous: Mapping[str, Any] = {}
    if state_path.is_file() and (not state_path.is_symlink()):
        try:
            previous = _load_json_object(state_path, context='V6 run state')
        except Exception:
            previous = {}
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    payload = {'schema_version': 'mkd2world-v6-run-state-v1', 'run_id': V6_FORMAL_RUN_ID, 'status': str(status), 'phase': str(phase), 'global_step': int(global_step), 'max_steps': int(max_steps), 'updated_at': now, 'last_step_at': last_step_at if last_step_at is not None else previous.get('last_step_at') or now, 'last_error': last_error, 'retryable': bool(retryable)}
    _atomic_write_json(state_path, payload)
    return payload

def _validate_run_config_for_resume(run_config: Mapping[str, Any], *, run_id: str, output_root: Path | None=None, certify_receipt: Path | None=None) -> None:
    if output_root is None:
        output_root = CANONICAL_OUTPUT_ROOT
    if certify_receipt is None:
        certify_receipt = CANONICAL_V6_LOG_ROOT / 'certify_receipt.json'
    model = run_config.get('model')
    distributed = run_config.get('distributed')
    loss = run_config.get('loss')
    labeler = run_config.get('focus_labeler')
    data = run_config.get('data')
    save_interval = run_config.get('save_interval')
    completion = run_config.get('completion_contract')
    arguments = run_config.get('args')
    if not all((isinstance(value, Mapping) for value in (model, distributed, loss, labeler, data, save_interval, completion, arguments))):
        raise V6SftError('resume run_config lacks V6 provenance sections')
    if run_config.get('schema_version') != V6_RUN_CONFIG_SCHEMA or run_config.get('run_id') != run_id or run_config.get('certify_run') is not False or (run_config.get('v5_parity', {}).get('status') != 'passed') or (run_config.get('v5_parity', {}).get('loss_equal') is not True) or (model.get('base_model') != CANONICAL_BASE_MODEL) or (model.get('init_adapter') != str(CANONICAL_INIT_ADAPTER)) or (model.get('adapter_initialization') != v6_adapter_initialization_contract()) or (loss != v6_loss_contract()) or (distributed.get('world_size') != V6_WORLD_SIZE) or (distributed.get('cuda_visible_devices') != V6_CUDA_VISIBLE_DEVICES) or (distributed.get('global_batch_size') != V6_GLOBAL_BATCH_SIZE) or (labeler.get('schema_version') != focus_contract.AUDIT_LABEL_SCHEMA_VERSION) or (data.get('train_jsonl') != str(CANONICAL_TRAIN_JSONL)) or (data.get('focus_sidecar') != str(CANONICAL_FOCUS_SIDECAR)) or (save_interval.get('steps') != V6_SAVE_STEPS) or (save_interval.get('examples') != V6_SAVE_INTERVAL_EXAMPLES) or (save_interval.get('checkpoint_steps', list(v6_checkpoint_steps())) != list(v6_checkpoint_steps())) or (save_interval.get('validation_steps', list(V6_VALIDATION_STEPS)) != list(V6_VALIDATION_STEPS)) or (completion != v6_formal_completion_contract()):
        raise V6SftError('resume run_config is not the same V3-Cur-initialized V6 contract')
    sidecar_path = Path(str(data.get('focus_sidecar')))
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise V6SftError('resume run_config sidecar SHA256 does not match the V6 run')
    canonical_modules = _validate_target_modules(model.get('target_modules'), context='resume run_config model.target_modules')
    expected_loaded_graph = {'module_count': EXPECTED_EXPANDED_MODULES, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'only_lora_parameters_trainable': True}
    if model.get('loaded_adapter_graph') != expected_loaded_graph:
        raise V6SftError('resume run_config loaded adapter graph drifted')
    expected_arguments = {'base_model': CANONICAL_BASE_MODEL, 'train_jsonl': str(CANONICAL_TRAIN_JSONL), 'focus_sidecar': str(CANONICAL_FOCUS_SIDECAR), 'output_dir': str(output_root), 'run_id': V6_FORMAL_RUN_ID, 'prompt_key': 'localwm_next', 'init_adapter': str(CANONICAL_INIT_ADAPTER), 'resume_from_checkpoint': None, 'dry_run': False, 'per_device_train_batch_size': V6_PER_DEVICE_TRAIN_BATCH_SIZE, 'gradient_accumulation_steps': V6_GRADIENT_ACCUMULATION_STEPS, 'num_train_epochs': 2.0, 'learning_rate': 2e-05, 'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.03, 'weight_decay': 0.01, 'max_grad_norm': 1.0, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-08, 'seed': 0, 'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'target_modules': 'qwen35_expanded', 'max_length': 20000, 'max_output_tokens': 6000, 'image_min_pixels': 200704, 'image_max_pixels': 802816, 'min_supervised_tokens': 16, 'assistant_loss_scope': 'full', 'dataset_loading_mode': 'direct', 'skip_missing_images': False, 'skip_overlong_samples': False, 'device_mode': 'ddp', 'gradient_checkpointing': True, 'bf16': True, 'fp16': False, 'save_steps': V6_SAVE_STEPS, 'logging_steps': 1, 'wm_eval_max_samples': 500, 'certify_run': False, 'certify_sample_plan': None, 'certify_receipt': str(certify_receipt), 'resume_validation_only': False}
    expected_arguments.update({'loss_profile': V6_LOSS_PROFILE, 'base_loss_granularity': V6_BASE_LOSS_GRANULARITY, 'focus_loss_granularity': V6_FOCUS_LOSS_GRANULARITY, 'wm_eval_jsonl': str(CANONICAL_VALIDATION_JSONL), 'wm_eval_workers_per_gpu': 3, 'wm_eval_embedding_model': str(CANONICAL_EMBEDDING_MODEL), 'wm_eval_embedding_device': 'cuda:3', 'wm_eval_stop_after_bad_checkpoints': 0, 'transition_cache_root': str(CANONICAL_TRANSITION_CACHE_ROOT), 'transition_output_root': str(CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID / 'transition_metrics')})
    argument_mismatches = {name: {'expected': expected, 'actual': arguments.get(name)} for (name, expected) in expected_arguments.items() if arguments.get(name) != expected}
    if argument_mismatches:
        raise V6SftError(f'resume run_config frozen arguments drifted: {argument_mismatches}')

def write_v6_checkpoint_binding(checkpoint_dir: Path, *, run_dir: Path, run_id: str, global_step: int) -> Path:
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    if checkpoint_dir.parent != run_dir / 'checkpoints':
        raise V6SftError('checkpoint binding target is outside this V6 run')
    if _checkpoint_step(checkpoint_dir) != global_step:
        raise V6SftError('checkpoint binding step/path mismatch')
    run_config_path = run_dir / 'run_config.json'
    integrity_path = checkpoint_dir / 'checkpoint_integrity.json'
    run_config = _load_json_object(run_config_path, context='V6 run_config')
    certify_run = run_config.get('certify_run') is True
    if certify_run:
        model = run_config.get('model')
        distributed = run_config.get('distributed')
        data = run_config.get('data')
        arguments = run_config.get('args')
        certify_plan = run_config.get('certify_sample_plan')
        expected_completion = {'canonical_training_rows': V6_CERTIFY_ROWS, 'epochs': 1.0, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'expected_final_step': V6_CERTIFY_EXPECTED_STEPS, 'expected_trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'formal_resume_allowed': False}
        if run_id != V6_CERTIFY_RUN_ID or run_config.get('schema_version') != V6_RUN_CONFIG_SCHEMA or (not isinstance(model, Mapping)) or (not isinstance(distributed, Mapping)) or (not isinstance(data, Mapping)) or (not isinstance(arguments, Mapping)) or (not isinstance(certify_plan, Mapping)) or (model.get('base_model') != CANONICAL_BASE_MODEL) or (model.get('init_adapter') != str(CANONICAL_INIT_ADAPTER)) or (model.get('adapter_initialization') != v6_adapter_initialization_contract()) or (run_config.get('loss') != v6_loss_contract()) or (distributed.get('world_size') != V6_WORLD_SIZE) or (distributed.get('cuda_visible_devices') != V6_CUDA_VISIBLE_DEVICES) or (distributed.get('global_batch_size') != V6_GLOBAL_BATCH_SIZE) or (data.get('train_jsonl') != str(CANONICAL_TRAIN_JSONL)) or (data.get('focus_sidecar') != str(CANONICAL_FOCUS_SIDECAR)) or (run_config.get('completion_contract') != expected_completion) or (arguments.get('run_id') != V6_CERTIFY_RUN_ID) or (arguments.get('output_dir') != str(CANONICAL_CERTIFY_OUTPUT_ROOT)) or (arguments.get('certify_run') is not True) or (arguments.get('num_train_epochs') != 1.0) or (arguments.get('per_device_train_batch_size') != V6_PER_DEVICE_TRAIN_BATCH_SIZE) or (arguments.get('gradient_accumulation_steps') != V6_GRADIENT_ACCUMULATION_STEPS) or (arguments.get('certify_sample_plan') != certify_plan.get('path')):
            raise V6SftError('checkpoint binding certify run_config contract drifted')
        certify_plan_path = Path(str(certify_plan.get('path') or ''))
        sidecar_path = Path(str(data.get('focus_sidecar') or ''))
        if certify_plan_path.is_symlink() or not certify_plan_path.is_file() or sidecar_path.is_symlink() or (not sidecar_path.is_file()):
            raise V6SftError('checkpoint binding certify data authority drifted')
    else:
        _validate_run_config_for_resume(run_config, run_id=run_id)
    integrity = _load_json_object(integrity_path, context='generic checkpoint integrity manifest')
    if integrity.get('status') != 'passed' or integrity.get('global_step') != global_step or integrity.get('world_size') != V6_WORLD_SIZE or (integrity.get('adapter_tensor_count') != EXPECTED_ADAPTER_TENSORS):
        raise V6SftError('generic checkpoint integrity is incompatible with V6')
    target_modules = _validate_target_modules(run_config.get('model', {}).get('target_modules'), context='checkpoint binding target_modules')
    adapter_contract = validate_expanded_adapter_files(checkpoint_dir, expected_modules=target_modules)
    binding = {'schema_version': V6_CHECKPOINT_BINDING_SCHEMA, 'status': 'passed', 'run_id': run_id, 'global_step': global_step, 'run_config_sha256': _sha256_file(run_config_path), 'checkpoint_integrity_sha256': _sha256_file(integrity_path), 'matcher_schema': focus_contract.AUDIT_LABEL_SCHEMA_VERSION, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'sidecar_sha256': run_config.get('data', {}).get('focus_sidecar_sha256'), 'adapter_initialization_mode': V6_ADAPTER_INIT_MODE, 'adapter_config_sha256': adapter_contract['config_sha256'], 'adapter_weights_sha256': adapter_contract['weights_sha256'], 'adapter_tensor_count': adapter_contract['tensor_count'], 'adapter_tensor_parameters': adapter_contract['tensor_parameters'], 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'world_size': V6_WORLD_SIZE, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'loss_ramp': False, 'resume_scope': 'certify_checkpoint_forbidden_for_formal_resume' if certify_run else 'same_run_v6_checkpoint_only', 'certify_run': certify_run, 'expected_final_step': V6_CERTIFY_EXPECTED_STEPS if certify_run else EXPECTED_FINAL_STEP}
    path = checkpoint_dir / 'v6_checkpoint_binding.json'
    if path.exists() or path.is_symlink():
        raise V6SftError(f'refusing to overwrite checkpoint binding: {path}')
    temporary = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)
    return path

def validate_v6_resume_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Accept only a cryptographically bound checkpoint from the same V6 run."""
    if args.resume_from_checkpoint is None:
        return {'mode': 'new_run', 'policy': 'same_run_v6_checkpoint_only', 'resume_from_checkpoint': None}
    if not args.run_id or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id) is None:
        raise V6SftError('V6 resume requires an explicit safe --run-id')
    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / args.run_id
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise V6SftError(f'V6 resume run directory is missing: {run_dir}')
    run_dir = run_dir.resolve(strict=True)
    checkpoint_input = Path(args.resume_from_checkpoint)
    if checkpoint_input.is_symlink() or not checkpoint_input.is_dir():
        raise V6SftError(f'V6 resume checkpoint is missing or is a symlink: {checkpoint_input}')
    checkpoint = checkpoint_input.resolve(strict=True)
    if checkpoint.parent != run_dir / 'checkpoints':
        raise V6SftError('--resume-from-checkpoint must belong to --output-dir/--run-id/checkpoints')
    step = _checkpoint_step(checkpoint)
    run_config_path = run_dir / 'run_config.json'
    run_config = _load_json_object(run_config_path, context='V6 run_config')
    _validate_run_config_for_resume(run_config, run_id=args.run_id)
    integrity_path = checkpoint / 'checkpoint_integrity.json'
    integrity = _load_json_object(integrity_path, context='generic checkpoint integrity manifest')
    if integrity.get('status') != 'passed' or integrity.get('global_step') != step or integrity.get('world_size') != V6_WORLD_SIZE or (integrity.get('adapter_tensor_count') != EXPECTED_ADAPTER_TENSORS):
        raise V6SftError('resume checkpoint generic integrity contract failed')
    required = set(_required_checkpoint_files(V6_WORLD_SIZE))
    recorded = integrity.get('required_files')
    if not isinstance(recorded, Mapping) or not required.issubset(recorded):
        raise V6SftError('resume checkpoint integrity is missing required files')
    for name in sorted(required):
        path = checkpoint / name
        metadata = recorded.get(name)
        if not isinstance(metadata, Mapping) or path.is_symlink() or (not path.is_file()) or (metadata.get('size_bytes') != path.stat().st_size):
            raise V6SftError(f'resume checkpoint file integrity failed: {name}')
    target_modules = _validate_target_modules(run_config.get('model', {}).get('target_modules'), context='resume checkpoint target_modules')
    adapter_contract = validate_expanded_adapter_files(checkpoint, expected_modules=target_modules)
    trainer_state = _load_json_object(checkpoint / 'trainer_state.json', context='resume trainer_state')
    if int(trainer_state.get('global_step', -1)) != step:
        raise V6SftError('resume trainer_state global_step does not match checkpoint')
    binding_path = checkpoint / 'v6_checkpoint_binding.json'
    binding = _load_json_object(binding_path, context='V6 checkpoint binding')
    expected_binding = {'schema_version': V6_CHECKPOINT_BINDING_SCHEMA, 'status': 'passed', 'run_id': args.run_id, 'global_step': step, 'run_config_sha256': _sha256_file(run_config_path), 'checkpoint_integrity_sha256': _sha256_file(integrity_path), 'matcher_schema': focus_contract.AUDIT_LABEL_SCHEMA_VERSION, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'sidecar_sha256': run_config.get('data', {}).get('focus_sidecar_sha256'), 'adapter_initialization_mode': V6_ADAPTER_INIT_MODE, 'adapter_config_sha256': adapter_contract['config_sha256'], 'adapter_weights_sha256': adapter_contract['weights_sha256'], 'adapter_tensor_count': adapter_contract['tensor_count'], 'adapter_tensor_parameters': adapter_contract['tensor_parameters'], 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'world_size': V6_WORLD_SIZE, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'loss_ramp': False, 'resume_scope': 'same_run_v6_checkpoint_only', 'certify_run': False, 'expected_final_step': EXPECTED_FINAL_STEP}
    if binding != expected_binding:
        raise V6SftError('resume checkpoint is not bound to this exact V6 run')
    return {'mode': 'resume_same_run', 'policy': 'same_run_v6_checkpoint_only', 'resume_from_checkpoint': str(checkpoint), 'run_id': args.run_id, 'global_step': step, 'run_config_sha256': expected_binding['run_config_sha256'], 'checkpoint_integrity_sha256': expected_binding['checkpoint_integrity_sha256'], 'checkpoint_binding_sha256': _sha256_file(binding_path)}

def _iter_jsonl_objects(path: Path):
    with path.open(encoding='utf-8') as handle:
        for (line_number, line) in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V6SftError(f'invalid JSON at {path}:{line_number}') from exc
            if not isinstance(row, dict):
                raise V6SftError(f'non-object JSON row at {path}:{line_number}')
            yield (line_number, row)

def _normalize_v6_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize model inputs while retaining sidecar contract metadata."""
    from mkd2world.training.sft import normalize_row_for_dataset
    normalized = normalize_row_for_dataset(dict(row))
    normalized['transition_class'] = str(row.get('transition_class') or '')
    normalized['action_bucket'] = str(row.get('action_bucket') or '')
    return normalized

def validate_v6_data_contract(train_jsonl: str | Path, focus_sidecar: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    train_path = Path(train_jsonl)
    sidecar_path = Path(focus_sidecar)
    if not train_path.is_file():
        raise V6SftError(f'missing canonical train JSONL: {train_path}')
    if not sidecar_path.is_file():
        raise V6SftError(f'full V6 focus sidecar is not available; phase-one audit approval is required before creating {sidecar_path}')
    manifest_path = sidecar_path.parent / 'manifest.json'
    if not manifest_path.is_file():
        raise V6SftError(f'missing focus sidecar manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != focus_contract.TRAINING_SIDECAR_MANIFEST_VERSION:
        raise V6SftError('unsupported focus sidecar manifest schema')
    labeler_path = Path(focus_contract.focus_labeler.__file__).resolve()
    labeler_sha256 = _sha256_file(labeler_path)
    dependency_path = Path(focus_contract.focus_dependency.__file__).resolve()
    dependency_sha256 = _sha256_file(dependency_path)
    builder_path = Path(__file__).resolve().parents[3] / 'scripts' / 'training' / 'build_v4_focus_sidecar.py'
    if not builder_path.is_file():
        raise V6SftError(f'missing canonical focus sidecar builder: {builder_path}')
    sidecars = load_focus_sidecar(sidecar_path)
    dataset_ids: list[str] = []
    seen_ids: set[str] = set()
    transition_counts: dict[str, int] = {}
    for (line_number, row) in _iter_jsonl_objects(train_path):
        sample_id = str(row.get('id') or '').strip()
        if not sample_id:
            raise V6SftError(f'training row {line_number} is missing id')
        if sample_id in seen_ids:
            raise V6SftError(f'duplicate training id {sample_id!r}')
        seen_ids.add(sample_id)
        dataset_ids.append(sample_id)
        transition = str(row.get('transition_class') or 'unknown')
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        image_path = Path(str(row.get('current_image_path') or ''))
        if not image_path.is_file():
            raise V6SftError(f'sample {sample_id} current screenshot does not exist: {image_path}')
    if train_path.resolve() == CANONICAL_TRAIN_JSONL.resolve():
        raise V6SftError('canonical train row/SHA/ordered-ID identity drifted')
    sidecar_ids = set(sidecars)
    if seen_ids != sidecar_ids:
        missing = sorted(seen_ids - sidecar_ids)
        extra = sorted(sidecar_ids - seen_ids)
        raise V6SftError(f'dataset/sidecar ID coverage mismatch: missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}')
    focus_kind_counts: dict[str, int] = {}
    focus_field_counts: dict[str, int] = {}
    total_base_units = 0
    total_focus_units = 0
    for (_, row) in _iter_jsonl_objects(train_path):
        normalized = _normalize_v6_training_row(row)
        annotation = sidecars[str(row['id'])]
        validated = validate_sidecar(normalized, annotation)
        total_base_units += len(validated.base_spans)
        total_focus_units += len(validated.focus_spans)
        for unit in annotation['focus_units']:
            kind = str(unit['kind'])
            focus_kind_counts[kind] = focus_kind_counts.get(kind, 0) + 1
            for field in unit['fields']:
                focus_field_counts[field] = focus_field_counts.get(field, 0) + 1
    ordered_ids_sha256 = _sha256_text('\n'.join(dataset_ids) + '\n')
    train_sha256 = _sha256_file(train_path)
    sidecar_sha256 = _sha256_file(sidecar_path)
    expected_manifest = {'source_sha256': train_sha256, 'source_rows': len(dataset_ids), 'sidecar_sha256': sidecar_sha256, 'sidecar_rows': len(dataset_ids), 'ordered_ids_sha256': ordered_ids_sha256, 'stores_full_target': False, 'labeler_sha256': labeler_sha256, 'labeler_parent_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'adapter_sha256': _sha256_file(Path(focus_contract.__file__).resolve()), 'labeler_dependency_sha256': dependency_sha256, 'builder_sha256': _sha256_file(builder_path), 'label_schema_version': focus_contract.AUDIT_LABEL_SCHEMA_VERSION, 'training_schema_version': focus_contract.TRAINING_SIDECAR_VERSION, 'labeler_version': focus_contract.FINAL_MATCHER_VERSION, 'labeler_path': str(labeler_path), 'freeze_path': str(focus_contract.FINAL_FREEZE_PATH), 'freeze_manifest_sha256': focus_contract.FINAL_FREEZE_SHA256SUMS_SHA256, 'freeze_metadata_sha256': focus_contract.FINAL_FREEZE_METADATA_SHA256, 'dependency_path': str(dependency_path), 'adapter_path': str(Path(focus_contract.__file__).resolve()), 'builder_path': str(builder_path), 'trainer_path': str(Path(__file__).resolve()), 'runner_path': str(Path(__file__).resolve().parents[3] / 'scripts/training/run_v6_full_sft.sh'), 'trainer_sha256': _sha256_file(Path(__file__).resolve()), 'runner_sha256': _sha256_file(Path(__file__).resolve().parents[3] / 'scripts/training/run_v6_full_sft.sh'), 'complete_base_supervision': True, 'focus_spans_subset_of_base': True, 'full_next_markdown_unchanged': True, 'transition_counts': dict(sorted(transition_counts.items())), 'focus_kind_counts': dict(sorted(focus_kind_counts.items())), 'focus_field_counts': dict(sorted(focus_field_counts.items()))}
    actual_manifest = {'source_sha256': manifest.get('source', {}).get('sha256'), 'source_rows': manifest.get('source', {}).get('rows'), 'sidecar_sha256': manifest.get('sidecar', {}).get('sha256'), 'sidecar_rows': manifest.get('sidecar', {}).get('rows'), 'ordered_ids_sha256': manifest.get('sidecar', {}).get('ordered_ids_sha256'), 'stores_full_target': manifest.get('sidecar', {}).get('stores_full_target'), 'labeler_sha256': manifest.get('labeler', {}).get('sha256'), 'labeler_parent_sha256': manifest.get('labeler', {}).get('parent_frozen_sha256'), 'adapter_sha256': manifest.get('adapter', {}).get('sha256'), 'labeler_dependency_sha256': manifest.get('labeler', {}).get('dependency_sha256'), 'builder_sha256': manifest.get('builder', {}).get('sha256'), 'label_schema_version': manifest.get('labeler', {}).get('schema_version'), 'training_schema_version': manifest.get('adapter', {}).get('schema_version'), 'labeler_version': manifest.get('labeler', {}).get('version'), 'labeler_path': manifest.get('labeler', {}).get('path'), 'freeze_path': manifest.get('labeler', {}).get('freeze_path'), 'freeze_manifest_sha256': manifest.get('labeler', {}).get('freeze_manifest_sha256'), 'freeze_metadata_sha256': manifest.get('labeler', {}).get('freeze_metadata_sha256'), 'dependency_path': manifest.get('labeler', {}).get('dependency_path'), 'adapter_path': manifest.get('adapter', {}).get('path'), 'builder_path': manifest.get('builder', {}).get('path'), 'trainer_path': manifest.get('runtime', {}).get('trainer_path'), 'runner_path': manifest.get('runtime', {}).get('runner_path'), 'trainer_sha256': manifest.get('runtime', {}).get('trainer_sha256'), 'runner_sha256': manifest.get('runtime', {}).get('runner_sha256'), 'complete_base_supervision': manifest.get('invariants', {}).get('complete_base_supervision'), 'focus_spans_subset_of_base': manifest.get('invariants', {}).get('focus_spans_subset_of_base'), 'full_next_markdown_unchanged': manifest.get('invariants', {}).get('full_next_markdown_unchanged'), 'transition_counts': manifest.get('counts', {}).get('transition_classes'), 'focus_kind_counts': manifest.get('counts', {}).get('focus_kinds'), 'focus_field_counts': manifest.get('counts', {}).get('focus_fields')}
    comparison_expected = dict(expected_manifest)
    comparison_actual = dict(actual_manifest)
    if sidecar_path.resolve() == CANONICAL_FOCUS_SIDECAR.resolve():
        manifest_sha256 = _sha256_file(manifest_path)
        for provenance_key in ('labeler_path', 'dependency_path', 'adapter_path', 'builder_path', 'trainer_path', 'runner_path', 'trainer_sha256', 'runner_sha256'):
            comparison_expected.pop(provenance_key, None)
            comparison_actual.pop(provenance_key, None)
    if comparison_actual != comparison_expected:
        raise V6SftError(f'focus sidecar manifest mismatch: expected={comparison_expected}, actual={comparison_actual}')
    report = {'status': 'passed', 'train_jsonl': str(train_path), 'train_sha256': train_sha256, 'focus_sidecar': str(sidecar_path), 'focus_sidecar_sha256': sidecar_sha256, 'focus_manifest': str(manifest_path), 'focus_manifest_sha256': _sha256_file(manifest_path), 'rows': len(dataset_ids), 'unique_ids': len(seen_ids), 'ordered_ids_sha256': ordered_ids_sha256, 'transition_counts': transition_counts, 'total_base_units': total_base_units, 'total_focus_units': total_focus_units, 'focus_kind_counts': focus_kind_counts, 'focus_field_counts': focus_field_counts, 'full_target_source': 'canonical train JSONL; sidecar stores hashes/spans only', 'complete_base_supervision': True, 'focus_spans_subset_of_base': True, 'labeler_version': focus_contract.FINAL_MATCHER_VERSION, 'labeler_schema': focus_contract.FINAL_MATCHER_SCHEMA, 'labeler_sha256': focus_contract.FINAL_MATCHER_SHA256}
    return (sidecars, report)

def validate_v6_token_contract(train_jsonl: str | Path, sidecars: Mapping[str, Mapping[str, Any]], tokenizer: Any, *, max_output_tokens: int) -> dict[str, Any]:
    """Map every full target with the real tokenizer before any GPU launch."""
    total_tokens = 0
    current_total_target_tokens = 0
    maximum_tokens = 0
    maximum_sample_id = ''
    current_max_target_tokens = 0
    max_current_target_sample_id = ''
    token_equal_changed = 0
    token_equal_nonchanged = 0
    text_different_token_equal = 0
    rows = 0
    for (_, row) in _iter_jsonl_objects(Path(train_jsonl)):
        sample_id = str(row['id'])
        validated = validate_sidecar(_normalize_v6_training_row(row), sidecars[sample_id])
        (token_ids, _, _) = map_target_tokens_to_units(tokenizer, validated)
        current_target = canonical_v6_current_target(row)
        current_validated = build_v6_current_sidecar(row)
        if current_validated.target_text != current_target:
            raise V6SftError(f'sample {sample_id} Current parser target differs from current_md')
        (current_token_ids, _, _) = map_target_tokens_to_units(tokenizer, current_validated)
        token_count = len(token_ids)
        current_token_count = len(current_token_ids)
        if token_count > max_output_tokens:
            raise V6SftError(f'sample {sample_id} has {token_count} target tokens, exceeding max_output_tokens={max_output_tokens}')
        if current_token_count > max_output_tokens:
            raise V6SftError(f'sample {sample_id} has {current_token_count} Current target tokens, exceeding max_output_tokens={max_output_tokens}')
        token_equal = token_ids == current_token_ids
        text_equal = validated.target_text == current_validated.target_text
        transition = str(row.get('transition_class') or '')
        if token_equal and transition == 'changed':
            token_equal_changed += 1
        elif token_equal:
            token_equal_nonchanged += 1
        if token_equal and (not text_equal):
            text_different_token_equal += 1
        rows += 1
        total_tokens += token_count
        current_total_target_tokens += current_token_count
        if token_count > maximum_tokens:
            maximum_tokens = token_count
            maximum_sample_id = sample_id
        if current_token_count > current_max_target_tokens:
            current_max_target_tokens = current_token_count
            max_current_target_sample_id = sample_id
    expected = (EXPECTED_TRAINING_ROWS, EXPECTED_TOTAL_TARGET_TOKENS, EXPECTED_MAX_TARGET_TOKENS, EXPECTED_MAX_TARGET_SAMPLE_ID, EXPECTED_TOTAL_CURRENT_TARGET_TOKENS, EXPECTED_MAX_CURRENT_TARGET_TOKENS, EXPECTED_MAX_CURRENT_TARGET_SAMPLE_ID, 2, EXPECTED_NONCHANGED_TOKEN_EQUAL_ROWS, 0)
    actual = (rows, total_tokens, maximum_tokens, maximum_sample_id, current_total_target_tokens, current_max_target_tokens, max_current_target_sample_id, token_equal_changed, token_equal_nonchanged, text_different_token_equal)
    if actual != expected:
        raise V6SftError(f'full Next/Current tokenizer contract drifted: expected={expected}, actual={actual}')
    return {'status': 'passed', 'rows': rows, 'total_target_tokens': total_tokens, 'max_target_tokens': maximum_tokens, 'max_target_sample_id': maximum_sample_id, 'current_total_target_tokens': current_total_target_tokens, 'current_max_target_tokens': current_max_target_tokens, 'max_current_target_sample_id': max_current_target_sample_id, 'changed_token_equal_rows': token_equal_changed, 'nonchanged_token_equal_rows': token_equal_nonchanged, 'text_different_token_equal_rows': text_different_token_equal, 'rank_eligible_rows': 25005, 'max_output_tokens': max_output_tokens}

def _select_v6_collator_preflight_rows(train_jsonl: str | Path, sidecars: Mapping[str, Mapping[str, Any]], *, max_target_sample_id: str) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Select a deterministic diagnostic set spanning the sidecar contract."""
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()
    required = {'transition:changed', 'transition:near_noop', 'transition:exact_noop', 'kind:ADD', 'kind:UPDATE', 'changed:base_only', 'update:multi_field', 'bucket:click', 'bucket:text_key', 'bucket:scroll', 'bucket:navigation', 'bucket:other'}
    max_target_row: dict[str, Any] | None = None
    max_focus_row: dict[str, Any] | None = None
    max_focus_units = -1
    for (_, raw_row) in _iter_jsonl_objects(Path(train_jsonl)):
        sample_id = str(raw_row['id'])
        sidecar = sidecars[sample_id]
        transition = str(raw_row.get('transition_class') or '')
        bucket = str(raw_row.get('action_bucket') or '')
        units = sidecar['focus_units']
        row_criteria = {f'transition:{transition}', f'bucket:{bucket}'}
        row_criteria.update((f"kind:{unit['kind']}" for unit in units))
        if transition == 'changed' and (not units):
            row_criteria.add('changed:base_only')
        if any((unit['kind'] == 'UPDATE' and len(unit['fields']) > 1 for unit in units)):
            row_criteria.add('update:multi_field')
        new_criteria = (row_criteria & required) - covered
        if new_criteria:
            if sample_id not in selected_ids:
                selected.append(_normalize_v6_training_row(raw_row))
                selected_ids.add(sample_id)
            covered.update(new_criteria)
        if sample_id == max_target_sample_id:
            max_target_row = _normalize_v6_training_row(raw_row)
        if len(units) > max_focus_units:
            max_focus_units = len(units)
            max_focus_row = _normalize_v6_training_row(raw_row)
    missing = required - covered
    if missing:
        raise V6SftError(f'collator preflight selection is missing contract cases: {sorted(missing)}')
    if max_target_row is None:
        raise V6SftError(f'max-target sample {max_target_sample_id!r} is absent from training data')
    if max_focus_row is None:
        raise V6SftError('collator preflight could not select a max-focus sample')
    for diagnostic_row in (max_target_row, max_focus_row):
        sample_id = str(diagnostic_row['id'])
        if sample_id not in selected_ids:
            selected.append(diagnostic_row)
            selected_ids.add(sample_id)
    diagnostics = {'max_target_sample_id': max_target_sample_id, 'max_focus_sample_id': str(max_focus_row['id']), 'max_focus_units': max_focus_units}
    return (selected, covered, diagnostics)

def build_v6_certify_sample_plan(train_jsonl: str | Path, sidecars: Mapping[str, Mapping[str, Any]], *, max_target_sample_id: str, max_current_target_sample_id: str, _authority: Mapping[str, Any] | None=None) -> dict[str, Any]:
    rows_by_bucket: dict[str, list[str]] = {bucket: [] for bucket in V6_ACTION_BUCKETS}
    sample_bucket: dict[str, str] = {}
    max_focus_units = -1
    max_focus_sample_id = ''
    contract_witnesses: dict[str, str] = {}
    seen: set[str] = set()
    source_ids: list[str] = []
    for (_, row) in _iter_jsonl_objects(Path(train_jsonl)):
        sample_id = str(row.get('id') or '')
        bucket = str(row.get('action_bucket') or '')
        if not sample_id or sample_id in seen:
            raise V6SftError('certify-plan source contains missing/duplicate sample IDs')
        if bucket not in rows_by_bucket:
            raise V6SftError(f'certify-plan source has unsupported bucket {bucket!r}')
        if sample_id not in sidecars:
            raise V6SftError(f'certify-plan sidecar is missing {sample_id}')
        seen.add(sample_id)
        source_ids.append(sample_id)
        sample_bucket[sample_id] = bucket
        rows_by_bucket[bucket].append(sample_id)
        focus_units = len(sidecars[sample_id]['focus_units'])
        transition = str(row.get('transition_class') or '')
        units = sidecars[sample_id]['focus_units']
        criteria = {f'transition:{transition}'}
        criteria.update((f"kind:{unit['kind']}" for unit in units))
        if transition == 'changed' and (not units):
            criteria.add('changed:base_only')
        if any((unit['kind'] == 'UPDATE' and len(unit['fields']) > 1 for unit in units)):
            criteria.add('update:multi_field')
        for criterion in V6_CERTIFY_CONTRACT_CASES:
            if criterion in criteria and criterion not in contract_witnesses:
                contract_witnesses[criterion] = sample_id
        if focus_units > max_focus_units:
            max_focus_units = focus_units
            max_focus_sample_id = sample_id
    if max_target_sample_id not in seen:
        raise V6SftError('certify-plan max-target sample is absent')
    if max_current_target_sample_id != EXPECTED_MAX_CURRENT_TARGET_SAMPLE_ID:
        raise V6SftError('certify-plan Current max-target witness differs from the full token audit')
    for required_current_id in (max_current_target_sample_id, EXPECTED_MAX_CURRENT_BASE_UNIT_SAMPLE_ID):
        if required_current_id not in seen:
            raise V6SftError(f'certify-plan required Current stress sample is absent: {required_current_id}')
    if not max_focus_sample_id:
        raise V6SftError('certify-plan max-Focus sample is absent')
    missing_contract_cases = set(V6_CERTIFY_CONTRACT_CASES) - set(contract_witnesses)
    if missing_contract_cases:
        raise V6SftError(f'certify plan cannot cover canonical contract cases: {sorted(missing_contract_cases)}')
    per_bucket = V6_CERTIFY_ROWS // len(V6_ACTION_BUCKETS)
    if per_bucket * len(V6_ACTION_BUCKETS) != V6_CERTIFY_ROWS:
        raise V6SftError('certify row count is not divisible by action buckets')
    mandatory = {max_target_sample_id, max_focus_sample_id, max_current_target_sample_id, EXPECTED_MAX_CURRENT_BASE_UNIT_SAMPLE_ID, *contract_witnesses.values()}
    selected: list[str] = []
    bucket_counts: dict[str, int] = {}
    for bucket in V6_ACTION_BUCKETS:
        candidates = rows_by_bucket[bucket]
        forced = [sample_id for sample_id in candidates if sample_id in mandatory]
        chosen = forced + [sample_id for sample_id in candidates if sample_id not in mandatory][:per_bucket - len(forced)]
        if len(chosen) != per_bucket:
            raise V6SftError(f'certify plan cannot select {per_bucket} rows for {bucket}')
        selected.extend(chosen)
        bucket_counts[bucket] = len(chosen)
    if len(selected) != V6_CERTIFY_ROWS or len(set(selected)) != V6_CERTIFY_ROWS:
        raise V6SftError(f'certify plan must contain {V6_CERTIFY_ROWS} unique sample IDs')
    if not mandatory.issubset(selected):
        raise V6SftError('certify plan lost a max-target/max-Focus sample')
    authority = dict(_authority or {})
    train_path = Path(train_jsonl)
    sidecar_path = Path(authority.get('focus_sidecar', CANONICAL_FOCUS_SIDECAR))
    manifest_path = Path(authority.get('focus_manifest', sidecar_path.parent / 'manifest.json'))
    trainer_path = Path(authority.get('trainer_path', Path(__file__).resolve()))
    runner_path = Path(authority.get('runner_path', Path(__file__).resolve().parents[3] / v6_runner_relative_path()))
    train_record_path = Path(authority.get('train_record_path', train_path))
    sidecar_record_path = Path(authority.get('focus_sidecar_record_path', sidecar_path))
    manifest_record_path = Path(authority.get('focus_manifest_record_path', manifest_path))
    ordered_ids_sha256 = _sha256_text('\n'.join(source_ids) + '\n')
    return {'schema_version': V6_CERTIFY_PLAN_SCHEMA, 'status': 'passed', 'rows': V6_CERTIFY_ROWS, 'selected_sample_ids': selected, 'selected_sample_ids_sha256': _sha256_text('\n'.join(selected) + '\n'), 'bucket_counts': bucket_counts, 'required_coverage': {'max_target_sample_id': max_target_sample_id, 'max_focus_sample_id': max_focus_sample_id, 'max_focus_units': max_focus_units, 'max_current_target_sample_id': max_current_target_sample_id, 'max_current_base_unit_sample_id': EXPECTED_MAX_CURRENT_BASE_UNIT_SAMPLE_ID, 'action_buckets': list(V6_ACTION_BUCKETS), 'contract_cases': list(V6_CERTIFY_CONTRACT_CASES), 'contract_witness_sample_ids': {criterion: contract_witnesses[criterion] for criterion in V6_CERTIFY_CONTRACT_CASES}}, 'canonical': {'train_jsonl': str(train_record_path), 'train_sha256': _sha256_file(train_path), 'ordered_ids_sha256': ordered_ids_sha256, 'focus_sidecar': str(sidecar_record_path), 'focus_sidecar_sha256': _sha256_file(sidecar_path), 'focus_manifest': str(manifest_record_path), 'focus_manifest_sha256': _sha256_file(manifest_path), 'matcher_schema': focus_contract.AUDIT_LABEL_SCHEMA_VERSION, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'trainer_sha256': _sha256_file(trainer_path), 'runner_sha256': _sha256_file(runner_path)}, 'training_contract': {'run_id': V6_CERTIFY_RUN_ID, 'world_size': V6_WORLD_SIZE, 'per_device_train_batch_size': V6_PER_DEVICE_TRAIN_BATCH_SIZE, 'gradient_accumulation_steps': V6_GRADIENT_ACCUMULATION_STEPS, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'epochs': 1.0, 'expected_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'adapter_initialization': v6_adapter_initialization_contract(), 'loss': v6_loss_contract()}}

def validate_v6_certify_sample_plan(plan_path: str | Path, train_jsonl: str | Path, sidecars: Mapping[str, Mapping[str, Any]], *, max_target_sample_id: str, max_current_target_sample_id: str, _authority: Mapping[str, Any] | None=None) -> dict[str, Any]:
    path = Path(plan_path)
    actual = _load_json_object(path, context='V6 certify sample plan')
    expected = build_v6_certify_sample_plan(train_jsonl, sidecars, max_target_sample_id=max_target_sample_id, max_current_target_sample_id=max_current_target_sample_id, _authority=_authority)
    if actual != expected:
        raise V6SftError('V6 certify sample plan differs from canonical recomputation')
    return {'path': str(path), 'sha256': _sha256_file(path), 'rows': V6_CERTIFY_ROWS, 'selected_sample_ids': list(actual['selected_sample_ids']), 'selected_sample_ids_sha256': actual['selected_sample_ids_sha256'], 'bucket_counts': dict(actual['bucket_counts']), 'required_coverage': dict(actual['required_coverage'])}

def publish_v6_certify_sample_plan(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise V6SftError(f'certify sample plan path is a symlink: {path}')
    try:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing_text = path.read_text(encoding='utf-8')
        if existing_text != text:
            existing = _load_json_object(path, context='existing V6 certify plan')
            existing_comparable = json.loads(json.dumps(existing))
            new_comparable = json.loads(json.dumps(plan))
            for value in (existing_comparable, new_comparable):
                canonical = value.get('canonical')
                if isinstance(canonical, dict):
                    canonical.pop('trainer_sha256', None)
                    canonical.pop('runner_sha256', None)
            certify_receipt = CANONICAL_V6_LOG_ROOT / 'certify_receipt.json'
            certify_run_dir = CANONICAL_CERTIFY_OUTPUT_ROOT / V6_CERTIFY_RUN_ID
            if existing_comparable != new_comparable or certify_receipt.exists() or certify_run_dir.exists():
                raise V6SftError(f'refusing to overwrite different certify plan: {path}')
            _atomic_write_json(path, plan)
            publication = 'refreshed_pre_certification_implementation_hashes'
        else:
            publication = 'present_exact_no_overwrite'
    else:
        publication = 'created_exclusive'
    return {'path': str(path), 'sha256': _sha256_file(path), 'publication': publication, 'rows': V6_CERTIFY_ROWS}

def validate_v6_collator_contract(train_jsonl: str | Path, sidecars: Mapping[str, Mapping[str, Any]], base_collator: Any, *, max_target_sample_id: str, preflight_batch_size: int) -> dict[str, Any]:
    (rows, covered, diagnostics) = _select_v6_collator_preflight_rows(train_jsonl, sidecars, max_target_sample_id=max_target_sample_id)
    if preflight_batch_size < 1:
        raise V6SftError('collator preflight requires a positive batch size')

    class CapturingCollator:
        prompt_key = 'localwm_next'

        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            self.processor = wrapped.processor
            self.last_batch: dict[str, Any] | None = None

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            batch = self.wrapped(features)
            self.last_batch = {key: value.clone() if isinstance(value, torch.Tensor) else value for (key, value) in batch.items()}
            return batch
    capturing = CapturingCollator(base_collator)
    collator = V6Collator(capturing, sidecars)
    total_supervised_tokens = 0
    total_focus_tokens = 0

    def validate_batch(features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = collator(features)
        before = capturing.last_batch
        if before is None:
            raise V6SftError('base collator preflight did not capture a batch')
        for (key, before_value) in before.items():
            after_value = batch[key]
            if isinstance(before_value, torch.Tensor):
                if not torch.equal(before_value, after_value):
                    raise V6SftError(f'V6Collator changed base-collator tensor {key!r}')
            elif before_value != after_value:
                raise V6SftError(f'V6Collator changed base-collator value {key!r}')
        labels = batch['labels']
        base_ids = batch[BASE_UNIT_IDS_KEY]
        focus_ids = batch[FOCUS_UNIT_IDS_KEY]
        supervised = labels.ne(-100)
        if not torch.equal(supervised, base_ids.ge(0)):
            raise V6SftError(f"sample {row['id']} supervised labels are not fully base-covered")
        if bool(focus_ids.ge(0).logical_and(~supervised).any()):
            raise V6SftError('collator preflight has Focus outside the full target labels')
        return batch
    for row in rows:
        batch = validate_batch([row])
        focus_ids = batch[FOCUS_UNIT_IDS_KEY][0]
        mapped_focus_ids = {int(value) for value in focus_ids[focus_ids.ge(0)].unique().tolist()}
        expected_focus_ids = set(range(len(sidecars[str(row['id'])]['focus_units'])))
        if mapped_focus_ids != expected_focus_ids:
            raise V6SftError(f"sample {row['id']} did not map every Focus group: expected={sorted(expected_focus_ids)}, actual={sorted(mapped_focus_ids)}")
        total_supervised_tokens += int(batch['labels'].ne(-100).sum().item())
        total_focus_tokens += int(focus_ids.ge(0).sum().item())
    batch_rows = [next((row for row in rows if str(row['id']) == max_target_sample_id))]
    batch_rows.extend((row for row in rows if str(row['id']) != max_target_sample_id))
    batch_rows = batch_rows[:preflight_batch_size]
    if len(batch_rows) != preflight_batch_size:
        raise V6SftError(f'collator preflight cannot construct batch size {preflight_batch_size}')
    validate_batch(batch_rows)
    return {'status': 'passed', 'samples': len(rows), 'sample_ids': [str(row['id']) for row in rows], 'covered_contract_cases': sorted(covered), 'supervised_tokens': total_supervised_tokens, 'focus_tokens': total_focus_tokens, 'batch_sizes_tested': [1, preflight_batch_size], 'batch_sample_ids': [str(row['id']) for row in batch_rows], **diagnostics, 'base_labels_bitwise_unchanged': True, 'all_supervised_tokens_base_covered': True}

class StrictV6JsonlDataset:

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
            raise V6SftError(f'training dataset is empty: {self.path}')
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_handle'] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        if self._handle is None:
            self._handle = self.path.open('r', encoding='utf-8')
        self._handle.seek(self.offsets[index])
        try:
            row = json.loads(self._handle.readline())
        except json.JSONDecodeError as exc:
            raise V6SftError(f'invalid JSON at dataset index {index}') from exc
        if not isinstance(row, dict):
            raise V6SftError(f'non-object JSON at dataset index {index}')
        try:
            return _normalize_v6_training_row(row)
        except ValueError as exc:
            raise V6SftError(f'invalid training sample at index {index}: {exc}') from exc

class StrictV6CertifyDataset:

    def __init__(self, path: str | Path, sample_ids: Sequence[str]) -> None:
        self.base = StrictV6JsonlDataset(path)
        index_by_id: dict[str, int] = {}
        for (index, (_, row)) in enumerate(_iter_jsonl_objects(Path(path))):
            sample_id = str(row.get('id') or '')
            if sample_id in index_by_id:
                raise V6SftError(f'duplicate source ID in certify dataset: {sample_id}')
            index_by_id[sample_id] = index
        requested = list(sample_ids)
        if len(requested) != V6_CERTIFY_ROWS or len(set(requested)) != V6_CERTIFY_ROWS:
            raise V6SftError(f'certify dataset requires {V6_CERTIFY_ROWS} unique frozen IDs')
        missing = [sample_id for sample_id in requested if sample_id not in index_by_id]
        if missing:
            raise V6SftError(f'certify dataset IDs are absent: {missing[:5]}')
        self.indices = [index_by_id[sample_id] for sample_id in requested]
        self.sample_ids = tuple(requested)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self.indices)
        if index < 0 or index >= len(self.indices):
            raise IndexError(index)
        row = self.base[self.indices[index]]
        if str(row.get('id')) != self.sample_ids[index]:
            raise V6SftError('certify dataset source order changed after validation')
        return row

def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        equal = actual == expected
    if not equal:
        raise V6SftError(f'frozen V6 contract requires {name}={expected!r}, got {actual!r}')

def _validate_v6_certify_receipt_legacy(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path)
    receipt = _load_json_object(receipt_path, context='V6 six-GPU certify receipt')
    sidecar_path = Path(CANONICAL_FOCUS_SIDECAR)
    runner_path = Path(__file__).resolve().parents[3] / v6_runner_relative_path()
    plan_path = Path(str(receipt.get('certify_sample_plan') or ''))
    memory_audit_path = Path(str(receipt.get('gpu_memory_audit') or ''))
    summary_path = Path(str(receipt.get('train_summary') or ''))
    for (candidate, context) in ((plan_path, 'certify sample plan'), (memory_audit_path, 'certify GPU memory audit'), (summary_path, 'certify train summary')):
        if candidate.is_symlink() or not candidate.is_file():
            raise V6SftError(f'{context} is missing or a symlink: {candidate}')
    plan = _load_json_object(plan_path, context='V6 certify sample plan')
    summary = _load_json_object(summary_path, context='V6 certify train summary')
    selected_ids = plan.get('selected_sample_ids')
    bucket_counts = plan.get('bucket_counts')
    required_coverage = plan.get('required_coverage')
    canonical = plan.get('canonical')
    training_contract = plan.get('training_contract')
    if plan.get('schema_version') != V6_CERTIFY_PLAN_SCHEMA or plan.get('status') != 'passed' or plan.get('rows') != V6_CERTIFY_ROWS or (not isinstance(selected_ids, list)) or (len(selected_ids) != V6_CERTIFY_ROWS) or any((not isinstance(sample_id, str) for sample_id in selected_ids)) or (len(set(selected_ids)) != V6_CERTIFY_ROWS) or (not isinstance(bucket_counts, Mapping)) or (bucket_counts != {bucket: V6_CERTIFY_ROWS // len(V6_ACTION_BUCKETS) for bucket in V6_ACTION_BUCKETS}) or (not isinstance(required_coverage, Mapping)) or (required_coverage.get('max_target_sample_id') not in selected_ids) or (required_coverage.get('max_focus_sample_id') not in selected_ids) or (required_coverage.get('max_current_target_sample_id') != EXPECTED_MAX_CURRENT_TARGET_SAMPLE_ID) or (EXPECTED_MAX_CURRENT_TARGET_SAMPLE_ID not in selected_ids) or (required_coverage.get('max_current_base_unit_sample_id') != EXPECTED_MAX_CURRENT_BASE_UNIT_SAMPLE_ID) or (EXPECTED_MAX_CURRENT_BASE_UNIT_SAMPLE_ID not in selected_ids) or (required_coverage.get('action_buckets') != list(V6_ACTION_BUCKETS)) or (required_coverage.get('contract_cases') != list(V6_CERTIFY_CONTRACT_CASES)) or (not isinstance(required_coverage.get('contract_witness_sample_ids'), Mapping)) or (set(required_coverage['contract_witness_sample_ids']) != set(V6_CERTIFY_CONTRACT_CASES)) or any((required_coverage['contract_witness_sample_ids'][criterion] not in selected_ids for criterion in V6_CERTIFY_CONTRACT_CASES)) or (not isinstance(canonical, Mapping)) or (canonical.get('focus_sidecar') != str(CANONICAL_FOCUS_SIDECAR)) or (not isinstance(training_contract, Mapping)) or (training_contract.get('run_id') != V6_CERTIFY_RUN_ID) or (training_contract.get('world_size') != V6_WORLD_SIZE) or (training_contract.get('global_batch_size') != V6_GLOBAL_BATCH_SIZE) or (training_contract.get('expected_optimizer_steps') != V6_CERTIFY_EXPECTED_STEPS) or (training_contract.get('adapter_initialization') != v6_adapter_initialization_contract()) or (training_contract.get('loss') != v6_loss_contract()):
        raise V6SftError('certify sample plan contract is invalid')
    summary_plan = summary.get('certify_sample_plan')
    loaded_graph = summary.get('loaded_adapter_graph')
    if summary.get('status') != 'passed' or summary.get('run_id') != V6_CERTIFY_RUN_ID or summary.get('certify_run') is not True or (summary.get('formal_training_started') is not False) or (summary.get('final_global_step') != V6_CERTIFY_EXPECTED_STEPS) or (summary.get('expected_final_step') != V6_CERTIFY_EXPECTED_STEPS) or (summary.get('trainable_parameters') != EXPECTED_TRAINABLE_PARAMS) or (summary.get('adapter_initialization') != v6_adapter_initialization_contract()) or (not isinstance(loaded_graph, Mapping)) or (loaded_graph.get('module_count') != EXPECTED_EXPANDED_MODULES) or (loaded_graph.get('trainable_parameters') != EXPECTED_TRAINABLE_PARAMS) or (summary.get('focus_loss_active') is not True) or (summary.get('loss_ramp') is not False) or (summary.get('finite_loss_observed') is not True) or (not isinstance(summary_plan, Mapping)) or (Path(str(summary_plan.get('path') or '')).resolve() != plan_path.resolve()) or (summary_plan.get('bucket_counts') != plan.get('bucket_counts')) or (summary_plan.get('required_coverage') != plan.get('required_coverage')):
        raise V6SftError('certify train summary is not bound to the validated plan')
    lines = memory_audit_path.read_text(encoding='utf-8').splitlines()
    expected_header = ['timestamp', 'gpu_index', 'memory_used_mib', 'utilization_percent', 'phase', 'minimum_active_mib']
    if not lines or lines[0].split('\t') != expected_header:
        raise V6SftError('certify GPU memory audit header is invalid')
    snapshots: dict[str, dict[str, dict[str, Any]]] = {}
    for (line_number, line) in enumerate(lines[1:], start=2):
        columns = line.split('\t')
        if len(columns) != len(expected_header):
            raise V6SftError(f'certify GPU memory audit has invalid row {line_number}')
        row = dict(zip(expected_header, columns))
        try:
            gpu_index = int(row['gpu_index'])
            used = int(row['memory_used_mib'])
            utilization = int(row['utilization_percent'])
            minimum = int(row['minimum_active_mib'])
        except ValueError as exc:
            raise V6SftError(f'certify GPU memory audit has non-integer row {line_number}') from exc
        if gpu_index not in range(V6_WORLD_SIZE) or used < 0 or utilization not in range(101) or (minimum < V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB) or (row['phase'] not in {'loading', 'active'}) or (not row['timestamp']):
            raise V6SftError(f'certify GPU memory audit has out-of-contract row {line_number}')
        timestamp_rows = snapshots.setdefault(row['timestamp'], {})
        key = str(gpu_index)
        if key in timestamp_rows:
            raise V6SftError('certify GPU memory audit repeats a GPU within one snapshot')
        timestamp_rows[key] = {'used': used, 'utilization': utilization, 'phase': row['phase'], 'minimum': minimum}
    qualifying_memory: list[dict[str, int]] = []
    expected_gpu_keys = {str(index) for index in range(V6_WORLD_SIZE)}
    for timestamp_rows in snapshots.values():
        if set(timestamp_rows) != expected_gpu_keys:
            continue
        if all((row['phase'] == 'active' and row['used'] >= V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB for row in timestamp_rows.values())):
            qualifying_memory.append({key: int(timestamp_rows[key]['used']) for key in sorted(timestamp_rows)})
    if not qualifying_memory:
        raise V6SftError('certify GPU memory audit lacks one simultaneous active six-GPU snapshot')
    memory_by_gpu = receipt.get('simultaneous_memory_used_mib')
    if not isinstance(memory_by_gpu, Mapping):
        raise V6SftError('certify receipt lacks simultaneous per-GPU memory evidence')
    try:
        normalized_receipt_memory = {str(key): int(value) for (key, value) in memory_by_gpu.items()}
        memory_ok = set(normalized_receipt_memory) == expected_gpu_keys and all((normalized_receipt_memory[str(index)] >= V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB for index in range(V6_WORLD_SIZE)))
    except (KeyError, TypeError, ValueError) as exc:
        raise V6SftError('certify receipt per-GPU memory evidence is invalid') from exc
    expected = {'schema_version': V6_CERTIFY_RECEIPT_SCHEMA, 'status': 'passed', 'formal_training_started': False, 'run_id': V6_CERTIFY_RUN_ID, 'world_size': V6_WORLD_SIZE, 'physical_gpus': list(range(V6_WORLD_SIZE)), 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'expected_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'final_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'sidecar_sha256': _sha256_file(sidecar_path), 'trainer_sha256': _sha256_file(Path(__file__).resolve()), 'runner_sha256': _sha256_file(runner_path), 'certify_sample_plan_sha256': _sha256_file(plan_path), 'gpu_memory_audit_sha256': _sha256_file(memory_audit_path), 'train_summary_sha256': _sha256_file(summary_path), 'adapter_initialization_mode': V6_ADAPTER_INIT_MODE, 'expanded_lora_modules': EXPECTED_EXPANDED_MODULES, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'focus_loss_active': True, 'loss_ramp': False, 'finite_loss_observed': True, 'simultaneous_memory_floor_mib': V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB, 'no_training_processes_remaining': True, 'certify_checkpoint_forbidden_for_formal_resume': True}
    mismatches = {key: {'expected': expected_value, 'actual': receipt.get(key)} for (key, expected_value) in expected.items() if receipt.get(key) != expected_value}
    if mismatches or not memory_ok or normalized_receipt_memory not in qualifying_memory:
        raise V6SftError(f'six-GPU certify receipt does not authorize formal V6 training: {mismatches}')
    return {'path': str(receipt_path), 'sha256': _sha256_file(receipt_path), 'run_id': V6_CERTIFY_RUN_ID, 'final_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'simultaneous_memory_floor_mib': V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB}

def _exact_regular_file(raw_value: object, expected: Path, *, context: str, recorded: str | Path | None=None) -> Path:
    """Require both the recorded lexical name and resolved file identity."""
    anchor = expected
    recorded_name = str(anchor if recorded is None else recorded)
    if not isinstance(raw_value, str) or raw_value != recorded_name:
        raise V6SftError(f'{context} must record exact canonical name {recorded_name!r}')
    if anchor.is_symlink() or not anchor.is_file():
        raise V6SftError(f'{context} is missing or a symlink: {anchor}')
    entity = anchor.resolve(strict=True)
    if entity != anchor:
        raise V6SftError(f'{context} resolves outside its canonical anchor')
    return entity

def _parse_gpu_audit(path: Path, *, start_byte: int, start_line: int, context: str) -> dict[str, Any]:
    """Validate the complete TSV, then isolate one launch by byte and line."""
    if path.is_symlink() or not path.is_file():
        raise V6SftError(f'{context} is missing or a symlink')
    header = 'timestamp\tgpu_index\tmemory_used_mib\tutilization_percent\tphase\tminimum_active_mib'
    full = path.read_bytes()
    try:
        full_text = full.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise V6SftError(f'{context} is not UTF-8') from exc
    full_lines = full_text.splitlines()
    if not full_lines or full_lines[0] != header or full_lines.count(header) != 1:
        raise V6SftError(f'{context} must contain exactly one canonical header')

    def parse_rows(lines: Sequence[str], *, allow_header: bool) -> dict[str, dict[str, dict[str, Any]]]:
        snapshots: dict[str, dict[str, dict[str, Any]]] = {}
        for (line_number, line) in enumerate(lines, start=1):
            if allow_header and line_number == 1 and (line == header):
                continue
            columns = line.split('\t')
            if len(columns) != 6:
                raise V6SftError(f'{context} has invalid row {line_number}')
            (timestamp, index_raw, used_raw, util_raw, phase, minimum_raw) = columns
            try:
                index = int(index_raw)
                used = int(used_raw)
                utilization = int(util_raw)
                minimum = int(minimum_raw)
            except ValueError as exc:
                raise V6SftError(f'{context} has non-integer evidence') from exc
            if not timestamp or index not in range(V6_WORLD_SIZE) or used < 0 or (utilization not in range(101)) or (phase not in {'loading', 'active'}) or (minimum < V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB):
                raise V6SftError(f'{context} has out-of-contract evidence')
            group = snapshots.setdefault(timestamp, {})
            key = str(index)
            if key in group:
                raise V6SftError(f'{context} repeats GPU {index} at {timestamp}')
            group[key] = {'used': used, 'utilization': utilization, 'phase': phase, 'minimum': minimum}
        return snapshots
    parse_rows(full_lines, allow_header=True)
    appended = _read_launch_append(path, start_byte, start_line, context=f'{context} launch window').decode('utf-8', errors='strict')
    launch_lines = appended.splitlines()
    if launch_lines and launch_lines[0] == header:
        launch_lines = launch_lines[1:]
    if header in launch_lines:
        raise V6SftError(f'{context} launch window contains a duplicate header')
    snapshots = parse_rows(launch_lines, allow_header=False)
    expected_keys = {str(index) for index in range(V6_WORLD_SIZE)}
    qualifying: dict[str, dict[str, int]] = {}
    for (timestamp, rows) in snapshots.items():
        if set(rows) == expected_keys and all((row['phase'] == 'active' and int(row['used']) >= V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB for row in rows.values())):
            qualifying[timestamp] = {key: int(rows[key]['used']) for key in sorted(rows)}
    if not qualifying:
        raise V6SftError(f'{context} lacks a current-launch simultaneous six-GPU snapshot')
    return {'sha256': _sha256_file(path), 'qualifying_snapshots': qualifying}

def _parse_v6_launch_events(path: Path, *, start_byte: int, start_line: int, context: str) -> dict[str, Any]:
    appended = _read_launch_append(path, start_byte, start_line, context=context).decode('utf-8', errors='strict')
    params: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    for raw_line in appended.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('event') == 'v6_trainable_parameters':
            params.append(event)
        elif event.get('event') == 'v6_loss_finite':
            losses.append(event)
    params_ok = any((event.get('trainable_parameters') == EXPECTED_TRAINABLE_PARAMS and event.get('expanded_lora_modules') == EXPECTED_EXPANDED_MODULES and (event.get('trainable_tensor_count') == EXPECTED_ADAPTER_TENSORS) and (event.get('target_modules_sha256') == EXPECTED_TARGET_MODULES_SHA256) and (event.get('adapter_initialization_mode') == V6_ADAPTER_INIT_MODE) and (event.get('single_adapter') is True) and (event.get('only_lora_parameters_trainable') is True) and (event.get('base_parameters_frozen') is True) and (event.get('init_adapter_weights_sha256') == EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256) for event in params))
    finite_events: dict[int, dict[str, Any]] = {}
    for event in losses:
        try:
            step = int(event.get('global_step'))
            loss = float(event.get('loss'))
            beta = float(event.get('beta'))
            eligible_count = float(event.get('eligible_count'))
            replay_error = float(event.get('current_replay_max_error'))
            gate_flip_count = float(event.get('gate_flip_count'))
        except (TypeError, ValueError):
            continue
        if step > 0 and math.isfinite(loss) and math.isfinite(beta) and math.isfinite(eligible_count) and math.isfinite(replay_error) and math.isfinite(gate_flip_count) and (event.get('focus_loss_active') is True) and (event.get('loss_ramp') is False):
            if step in finite_events:
                raise V6SftError(f'{context} repeats finite-loss step {step}')
            finite_events[step] = event
    if not params_ok or not finite_events:
        raise V6SftError(f'{context} lacks current-launch params/finite loss')
    expected_steps = list(range(1, V6_CERTIFY_EXPECTED_STEPS + 1))
    if 'certify' in context.lower():
        if sorted(finite_events) != expected_steps:
            raise V6SftError(f'{context} finite-loss steps differ from {expected_steps}')
        for step in expected_steps:
            event = finite_events[step]
            expected_beta = V6_BETA_MAX * min(1.0, step / V6_BETA_WARMUP_STEPS)
            if not math.isclose(float(event['beta']), expected_beta, rel_tol=0.0, abs_tol=1e-12) or float(event['eligible_count']) <= 0.0 or float(event['current_replay_max_error']) > V6_REPLAY_ATOL or (float(event['gate_flip_count']) != 0.0):
                raise V6SftError(f'{context} step {step} violates beta/eligible/replay/gate contract')
    return {'sha256': _sha256_file(path), 'finite_loss_steps': sorted(finite_events), 'finite_loss_events': [{'global_step': step, 'loss': float(finite_events[step]['loss']), 'beta': float(finite_events[step]['beta']), 'eligible_count': float(finite_events[step]['eligible_count']), 'current_replay_max_error': float(finite_events[step]['current_replay_max_error']), 'gate_flip_count': float(finite_events[step]['gate_flip_count'])} for step in sorted(finite_events)], 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS}

def _validate_final_adapter_chain(*, run_dir: Path, recorded_run_dir: Path, summary_path: Path, run_config_path: Path, run_id: str, certify_run: bool, expected_step: int, data_authority: Mapping[str, Any], certify_plan_path: Path | None=None, certify_plan_record_path: Path | None=None) -> dict[str, Any]:
    """Deep-validate completion artifacts without trusting summary hashes."""
    summary = _load_json_object(summary_path, context=f'{run_id} train summary')
    run_config = _load_json_object(run_config_path, context=f'{run_id} run_config')
    target_modules = _validate_target_modules(run_config.get('model', {}).get('target_modules'), context=f'{run_id} target_modules')
    expected_graph = {'module_count': EXPECTED_EXPANDED_MODULES, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'only_lora_parameters_trainable': True}
    data = run_config.get('data')
    arguments = run_config.get('args')
    expected_data = {'train_jsonl': str(data_authority['train_jsonl']), 'train_sha256': _sha256_file(Path(data_authority.get('train_file', data_authority['train_jsonl']))), 'focus_sidecar': str(data_authority['focus_sidecar']), 'focus_sidecar_sha256': _sha256_file(Path(data_authority.get('focus_sidecar_file', data_authority['focus_sidecar']))), 'focus_manifest': str(data_authority['focus_manifest']), 'focus_manifest_sha256': _sha256_file(Path(data_authority.get('focus_manifest_file', data_authority['focus_manifest']))), 'rows': int(data_authority.get('rows', EXPECTED_TRAINING_ROWS))}
    if not isinstance(data, Mapping) or any((data.get(key) != value for (key, value) in expected_data.items())):
        raise V6SftError(f'{run_id} run_config canonical data identity drifted')
    expected_args = {'base_model': CANONICAL_BASE_MODEL, 'train_jsonl': str(data_authority['train_jsonl']), 'focus_sidecar': str(data_authority['focus_sidecar']), 'output_dir': str(data_authority.get('output_root', run_dir.parent)), 'run_id': run_id, 'prompt_key': 'localwm_next', 'init_adapter': str(CANONICAL_INIT_ADAPTER), 'resume_from_checkpoint': None, 'dry_run': False, 'per_device_train_batch_size': V6_PER_DEVICE_TRAIN_BATCH_SIZE, 'gradient_accumulation_steps': V6_GRADIENT_ACCUMULATION_STEPS, 'num_train_epochs': 1.0 if certify_run else 2.0, 'learning_rate': 2e-05, 'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.03, 'weight_decay': 0.01, 'seed': 0, 'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'target_modules': 'qwen35_expanded', 'max_length': 20000, 'max_output_tokens': 6000, 'image_min_pixels': 200704, 'image_max_pixels': 802816, 'min_supervised_tokens': 16, 'assistant_loss_scope': 'full', 'dataset_loading_mode': 'direct', 'skip_missing_images': False, 'skip_overlong_samples': False, 'device_mode': 'ddp', 'gradient_checkpointing': True, 'bf16': True, 'fp16': False, 'save_steps': V6_SAVE_STEPS, 'logging_steps': 1, 'wm_eval_max_samples': 500, 'certify_run': certify_run, 'certify_sample_plan': str(certify_plan_record_path) if certify_run else None, 'certify_receipt': None if certify_run else str(data_authority['certify_receipt']), 'resume_validation_only': False}
    if not isinstance(arguments, Mapping) or any((arguments.get(key) != value for (key, value) in expected_args.items())):
        raise V6SftError(f'{run_id} run_config frozen args drifted')
    if run_config.get('schema_version') != V6_RUN_CONFIG_SCHEMA or run_config.get('run_id') != run_id or run_config.get('certify_run') is not certify_run or (run_config.get('v5_parity', {}).get('status') != 'passed') or (run_config.get('v5_parity', {}).get('loss_equal') is not True) or (run_config.get('model', {}).get('base_model') != CANONICAL_BASE_MODEL) or (run_config.get('model', {}).get('init_adapter') != str(CANONICAL_INIT_ADAPTER)) or (run_config.get('model', {}).get('adapter_initialization') != v6_adapter_initialization_contract()) or (run_config.get('model', {}).get('loaded_adapter_graph') != expected_graph) or (run_config.get('loss') != v6_loss_contract()) or (run_config.get('distributed', {}).get('world_size') != V6_WORLD_SIZE) or (run_config.get('distributed', {}).get('cuda_visible_devices') != V6_CUDA_VISIBLE_DEVICES) or (run_config.get('distributed', {}).get('global_batch_size') != V6_GLOBAL_BATCH_SIZE):
        raise V6SftError(f'{run_id} run_config completion contract drifted')
    if certify_run:
        expected_completion = {'canonical_training_rows': V6_CERTIFY_ROWS, 'epochs': 1.0, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'expected_final_step': V6_CERTIFY_EXPECTED_STEPS, 'expected_trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'formal_resume_allowed': False}
        certify_plan_report = run_config.get('certify_sample_plan')
        if run_config.get('completion_contract') != expected_completion or run_config.get('certify_receipt_authorization') is not None or (not isinstance(certify_plan_report, Mapping)) or (certify_plan_path is None) or (certify_plan_record_path is None) or (certify_plan_report.get('path') != str(certify_plan_record_path)):
            raise V6SftError(f'{run_id} certify run_config contract drifted')
    final_adapter = run_dir / 'final_adapter'
    recorded_final_adapter = recorded_run_dir / 'final_adapter'
    if not isinstance(summary.get('final_adapter'), str) or summary.get('final_adapter') != str(recorded_final_adapter) or final_adapter.is_symlink() or (not final_adapter.is_dir()) or (final_adapter.resolve(strict=True) != run_dir.resolve(strict=True) / 'final_adapter'):
        raise V6SftError(f'{run_id} final_adapter path is not canonical')
    adapter_report = validate_expanded_adapter_files(final_adapter, expected_modules=target_modules)
    adapter_report['path'] = str(recorded_final_adapter)
    integrity_path = final_adapter / 'adapter_integrity.json'
    integrity = _load_json_object(integrity_path, context=f'{run_id} final adapter integrity')
    required = integrity.get('required_files')
    if integrity.get('status') != 'passed' or integrity.get('global_step') != expected_step or integrity.get('adapter_tensor_count') != EXPECTED_ADAPTER_TENSORS or (not isinstance(required, Mapping)) or (set(required) != {'adapter_model.safetensors', 'adapter_config.json'}):
        raise V6SftError(f'{run_id} final adapter integrity contract failed')
    for name in sorted(required):
        artifact = final_adapter / name
        metadata = required[name]
        if artifact.is_symlink() or not artifact.is_file() or (not isinstance(metadata, Mapping)) or (metadata.get('size_bytes') != artifact.stat().st_size):
            raise V6SftError(f'{run_id} final adapter integrity failed: {name}')
    latest = run_dir / 'latest_successful_checkpoint.txt'
    if latest.is_symlink() or not latest.is_file() or latest.read_text(encoding='utf-8') != f'{recorded_final_adapter}\n':
        raise V6SftError(f'{run_id} latest-success pointer is invalid')
    if summary.get('status') != 'passed' or summary.get('run_id') != run_id or summary.get('certify_run') is not certify_run or (summary.get('formal_training_started') is certify_run) or (summary.get('final_global_step') != expected_step) or (summary.get('expected_final_step') != expected_step) or (summary.get('adapter_initialization') != v6_adapter_initialization_contract()) or (summary.get('loaded_adapter_graph') != expected_graph) or (summary.get('trainable_parameters') != EXPECTED_TRAINABLE_PARAMS) or (summary.get('trainable_tensor_count') != EXPECTED_ADAPTER_TENSORS) or (summary.get('focus_loss_active') is not True) or (summary.get('finite_loss_observed') is not True) or (summary.get('loss_ramp') is not False) or (summary.get('global_batch_size') != V6_GLOBAL_BATCH_SIZE) or (summary.get('final_adapter_contract') != adapter_report):
        raise V6SftError(f'{run_id} train summary completion contract failed')
    return {'summary': summary, 'run_config': run_config, 'adapter_contract': adapter_report, 'integrity_path': str(integrity_path), 'integrity_sha256': _sha256_file(integrity_path), 'latest_pointer': str(latest), 'latest_pointer_sha256': _sha256_file(latest)}

def _certify_authority(test_authority: Mapping[str, Any] | None) -> dict[str, Any]:
    trainer = Path(__file__).resolve()
    repo_root = trainer.parents[3]
    runner = repo_root / v6_runner_relative_path()
    if test_authority is None:
        log_root = repo_root / CANONICAL_V6_LOG_ROOT
        sidecar = repo_root / CANONICAL_FOCUS_SIDECAR
        return {'test': False, 'log_root': log_root, 'recorded_log_root': CANONICAL_V6_LOG_ROOT, 'train_jsonl': repo_root / CANONICAL_TRAIN_JSONL, 'recorded_train_jsonl': CANONICAL_TRAIN_JSONL, 'focus_sidecar': sidecar, 'recorded_focus_sidecar': CANONICAL_FOCUS_SIDECAR, 'focus_manifest': sidecar.parent / 'manifest.json', 'recorded_focus_manifest': CANONICAL_FOCUS_SIDECAR.parent / 'manifest.json', 'certify_run_dir': repo_root / CANONICAL_CERTIFY_OUTPUT_ROOT / V6_CERTIFY_RUN_ID, 'recorded_certify_run_dir': CANONICAL_CERTIFY_OUTPUT_ROOT / V6_CERTIFY_RUN_ID, 'trainer_path': trainer, 'recorded_trainer_path': trainer, 'runner_path': runner, 'recorded_runner_path': runner, 'token_contract': canonical_v6_token_contract()}
    required = {'log_root', 'train_jsonl', 'focus_sidecar', 'focus_manifest', 'certify_run_dir', 'trainer_path', 'runner_path', 'token_contract'}
    if not required.issubset(test_authority):
        raise V6SftError('_test_authority lacks isolated certify authority paths')
    authority = {key: Path(test_authority[key]) for key in required if key != 'token_contract'}
    authority['token_contract'] = dict(test_authority['token_contract'])
    for key in ('log_root', 'train_jsonl', 'focus_sidecar', 'focus_manifest', 'certify_run_dir', 'trainer_path', 'runner_path'):
        authority[f'recorded_{key}'] = Path(test_authority.get(f'recorded_{key}', test_authority[key]))
    authority['test'] = True
    return authority

def validate_v6_certify_receipt(path: str | Path, *, _test_authority: Mapping[str, Any] | None=None) -> dict[str, Any]:
    authority = _certify_authority(_test_authority)
    log_root = authority['log_root']
    expected_receipt = log_root / 'certify_receipt.json'
    receipt_path = _exact_regular_file(str(path), expected_receipt, context='V6 certify receipt', recorded=authority['recorded_log_root'] / 'certify_receipt.json')
    receipt = _load_json_object(receipt_path, context='V6 certify receipt')
    try:
        pid = int(receipt.get('training_pid'))
        pgid = int(receipt.get('training_pgid'))
        log_byte = int(receipt.get('certify_log_start_byte'))
        log_line = int(receipt.get('certify_log_start_line'))
        audit_byte = int(receipt.get('gpu_audit_start_byte'))
        audit_line = int(receipt.get('gpu_audit_start_line'))
    except (TypeError, ValueError) as exc:
        raise V6SftError('certify receipt launch identity is invalid') from exc
    if min(pid, pgid) <= 0 or min(log_byte, log_line, audit_byte, audit_line) < 0:
        raise V6SftError('certify receipt launch identity is invalid')
    expected_paths = {'preflight': log_root / 'preflight.json', 'certify_sample_plan': log_root / 'certify_sample_plan.json', 'certify_log': log_root / 'certify.log', 'gpu_memory_audit': log_root / 'certify_gpu_memory.tsv', 'monitor_status': log_root / f'certify_monitor_status.{pid}.json', 'train_summary': authority['certify_run_dir'] / 'train_summary.json', 'run_config': authority['certify_run_dir'] / 'run_config.json', 'certify_checkpoint_integrity': authority['certify_run_dir'] / 'checkpoints/checkpoint-2/checkpoint_integrity.json', 'certify_checkpoint_binding': authority['certify_run_dir'] / 'checkpoints/checkpoint-2/v6_checkpoint_binding.json', 'certify_checkpoint_cold_load': log_root / 'certify_checkpoint_cold_load.json'}
    recorded_paths = {'preflight': authority['recorded_log_root'] / 'preflight.json', 'certify_sample_plan': authority['recorded_log_root'] / 'certify_sample_plan.json', 'certify_log': authority['recorded_log_root'] / 'certify.log', 'gpu_memory_audit': authority['recorded_log_root'] / 'certify_gpu_memory.tsv', 'monitor_status': authority['recorded_log_root'] / f'certify_monitor_status.{pid}.json', 'train_summary': authority['recorded_certify_run_dir'] / 'train_summary.json', 'run_config': authority['recorded_certify_run_dir'] / 'run_config.json', 'certify_checkpoint_integrity': authority['recorded_certify_run_dir'] / 'checkpoints/checkpoint-2/checkpoint_integrity.json', 'certify_checkpoint_binding': authority['recorded_certify_run_dir'] / 'checkpoints/checkpoint-2/v6_checkpoint_binding.json', 'certify_checkpoint_cold_load': authority['recorded_log_root'] / 'certify_checkpoint_cold_load.json'}
    files = {name: _exact_regular_file(receipt.get(name), expected, context=f'certify {name}', recorded=recorded_paths[name]) for (name, expected) in expected_paths.items()}
    trainer = _exact_regular_file(receipt.get('trainer_path'), authority['trainer_path'], context='certify trainer', recorded=authority['recorded_trainer_path'])
    runner = _exact_regular_file(receipt.get('runner_path'), authority['runner_path'], context='certify runner', recorded=authority['recorded_runner_path'])
    sidecar = _exact_regular_file(str(authority['focus_sidecar']), authority['focus_sidecar'], context='certify focus sidecar', recorded=authority['focus_sidecar'])
    manifest = _exact_regular_file(str(authority['focus_manifest']), authority['focus_manifest'], context='certify focus manifest', recorded=authority['focus_manifest'])
    train = _exact_regular_file(str(authority['train_jsonl']), authority['train_jsonl'], context='certify train JSONL', recorded=authority['train_jsonl'])
    preflight = _load_json_object(files['preflight'], context='V6 preflight')
    if authority['test']:
        sidecars = load_focus_sidecar(sidecar)
        dataset_ids = [str(row.get('id') or '') for (_, row) in _iter_jsonl_objects(train)]
        data_report = {'train_jsonl': str(train), 'train_sha256': _sha256_file(train), 'focus_sidecar': str(sidecar), 'focus_sidecar_sha256': _sha256_file(sidecar), 'focus_manifest': str(manifest), 'focus_manifest_sha256': _sha256_file(manifest), 'rows': len(dataset_ids), 'ordered_ids_sha256': _sha256_text('\n'.join(dataset_ids) + '\n')}
    else:
        (sidecars, data_report) = validate_v6_data_contract(train, sidecar)
        data_report['train_jsonl'] = str(authority['recorded_train_jsonl'])
        data_report['focus_sidecar'] = str(authority['recorded_focus_sidecar'])
        data_report['focus_manifest'] = str(authority['recorded_focus_manifest'])
    token_alignment = preflight.get('token_alignment')
    token_contract = authority['token_contract']
    expected_token_alignment = {'status': 'passed', 'rows': token_contract['rows'], 'total_target_tokens': token_contract['total_target_tokens'], 'max_target_tokens': token_contract['max_target_tokens'], 'max_target_sample_id': token_contract['max_target_sample_id'], 'current_total_target_tokens': EXPECTED_TOTAL_CURRENT_TARGET_TOKENS, 'current_max_target_tokens': EXPECTED_MAX_CURRENT_TARGET_TOKENS, 'max_current_target_sample_id': EXPECTED_MAX_CURRENT_TARGET_SAMPLE_ID, 'changed_token_equal_rows': 2, 'nonchanged_token_equal_rows': EXPECTED_NONCHANGED_TOKEN_EQUAL_ROWS, 'text_different_token_equal_rows': 0, 'rank_eligible_rows': 25005, 'max_output_tokens': token_contract['max_output_tokens']}
    if token_alignment != expected_token_alignment:
        raise V6SftError('preflight differs from independently frozen tokenizer contract')
    max_target_id = str(token_contract['max_target_sample_id'])
    plan_authority = {'focus_sidecar': sidecar, 'focus_manifest': manifest, 'trainer_path': trainer, 'runner_path': runner}
    if not authority['test']:
        plan_authority.update({'train_record_path': authority['recorded_train_jsonl'], 'focus_sidecar_record_path': authority['recorded_focus_sidecar'], 'focus_manifest_record_path': authority['recorded_focus_manifest']})
    expected_plan = build_v6_certify_sample_plan(train, sidecars, max_target_sample_id=max_target_id, max_current_target_sample_id=EXPECTED_MAX_CURRENT_TARGET_SAMPLE_ID, _authority=plan_authority)
    plan = _load_json_object(files['certify_sample_plan'], context='V6 certify plan')
    if plan != expected_plan:
        raise V6SftError('certify plan differs from deterministic canonical recomputation')
    publication = preflight.get('certify_sample_plan', {}).get('publication')
    preflight_expected = {'status': 'passed', 'can_start_training': True, 'formal_training_started': False, 'adapter_initialization': v6_adapter_initialization_contract(), 'loss': v6_loss_contract()}
    if any((preflight.get(key) != value for (key, value) in preflight_expected.items())):
        raise V6SftError('preflight top-level V6 contract drifted')
    for key in ('train_jsonl', 'train_sha256', 'focus_sidecar', 'focus_sidecar_sha256', 'focus_manifest', 'focus_manifest_sha256', 'rows', 'ordered_ids_sha256'):
        if preflight.get('data', {}).get(key) != data_report.get(key):
            raise V6SftError(f'preflight data identity drifted: {key}')
    training = preflight.get('training')
    if not isinstance(training, Mapping) or training.get('world_size') != V6_WORLD_SIZE or training.get('per_device_train_batch_size') != V6_PER_DEVICE_TRAIN_BATCH_SIZE or (training.get('gradient_accumulation_steps') != V6_GRADIENT_ACCUMULATION_STEPS) or (training.get('global_batch_size') != V6_GLOBAL_BATCH_SIZE) or (training.get('epochs') != 2.0) or (training.get('expected_trainable_parameters') != EXPECTED_TRAINABLE_PARAMS) or (training.get('expected_final_step') != EXPECTED_FINAL_STEP) or (not isinstance(publication, Mapping)) or (publication.get('path') != str(recorded_paths['certify_sample_plan'])):
        raise V6SftError('preflight training/plan publication contract drifted')
    launch_events = _parse_v6_launch_events(files['certify_log'], start_byte=log_byte, start_line=log_line, context='V6 certify log')
    audit = _parse_gpu_audit(files['gpu_memory_audit'], start_byte=audit_byte, start_line=audit_line, context='V6 certify GPU audit')
    monitor = _load_json_object(files['monitor_status'], context='certify monitor')
    timestamp = receipt.get('qualifying_snapshot_timestamp')
    memory = receipt.get('simultaneous_memory_used_mib')
    if not isinstance(timestamp, str) or timestamp not in audit['qualifying_snapshots']:
        raise V6SftError('certify receipt qualifying GPU timestamp is invalid')
    if memory != audit['qualifying_snapshots'][timestamp]:
        raise V6SftError('certify receipt GPU memory differs from qualifying snapshot')
    monitor_expected = {'schema_version': 'local-wm-v6-gpu-monitor-v2', 'status': 'passed', 'kind': 'certify', 'training_pid': pid, 'training_pgid': pgid, 'gpu_memory_audit': str(recorded_paths['gpu_memory_audit']), 'gpu_memory_audit_sha256': audit['sha256'], 'simultaneous_memory_used_mib': memory, 'qualifying_snapshot_timestamp': timestamp, 'parameters_observed': True, 'finite_loss_observed': True, 'simultaneous_memory_floor_observed': True, 'monitor_failed': False, 'launch_log_start_byte': log_byte, 'launch_log_start_line': log_line, 'gpu_audit_start_byte': audit_byte, 'gpu_audit_start_line': audit_line}
    if monitor != monitor_expected:
        raise V6SftError('certify monitor is not the exact receipt launch evidence')
    completion = _validate_final_adapter_chain(run_dir=authority['certify_run_dir'], recorded_run_dir=authority['recorded_certify_run_dir'], summary_path=files['train_summary'], run_config_path=files['run_config'], run_id=V6_CERTIFY_RUN_ID, certify_run=True, expected_step=V6_CERTIFY_EXPECTED_STEPS, data_authority={'train_jsonl': authority['recorded_train_jsonl'], 'train_file': train, 'focus_sidecar': authority['recorded_focus_sidecar'], 'focus_sidecar_file': sidecar, 'focus_manifest': authority['recorded_focus_manifest'], 'focus_manifest_file': manifest, 'rows': data_report['rows'], 'output_root': authority['recorded_certify_run_dir'].parent}, certify_plan_path=files['certify_sample_plan'], certify_plan_record_path=authority['recorded_log_root'] / 'certify_sample_plan.json')
    checkpoint_integrity = _load_json_object(files['certify_checkpoint_integrity'], context='V6 certify checkpoint integrity')
    checkpoint_binding = _load_json_object(files['certify_checkpoint_binding'], context='V6 certify checkpoint binding')
    checkpoint_cold_load = _load_json_object(files['certify_checkpoint_cold_load'], context='V6 certify checkpoint cold-load evidence')
    required_files = checkpoint_integrity.get('required_files')
    expected_required_names = set(_required_checkpoint_files(V6_WORLD_SIZE))
    if checkpoint_integrity.get('status') != 'passed' or checkpoint_integrity.get('global_step') != V6_CERTIFY_EXPECTED_STEPS or checkpoint_integrity.get('world_size') != V6_WORLD_SIZE or (checkpoint_integrity.get('adapter_tensor_count') != EXPECTED_ADAPTER_TENSORS) or (not isinstance(required_files, Mapping)) or (set(required_files) != expected_required_names) or (checkpoint_binding.get('status') != 'passed') or (checkpoint_binding.get('run_id') != V6_CERTIFY_RUN_ID) or (checkpoint_binding.get('global_step') != V6_CERTIFY_EXPECTED_STEPS) or (checkpoint_binding.get('certify_run') is not True):
        raise V6SftError('V6 certify checkpoint recovery chain drifted')
    expected_cold_load = {'schema_version': 'local-wm-v6-certify-cold-load-v1', 'status': 'passed', 'global_step': V6_CERTIFY_EXPECTED_STEPS, 'checkpoint': str(authority['recorded_certify_run_dir'] / 'checkpoints/checkpoint-2'), 'checkpoint_integrity_sha256': _sha256_file(files['certify_checkpoint_integrity']), 'checkpoint_binding_sha256': _sha256_file(files['certify_checkpoint_binding']), 'required_file_sha256': {name: required_files[name]['sha256'] for name in _required_checkpoint_files(V6_WORLD_SIZE)}, 'adapter_tensor_count': EXPECTED_ADAPTER_TENSORS, 'adapter_tensor_parameters': EXPECTED_TRAINABLE_PARAMS, 'adapter_all_tensors_cold_loaded': True, 'optimizer_cold_loaded': True, 'scheduler_cold_loaded': True, 'training_args_cold_loaded': True, 'rng_state_ranks_cold_loaded': list(range(V6_WORLD_SIZE))}
    if checkpoint_cold_load != expected_cold_load:
        raise V6SftError('V6 certify checkpoint cold-load evidence drifted')
    summary = completion['summary']
    if summary.get('resume') != {'mode': 'new_certify_run', 'policy': 'certify_checkpoint_forbidden_for_formal_resume', 'resume_from_checkpoint': None} or summary.get('certify_sample_plan', {}).get('path') != str(authority['recorded_log_root'] / 'certify_sample_plan.json'):
        raise V6SftError('certify summary plan/resume contract drifted')
    expected_hashes = {'preflight_sha256': _sha256_file(files['preflight']), 'certify_sample_plan_sha256': _sha256_file(files['certify_sample_plan']), 'certify_log_sha256': launch_events['sha256'], 'gpu_memory_audit_sha256': audit['sha256'], 'monitor_status_sha256': _sha256_file(files['monitor_status']), 'train_summary_sha256': _sha256_file(files['train_summary']), 'run_config_sha256': _sha256_file(files['run_config']), 'certify_checkpoint_integrity_sha256': _sha256_file(files['certify_checkpoint_integrity']), 'certify_checkpoint_binding_sha256': _sha256_file(files['certify_checkpoint_binding']), 'certify_checkpoint_cold_load_sha256': _sha256_file(files['certify_checkpoint_cold_load']), 'final_adapter_integrity_sha256': completion['integrity_sha256'], 'latest_success_pointer_sha256': completion['latest_pointer_sha256'], 'sidecar_sha256': _sha256_file(sidecar), 'focus_manifest_sha256': _sha256_file(manifest), 'trainer_sha256': _sha256_file(trainer), 'runner_sha256': _sha256_file(runner)}
    expected_static = {'schema_version': V6_CERTIFY_RECEIPT_SCHEMA, 'status': 'passed', 'formal_training_started': False, 'run_id': V6_CERTIFY_RUN_ID, 'world_size': V6_WORLD_SIZE, 'physical_gpus': list(range(V6_WORLD_SIZE)), 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'expected_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'final_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'adapter_initialization_mode': V6_ADAPTER_INIT_MODE, 'expanded_lora_modules': EXPECTED_EXPANDED_MODULES, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'focus_loss_active': True, 'loss_ramp': False, 'finite_loss_observed': True, 'simultaneous_memory_floor_mib': V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB, 'no_training_processes_remaining': True, 'certify_checkpoint_forbidden_for_formal_resume': True, 'certification_runs_validation500': False, 'finite_loss_steps': list(range(1, V6_CERTIFY_EXPECTED_STEPS + 1)), 'finite_loss_events': launch_events['finite_loss_events'], **expected_hashes}
    mismatches = {key: {'expected': value, 'actual': receipt.get(key)} for (key, value) in expected_static.items() if receipt.get(key) != value}
    if mismatches:
        raise V6SftError(f'certify receipt artifact binding failed: {mismatches}')
    return {'path': str(authority['recorded_log_root'] / 'certify_receipt.json'), 'sha256': _sha256_file(receipt_path), 'run_id': V6_CERTIFY_RUN_ID, 'final_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'qualifying_snapshot_timestamp': timestamp, 'simultaneous_memory_floor_mib': V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB}

def _read_launch_append(path: Path, start_byte: int, start_line: int, *, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise V6SftError(f'{context} is missing or a symlink: {path}')
    if isinstance(start_byte, bool) or not isinstance(start_byte, int) or start_byte < 0:
        raise V6SftError(f'{context} start byte must be a non-negative integer')
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 0:
        raise V6SftError(f'{context} start line must be a non-negative integer')
    size = path.stat().st_size
    if start_byte > size:
        raise V6SftError(f'{context} start byte exceeds file size')
    with path.open('rb') as handle:
        prefix = handle.read(start_byte)
        if prefix.count(b'\n') != start_line:
            raise V6SftError(f'{context} start byte/line boundary mismatch')
        if start_byte:
            if prefix[-1:] != b'\n':
                raise V6SftError(f'{context} start byte is not on a line boundary')
        handle.seek(start_byte)
        return handle.read()

def _validate_v6_formal_monitor_evidence_legacy(*, train_log: str | Path, train_log_start_byte: int, train_log_start_line: int, gpu_memory_audit: str | Path, gpu_audit_start_byte: int, gpu_audit_start_line: int, monitor_status: str | Path) -> dict[str, Any]:
    train_path = Path(train_log)
    audit_path = Path(gpu_memory_audit)
    monitor_path = Path(monitor_status)
    monitor = _load_json_object(monitor_path, context='formal V6 monitor status')
    train_append = _read_launch_append(train_path, train_log_start_byte, train_log_start_line, context='formal V6 train log')
    audit_append = _read_launch_append(audit_path, gpu_audit_start_byte, gpu_audit_start_line, context='formal V6 GPU audit')
    params_events: list[dict[str, Any]] = []
    loss_events: list[dict[str, Any]] = []
    for raw_line in train_append.decode('utf-8', errors='strict').splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('event') == 'v6_trainable_parameters':
            params_events.append(event)
        elif event.get('event') == 'v6_loss_finite':
            loss_events.append(event)
    params_ok = any((event.get('trainable_parameters') == EXPECTED_TRAINABLE_PARAMS and event.get('expanded_lora_modules') == EXPECTED_EXPANDED_MODULES and (event.get('trainable_tensor_count') == EXPECTED_ADAPTER_TENSORS) and (event.get('target_modules_sha256') == EXPECTED_TARGET_MODULES_SHA256) and (event.get('adapter_initialization_mode') == V6_ADAPTER_INIT_MODE) for event in params_events))
    finite_events: list[dict[str, Any]] = []
    for event in loss_events:
        try:
            loss = float(event.get('loss'))
            step = int(event.get('global_step'))
        except (TypeError, ValueError):
            continue
        if math.isfinite(loss) and step > 0 and (event.get('focus_loss_active') is True) and (event.get('loss_ramp') is False):
            finite_events.append(event)
    if not params_ok or not finite_events:
        raise V6SftError('formal monitor evidence lacks current-launch parameters/finite progress')
    decoded_audit = audit_append.decode('utf-8', errors='strict').splitlines()
    header = ['timestamp', 'gpu_index', 'memory_used_mib', 'utilization_percent', 'phase', 'minimum_active_mib']
    if decoded_audit and decoded_audit[0].split('\t') == header:
        decoded_audit = decoded_audit[1:]
    snapshots: dict[str, dict[str, dict[str, int | str]]] = {}
    for (line_number, line) in enumerate(decoded_audit, start=1):
        columns = line.split('\t')
        if len(columns) != len(header):
            raise V6SftError(f'formal GPU audit has invalid appended row {line_number}')
        row = dict(zip(header, columns))
        try:
            index = int(row['gpu_index'])
            used = int(row['memory_used_mib'])
            utilization = int(row['utilization_percent'])
            minimum = int(row['minimum_active_mib'])
        except ValueError as exc:
            raise V6SftError('formal GPU audit contains non-integer evidence') from exc
        if index not in range(V6_WORLD_SIZE) or used < 0 or utilization not in range(101) or (minimum < V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB) or (row['phase'] not in {'loading', 'active'}) or (not row['timestamp']):
            raise V6SftError('formal GPU audit contains out-of-contract evidence')
        timestamp_rows = snapshots.setdefault(row['timestamp'], {})
        key = str(index)
        if key in timestamp_rows:
            raise V6SftError('formal GPU audit repeats a GPU in one snapshot')
        timestamp_rows[key] = {'used': used, 'utilization': utilization, 'phase': row['phase'], 'minimum': minimum}
    expected_gpu_keys = {str(index) for index in range(V6_WORLD_SIZE)}
    qualifying: list[dict[str, int]] = []
    for rows in snapshots.values():
        if set(rows) != expected_gpu_keys:
            continue
        if all((row['phase'] == 'active' and int(row['used']) >= V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB for row in rows.values())):
            qualifying.append({key: int(rows[key]['used']) for key in sorted(rows)})
    if not qualifying:
        raise V6SftError('formal GPU audit lacks a current-launch simultaneous six-GPU snapshot')
    monitor_memory = monitor.get('simultaneous_memory_used_mib')
    try:
        normalized_monitor_memory = {str(key): int(value) for (key, value) in monitor_memory.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise V6SftError('formal monitor memory snapshot is invalid') from exc
    expected_monitor = {'schema_version': 'local-wm-v6-gpu-monitor-v1', 'status': 'passed', 'kind': 'formal', 'gpu_memory_audit': str(audit_path), 'gpu_memory_audit_sha256': _sha256_file(audit_path), 'parameters_observed': True, 'finite_loss_observed': True, 'simultaneous_memory_floor_observed': True, 'monitor_failed': False, 'launch_log_start_byte': train_log_start_byte, 'launch_log_start_line': train_log_start_line, 'gpu_audit_start_byte': gpu_audit_start_byte, 'gpu_audit_start_line': gpu_audit_start_line}
    mismatches = {key: {'expected': value, 'actual': monitor.get(key)} for (key, value) in expected_monitor.items() if monitor.get(key) != value}
    if mismatches or normalized_monitor_memory not in qualifying:
        raise V6SftError(f'formal monitor status is not backed by artifacts: {mismatches}')
    return {'train_log': str(train_path), 'train_log_sha256': _sha256_file(train_path), 'train_log_start_byte': train_log_start_byte, 'train_log_start_line': train_log_start_line, 'gpu_memory_audit': str(audit_path), 'gpu_memory_audit_sha256': _sha256_file(audit_path), 'gpu_audit_start_byte': gpu_audit_start_byte, 'gpu_audit_start_line': gpu_audit_start_line, 'monitor_status': str(monitor_path), 'monitor_status_sha256': _sha256_file(monitor_path), 'simultaneous_memory_used_mib': normalized_monitor_memory, 'trainable_parameters_observed': EXPECTED_TRAINABLE_PARAMS, 'finite_loss_steps': sorted({int(event['global_step']) for event in finite_events}), 'focus_loss_active': True, 'loss_ramp': False}

def validate_v6_formal_monitor_evidence(*, train_log: str | Path, train_log_start_byte: int, train_log_start_line: int, gpu_memory_audit: str | Path, gpu_audit_start_byte: int, gpu_audit_start_line: int, monitor_status: str | Path, training_pid: int, training_pgid: int, qualifying_snapshot_timestamp: str, gpu_memory_audit_recorded: str | Path | None=None) -> dict[str, Any]:
    train_path = Path(train_log)
    audit_path = Path(gpu_memory_audit)
    monitor_path = Path(monitor_status)
    events = _parse_v6_launch_events(train_path, start_byte=train_log_start_byte, start_line=train_log_start_line, context='formal V6 train log')
    audit = _parse_gpu_audit(audit_path, start_byte=gpu_audit_start_byte, start_line=gpu_audit_start_line, context='formal V6 GPU audit')
    if not isinstance(training_pid, int) or isinstance(training_pid, bool) or training_pid <= 0 or (not isinstance(training_pgid, int)) or isinstance(training_pgid, bool) or (training_pgid <= 0) or (not isinstance(qualifying_snapshot_timestamp, str)) or (qualifying_snapshot_timestamp not in audit['qualifying_snapshots']):
        raise V6SftError('formal launch PID/PGID/snapshot identity is invalid')
    memory = audit['qualifying_snapshots'][qualifying_snapshot_timestamp]
    monitor = _load_json_object(monitor_path, context='formal V6 monitor status')
    expected_monitor = {'schema_version': 'local-wm-v6-gpu-monitor-v2', 'status': 'passed', 'kind': 'formal', 'training_pid': training_pid, 'training_pgid': training_pgid, 'gpu_memory_audit': str(audit_path if gpu_memory_audit_recorded is None else gpu_memory_audit_recorded), 'gpu_memory_audit_sha256': audit['sha256'], 'simultaneous_memory_used_mib': memory, 'qualifying_snapshot_timestamp': qualifying_snapshot_timestamp, 'parameters_observed': True, 'finite_loss_observed': True, 'simultaneous_memory_floor_observed': True, 'monitor_failed': False, 'launch_log_start_byte': train_log_start_byte, 'launch_log_start_line': train_log_start_line, 'gpu_audit_start_byte': gpu_audit_start_byte, 'gpu_audit_start_line': gpu_audit_start_line}
    if monitor != expected_monitor:
        raise V6SftError('formal monitor status is not exact current-launch evidence')
    return {'train_log': str(train_path), 'train_log_sha256': events['sha256'], 'train_log_start_byte': train_log_start_byte, 'train_log_start_line': train_log_start_line, 'gpu_memory_audit': str(audit_path), 'gpu_memory_audit_sha256': audit['sha256'], 'gpu_audit_start_byte': gpu_audit_start_byte, 'gpu_audit_start_line': gpu_audit_start_line, 'monitor_status': str(monitor_path), 'monitor_status_sha256': _sha256_file(monitor_path), 'training_pid': training_pid, 'training_pgid': training_pgid, 'qualifying_snapshot_timestamp': qualifying_snapshot_timestamp, 'simultaneous_memory_used_mib': memory, 'trainable_parameters_observed': EXPECTED_TRAINABLE_PARAMS, 'trainable_tensor_count_observed': EXPECTED_ADAPTER_TENSORS, 'finite_loss_steps': events['finite_loss_steps'], 'focus_loss_active': True, 'loss_ramp': False}

def _validate_v6_formal_receipt_legacy(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path)
    receipt = _load_json_object(receipt_path, context='formal V6 receipt')
    runner_path = Path(__file__).resolve().parents[3] / v6_runner_relative_path()
    trainer_path = Path(__file__).resolve()
    sidecar_path = Path(CANONICAL_FOCUS_SIDECAR)
    manifest_path = sidecar_path.parent / 'manifest.json'
    referenced = {'train_log': Path(str(receipt.get('train_log') or '')), 'gpu_memory_audit': Path(str(receipt.get('gpu_memory_audit') or '')), 'monitor_status': Path(str(receipt.get('monitor_status') or '')), 'train_summary': Path(str(receipt.get('train_summary') or '')), 'run_config': Path(str(receipt.get('run_config') or '')), 'stable_marker': Path(str(receipt.get('stable_marker') or '')), 'certify_receipt': Path(str(receipt.get('certify_receipt') or ''))}
    for (name, candidate) in referenced.items():
        if candidate.is_symlink() or not candidate.is_file():
            raise V6SftError(f'formal receipt {name} is missing or a symlink')
    evidence = validate_v6_formal_monitor_evidence(train_log=referenced['train_log'], train_log_start_byte=receipt.get('train_log_start_byte'), train_log_start_line=receipt.get('train_log_start_line'), gpu_memory_audit=referenced['gpu_memory_audit'], gpu_audit_start_byte=receipt.get('gpu_audit_start_byte'), gpu_audit_start_line=receipt.get('gpu_audit_start_line'), monitor_status=referenced['monitor_status'])
    summary = _load_json_object(referenced['train_summary'], context='formal V6 train summary')
    run_config = _load_json_object(referenced['run_config'], context='formal V6 run_config')
    stable = _load_json_object(referenced['stable_marker'], context='formal V6 stable marker')
    certify_report = validate_v6_certify_receipt(referenced['certify_receipt'])
    _validate_run_config_for_resume(run_config, run_id=V6_FORMAL_RUN_ID)
    manifest = _load_json_object(manifest_path, context='final V20 sidecar manifest')
    launch_argv = stable.get('launch_argv')
    launch_environment = stable.get('launch_environment')
    if not isinstance(launch_argv, list) or not isinstance(launch_environment, Mapping):
        raise V6SftError('stable marker lacks canonical launch arguments/environment')
    if launch_environment.get('PYTHONUNBUFFERED') != '1' or launch_environment.get('TORCH_NCCL_ASYNC_ERROR_HANDLING') != '1':
        raise V6SftError('stable marker launch hash/environment is invalid')
    expected_files = {'train_log_sha256': _sha256_file(referenced['train_log']), 'gpu_memory_audit_sha256': _sha256_file(referenced['gpu_memory_audit']), 'monitor_status_sha256': _sha256_file(referenced['monitor_status']), 'train_summary_sha256': _sha256_file(referenced['train_summary']), 'run_config_sha256': _sha256_file(referenced['run_config']), 'stable_marker_sha256': _sha256_file(referenced['stable_marker']), 'certify_receipt_sha256': _sha256_file(referenced['certify_receipt']), 'trainer_sha256': _sha256_file(trainer_path), 'runner_sha256': _sha256_file(runner_path), 'focus_sidecar_sha256': _sha256_file(sidecar_path), 'focus_manifest_sha256': _sha256_file(manifest_path)}
    expected = {'schema_version': V6_FORMAL_RECEIPT_SCHEMA, 'status': 'passed', 'run_id': V6_FORMAL_RUN_ID, 'formal_training_started': True, 'formal_training_completed': True, 'world_size': V6_WORLD_SIZE, 'physical_gpus': list(range(V6_WORLD_SIZE)), 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'final_global_step': EXPECTED_FINAL_STEP, 'expected_final_step': EXPECTED_FINAL_STEP, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'trainer_path': str(trainer_path), 'runner_path': str(runner_path), 'focus_sidecar': str(sidecar_path), 'focus_manifest': str(manifest_path), 'parameters_observed': True, 'finite_loss_observed': True, 'progress_observed': True, 'focus_loss_active': True, 'loss_ramp': False, 'no_training_processes_remaining': True, **expected_files}
    mismatches = {key: {'expected': value, 'actual': receipt.get(key)} for (key, value) in expected.items() if receipt.get(key) != value}
    if mismatches or receipt.get('simultaneous_memory_used_mib') != evidence['simultaneous_memory_used_mib'] or summary.get('status') != 'passed' or (summary.get('run_id') != V6_FORMAL_RUN_ID) or (summary.get('certify_run') is not False) or (summary.get('formal_training_started') is not True) or (summary.get('final_global_step') != EXPECTED_FINAL_STEP) or (summary.get('expected_final_step') != EXPECTED_FINAL_STEP) or (summary.get('trainable_parameters') != EXPECTED_TRAINABLE_PARAMS) or (summary.get('finite_loss_observed') is not True) or (summary.get('focus_loss_active') is not True) or (summary.get('loss_ramp') is not False) or (summary.get('certify_receipt_authorization') != certify_report) or (stable.get('schema_version') != 'local-wm-v6-formal-stable-v3') or (stable.get('status') != 'stable') or (stable.get('run_id') != V6_FORMAL_RUN_ID) or (stable.get('runner_path') != str(authority['recorded_runner_path'])) or (stable.get('trainer_path') != str(authority['recorded_trainer_path'])) or (stable.get('focus_sidecar') != str(authority['recorded_focus_sidecar'])) or (stable.get('focus_manifest') != str(authority['recorded_focus_manifest'])) or (stable.get('certify_receipt') != str(recorded_paths['certify_receipt'])) or (stable.get('train_log') != str(recorded_paths['train_log'])) or (stable.get('resume_checkpoint') != receipt.get('resume_checkpoint')) or (stable.get('training_pid') != receipt.get('training_pid')) or (stable.get('train_log_start_byte') != receipt.get('train_log_start_byte')) or (stable.get('train_log_start_line') != receipt.get('train_log_start_line')) or (stable.get('gpu_audit_start_byte') != receipt.get('gpu_audit_start_byte')) or (stable.get('gpu_audit_start_line') != receipt.get('gpu_audit_start_line')) or (receipt.get('training_pid') != _load_json_object(referenced['monitor_status'], context='formal monitor status').get('training_pid')):
        raise V6SftError(f'formal V6 receipt validation failed: {mismatches}')
    try:
        if int(receipt.get('training_pid')) <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise V6SftError('formal receipt training PID is invalid') from exc
    resume = receipt.get('resume_checkpoint')
    binding_sha = receipt.get('resume_checkpoint_binding_sha256')
    if resume is None:
        if binding_sha is not None:
            raise V6SftError('new formal run receipt unexpectedly binds a checkpoint')
    else:
        checkpoint = Path(str(resume))
        binding = checkpoint / 'v6_checkpoint_binding.json'
        if checkpoint.is_symlink() or not checkpoint.is_dir() or checkpoint.resolve().parent != (CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID / 'checkpoints').resolve() or binding.is_symlink() or (not binding.is_file()):
            raise V6SftError('formal receipt resume checkpoint binding is invalid')
    return {'path': str(receipt_path), 'sha256': _sha256_file(receipt_path), 'run_id': V6_FORMAL_RUN_ID, 'final_global_step': EXPECTED_FINAL_STEP, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'monitor_evidence': evidence}

def _formal_authority(test_authority: Mapping[str, Any] | None) -> dict[str, Any]:
    certify = _certify_authority(test_authority.get('certify_authority') if test_authority is not None else None)
    if test_authority is None:
        repo_root = Path(__file__).resolve().parents[3]
        return {**certify, 'formal_run_dir': repo_root / CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID, 'recorded_formal_run_dir': CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID, 'python_bin': repo_root / 'envs/qwen3.5/bin/python', 'recorded_python_bin': repo_root / 'envs/qwen3.5/bin/python'}
    required = {'log_root', 'formal_run_dir', 'python_bin', 'certify_authority'}
    if not required.issubset(test_authority):
        raise V6SftError('_test_authority lacks isolated formal authority paths')
    authority = {**certify, 'log_root': Path(test_authority['log_root']), 'formal_run_dir': Path(test_authority['formal_run_dir']), 'python_bin': Path(test_authority['python_bin']), 'test': True}
    authority['recorded_log_root'] = Path(test_authority.get('recorded_log_root', test_authority['log_root']))
    authority['recorded_formal_run_dir'] = Path(test_authority.get('recorded_formal_run_dir', test_authority['formal_run_dir']))
    authority['recorded_python_bin'] = Path(test_authority.get('recorded_python_bin', test_authority['python_bin']))
    return authority

def _validate_resume_semantics(*, resume: object, run_dir: Path) -> dict[str, Any]:
    if not isinstance(resume, Mapping):
        raise V6SftError('formal completion lacks resume semantics')
    if resume.get('mode') == 'new_run':
        expected = {'mode': 'new_run', 'policy': 'same_run_v6_checkpoint_only', 'resume_from_checkpoint': None}
        if dict(resume) != expected:
            raise V6SftError('fresh formal resume semantics drifted')
        return expected
    checkpoint_raw = resume.get('resume_from_checkpoint')
    if not isinstance(checkpoint_raw, str):
        raise V6SftError('formal resume checkpoint path is invalid')
    checkpoint = Path(checkpoint_raw)
    if checkpoint.is_symlink() or not checkpoint.is_dir() or checkpoint.resolve(strict=True).parent != run_dir.resolve(strict=True) / 'checkpoints':
        raise V6SftError('formal resume checkpoint is outside canonical run')
    args = argparse.Namespace(resume_from_checkpoint=checkpoint, run_id=V6_FORMAL_RUN_ID, output_dir=run_dir.parent)
    expected = validate_v6_resume_checkpoint(args)
    if dict(resume) != expected:
        raise V6SftError('formal resume semantics differ from checkpoint validation')
    return expected

def validate_v6_formal_receipt(path: str | Path, *, _test_authority: Mapping[str, Any] | None=None) -> dict[str, Any]:
    authority = _formal_authority(_test_authority)
    log_root = authority['log_root']
    receipt_path = _exact_regular_file(str(path), log_root / 'formal_receipt.json', context='formal receipt', recorded=authority['recorded_log_root'] / 'formal_receipt.json')
    receipt = _load_json_object(receipt_path, context='formal V6 receipt')
    try:
        pid = int(receipt.get('training_pid'))
        pgid = int(receipt.get('training_pgid'))
        train_byte = int(receipt.get('train_log_start_byte'))
        train_line = int(receipt.get('train_log_start_line'))
        gpu_byte = int(receipt.get('gpu_audit_start_byte'))
        gpu_line = int(receipt.get('gpu_audit_start_line'))
    except (TypeError, ValueError) as exc:
        raise V6SftError('formal receipt launch identity is invalid') from exc
    if min(pid, pgid) <= 0 or min(train_byte, train_line, gpu_byte, gpu_line) < 0:
        raise V6SftError('formal receipt launch identity is invalid')
    timestamp = receipt.get('qualifying_snapshot_timestamp')
    if not isinstance(timestamp, str) or not timestamp:
        raise V6SftError('formal receipt qualifying timestamp is invalid')
    run_dir = authority['formal_run_dir']
    expected_paths = {'train_log': log_root / 'train.log', 'gpu_memory_audit': log_root / 'gpu_memory.tsv', 'monitor_status': log_root / f'formal_monitor_status.{pid}.json', 'train_summary': run_dir / 'train_summary.json', 'run_config': run_dir / 'run_config.json', 'stable_marker': log_root / 'stable.json', 'completion_ready': log_root / 'completion_ready.json', 'certify_receipt': log_root / 'certify_receipt.json'}
    recorded_paths = {'train_log': authority['recorded_log_root'] / 'train.log', 'gpu_memory_audit': authority['recorded_log_root'] / 'gpu_memory.tsv', 'monitor_status': authority['recorded_log_root'] / f'formal_monitor_status.{pid}.json', 'train_summary': authority['recorded_formal_run_dir'] / 'train_summary.json', 'run_config': authority['recorded_formal_run_dir'] / 'run_config.json', 'stable_marker': authority['recorded_log_root'] / 'stable.json', 'completion_ready': authority['recorded_log_root'] / 'completion_ready.json', 'certify_receipt': authority['recorded_log_root'] / 'certify_receipt.json'}
    files = {name: _exact_regular_file(receipt.get(name), expected, context=f'formal {name}', recorded=recorded_paths[name]) for (name, expected) in expected_paths.items()}
    trainer = _exact_regular_file(receipt.get('trainer_path'), authority['trainer_path'], context='formal trainer', recorded=authority['recorded_trainer_path'])
    runner = _exact_regular_file(receipt.get('runner_path'), authority['runner_path'], context='formal runner', recorded=authority['recorded_runner_path'])
    sidecar = _exact_regular_file(receipt.get('focus_sidecar'), authority['focus_sidecar'], context='formal sidecar', recorded=authority['recorded_focus_sidecar'])
    manifest = _exact_regular_file(receipt.get('focus_manifest'), authority['focus_manifest'], context='formal manifest', recorded=authority['recorded_focus_manifest'])
    certify_report = validate_v6_certify_receipt(recorded_paths['certify_receipt'], _test_authority=_test_authority['certify_authority'] if _test_authority is not None else None)
    evidence = validate_v6_formal_monitor_evidence(train_log=files['train_log'], train_log_start_byte=train_byte, train_log_start_line=train_line, gpu_memory_audit=files['gpu_memory_audit'], gpu_audit_start_byte=gpu_byte, gpu_audit_start_line=gpu_line, monitor_status=files['monitor_status'], training_pid=pid, training_pgid=pgid, qualifying_snapshot_timestamp=timestamp, gpu_memory_audit_recorded=recorded_paths['gpu_memory_audit'])
    completion = _validate_final_adapter_chain(run_dir=run_dir, recorded_run_dir=authority['recorded_formal_run_dir'], summary_path=files['train_summary'], run_config_path=files['run_config'], run_id=V6_FORMAL_RUN_ID, certify_run=False, expected_step=EXPECTED_FINAL_STEP, data_authority={'train_jsonl': authority['recorded_train_jsonl'], 'train_file': authority['train_jsonl'], 'focus_sidecar': authority['recorded_focus_sidecar'], 'focus_sidecar_file': sidecar, 'focus_manifest': authority['recorded_focus_manifest'], 'focus_manifest_file': manifest, 'rows': EXPECTED_TRAINING_ROWS, 'output_root': authority['recorded_formal_run_dir'].parent, 'certify_receipt': recorded_paths['certify_receipt']})
    run_config = completion['run_config']
    _validate_run_config_for_resume(run_config, run_id=V6_FORMAL_RUN_ID, output_root=authority['recorded_formal_run_dir'].parent, certify_receipt=recorded_paths['certify_receipt'])
    summary = completion['summary']
    if summary.get('certify_receipt_authorization') != certify_report:
        raise V6SftError('formal summary certify authorization drifted')
    resume = _validate_resume_semantics(resume=summary.get('resume'), run_dir=run_dir)
    if receipt.get('resume') != resume:
        raise V6SftError('formal receipt resume semantics differ from summary')
    stable = _load_json_object(files['stable_marker'], context='formal stable marker')
    launch_argv = stable.get('launch_argv')
    launch_environment = stable.get('launch_environment')
    if stable.get('schema_version') != 'local-wm-v6-formal-stable-v6' or stable.get('status') != 'stable' or stable.get('run_id') != V6_FORMAL_RUN_ID or (stable.get('training_pid') != pid) or (stable.get('training_pgid') != pgid) or (stable.get('train_log_start_byte') != train_byte) or (stable.get('train_log_start_line') != train_line) or (stable.get('gpu_audit_start_byte') != gpu_byte) or (stable.get('gpu_audit_start_line') != gpu_line) or (stable.get('qualifying_snapshot_timestamp') != timestamp) or (stable.get('simultaneous_memory_used_mib') != evidence['simultaneous_memory_used_mib']) or (stable.get('resume') != resume) or (stable.get('expected_final_step') != EXPECTED_FINAL_STEP) or (stable.get('expected_trainable_parameters') != EXPECTED_TRAINABLE_PARAMS) or (stable.get('expected_trainable_tensor_count') != EXPECTED_ADAPTER_TENSORS) or (not isinstance(launch_argv, list)) or (not isinstance(launch_environment, Mapping)) or any((launch_environment.get(key) != value for (key, value) in {'CUDA_VISIBLE_DEVICES': V6_CUDA_VISIBLE_DEVICES, 'OMP_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': 'src', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'V6_RUN_ID': V6_FORMAL_RUN_ID, 'V6_MANAGED_TOKEN_FILE': str(CANONICAL_V6_LOG_ROOT / 'managed_process.token')}.items())):
        raise V6SftError('formal stable marker is split-brain or incomplete')
    if len(launch_argv) < 10 or Path(str(launch_argv[0])).resolve(strict=True) != authority['python_bin'].resolve(strict=True) or launch_argv[1:9] != ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=6', '-m', 'mkd2world.training.next_tsm_sft', 'train', '--manifest'] or (launch_argv[9] != str(CANONICAL_V6_MANIFEST)):
        raise V6SftError('formal stable launch command prefix drifted')
    if resume['mode'] == 'new_run':
        if '--resume-from-checkpoint' in launch_argv:
            raise V6SftError('fresh stable marker unexpectedly resumes')
    elif '--resume-from-checkpoint' not in launch_argv or Path(launch_argv[launch_argv.index('--resume-from-checkpoint') + 1]).resolve(strict=True) != Path(resume['resume_from_checkpoint']).resolve(strict=True):
        raise V6SftError('stable marker resumed a different checkpoint')
    hashes = {'train_log_sha256': _sha256_file(files['train_log']), 'gpu_memory_audit_sha256': _sha256_file(files['gpu_memory_audit']), 'monitor_status_sha256': _sha256_file(files['monitor_status']), 'train_summary_sha256': _sha256_file(files['train_summary']), 'run_config_sha256': _sha256_file(files['run_config']), 'stable_marker_sha256': _sha256_file(files['stable_marker']), 'certify_receipt_sha256': _sha256_file(files['certify_receipt']), 'trainer_sha256': _sha256_file(trainer), 'runner_sha256': _sha256_file(runner), 'focus_sidecar_sha256': _sha256_file(sidecar), 'focus_manifest_sha256': _sha256_file(manifest), 'final_adapter_integrity_sha256': completion['integrity_sha256'], 'latest_success_pointer_sha256': completion['latest_pointer_sha256']}
    for (key, value) in hashes.items():
        if stable.get(key) not in (None, value) or receipt.get(key) != value:
            raise V6SftError(f'formal artifact hash binding drifted: {key}')
    ready = _load_json_object(files['completion_ready'], context='formal completion-ready')
    ready_expected = {'schema_version': 'local-wm-v6-completion-ready-v1', 'status': 'passed', 'run_id': V6_FORMAL_RUN_ID, 'training_pid': pid, 'training_pgid': pgid, 'exit_code': 0, 'train_log_start_byte': train_byte, 'train_log_start_line': train_line, 'gpu_audit_start_byte': gpu_byte, 'gpu_audit_start_line': gpu_line, 'qualifying_snapshot_timestamp': timestamp, 'simultaneous_memory_used_mib': evidence['simultaneous_memory_used_mib'], 'resume': resume, 'no_training_processes_remaining': True, **hashes}
    ready_mismatches = {key: {'expected': value, 'actual': ready.get(key)} for (key, value) in ready_expected.items() if ready.get(key) != value}
    if ready_mismatches:
        raise V6SftError(f'formal completion-ready proof failed: {ready_mismatches}')
    ready_sha = _sha256_file(files['completion_ready'])
    validation_binding: dict[str, Any] = {}
    for step in V6_VALIDATION_STEPS:
        validation = validate_v6_inline_evaluation_receipt(run_dir=run_dir, step=step)
        validation_path = run_dir / 'wm_eval' / f'step_{step:06d}' / 'validation/evaluation_receipt.json'
        validation_binding[str(step)] = {'path': str(validation_path), 'sha256': _sha256_file(validation_path), 'display_name': validation['display_name'], 'completed_judgments': 500, 'strict_complete': True}
    expected_receipt = {'schema_version': V6_FORMAL_RECEIPT_SCHEMA, 'status': 'passed', 'run_id': V6_FORMAL_RUN_ID, 'formal_training_started': True, 'formal_training_completed': True, 'world_size': V6_WORLD_SIZE, 'physical_gpus': list(range(V6_WORLD_SIZE)), 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'final_global_step': EXPECTED_FINAL_STEP, 'expected_final_step': EXPECTED_FINAL_STEP, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'training_pid': pid, 'training_pgid': pgid, 'train_log_start_byte': train_byte, 'train_log_start_line': train_line, 'gpu_audit_start_byte': gpu_byte, 'gpu_audit_start_line': gpu_line, 'qualifying_snapshot_timestamp': timestamp, 'simultaneous_memory_used_mib': evidence['simultaneous_memory_used_mib'], 'resume': resume, 'completion_ready_sha256': ready_sha, 'parameters_observed': True, 'finite_loss_observed': True, 'progress_observed': True, 'focus_loss_active': True, 'loss_ramp': False, 'no_training_processes_remaining': True, **hashes}
    expected_receipt.update({'validation_steps': list(V6_VALIDATION_STEPS), 'validation_receipts': validation_binding})
    receipt_mismatches = {key: {'expected': value, 'actual': receipt.get(key)} for (key, value) in expected_receipt.items() if receipt.get(key) != value}
    if receipt_mismatches:
        raise V6SftError(f'formal receipt contract failed: {receipt_mismatches}')
    return {'path': str(authority['recorded_log_root'] / 'formal_receipt.json'), 'sha256': _sha256_file(receipt_path), 'run_id': V6_FORMAL_RUN_ID, 'final_global_step': EXPECTED_FINAL_STEP, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'completion_ready_sha256': ready_sha, 'monitor_evidence': evidence}

def validate_v6_training_args(args: argparse.Namespace, *, require_ddp: bool) -> None:
    configure_v6_loss_profile(args.loss_profile)
    expected_granularities = V6_LOSS_PROFILES[args.loss_profile]
    actual_granularities = (args.base_loss_granularity, args.focus_loss_granularity)
    if actual_granularities != expected_granularities:
        raise V6SftError(f'--loss-profile disagrees with explicit base/focus granularities: profile={args.loss_profile!r} expects={expected_granularities!r} actual={actual_granularities!r}')
    'Reject drift from the bare-base, six-GPU V6-FullSFT recipe.'
    if args.train_jsonl is None:
        raise V6SftError('--train-jsonl is required')
    train_path = _require_lexical_path(args.train_jsonl, CANONICAL_TRAIN_JSONL, context='--train-jsonl')
    if train_path.is_symlink() or not train_path.is_file():
        raise V6SftError('canonical --train-jsonl is missing or a symlink')
    if args.focus_sidecar is None:
        raise V6SftError('--focus-sidecar is required')
    sidecar_path = _require_lexical_path(args.focus_sidecar, CANONICAL_FOCUS_SIDECAR, context='--focus-sidecar')
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise V6SftError('canonical --focus-sidecar is missing or a symlink')
    _require_lexical_path(args.base_model, CANONICAL_BASE_MODEL, context='--base-model')
    init_adapter = _require_lexical_path(args.init_adapter, CANONICAL_INIT_ADAPTER, context='V6 --init-adapter')
    if init_adapter.is_symlink() or not init_adapter.is_dir():
        raise V6SftError('canonical V3-Cur-1200 --init-adapter is unavailable')
    if args.output_dir is None:
        raise V6SftError('--output-dir is required')
    expected_output_root = CANONICAL_CERTIFY_OUTPUT_ROOT if args.certify_run else CANONICAL_OUTPUT_ROOT
    output_path = _require_lexical_path(args.output_dir, expected_output_root, context='--output-dir')
    if output_path.is_symlink() or output_path.parent.is_symlink():
        raise V6SftError('--output-dir and its parent may not be symlinks')
    frozen = {'base_model': CANONICAL_BASE_MODEL, 'prompt_key': 'localwm_next', 'per_device_train_batch_size': V6_PER_DEVICE_TRAIN_BATCH_SIZE, 'gradient_accumulation_steps': V6_GRADIENT_ACCUMULATION_STEPS, 'learning_rate': 2e-05, 'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.03, 'weight_decay': 0.01, 'seed': 0, 'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'target_modules': 'qwen35_expanded', 'max_length': 20000, 'max_output_tokens': 6000, 'image_min_pixels': 200704, 'image_max_pixels': 802816, 'min_supervised_tokens': 16, 'assistant_loss_scope': 'full', 'gradient_checkpointing': True, 'bf16': True, 'fp16': False, 'device_mode': 'ddp', 'max_grad_norm': 1.0, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-08, 'logging_steps': 1, 'save_steps': V6_SAVE_STEPS, 'save_total_limit': None}
    for (name, expected) in frozen.items():
        _require_equal(name, getattr(args, name), expected)
    parity_report = validate_v6_v5_parity_contract(args)
    expected_run_id = V6_CERTIFY_RUN_ID if args.certify_run else V6_FORMAL_RUN_ID
    _require_equal('run_id', args.run_id, expected_run_id)
    _require_equal('num_train_epochs', args.num_train_epochs, 1.0 if args.certify_run else 2.0)
    if args.eval_jsonl is not None or args.max_train_samples is not None:
        raise V6SftError('V6-FullSFT forbids eval subsampling and train subsampling')
    if args.skip_missing_images or args.skip_overlong_samples:
        raise V6SftError('V6-FullSFT forbids silent image/length skipping')
    if args.dataset_loading_mode != 'direct':
        raise V6SftError('V6-FullSFT requires direct strict dataset loading')
    if not args.certify_run:
        _require_lexical_path(args.wm_eval_jsonl, CANONICAL_VALIDATION_JSONL, context='V6 --wm-eval-jsonl')
        if args.wm_eval_max_samples != 500:
            raise V6SftError('V6 formal training requires Validation500')
        if args.wm_eval_workers_per_gpu != 3:
            raise V6SftError('V6 formal training requires 3 validation workers per GPU')
        _require_lexical_path(args.wm_eval_embedding_model, CANONICAL_EMBEDDING_MODEL, context='V6 --wm-eval-embedding-model')
        if args.wm_eval_embedding_device != 'cuda:3':
            raise V6SftError('V6 formal training requires WM EleF1 on cuda:3')
        if args.wm_eval_stop_after_bad_checkpoints != 0:
            raise V6SftError('V6 validation health may not early-stop formal training')
        _require_lexical_path(args.transition_cache_root, CANONICAL_TRANSITION_CACHE_ROOT, context='V6 --transition-cache-root')
        _require_lexical_path(args.transition_output_root, CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID / 'transition_metrics', context='V6 --transition-output-root')
    if args.per_device_train_batch_size * V6_WORLD_SIZE * args.gradient_accumulation_steps != V6_GLOBAL_BATCH_SIZE:
        raise V6SftError(f'V6-FullSFT global batch size must be {V6_GLOBAL_BATCH_SIZE}')
    if args.certify_run:
        if args.certify_sample_plan is None:
            raise V6SftError('--certify-run requires --certify-sample-plan')
        if args.resume_from_checkpoint is not None:
            raise V6SftError('V6 certify may not resume from any checkpoint')
        if args.certify_receipt is not None:
            raise V6SftError('V6 certify may not consume its own certify receipt')
        _require_lexical_path(args.wm_eval_jsonl, CANONICAL_VALIDATION_JSONL, context='V6 certify --wm-eval-jsonl')
        if args.wm_eval_max_samples != 500:
            raise V6SftError('V6 certify must retain the V5 wm_eval_max_samples argument; certify validation_steps remains empty')
        if args.wm_eval_workers_per_gpu != V6_VALIDATION_WORKERS_PER_GPU:
            raise V6SftError('V6 certify must retain the V5 validation worker argument')
        _require_lexical_path(args.wm_eval_embedding_model, CANONICAL_EMBEDDING_MODEL, context='V6 certify --wm-eval-embedding-model')
        if args.wm_eval_embedding_device != 'cuda:3':
            raise V6SftError('V6 certify must retain the V5 EleF1 device argument')
    elif args.certify_sample_plan is not None:
        raise V6SftError('--certify-sample-plan is valid only with --certify-run')
    if args.write_certify_sample_plan is not None and (not args.dry_run or args.certify_run or args.resume_validation_only):
        raise V6SftError('--write-certify-sample-plan is valid only for a formal CPU dry-run')
    if args.resume_validation_only:
        if args.resume_from_checkpoint is None or args.dry_run or args.certify_run:
            raise V6SftError('--resume-validation-only requires one formal resume checkpoint')
    if not args.certify_run:
        validate_v6_resume_checkpoint(args)
    if not args.dry_run and (not args.resume_validation_only) and (not args.certify_run):
        if args.certify_receipt is None:
            raise V6SftError('formal V6-FullSFT requires --certify-receipt')
        validate_v6_certify_receipt(args.certify_receipt)
    if require_ddp:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != V6_CUDA_VISIBLE_DEVICES:
            raise V6SftError(f'formal V6-FullSFT requires CUDA_VISIBLE_DEVICES={V6_CUDA_VISIBLE_DEVICES}')
        if int(os.environ.get('WORLD_SIZE', '1')) != V6_WORLD_SIZE:
            raise V6SftError(f'formal V6-FullSFT requires exactly {V6_WORLD_SIZE} torchrun processes')
        if os.environ.get('V6_RUN_ID') != V6_FORMAL_RUN_ID:
            raise V6SftError('DDP V6 launch requires the manifest V6_RUN_ID')
        if not args.certify_run:
            managed_token = os.environ.get('V6_MANAGED_PROCESS_TOKEN', '')
            if len(managed_token) < 49 or not managed_token.startswith(f'{V6_FORMAL_RUN_ID}:'):
                raise V6SftError('formal V6 inline validation requires a unique V6_MANAGED_PROCESS_TOKEN')
            _require_lexical_path(os.environ.get('V6_MANAGED_TOKEN_FILE'), CANONICAL_V6_LOG_ROOT / 'managed_process.token', context='V6_MANAGED_TOKEN_FILE')
    args._v6_v5_parity_report = parity_report

def build_v6_arg_parser() -> argparse.ArgumentParser:
    from mkd2world.training import sft as sft_module
    parser = sft_module.build_arg_parser()
    parser.description = 'Run strict Local WM V6 V4-EE plus page-margin FullSFT.'
    parser.add_argument('--loss-profile', choices=tuple(V6_LOSS_PROFILES), default=V6_LOSS_PROFILE)
    parser.add_argument('--base-loss-granularity', choices=V6_LOSS_GRANULARITIES, default=V6_BASE_LOSS_GRANULARITY)
    parser.add_argument('--focus-loss-granularity', choices=V6_LOSS_GRANULARITIES, default=V6_FOCUS_LOSS_GRANULARITY)
    parser.add_argument('--transition-cache-root', type=sft_module.optional_path)
    parser.add_argument('--transition-output-root', type=sft_module.optional_path)
    parser.add_argument('--focus-sidecar', type=sft_module.optional_path, required=True)
    parser.add_argument('--certify-run', action='store_true')
    parser.add_argument('--certify-sample-plan', type=sft_module.optional_path)
    parser.add_argument('--write-certify-sample-plan', type=sft_module.optional_path)
    parser.add_argument('--certify-receipt', type=sft_module.optional_path)
    parser.add_argument('--resume-validation-only', action='store_true')
    parser.set_defaults(base_model=CANONICAL_BASE_MODEL, train_jsonl=CANONICAL_TRAIN_JSONL, prompt_key='localwm_next', init_adapter=CANONICAL_INIT_ADAPTER, output_dir=CANONICAL_OUTPUT_ROOT, run_id=V6_FORMAL_RUN_ID, per_device_train_batch_size=V6_PER_DEVICE_TRAIN_BATCH_SIZE, gradient_accumulation_steps=V6_GRADIENT_ACCUMULATION_STEPS, num_train_epochs=2.0, learning_rate=2e-05, lr_scheduler_type='cosine', warmup_ratio=0.03, weight_decay=0.01, seed=0, lora_r=32, lora_alpha=64, lora_dropout=0.05, target_modules='qwen35_expanded', max_length=20000, max_output_tokens=6000, save_steps=V6_SAVE_STEPS, dataset_loading_mode='direct', skip_missing_images=False, skip_overlong_samples=False, device_mode='ddp', wm_eval_max_samples=500)
    return parser

def build_v6_args_from_manifest(command: str, manifest_path: str | Path, *, resume_from_checkpoint: str | Path | None=None) -> argparse.Namespace:
    if command not in {'preflight', 'certify', 'train'}:
        raise V6SftError(f'unsupported manifest trainer command: {command}')
    manifest = validate_v6_manifest(manifest_path)
    v5_run_config = _load_json_object(CANONICAL_V5_RUN_CONFIG, context='frozen V5 formal run_config')
    v5_args = v5_run_config.get('args')
    if not isinstance(v5_args, Mapping):
        raise V6SftError('frozen V5 run_config args are missing')
    parser = build_v6_arg_parser()
    args = parser.parse_args(['--focus-sidecar', str(manifest['data']['focus_sidecar'])])
    for name in vars(args):
        if name in v5_args:
            setattr(args, name, v5_args[name])
    artifacts = manifest['artifacts']
    data = manifest['data']
    initialization = manifest['initialization']
    args.base_model = manifest['model']['base_model']
    args.init_adapter = initialization['init_adapter']
    args.train_jsonl = data['train_jsonl']
    args.focus_sidecar = data['focus_sidecar']
    args.prompt_key = data['prompt_key']
    args.wm_eval_jsonl = data['validation_jsonl']
    args.transition_cache_root = data['transition_cache_root']
    args.loss_profile = 'ee'
    args.base_loss_granularity = 'element'
    args.focus_loss_granularity = 'element'
    args.resume_validation_only = False
    args.resume_from_checkpoint = str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
    args.write_certify_sample_plan = None
    args.certify_sample_plan = None
    args.certify_receipt = None
    args.certify_run = command == 'certify'
    args.dry_run = command == 'preflight'
    if command == 'certify':
        args.output_dir = artifacts['certify_root']
        args.run_id = V6_CERTIFY_RUN_ID
        args.num_train_epochs = 1.0
        args.certify_sample_plan = str(Path(artifacts['log_root']) / 'certify_sample_plan.json')
        args.transition_output_root = str(Path(artifacts['certify_root']) / V6_CERTIFY_RUN_ID / 'transition_metrics')
    else:
        args.output_dir = artifacts['output_root']
        args.run_id = manifest['identity']['run_id']
        args.num_train_epochs = float(manifest['training']['epochs'])
        args.transition_output_root = str(Path(artifacts['run_dir']) / 'transition_metrics')
        if command == 'preflight':
            args.write_certify_sample_plan = str(Path(artifacts['log_root']) / 'certify_sample_plan.json')
        else:
            args.certify_receipt = str(Path(artifacts['log_root']) / 'certify_receipt.json')
    validate_v6_training_args(args, require_ddp=False)
    return args

def build_v6_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Manifest-driven V6-Cur-FullSFT control surface.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('preflight', 'certify', 'train', 'status'):
        subparser = subparsers.add_parser(command)
        subparser.add_argument('--manifest', type=Path, required=True)
        if command == 'train':
            subparser.add_argument('--resume-from-checkpoint', type=Path)
    return parser

def build_v6_status_report(manifest_path: str | Path) -> dict[str, Any]:
    manifest = validate_v6_manifest(manifest_path)
    artifacts = manifest['artifacts']
    log_root = Path(artifacts['log_root'])
    run_dir = Path(artifacts['run_dir'])
    certify_dir = Path(artifacts['certify_root']) / V6_CERTIFY_RUN_ID
    run_state_path = Path(artifacts.get('run_state', log_root / 'run_state.json'))
    formal_receipt_path = Path(artifacts['formal_receipt'])
    certify_receipt_path = log_root / 'certify_receipt.json'

    def json_artifact(path: Path) -> dict[str, Any]:
        if path.is_symlink():
            return {'path': str(path), 'exists': True, 'valid': False, 'error': 'symlink'}
        if not path.is_file():
            return {'path': str(path), 'exists': False, 'valid': False}
        try:
            payload = _load_json_object(path, context=f'status artifact {path.name}')
            return {'path': str(path), 'exists': True, 'valid': True, 'sha256': _sha256_file(path), 'payload': payload}
        except Exception as exc:
            return {'path': str(path), 'exists': True, 'valid': False, 'error': f'{type(exc).__name__}: {exc}'}
    checkpoint_steps: list[int] = []
    checkpoint_root = run_dir / 'checkpoints'
    if checkpoint_root.is_dir() and (not checkpoint_root.is_symlink()):
        for path in checkpoint_root.glob('checkpoint-*'):
            match = re.fullmatch('checkpoint-(\\d+)', path.name)
            if match and path.is_dir() and (not path.is_symlink()):
                checkpoint_steps.append(int(match.group(1)))
    checkpoint_steps.sort()
    formal_artifact = json_artifact(formal_receipt_path)
    completion: dict[str, Any] = {'complete': False, 'deep_valid': False, 'reason': 'formal_receipt_missing'}
    status = 'passed'
    if formal_receipt_path.is_file() and (not formal_receipt_path.is_symlink()):
        try:
            completion = deep_validate_v6_formal_receipt(manifest_path)
        except Exception as exc:
            status = 'invalid'
            completion = {'complete': False, 'deep_valid': False, 'reason': 'formal_receipt_invalid', 'error_type': type(exc).__name__, 'error': str(exc)}
    return {'schema_version': 'mkd2world-v6-status-v1', 'status': status, 'run_id': manifest['identity']['run_id'], 'manifest': {'path': str(CANONICAL_V6_MANIFEST), 'sha256': _sha256_file(CANONICAL_V6_MANIFEST)}, 'run_state': json_artifact(run_state_path), 'certify_receipt': json_artifact(certify_receipt_path), 'formal_receipt': formal_artifact, 'completion': completion, 'formal_run_dir': str(run_dir), 'certify_run_dir': str(certify_dir), 'checkpoint_steps': checkpoint_steps, 'expected_checkpoint_steps': list(v6_checkpoint_steps())}

def deep_validate_v6_formal_receipt(manifest_path: str | Path) -> dict[str, Any]:
    """Public completion gate used by the runner watchdog."""
    manifest = validate_v6_manifest(manifest_path)
    receipt_path = Path(manifest['artifacts']['formal_receipt'])
    report = validate_v6_formal_receipt(receipt_path)
    return {'status': 'passed', 'complete': True, 'deep_valid': True, 'run_id': manifest['identity']['run_id'], 'receipt_path': str(receipt_path), 'receipt_sha256': _sha256_file(receipt_path), 'receipt': report}

def deep_validate_v6_certify_receipt(manifest_path: str | Path) -> dict[str, Any]:
    manifest = validate_v6_manifest(manifest_path)
    receipt_path = Path(manifest['artifacts'].get('certify_receipt', Path(manifest['artifacts']['log_root']) / 'certify_receipt.json'))
    report = validate_v6_certify_receipt(receipt_path)
    return {'status': 'passed', 'complete': True, 'deep_valid': True, 'run_id': manifest['identity']['run_id'], 'receipt_path': str(receipt_path), 'receipt_sha256': _sha256_file(receipt_path), 'receipt': report}

def _v6_file_window(path: Path) -> tuple[int, int]:
    if path.is_symlink():
        raise V6SftError(f'launch evidence path may not be a symlink: {path}')
    if not path.exists():
        return (0, 0)
    if not path.is_file():
        raise V6SftError(f'launch evidence is not a regular file: {path}')
    payload = path.read_bytes()
    return (len(payload), payload.count(b'\n'))

def _v6_process_run_id(pid: int) -> str | None:
    try:
        environ = Path(f'/proc/{pid}/environ').read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    prefix = b'V6_RUN_ID='
    for item in environ.split(b'\x00'):
        if item.startswith(prefix):
            return item[len(prefix):].decode('utf-8', errors='replace')
    return None

def _v6_certify_token_processes(certify_token: str) -> dict[int, tuple[int, int]]:
    matches: dict[int, tuple[int, int]] = {}
    control_plane_pids = {os.getpid(), os.getppid()}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in control_plane_pids:
            continue
        if _v6_process_run_id(pid) != V6_FORMAL_RUN_ID or _v6_process_token(pid) != certify_token:
            continue
        identity = _v6_process_identity(pid)
        if identity is not None:
            matches[pid] = identity
    return matches

def _terminate_v6_certify_token_processes(certify_token: str) -> None:
    """Precisely terminate only processes matching token, uid and starttime."""
    candidates = _v6_certify_token_processes(certify_token)
    for (pid, expected_identity) in candidates.items():
        if expected_identity[0] != os.getuid():
            raise V6SftError(f'refusing to terminate foreign certify PID {pid}')
        if _v6_process_identity(pid) == expected_identity and _v6_process_run_id(pid) == V6_FORMAL_RUN_ID and (_v6_process_token(pid) == certify_token):
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        remaining = {pid: identity for (pid, identity) in candidates.items() if _v6_process_identity(pid) == identity and _v6_process_run_id(pid) == V6_FORMAL_RUN_ID and (_v6_process_token(pid) == certify_token)}
        if not remaining:
            return
        time.sleep(0.25)
    for (pid, expected_identity) in remaining.items():
        if _v6_process_identity(pid) == expected_identity and _v6_process_run_id(pid) == V6_FORMAL_RUN_ID and (_v6_process_token(pid) == certify_token):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _v6_certify_token_processes(certify_token):
            return
        time.sleep(0.25)
    raise V6SftError('V6 certify token processes remain after precise cleanup')

def _cold_validate_v6_certify_checkpoint(checkpoint_dir: Path, *, integrity_path: Path, binding_path: Path) -> dict[str, Any]:
    integrity = _load_json_object(integrity_path, context='V6 certify checkpoint integrity')
    required_names = _required_checkpoint_files(V6_WORLD_SIZE)
    recorded = integrity.get('required_files')
    if not isinstance(recorded, Mapping) or set(recorded) != set(required_names):
        raise V6SftError('V6 certify checkpoint integrity file set is incomplete')
    required_hashes: dict[str, str] = {}
    for name in required_names:
        artifact = checkpoint_dir / name
        metadata = recorded.get(name)
        if artifact.is_symlink() or not artifact.is_file() or (not isinstance(metadata, Mapping)) or (metadata.get('size_bytes') != artifact.stat().st_size):
            raise V6SftError(f'V6 certify checkpoint artifact failed: {name}')
        required_hashes[name] = str(metadata['sha256'])
    adapter = validate_expanded_adapter_files(checkpoint_dir)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise V6SftError('V6 certify cold load requires safetensors') from exc
    loaded_tensors = 0
    loaded_parameters = 0
    with safe_open(str(checkpoint_dir / 'adapter_model.safetensors'), framework='pt', device='cpu') as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if not bool(torch.isfinite(tensor).all().item()):
                raise V6SftError(f'V6 certify adapter tensor is non-finite: {name}')
            loaded_tensors += 1
            loaded_parameters += int(tensor.numel())
    if loaded_tensors != EXPECTED_ADAPTER_TENSORS or loaded_parameters != EXPECTED_TRAINABLE_PARAMS:
        raise V6SftError('V6 certify cold-loaded adapter graph drifted')
    load_kwargs = {'map_location': 'cpu', 'weights_only': False}
    optimizer = torch.load(checkpoint_dir / 'optimizer.pt', **load_kwargs)
    scheduler = torch.load(checkpoint_dir / 'scheduler.pt', **load_kwargs)
    training_args = torch.load(checkpoint_dir / 'training_args.bin', **load_kwargs)
    rng_states = [torch.load(checkpoint_dir / f'rng_state_{rank}.pth', **load_kwargs) for rank in range(V6_WORLD_SIZE)]
    trainer_state = _load_json_object(checkpoint_dir / 'trainer_state.json', context='V6 certify trainer_state')
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get('state'), Mapping) or (not isinstance(optimizer.get('param_groups'), list)) or (not optimizer['param_groups']) or (not isinstance(scheduler, Mapping)) or (int(scheduler.get('last_epoch', -1)) != V6_CERTIFY_EXPECTED_STEPS) or (int(trainer_state.get('global_step', -1)) != V6_CERTIFY_EXPECTED_STEPS) or any((not isinstance(state, Mapping) for state in rng_states)) or (int(getattr(training_args, 'gradient_accumulation_steps', -1)) != V6_GRADIENT_ACCUMULATION_STEPS) or (int(getattr(training_args, 'per_device_train_batch_size', -1)) != V6_PER_DEVICE_TRAIN_BATCH_SIZE):
        raise V6SftError('V6 certify optimizer/scheduler/state cold load failed')
    report = {'schema_version': 'local-wm-v6-certify-cold-load-v1', 'status': 'passed', 'global_step': V6_CERTIFY_EXPECTED_STEPS, 'checkpoint': str(CANONICAL_CERTIFY_OUTPUT_ROOT / V6_CERTIFY_RUN_ID / 'checkpoints/checkpoint-2'), 'checkpoint_integrity_sha256': _sha256_file(integrity_path), 'checkpoint_binding_sha256': _sha256_file(binding_path), 'required_file_sha256': required_hashes, 'adapter_tensor_count': adapter['tensor_count'], 'adapter_tensor_parameters': adapter['tensor_parameters'], 'adapter_all_tensors_cold_loaded': True, 'optimizer_cold_loaded': True, 'scheduler_cold_loaded': True, 'training_args_cold_loaded': True, 'rng_state_ranks_cold_loaded': list(range(V6_WORLD_SIZE))}
    del optimizer, scheduler, training_args, rng_states
    return report

def _v6_certify_gpu_snapshot(*, certify_token: str) -> tuple[str, list[dict[str, Any]]]:
    gpu_query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows: dict[str, dict[str, Any]] = {}
    uuid_to_index: dict[str, str] = {}
    for raw in gpu_query.stdout.splitlines():
        parts = [part.strip() for part in raw.split(',', 2)]
        if len(parts) != 3 or parts[0] not in {str(index) for index in range(6)}:
            continue
        (index, uuid, utilization) = parts
        uuid_to_index[uuid] = index
        rows[index] = {'gpu_index': int(index), 'memory_used_mib': 0, 'utilization_percent': int(utilization), 'managed_pids': []}
    if set(rows) != {str(index) for index in range(V6_WORLD_SIZE)}:
        raise V6SftError('certify nvidia-smi did not return physical GPUs 0-5')
    process_query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    for raw in process_query.stdout.splitlines():
        parts = [part.strip() for part in raw.split(',', 2)]
        if len(parts) != 3 or parts[0] not in uuid_to_index:
            continue
        try:
            pid = int(parts[1])
            memory_mib = int(parts[2])
        except ValueError:
            continue
        if _v6_process_run_id(pid) != V6_FORMAL_RUN_ID or _v6_process_token(pid) != certify_token:
            continue
        row = rows[uuid_to_index[parts[0]]]
        row['memory_used_mib'] += memory_mib
        row['managed_pids'].append(pid)
    timestamp = datetime.now().astimezone().isoformat(timespec='milliseconds')
    result: list[dict[str, Any]] = []
    for index in range(V6_WORLD_SIZE):
        row = rows[str(index)]
        row['managed_pids'] = sorted(set(row['managed_pids']))
        row['phase'] = 'active' if row['managed_pids'] and row['memory_used_mib'] >= V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB else 'loading'
        row['minimum_active_mib'] = V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB
        result.append(row)
    return (timestamp, result)

def _run_v6_certify_supervisor_once(manifest_path: str | Path, python_bin: str | Path) -> dict[str, Any]:
    manifest = validate_v6_manifest(manifest_path)
    repo_root = Path(__file__).resolve().parents[3]
    log_root = repo_root / Path(manifest['artifacts']['log_root'])
    receipt_path = repo_root / Path(manifest['artifacts'].get('certify_receipt', Path(manifest['artifacts']['log_root']) / 'certify_receipt.json'))
    if receipt_path.is_file() and (not receipt_path.is_symlink()):
        return deep_validate_v6_certify_receipt(manifest_path)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise V6SftError('V6 certify receipt path exists but is not a regular file')
    certify_run_dir = repo_root / CANONICAL_CERTIFY_OUTPUT_ROOT / V6_CERTIFY_RUN_ID
    if certify_run_dir.exists() or certify_run_dir.is_symlink():
        raise V6SftError('incomplete V6 certify run directory exists; refuse implicit overwrite')
    python_path = Path(python_bin)
    if not python_path.is_file():
        raise V6SftError(f'V6 certify Python is unavailable: {python_path}')
    python_path = python_path.resolve(strict=True)
    if not python_path.is_file():
        raise V6SftError(f'V6 certify Python does not resolve to a file: {python_path}')
    log_root.mkdir(parents=True, exist_ok=True)
    certify_log = log_root / 'certify.log'
    gpu_audit = log_root / 'certify_gpu_memory.tsv'
    cold_load_path = log_root / 'certify_checkpoint_cold_load.json'
    for path in (certify_log, gpu_audit):
        if path.is_symlink():
            raise V6SftError(f'V6 certify evidence path is a symlink: {path}')
    if cold_load_path.exists() or cold_load_path.is_symlink():
        raise V6SftError('V6 certify cold-load evidence exists without a valid receipt')
    if not gpu_audit.exists() or gpu_audit.stat().st_size == 0:
        gpu_audit.write_text('timestamp\tgpu_index\tmemory_used_mib\tutilization_percent\tphase\tminimum_active_mib\n', encoding='utf-8')
    audit_header = 'timestamp\tgpu_index\tmemory_used_mib\tutilization_percent\tphase\tminimum_active_mib'
    audit_payload = gpu_audit.read_bytes()
    if not audit_payload.endswith(b'\n') or gpu_audit.read_text(encoding='utf-8').splitlines()[0] != audit_header:
        raise V6SftError('V6 certify GPU audit header drifted')
    if certify_log.exists() and certify_log.stat().st_size:
        if not certify_log.read_bytes().endswith(b'\n'):
            raise V6SftError('V6 certify log is not on an append line boundary')
    (log_byte, log_line) = _v6_file_window(certify_log)
    (audit_byte, audit_line) = _v6_file_window(gpu_audit)
    command = [str(python_path), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=6', '-m', 'mkd2world.training.next_tsm_sft', 'certify', '--manifest', str(CANONICAL_V6_MANIFEST)]
    certify_token = f'{V6_FORMAL_RUN_ID}:certify:{os.urandom(24).hex()}'
    environment = dict(os.environ)
    environment.update({'CUDA_VISIBLE_DEVICES': V6_CUDA_VISIBLE_DEVICES, 'OMP_NUM_THREADS': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': 'src', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'V6_RUN_ID': V6_FORMAL_RUN_ID, 'V6_MANAGED_PROCESS_TOKEN': certify_token})
    write_v6_run_state(status='running', phase='certify_supervisor', global_step=0, max_steps=V6_CERTIFY_EXPECTED_STEPS, retryable=False)
    qualifying: tuple[str, dict[str, int]] | None = None
    peak_managed = {str(index): 0 for index in range(V6_WORLD_SIZE)}
    with certify_log.open('a', encoding='utf-8') as log_handle, gpu_audit.open('a', encoding='utf-8') as audit_handle:
        process = subprocess.Popen(command, cwd=repo_root, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        pid = int(process.pid)
        pgid = int(os.getpgid(pid))
        try:
            while process.poll() is None:
                (timestamp, snapshot) = _v6_certify_gpu_snapshot(certify_token=certify_token)
                memory = {str(row['gpu_index']): int(row['memory_used_mib']) for row in snapshot}
                for row in snapshot:
                    key = str(row['gpu_index'])
                    peak_managed[key] = max(peak_managed[key], memory[key])
                    audit_handle.write('\t'.join([timestamp, key, str(memory[key]), str(row['utilization_percent']), str(row['phase']), str(row['minimum_active_mib'])]) + '\n')
                audit_handle.flush()
                os.fsync(audit_handle.fileno())
                if all((row['phase'] == 'active' and int(row['memory_used_mib']) >= V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB for row in snapshot)):
                    qualifying = qualifying or (timestamp, memory)
                time.sleep(2.0)
            return_code = int(process.wait())
        except BaseException as exc:
            _terminate_v6_certify_token_processes(certify_token)
            write_v6_run_state(status='failed', phase='certify_failed', global_step=0, max_steps=V6_CERTIFY_EXPECTED_STEPS, last_error=f'{type(exc).__name__}: {exc}', retryable=True)
            raise
        finally:
            log_handle.flush()
            os.fsync(log_handle.fileno())
    remaining = _v6_certify_token_processes(certify_token)
    if remaining:
        _terminate_v6_certify_token_processes(certify_token)
        raise V6SftError(f'V6 Tier-2 token processes remained after torchrun exit: {sorted(remaining)}')
    if return_code != 0:
        raise V6SftError(f'V6 Tier-2 torchrun failed with exit code {return_code}')
    if qualifying is None:
        raise V6SftError('V6 Tier-2 lacks one simultaneous six-GPU active snapshot')
    launch_events = _parse_v6_launch_events(certify_log, start_byte=log_byte, start_line=log_line, context='V6 certify log')
    audit = _parse_gpu_audit(gpu_audit, start_byte=audit_byte, start_line=audit_line, context='V6 certify GPU audit')
    (qualifying_timestamp, qualifying_memory) = qualifying
    if audit['qualifying_snapshots'].get(qualifying_timestamp) != qualifying_memory:
        raise V6SftError('V6 Tier-2 qualifying snapshot changed after write')
    monitor_path = log_root / f'certify_monitor_status.{pid}.json'
    monitor = {'schema_version': 'local-wm-v6-gpu-monitor-v2', 'status': 'passed', 'kind': 'certify', 'training_pid': pid, 'training_pgid': pgid, 'gpu_memory_audit': str(CANONICAL_V6_LOG_ROOT / 'certify_gpu_memory.tsv'), 'gpu_memory_audit_sha256': audit['sha256'], 'simultaneous_memory_used_mib': qualifying_memory, 'qualifying_snapshot_timestamp': qualifying_timestamp, 'parameters_observed': True, 'finite_loss_observed': True, 'simultaneous_memory_floor_observed': True, 'monitor_failed': False, 'launch_log_start_byte': log_byte, 'launch_log_start_line': log_line, 'gpu_audit_start_byte': audit_byte, 'gpu_audit_start_line': audit_line}
    _atomic_write_json(monitor_path, monitor)
    authority = _certify_authority(None)
    preflight_path = log_root / 'preflight.json'
    plan_path = log_root / 'certify_sample_plan.json'
    summary_path = certify_run_dir / 'train_summary.json'
    run_config_path = certify_run_dir / 'run_config.json'
    checkpoint_dir = certify_run_dir / 'checkpoints' / 'checkpoint-2'
    checkpoint_integrity = checkpoint_dir / 'checkpoint_integrity.json'
    checkpoint_binding = checkpoint_dir / 'v6_checkpoint_binding.json'
    for required in (preflight_path, plan_path, summary_path, run_config_path, checkpoint_integrity, checkpoint_binding):
        if required.is_symlink() or not required.is_file():
            raise V6SftError(f'V6 certify completion artifact is missing: {required}')
    integrity = _load_json_object(checkpoint_integrity, context='V6 certify checkpoint integrity')
    binding = _load_json_object(checkpoint_binding, context='V6 certify checkpoint binding')
    if integrity.get('status') != 'passed' or integrity.get('global_step') != V6_CERTIFY_EXPECTED_STEPS or integrity.get('world_size') != V6_WORLD_SIZE or (integrity.get('adapter_tensor_count') != EXPECTED_ADAPTER_TENSORS) or (binding.get('status') != 'passed') or (binding.get('run_id') != V6_CERTIFY_RUN_ID) or (binding.get('global_step') != V6_CERTIFY_EXPECTED_STEPS) or (binding.get('certify_run') is not True):
        raise V6SftError('V6 certify checkpoint-2 integrity/binding failed')
    cold_load_report = _cold_validate_v6_certify_checkpoint(checkpoint_dir, integrity_path=checkpoint_integrity, binding_path=checkpoint_binding)
    _atomic_write_json(cold_load_path, cold_load_report)
    preflight = _load_json_object(preflight_path, context='V6 CPU preflight')
    data_report = preflight.get('data') or {}
    completion = _validate_final_adapter_chain(run_dir=authority['certify_run_dir'], recorded_run_dir=authority['recorded_certify_run_dir'], summary_path=summary_path, run_config_path=run_config_path, run_id=V6_CERTIFY_RUN_ID, certify_run=True, expected_step=V6_CERTIFY_EXPECTED_STEPS, data_authority={'train_jsonl': authority['recorded_train_jsonl'], 'train_file': authority['train_jsonl'], 'focus_sidecar': authority['recorded_focus_sidecar'], 'focus_sidecar_file': authority['focus_sidecar'], 'focus_manifest': authority['recorded_focus_manifest'], 'focus_manifest_file': authority['focus_manifest'], 'rows': data_report.get('rows'), 'output_root': authority['recorded_certify_run_dir'].parent}, certify_plan_path=plan_path, certify_plan_record_path=CANONICAL_V6_LOG_ROOT / 'certify_sample_plan.json')
    recorded_monitor = CANONICAL_V6_LOG_ROOT / monitor_path.name
    receipt = {'schema_version': V6_CERTIFY_RECEIPT_SCHEMA, 'status': 'passed', 'formal_training_started': False, 'run_id': V6_CERTIFY_RUN_ID, 'world_size': V6_WORLD_SIZE, 'physical_gpus': list(range(V6_WORLD_SIZE)), 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'expected_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'final_optimizer_steps': V6_CERTIFY_EXPECTED_STEPS, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'adapter_initialization_mode': V6_ADAPTER_INIT_MODE, 'expanded_lora_modules': EXPECTED_EXPANDED_MODULES, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'focus_loss_active': True, 'loss_ramp': False, 'finite_loss_observed': True, 'simultaneous_memory_floor_mib': V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB, 'no_training_processes_remaining': True, 'certify_checkpoint_forbidden_for_formal_resume': True, 'certification_runs_validation500': False, 'training_pid': pid, 'training_pgid': pgid, 'certify_log_start_byte': log_byte, 'certify_log_start_line': log_line, 'gpu_audit_start_byte': audit_byte, 'gpu_audit_start_line': audit_line, 'qualifying_snapshot_timestamp': qualifying_timestamp, 'simultaneous_memory_used_mib': qualifying_memory, 'peak_managed_memory_mib': peak_managed, 'finite_loss_steps': launch_events['finite_loss_steps'], 'finite_loss_events': launch_events['finite_loss_events'], 'preflight': str(CANONICAL_V6_LOG_ROOT / 'preflight.json'), 'certify_sample_plan': str(CANONICAL_V6_LOG_ROOT / 'certify_sample_plan.json'), 'certify_log': str(CANONICAL_V6_LOG_ROOT / 'certify.log'), 'gpu_memory_audit': str(CANONICAL_V6_LOG_ROOT / 'certify_gpu_memory.tsv'), 'monitor_status': str(recorded_monitor), 'train_summary': str(authority['recorded_certify_run_dir'] / 'train_summary.json'), 'run_config': str(authority['recorded_certify_run_dir'] / 'run_config.json'), 'trainer_path': str(authority['recorded_trainer_path']), 'runner_path': str(authority['recorded_runner_path']), 'certify_checkpoint_integrity': str(authority['recorded_certify_run_dir'] / 'checkpoints/checkpoint-2/checkpoint_integrity.json'), 'certify_checkpoint_binding': str(authority['recorded_certify_run_dir'] / 'checkpoints/checkpoint-2/v6_checkpoint_binding.json'), 'certify_checkpoint_cold_load': str(CANONICAL_V6_LOG_ROOT / 'certify_checkpoint_cold_load.json'), 'preflight_sha256': _sha256_file(preflight_path), 'certify_sample_plan_sha256': _sha256_file(plan_path), 'certify_log_sha256': launch_events['sha256'], 'gpu_memory_audit_sha256': audit['sha256'], 'monitor_status_sha256': _sha256_file(monitor_path), 'train_summary_sha256': _sha256_file(summary_path), 'run_config_sha256': _sha256_file(run_config_path), 'final_adapter_integrity_sha256': completion['integrity_sha256'], 'latest_success_pointer_sha256': completion['latest_pointer_sha256'], 'certify_checkpoint_integrity_sha256': _sha256_file(checkpoint_integrity), 'certify_checkpoint_binding_sha256': _sha256_file(checkpoint_binding), 'certify_checkpoint_cold_load_sha256': _sha256_file(cold_load_path), 'sidecar_sha256': _sha256_file(authority['focus_sidecar']), 'focus_manifest_sha256': _sha256_file(authority['focus_manifest']), 'trainer_sha256': _sha256_file(authority['trainer_path']), 'runner_sha256': _sha256_file(authority['runner_path'])}
    _atomic_write_json(receipt_path, receipt)
    report = deep_validate_v6_certify_receipt(manifest_path)
    write_v6_run_state(status='ready', phase='certification_complete', global_step=0, max_steps=EXPECTED_FINAL_STEP, retryable=False)
    return report

def run_v6_certify_supervisor(manifest_path: str | Path, python_bin: str | Path) -> dict[str, Any]:
    """Public idempotent Tier-2 supervisor with atomic failure state."""
    try:
        return _run_v6_certify_supervisor_once(manifest_path, python_bin)
    except BaseException as exc:
        repo_root = Path(__file__).resolve().parents[3]
        residue = any((path.exists() or path.is_symlink() for path in (repo_root / CANONICAL_CERTIFY_OUTPUT_ROOT / V6_CERTIFY_RUN_ID, repo_root / CANONICAL_V6_LOG_ROOT / 'certify_receipt.json', repo_root / CANONICAL_V6_LOG_ROOT / 'certify_checkpoint_cold_load.json')))
        write_v6_run_state(status='blocked' if residue else 'failed', phase='certify_manual_quarantine' if residue else 'certify_failed_before_artifacts', global_step=0, max_steps=V6_CERTIFY_EXPECTED_STEPS, last_error=f'{type(exc).__name__}: {exc}', retryable=not residue)
        raise
_V6_GPU_AUDIT_HEADER = 'timestamp\tgpu_index\tmemory_used_mib\tutilization_percent\tphase\tminimum_active_mib'

def _v6_write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise V6SftError(f'V6 control artifact is a symlink: {path}')
    if path.is_file():
        existing = _load_json_object(path, context=f'existing {path.name}')
        if existing != dict(payload):
            raise V6SftError(f'existing V6 control artifact drifted: {path}')
        return
    if path.exists():
        raise V6SftError(f'V6 control artifact is not a regular file: {path}')
    _atomic_write_json(path, payload)

def _v6_formal_managed_token(repo_root: Path) -> str:
    token_file_raw = os.environ.get('V6_MANAGED_TOKEN_FILE')
    if token_file_raw != str(CANONICAL_V6_LOG_ROOT / 'managed_process.token'):
        raise V6SftError('formal supervisor requires the canonical managed token file')
    token_path = repo_root / Path(token_file_raw)
    if token_path.is_symlink() or not token_path.is_file():
        raise V6SftError('formal managed token file is missing or a symlink')
    if token_path.stat().st_mode & 511 != 384:
        raise V6SftError('formal managed token file mode must be 0600')
    token = token_path.read_text(encoding='utf-8').strip()
    if len(token) < 49 or not token.startswith(f'{V6_FORMAL_RUN_ID}:') or os.environ.get('V6_MANAGED_PROCESS_TOKEN') != token:
        raise V6SftError('formal managed process token is malformed or split-brain')
    return token

def _v6_resume_namespace(run_dir: Path, checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(resume_from_checkpoint=checkpoint, run_id=V6_FORMAL_RUN_ID, output_dir=run_dir.parent)

def _v6_validate_resume_candidate(run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    """Validate a candidate plus every preceding checkpoint/evaluation commit."""
    step = _checkpoint_step(checkpoint)
    scheduled = tuple(v6_checkpoint_steps())
    if step not in scheduled:
        raise V6SftError(f'formal resume step is outside the schedule: {step}')
    report = validate_v6_resume_checkpoint(_v6_resume_namespace(run_dir, checkpoint))
    for prior_step in scheduled:
        if prior_step > step:
            break
        prior = run_dir / 'checkpoints' / f'checkpoint-{prior_step}'
        validate_v6_resume_checkpoint(_v6_resume_namespace(run_dir, prior))
        if prior_step in V6_VALIDATION_STEPS:
            validate_v6_inline_evaluation_receipt(run_dir=run_dir, step=prior_step)
    return report

def _v6_select_latest_resume_checkpoint(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    if not run_dir.exists():
        return (None, {'mode': 'new_run', 'policy': 'same_run_v6_checkpoint_only', 'resume_from_checkpoint': None})
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise V6SftError(f'formal run path is invalid: {run_dir}')
    checkpoint_root = run_dir / 'checkpoints'
    candidates: list[tuple[int, Path]] = []
    if checkpoint_root.is_dir() and (not checkpoint_root.is_symlink()):
        for candidate in checkpoint_root.glob('checkpoint-*'):
            try:
                step = _checkpoint_step(candidate)
            except V6SftError:
                continue
            if candidate.is_dir() and (not candidate.is_symlink()):
                candidates.append((step, candidate))
    for (_, candidate) in sorted(candidates, reverse=True):
        try:
            return (candidate, _v6_validate_resume_candidate(run_dir, candidate))
        except Exception:
            continue
    raise V6SftError('formal run directory exists but has no fully committed same-run V6 checkpoint')

def _v6_archive_formal_residue(*, run_dir: Path, log_root: Path, selected_step: int | None) -> Path | None:
    """Recoverably quarantine only artifacts newer than a selected checkpoint."""
    paths: list[Path] = []
    for control in (log_root / 'stable.json', log_root / 'completion_ready.json'):
        if control.exists() or control.is_symlink():
            paths.append(control)
    paths.extend(sorted(log_root.glob('stable.attempt.*.json')))
    if selected_step is not None:
        checkpoint_root = run_dir / 'checkpoints'
        if checkpoint_root.is_dir() and (not checkpoint_root.is_symlink()):
            for path in checkpoint_root.glob('checkpoint-*'):
                try:
                    if _checkpoint_step(path) > selected_step:
                        paths.append(path)
                except V6SftError:
                    paths.append(path)
        eval_root = run_dir / 'wm_eval'
        if eval_root.is_dir() and (not eval_root.is_symlink()):
            for path in eval_root.glob('step_*'):
                match = re.fullmatch('step_(\\d+)', path.name)
                if match is None or int(match.group(1)) > selected_step:
                    paths.append(path)
        transition_root = run_dir / 'transition_metrics'
        if transition_root.is_dir() and (not transition_root.is_symlink()):
            for path in transition_root.iterdir():
                match = re.search('-(\\d+)$', path.name)
                if match is None or int(match.group(1)) > selected_step:
                    paths.append(path)
        for final_name in ('final_adapter', 'train_summary.json', 'latest_successful_checkpoint.txt'):
            path = run_dir / final_name
            if path.exists() or path.is_symlink():
                paths.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        return None
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
    quarantine = log_root / 'recovery_quarantine' / f'{stamp}.{os.getpid()}'
    quarantine.mkdir(parents=True, exist_ok=False)
    for (index, source) in enumerate(unique):
        if not (source.exists() or source.is_symlink()):
            continue
        if source.is_relative_to(run_dir):
            relative = Path('run') / source.relative_to(run_dir)
        elif source.is_relative_to(log_root):
            relative = Path('log') / source.relative_to(log_root)
        else:
            raise V6SftError(f'refusing to archive non-canonical residue: {source}')
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination = destination.with_name(f'{destination.name}.{index}')
        shutil.move(str(source), str(destination))
    return quarantine

def _v6_prepare_append_evidence(path: Path, *, header: str | None=None) -> tuple[int, int]:
    if path.is_symlink():
        raise V6SftError(f'formal evidence path is a symlink: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if header is None:
            path.touch()
        else:
            path.write_text(header + '\n', encoding='utf-8')
    if not path.is_file():
        raise V6SftError(f'formal evidence path is not a regular file: {path}')
    payload = path.read_bytes()
    if payload and (not payload.endswith(b'\n')):
        raise V6SftError(f'formal evidence lacks an append line boundary: {path}')
    if header is not None:
        text = payload.decode('utf-8', errors='strict').splitlines()
        if not text or text[0] != header or text.count(header) != 1:
            raise V6SftError('formal GPU audit header drifted')
    return (len(payload), payload.count(b'\n'))

def _v6_formal_log_flags(path: Path, *, start_byte: int) -> tuple[bool, bool]:
    payload = path.read_bytes()[start_byte:].decode('utf-8', errors='strict')
    return ('"event": "v6_trainable_parameters"' in payload, '"event": "v6_loss_finite"' in payload)

def _v6_canonical_mapping_sha(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')))

def _v6_build_formal_stable(*, python_path: Path, pid: int, pgid: int, train_byte: int, train_line: int, gpu_byte: int, gpu_line: int, timestamp: str, memory: Mapping[str, int], resume: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    launch_argv = list(command)
    launch_environment = {'CUDA_VISIBLE_DEVICES': V6_CUDA_VISIBLE_DEVICES, 'OMP_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': 'src', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'V6_RUN_ID': V6_FORMAL_RUN_ID, 'V6_MANAGED_TOKEN_FILE': str(CANONICAL_V6_LOG_ROOT / 'managed_process.token')}
    return {'schema_version': 'local-wm-v6-formal-stable-v6', 'status': 'stable', 'run_id': V6_FORMAL_RUN_ID, 'training_pid': int(pid), 'training_pgid': int(pgid), 'train_log_start_byte': int(train_byte), 'train_log_start_line': int(train_line), 'gpu_audit_start_byte': int(gpu_byte), 'gpu_audit_start_line': int(gpu_line), 'qualifying_snapshot_timestamp': str(timestamp), 'simultaneous_memory_used_mib': {str(key): int(value) for (key, value) in memory.items()}, 'resume': dict(resume), 'expected_final_step': EXPECTED_FINAL_STEP, 'expected_trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'expected_trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'launch_argv': launch_argv, 'launch_environment': launch_environment, 'launch_argv_sha256': _v6_canonical_mapping_sha(launch_argv), 'launch_environment_sha256': _v6_canonical_mapping_sha(launch_environment)}

def _v6_validate_formal_completion_chain(authority: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the completed model, checkpoint schedule and all 11 evaluations."""
    run_dir = Path(authority['formal_run_dir'])
    certify_report = validate_v6_certify_receipt(Path(authority['recorded_log_root']) / 'certify_receipt.json')
    completion = _validate_final_adapter_chain(run_dir=run_dir, recorded_run_dir=Path(authority['recorded_formal_run_dir']), summary_path=run_dir / 'train_summary.json', run_config_path=run_dir / 'run_config.json', run_id=V6_FORMAL_RUN_ID, certify_run=False, expected_step=EXPECTED_FINAL_STEP, data_authority={'train_jsonl': authority['recorded_train_jsonl'], 'train_file': authority['train_jsonl'], 'focus_sidecar': authority['recorded_focus_sidecar'], 'focus_sidecar_file': authority['focus_sidecar'], 'focus_manifest': authority['recorded_focus_manifest'], 'focus_manifest_file': authority['focus_manifest'], 'rows': EXPECTED_TRAINING_ROWS, 'output_root': Path(authority['recorded_formal_run_dir']).parent, 'certify_receipt': Path(authority['recorded_log_root']) / 'certify_receipt.json'})
    summary = completion['summary']
    if summary.get('certify_receipt_authorization') != certify_report:
        raise V6SftError('formal summary certify authorization drifted')
    resume = _validate_resume_semantics(resume=summary.get('resume'), run_dir=run_dir)
    actual_steps: list[int] = []
    checkpoint_root = run_dir / 'checkpoints'
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise V6SftError('formal checkpoint root is missing or a symlink')
    for path in checkpoint_root.glob('checkpoint-*'):
        if path.is_dir() and (not path.is_symlink()):
            actual_steps.append(_checkpoint_step(path))
    if sorted(actual_steps) != list(v6_checkpoint_steps()):
        raise V6SftError('formal checkpoint schedule is incomplete or contains extras')
    validation_binding: dict[str, Any] = {}
    for step in v6_checkpoint_steps():
        validate_v6_resume_checkpoint(_v6_resume_namespace(run_dir, checkpoint_root / f'checkpoint-{step}'))
    for step in V6_VALIDATION_STEPS:
        validation = validate_v6_inline_evaluation_receipt(run_dir=run_dir, step=step)
        validation_path = run_dir / 'wm_eval' / f'step_{step:06d}' / 'validation/evaluation_receipt.json'
        validation_binding[str(step)] = {'path': str(validation_path), 'sha256': _sha256_file(validation_path), 'display_name': validation['display_name'], 'completed_judgments': 500, 'strict_complete': True}
    return (completion, resume, validation_binding)

def _v6_finalize_formal_receipt(*, manifest_path: str | Path, authority: Mapping[str, Any], stable: Mapping[str, Any], managed_token: str) -> dict[str, Any]:
    """Freeze the finished launch evidence and atomically commit V6 completion."""
    log_root = Path(authority['log_root'])
    run_dir = Path(authority['formal_run_dir'])
    pid = int(stable['training_pid'])
    pgid = int(stable['training_pgid'])
    train_byte = int(stable['train_log_start_byte'])
    train_line = int(stable['train_log_start_line'])
    gpu_byte = int(stable['gpu_audit_start_byte'])
    gpu_line = int(stable['gpu_audit_start_line'])
    timestamp = str(stable['qualifying_snapshot_timestamp'])
    memory = {str(key): int(value) for (key, value) in dict(stable['simultaneous_memory_used_mib']).items()}
    launch_argv = stable.get('launch_argv')
    launch_environment = stable.get('launch_environment')
    expected_prefix = [str(Path(authority['python_bin']).resolve(strict=True)), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=6', '-m', 'mkd2world.training.next_tsm_sft', 'train', '--manifest', str(CANONICAL_V6_MANIFEST)]
    if stable.get('schema_version') != 'local-wm-v6-formal-stable-v6' or stable.get('status') != 'stable' or stable.get('run_id') != V6_FORMAL_RUN_ID or (set(memory) != {str(index) for index in range(V6_WORLD_SIZE)}) or (not isinstance(launch_argv, list)) or (launch_argv[:len(expected_prefix)] != expected_prefix) or (not isinstance(launch_environment, Mapping)) or any((launch_environment.get(key) != value for (key, value) in {'CUDA_VISIBLE_DEVICES': V6_CUDA_VISIBLE_DEVICES, 'OMP_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': 'src', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'V6_RUN_ID': V6_FORMAL_RUN_ID, 'V6_MANAGED_TOKEN_FILE': str(CANONICAL_V6_LOG_ROOT / 'managed_process.token')}.items())):
        raise V6SftError('formal stable marker is invalid before finalization')
    train_log = log_root / 'train.log'
    gpu_audit = log_root / 'gpu_memory.tsv'
    stable_path = log_root / 'stable.json'
    if _v6_certify_token_processes(managed_token):
        raise V6SftError('formal token processes remain before finalization')
    events = _parse_v6_launch_events(train_log, start_byte=train_byte, start_line=train_line, context='formal V6 train log')
    audit = _parse_gpu_audit(gpu_audit, start_byte=gpu_byte, start_line=gpu_line, context='formal V6 GPU audit')
    if audit['qualifying_snapshots'].get(timestamp) != memory:
        raise V6SftError('formal stable snapshot differs from the frozen GPU audit')
    if _load_json_object(stable_path, context='formal stable marker') != dict(stable):
        raise V6SftError('canonical formal stable marker changed before finalization')
    monitor_path = log_root / f'formal_monitor_status.{pid}.json'
    monitor = {'schema_version': 'local-wm-v6-gpu-monitor-v2', 'status': 'passed', 'kind': 'formal', 'training_pid': pid, 'training_pgid': pgid, 'gpu_memory_audit': str(Path(authority['recorded_log_root']) / 'gpu_memory.tsv'), 'gpu_memory_audit_sha256': audit['sha256'], 'simultaneous_memory_used_mib': memory, 'qualifying_snapshot_timestamp': timestamp, 'parameters_observed': True, 'finite_loss_observed': True, 'simultaneous_memory_floor_observed': True, 'monitor_failed': False, 'launch_log_start_byte': train_byte, 'launch_log_start_line': train_line, 'gpu_audit_start_byte': gpu_byte, 'gpu_audit_start_line': gpu_line}
    _v6_write_once_json(monitor_path, monitor)
    validate_v6_formal_monitor_evidence(train_log=train_log, train_log_start_byte=train_byte, train_log_start_line=train_line, gpu_memory_audit=gpu_audit, gpu_audit_start_byte=gpu_byte, gpu_audit_start_line=gpu_line, monitor_status=monitor_path, training_pid=pid, training_pgid=pgid, qualifying_snapshot_timestamp=timestamp, gpu_memory_audit_recorded=Path(authority['recorded_log_root']) / 'gpu_memory.tsv')
    (completion, resume, validation_binding) = _v6_validate_formal_completion_chain(authority)
    if stable.get('resume') != resume:
        raise V6SftError('formal stable marker resume semantics differ from summary')
    if resume['mode'] == 'new_run':
        if '--resume-from-checkpoint' in launch_argv:
            raise V6SftError('fresh formal stable marker unexpectedly resumes')
    elif '--resume-from-checkpoint' not in launch_argv or Path(launch_argv[launch_argv.index('--resume-from-checkpoint') + 1]).resolve(strict=True) != Path(resume['resume_from_checkpoint']).resolve(strict=True):
        raise V6SftError('formal stable marker resumed a different checkpoint')
    actual_paths = {'train_log': train_log, 'gpu_memory_audit': gpu_audit, 'monitor_status': monitor_path, 'train_summary': run_dir / 'train_summary.json', 'run_config': run_dir / 'run_config.json', 'stable_marker': stable_path, 'certify_receipt': log_root / 'certify_receipt.json', 'trainer': Path(authority['trainer_path']), 'runner': Path(authority['runner_path']), 'focus_sidecar': Path(authority['focus_sidecar']), 'focus_manifest': Path(authority['focus_manifest']), 'final_adapter_integrity': Path(completion['integrity_path']), 'latest_success_pointer': Path(completion['latest_pointer'])}
    for (name, path) in actual_paths.items():
        if path.is_symlink() or not path.is_file():
            raise V6SftError(f'formal finalization dependency is missing: {name}')
    hashes = {'train_log_sha256': events['sha256'], 'gpu_memory_audit_sha256': audit['sha256'], 'monitor_status_sha256': _sha256_file(actual_paths['monitor_status']), 'train_summary_sha256': _sha256_file(actual_paths['train_summary']), 'run_config_sha256': _sha256_file(actual_paths['run_config']), 'stable_marker_sha256': _sha256_file(actual_paths['stable_marker']), 'certify_receipt_sha256': _sha256_file(actual_paths['certify_receipt']), 'trainer_sha256': _sha256_file(actual_paths['trainer']), 'runner_sha256': _sha256_file(actual_paths['runner']), 'focus_sidecar_sha256': _sha256_file(actual_paths['focus_sidecar']), 'focus_manifest_sha256': _sha256_file(actual_paths['focus_manifest']), 'final_adapter_integrity_sha256': completion['integrity_sha256'], 'latest_success_pointer_sha256': completion['latest_pointer_sha256']}
    ready = {'schema_version': 'local-wm-v6-completion-ready-v1', 'status': 'passed', 'run_id': V6_FORMAL_RUN_ID, 'training_pid': pid, 'training_pgid': pgid, 'exit_code': 0, 'train_log_start_byte': train_byte, 'train_log_start_line': train_line, 'gpu_audit_start_byte': gpu_byte, 'gpu_audit_start_line': gpu_line, 'qualifying_snapshot_timestamp': timestamp, 'simultaneous_memory_used_mib': memory, 'resume': resume, 'no_training_processes_remaining': True, **hashes}
    ready_path = log_root / 'completion_ready.json'
    _v6_write_once_json(ready_path, ready)
    ready_sha = _sha256_file(ready_path)
    recorded_log = Path(authority['recorded_log_root'])
    recorded_run = Path(authority['recorded_formal_run_dir'])
    receipt = {'schema_version': V6_FORMAL_RECEIPT_SCHEMA, 'status': 'passed', 'run_id': V6_FORMAL_RUN_ID, 'formal_training_started': True, 'formal_training_completed': True, 'world_size': V6_WORLD_SIZE, 'physical_gpus': list(range(V6_WORLD_SIZE)), 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'final_global_step': EXPECTED_FINAL_STEP, 'expected_final_step': EXPECTED_FINAL_STEP, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'matcher_sha256': focus_contract.AUDIT_LABELER_PARENT_SHA256, 'training_pid': pid, 'training_pgid': pgid, 'train_log': str(recorded_log / 'train.log'), 'train_log_start_byte': train_byte, 'train_log_start_line': train_line, 'gpu_memory_audit': str(recorded_log / 'gpu_memory.tsv'), 'gpu_audit_start_byte': gpu_byte, 'gpu_audit_start_line': gpu_line, 'monitor_status': str(recorded_log / monitor_path.name), 'train_summary': str(recorded_run / 'train_summary.json'), 'run_config': str(recorded_run / 'run_config.json'), 'stable_marker': str(recorded_log / 'stable.json'), 'completion_ready': str(recorded_log / 'completion_ready.json'), 'certify_receipt': str(recorded_log / 'certify_receipt.json'), 'trainer_path': str(authority['recorded_trainer_path']), 'runner_path': str(authority['recorded_runner_path']), 'focus_sidecar': str(authority['recorded_focus_sidecar']), 'focus_manifest': str(authority['recorded_focus_manifest']), 'qualifying_snapshot_timestamp': timestamp, 'simultaneous_memory_used_mib': memory, 'resume': resume, 'completion_ready_sha256': ready_sha, 'parameters_observed': True, 'finite_loss_observed': True, 'progress_observed': True, 'focus_loss_active': True, 'loss_ramp': False, 'no_training_processes_remaining': True, 'validation_steps': list(V6_VALIDATION_STEPS), 'validation_receipts': validation_binding, **hashes}
    receipt_path = log_root / 'formal_receipt.json'
    if receipt_path.exists() or receipt_path.is_symlink():
        raise V6SftError('refusing to overwrite an existing formal receipt')
    _atomic_write_json(receipt_path, receipt)
    report = deep_validate_v6_formal_receipt(manifest_path)
    write_v6_run_state(status='complete', phase='formal_receipt_complete', global_step=EXPECTED_FINAL_STEP, max_steps=EXPECTED_FINAL_STEP, retryable=False)
    return report

def _run_v6_formal_supervisor_once(manifest_path: str | Path, python_bin: str | Path) -> dict[str, Any]:
    manifest = validate_v6_manifest(manifest_path)
    authority = _formal_authority(None)
    repo_root = Path(__file__).resolve().parents[3]
    log_root = Path(authority['log_root'])
    run_dir = Path(authority['formal_run_dir'])
    receipt_path = log_root / 'formal_receipt.json'
    if receipt_path.is_file() and (not receipt_path.is_symlink()):
        return deep_validate_v6_formal_receipt(manifest_path)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise V6SftError('formal receipt path exists but is not a regular file')
    deep_validate_v6_certify_receipt(manifest_path)
    managed_token = _v6_formal_managed_token(repo_root)
    python_path = Path(python_bin).resolve(strict=True)
    if python_path != Path(authority['python_bin']).resolve(strict=True):
        raise V6SftError('formal supervisor Python differs from manifest authority')
    if _v6_certify_token_processes(managed_token):
        raise V6SftError('formal token already has live child processes')
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / 'train_summary.json'
    if summary_path.is_file() and (not summary_path.is_symlink()):
        stable_path = log_root / 'stable.json'
        if stable_path.is_symlink() or not stable_path.is_file():
            raise V6SftError('completed formal summary lacks its immutable launch stable marker')
        stable = _load_json_object(stable_path, context='formal stable marker')
        return _v6_finalize_formal_receipt(manifest_path=manifest_path, authority=authority, stable=stable, managed_token=managed_token)
    if summary_path.exists() or summary_path.is_symlink():
        raise V6SftError('existing formal train summary is invalid')
    resume_checkpoint: Path | None
    resume: dict[str, Any]
    if run_dir.exists() or run_dir.is_symlink():
        try:
            (resume_checkpoint, resume) = _v6_select_latest_resume_checkpoint(run_dir)
        except V6SftError:
            stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
            quarantine = log_root / 'recovery_quarantine' / f'pre_checkpoint_run.{stamp}.{os.getpid()}'
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            if run_dir.is_symlink() or not run_dir.is_dir():
                raise
            shutil.move(str(run_dir), str(quarantine))
            (resume_checkpoint, resume) = (None, {'mode': 'new_run', 'policy': 'same_run_v6_checkpoint_only', 'resume_from_checkpoint': None})
    else:
        (resume_checkpoint, resume) = (None, {'mode': 'new_run', 'policy': 'same_run_v6_checkpoint_only', 'resume_from_checkpoint': None})
    selected_step = _checkpoint_step(resume_checkpoint) if resume_checkpoint is not None else None
    _v6_archive_formal_residue(run_dir=run_dir, log_root=log_root, selected_step=selected_step)
    train_log = log_root / 'train.log'
    gpu_audit = log_root / 'gpu_memory.tsv'
    (train_byte, train_line) = _v6_prepare_append_evidence(train_log)
    (gpu_byte, gpu_line) = _v6_prepare_append_evidence(gpu_audit, header=_V6_GPU_AUDIT_HEADER)
    command = [str(python_path), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=6', '-m', 'mkd2world.training.next_tsm_sft', 'train', '--manifest', str(CANONICAL_V6_MANIFEST)]
    if resume_checkpoint is not None:
        command.extend(['--resume-from-checkpoint', str(resume_checkpoint.resolve(strict=True))])
    environment = dict(os.environ)
    environment.update({'CUDA_VISIBLE_DEVICES': V6_CUDA_VISIBLE_DEVICES, 'OMP_NUM_THREADS': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': 'src', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'V6_RUN_ID': V6_FORMAL_RUN_ID, 'V6_MANAGED_PROCESS_TOKEN': managed_token, 'V6_MANAGED_TOKEN_FILE': str(CANONICAL_V6_LOG_ROOT / 'managed_process.token')})
    write_v6_run_state(status='running', phase='formal_supervisor', global_step=selected_step or 0, max_steps=EXPECTED_FINAL_STEP, retryable=False)
    qualifying: tuple[str, dict[str, int]] | None = None
    stable: dict[str, Any] | None = None
    process: subprocess.Popen[str] | None = None
    try:
        with train_log.open('a', encoding='utf-8') as log_handle, gpu_audit.open('a', encoding='utf-8') as audit_handle:
            process = subprocess.Popen(command, cwd=repo_root, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            pid = int(process.pid)
            pgid = int(os.getpgid(pid))
            if pid != pgid or pgid == os.getpgrp():
                raise V6SftError('formal torchrun lacks an isolated PID/PGID')
            attempt_path = log_root / f'stable.attempt.{pid}.json'
            while process.poll() is None:
                (timestamp, snapshot) = _v6_certify_gpu_snapshot(certify_token=managed_token)
                memory = {str(row['gpu_index']): int(row['memory_used_mib']) for row in snapshot}
                for row in snapshot:
                    audit_handle.write('\t'.join([timestamp, str(row['gpu_index']), str(row['memory_used_mib']), str(row['utilization_percent']), str(row['phase']), str(row['minimum_active_mib'])]) + '\n')
                audit_handle.flush()
                os.fsync(audit_handle.fileno())
                if qualifying is None and all((row['phase'] == 'active' and int(row['memory_used_mib']) >= V6_TRAINING_MEMORY_EVIDENCE_FLOOR_MIB for row in snapshot)):
                    qualifying = (timestamp, memory)
                log_handle.flush()
                (params_seen, finite_seen) = _v6_formal_log_flags(train_log, start_byte=train_byte)
                if stable is None and qualifying is not None and params_seen and finite_seen:
                    stable = _v6_build_formal_stable(python_path=python_path, pid=pid, pgid=pgid, train_byte=train_byte, train_line=train_line, gpu_byte=gpu_byte, gpu_line=gpu_line, timestamp=qualifying[0], memory=qualifying[1], resume=resume, command=command)
                    _v6_write_once_json(attempt_path, stable)
                    _v6_write_once_json(log_root / 'stable.json', stable)
                time.sleep(2.0)
            return_code = int(process.wait())
            log_handle.flush()
            os.fsync(log_handle.fileno())
            audit_handle.flush()
            os.fsync(audit_handle.fileno())
    except BaseException as exc:
        _terminate_v6_certify_token_processes(managed_token)
        write_v6_run_state(status='failed', phase='formal_supervisor_failed', global_step=selected_step or 0, max_steps=EXPECTED_FINAL_STEP, last_error=f'{type(exc).__name__}: {exc}', retryable=True)
        raise
    remaining = _v6_certify_token_processes(managed_token)
    if remaining:
        _terminate_v6_certify_token_processes(managed_token)
        raise V6SftError(f'formal token processes remained after torchrun exit: {sorted(remaining)}')
    if return_code != 0:
        write_v6_run_state(status='failed', phase='formal_torchrun_failed', global_step=selected_step or 0, max_steps=EXPECTED_FINAL_STEP, last_error=f'formal torchrun exit code {return_code}', retryable=True)
        raise V6SftError(f'formal torchrun failed with exit code {return_code}')
    if stable is None:
        stable_path = log_root / 'stable.json'
        if stable_path.is_file() and (not stable_path.is_symlink()):
            stable = _load_json_object(stable_path, context='formal stable marker')
        else:
            raise V6SftError('formal training completed without stable GPU/loss evidence')
    return _v6_finalize_formal_receipt(manifest_path=manifest_path, authority=authority, stable=stable, managed_token=managed_token)

def run_v6_formal_supervisor(manifest_path: str | Path, python_bin: str | Path) -> dict[str, Any]:
    try:
        return _run_v6_formal_supervisor_once(manifest_path, python_bin)
    except BaseException as exc:
        state_path = CANONICAL_V6_LOG_ROOT / 'run_state.json'
        retryable = False
        current_step = 0
        if state_path.is_file() and (not state_path.is_symlink()):
            try:
                state = _load_json_object(state_path, context='V6 failed formal state')
                retryable = state.get('retryable') is True
                current_step = int(state.get('global_step', 0))
            except Exception:
                pass
        if not retryable:
            write_v6_run_state(status='blocked', phase='formal_manual_review', global_step=current_step, max_steps=EXPECTED_FINAL_STEP, last_error=f'{type(exc).__name__}: {exc}', retryable=False)
        raise

def v6_loss_contract(base_loss_granularity: str | None=None, focus_loss_granularity: str | None=None) -> dict[str, Any]:
    base_granularity = base_loss_granularity or V6_BASE_LOSS_GRANULARITY
    focus_granularity = focus_loss_granularity or V6_FOCUS_LOSS_GRANULARITY
    for granularity in (base_granularity, focus_granularity):
        if granularity not in V6_LOSS_GRANULARITIES:
            raise V6SftError(f'invalid loss granularity {granularity!r}')
    if V6_LOSS_PROFILE == 'ee' and base_granularity == 'element' and (focus_granularity == 'element'):
        return {'profile': 'v4_ee_plus_page_element_margin', 'base_reduction': 'mean token CE per complete target element line', 'neutral_group': 'Environment plus assistant EOS/im_end', 'focus_kinds': ['ADD', 'UPDATE'], 'delete_focus': False, 'lambda': '0 if C=0 else min(3,sqrt(N/C))', 'sample_loss': '(sum(base_means)+lambda*sum(focus_means))/(N+lambda*C)', 'batch_loss': 'mean(sample_loss)', 'ramp': False, 'ranking': {'eligible': 'changed and Current/Next assistant token ids differ', 'next_target': 'raw target_next_md', 'current_target': 'raw current_md', 'page_reduction': 'element-token-mean then equal element mean; no focus', 'margin': V6_MARGIN, 'beta_max': V6_BETA_MAX, 'beta_warmup_optimizer_steps': V6_BETA_WARMUP_STEPS, 'recompute': '3F+2B exact Current gate/replay', 'normalization': 'complete local microbatch mean; DDP+GA global mean'}}
    return {'profile': V6_LOSS_PROFILE, 'base_granularity': base_granularity, 'focus_granularity': focus_granularity, 'base_reduction': 'mean token CE per complete target element line' if base_granularity == 'element' else 'sum token CE over all supervised base tokens', 'focus_reduction': 'mean token CE per ADD/UPDATE focus element' if focus_granularity == 'element' else 'sum token CE over all ADD/UPDATE focus tokens', 'neutral_group': 'Environment plus assistant EOS/im_end', 'focus_kinds': ['ADD', 'UPDATE'], 'delete_focus': False, 'lambda': '0 if C=0 else min(3,sqrt(N/C))', 'count_semantics': 'N and C count the selected base/focus units', 'sample_loss': '(sum(base_selected)+lambda*sum(focus_selected))/(N+lambda*C)', 'batch_loss': 'mean(sample_loss)', 'ramp': False}

def build_v6_dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    validate_v6_training_args(args, require_ddp=False)
    v5_preflight = _load_json_object(CANONICAL_V5_PREFLIGHT, context='frozen V5 CPU preflight')
    base_model_report = dict(v5_preflight.get('base_model') or {})
    data_report = dict(v5_preflight.get('data') or {})
    token_report = dict(v5_preflight.get('token_alignment') or {})
    collator_report = dict(v5_preflight.get('collator_alignment') or {})
    if v5_preflight.get('status') != 'passed' or v5_preflight.get('can_start_training') is not True or data_report.get('rows') != EXPECTED_TRAINING_ROWS or (token_report.get('status') != 'passed') or (token_report.get('rows') != EXPECTED_TRAINING_ROWS) or (token_report.get('rank_eligible_rows') != 25005) or (collator_report.get('status') != 'passed'):
        raise V6SftError('frozen V5 preflight cannot certify V6 shared contracts')
    base_path = _require_lexical_path(args.base_model, CANONICAL_BASE_MODEL, context='V6 bare base model')
    for (name, (size_bytes, _sha256)) in EXPECTED_BASE_SHARDS.items():
        shard = base_path / name
        if not shard.is_file() or shard.is_symlink() or shard.stat().st_size != size_bytes:
            raise V6SftError(f'live bare-Qwen shard identity drifted: {name}')
    init_adapter_report = validate_v3_cur_1200_init_adapter(args.init_adapter)
    resume_report = validate_v6_resume_checkpoint(args)
    manifest = validate_v6_manifest(CANONICAL_V6_MANIFEST)
    manifest_data = manifest['data']
    sidecars = load_focus_sidecar(args.focus_sidecar)
    if len(sidecars) != EXPECTED_TRAINING_ROWS:
        raise V6SftError('live V20 sidecar row count drifted')
    certify_plan = build_v6_certify_sample_plan(args.train_jsonl, sidecars, max_target_sample_id=token_report['max_target_sample_id'], max_current_target_sample_id=token_report['max_current_target_sample_id'])
    certify_plan_publication = None
    if args.write_certify_sample_plan is not None:
        certify_plan_publication = publish_v6_certify_sample_plan(Path(args.write_certify_sample_plan), certify_plan)
    return {'status': 'passed', 'can_start_training': True, 'formal_training_started': False, 'display_name_pattern': DISPLAY_NAME_PATTERN, 'data': data_report, 'token_alignment': token_report, 'collator_alignment': collator_report, 'certify_sample_plan': {'rows': certify_plan['rows'], 'selected_sample_ids_sha256': certify_plan['selected_sample_ids_sha256'], 'bucket_counts': certify_plan['bucket_counts'], 'required_coverage': certify_plan['required_coverage'], 'publication': certify_plan_publication}, 'base_model': base_model_report, 'init_adapter': init_adapter_report, 'adapter_initialization': v6_adapter_initialization_contract(), 'manifest': {'path': str(CANONICAL_V6_MANIFEST), 'sha256': _sha256_file(CANONICAL_V6_MANIFEST)}, 'reused_certification': {'source': str(CANONICAL_V5_PREFLIGHT), 'source_sha256': EXPECTED_V5_PREFLIGHT_SHA256, 'scope': 'shared base/data/token/collator contracts', 'reran_full_token_or_validation500_smoke': False}, 'implementation': {'trainer_path': str(Path(__file__).resolve()), 'trainer_sha256': _sha256_file(Path(__file__).resolve()), 'runner_path': str(Path(__file__).resolve().parents[3] / v6_runner_relative_path()), 'runner_sha256': _sha256_file(Path(__file__).resolve().parents[3] / v6_runner_relative_path())}, 'v5_parity': getattr(args, '_v6_v5_parity_report'), 'resume': resume_report, 'loss': v6_loss_contract(args.base_loss_granularity, args.focus_loss_granularity), 'training': {'gpus': list(range(V6_WORLD_SIZE)), 'world_size': V6_WORLD_SIZE, 'per_device_train_batch_size': args.per_device_train_batch_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'epochs': args.num_train_epochs, 'learning_rate': args.learning_rate, 'scheduler': args.lr_scheduler_type, 'warmup_ratio': args.warmup_ratio, 'weight_decay': args.weight_decay, 'max_length': args.max_length, 'max_output_tokens': args.max_output_tokens, 'seed': args.seed, 'save_steps': args.save_steps, 'checkpoint_steps': list(v6_checkpoint_steps()), 'validation_steps': list(v6_validation_steps()), 'save_interval_examples': V6_SAVE_INTERVAL_EXAMPLES, 'save_interval_equivalence': '100*60 == 6000', 'expected_trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'expected_final_step': EXPECTED_FINAL_STEP, 'completion_contract': v6_formal_completion_contract(), 'loss_ramp': False}}

def _create_distributed_run_directory(output_dir: Path, *, is_main_process: bool, torch_module: Any, distributed_module: Any, status_device: Any) -> None:
    """Atomically reserve one run directory and publish rank-0's decision."""
    create_error: str | None = None
    if is_main_process:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            create_error = f'refusing to overwrite V6 run directory: {output_dir}'
        except OSError as exc:
            create_error = f'failed to create V6 run directory {output_dir}: {exc}'
    status = torch_module.tensor([1 if is_main_process and create_error is None else 0], dtype=torch_module.int32, device=status_device)
    distributed_module.broadcast(status, src=0)
    if int(status.item()) != 1:
        raise V6SftError(create_error or f'rank 0 refused to create V6 run directory: {output_dir}')
    distributed_module.barrier()

def build_v6_checkpoint_binding_callback(TrainerCallback: Any, *, torch_module: Any, checkpoint_root: Path, run_dir: Path, run_id: str, rank: int):
    """Write V6 provenance only after the generic checkpoint integrity callback."""
    from mkd2world.training import sft as sft_module

    class V6CheckpointBindingCallback(TrainerCallback):

        def on_save(self, training_args, state, control, **kwargs):
            step = int(state.global_step)
            if rank == 0:
                binding_path = write_v6_checkpoint_binding(checkpoint_root / f'checkpoint-{step}', run_dir=run_dir, run_id=run_id, global_step=step)
                print(json.dumps({'event': 'v6_checkpoint_binding_passed', 'global_step': step, 'binding': str(binding_path)}, ensure_ascii=False), flush=True)
            sft_module.distributed_barrier(torch_module)
            return control
    return V6CheckpointBindingCallback()

def configure_v6_checkpoint_saving_policy(training_args: Any, *, certify_run: bool) -> Any:
    strategy = getattr(training_args, 'save_strategy', None)
    strategy_value = getattr(strategy, 'value', strategy)
    if strategy_value != 'steps':
        raise V6SftError('V6 TrainingArguments must start with save_strategy=steps')
    del certify_run
    return training_args

def v6_checkpoint_steps() -> tuple[int, ...]:
    regular = tuple(range(V6_SAVE_STEPS, 1001, V6_SAVE_STEPS))
    return tuple(sorted(set(regular).union(V6_EXTRA_CHECKPOINT_STEPS)))

def v6_validation_steps() -> tuple[int, ...]:
    """Return checkpoints which synchronously run Validation500."""
    return V6_VALIDATION_STEPS

def build_v6_extra_checkpoint_callback(TrainerCallback: Any, *, certify_run: bool=False):
    extra_steps = frozenset((V6_CERTIFY_EXPECTED_STEPS,) if certify_run else V6_EXTRA_CHECKPOINT_STEPS)

    class V6ExtraCheckpointCallback(TrainerCallback):

        def on_step_end(self, training_args, state, control, **kwargs):
            del training_args, kwargs
            if int(state.global_step) in extra_steps:
                control.should_save = True
            return control
    return V6ExtraCheckpointCallback()

def _jsonl_row_count(path: Path) -> int:
    with path.open(encoding='utf-8') as handle:
        return sum((1 for line in handle if line.strip()))

def _validate_v6_validation500_predictions(*, dataset_jsonl: Path, predictions_jsonl: Path, step: int) -> list[dict[str, Any]]:
    from mkd2world.training import sft as sft_module
    source_rows = [row for (_line, row) in _iter_jsonl_objects(dataset_jsonl)]
    prediction_rows = [row for (_line, row) in _iter_jsonl_objects(predictions_jsonl)]
    if len(source_rows) != 500 or len(prediction_rows) != 500:
        raise V6SftError('inline Validation500 must cover exactly 500 source rows')
    by_index: dict[int, dict[str, Any]] = {}
    for row in prediction_rows:
        index = int(row.get('source_eval_index', -1))
        if index in by_index:
            raise V6SftError(f'duplicate Validation500 source_eval_index {index}')
        by_index[index] = row
    if set(by_index) != set(range(500)):
        raise V6SftError('inline Validation500 source indexes must equal 0..499')
    for (index, source) in enumerate(source_rows):
        normalized = sft_module.normalize_row_for_dataset(dict(source))
        prediction = by_index[index]
        expected = {'id': normalized['id'], 'target_next_md': normalized['target_next_md'], 'candidate_action_json': normalized['candidate_action_json'], 'meta_json': normalized['meta_json']}
        mismatches = {key: {'expected': value, 'actual': prediction.get(key)} for (key, value) in expected.items() if prediction.get(key) != value}
        if mismatches or int(prediction.get('global_step', -1)) != int(step) or int(prediction.get('rank', -1)) not in range(V6_VALIDATION_WORKERS) or (prediction.get('prediction_replay_valid') is not True) or (prediction.get('model_output') != prediction.get('model_output_next_md')) or (not str(prediction.get('model_output_next_md') or '').strip()):
            raise V6SftError(f'inline Validation500 row contract failed at index {index}: {mismatches}')
    return [by_index[index] for index in range(500)]

def _validate_v6_transition_outputs(*, transition_metrics_path: Path, predictions_path: Path, source_prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    transition = _load_json_object(transition_metrics_path, context='inline transition metrics')
    all_metrics = transition.get('metrics', {}).get('all', {})
    if transition.get('schema_version') != 'localwm_transition_metrics_v2' or transition.get('metrics', {}).get('ntm_scope') != 'all_samples' or int(all_metrics.get('rows', -1)) != 500 or (sum((int(all_metrics.get(key, -1000)) for key in ('transition_win_rows', 'transition_tie_rows', 'transition_loss_rows'))) != 500):
        raise V6SftError('inline transition metrics all-sample contract failed')
    prediction_files = transition.get('prediction_files')
    expected_prediction_binding = [{'path': str(predictions_path), 'sha256': _sha256_file(predictions_path)}]
    if prediction_files != expected_prediction_binding:
        raise V6SftError('transition metrics are not bound to canonical predictions')
    per_sample_path = Path(str(transition.get('per_sample_path') or ''))
    if per_sample_path.is_symlink() or not per_sample_path.is_file():
        raise V6SftError('inline transition per-sample output is missing')
    per_sample_rows = [row for (_line, row) in _iter_jsonl_objects(per_sample_path)]
    if len(per_sample_rows) != 500:
        raise V6SftError('inline transition per-sample output must contain 500 rows')
    seen_indexes: set[int] = set()
    for row in per_sample_rows:
        index = int(row.get('source_eval_index', -1))
        if index in seen_indexes or index not in range(500):
            raise V6SftError('inline transition per-sample indexes are invalid')
        seen_indexes.add(index)
        prediction = source_prediction_rows[index]
        if row.get('schema_version') != 'localwm_transition_metrics_v2' or row.get('id') != prediction.get('id') or row.get('prediction_mode') != 'next' or (row.get('prediction_replay_valid') is not True) or any((not math.isfinite(float(row[key])) for key in ('next_elef1', 'prediction_current_elef1', 'Copy-EleF1', 'NTM'))):
            raise V6SftError(f'inline transition per-sample row failed at index {index}')
    return transition

def _v6_process_identity(pid: int) -> tuple[int, int] | None:
    """Return ``(real_uid, starttime)`` or None for an exited process."""
    try:
        status = Path(f'/proc/{pid}/status').read_text(encoding='utf-8')
        stat = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    uid_line = next((line for line in status.splitlines() if line.startswith('Uid:')), None)
    if uid_line is None:
        raise V6SftError(f'cannot read real uid for GPU process {pid}')
    real_uid = int(uid_line.split()[1])
    closing = stat.rfind(')')
    fields_after_comm = stat[closing + 2:].split()
    if closing < 0 or len(fields_after_comm) <= 19:
        raise V6SftError(f'cannot read starttime for GPU process {pid}')
    return (real_uid, int(fields_after_comm[19]))

def _v6_process_token(pid: int) -> str | None:
    try:
        environ = Path(f'/proc/{pid}/environ').read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    prefix = b'V6_MANAGED_PROCESS_TOKEN='
    for item in environ.split(b'\x00'):
        if item.startswith(prefix):
            try:
                return item[len(prefix):].decode('utf-8', errors='strict')
            except UnicodeDecodeError:
                return None
    return None

def _v6_gpu_sample(*, managed_token: str, known_managed: dict[int, tuple[int, int]]) -> dict[str, Any]:
    """Classify one nvidia-smi snapshot without treating foreign PIDs as fatal."""
    gpu_query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    gpu_rows: dict[str, dict[str, Any]] = {}
    uuid_to_index: dict[str, str] = {}
    for raw in gpu_query.stdout.splitlines():
        (index, uuid, total_memory) = (part.strip() for part in raw.split(',', 2))
        if index not in {str(value) for value in range(V6_WORLD_SIZE)}:
            continue
        uuid_to_index[uuid] = index
        gpu_rows[index] = {'total_memory_used_mib': int(total_memory), 'managed_memory_used_mib': 0, 'managed_pids': [], 'foreign_pids': []}
    if set(gpu_rows) != {str(value) for value in range(V6_WORLD_SIZE)}:
        raise V6SftError(f'nvidia-smi did not return physical GPUs 0-5: {gpu_rows}')
    process_query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    for raw in process_query.stdout.splitlines():
        parts = [part.strip() for part in raw.split(',', 2)]
        if len(parts) != 3 or parts[0] not in uuid_to_index:
            continue
        gpu_index = uuid_to_index[parts[0]]
        try:
            pid = int(parts[1])
            memory_mib = int(parts[2])
        except ValueError:
            continue
        identity_before = _v6_process_identity(pid)
        if identity_before is None:
            gpu_rows[gpu_index]['foreign_pids'].append(pid)
            continue
        token = _v6_process_token(pid)
        identity_after = _v6_process_identity(pid)
        if identity_after is None:
            continue
        managed = False
        if identity_before == identity_after and token == managed_token:
            known_managed[pid] = identity_after
            managed = True
        elif token is None and identity_before == identity_after and (known_managed.get(pid) == identity_after):
            managed = True
        if managed:
            gpu_rows[gpu_index]['managed_memory_used_mib'] += memory_mib
            gpu_rows[gpu_index]['managed_pids'].append(pid)
        else:
            gpu_rows[gpu_index]['foreign_pids'].append(pid)
    for row in gpu_rows.values():
        row['managed_pids'].sort()
        row['foreign_pids'] = sorted(set(row['foreign_pids']))
    return {'timestamp': datetime.now().astimezone().isoformat(timespec='milliseconds'), 'gpus': gpu_rows}

class V6ManagedGpuAudit:
    """Background per-checkpoint audit for the 18 managed validation workers."""

    def __init__(self, *, run_dir: Path, managed_token: str) -> None:
        if not managed_token:
            raise V6SftError('V6_MANAGED_PROCESS_TOKEN must be non-empty')
        self.run_dir = run_dir
        self.managed_token = managed_token
        self._step: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._known_managed: dict[int, tuple[int, int]] = {}

    def start(self, step: int) -> None:
        if self._thread is not None:
            raise V6SftError('a V6 managed GPU audit is already active')
        self._step = int(step)
        self._stop.clear()
        self._samples = []
        self._errors = []
        self._known_managed = {}

        def monitor() -> None:
            while not self._stop.is_set():
                try:
                    self._samples.append(_v6_gpu_sample(managed_token=self.managed_token, known_managed=self._known_managed))
                except Exception as exc:
                    self._errors.append(f'{type(exc).__name__}: {exc}')
                self._stop.wait(2.0)
        self._thread = threading.Thread(target=monitor, name=f'v6-gpu-audit-step-{step}', daemon=True)
        self._thread.start()

    def finish(self, step: int) -> dict[str, Any]:
        if self._step != int(step) or self._thread is None:
            raise V6SftError(f'no active managed GPU audit for step {step}')
        self._stop.set()
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            raise V6SftError(f'managed GPU audit did not stop at step {step}')
        qualifying = []
        for sample in self._samples:
            if all((int(sample['gpus'][str(gpu)]['managed_memory_used_mib']) >= 40960 and len(sample['gpus'][str(gpu)]['managed_pids']) >= 3 for gpu in range(V6_WORLD_SIZE))):
                qualifying.append(sample)
        audit_path = self.run_dir / 'wm_eval' / 'gpu_audit' / f'step_{step:06d}.json'
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'schema_version': 'v6-inline-validation-managed-gpu-audit-v1', 'global_step': int(step), 'managed_token_sha256': _sha256_text(self.managed_token), 'physical_gpus': list(range(V6_WORLD_SIZE)), 'workers_per_gpu': 3, 'minimum_managed_memory_mib': 40960, 'foreign_pid_policy': 'record_only_excluded_from_gate', 'samples': self._samples, 'query_errors': self._errors, 'qualifying_snapshot': qualifying[0] if qualifying else None}
        temporary = audit_path.with_name(f'{audit_path.name}.tmp.{os.getpid()}')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, audit_path)
        self._thread = None
        self._step = None
        if not qualifying:
            raise V6SftError(f'step {step} never reached 3 managed workers and 40960 MiB on every GPU')
        return validate_v6_gpu_audit_file(audit_path, step=step)

def validate_v6_gpu_audit_file(path: Path, *, step: int) -> dict[str, Any]:
    payload = _load_json_object(path, context=f'managed GPU audit step {step}')
    qualifying = payload.get('qualifying_snapshot')
    if payload.get('schema_version') != 'v6-inline-validation-managed-gpu-audit-v1' or int(payload.get('global_step', -1)) != int(step) or payload.get('physical_gpus') != list(range(V6_WORLD_SIZE)) or (int(payload.get('workers_per_gpu', -1)) != 3) or (int(payload.get('minimum_managed_memory_mib', -1)) != 40960) or (payload.get('foreign_pid_policy') != 'record_only_excluded_from_gate') or (not isinstance(qualifying, dict)) or any((int(qualifying['gpus'][str(gpu)]['managed_memory_used_mib']) < 40960 or len(qualifying['gpus'][str(gpu)]['managed_pids']) < 3 for gpu in range(V6_WORLD_SIZE))):
        raise V6SftError(f'managed GPU audit contract failed at step {step}')
    return {'path': str(path), 'sha256': _sha256_file(path), 'managed_token_sha256': payload.get('managed_token_sha256'), 'sample_count': len(payload.get('samples', [])), 'query_error_count': len(payload.get('query_errors', [])), 'qualifying_snapshot': qualifying}

class V6InlineValidationStateGuard:

    def __init__(self) -> None:
        self._active_step: int | None = None
        self._python_rng: object | None = None
        self._numpy_rng: Any = None
        self._torch_cpu_rng: torch.Tensor | None = None
        self._torch_cuda_rng: torch.Tensor | None = None
        self._cuda_device: int | None = None
        self._model_training: bool | None = None
        self._use_cache: Any = None
        self._has_use_cache = False
        self._gradient_checkpointing: bool | None = None
        self._parameter_versions: tuple[tuple[str, int], ...] = ()

    @staticmethod
    def _core_model(model: Any) -> Any:
        core = model
        seen: set[int] = set()
        while hasattr(core, 'module') and id(core) not in seen:
            seen.add(id(core))
            candidate = getattr(core, 'module')
            if candidate is core:
                break
            core = candidate
        return core

    def capture(self, *, step: int, model: Any) -> None:
        if self._active_step is not None:
            raise V6SftError(f'inline validation state guard already active at step {self._active_step}')
        if not torch.cuda.is_available():
            raise V6SftError('inline validation state guard requires CUDA')
        try:
            import numpy as np
        except ImportError as exc:
            raise V6SftError('inline validation state guard requires NumPy') from exc
        core = self._core_model(model)
        config = getattr(core, 'config', None)
        self._active_step = int(step)
        self._python_rng = random.getstate()
        self._numpy_rng = np.random.get_state()
        self._torch_cpu_rng = torch.get_rng_state().clone()
        self._cuda_device = int(torch.cuda.current_device())
        self._torch_cuda_rng = torch.cuda.get_rng_state(self._cuda_device).clone()
        self._model_training = bool(model.training)
        self._has_use_cache = config is not None and hasattr(config, 'use_cache')
        self._use_cache = getattr(config, 'use_cache', None)
        self._gradient_checkpointing = bool(getattr(core, 'is_gradient_checkpointing', False))
        self._parameter_versions = tuple(((name, int(parameter._version)) for (name, parameter) in core.named_parameters() if parameter.requires_grad))

    def restore(self, *, step: int, model: Any) -> None:
        if self._active_step != int(step):
            raise V6SftError(f'inline validation state guard step mismatch: active={self._active_step}, finishing={step}')
        if self._python_rng is None or self._torch_cpu_rng is None or self._torch_cuda_rng is None or (self._cuda_device is None) or (self._model_training is None) or (self._gradient_checkpointing is None):
            raise V6SftError('inline validation state guard has incomplete state')
        try:
            import numpy as np
        except ImportError as exc:
            raise V6SftError('inline validation state guard requires NumPy') from exc
        core = self._core_model(model)
        actual_versions = tuple(((name, int(parameter._version)) for (name, parameter) in core.named_parameters() if parameter.requires_grad))
        if actual_versions != self._parameter_versions:
            raise V6SftError(f'inline validation mutated trainable parameters at step {step}')
        config = getattr(core, 'config', None)
        if self._has_use_cache:
            if config is None or not hasattr(config, 'use_cache'):
                raise V6SftError('inline validation removed model.config.use_cache')
            config.use_cache = self._use_cache
        actual_gc = bool(getattr(core, 'is_gradient_checkpointing', False))
        if actual_gc != self._gradient_checkpointing:
            method_name = 'gradient_checkpointing_enable' if self._gradient_checkpointing else 'gradient_checkpointing_disable'
            method = getattr(core, method_name, None)
            if not callable(method):
                raise V6SftError(f'inline validation changed gradient checkpointing without {method_name}')
            method()
        model.train(self._model_training)
        random.setstate(self._python_rng)
        np.random.set_state(self._numpy_rng)
        torch.set_rng_state(self._torch_cpu_rng)
        torch.cuda.set_rng_state(self._torch_cuda_rng, self._cuda_device)
        if not torch.equal(torch.get_rng_state(), self._torch_cpu_rng) or not torch.equal(torch.cuda.get_rng_state(self._cuda_device), self._torch_cuda_rng):
            raise V6SftError(f'inline validation RNG restoration failed at step {step}')
        self._active_step = None
        self._python_rng = None
        self._numpy_rng = None
        self._torch_cpu_rng = None
        self._torch_cuda_rng = None
        self._cuda_device = None
        self._model_training = None
        self._use_cache = None
        self._has_use_cache = False
        self._gradient_checkpointing = None
        self._parameter_versions = ()

def build_v6_state_guard_start_callback(TrainerCallback: Any, *, guard: V6InlineValidationStateGuard, allowed_steps: Sequence[int]=V6_VALIDATION_STEPS):
    allowed_step_set = frozenset((int(step) for step in allowed_steps))

    class V6StateGuardStartCallback(TrainerCallback):

        def on_save(self, training_args, state, control, **kwargs):
            del training_args
            step = int(state.global_step)
            if step in allowed_step_set:
                model = kwargs.get('model')
                if model is None:
                    raise V6SftError('state guard start callback did not receive model')
                guard.capture(step=step, model=model)
            return control
    return V6StateGuardStartCallback()

def build_v6_state_guard_finish_callback(TrainerCallback: Any, *, guard: V6InlineValidationStateGuard, allowed_steps: Sequence[int]=V6_VALIDATION_STEPS):
    allowed_step_set = frozenset((int(step) for step in allowed_steps))

    class V6StateGuardFinishCallback(TrainerCallback):

        def on_save(self, training_args, state, control, **kwargs):
            del training_args
            step = int(state.global_step)
            if step in allowed_step_set:
                model = kwargs.get('model')
                if model is None:
                    raise V6SftError('state guard finish callback did not receive model')
                guard.restore(step=step, model=model)
            return control
    return V6StateGuardFinishCallback()

def build_v6_gpu_audit_start_callback(TrainerCallback: Any, *, rank: int, audit: V6ManagedGpuAudit, allowed_steps: Sequence[int]=V6_VALIDATION_STEPS):
    allowed_step_set = frozenset((int(step) for step in allowed_steps))

    class V6GpuAuditStartCallback(TrainerCallback):

        def on_save(self, training_args, state, control, **kwargs):
            del training_args, kwargs
            step = int(state.global_step)
            if rank == 0 and step in allowed_step_set:
                audit.start(step)
            return control
    return V6GpuAuditStartCallback()

def _run_transition_metric_shards(*, dataset_jsonl: Path, predictions_jsonl: Path, cache_root: Path, output_root: Path, model_name: str, log_dir: Path) -> Path:
    """Score one completed Validation500 on all six physical GPUs."""
    project_root = Path(__file__).resolve().parents[3]
    scorer = project_root / 'scripts/evaluation/datasetV4_eval_step/score_transition_metrics.py'
    if not scorer.is_file():
        raise V6SftError(f'transition scorer is missing: {scorer}')
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[Any], Any, Path]] = []
    for shard_index in range(V6_WORLD_SIZE):
        log_path = log_dir / f'score_rank{shard_index}.log'
        log_handle = log_path.open('w', encoding='utf-8')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(shard_index)
        env['PYTHONPATH'] = str(project_root / 'src')
        command = [sys.executable, str(scorer), 'score-shard', '--split', 'validation', '--dataset-jsonl', str(dataset_jsonl), '--cache-root', str(cache_root), '--output-root', str(output_root), '--model-name', model_name, '--predictions-jsonl', str(predictions_jsonl), '--prediction-field', 'model_output_next_md', '--prediction-mode', 'next', '--physical-gpu', str(shard_index), '--shard-index', str(shard_index), '--num-shards', str(V6_WORLD_SIZE)]
        process = subprocess.Popen(command, cwd=project_root, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((process, log_handle, log_path))
    failures = []
    for (process, log_handle, log_path) in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append({'pid': process.pid, 'return_code': return_code, 'log': str(log_path)})
    if failures:
        raise V6SftError(f'transition score shards failed: {failures}')
    merge_log = log_dir / 'merge.log'
    merge_command = [sys.executable, str(scorer), 'merge-scores', '--split', 'validation', '--dataset-jsonl', str(dataset_jsonl), '--cache-root', str(cache_root), '--output-root', str(output_root), '--model-name', model_name, '--prediction-field', 'model_output_next_md', '--prediction-mode', 'next', '--num-shards', str(V6_WORLD_SIZE)]
    with merge_log.open('w', encoding='utf-8') as handle:
        completed = subprocess.run(merge_command, cwd=project_root, env={**os.environ, 'PYTHONPATH': str(project_root / 'src')}, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise V6SftError(f'transition score merge failed rc={completed.returncode}: {merge_log}')
    metrics_path = output_root / model_name / 'validation' / 'metrics.json'
    if not metrics_path.is_file():
        raise V6SftError(f'transition metrics are missing: {metrics_path}')
    return metrics_path

def _validate_v6_transition_gpu_audits(*, transition_metrics_path: Path, dataset_jsonl: Path, predictions_jsonl: Path) -> list[dict[str, Any]]:
    """Bind all six transition-scoring shards to physical CUDA GPUs 0--5."""
    validation_root = transition_metrics_path.parent
    expected_dataset_sha = _sha256_file(dataset_jsonl)
    expected_predictions_sha = _sha256_file(predictions_jsonl)
    artifacts: list[dict[str, Any]] = []
    output_rows = 0
    physical_gpus: set[int] = set()
    for rank in range(V6_WORLD_SIZE):
        shard_path = validation_root / 'shards' / f'per_sample.rank{rank}.jsonl'
        audit_path = shard_path.with_suffix('.audit.json')
        for path in (shard_path, audit_path):
            if path.is_symlink() or not path.is_file():
                raise V6SftError(f'transition scoring shard artifact is missing: {path}')
        audit = _load_json_object(audit_path, context=f'transition scoring GPU audit rank {rank}')
        prediction_files = audit.get('prediction_files')
        expected_prediction = {'path': str(predictions_jsonl), 'sha256': expected_predictions_sha}
        physical_gpu = int(audit.get('physical_gpu', -1))
        rows = int(audit.get('output_rows', -1))
        if audit.get('mode') != 'score_shard' or int(audit.get('shard_index', -1)) != rank or int(audit.get('num_shards', -1)) != V6_WORLD_SIZE or (physical_gpu != rank) or (audit.get('cuda_visible_devices') != str(rank)) or (str(audit.get('torch_device', '')).lower() != 'cuda:0') or ('cuda' not in str(audit.get('embedder_device_verified', '')).lower()) or (audit.get('cpu_fallback_allowed') is not False) or (int(audit.get('dataset_rows', -1)) != 500) or (prediction_files != [expected_prediction]) or (Path(str(audit.get('output', ''))).resolve() != shard_path.resolve()) or (rows != _jsonl_row_count(shard_path)) or (rows <= 0):
            raise V6SftError(f'transition scoring GPU audit contract failed for rank {rank}')
        physical_gpus.add(physical_gpu)
        output_rows += rows
        artifacts.append({'rank': rank, 'physical_gpu': physical_gpu, 'path': str(audit_path), 'sha256': _sha256_file(audit_path), 'output_path': str(shard_path), 'output_sha256': _sha256_file(shard_path), 'output_rows': rows, 'cpu_fallback_allowed': False, 'embedder_device_verified': audit['embedder_device_verified']})
    if physical_gpus != set(range(V6_WORLD_SIZE)) or output_rows != 500:
        raise V6SftError(f'transition scoring must cover 500 rows on physical GPUs 0--5: gpus={sorted(physical_gpus)}, rows={output_rows}')
    return artifacts

def _append_v6_experiment_record(receipt: Mapping[str, Any]) -> None:
    record_path = Path('logs/experiment_notes/exp1/V6-Cur-FullSFT-20260815.md')
    record_path.parent.mkdir(parents=True, exist_ok=True)
    transition = receipt['transition_metrics']
    metrics = transition['metrics']
    all_metrics = metrics['all']
    derived = transition['derived']
    wins = all_metrics['transition_win_rows']
    ties = all_metrics['transition_tie_rows']
    losses = all_metrics['transition_loss_rows']
    row = f"| {receipt['display_name']} | {float(all_metrics['Next EleF1']) * 100:.4f} | {float(derived['Prediction-Current EleF1']) * 100:.4f} | {float(all_metrics['Copy-EleF1']) * 100:.4f} | {float(all_metrics['NTM']) * 100:.4f} | {float(all_metrics['Transition Win Rate']) * 100:.4f} | {wins}/{ties}/{losses} | 500 | true | {receipt['created_at']} | {receipt['receipt_path']} |\n"
    with record_path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = handle.read()
            if f"| {receipt['display_name']} |" in existing:
                return
            handle.seek(0, os.SEEK_END)
            if not existing:
                handle.write('# V6 world-model SFT Validation500\n\nNTM and TWR use all 500 samples.\n\n| model | Next EleF1 | P-Current | Copy-EleF1 | NTM | TWR | W/T/L | completed_judgments | strict_complete | time | receipt |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n')
            handle.write(row)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def finalize_v6_inline_evaluation(*, run_dir: Path, checkpoint_root: Path, dataset_jsonl: Path, transition_cache_root: Path, transition_output_root: Path, step: int, gpu_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Validation500, score all-sample metrics, and atomically commit."""
    if step not in V6_VALIDATION_STEPS:
        raise V6SftError(f'step {step} is not in the inline validation schedule')
    expected_run_dir = CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID
    if run_dir.resolve() != expected_run_dir.resolve():
        raise V6SftError(f'inline validation run_dir must be {expected_run_dir}')
    if checkpoint_root.resolve() != (run_dir / 'checkpoints').resolve():
        raise V6SftError('inline validation checkpoint_root is not canonical')
    if dataset_jsonl.resolve() != CANONICAL_VALIDATION_JSONL.resolve():
        raise V6SftError('inline validation dataset path is not canonical')
    if transition_cache_root.resolve() != CANONICAL_TRANSITION_CACHE_ROOT.resolve():
        raise V6SftError('inline transition cache root is not canonical')
    if transition_output_root.resolve() != (run_dir / 'transition_metrics').resolve():
        raise V6SftError('inline transition output root is not canonical')
    step_dir = run_dir / 'wm_eval' / f'step_{step:06d}' / 'validation'
    receipt_path = step_dir / 'evaluation_receipt.json'
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = validate_v6_inline_evaluation_receipt(run_dir=run_dir, step=step)
        _append_v6_experiment_record(receipt)
        return receipt
    predictions_path = step_dir / 'predictions.jsonl'
    standard_metrics_path = step_dir / 'metrics.json'
    checkpoint_path = checkpoint_root / f'checkpoint-{step}'
    binding_path = checkpoint_path / 'v6_checkpoint_binding.json'
    for path in (predictions_path, standard_metrics_path, binding_path):
        if path.is_symlink() or not path.is_file():
            raise V6SftError(f'inline Validation500 artifact is missing: {path}')
    prediction_rows = _validate_v6_validation500_predictions(dataset_jsonl=dataset_jsonl, predictions_jsonl=predictions_path, step=step)
    standard_metrics = _load_json_object(standard_metrics_path, context='inline Validation500 metrics')
    if int(standard_metrics.get('evaluated_rows', -1)) != 500:
        raise V6SftError('inline Validation500 metrics are incomplete')
    display_name = DISPLAY_NAME_PATTERN.format(step=step)
    transition_metrics_path = _run_transition_metric_shards(dataset_jsonl=dataset_jsonl, predictions_jsonl=predictions_path, cache_root=transition_cache_root, output_root=transition_output_root, model_name=display_name, log_dir=step_dir / 'transition_score_logs')
    transition_metrics = _validate_v6_transition_outputs(transition_metrics_path=transition_metrics_path, predictions_path=predictions_path, source_prediction_rows=prediction_rows)
    transition_gpu_audits = _validate_v6_transition_gpu_audits(transition_metrics_path=transition_metrics_path, dataset_jsonl=dataset_jsonl, predictions_jsonl=predictions_path)
    all_metrics = transition_metrics.get('metrics', {}).get('all', {})
    required_metric_keys = {'Next EleF1', 'Copy-EleF1', 'NTM', 'Transition Win Rate', 'transition_win_rows', 'transition_tie_rows', 'transition_loss_rows'}
    if int(all_metrics.get('rows', -1)) != 500 or not required_metric_keys.issubset(all_metrics):
        raise V6SftError('inline transition metrics are incomplete')
    per_sample_path = Path(str(transition_metrics.get('per_sample_path')))
    prediction_current_values = []
    with per_sample_path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                prediction_current_values.append(float(json.loads(line)['prediction_current_elef1']))
    if len(prediction_current_values) != 500:
        raise V6SftError('inline transition per-sample output must contain 500 rows')
    transition_derived = {'Prediction-Current EleF1': sum(prediction_current_values) / 500, 'per_sample_path': str(per_sample_path), 'per_sample_sha256': _sha256_file(per_sample_path)}
    receipt = {'schema_version': 'v6-full-sft-page-margin-inline-validation-v1', 'status': 'passed', 'strict_complete': True, 'created_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'run_id': V6_FORMAL_RUN_ID, 'loss_profile': V6_LOSS_PROFILE, 'display_name': display_name, 'global_step': step, 'completed_judgments': 500, 'dataset': {'path': str(dataset_jsonl), 'sha256': _sha256_file(dataset_jsonl)}, 'checkpoint': {'path': str(checkpoint_path), 'binding_path': str(binding_path), 'binding_sha256': _sha256_file(binding_path)}, 'validation': {'predictions_path': str(predictions_path), 'predictions_sha256': _sha256_file(predictions_path), 'metrics_path': str(standard_metrics_path), 'metrics_sha256': _sha256_file(standard_metrics_path), 'metrics': standard_metrics}, 'transition_metrics': {'path': str(transition_metrics_path), 'sha256': _sha256_file(transition_metrics_path), 'derived': transition_derived, 'gpu_audits': transition_gpu_audits, **transition_metrics}, 'gpu_contract': {'physical_gpus': list(range(V6_WORLD_SIZE)), 'workers_per_gpu': 3, 'minimum_managed_memory_mib': 40960, 'foreign_pid_policy': 'record_only_excluded_from_gate', 'worker_exit_policy': 'wait_for_all_18_shards', 'cpu_fallback_allowed': False, 'audit': dict(gpu_audit)}, 'receipt_path': str(receipt_path)}
    temporary = receipt_path.with_name(f'{receipt_path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, receipt_path)
    _append_v6_experiment_record(receipt)
    return receipt

def validate_v6_inline_evaluation_receipt(*, run_dir: Path, step: int) -> dict[str, Any]:
    """Deeply validate one committed inline evaluation for resume/completion."""
    if step not in V6_VALIDATION_STEPS:
        raise V6SftError(f'step {step} is outside the validation schedule')
    expected_run_dir = CANONICAL_OUTPUT_ROOT / V6_FORMAL_RUN_ID
    if run_dir.resolve() != expected_run_dir.resolve():
        raise V6SftError(f'inline receipt run_dir must be {expected_run_dir}')
    step_dir = run_dir / 'wm_eval' / f'step_{step:06d}' / 'validation'
    receipt_path = step_dir / 'evaluation_receipt.json'
    receipt = _load_json_object(receipt_path, context=f'inline evaluation receipt step {step}')
    dataset = receipt.get('dataset', {})
    checkpoint = receipt.get('checkpoint', {})
    validation = receipt.get('validation', {})
    transition = receipt.get('transition_metrics', {})
    gpu_audit = receipt.get('gpu_contract', {}).get('audit', {})
    expected_paths = {'dataset': Path(str(dataset.get('path'))), 'binding': Path(str(checkpoint.get('binding_path'))), 'predictions': Path(str(validation.get('predictions_path'))), 'validation_metrics': Path(str(validation.get('metrics_path'))), 'transition_metrics': Path(str(transition.get('path'))), 'per_sample': Path(str(transition.get('derived', {}).get('per_sample_path'))), 'gpu_audit': Path(str(gpu_audit.get('path')))}
    canonical_paths = {'dataset': CANONICAL_VALIDATION_JSONL, 'binding': run_dir / 'checkpoints' / f'checkpoint-{step}' / 'v6_checkpoint_binding.json', 'predictions': step_dir / 'predictions.jsonl', 'validation_metrics': step_dir / 'metrics.json', 'transition_metrics': run_dir / 'transition_metrics' / DISPLAY_NAME_PATTERN.format(step=step) / 'validation/metrics.json', 'per_sample': run_dir / 'transition_metrics' / DISPLAY_NAME_PATTERN.format(step=step) / 'validation/per_sample.jsonl', 'gpu_audit': run_dir / 'wm_eval/gpu_audit' / f'step_{step:06d}.json'}
    path_mismatches = {name: {'expected': str(canonical), 'actual': str(expected_paths[name])} for (name, canonical) in canonical_paths.items() if expected_paths[name].resolve() != canonical.resolve()}
    if path_mismatches:
        raise V6SftError(f'inline receipt path drift: {path_mismatches}')
    for (name, path) in expected_paths.items():
        if path.is_symlink() or not path.is_file():
            raise V6SftError(f'inline receipt dependency missing: {name}={path}')
    validated_gpu_audit = validate_v6_gpu_audit_file(expected_paths['gpu_audit'], step=step)
    prediction_rows = _validate_v6_validation500_predictions(dataset_jsonl=expected_paths['dataset'], predictions_jsonl=expected_paths['predictions'], step=step)
    transition_payload = _validate_v6_transition_outputs(transition_metrics_path=expected_paths['transition_metrics'], predictions_path=expected_paths['predictions'], source_prediction_rows=prediction_rows)
    validated_transition_gpu_audits = _validate_v6_transition_gpu_audits(transition_metrics_path=expected_paths['transition_metrics'], dataset_jsonl=expected_paths['dataset'], predictions_jsonl=expected_paths['predictions'])
    standard_metrics = _load_json_object(expected_paths['validation_metrics'], context='inline Validation500 metrics')
    transition_all = transition_payload.get('metrics', {}).get('all', {})
    if receipt.get('schema_version') != 'v6-full-sft-page-margin-inline-validation-v1' or receipt.get('status') != 'passed' or receipt.get('strict_complete') is not True or (receipt.get('run_id') != V6_FORMAL_RUN_ID) or (receipt.get('loss_profile') != V6_LOSS_PROFILE) or (receipt.get('display_name') != DISPLAY_NAME_PATTERN.format(step=step)) or (int(receipt.get('global_step', -1)) != step) or (int(receipt.get('completed_judgments', -1)) != 500) or (_jsonl_row_count(expected_paths['predictions']) != 500) or (_jsonl_row_count(expected_paths['per_sample']) != 500) or (dict(gpu_audit) != validated_gpu_audit) or (transition.get('gpu_audits') != validated_transition_gpu_audits) or (int(standard_metrics.get('evaluated_rows', -1)) != 500) or (int(standard_metrics.get('invalid_action_rows', -1)) != 0) or (set(standard_metrics.get('by_action_bucket', {})) != set(V6_ACTION_BUCKETS)) or (not math.isclose(float(standard_metrics.get('EleF1', float('nan'))), float(transition_all.get('Next EleF1', float('nan'))), rel_tol=0.0, abs_tol=1e-08)):
        raise V6SftError(f'inline evaluation receipt contract failed at step {step}')
    return receipt

def finalize_v6_certify_resident_validation(*, run_dir: Path, checkpoint_root: Path, dataset_jsonl: Path, step: int, gpu_audit: Mapping[str, Any]) -> dict[str, Any]:
    if step not in V6_CERTIFY_VALIDATION_STEPS:
        raise V6SftError(f'step {step} is not the certify validation step')
    expected_run_dir = CANONICAL_CERTIFY_OUTPUT_ROOT / V6_CERTIFY_RUN_ID
    if run_dir.resolve() != expected_run_dir.resolve():
        raise V6SftError(f'certify validation run_dir must be {expected_run_dir}')
    if checkpoint_root.resolve() != (run_dir / 'checkpoints').resolve():
        raise V6SftError('certify validation checkpoint_root is not canonical')
    if dataset_jsonl.resolve() != CANONICAL_VALIDATION_JSONL.resolve():
        raise V6SftError('certify validation dataset path is not canonical')
    step_dir = run_dir / 'wm_eval' / f'step_{step:06d}' / 'validation'
    receipt_path = step_dir / 'resident_validation_receipt.json'
    if receipt_path.exists() or receipt_path.is_symlink():
        raise V6SftError(f'refusing to overwrite certify resident-validation receipt: {receipt_path}')
    predictions_path = step_dir / 'predictions.jsonl'
    metrics_path = step_dir / 'metrics.json'
    checkpoint_path = checkpoint_root / f'checkpoint-{step}'
    binding_path = checkpoint_path / 'v6_checkpoint_binding.json'
    for path in (predictions_path, metrics_path, binding_path):
        if path.is_symlink() or not path.is_file():
            raise V6SftError(f'certify resident-validation artifact is missing: {path}')
    prediction_rows = _validate_v6_validation500_predictions(dataset_jsonl=dataset_jsonl, predictions_jsonl=predictions_path, step=step)
    metrics = _load_json_object(metrics_path, context='certify resident EleF1 metrics')
    if int(metrics.get('evaluated_rows', -1)) != 500:
        raise V6SftError('certify resident Validation500 metrics are incomplete')
    worker_files = sorted((step_dir / 'workers').glob('predictions_rank*.jsonl'))
    if len(worker_files) != V6_VALIDATION_WORKERS:
        raise V6SftError('certify resident Validation500 requires exactly 18 worker shards')
    worker_rows = sum((_jsonl_row_count(path) for path in worker_files))
    if worker_rows != 500 or len(prediction_rows) != 500:
        raise V6SftError('certify resident worker shards do not cover 500 rows')
    audit_path = Path(str(gpu_audit.get('path') or ''))
    validated_audit = validate_v6_gpu_audit_file(audit_path, step=step)
    if dict(gpu_audit) != validated_audit:
        raise V6SftError('certify resident managed-GPU audit changed after completion')
    receipt = {'schema_version': 'v6-full-sft-resident-validation500-certify-v1', 'status': 'passed', 'strict_complete': True, 'created_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'run_id': V6_CERTIFY_RUN_ID, 'global_step': step, 'completed_judgments': 500, 'resident_training_model': True, 'physical_gpus': list(range(V6_WORLD_SIZE)), 'workers_per_gpu': V6_VALIDATION_WORKERS_PER_GPU, 'worker_shards': V6_VALIDATION_WORKERS, 'dataset_path': str(dataset_jsonl), 'dataset_sha256': _sha256_file(dataset_jsonl), 'checkpoint_path': str(checkpoint_path), 'checkpoint_binding_path': str(binding_path), 'checkpoint_binding_sha256': _sha256_file(binding_path), 'predictions_path': str(predictions_path), 'predictions_sha256': _sha256_file(predictions_path), 'metrics_path': str(metrics_path), 'metrics_sha256': _sha256_file(metrics_path), 'worker_rows': worker_rows, 'managed_gpu_audit': validated_audit, 'cpu_fallback_allowed': False, 'receipt_path': str(receipt_path)}
    temporary = receipt_path.with_name(f'{receipt_path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, receipt_path)
    return receipt

def build_v6_inline_evaluation_receipt_callback(TrainerCallback: Any, *, torch_module: Any, rank: int, run_dir: Path, checkpoint_root: Path, dataset_jsonl: Path, transition_cache_root: Path, transition_output_root: Path, gpu_audit: V6ManagedGpuAudit, certify_run: bool=False):
    """Commit transition metrics and a strict receipt before training resumes."""
    allowed_steps = frozenset(V6_CERTIFY_VALIDATION_STEPS if certify_run else V6_VALIDATION_STEPS)

    class V6InlineEvaluationReceiptCallback(TrainerCallback):

        def on_save(self, training_args, state, control, **kwargs):
            del training_args, kwargs
            step = int(state.global_step)
            if step not in allowed_steps:
                return control
            if rank == 0:
                gpu_evidence = gpu_audit.finish(step)
                if certify_run:
                    receipt = finalize_v6_certify_resident_validation(run_dir=run_dir, checkpoint_root=checkpoint_root, dataset_jsonl=dataset_jsonl, step=step, gpu_audit=gpu_evidence)
                else:
                    receipt = finalize_v6_inline_evaluation(run_dir=run_dir, checkpoint_root=checkpoint_root, dataset_jsonl=dataset_jsonl, transition_cache_root=transition_cache_root, transition_output_root=transition_output_root, step=step, gpu_audit=gpu_evidence)
                print(json.dumps({'event': 'v6_certify_resident_validation_passed' if certify_run else 'v6_inline_validation_passed', 'global_step': step, 'completed_judgments': 500, 'strict_complete': True, 'receipt': receipt['receipt_path']}, ensure_ascii=False), flush=True)
            sft_module = __import__('mkd2world.training.sft', fromlist=['distributed_barrier'])
            sft_module.distributed_barrier(torch_module)
            return control
    return V6InlineEvaluationReceiptCallback()

def validate_v6_checkpoint_callback_order(callbacks: Sequence[Any]) -> None:
    """Require generic integrity to finish before V6 provenance is written."""
    names = [type(callback).__name__ for callback in callbacks]
    try:
        integrity_index = names.index('CheckpointIntegrityCallback')
        binding_index = names.index('V6CheckpointBindingCallback')
    except ValueError as exc:
        raise V6SftError(f'V6 checkpoint callbacks are incomplete: {names}') from exc
    if integrity_index >= binding_index:
        raise V6SftError('V6 checkpoint binding callback must run after generic integrity')
    for callback_name in ('V6StateGuardStartCallback', 'V6GpuAuditStartCallback', 'EleF1EvalCallback', 'V6StateGuardFinishCallback', 'V6InlineEvaluationReceiptCallback'):
        if callback_name in names and names.index(callback_name) <= binding_index:
            raise V6SftError(f'{callback_name} must run after V6 checkpoint binding')
    inline_names = ('V6StateGuardStartCallback', 'V6GpuAuditStartCallback', 'EleF1EvalCallback', 'V6StateGuardFinishCallback', 'V6InlineEvaluationReceiptCallback')
    if any((name in names for name in inline_names)):
        missing = [name for name in inline_names if name not in names]
        if missing:
            raise V6SftError(f'V6 inline validation callbacks are incomplete: missing={missing}')
        indexes = [names.index(name) for name in inline_names]
        if indexes != sorted(indexes) or len(set(indexes)) != len(indexes):
            raise V6SftError('V6 inline callback order must be StateGuardStart -> GPU audit -> Validation500 -> StateGuardFinish -> receipt')
    if 'V6GpuAuditStartCallback' in names and 'EleF1EvalCallback' in names and (names.index('V6GpuAuditStartCallback') >= names.index('EleF1EvalCallback')):
        raise V6SftError('managed GPU auditing must start before Validation500 generation')
    if 'EleF1EvalCallback' in names and 'V6InlineEvaluationReceiptCallback' in names and (names.index('EleF1EvalCallback') >= names.index('V6InlineEvaluationReceiptCallback')):
        raise V6SftError('Validation500 generation/scoring must finish before its strict receipt')

def build_v6_finite_loss_callback(TrainerCallback: Any, *, rank: int):
    """Emit machine-readable evidence and abort on any non-finite logged loss."""

    class V6FiniteLossCallback(TrainerCallback):

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs is None or 'loss' not in logs:
                return control
            loss = float(logs['loss'])
            if not math.isfinite(loss):
                raise V6SftError(f'V6 training emitted non-finite loss: {loss}')
            required_metrics = ('beta', 'eligible_count', 'current_replay_max_error', 'gate_flip_count')
            missing = [name for name in required_metrics if name not in logs]
            if missing:
                raise V6SftError(f'V6 finite-loss event lacks ranking diagnostics: {missing}')
            diagnostics = {name: float(logs[name]) for name in required_metrics}
            if not all((math.isfinite(value) for value in diagnostics.values())) or diagnostics['current_replay_max_error'] > V6_REPLAY_ATOL or diagnostics['gate_flip_count'] != 0.0:
                raise V6SftError(f'V6 training emitted invalid ranking diagnostics: {diagnostics}')
            if rank == 0:
                print(json.dumps({'event': 'v6_loss_finite', 'global_step': int(state.global_step), 'loss': loss, 'focus_loss_active': True, 'loss_ramp': False, **diagnostics}, ensure_ascii=False), flush=True)
            return control
    return V6FiniteLossCallback()

class V6RunStateManager:
    """Publish optimizer progress and keep long inline validation visibly alive."""

    def __init__(self, *, rank: int, max_steps: int, phase: str) -> None:
        self.rank = int(rank)
        self.max_steps = int(max_steps)
        self.training_phase = str(phase)
        self._last_step_at: str | None = None
        self._validation_step: int | None = None
        self._validation_stop = threading.Event()
        self._validation_thread: threading.Thread | None = None
        self._initialization_stop = threading.Event()
        self._initialization_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _write(self, *, status: str, phase: str, step: int, last_error: str | None=None, retryable: bool=False) -> None:
        if self.rank != 0:
            return
        with self._lock:
            write_v6_run_state(status=status, phase=phase, global_step=int(step), max_steps=self.max_steps, last_step_at=self._last_step_at, last_error=last_error, retryable=retryable)

    def start(self, step: int) -> None:
        self.finish_initializing()
        self._last_step_at = datetime.now().astimezone().isoformat(timespec='seconds')
        self._write(status='running', phase=self.training_phase, step=step)

    def start_initializing(self, step: int) -> None:
        if self.rank != 0:
            return
        if self._initialization_thread is not None:
            raise V6SftError('V6 initialization heartbeat is already active')
        self._last_step_at = datetime.now().astimezone().isoformat(timespec='seconds')
        self._initialization_stop.clear()
        self._write(status='running', phase='initializing', step=step)

        def heartbeat() -> None:
            while not self._initialization_stop.wait(30.0):
                self._write(status='running', phase='initializing', step=step)
        self._initialization_thread = threading.Thread(target=heartbeat, name='v6-run-state-initializing', daemon=True)
        self._initialization_thread.start()

    def finish_initializing(self) -> None:
        if self.rank != 0 or self._initialization_thread is None:
            return
        self._initialization_stop.set()
        self._initialization_thread.join(timeout=5.0)
        if self._initialization_thread.is_alive():
            raise V6SftError('V6 initialization heartbeat did not stop')
        self._initialization_thread = None

    def optimizer_step(self, step: int) -> None:
        self._last_step_at = datetime.now().astimezone().isoformat(timespec='seconds')
        self._write(status='running', phase=self.training_phase, step=step)

    def start_validation(self, step: int) -> None:
        if self.rank != 0:
            return
        if self._validation_thread is not None:
            raise V6SftError('V6 run-state validation heartbeat is already active')
        self._validation_step = int(step)
        self._validation_stop.clear()
        self._write(status='running', phase='inline_validation', step=step)

        def heartbeat() -> None:
            while not self._validation_stop.wait(30.0):
                self._write(status='running', phase='inline_validation', step=int(self._validation_step or step))
        self._validation_thread = threading.Thread(target=heartbeat, name=f'v6-run-state-validation-{step}', daemon=True)
        self._validation_thread.start()

    def finish_validation(self, step: int) -> None:
        if self.rank != 0:
            return
        if self._validation_step != int(step) or self._validation_thread is None:
            raise V6SftError(f'V6 validation run-state mismatch at step {step}')
        self._validation_stop.set()
        self._validation_thread.join(timeout=5.0)
        if self._validation_thread.is_alive():
            raise V6SftError('V6 validation run-state heartbeat did not stop')
        self._validation_thread = None
        self._validation_step = None
        self._write(status='running', phase=self.training_phase, step=step)

    def finalizing(self, step: int) -> None:
        self._write(status='running', phase='finalizing', step=step)

    def complete(self, step: int) -> None:
        if self.training_phase == 'certify_training':
            self._write(status='certify_training_complete', phase='certify_finalize', step=step)
        else:
            self._write(status='complete', phase='complete', step=step)

def build_v6_run_state_training_callback(TrainerCallback: Any, *, manager: V6RunStateManager):

    class V6RunStateTrainingCallback(TrainerCallback):

        def on_train_begin(self, training_args, state, control, **kwargs):
            del training_args, kwargs
            manager.start(int(state.global_step))
            return control

        def on_step_end(self, training_args, state, control, **kwargs):
            del training_args, kwargs
            manager.optimizer_step(int(state.global_step))
            return control

        def on_train_end(self, training_args, state, control, **kwargs):
            del training_args, kwargs
            manager.finalizing(int(state.global_step))
            return control
    return V6RunStateTrainingCallback()

def build_v6_run_state_validation_callback(TrainerCallback: Any, *, manager: V6RunStateManager, allowed_steps: Sequence[int], finish: bool):
    allowed = frozenset((int(step) for step in allowed_steps))

    class V6RunStateValidationCallback(TrainerCallback):

        def on_save(self, training_args, state, control, **kwargs):
            del training_args, kwargs
            step = int(state.global_step)
            if step in allowed:
                if finish:
                    manager.finish_validation(step)
                else:
                    manager.start_validation(step)
            return control
    return V6RunStateValidationCallback()

def run_v6_training(args: argparse.Namespace) -> Path:
    """Run the independent V6-FullSFT entrypoint after phase-one approval."""
    from mkd2world.prompts import PromptRegistry
    from mkd2world.training import sft as sft_module
    validate_v6_training_args(args, require_ddp=True)
    rank = int(os.environ.get('RANK', '0'))
    initial_step = _checkpoint_step(Path(args.resume_from_checkpoint)) if args.resume_from_checkpoint is not None else 0
    run_state_manager = V6RunStateManager(rank=rank, max_steps=V6_CERTIFY_EXPECTED_STEPS if args.certify_run else EXPECTED_FINAL_STEP, phase='certify_training' if args.certify_run else 'training')
    run_state_manager.start_initializing(initial_step)
    if args.certify_run:
        resume_report = {'mode': 'new_certify_run', 'policy': 'certify_checkpoint_forbidden_for_formal_resume', 'resume_from_checkpoint': None}
        certify_receipt_report = None
    else:
        resume_report = validate_v6_resume_checkpoint(args)
        certify_receipt_report = validate_v6_certify_receipt(args.certify_receipt)
    preflight_path = CANONICAL_V6_LOG_ROOT / 'preflight.json'
    preflight = _load_json_object(preflight_path, context='V6 CPU preflight')
    if preflight.get('status') != 'passed' or preflight.get('can_start_training') is not True or preflight.get('formal_training_started') is not False:
        raise V6SftError('V6 CPU preflight is missing or not passed')
    data_report = dict(preflight.get('data') or {})
    token_report = dict(preflight.get('token_alignment') or {})
    collator_report = dict(preflight.get('collator_alignment') or {})
    base_model_report = dict(preflight.get('base_model') or {})
    if data_report.get('rows') != EXPECTED_TRAINING_ROWS or token_report.get('status') != 'passed' or token_report.get('rows') != EXPECTED_TRAINING_ROWS or (token_report.get('rank_eligible_rows') != 25005) or (collator_report.get('status') != 'passed'):
        raise V6SftError('V6 CPU preflight report violates the frozen contract')
    sidecars = load_focus_sidecar(args.focus_sidecar)
    if len(sidecars) != EXPECTED_TRAINING_ROWS:
        raise V6SftError('V20 focus sidecar row count drifted after preflight')
    data_report['current_pages'] = {'rows': EXPECTED_TRAINING_ROWS, 'base_units': 1240544, 'max_base_units': 146, 'transition_counts': {'changed': 25007, 'near_noop': 4296, 'exact_noop': 1942}, 'rank_eligible_rows': 25005, 'identity_changed_ids': ['458140f82be74180495b5e61886ad6863f234252a75010a81f49400dd41fd1b6', 'ecc7ccd28c88f1ad27f97c4d0cba9483e6ab46660bbc9c2d4bab5dedec45bc26'], 'source_field': 'current_md', 'validated_by': str(preflight_path)}
    current_sidecars: dict[str, ValidatedSidecar] = {}
    (torch_module, _load_dataset, _LoraConfig, PeftModel, _get_peft_model, Image, AutoModelForImageTextToText, AutoProcessor, _Trainer, TrainerCallback, TrainingArguments) = sft_module.import_training_dependencies()
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch_module.cuda.is_available():
        raise V6SftError('formal V6-FullSFT requires CUDA')
    torch_module.cuda.set_device(local_rank)
    is_main_process = rank == 0
    init_adapter_report = validate_v3_cur_1200_init_adapter(args.init_adapter) if is_main_process else v6_adapter_initialization_contract()
    run_id = str(args.run_id)
    processor = AutoProcessor.from_pretrained(args.base_model, min_pixels=args.image_min_pixels, max_pixels=args.image_max_pixels, trust_remote_code=True)
    data_report['token_alignment'] = token_report
    base_collator = sft_module.LocalWMSftCollator(processor=processor, image_module=Image, max_length=args.max_length, max_output_tokens=args.max_output_tokens, assistant_loss_scope='full', min_supervised_tokens=args.min_supervised_tokens, prompt_key='localwm_next')
    data_report['collator_alignment'] = collator_report
    wm_eval_rows: list[dict[str, str]] = []
    wm_eval_length_validation: dict[str, Any] | None = None
    validation_steps = V6_CERTIFY_VALIDATION_STEPS if args.certify_run else V6_VALIDATION_STEPS
    if validation_steps:
        wm_eval_rows = sft_module.load_wm_eval_rows(Path(args.wm_eval_jsonl), max_samples=args.wm_eval_max_samples)
        if len(wm_eval_rows) != 500:
            raise V6SftError(f'V6 inline Validation500 requires 500 rows, got {len(wm_eval_rows)}')
        wm_eval_length_validation = sft_module.build_wm_eval_length_validation(wm_eval_rows, processor=processor, max_length=args.max_length, max_output_tokens=args.max_output_tokens, prompt_key=args.prompt_key)
        if wm_eval_length_validation.get('status') != 'passed':
            raise V6SftError('V6 inline Validation500 violates the length contract')
    certify_plan_report = None
    if args.certify_run:
        certify_plan_report = validate_v6_certify_sample_plan(args.certify_sample_plan, args.train_jsonl, sidecars, max_target_sample_id=token_report['max_target_sample_id'], max_current_target_sample_id=token_report['max_current_target_sample_id'])
    distributed_module = torch_module.distributed
    if not distributed_module.is_initialized():
        distributed_module.init_process_group(backend='nccl', device_id=torch_module.device('cuda', local_rank), timeout=timedelta(seconds=args.wm_eval_ddp_timeout_seconds))
    distributed_module.barrier()
    output_dir = Path(args.output_dir) / run_id
    checkpoint_dir = output_dir / 'checkpoints'
    final_dir = output_dir / 'final_adapter'
    if args.resume_from_checkpoint is None:
        _create_distributed_run_directory(output_dir, is_main_process=is_main_process, torch_module=torch_module, distributed_module=distributed_module, status_device=torch_module.device('cuda', local_rank))
    else:
        if output_dir.resolve(strict=True) != Path(args.resume_from_checkpoint).resolve(strict=True).parent.parent:
            raise V6SftError('resume checkpoint/run directory changed after preflight')
        distributed_module.barrier()
    if args.certify_run:
        train_dataset = StrictV6CertifyDataset(args.train_jsonl, certify_plan_report['selected_sample_ids'])
        if len(train_dataset) != V6_CERTIFY_ROWS:
            raise V6SftError('strict certify dataset row count changed after preflight')
    else:
        train_dataset = StrictV6JsonlDataset(args.train_jsonl)
        if len(train_dataset) != data_report['rows']:
            raise V6SftError('strict dataset row count changed after preflight')
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
    target_modules = sft_module.resolve_lora_target_modules(model, args.target_modules)
    if len(target_modules) != EXPECTED_EXPANDED_MODULES:
        raise V6SftError('base model no longer exposes 358 Expanded-LoRA modules')
    model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    loaded_adapter_report = validate_loaded_expanded_modules(model, target_modules)
    if is_main_process and hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    if is_main_process:
        print(json.dumps({'event': 'v6_trainable_parameters', 'trainable_parameters': loaded_adapter_report['trainable_parameters'], 'expanded_lora_modules': loaded_adapter_report['module_count'], 'trainable_tensor_count': loaded_adapter_report['trainable_tensor_count'], 'target_modules_sha256': loaded_adapter_report['target_modules_sha256'], 'adapter_initialization_mode': V6_ADAPTER_INIT_MODE, 'init_adapter': str(args.init_adapter), 'init_adapter_weights_sha256': EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256, 'single_adapter': True, 'only_lora_parameters_trainable': True, 'base_parameters_frozen': True}, ensure_ascii=False), flush=True)
    prompt_path = PromptRegistry.from_project().path_for(args.prompt_key)
    next_collator = V6Collator(base_collator, sidecars)
    current_collator = V6ExplicitCurrentCollator(processor=processor, image_module=Image, max_length=args.max_length, max_output_tokens=args.max_output_tokens, min_supervised_tokens=args.min_supervised_tokens, sidecars_by_id=current_sidecars)
    collator = V6PairedCollator(next_collator, current_collator)
    training_args = sft_module.build_training_arguments(TrainingArguments, args, checkpoint_dir, has_eval=False, distributed=True)
    configure_v6_checkpoint_saving_policy(training_args, certify_run=bool(args.certify_run))
    callbacks = [sft_module.build_interval_override_callback(TrainerCallback, args, has_eval=False), sft_module.build_training_timing_callback(TrainerCallback, args), build_v6_finite_loss_callback(TrainerCallback, rank=rank), build_v6_run_state_training_callback(TrainerCallback, manager=run_state_manager), build_v6_extra_checkpoint_callback(TrainerCallback, certify_run=bool(args.certify_run)), sft_module.build_checkpoint_integrity_callback(TrainerCallback, torch_module=torch_module, checkpoint_root=checkpoint_dir, rank=rank, world_size=world_size), build_v6_checkpoint_binding_callback(TrainerCallback, torch_module=torch_module, checkpoint_root=checkpoint_dir, run_dir=output_dir, run_id=run_id, rank=rank)]
    if wm_eval_rows:
        gpu_audit = V6ManagedGpuAudit(run_dir=output_dir, managed_token=os.environ.get('V6_MANAGED_PROCESS_TOKEN', ''))
        state_guard = V6InlineValidationStateGuard()
        callbacks.append(build_v6_run_state_validation_callback(TrainerCallback, manager=run_state_manager, allowed_steps=validation_steps, finish=False))
        callbacks.append(build_v6_state_guard_start_callback(TrainerCallback, guard=state_guard, allowed_steps=validation_steps))
        callbacks.append(build_v6_gpu_audit_start_callback(TrainerCallback, rank=rank, audit=gpu_audit, allowed_steps=validation_steps))
        callbacks.append(sft_module.build_wm_score_eval_callback(TrainerCallback, torch_module=torch_module, processor=processor, image_module=Image, eval_rows=wm_eval_rows, eval_jsonl_path=Path(args.wm_eval_jsonl), length_validation=wm_eval_length_validation, split_name='validation', embedder_cache={}, output_dir=output_dir, checkpoint_root=checkpoint_dir, final_adapter_dir=final_dir, args=args, rank=rank, local_rank=local_rank, world_size=world_size, allowed_steps=validation_steps))
        callbacks.append(build_v6_state_guard_finish_callback(TrainerCallback, guard=state_guard, allowed_steps=validation_steps))
        callbacks.append(build_v6_inline_evaluation_receipt_callback(TrainerCallback, torch_module=torch_module, rank=rank, run_dir=output_dir, checkpoint_root=checkpoint_dir, dataset_jsonl=Path(args.wm_eval_jsonl), transition_cache_root=Path(args.transition_cache_root or CANONICAL_TRANSITION_CACHE_ROOT), transition_output_root=Path(args.transition_output_root or output_dir / 'transition_metrics'), gpu_audit=gpu_audit, certify_run=bool(args.certify_run)))
        callbacks.append(build_v6_run_state_validation_callback(TrainerCallback, manager=run_state_manager, allowed_steps=validation_steps, finish=True))
    validate_v6_checkpoint_callback_order(callbacks)
    if is_main_process and args.resume_from_checkpoint is None:
        run_config = {'schema_version': V6_RUN_CONFIG_SCHEMA, 'run_id': run_id, 'certify_run': bool(args.certify_run), 'display_name_pattern': DISPLAY_NAME_PATTERN, 'created_at': datetime.now().isoformat(timespec='seconds'), 'manifest': {'path': str(CANONICAL_V6_MANIFEST), 'sha256': _sha256_file(CANONICAL_V6_MANIFEST)}, 'v5_parity': getattr(args, '_v6_v5_parity_report'), 'prompt': {'key': args.prompt_key, 'path': str(prompt_path), 'sha256': _sha256_file(prompt_path)}, 'data': data_report, 'model': {'base_model': args.base_model, 'base_model_contract': base_model_report, 'init_adapter': str(args.init_adapter), 'adapter_initialization': v6_adapter_initialization_contract(), 'source_adapter_contract': init_adapter_report, 'loaded_adapter_graph': loaded_adapter_report, 'target_modules': target_modules}, 'resume_policy': {'scope': 'same_run_v6_checkpoint_only', 'initial_resume_from_checkpoint': None}, 'loss': v6_loss_contract(args.base_loss_granularity, args.focus_loss_granularity), 'certify_sample_plan': certify_plan_report, 'certify_receipt_authorization': certify_receipt_report, 'distributed': {'world_size': world_size, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'global_batch_size': V6_GLOBAL_BATCH_SIZE}, 'save_interval': {'steps': V6_SAVE_STEPS, 'checkpoint_steps': list(v6_checkpoint_steps()), 'extra_checkpoint_steps': list(V6_EXTRA_CHECKPOINT_STEPS), 'validation_steps': list(V6_VALIDATION_STEPS), 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'examples': V6_SAVE_INTERVAL_EXAMPLES, 'equivalent_historical_recipe': '100*60=6000'}, 'completion_contract': {'canonical_training_rows': V6_CERTIFY_ROWS, 'epochs': 1.0, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'expected_final_step': V6_CERTIFY_EXPECTED_STEPS, 'expected_trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'formal_resume_allowed': False} if args.certify_run else v6_formal_completion_contract(), 'args': vars(args), 'focus_labeler': {'schema_version': focus_contract.AUDIT_LABEL_SCHEMA_VERSION, 'training_sidecar_schema_version': focus_contract.TRAINING_SIDECAR_VERSION, 'path': str(Path(focus_contract.focus_labeler.__file__).resolve()), 'sha256': _sha256_file(Path(focus_contract.focus_labeler.__file__).resolve()), 'adapter_path': str(Path(focus_contract.__file__).resolve()), 'adapter_sha256': _sha256_file(Path(focus_contract.__file__).resolve()), 'dependency_path': str(Path(focus_contract.focus_dependency.__file__).resolve()), 'dependency_sha256': _sha256_file(Path(focus_contract.focus_dependency.__file__).resolve())}, 'git_status_short': sft_module.git_status_short()}
        (output_dir / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
        shutil.copyfile(prompt_path, output_dir / 'prompt_snapshot.md')
    distributed_module.barrier()
    trainer = V6Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator, callbacks=callbacks, loss_profile=args.loss_profile, base_loss_granularity=args.base_loss_granularity, focus_loss_granularity=args.focus_loss_granularity)
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint is not None else None)
    final_step = int(trainer.state.global_step)
    finite_logged_losses = [float(entry['loss']) for entry in trainer.state.log_history if isinstance(entry, Mapping) and 'loss' in entry]
    if not finite_logged_losses or not all((math.isfinite(loss) for loss in finite_logged_losses)):
        raise V6SftError('V6 training completion requires at least one finite logged loss')
    expected_final_step = V6_CERTIFY_EXPECTED_STEPS if args.certify_run else EXPECTED_FINAL_STEP
    if final_step != expected_final_step:
        raise V6SftError(f'V6 training final step violates the frozen completion contract: expected={expected_final_step}, actual={final_step}')
    if not args.certify_run:
        actual_checkpoint_steps = []
        for path in checkpoint_dir.glob('checkpoint-*'):
            match = re.fullmatch('checkpoint-(\\d+)', path.name)
            if match and path.is_dir() and (not path.is_symlink()):
                actual_checkpoint_steps.append(int(match.group(1)))
        actual_checkpoint_steps.sort()
        expected_checkpoint_steps = list(v6_checkpoint_steps())
        if actual_checkpoint_steps != expected_checkpoint_steps:
            raise V6SftError(f'formal checkpoint schedule is incomplete or contains extras: expected={expected_checkpoint_steps}, actual={actual_checkpoint_steps}')
        for validation_step in V6_VALIDATION_STEPS:
            validate_v6_inline_evaluation_receipt(run_dir=output_dir, step=validation_step)
    trainer.save_model(str(final_dir))
    sft_module.distributed_barrier(torch_module)
    if trainer.is_world_process_zero():
        sft_module.write_final_adapter_integrity_manifest(final_dir, global_step=final_step)
        final_adapter_report = validate_expanded_adapter_files(final_dir, expected_modules=target_modules)
        processor.save_pretrained(str(final_dir))
        summary = {'status': 'passed', 'run_id': run_id, 'formal_training_started': not args.certify_run, 'display_name': DISPLAY_NAME_PATTERN.format(step=final_step), 'loss_profile': args.loss_profile, 'base_loss_granularity': args.base_loss_granularity, 'focus_loss_granularity': args.focus_loss_granularity, 'finished_at': datetime.now().isoformat(timespec='seconds'), 'final_global_step': final_step, 'expected_final_step': expected_final_step, 'final_adapter': str(final_dir), 'final_adapter_contract': final_adapter_report, 'adapter_initialization': v6_adapter_initialization_contract(), 'loaded_adapter_graph': loaded_adapter_report, 'trainable_parameters': EXPECTED_TRAINABLE_PARAMS, 'trainable_tensor_count': EXPECTED_ADAPTER_TENSORS, 'target_modules_sha256': EXPECTED_TARGET_MODULES_SHA256, 'focus_loss_active': True, 'finite_loss_observed': True, 'resume': resume_report, 'certify_run': bool(args.certify_run), 'certify_sample_plan': certify_plan_report, 'certify_receipt_authorization': certify_receipt_report, 'global_batch_size': V6_GLOBAL_BATCH_SIZE, 'loss_ramp': False, 'checkpoint_steps': list(v6_checkpoint_steps()), 'validation_steps': list(v6_validation_steps()), 'final_adapter_only_step': EXPECTED_FINAL_STEP}
        summary_path = output_dir / 'train_summary.json'
        if summary_path.exists() or summary_path.is_symlink():
            raise V6SftError(f'refusing to overwrite train summary: {summary_path}')
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        latest_path = output_dir / 'latest_successful_checkpoint.txt'
        if latest_path.exists() or latest_path.is_symlink():
            raise V6SftError(f'refusing to overwrite success pointer: {latest_path}')
        latest_path.write_text(str(final_dir) + '\n', encoding='utf-8')
        run_state_manager.complete(final_step)
    sft_module.distributed_barrier(torch_module)
    return final_dir

def _run_v6_manifest_command(argv: list[str]) -> None:
    control = build_v6_command_parser().parse_args(argv)
    manifest = validate_v6_manifest(control.manifest)
    log_root = Path(manifest['artifacts']['log_root'])
    log_root.mkdir(parents=True, exist_ok=True)
    if control.command == 'status':
        report = build_v6_status_report(control.manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report.get('status') == 'invalid':
            raise SystemExit(2)
        return
    if control.command == 'preflight':
        preflight_path = log_root / 'preflight.json'
        if preflight_path.is_file() and (not preflight_path.is_symlink()):
            report = _load_json_object(preflight_path, context='V6 CPU preflight')
            publication = report.get('certify_sample_plan', {}).get('publication')
            implementation = report.get('implementation')
            if report.get('status') != 'passed' or report.get('can_start_training') is not True or (not isinstance(publication, Mapping)) or (not isinstance(implementation, Mapping)):
                raise V6SftError('existing V6 preflight is stale; refuse silent overwrite')
            report = {**report, 'reused_existing_preflight': True}
        else:
            args = build_v6_args_from_manifest('preflight', control.manifest)
            report = build_v6_dry_run_report(args)
            _atomic_write_json(preflight_path, report)
        write_v6_run_state(status='ready', phase='preflight_complete', global_step=0, max_steps=EXPECTED_FINAL_STEP, retryable=False)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    args = build_v6_args_from_manifest(control.command, control.manifest, resume_from_checkpoint=getattr(control, 'resume_from_checkpoint', None))
    try:
        final_dir = sft_module.run_with_distributed_cleanup(lambda : run_v6_training(args))
    except BaseException as exc:
        if int(os.environ.get('RANK', '0')) == 0:
            state_path = log_root / 'run_state.json'
            current_step = 0
            if state_path.is_file() and (not state_path.is_symlink()):
                try:
                    current_step = int(_load_json_object(state_path, context='V6 failed run state').get('global_step', 0))
                except Exception:
                    current_step = 0
            write_v6_run_state(status='failed', phase=f'{control.command}_failed', global_step=current_step, max_steps=V6_CERTIFY_EXPECTED_STEPS if control.command == 'certify' else EXPECTED_FINAL_STEP, last_error=f'{type(exc).__name__}: {exc}', retryable=not isinstance(exc, V6SftError))
            print(json.dumps({'status': 'failed', 'command': control.command, 'run_id': args.run_id, 'error_type': type(exc).__name__, 'error': str(exc)}, ensure_ascii=False), flush=True)
        raise
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps({'status': 'passed', 'command': control.command, 'run_id': args.run_id, 'final_adapter': str(final_dir), 'certify_receipt': str(log_root / 'certify_receipt.json'), 'formal_receipt': str(log_root / 'formal_receipt.json'), 'run_state': str(log_root / 'run_state.json')}, ensure_ascii=False))

def main(argv: list[str] | None=None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in {'preflight', 'certify', 'train', 'status', '-h', '--help'}:
        _run_v6_manifest_command(raw_argv)
        return
    parser = build_v6_arg_parser()
    args = parser.parse_args(raw_argv)
    if args.resume_validation_only:
        try:
            validate_v6_training_args(args, require_ddp=False)
            report = validate_v6_resume_checkpoint(args)
        except Exception as exc:
            print(json.dumps({'status': 'blocked', 'resume_valid': False, 'formal_training_started': False, 'error_type': type(exc).__name__, 'error': str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(2) from exc
        print(json.dumps({'status': 'passed', 'resume_valid': True, 'formal_training_started': False, 'resume': report}, ensure_ascii=False, indent=2))
        return
    if args.dry_run:
        try:
            report = build_v6_dry_run_report(args)
        except Exception as exc:
            print(json.dumps({'status': 'blocked', 'can_start_training': False, 'formal_training_started': False, 'error_type': type(exc).__name__, 'error': str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(2) from exc
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    from mkd2world.training import sft as sft_module
    final_dir = sft_module.run_with_distributed_cleanup(lambda : run_v6_training(args))
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps({'final_adapter': str(final_dir)}, ensure_ascii=False))
__all__ = ['BASE_UNIT_IDS_KEY', 'DISPLAY_NAME_PATTERN', 'FOCUS_UNIT_IDS_KEY', 'IGNORE_UNIT_ID', 'MAX_FOCUS_LAMBDA', 'EXPECTED_FINAL_STEP', 'EXPECTED_TRAINABLE_PARAMS', 'EXPECTED_TRAINING_ROWS', 'V6_ADAPTER_INIT_MODE', 'V6_CUDA_VISIBLE_DEVICES', 'V6_GLOBAL_BATCH_SIZE', 'V6_FORMAL_RUN_ID', 'V6_FORMAL_RECEIPT_SCHEMA', 'V6_SAVE_INTERVAL_EXAMPLES', 'V6_SAVE_STEPS', 'V6_EXTRA_CHECKPOINT_STEPS', 'V6_VALIDATION_STEPS', 'V6_LOSS_PROFILE', 'V6_LOSS_PROFILES', 'V6_CERTIFY_EXPECTED_STEPS', 'V6_CERTIFY_PLAN_SCHEMA', 'V6_CERTIFY_RECEIPT_SCHEMA', 'V6_CERTIFY_ROWS', 'V6_CERTIFY_RUN_ID', 'V6_CERTIFY_CONTRACT_CASES', 'V6_WORLD_SIZE', 'V6Collator', 'V6SftCollator', 'V6SftError', 'V6Trainer', 'ValidatedSidecar', 'canonical_v6_target', 'compute_causal_v6_loss', 'compute_v6_loss', 'load_focus_sidecar', 'map_target_tokens_to_units', 'validate_sidecar', 'StrictV6JsonlDataset', 'StrictV6CertifyDataset', 'build_v6_arg_parser', 'build_v6_args_from_manifest', 'build_v6_command_parser', 'build_v6_status_report', 'build_v6_checkpoint_binding_callback', 'build_v6_dry_run_report', 'build_v6_finite_loss_callback', 'build_v6_extra_checkpoint_callback', 'build_v6_inline_evaluation_receipt_callback', 'build_v6_certify_sample_plan', 'configure_v6_checkpoint_saving_policy', 'configure_v6_loss_profile', 'deep_validate_v6_certify_receipt', 'deep_validate_v6_formal_receipt', 'finalize_v6_inline_evaluation', 'publish_v6_certify_sample_plan', 'run_v6_certify_supervisor', 'run_v6_formal_supervisor', 'run_v6_training', 'validate_expanded_adapter_files', 'validate_bare_base_model', 'validate_v6_checkpoint_callback_order', 'validate_v6_collator_contract', 'validate_v6_data_contract', 'validate_v6_training_args', 'validate_v6_manifest', 'validate_v6_inline_evaluation_receipt', 'validate_v6_resume_checkpoint', 'validate_v6_formal_monitor_evidence', 'validate_v6_formal_receipt', 'validate_v6_certify_receipt', 'validate_v6_certify_sample_plan', 'validate_v6_token_contract', 'v6_loss_contract', 'v6_checkpoint_steps', 'v6_validation_steps', 'v6_adapter_initialization_contract', 'v6_formal_completion_contract', 'write_v6_checkpoint_binding', 'write_v6_run_state']
if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
