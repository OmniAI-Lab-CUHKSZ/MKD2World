from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping
from mkd2world.training.markdown import build_assistant_output
from mkd2world.training.prompts import local_world_model_target
AUDIT_LABEL_SCHEMA_VERSION = 'v4-focus-v20-hover-compact-global-support'
TRAINING_SIDECAR_VERSION = 'local-wm-v4-focus-training-sidecar-v20-v1'

def _load_final_labeler():
    from mkd2world.training import focus_annotator
    if getattr(focus_annotator, 'FOCUS_LABEL_VERSION', None) != AUDIT_LABEL_SCHEMA_VERSION:
        raise RuntimeError('focus matcher schema drifted')
    return focus_annotator
focus_labeler = _load_final_labeler()
ALLOWED_TRANSITION_CLASSES = frozenset({'changed', 'near_noop', 'exact_noop'})
ALLOWED_UPDATE_FIELDS = frozenset({'bbox', 'content', 'state'})
SIDECAR_KEYS = frozenset({'schema_version', 'label_schema_version', 'sample_id', 'transition_class', 'target_sha256', 'focus_units'})
FOCUS_UNIT_KEYS = frozenset({'focus_id', 'kind', 'next_element_index', 'current_element_index', 'fields', 'spans'})
SPAN_KEYS = frozenset({'field', 'start', 'end', 'text', 'text_sha256'})

class V4TrainingSidecarError(ValueError):
    """Raised when the compact V12 training contract is violated."""

@dataclass(frozen=True)
class CharacterSpan:
    start: int
    end: int

@dataclass(frozen=True)
class ValidatedTrainingSidecar:
    sample_id: str
    target_text: str
    base_spans: tuple[CharacterSpan, ...]
    focus_spans: tuple[tuple[CharacterSpan, ...], ...]

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V4TrainingSidecarError(f'{context} must be an integer')
    return value

def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise V4TrainingSidecarError(f'{context} schema keys mismatch: missing={missing}, extra={extra}')

def canonical_training_target(sample: Mapping[str, Any]) -> str:
    """Return the unchanged full next-Markdown assistant target."""
    target_markdown = local_world_model_target(dict(sample), prompt_key='localwm_next')
    if not target_markdown:
        sample_id = str(sample.get('id') or '<unknown>')
        raise V4TrainingSidecarError(f'sample {sample_id} has an empty localwm_next target')
    return build_assistant_output('', target_markdown)

def canonicalize_field_span_pairs(fields: object, spans: object, *, context: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Sort parallel ``(field, span)`` pairs without breaking their binding."""
    if not isinstance(fields, list) or not fields:
        raise V4TrainingSidecarError(f'{context}.fields must be a non-empty list')
    if not isinstance(spans, list) or len(spans) != len(fields):
        raise V4TrainingSidecarError(f'{context}.spans must have the same length as fields')
    pairs: list[tuple[str, dict[str, Any]]] = []
    for (index, (field, span)) in enumerate(zip(fields, spans)):
        if not isinstance(field, str) or not field:
            raise V4TrainingSidecarError(f'{context}.fields[{index}] must be a non-empty string')
        if not isinstance(span, Mapping):
            raise V4TrainingSidecarError(f'{context}.spans[{index}] must be an object')
        if span.get('field') != field:
            raise V4TrainingSidecarError(f'{context} field/span binding mismatch at index {index}')
        pairs.append((field, dict(span)))
    if len({field for (field, _) in pairs}) != len(pairs):
        raise V4TrainingSidecarError(f'{context}.fields must be unique')
    pairs.sort(key=lambda pair: pair[0])
    return ([field for (field, _) in pairs], [span for (_, span) in pairs])

def project_focus_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one rich final annotation and project its loss-relevant fields."""
    rich = dict(annotation)
    try:
        focus_labeler.validate_focus_annotation(rich)
    except focus_labeler.V4FocusError as exc:
        raise V4TrainingSidecarError(f'rich final annotation failed validation: {exc}') from exc
    compact_units: list[dict[str, Any]] = []
    for (index, raw_unit) in enumerate(rich['focus_units']):
        context = f'focus_units[{index}]'
        (fields, spans) = canonicalize_field_span_pairs(raw_unit.get('fields'), raw_unit.get('spans'), context=context)
        compact_units.append({'focus_id': index, 'kind': raw_unit['kind'], 'next_element_index': raw_unit['next_element_index'], 'current_element_index': raw_unit.get('current_element_index'), 'fields': fields, 'spans': spans})
    return {'schema_version': TRAINING_SIDECAR_VERSION, 'label_schema_version': AUDIT_LABEL_SCHEMA_VERSION, 'sample_id': rich['sample_id'], 'transition_class': rich['transition_class'], 'target_sha256': rich['target_sha256'], 'focus_units': compact_units}

def build_training_sidecar(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build one compact row from V12 or emit base-only for canonical no-ops."""
    sample_id = str(row.get('id') or '').strip()
    if not sample_id:
        raise V4TrainingSidecarError('sample is missing id')
    transition_class = str(row.get('transition_class') or '').strip()
    if transition_class not in ALLOWED_TRANSITION_CLASSES:
        raise V4TrainingSidecarError(f'sample {sample_id} has unsupported transition_class={transition_class!r}')
    target = canonical_training_target(row)
    if transition_class == 'changed':
        rich = focus_labeler.build_focus_annotation(dict(row))
        if rich.get('target_next_md') != target:
            raise V4TrainingSidecarError(f'sample {sample_id} V12 target differs from canonical localwm_next target')
        compact = project_focus_annotation(rich)
    else:
        compact = {'schema_version': TRAINING_SIDECAR_VERSION, 'label_schema_version': AUDIT_LABEL_SCHEMA_VERSION, 'sample_id': sample_id, 'transition_class': transition_class, 'target_sha256': _sha256_text(target), 'focus_units': []}
    validate_training_sidecar(row, compact)
    return compact

def _read_span(raw_span: object, *, target_text: str, context: str) -> CharacterSpan:
    if not isinstance(raw_span, Mapping):
        raise V4TrainingSidecarError(f'{context} must be an object')
    _require_exact_keys(raw_span, SPAN_KEYS, context=context)
    start = _require_int(raw_span.get('start'), context=f'{context}.start')
    end = _require_int(raw_span.get('end'), context=f'{context}.end')
    if start < 0 or end <= start or end > len(target_text):
        raise V4TrainingSidecarError(f'{context} is outside target: {start}:{end}/{len(target_text)}')
    expected_text = target_text[start:end]
    if raw_span.get('text') != expected_text:
        raise V4TrainingSidecarError(f'{context}.text does not match target span')
    if raw_span.get('text_sha256') != _sha256_text(expected_text):
        raise V4TrainingSidecarError(f'{context}.text_sha256 does not match target span')
    return CharacterSpan(start=start, end=end)

def validate_training_sidecar(sample: Mapping[str, Any], raw_sidecar: Mapping[str, Any]) -> ValidatedTrainingSidecar:
    """Validate the compact sidecar against the canonical full next target."""
    sample_id = str(sample.get('id') or '').strip()
    if not sample_id:
        raise V4TrainingSidecarError('training sample is missing id')
    _require_exact_keys(raw_sidecar, SIDECAR_KEYS, context=f'sample {sample_id} sidecar')
    if str(raw_sidecar.get('sample_id') or '').strip() != sample_id:
        raise V4TrainingSidecarError(f'sample {sample_id} sidecar id mismatch')
    if raw_sidecar.get('schema_version') != TRAINING_SIDECAR_VERSION:
        raise V4TrainingSidecarError(f'sample {sample_id} requires schema_version {TRAINING_SIDECAR_VERSION!r}')
    if raw_sidecar.get('label_schema_version') != AUDIT_LABEL_SCHEMA_VERSION:
        raise V4TrainingSidecarError(f'sample {sample_id} requires label_schema_version {AUDIT_LABEL_SCHEMA_VERSION!r}')
    transition_class = str(raw_sidecar.get('transition_class') or '').strip()
    if transition_class not in ALLOWED_TRANSITION_CLASSES:
        raise V4TrainingSidecarError(f'sample {sample_id} has invalid sidecar transition_class')
    sample_transition = str(sample.get('transition_class') or '').strip()
    if sample_transition and sample_transition != transition_class:
        raise V4TrainingSidecarError(f'sample {sample_id} transition_class mismatch')
    target = canonical_training_target(sample)
    if raw_sidecar.get('target_sha256') != _sha256_text(target):
        raise V4TrainingSidecarError(f'sample {sample_id} target_sha256 mismatch')
    try:
        parsed_target = focus_labeler.parse_markdown_spans(target)
        parsed_current = focus_labeler.parse_markdown_spans(sample.get('current_md'))
    except focus_labeler.V4FocusError as exc:
        raise V4TrainingSidecarError(f'sample {sample_id} has non-canonical Markdown: {exc}') from exc
    if parsed_target.text != target:
        raise V4TrainingSidecarError(f'sample {sample_id} target is not canonical final-labeler Markdown')
    raw_units = raw_sidecar.get('focus_units')
    if not isinstance(raw_units, list):
        raise V4TrainingSidecarError(f'sample {sample_id} focus_units must be a list')
    if transition_class != 'changed' and raw_units:
        raise V4TrainingSidecarError(f'sample {sample_id} non-changed transition must be base-only')
    base_spans = tuple((CharacterSpan(span.start, span.end) for span in parsed_target.base_units))
    cursor = 0
    for (index, span) in enumerate(base_spans):
        if span.start != cursor:
            raise V4TrainingSidecarError(f'sample {sample_id} base supervision gap/overlap before unit {index}: expected_start={cursor}, actual_start={span.start}')
        cursor = span.end
    if cursor != len(target):
        raise V4TrainingSidecarError(f'sample {sample_id} base supervision ends at {cursor}, target ends at {len(target)}')
    focus_spans: list[tuple[CharacterSpan, ...]] = []
    seen_next: set[int] = set()
    occupied: list[tuple[CharacterSpan, int]] = []
    for (focus_index, raw_unit) in enumerate(raw_units):
        context = f'sample {sample_id} focus_units[{focus_index}]'
        if not isinstance(raw_unit, Mapping):
            raise V4TrainingSidecarError(f'{context} must be an object')
        _require_exact_keys(raw_unit, FOCUS_UNIT_KEYS, context=context)
        if _require_int(raw_unit.get('focus_id'), context=f'{context}.focus_id') != focus_index:
            raise V4TrainingSidecarError(f'{context}.focus_id must be contiguous')
        kind = raw_unit.get('kind')
        if kind not in {'ADD', 'UPDATE'}:
            raise V4TrainingSidecarError(f'{context} has unsupported kind {kind!r}')
        next_index = _require_int(raw_unit.get('next_element_index'), context=f'{context}.next_element_index')
        if next_index < 0 or next_index >= len(parsed_target.elements):
            raise V4TrainingSidecarError(f'{context} next element is out of range')
        if next_index in seen_next:
            raise V4TrainingSidecarError(f'{context} duplicates a next element')
        seen_next.add(next_index)
        (fields, sorted_spans) = canonicalize_field_span_pairs(raw_unit.get('fields'), raw_unit.get('spans'), context=context)
        if raw_unit.get('fields') != fields or raw_unit.get('spans') != sorted_spans:
            raise V4TrainingSidecarError(f'{context} fields and bound spans must use canonical field order')
        element = parsed_target.elements[next_index]
        if kind == 'ADD':
            if raw_unit.get('current_element_index') is not None:
                raise V4TrainingSidecarError(f'{context} ADD current index must be null')
            if fields != ['line']:
                raise V4TrainingSidecarError(f'{context} ADD must focus the full line')
            expected_by_field = {'line': element.line_span}
        else:
            current_index = _require_int(raw_unit.get('current_element_index'), context=f'{context}.current_element_index')
            if current_index < 0 or current_index >= len(parsed_current.elements):
                raise V4TrainingSidecarError(f'{context} current element is out of range')
            if not set(fields).issubset(ALLOWED_UPDATE_FIELDS):
                raise V4TrainingSidecarError(f'{context} UPDATE permits only bbox/content/state')
            expected_by_field = {'bbox': element.bbox_span, 'content': element.content_span, 'state': element.state_span}
        unit_spans: list[CharacterSpan] = []
        owner_indexes: set[int] = set()
        for (span_index, (field, raw_span)) in enumerate(zip(fields, sorted_spans)):
            span = _read_span(raw_span, target_text=target, context=f'{context}.spans[{span_index}]')
            expected = expected_by_field.get(field)
            if expected is None or (span.start, span.end) != (expected.start, expected.end):
                raise V4TrainingSidecarError(f'{context}.spans[{span_index}] does not match target field {field}')
            owners = [index for (index, base) in enumerate(base_spans) if span.start >= base.start and span.end <= base.end]
            if len(owners) != 1:
                raise V4TrainingSidecarError(f'{context}.spans[{span_index}] crosses a base unit')
            owner_indexes.add(owners[0])
            for (previous, previous_group) in occupied:
                if span.start < previous.end and span.end > previous.start:
                    raise V4TrainingSidecarError(f'{context}.spans[{span_index}] overlaps focus group {previous_group}')
            occupied.append((span, focus_index))
            unit_spans.append(span)
        if len(owner_indexes) != 1:
            raise V4TrainingSidecarError(f'{context} spans more than one base unit')
        focus_spans.append(tuple(unit_spans))
    return ValidatedTrainingSidecar(sample_id=sample_id, target_text=target, base_spans=base_spans, focus_spans=tuple(focus_spans))
__all__ = ['ALLOWED_TRANSITION_CLASSES', 'AUDIT_LABEL_SCHEMA_VERSION', 'CharacterSpan', 'TRAINING_SIDECAR_VERSION', 'V4TrainingSidecarError', 'ValidatedTrainingSidecar', 'build_training_sidecar', 'canonical_training_target', 'canonicalize_field_span_pairs', 'project_focus_annotation', 'validate_training_sidecar']
