from __future__ import annotations
import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence
from urllib.parse import urlparse
from openai import AsyncOpenAI
from PIL import Image, ImageDraw
from playwright.async_api import BrowserContext, Page, async_playwright
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mkd2world.actions import action_to_prompt_dict
from mkd2world.prompts import PromptRegistry
SUPPORTED_ACTIONS = 'left_click,double_click,right_click,type_text,scroll,key_press,hotkey,navigate,go_back,go_forward,refresh,wait,select_option,toggle_check,drag,hover,move_to,clear,finish,fail'
WORLD_MODEL_MAX_OUTPUT_TOKENS = 6000
P2_DIGEST_MAX_PREDICTED_MD_LINE_CHARS = 512
COORDINATE_ACTIONS = {'left_click', 'double_click', 'right_click', 'toggle_check', 'hover', 'move_to'}
NATIVE_COORDINATE_ACTIONS = {'middle_click', 'triple_click'}
NO_ARG_ACTIONS = {'go_back', 'go_forward', 'refresh', 'wait'}
ALLOWED_ACTIONS = set(SUPPORTED_ACTIONS.split(',')) | NATIVE_COORDINATE_ACTIONS
CANDIDATE_ACTIONS_CORRECTION = 'Regenerate the entire candidate_actions array from scratch and validate every unified_action independently; do not preserve any invalid candidate. Use exactly one of these parameter schemas and no extra fields: left_click/double_click/right_click/toggle_check/hover/move_to={type,coordinate}; type_text={type,text} with optional coordinate,clear,enter; scroll={type,delta_x,delta_y}; key_press={type,key}; hotkey={type,keys}; navigate={type,url}; go_back/go_forward/refresh/wait={type} only; select_option={type,coordinate,option_text}; drag={type,from_coordinate,to_coordinate}; clear={type} with optional coordinate; finish={type,answer}; fail={type,reason}. Return 1..4 corrected candidates as JSON.'
ACTION_RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'online_mind2web_gui_action', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['selected_action'], 'properties': {'selected_action': {'type': 'object', 'additionalProperties': False, 'required': ['unified_action', 'action_intent'], 'properties': {'unified_action': {'type': 'object', 'required': ['type'], 'additionalProperties': True, 'properties': {'type': {'type': 'string', 'enum': sorted(ALLOWED_ACTIONS)}}}, 'action_intent': {'type': 'string'}}}}}}}
CANDIDATE_ACTIONS_RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'online_mind2web_gui_action_candidates', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['candidate_actions'], 'properties': {'candidate_actions': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {'type': 'object', 'additionalProperties': False, 'required': ['unified_action', 'action_intent', 'evidence', 'confidence'], 'properties': {'unified_action': {'type': 'object', 'required': ['type'], 'additionalProperties': True, 'properties': {'type': {'type': 'string', 'enum': sorted(ALLOWED_ACTIONS)}}}, 'action_intent': {'type': 'string'}, 'evidence': {'type': 'string'}, 'confidence': {'type': 'integer', 'minimum': 0, 'maximum': 10}}}}}}}}
BEST_ACTION_RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'online_mind2web_gui_best_action', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['best_action'], 'properties': {'best_action': {'type': 'object', 'additionalProperties': False, 'required': ['candidate_index', 'unified_action', 'action_intent', 'evidence', 'confidence'], 'properties': {'candidate_index': {'type': 'integer', 'minimum': 0, 'maximum': 3}, 'unified_action': {'type': 'object', 'required': ['type'], 'additionalProperties': True, 'properties': {'type': {'type': 'string', 'enum': sorted(ALLOWED_ACTIONS)}}}, 'action_intent': {'type': 'string'}, 'evidence': {'type': 'string'}, 'confidence': {'type': 'integer', 'minimum': 0, 'maximum': 10}}}}}}}
V3_LOCAL_CANDIDATE_ACTIONS_RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'online_mind2web_v3_local_gui_action_candidates', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['candidate_actions'], 'properties': {'candidate_actions': {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': {'type': 'object', 'additionalProperties': False, 'required': ['unified_action', 'action_intent', 'confidence'], 'properties': {'unified_action': {'type': 'object', 'required': ['type'], 'additionalProperties': True, 'properties': {'type': {'type': 'string', 'enum': sorted(ALLOWED_ACTIONS)}}}, 'action_intent': {'type': 'string'}, 'confidence': {'type': 'integer', 'minimum': 0, 'maximum': 10}}}}}}}}
V3_LOCAL_BEST_ACTION_RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'online_mind2web_v3_local_gui_best_action', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['best_action'], 'properties': {'best_action': {'type': 'object', 'additionalProperties': False, 'required': ['candidate_index', 'unified_action', 'action_intent'], 'properties': {'candidate_index': {'type': 'integer', 'minimum': 0, 'maximum': 2}, 'unified_action': {'type': 'object', 'required': ['type'], 'additionalProperties': True, 'properties': {'type': {'type': 'string', 'enum': sorted(ALLOWED_ACTIONS)}}}, 'action_intent': {'type': 'string'}}}}}}}
P2_DIGEST_RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'online_mind2web_p2_candidate_digest', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['predicted_change', 'intent_alignment', 'task_progress', 'evidence_refs'], 'properties': {'predicted_change': {'type': 'string'}, 'intent_alignment': {'type': 'string', 'enum': ['yes', 'partial', 'no', 'unclear']}, 'task_progress': {'type': 'string', 'enum': ['positive', 'neutral', 'negative', 'unclear']}, 'evidence_refs': {'type': 'array', 'minItems': 0, 'maxItems': 3, 'items': {'type': 'string'}}}}}}
P24_DIGEST_RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'online_mind2web_p24_candidate_digest', 'strict': True, 'schema': {'type': 'object', 'additionalProperties': False, 'required': ['predicted_change', 'task_progress', 'evidence_refs'], 'properties': {'predicted_change': {'type': 'string'}, 'task_progress': {'type': 'string', 'enum': ['positive', 'neutral', 'negative', 'unclear']}, 'evidence_refs': {'type': 'array', 'minItems': 0, 'maxItems': 3, 'items': {'type': 'string'}}}}}}
BLOCKED_INTENT_RE = re.compile('\\b(place\\s+(?:the\\s+)?order|complete\\s+(?:the\\s+)?purchase|confirm\\s+(?:the\\s+)?payment|pay\\s+now|send\\s+(?:the\\s+)?(?:message|email)|submit\\s+(?:the\\s+)?(?:application|order)|delete\\s+(?:the\\s+)?(?:account|record|item)|book\\s+(?:the\\s+)?(?:flight|hotel)|reserve\\s+(?:the\\s+)?(?:room|table))\\b', re.IGNORECASE)
CAPTCHA_RE = re.compile('captcha|verify you are human|checking your browser|access denied|unusual traffic|security check', re.IGNORECASE)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_task_ids(path: Path) -> list[str]:
    """Load a deterministic task selection from a probe or mask JSON file."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        values = [item.get('task_id', item) if isinstance(item, dict) else item for item in payload]
    elif isinstance(payload, dict):
        values = None
        for key in ('task_ids', 'included_task_ids', 'failed_task_ids', 'excluded_task_ids'):
            if isinstance(payload.get(key), list):
                values = payload[key]
                break
        if values is None and isinstance(payload.get('tasks'), list):
            values = [item['task_id'] for item in payload['tasks'] if isinstance(item, dict) and 'task_id' in item]
        if values is None:
            raise ValueError(f'No task IDs found in task manifest: {path}')
    else:
        raise ValueError(f'Unsupported task manifest: {path}')
    task_ids = [str(value) for value in values]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f'Task manifest contains duplicate task IDs: {path}')
    return task_ids

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(payload, encoding='utf-8')
    os.replace(temporary, path)

def _balanced_json_value(text: str, start: int) -> tuple[object, int] | None:
    """Decode one JSON value beginning at start."""
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None

def _repair_split_selected_action(text: str) -> dict | None:
    """Recover GUI-Owl responses that split selected_action across fragments."""
    if any((text.count(key) != 1 for key in ('"selected_action"', '"unified_action"', '"action_intent"'))):
        return None
    marker = '"unified_action"'
    marker_pos = text.find(marker)
    if marker_pos < 0:
        return None
    colon = text.find(':', marker_pos + len(marker))
    if colon < 0:
        return None
    action_start = colon + 1
    while action_start < len(text) and text[action_start].isspace():
        action_start += 1
    decoded = _balanced_json_value(text, action_start)
    if decoded is None or not isinstance(decoded[0], dict):
        return None
    unified_action = decoded[0]
    intent_marker = '"action_intent"'
    intent_pos = text.find(intent_marker, action_start + decoded[1])
    if intent_pos < 0:
        return None
    intent_colon = text.find(':', intent_pos + len(intent_marker))
    if intent_colon < 0:
        return None
    intent_start = intent_colon + 1
    while intent_start < len(text) and text[intent_start].isspace():
        intent_start += 1
    intent = _balanced_json_value(text, intent_start)
    if intent is None or not isinstance(intent[0], str):
        return None
    return {'selected_action': {'unified_action': unified_action, 'action_intent': intent[0]}}

def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    for suffix in ('<|im_end|>', '<|endoftext|>'):
        while text.endswith(suffix):
            text = text[:-len(suffix)].rstrip()
    if text.startswith('```'):
        text = re.sub('^```(?:json)?\\s*', '', text, flags=re.IGNORECASE)
        text = re.sub('\\s*```$', '', text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _repair_split_selected_action(text)
        if payload is None:
            (start, end) = (text.find('{'), text.rfind('}'))
            if start < 0 or end <= start:
                raise ValueError('model response contains no JSON object')
            try:
                payload = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                raise ValueError('model response contains no valid JSON object')
    if not isinstance(payload, dict) or not isinstance(payload.get('selected_action'), dict):
        raise ValueError('response must contain selected_action object')
    selected = payload['selected_action']
    if os.environ.get('GUI_OWL_COMPAT') == '1' and set(selected) == {'unified_action'}:
        selected['action_intent'] = ''
    action = selected.get('unified_action')
    if not isinstance(action, dict):
        raise ValueError('selected_action.unified_action must be an object')
    action_type = action.get('type')
    if action_type not in ALLOWED_ACTIONS:
        raise ValueError(f'unsupported action type: {action_type!r}')
    validate_action(action)
    if set(selected) != {'unified_action', 'action_intent'}:
        raise ValueError('selected_action must contain exactly unified_action and action_intent')
    if not isinstance(selected.get('action_intent'), str):
        raise ValueError('selected_action.action_intent must be a string')
    return payload

def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub('^```(?:json)?\\s*', '', text, flags=re.IGNORECASE)
        text = re.sub('\\s*```$', '', text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        (start, end) = (text.find('{'), text.rfind('}'))
        if start < 0 or end <= start:
            raise ValueError('model response contains no JSON object')
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError('model response must be a JSON object')
    return payload

def _floor_action_numbers(action: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize numeric GUI action parameters with floor semantics."""
    normalized = dict(action)
    for field in ('coordinate', 'from_coordinate', 'to_coordinate'):
        value = normalized.get(field)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            if all((not isinstance(item, bool) and isinstance(item, (int, float)) for item in value)):
                normalized[field] = [math.floor(float(item)) for item in value]
    if normalized.get('type') == 'scroll':
        for field in ('delta_x', 'delta_y'):
            value = normalized.get(field)
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                normalized[field] = math.floor(float(value))
    return normalized

def _validate_selected_action(selected: Any) -> dict[str, Any]:
    if not isinstance(selected, dict):
        raise ValueError('action candidate must be an object')
    action = selected.get('unified_action')
    if not isinstance(action, dict):
        raise ValueError('unified_action must be an object')
    selected = dict(selected)
    action = dict(action)
    action_type = action.get('type')
    if action_type not in ALLOWED_ACTIONS:
        raise ValueError(f'unsupported action type: {action_type!r}')
    if action_type in NO_ARG_ACTIONS:
        for field in ('coordinate', 'delay', 'delta_x', 'delta_y'):
            action.pop(field, None)
    elif action_type == 'scroll':
        action.pop('coordinate', None)
    action = _floor_action_numbers(action)
    selected['unified_action'] = action
    validate_action(action)
    confidence = selected.get('confidence')
    if not isinstance(confidence, int) or not 0 <= confidence <= 10:
        raise ValueError('confidence must be an integer in [0, 10]')
    for field in ('action_intent', 'evidence'):
        if not isinstance(selected.get(field), str):
            raise ValueError(f'{field} must be a string')
    return selected

def parse_candidate_actions_json(text: str) -> list[dict[str, Any]]:
    payload = _parse_json_object(text)
    candidates = payload.get('candidate_actions')
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
        raise ValueError('candidate_actions must contain 1..4 items')
    return [_validate_selected_action(item) for item in candidates]

def parse_best_action_index(text: str, candidate_count: int) -> int:
    payload = _parse_json_object(text)
    best = payload.get('best_action')
    _validate_selected_action(best)
    try:
        raw_index = int(best.get('candidate_index'))
    except Exception as exc:
        raise ValueError('best_action.candidate_index must be an integer') from exc
    if not 0 <= raw_index < candidate_count:
        raise ValueError(f'candidate_index {raw_index} out of range for {candidate_count} candidates')
    return raw_index

def parse_best_action_selection(text: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:

    if not candidates:
        raise ValueError('best-action selection requires at least one candidate')
    payload = _parse_json_object(text)
    best = _validate_selected_action(payload.get('best_action'))
    reported_index = parse_best_action_index(text, len(candidates))
    best_action = best['unified_action']
    matches = [index for (index, candidate) in enumerate(candidates) if candidate['unified_action'] == best_action]
    if not matches:
        raise ValueError('best_action.unified_action must exactly match one supplied candidate action')
    selected_index = reported_index if reported_index in matches else matches[0]
    return {'selected_index': selected_index, 'reported_index': reported_index, 'index_action_consistent': reported_index in matches, 'matching_indices': matches}

def parse_strict_best_action_selection(text: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selection = parse_best_action_selection(text, candidates)
    if not selection['index_action_consistent']:
        raise ValueError('best_action candidate_index and exact copied unified_action disagree')
    return selection

def _validate_v3_local_candidate(selected: Any) -> dict[str, Any]:
    if not isinstance(selected, dict):
        raise ValueError('V3 Local WM action candidate must be an object')
    if set(selected) != {'unified_action', 'action_intent', 'confidence'}:
        raise ValueError('V3 Local WM action candidate must contain exactly unified_action, action_intent, and confidence')
    action = selected.get('unified_action')
    if not isinstance(action, dict):
        raise ValueError('unified_action must be an object')
    action = dict(action)
    action_type = action.get('type')
    if action_type not in ALLOWED_ACTIONS:
        raise ValueError(f'unsupported action type: {action_type!r}')
    if action_type in NO_ARG_ACTIONS:
        for field in ('coordinate', 'delay', 'delta_x', 'delta_y'):
            action.pop(field, None)
    elif action_type == 'scroll':
        action.pop('coordinate', None)
    action = _floor_action_numbers(action)
    validate_action(action)
    action_intent = selected.get('action_intent')
    if not isinstance(action_intent, str):
        raise ValueError('action_intent must be a string')
    confidence = selected.get('confidence')
    if type(confidence) is not int or not 0 <= confidence <= 10:
        raise ValueError('confidence must be an integer in [0, 10]')
    return {'unified_action': action, 'action_intent': action_intent, 'confidence': confidence}

def parse_v3_local_candidate_actions_json(text: str) -> list[dict[str, Any]]:
    payload = _parse_json_object(text)
    if set(payload) != {'candidate_actions'}:
        raise ValueError('V3 Local WM response must contain only candidate_actions')
    candidates = payload['candidate_actions']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError('candidate_actions must contain 1..3 items')
    validated = [_validate_v3_local_candidate(item) for item in candidates]
    return validated
_P2_EVIDENCE_BULLET_RE = re.compile('^(?:[-*+]\\s+)+')
_P2_EVIDENCE_BBOX_RE = re.compile('<\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*>\\s*$')

def _ground_p2_evidence_reference(reference: str, predicted_next_md: str) -> str | None:
    """Return the exact forecast line for a narrowly repairable evidence ref."""
    if reference in predicted_next_md:
        return reference
    normalized_reference = _P2_EVIDENCE_BULLET_RE.sub('', reference.strip())
    reference_bbox_match = _P2_EVIDENCE_BBOX_RE.search(normalized_reference)
    reference_bbox = tuple((int(value) for value in reference_bbox_match.groups())) if reference_bbox_match else None
    reference_element = _P2_EVIDENCE_BBOX_RE.sub('', normalized_reference).rstrip()
    matches: list[tuple[int, str]] = []
    for raw_line in predicted_next_md.splitlines():
        exact_line = raw_line.strip()
        if not exact_line:
            continue
        normalized_line = _P2_EVIDENCE_BULLET_RE.sub('', exact_line)
        line_bbox_match = _P2_EVIDENCE_BBOX_RE.search(normalized_line)
        line_element = _P2_EVIDENCE_BBOX_RE.sub('', normalized_line).rstrip()
        if line_element != reference_element:
            continue
        if reference_bbox is None or line_bbox_match is None:
            distance = 0
        else:
            line_bbox = tuple((int(value) for value in line_bbox_match.groups()))
            distance = sum((abs(left - right) for (left, right) in zip(reference_bbox, line_bbox)))
        matches.append((distance, exact_line))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return None
    if reference_bbox is not None and matches[0][0] > 8:
        return None
    return matches[0][1]

def parse_p2_digest_json(text: str, predicted_next_md: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    expected_keys = {'predicted_change', 'intent_alignment', 'task_progress', 'evidence_refs'}
    if set(payload) != expected_keys:
        raise ValueError('P2 digest must contain exactly predicted_change, intent_alignment, task_progress, and evidence_refs')
    predicted_change = payload['predicted_change']
    if not isinstance(predicted_change, str) or not predicted_change.strip():
        raise ValueError('predicted_change must be a non-empty string')
    if len(predicted_change.split()) > 40:
        raise ValueError('predicted_change must contain at most 40 words')
    intent_alignment = payload['intent_alignment']
    if intent_alignment not in {'yes', 'partial', 'no', 'unclear'}:
        raise ValueError('invalid intent_alignment')
    task_progress = payload['task_progress']
    if task_progress not in {'positive', 'neutral', 'negative', 'unclear'}:
        raise ValueError('invalid task_progress')
    evidence_refs = payload['evidence_refs']
    if not isinstance(evidence_refs, list) or len(evidence_refs) > 3:
        raise ValueError('evidence_refs must contain zero to three strings')
    grounded_refs: list[str] = []
    for reference in evidence_refs:
        if not isinstance(reference, str) or not reference:
            continue
        grounded = _ground_p2_evidence_reference(reference, predicted_next_md)
        if grounded is not None and grounded not in grounded_refs:
            grounded_refs.append(grounded)
    if evidence_refs and (not grounded_refs):
        predicted_change = 'unclear'
        intent_alignment = 'unclear'
        task_progress = 'unclear'
    return {'predicted_change': predicted_change.strip(), 'intent_alignment': intent_alignment, 'task_progress': task_progress, 'evidence_refs': grounded_refs}

def parse_p24_digest_json(text: str, predicted_next_md: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    expected_keys = {'predicted_change', 'task_progress', 'evidence_refs'}
    if set(payload) != expected_keys:
        raise ValueError('P2.4 digest must contain exactly predicted_change, task_progress, and evidence_refs')
    predicted_change = payload['predicted_change']
    if not isinstance(predicted_change, str) or not predicted_change.strip():
        raise ValueError('predicted_change must be a non-empty string')
    if len(predicted_change.split()) > 40:
        raise ValueError('predicted_change must contain at most 40 words')
    task_progress = payload['task_progress']
    if task_progress not in {'positive', 'neutral', 'negative', 'unclear'}:
        raise ValueError('invalid task_progress')
    evidence_refs = payload['evidence_refs']
    if not isinstance(evidence_refs, list) or len(evidence_refs) > 3:
        raise ValueError('evidence_refs must contain zero to three strings')
    grounded_refs: list[str] = []
    for reference in evidence_refs:
        if not isinstance(reference, str) or not reference:
            continue
        grounded = _ground_p2_evidence_reference(reference, predicted_next_md)
        if grounded is not None and grounded not in grounded_refs:
            grounded_refs.append(grounded)
    if evidence_refs and (not grounded_refs):
        predicted_change = 'unclear'
        task_progress = 'unclear'
    return {'predicted_change': predicted_change.strip(), 'task_progress': task_progress, 'evidence_refs': grounded_refs}

def truncate_predicted_md_for_digest(predicted_next_md: str, *, max_line_chars: int=P2_DIGEST_MAX_PREDICTED_MD_LINE_CHARS) -> tuple[str, dict[str, Any]]:

    if max_line_chars < 128:
        raise ValueError('Digest predicted-Markdown line limit must be at least 128')
    sanitized_lines: list[str] = []
    truncated_lines: list[dict[str, Any]] = []
    for (line_index, line) in enumerate(predicted_next_md.splitlines()):
        if len(line) <= max_line_chars:
            sanitized_lines.append(line)
            continue
        line_sha256 = hashlib.sha256(line.encode('utf-8')).hexdigest()
        marker = f' ... [TRUNCATED_OVERSIZED_ELEMENT original_chars={len(line)} sha256={line_sha256[:16]}]'
        keep_chars = max_line_chars - len(marker)
        sanitized_lines.append(line[:keep_chars].rstrip() + marker)
        truncated_lines.append({'line_index_zero_based': line_index, 'original_chars': len(line), 'original_sha256': line_sha256, 'retained_chars': keep_chars})
    sanitized = '\n'.join(sanitized_lines)
    return (sanitized, {'applied': bool(truncated_lines), 'max_line_chars': max_line_chars, 'truncated_line_count': len(truncated_lines), 'truncated_lines': truncated_lines, 'original_chars': len(predicted_next_md), 'sanitized_chars': len(sanitized), 'original_sha256': hashlib.sha256(predicted_next_md.encode('utf-8')).hexdigest(), 'sanitized_sha256': hashlib.sha256(sanitized.encode('utf-8')).hexdigest()})

def parse_v3_local_best_action_selection(text: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError('best-action selection requires at least one candidate')
    payload = _parse_json_object(text)
    if set(payload) != {'best_action'} or not isinstance(payload['best_action'], dict):
        raise ValueError('V3 Local WM response must contain only best_action')
    best = payload['best_action']
    if set(best) != {'candidate_index', 'unified_action', 'action_intent'}:
        raise ValueError('best_action must contain exactly candidate_index, unified_action, and action_intent')
    reported_index = best['candidate_index']
    if type(reported_index) is not int:
        raise ValueError('best_action.candidate_index must be an integer')
    if not 0 <= reported_index < len(candidates):
        raise ValueError(f'candidate_index {reported_index} out of range for {len(candidates)} candidates')
    expected = candidates[reported_index]
    selected = _validate_v3_local_candidate({'unified_action': best['unified_action'], 'action_intent': best['action_intent'], 'confidence': expected['confidence']})
    if selected['unified_action'] != expected['unified_action']:
        raise ValueError('best_action.unified_action must exactly match candidate_actions[candidate_index]')
    if selected['action_intent'] != expected['action_intent']:
        raise ValueError('best_action.action_intent must exactly match candidate_actions_intent[candidate_index]')
    return {'selected_index': reported_index, 'reported_index': reported_index, 'index_action_consistent': True, 'matching_indices': [reported_index]}

def build_aligned_best_action_payload(common: dict[str, Any], candidate_records: list[dict[str, Any]], predicted_observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the selector contract shared by Local and native world models."""
    expected = list(range(len(candidate_records)))
    observed = [int(item['candidate_index']) for item in predicted_observations]
    if observed != expected:
        raise ValueError(f'predicted_observations must be complete and zero-based in candidate order: expected={expected}, observed={observed}')
    return common | {'candidate_actions': [item['unified_action'] for item in candidate_records], 'candidate_actions_intent': [item['action_intent'] for item in candidate_records], 'candidate_actions_evidence': [item['evidence'] for item in candidate_records], 'candidate_actions_confidence': [item['confidence'] for item in candidate_records], 'predicted_observations': predicted_observations}

def build_v3_local_best_action_payload(common: dict[str, Any], candidate_records: list[dict[str, Any]], predicted_markdowns: list[str]) -> dict[str, Any]:
    if len(candidate_records) != len(predicted_markdowns):
        raise ValueError('V3 Local WM predictions must align one-to-one with candidate actions')
    return common | {'candidate_actions': [candidate['unified_action'] for candidate in candidate_records], 'candidate_actions_intent': [candidate['action_intent'] for candidate in candidate_records], 'predicted_observation': predicted_markdowns}

def build_p2_digest_payload(*, goal: str, candidate: dict[str, Any], predicted_next_md: str) -> dict[str, Any]:
    return {'goal_text': goal, 'candidate_action': candidate['unified_action'], 'candidate_action_intent': candidate['action_intent'], 'predicted_next_md': predicted_next_md}

def build_p2_best_action_payload(common: dict[str, Any], candidate_records: list[dict[str, Any]], candidate_digests: list[dict[str, Any]], *, include_candidate_confidence: bool=False) -> dict[str, Any]:
    if len(candidate_records) != len(candidate_digests):
        raise ValueError('P2 digests must align one-to-one with candidate actions')
    payload = common | {'candidate_actions': [candidate['unified_action'] for candidate in candidate_records], 'candidate_actions_intent': [candidate['action_intent'] for candidate in candidate_records], 'candidate_digests': candidate_digests}
    if include_candidate_confidence:
        payload['candidate_actions_conf'] = [candidate['confidence'] for candidate in candidate_records]
    return payload

def build_p3_best_action_payload(common: dict[str, Any], candidate_records: list[dict[str, Any]], candidate_digests: list[dict[str, Any]], predicted_markdowns: list[str]) -> dict[str, Any]:
    if len(candidate_records) != len(predicted_markdowns):
        raise ValueError('P3 Local-WM predictions must align one-to-one with candidate actions')
    payload = build_p2_best_action_payload(common, candidate_records, candidate_digests)
    if not any((digest.get('task_progress') == 'positive' for digest in candidate_digests)):
        payload['candidate_predicted_next_mds'] = list(predicted_markdowns)
    return payload

def build_p2_prediction_audit(predictions: Sequence[Any]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for (index, prediction) in enumerate(predictions):
        predicted_next_md = str(prediction.predicted_markdown)
        encoded = predicted_next_md.encode('utf-8')
        audit.append({'candidate_index': index, 'predicted_next_md_sha256': hashlib.sha256(encoded).hexdigest(), 'predicted_next_md_chars': len(predicted_next_md), 'predicted_next_md_bytes': len(encoded), 'output_state_mode': prediction.output_state_mode})
    return audit

def build_v3_other_best_action_payload(common: dict[str, Any], candidate_records: list[dict[str, Any]], predicted_observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the compact V3 selector input without candidate confidence."""
    expected = list(range(len(candidate_records)))
    observed = [int(item['candidate_index']) for item in predicted_observations]
    if observed != expected:
        raise ValueError(f'predicted_observations must be complete and zero-based in candidate order: expected={expected}, observed={observed}')
    return common | {'candidate_actions': [candidate['unified_action'] for candidate in candidate_records], 'candidate_actions_intent': [candidate['action_intent'] for candidate in candidate_records], 'predicted_observations': [str(item['predicted_observation']) for item in predicted_observations]}

def wm_trigger_reason(candidate_records: list[dict[str, Any]]) -> str:
    """Use the shared V3 confidence policy for Local and comparison WMs."""
    return 'multiple_candidates' if len(candidate_records) > 1 else 'none'

def parse_native_tool_call(text: str) -> tuple[dict[str, Any], str]:
    """Parse the official FARA/GUI-Owl XML computer_use response."""
    matches = re.findall('<tool_call>\\s*(.*?)\\s*</tool_call>', text, flags=re.DOTALL | re.IGNORECASE)
    candidate = matches[-1] if matches else ''
    if not candidate:
        marker = text.rfind('{"name"')
        if marker < 0:
            marker = text.rfind("{'name'")
        if marker < 0:
            raise ValueError('native computer-use response contains no tool_call')
        decoder = json.JSONDecoder()
        (candidate, _) = decoder.raw_decode(text[marker:])
        payload = candidate
    else:
        payload = json.loads(candidate)
    if not isinstance(payload, dict) or payload.get('name') != 'computer_use':
        raise ValueError('native tool call must target computer_use')
    arguments = payload.get('arguments')
    if not isinstance(arguments, dict) or not isinstance(arguments.get('action'), str):
        raise ValueError('native computer_use arguments require an action')
    thought_match = re.search('<thinking>\\s*(.*?)\\s*</thinking>', text, flags=re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        thought = text[:text.find('<tool_call>')].strip() if '<tool_call>' in text else ''
    return (arguments, thought)

def native_action_to_selected(arguments: dict[str, Any], *, variant: str, width: int, height: int) -> dict[str, Any]:
    """Adapt native FARA/GUI-Owl actions to the browser runner action contract."""
    kind = str(arguments['action']).lower()

    def point(name: str='coordinate') -> list[int]:
        value = arguments.get(name)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f'native {kind} requires {name}=[x,y]')
        (x, y) = (float(value[0]), float(value[1]))
        if variant == 'fara':
            gx = round(x * 999 / max(width - 1, 1))
            gy = round(y * 999 / max(height - 1, 1))
        else:
            gx = round(x * 999 / 1000)
            gy = round(y * 999 / 1000)
        return [max(0, min(999, gx)), max(0, min(999, gy))]
    if kind == 'key':
        keys = arguments.get('keys')
        if not isinstance(keys, list) or not keys:
            raise ValueError('native key requires non-empty keys')
        action = {'type': 'key_press', 'key': str(keys[0])} if len(keys) == 1 else {'type': 'hotkey', 'keys': [str(k) for k in keys]}
    elif kind == 'type':
        action = {'type': 'type_text', 'text': str(arguments.get('text', ''))}
        if 'coordinate' in arguments:
            action['coordinate'] = point()
        if arguments.get('clear'):
            action['clear'] = True
        if arguments.get('enter') or arguments.get('press_enter'):
            action['enter'] = True
    elif kind in {'left_click', 'click'}:
        action = {'type': 'left_click', 'coordinate': point()}
    elif kind == 'double_click':
        action = {'type': 'double_click', 'coordinate': point()}
    elif kind == 'triple_click':
        action = {'type': 'triple_click', 'coordinate': point()}
    elif kind == 'right_click':
        action = {'type': 'right_click', 'coordinate': point()}
    elif kind == 'middle_click':
        action = {'type': 'middle_click', 'coordinate': point()}
    elif kind == 'mouse_move':
        action = {'type': 'hover', 'coordinate': point()}
    elif kind in {'left_click_drag', 'drag'}:
        action = {'type': 'drag', 'from_coordinate': point(), 'to_coordinate': point('coordinate2')}
    elif kind in {'scroll', 'hscroll'}:
        amount = float(arguments.get('pixels', 0))
        if abs(amount) <= 5:
            amount *= 200
        delta = max(-999, min(999, math.floor(-amount)))
        action = {'type': 'scroll', 'delta_x': delta if kind == 'hscroll' else 0, 'delta_y': 0 if kind == 'hscroll' else delta}
    elif kind == 'visit_url':
        action = {'type': 'navigate', 'url': str(arguments.get('url', ''))}
    elif kind == 'history_back':
        action = {'type': 'go_back'}
    elif kind == 'wait':
        action = {'type': 'wait'}
    elif kind == 'answer':
        action = {'type': 'finish', 'answer': str(arguments.get('text', ''))}
    elif kind == 'terminate':
        status = str(arguments.get('status', 'failure')).lower()
        action = {'type': 'finish', 'answer': 'Native agent reported task completion.'} if status == 'success' else {'type': 'fail', 'reason': 'Native agent reported failure.'}
    elif kind == 'pause_and_memorize_fact':
        action = {'type': 'wait'}
    elif kind == 'web_search':
        action = {'type': 'fail', 'reason': 'Search-engine detours are disabled for Online-Mind2Web.'}
    else:
        raise ValueError(f'unsupported native computer-use action: {kind}')
    validate_action(action)
    return {'unified_action': action, 'action_intent': f'Native computer_use action: {kind}', 'evidence': 'Selected directly by the native GUI agent from the current screenshot.', 'confidence': 0}

def _coordinate(value: Any, field: str='coordinate') -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{field} must be [x, y]')
    if any((isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)):
        raise ValueError(f'{field} entries must be numeric')
    point = tuple((math.floor(float(item)) for item in value))
    if any((item < 0 or item > 999 for item in point)):
        raise ValueError(f'{field} must use the 0..999 grid')
    return point

def validate_action(action: dict[str, Any]) -> None:
    action_type = action['type']
    required: dict[str, set[str]] = {**{name: {'type', 'coordinate'} for name in COORDINATE_ACTIONS}, **{name: {'type', 'coordinate'} for name in NATIVE_COORDINATE_ACTIONS}, **{name: {'type'} for name in NO_ARG_ACTIONS}, 'type_text': {'type', 'text'}, 'scroll': {'type', 'delta_x', 'delta_y'}, 'key_press': {'type', 'key'}, 'hotkey': {'type', 'keys'}, 'navigate': {'type', 'url'}, 'select_option': {'type', 'coordinate', 'option_text'}, 'drag': {'type', 'from_coordinate', 'to_coordinate'}, 'clear': {'type'}, 'finish': {'type', 'answer'}, 'fail': {'type', 'reason'}}
    allowed = set(required[action_type])
    if action_type == 'type_text':
        allowed.update({'coordinate', 'clear', 'enter'})
    if action_type == 'clear':
        allowed.add('coordinate')
    missing = required[action_type] - action.keys()
    extra = action.keys() - allowed
    if missing or extra:
        raise ValueError(f'invalid fields for {action_type}: missing={sorted(missing)}, extra={sorted(extra)}')
    if action_type in COORDINATE_ACTIONS or action_type in NATIVE_COORDINATE_ACTIONS or action_type in {'select_option'}:
        _coordinate(action['coordinate'])
    if action_type == 'type_text' and 'coordinate' in action:
        _coordinate(action['coordinate'])
    if action_type == 'type_text' and any((not isinstance(action.get(field), bool) for field in ('clear', 'enter') if field in action)):
        raise ValueError('type_text clear/enter must be booleans')
    if action_type == 'clear' and 'coordinate' in action:
        _coordinate(action['coordinate'])
    if action_type == 'drag':
        _coordinate(action['from_coordinate'], 'from_coordinate')
        _coordinate(action['to_coordinate'], 'to_coordinate')
    if action_type == 'scroll':
        for field in ('delta_x', 'delta_y'):
            value = action[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{field} must be numeric')
            action[field] = math.floor(float(value))
            if not -999 <= action[field] <= 999:
                raise ValueError(f'{field} must be an integer in [-999, 999]')
    if action_type == 'navigate':
        parsed = urlparse(str(action['url']))
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise ValueError('navigate URL must be an absolute http(s) URL')
    if action_type == 'hotkey' and (not isinstance(action['keys'], list) or not action['keys'] or (not all((isinstance(k, str) for k in action['keys'])))):
        raise ValueError('hotkey keys must be a non-empty string list')

def scale_coordinate(coordinate: list[int] | tuple[int, int], width: int, height: int) -> tuple[int, int]:
    (x, y) = _coordinate(list(coordinate))
    return (round(x * (width - 1) / 999), round(y * (height - 1) / 999))

def task_complete_action(answer: str) -> str:
    answer = ' '.join(str(answer).split()) or 'Task ended without a final answer.'
    return f'TASK_COMPLETE -> ANSWER: {answer}'

def validate_v2_result(result: dict[str, Any], task_dir: Path, schema_path: Path | None=None) -> None:
    if schema_path is not None:
        import jsonschema
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(result)
    if result.get('schema_version') != 'online-mind2web-v2':
        raise ValueError('invalid schema_version')
    steps = result.get('action_history')
    if not isinstance(steps, list) or not steps:
        raise ValueError('action_history must be non-empty')
    for (index, step) in enumerate(steps):
        if step.get('step') != index:
            raise ValueError(f'non-contiguous step at index {index}')
        screenshot = step.get('screenshot')
        if not isinstance(screenshot, str) or not (task_dir / 'trajectory' / screenshot).is_file():
            raise ValueError(f'missing screenshot at step {index}: {screenshot}')
    if not str(steps[-1].get('action', '')).startswith('TASK_COMPLETE'):
        raise ValueError('last action must start with TASK_COMPLETE')
    if len(steps) > 50:
        raise ValueError('trajectory exceeds the 50-step limit')

def model_fingerprint(model_path: Path) -> str:
    entries: list[dict[str, Any]] = []
    if not model_path.exists():
        return 'missing'
    for path in sorted(model_path.iterdir()):
        if path.is_file() and (path.suffix in {'.json', '.model'} or 'safetensors' in path.name):
            entry = {'name': path.name, 'size': path.stat().st_size}
            if path.stat().st_size < 32 * 1024 * 1024:
                entry['sha256'] = sha256_file(path)
            entries.append(entry)
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()

def local_adapter_fingerprint(adapter_path: Path) -> dict[str, Any]:
    """Record the adapter files used by an online world-model run."""
    required = ('adapter_config.json', 'adapter_model.safetensors')
    files = {}
    for name in required:
        path = adapter_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {'size': path.stat().st_size, 'path': str(path.resolve())}
    return {'files': files}

@dataclass
class RunnerConfig:
    endpoint: str
    model: str
    prompt: str
    mode: str
    markdown_endpoint: str | None
    width: int
    height: int
    max_steps: int
    navigation_timeout_ms: int
    action_wait_ms: int
    model_timeout_seconds: int
    model_retries: int
    reasoning_effort: str
    thinking_token_budget: int
    max_completion_tokens: int
    temperature: float = 1.0
    enable_thinking: bool = True
    agent_context_tokens: int = 0
    no_progress_limit: int = 4
    navigation_retries: int = 0
    browser_user_agent: str | None = None
    entry_wait_until: str = 'domcontentloaded'
    entry_settle_ms: int = 0
    compact_vanilla_input: bool = True
    gui_agent_use_current_md: bool = True

class OnlineMind2WebAgent(Protocol):

    async def close(self) -> None:
        ...

    async def render_markdown(self, screenshot: bytes) -> str:
        ...

    async def choose(self, goal: str, url: str, history: list[str], screenshot: bytes, markdown: str, *, page: Page | None=None) -> tuple[dict[str, Any], str, float]:
        ...

class ModelAuditError(RuntimeError):

    def __init__(self, message: str, *, audit_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit_payload = audit_payload

def classify_model_failure(error: Exception | str) -> str:
    audit_payload = getattr(error, 'audit_payload', None)
    if isinstance(audit_payload, dict) and audit_payload.get('error_kind') == 'context_overflow':
        return 'context_overflow'
    message = str(error).lower()
    if 'maximum context length' in message or 'exceeding native window' in message or 'context_overflow' in message:
        return 'context_overflow'
    return 'model_error'

def adjust_max_completion_tokens_from_context_error(error: Exception | str, current_max_tokens: int, *, safety_margin: int=16, retry_ceiling: int | None=None) -> int:

    match = re.search('maximum context length is\\s+(\\d+).*?requested\\s+(\\d+)\\s+output tokens.*?prompt contains at least\\s+(\\d+)\\s+input tokens', str(error), flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return current_max_tokens
    context_tokens = int(match.group(1))
    prompt_tokens = int(match.group(3))
    available = context_tokens - prompt_tokens - safety_margin
    if available <= 0:
        return current_max_tokens
    candidates = [current_max_tokens, available, max(1, current_max_tokens // 2)]
    if retry_ceiling is not None:
        candidates.append(max(1, retry_ceiling))
    return min(candidates)

def manifest_world_model_endpoint(endpoint: str | None, num_shards: int) -> str | None:
    """Return a lane-independent endpoint signature for shared run manifests."""
    if endpoint is None:
        return None
    return endpoint if num_shards == 1 else 'per_shard_local_service'

def parse_local_wm_devices(value: str) -> tuple[str, ...]:
    devices = tuple((item.strip() for item in value.split(',') if item.strip()))
    if not devices:
        raise argparse.ArgumentTypeError('Local WM replica devices cannot be empty')
    if devices == ('auto',):
        return devices
    if 'auto' in devices or any((re.fullmatch('cuda:[0-9]+', device) is None for device in devices)):
        raise argparse.ArgumentTypeError("Local WM replica devices must be 'auto' or unique cuda:N values")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError('Local WM replica devices must be unique')
    return devices

def reconcile_resume_manifest(existing: dict[str, Any], requested: dict[str, Any], *, resumed_at: str | None=None) -> dict[str, Any]:

    existing_parameters = dict(existing.get('parameters') or {})
    existing_parameters.setdefault('navigation_retries', 0)
    existing_parameters.setdefault('entry_wait_until', 'domcontentloaded')
    existing_parameters.setdefault('entry_settle_ms', 0)
    existing_parameters.setdefault('browser_executable_path', None)
    existing_parameters.setdefault('browser_user_agent', None)
    existing_parameters.setdefault('browser_disable_http2', False)
    existing_parameters.setdefault('browser_disable_crashpad', True)
    existing_parameters.setdefault('agent_context_tokens', None)
    existing_parameters.setdefault('task_scheduling', 'static')
    existing_parameters.setdefault('shared_queue_poll_seconds', None)
    existing_parameters.setdefault('task_claim_stale_seconds', None)
    existing_parameters.setdefault('gui_agent_use_current_md', True)
    existing_parameters.setdefault('local_wm_output_state_mode', 'replayed_next')
    existing_parameters.setdefault('local_wm_replica_count', 1)
    existing_parameters.setdefault('local_wm_replica_devices', ['auto'])
    requested_parameters = dict(requested.get('parameters') or {})
    requested_parameters.setdefault('task_scheduling', 'static')
    requested_parameters.setdefault('shared_queue_poll_seconds', None)
    requested_parameters.setdefault('task_claim_stale_seconds', None)
    requested_parameters.setdefault('gui_agent_use_current_md', True)
    requested_parameters.setdefault('local_wm_output_state_mode', 'replayed_next')
    requested_parameters.setdefault('local_wm_replica_count', 1)
    requested_parameters.setdefault('local_wm_replica_devices', ['auto'])
    old_context = existing_parameters.get('agent_context_tokens')
    new_context = requested_parameters.get('agent_context_tokens')
    expands_agent_context = requested.get('agent_strategy') == 'wm_compare' and isinstance(old_context, int) and isinstance(new_context, int) and (old_context > 0) and (new_context > old_context)
    switches_to_shared = existing_parameters.get('task_scheduling') == 'static' and requested_parameters.get('task_scheduling') == 'shared'
    old_shards = existing_parameters.get('num_shards')
    new_shards = requested_parameters.get('num_shards')
    old_execution_layout = existing_parameters.get('execution_layout')
    new_execution_layout = requested_parameters.get('execution_layout')
    expands_shared_shards = existing_parameters.get('task_scheduling') == 'shared' and requested_parameters.get('task_scheduling') == 'shared' and isinstance(old_shards, int) and isinstance(new_shards, int) and (old_shards > 0) and (new_shards > old_shards)
    old_local_wm_replica_count = existing_parameters.get('local_wm_replica_count')
    new_local_wm_replica_count = requested_parameters.get('local_wm_replica_count')
    expands_local_wm_replicas = requested.get('agent_strategy') in {'local_wm', 'p2_digest', 'p3_digest'} and isinstance(old_local_wm_replica_count, int) and isinstance(new_local_wm_replica_count, int) and (old_local_wm_replica_count > 0) and (new_local_wm_replica_count > old_local_wm_replica_count) and (new_local_wm_replica_count == len(requested_parameters.get('local_wm_replica_devices') or []))
    existing_for_compare = dict(existing)
    if expands_agent_context:
        existing_parameters['agent_context_tokens'] = new_context
    if switches_to_shared:
        for field in ('task_scheduling', 'shared_queue_poll_seconds', 'task_claim_stale_seconds'):
            existing_parameters[field] = requested_parameters.get(field)
    if expands_shared_shards:
        existing_parameters['num_shards'] = new_shards
        existing_parameters['execution_layout'] = new_execution_layout
    if expands_local_wm_replicas:
        existing_parameters['local_wm_replica_count'] = new_local_wm_replica_count
        existing_parameters['local_wm_replica_devices'] = requested_parameters.get('local_wm_replica_devices')
        existing_parameters['execution_layout'] = new_execution_layout
    existing_for_compare['parameters'] = existing_parameters
    existing_local_wm = existing_for_compare.get('local_wm')
    requested_local_wm = requested.get('local_wm')
    if isinstance(existing_local_wm, dict) and isinstance(requested_local_wm, dict) and ('output_state_mode' not in existing_local_wm) and (requested_local_wm.get('output_state_mode') == 'replayed_next'):
        normalized_local_wm = dict(existing_local_wm)
        normalized_local_wm['output_state_mode'] = 'replayed_next'
        existing_for_compare['local_wm'] = normalized_local_wm
        existing_local_wm = normalized_local_wm
    if isinstance(existing_local_wm, dict) and isinstance(requested_local_wm, dict) and isinstance(existing_local_wm.get('adapter_fingerprint'), str) and isinstance(requested_local_wm.get('adapter_fingerprint'), dict) and (existing_local_wm.get('adapter') == requested_local_wm.get('adapter')):
        normalized_local_wm = dict(existing_local_wm)
        normalized_local_wm['adapter_fingerprint'] = requested_local_wm.get('adapter_fingerprint')
        existing_for_compare['local_wm'] = normalized_local_wm
        existing_local_wm = normalized_local_wm
    if 'local_wm_output_state_mode' not in existing_parameters and requested_parameters.get('local_wm_output_state_mode') == 'replayed_next':
        existing_parameters['local_wm_output_state_mode'] = 'replayed_next'
    if isinstance(existing_local_wm, dict) and isinstance(requested_local_wm, dict) and ('prompt_key' not in existing_local_wm) and (requested_local_wm.get('prompt_key') == 'world_local') and existing_parameters['gui_agent_use_current_md']:
        normalized_local_wm = dict(existing_local_wm)
        for field in ('prompt_key', 'prompt', 'prompt_sha256', 'input_contract'):
            normalized_local_wm[field] = requested_local_wm.get(field)
        existing_for_compare['local_wm'] = normalized_local_wm
    existing_for_compare.setdefault('task_selection', None)
    if os.environ.get('ALLOW_RESUME_CONTEXT_SHRINK') == '1':
        if 'agent_context_tokens' in requested_parameters:
            existing_parameters['agent_context_tokens'] = requested_parameters.get('agent_context_tokens')
        existing_wm = existing_for_compare.get('world_model')
        requested_wm = requested.get('world_model')
        if isinstance(existing_wm, dict) and isinstance(requested_wm, dict):
            existing_wm = dict(existing_wm)
            for field in ('context_tokens', 'max_tokens'):
                if field in requested_wm:
                    existing_wm[field] = requested_wm.get(field)
            existing_for_compare['world_model'] = existing_wm
    requested_for_compare = dict(requested)
    requested_for_compare['parameters'] = requested_parameters
    requested_for_compare.setdefault('task_selection', None)
    stable = ('dataset_sha256', 'task_order', 'task_selection', 'mode', 'agent_strategy', 'model', 'model_fingerprint', 'prompt_sha256', 'best_prompt_sha256', 'digest_prompt_sha256', 'local_wm', 'p2_digest', 'p3_digest', 'world_model', 'parameters')
    if any((existing_for_compare.get(key) != requested_for_compare.get(key) for key in stable)):
        raise RuntimeError('resume configuration differs from the existing manifest')
    if not expands_agent_context and (not switches_to_shared) and (not expands_shared_shards) and (not expands_local_wm_replicas):
        return existing
    merged = dict(existing)
    merged.update(requested)
    merged['created_at'] = existing.get('created_at', requested.get('created_at'))
    resume_time = resumed_at or datetime.now().astimezone().isoformat(timespec='seconds')
    if expands_agent_context:
        history = list(existing.get('agent_context_expansion_history') or [])
        event = {'from_tokens': old_context, 'to_tokens': new_context, 'resumed_at': resume_time, 'reason': 'audited_gui_selector_context_overflow'}
        if not any((item.get('from_tokens') == old_context and item.get('to_tokens') == new_context for item in history if isinstance(item, dict))):
            history.append(event)
        merged['agent_context_expansion_history'] = history
    if switches_to_shared:
        history = list(existing.get('task_scheduling_history') or [])
        event = {'from': 'static', 'to': 'shared', 'resumed_at': resume_time, 'reason': 'dynamic_cross_lane_handoff'}
        if not any((item.get('from') == 'static' and item.get('to') == 'shared' for item in history if isinstance(item, dict))):
            history.append(event)
        merged['task_scheduling_history'] = history
    if expands_shared_shards:
        history = list(existing.get('shard_expansion_history') or [])
        event = {'from_shards': old_shards, 'to_shards': new_shards, 'resumed_at': resume_time, 'reason': 'shared_queue_capacity_expansion'}
        if old_execution_layout != new_execution_layout:
            event['from_execution_layout'] = old_execution_layout
            event['to_execution_layout'] = new_execution_layout
        if not any((item.get('from_shards') == old_shards and item.get('to_shards') == new_shards for item in history if isinstance(item, dict))):
            history.append(event)
        merged['shard_expansion_history'] = history
    if expands_local_wm_replicas:
        history = list(existing.get('local_wm_replica_expansion_history') or [])
        event = {'from_replicas': old_local_wm_replica_count, 'to_replicas': new_local_wm_replica_count, 'devices': requested_parameters['local_wm_replica_devices'], 'resumed_at': resume_time, 'reason': 'dual_replica_pipeline_capacity_expansion'}
        if not any((item.get('from_replicas') == old_local_wm_replica_count and item.get('to_replicas') == new_local_wm_replica_count for item in history if isinstance(item, dict))):
            history.append(event)
        merged['local_wm_replica_expansion_history'] = history
    return merged

def should_retry_gui_context_overflow(metadata: dict[str, Any], *, current_agent_context_tokens: int) -> bool:

    if metadata.get('status') != 'context_overflow':
        return False
    audit = metadata.get('model_error_audit')
    if not isinstance(audit, dict) or audit.get('component') != 'qwen35b_gui_agent':
        return False
    contexts = [int(value) for value in re.findall('maximum context length is\\s+(\\d+)', str(metadata.get('model_error', '')), flags=re.IGNORECASE)]
    return bool(contexts and current_agent_context_tokens > max(contexts))

def archive_gui_context_overflow(task_dir: Path) -> int:
    """Preserve a superseded GUI-overflow trajectory before technical retry."""
    attempt = 1
    while (task_dir / f'run_metadata.attempt_{attempt:02d}_gui_context_overflow.json').exists():
        attempt += 1
    suffix = f'attempt_{attempt:02d}_gui_context_overflow'
    trajectory = task_dir / 'trajectory'
    if trajectory.is_dir():
        shutil.copytree(trajectory, task_dir / f'trajectory.{suffix}')
    for stem in ('raw_steps', 'run_metadata', 'result'):
        source = task_dir / f'{stem}.json'
        if source.is_file():
            os.replace(source, task_dir / f'{stem}.{suffix}.json')
    return attempt

def build_vanilla_input_payload(*, goal: str, url: str, trace: str, compact: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {'goal_text': goal, 'current_url': url, 'current_subgoal_trace': trace, 'supported_unified_actions': SUPPORTED_ACTIONS}
    if not compact:
        payload['subgoal_history'] = []
        payload['current_subgoal_text'] = ''
    return payload

def build_local_wm_agent_common_payload(*, goal: str, url: str, trace: str, markdown: str, use_current_md: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {'goal_text': goal, 'current_url': url, 'current_subgoal_trace': trace, 'supported_unified_actions': SUPPORTED_ACTIONS}
    if use_current_md:
        if not markdown.strip():
            raise ValueError('GUI Agent current_md input is enabled but current Markdown is empty')
        payload['current_md'] = markdown
    return payload

def v3_local_wm_prompt_variant(path: Path, *, filename: str) -> str | None:
    """Return the V3 Local-WM prompt variant encoded by its directory."""
    valid_names = {filename}
    if filename == 'gui_agent_best.md':
        valid_names.update({'gui_agent_best_delta.md', 'gui_agent_best_full.md'})
    if path.name in valid_names and path.parent.name in {'gui_agent', 'gui_agent_omd'} and (path.parent.parent.name == 'templatesV3'):
        return path.parent.name
    return None

def is_v3_local_wm_prompt(path: Path, *, filename: str) -> bool:
    """Return whether a prompt belongs to the V3 Local WM contract."""
    return v3_local_wm_prompt_variant(path, filename=filename) is not None

def comparison_wm_prompt_version(path: Path, *, filename: str) -> str | None:
    """Return the policy prompt version encoded by a comparison-WM path."""
    if path.name != filename or path.parent.name != 'otherwm':
        return None
    policy_dir = path.parent.parent
    if policy_dir.name == 'templatesV3':
        return 'P2.3'
    if policy_dir.name in {'p2.1', 'p2.3'} and policy_dir.parent.name == 'prompts':
        return policy_dir.name.upper()
    return None

def is_v3_other_wm_prompt(path: Path, *, filename: str) -> bool:
    """Return whether a prompt belongs to a versioned comparison-WM contract."""
    return comparison_wm_prompt_version(path, filename=filename) is not None

def apply_prompt_contract_defaults(args: argparse.Namespace) -> None:
    """Apply prompt-specific runtime invariants before manifest creation."""
    requested_comparison_version = getattr(args, 'comparison_prompt_version', None)
    args.v3_local_wm_contract = False
    args.v3_other_wm_contract = False
    args.comparison_prompt_version = None
    args.p2_digest_contract = False
    args.p2_policy_version = None
    args.p3_digest_contract = False
    args.local_wm_output_state_mode = getattr(args, 'local_wm_output_state_mode', 'replayed_next')
    if args.compact_vanilla_input is None:
        args.compact_vanilla_input = args.agent_strategy == 'vanilla'
    if args.agent_strategy == 'local_wm':
        candidate_variant = v3_local_wm_prompt_variant(args.prompt, filename='gui_agent_candidate.md')
        best_variant = v3_local_wm_prompt_variant(args.best_prompt, filename='gui_agent_best.md')
        if (candidate_variant is None) != (best_variant is None):
            raise ValueError('V3 Local WM candidate and best-action prompts must be used together')
        if candidate_variant is not None and best_variant is not None and (candidate_variant != best_variant):
            raise ValueError('V3 Local WM candidate and best-action prompts must use the same variant')
        args.v3_local_wm_contract = candidate_variant is not None
        if candidate_variant is not None:
            prompt_uses_current_md = candidate_variant == 'gui_agent'
            if args.gui_agent_use_current_md != prompt_uses_current_md:
                expected = '--gui-agent-use-current-md' if prompt_uses_current_md else '--no-gui-agent-use-current-md'
                raise ValueError(f'{candidate_variant} prompts require {expected}')
        expected_local_prompts = {'world_local'} if args.gui_agent_use_current_md else {'localwm_next', 'localwm_delta'}
        if args.local_wm_prompt_key not in expected_local_prompts:
            raise ValueError(f'GUI Agent current_md policy and Local WM prompt key must align: expected one of {sorted(expected_local_prompts)!r}')
        if args.local_wm_prompt_key == 'localwm_delta':
            expected_best = 'gui_agent_best_delta.md' if args.local_wm_output_state_mode == 'raw_delta' else 'gui_agent_best_full.md'
            if args.best_prompt.name != expected_best:
                raise ValueError(f'{args.local_wm_output_state_mode} requires --best-prompt {expected_best}')
        elif args.local_wm_output_state_mode != 'replayed_next':
            raise ValueError('raw_delta output state requires --local-wm-prompt-key localwm_delta')
    if args.agent_strategy in {'p2_digest', 'p3_digest'}:
        policy_name = 'p2' if args.agent_strategy == 'p2_digest' else 'p3'
        policy_dir = args.digest_contract_prompt_dir if args.digest_contract_prompt_dir is not None else PROJECT_ROOT / ('mkd2world/prompts/p2.1' if policy_name == 'p2' else f'mkd2world/prompts/{policy_name}')
        expected_prompts = {'candidate': policy_dir / 'candidate.md', 'digest': policy_dir / 'digest.md', 'best': policy_dir / 'best.md'}
        configured_prompts = {'candidate': args.prompt, 'digest': args.digest_prompt, 'best': args.best_prompt}
        existing_manifest_path = args.output_dir / 'manifest.json'
        if args.resume and existing_manifest_path.is_file():
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding='utf-8'))
            if existing_manifest.get('agent_strategy') != args.agent_strategy:
                raise ValueError(f'{args.agent_strategy} cannot resume an output directory owned by another strategy')
        for (stage, configured_path) in configured_prompts.items():
            expected_path = expected_prompts[stage]
            if configured_path is None or not configured_path.is_file():
                raise ValueError(f'{args.agent_strategy} {stage} prompt is missing: {configured_path}')
            if not expected_path.is_file():
                raise ValueError(f'{args.agent_strategy} {stage} prompt is missing: {expected_path}')
            if configured_path.resolve() != expected_path.resolve():
                raise ValueError(f'{args.agent_strategy} {stage} prompt must be {expected_path}')
        if args.agent_strategy == 'p2_digest':
            args.p2_digest_contract = True
            args.p2_policy_version = 'p2.4' if policy_dir.name == 'p2.4' else 'p2.1'
        else:
            args.p3_digest_contract = True
        if args.gui_agent_use_current_md:
            raise ValueError(f'{args.agent_strategy} requires --no-gui-agent-use-current-md')
        if args.local_wm_prompt_key != 'localwm_next':
            raise ValueError(f'{args.agent_strategy} requires --local-wm-prompt-key localwm_next')
        if args.local_wm_output_state_mode != 'replayed_next':
            raise ValueError(f'{args.agent_strategy} requires full replayed_next Local-WM forecasts')
    if args.agent_strategy == 'wm_compare':
        candidate_version = comparison_wm_prompt_version(args.prompt, filename='gui_agent_candidate.md')
        best_version = comparison_wm_prompt_version(args.best_prompt, filename='gui_agent_best.md')
        if requested_comparison_version is not None:
            canonical = PROJECT_ROOT / 'mkd2world/prompts' / requested_comparison_version.lower() / 'otherwm'
            configured = {'candidate': args.prompt, 'best': args.best_prompt, 'world_model': args.world_model_prompt}
            canonical_names = {'candidate': 'gui_agent_candidate.md', 'best': 'gui_agent_best.md', 'world_model': 'world_agentworld_web.md' if args.world_model_kind == 'agentworld' else 'world_dreamer.md' if args.world_model_kind == 'dreamer' else 'world_webworld.md' if args.world_model_kind in {'webworld', 'webworld32b'} else 'world_wma_v2.md' if args.world_model_kind == 'wma' and getattr(args, 'wma_input_contract', 'official_contextual') == 'axtree_action_only' else 'world_wma.md'}
            for (name, configured_path) in configured.items():
                expected_path = canonical / canonical_names[name]
                if configured_path is None or not configured_path.is_file():
                    raise ValueError(f'Comparison-WM {name} prompt is missing: {configured_path}')
                if not expected_path.is_file():
                    raise ValueError(f'Comparison-WM {name} prompt is missing: {expected_path}')
                if configured_path.resolve() != expected_path.resolve():
                    raise ValueError(f'Comparison-WM {name} prompt must be {expected_path}')
            candidate_version = requested_comparison_version
            best_version = requested_comparison_version
        if candidate_version != best_version:
            raise ValueError('Comparison-WM candidate and best-action prompts must use the same policy prompt version')
        args.comparison_prompt_version = candidate_version
        args.v3_other_wm_contract = candidate_version is not None

def enforce_world_model_output_limit(args: argparse.Namespace) -> None:
    """Keep every evaluation World Model on the project-wide 6K ceiling."""
    if args.agent_strategy in {'local_wm', 'p2_digest', 'p3_digest'} and args.local_max_new_tokens != WORLD_MODEL_MAX_OUTPUT_TOKENS:
        raise ValueError(f'All Local-WM evaluation strategies require --local-max-new-tokens {WORLD_MODEL_MAX_OUTPUT_TOKENS}')
    if args.agent_strategy == 'wm_compare' and args.world_model_kind != 'none' and (args.world_model_max_tokens != WORLD_MODEL_MAX_OUTPUT_TOKENS):
        raise ValueError(f'All comparison-WM evaluations require --world-model-max-tokens {WORLD_MODEL_MAX_OUTPUT_TOKENS}')

def build_prompt_trace_entry(action: dict[str, Any], action_intent: str, execution_status: str) -> str:
    """Serialize one real action for future prompts without pixel conversion."""
    validate_action(action)
    if execution_status not in {'SUCCESS', 'FAILED'}:
        raise ValueError('execution_status must be SUCCESS or FAILED')
    return json.dumps({'action': action_to_prompt_dict(action), 'action_intent': ' '.join(action_intent.split()), 'execution_status': execution_status}, ensure_ascii=False, separators=(',', ':'))

def wma_previous_action_from_trace(trace: str | None) -> str:

    if not trace:
        return 'None'
    try:
        payload = json.loads(trace)
        action = payload['action']
        from mkd2world.online_mind2web_wm import TargetResolution, serialize_wma_action
        return serialize_wma_action(action, TargetResolution('not_applicable', None, None))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 'None'

class VanillaAgent:

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(base_url=config.endpoint.rstrip('/') + '/v1', api_key='none', timeout=config.model_timeout_seconds)

    async def close(self) -> None:
        await self.client.close()

    async def render_markdown(self, screenshot: bytes) -> str:
        if self.config.mode != 'screenshot_md':
            return ''
        if not self.config.markdown_endpoint:
            raise RuntimeError('screenshot_md mode requires --markdown-endpoint')
        import httpx
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
                    response = await client.post(self.config.markdown_endpoint, content=screenshot, headers={'content-type': 'image/png'})
                    if response.is_error:
                        body = response.text[:2000]
                        raise RuntimeError(f'Markdown renderer HTTP {response.status_code}: {body}')
                    payload = response.json()
                markdown = payload.get('markdown') if isinstance(payload, dict) else None
                if not isinstance(markdown, str) or not markdown.strip():
                    raise RuntimeError('Markdown renderer returned an empty/invalid response')
                return markdown
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f'Markdown renderer failed after 3 attempts: {last_error}') from last_error

    async def choose(self, goal: str, url: str, history: list[str], screenshot: bytes, markdown: str, *, page: Page | None=None) -> tuple[dict[str, Any], str, float]:
        trace = '\n'.join((f'{i + 1}. {item}' for (i, item) in enumerate(history[-20:]))) or '(none)'
        payload = build_vanilla_input_payload(goal=goal, url=url, trace=trace, compact=self.config.compact_vanilla_input)
        if self.config.mode == 'screenshot_md':
            payload['current_md'] = markdown
        encoded = base64.b64encode(screenshot).decode('ascii')
        messages = [{'role': 'system', 'content': self.config.prompt}, {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}, {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}', 'detail': 'high'}}]}]
        last_error: Exception | None = None
        retry_messages = list(messages)
        request_max_tokens = self.config.max_completion_tokens
        started = time.monotonic()
        for attempt in range(self.config.model_retries + 1):
            try:
                thinking_token_budget = min(self.config.thinking_token_budget, max(1, request_max_tokens - 1))
                request_kwargs: dict[str, Any] = {'model': self.config.model, 'messages': retry_messages, 'temperature': self.config.temperature, 'max_completion_tokens': request_max_tokens, 'response_format': ACTION_RESPONSE_FORMAT, 'extra_body': {'thinking_token_budget': thinking_token_budget, 'chat_template_kwargs': {'enable_thinking': self.config.enable_thinking}}}
                if self.config.enable_thinking:
                    request_kwargs['reasoning_effort'] = self.config.reasoning_effort
                response = await self.client.chat.completions.create(**request_kwargs)
                raw = response.choices[0].message.content or ''
                try:
                    return (parse_model_json(raw), raw, time.monotonic() - started)
                except Exception as exc:
                    last_error = ValueError(f'{exc}; raw_response={raw[:2000]!r}')
                    if attempt < self.config.model_retries:
                        retry_messages.extend([{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': f'The previous action failed validation: {exc}. Return one corrected action matching the required JSON fields exactly.'}])
                        await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as exc:
                last_error = exc
                adjusted = adjust_max_completion_tokens_from_context_error(exc, request_max_tokens)
                if adjusted < request_max_tokens:
                    request_max_tokens = adjusted
                if attempt < self.config.model_retries:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f'model failed after {self.config.model_retries + 1} attempts: {last_error}')

class NativeComputerUseAgent(VanillaAgent):
    """Official XML computer_use loop for FARA and GUI-Owl checkpoints."""

    def __init__(self, config: RunnerConfig, *, variant: str) -> None:
        super().__init__(config)
        if variant not in {'fara', 'gui_owl'}:
            raise ValueError(f'unsupported native variant: {variant}')
        self.variant = variant

    async def choose(self, goal: str, url: str, history: list[str], screenshot: bytes, markdown: str, *, page: Page | None=None) -> tuple[dict[str, Any], str, float]:
        trace = '\n'.join((f'{i + 1}. {item}' for (i, item) in enumerate(history[-20:]))) or '(none)'
        user_text = f'Please generate the next move according to the UI screenshot, instruction, and previous actions.\nInstruction: {goal}\nCurrent URL: {url}\nPrevious actions:\n{trace}'
        if self.config.mode == 'screenshot_md':
            user_text += f'\nCurrent interface Markdown (0..999 coordinates):\n{markdown}'
        encoded = base64.b64encode(screenshot).decode('ascii')
        messages: list[dict[str, Any]] = [{'role': 'system', 'content': self.config.prompt}, {'role': 'user', 'content': [{'type': 'text', 'text': user_text}, {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}', 'detail': 'high'}}]}]
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.config.model_retries + 1):
            try:
                response = await self.client.chat.completions.create(model=self.config.model, messages=messages, temperature=self.config.temperature, max_completion_tokens=self.config.max_completion_tokens)
                choice = response.choices[0]
                raw = choice.message.content or ''
                try:
                    (arguments, thought) = parse_native_tool_call(raw)
                    selected = native_action_to_selected(arguments, variant=self.variant, width=self.config.width, height=self.config.height)
                    usage = response.usage
                    diagnostics = {'strategy': self.variant, 'native_raw': raw, 'native_arguments': arguments, 'native_thought': thought, 'finish_reason': choice.finish_reason, 'prompt_tokens': getattr(usage, 'prompt_tokens', None), 'completion_tokens': getattr(usage, 'completion_tokens', None), 'total_tokens': getattr(usage, 'total_tokens', None), 'output_limit_hit': choice.finish_reason == 'length'}
                    if thought:
                        selected['evidence'] = thought
                    return ({'selected_action': selected}, json.dumps(diagnostics, ensure_ascii=False), time.monotonic() - started)
                except Exception as exc:
                    last_error = ValueError(f'{exc}; raw_response={raw[:2000]!r}')
                    if attempt < self.config.model_retries:
                        messages.extend([{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': f'The previous response could not be executed. Return exactly one valid computer_use tool call. Validation error: {exc}'}])
                        await asyncio.sleep(2 ** attempt)
            except Exception as exc:
                last_error = exc
                if attempt < self.config.model_retries:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f'native GUI agent failed after {self.config.model_retries + 1} attempts: {last_error}')

class LocalWMAgent(VanillaAgent):
    """Action-only GUI Agent using Local WM predictions for multi-candidate decisions."""

    def __init__(self, config: RunnerConfig, *, candidate_prompt: str, best_prompt: str, local_base_model: Path, local_adapter: Path | None, local_base_only: bool, local_batch_size: int, local_max_length: int, local_max_new_tokens: int, local_wm_prompt_key: str, local_wm_output_state_mode: str, local_wm_devices: Sequence[str], v3_contract: bool=False) -> None:
        super().__init__(config)
        from mkd2world.local_wm_runtime import TransformersLocalWorldModel
        self.candidate_prompt = candidate_prompt
        self.best_prompt = best_prompt
        self.v3_contract = v3_contract
        if not local_wm_devices:
            raise ValueError('Local WM requires at least one replica device')
        self.local_wm_devices = tuple(local_wm_devices)
        self.local_wms = [TransformersLocalWorldModel(base_model=local_base_model, adapter_path=local_adapter, base_only=local_base_only, device_map='auto' if device == 'auto' else {'': device}, torch_dtype='bfloat16', max_length=local_max_length, max_new_tokens=local_max_new_tokens, batch_size=local_batch_size, prompt_key=local_wm_prompt_key, delta_state_mode=local_wm_output_state_mode) for device in self.local_wm_devices]
        self.local_wm = self.local_wms[0]
        self._available_local_wms: asyncio.Queue[int] = asyncio.Queue()
        for replica_index in range(len(self.local_wms)):
            self._available_local_wms.put_nowait(replica_index)

    async def preload(self) -> None:
        """Load every replica before browsers claim formal tasks."""
        for local_wm in self.local_wms:
            await asyncio.to_thread(local_wm.load)
        print('LOCAL_WM_REPLICA_POOL_READY ' + json.dumps({'replica_count': len(self.local_wms), 'devices': list(self.local_wm_devices)}, ensure_ascii=False, sort_keys=True), flush=True)

    async def _predict_with_local_wm_replica(self, current_state: Any, candidates: Sequence[Any]) -> tuple[list[Any], dict[str, int], dict[str, int], int, str]:
        replica_index = await self._available_local_wms.get()
        local_wm = self.local_wms[replica_index]
        try:
            before_telemetry = local_wm.telemetry_snapshot()
            predictions = await asyncio.to_thread(local_wm.predict_next_states, current_state, candidates)
            after_telemetry = local_wm.telemetry_snapshot()
            return (predictions, before_telemetry, after_telemetry, replica_index, self.local_wm_devices[replica_index])
        finally:
            self._available_local_wms.put_nowait(replica_index)

    async def close(self) -> None:
        try:
            for local_wm in self.local_wms:
                await asyncio.to_thread(local_wm.close)
        finally:
            await super().close()

    async def _chat_json(self, *, system_prompt: str, payload: dict[str, Any], screenshot: bytes, response_format: dict[str, Any], parser: Any, correction: str) -> tuple[Any, str, dict[str, Any]]:
        encoded = base64.b64encode(screenshot).decode('ascii')
        messages: list[dict[str, Any]] = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}, {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}', 'detail': 'high'}}]}]
        last_error: Exception | None = None
        attempts: list[dict[str, Any]] = []
        request_max_tokens = self.config.max_completion_tokens
        for attempt in range(self.config.model_retries + 1):
            try:
                thinking_token_budget = min(self.config.thinking_token_budget, max(1, request_max_tokens - 1))
                response = await self.client.chat.completions.create(model=self.config.model, messages=messages, temperature=self.config.temperature, max_completion_tokens=request_max_tokens, reasoning_effort=self.config.reasoning_effort, response_format=response_format, extra_body={'thinking_token_budget': thinking_token_budget, 'chat_template_kwargs': {'enable_thinking': self.config.enable_thinking}})
                choice = response.choices[0]
                raw = choice.message.content or ''
                try:
                    parsed = parser(raw)
                    usage = response.usage
                    stats = {'finish_reason': choice.finish_reason, 'prompt_tokens': getattr(usage, 'prompt_tokens', None), 'completion_tokens': getattr(usage, 'completion_tokens', None), 'total_tokens': getattr(usage, 'total_tokens', None), 'output_limit_hit': choice.finish_reason == 'length'}
                    return (parsed, raw, stats)
                except Exception as exc:
                    last_error = ValueError(f'{exc}; raw_response={raw[:2000]!r}')
                    if attempt < self.config.model_retries:
                        messages.extend([{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': f'{correction} Validation error: {exc}'}])
                        await asyncio.sleep(2 ** attempt)
            except Exception as exc:
                last_error = exc
                adjusted = adjust_max_completion_tokens_from_context_error(exc, request_max_tokens)
                if adjusted < request_max_tokens:
                    request_max_tokens = adjusted
                if attempt < self.config.model_retries:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f'GUI Agent failed after {self.config.model_retries + 1} attempts: {last_error}')

    async def choose(self, goal: str, url: str, history: list[str], screenshot: bytes, markdown: str, *, page: Page | None=None) -> tuple[dict[str, Any], str, float]:
        from mkd2world.actions import UnifiedAction
        from mkd2world.contracts import CandidateAction, GuiState
        started = time.monotonic()
        trace = '\n'.join((f'{i + 1}. {item}' for (i, item) in enumerate(history[-20:]))) or '(none)'
        common = build_local_wm_agent_common_payload(goal=goal, url=url, trace=trace, markdown=markdown, use_current_md=self.config.gui_agent_use_current_md)
        candidate_response_format = V3_LOCAL_CANDIDATE_ACTIONS_RESPONSE_FORMAT if self.v3_contract else CANDIDATE_ACTIONS_RESPONSE_FORMAT
        candidate_parser = parse_v3_local_candidate_actions_json if self.v3_contract else parse_candidate_actions_json
        candidate_correction = 'Return exactly 1 candidate only when its confidence is 8..10; when confidence is 0..7, return 2 or 3 complementary candidates. Each item must contain exactly unified_action, action_intent, and confidence.' if self.v3_contract else CANDIDATE_ACTIONS_CORRECTION
        (candidates_payload, candidate_raw, candidate_stats) = await self._chat_json(system_prompt=self.candidate_prompt, payload=common, screenshot=screenshot, response_format=candidate_response_format, parser=candidate_parser, correction=candidate_correction)
        candidate_records = list(candidates_payload)
        candidates = [CandidateAction(unified_action=UnifiedAction.from_dict(item['unified_action']), action_intent=item['action_intent'], evidence=item.get('evidence', ''), confidence=item.get('confidence', 0)) for item in candidate_records]
        candidate_confidences = [candidate.confidence for candidate in candidates]
        trigger_reason = wm_trigger_reason(candidate_records) if self.v3_contract else 'multiple_candidates' if len(candidate_records) > 1 else 'none'
        diagnostics: dict[str, Any] = {'strategy': 'local_wm_only', 'candidate_count': len(candidates), 'candidate_confidences': candidate_confidences, 'wm_trigger_reason': trigger_reason, 'candidate_raw': candidate_raw, 'candidate_model_stats': candidate_stats, 'local_wm_called': False, 'best_action_called': False, 'gui_agent_use_current_md': self.config.gui_agent_use_current_md, 'local_wm_prompt_key': self.local_wm.prompt_key, 'local_wm_output_state_mode': self.local_wm.delta_state_mode}
        if trigger_reason == 'none':
            selected_index = 0
        else:
            (predictions, before_telemetry, after_telemetry, replica_index, replica_device) = await self._predict_with_local_wm_replica(GuiState(screenshot=screenshot, markdown=markdown, url=url), candidates)
            prediction_payload = [{'candidate_index': index, 'observation_format': 'markdown', 'predicted_observation': item.predicted_markdown, 'wm_called': True} for (index, item) in enumerate(predictions)]
            diagnostics.update({'local_wm_called': True, 'local_wm_replica_index': replica_index, 'local_wm_replica_device': replica_device, 'local_wm_predictions': [{'candidate_index': index, 'action_analysis': item.action_analysis, 'predicted_markdown': item.predicted_markdown, 'predicted_delta': item.predicted_delta, 'output_state_mode': item.output_state_mode, 'raw_output': item.raw_output} for (index, item) in enumerate(predictions)], 'local_wm_telemetry_before': before_telemetry, 'local_wm_telemetry_after': after_telemetry})
            if self.v3_contract:
                best_payload = build_v3_local_best_action_payload(common, candidate_records, [item.predicted_markdown for item in predictions])
                best_response_format = V3_LOCAL_BEST_ACTION_RESPONSE_FORMAT
                best_parser = lambda raw: parse_v3_local_best_action_selection(raw, candidate_records)
                best_correction = 'Return one corrected best_action containing exactly candidate_index, unified_action, and action_intent. The unified_action must exactly copy the indexed candidate.'
            else:
                best_payload = build_aligned_best_action_payload(common, candidate_records, prediction_payload)
                best_response_format = BEST_ACTION_RESPONSE_FORMAT
                best_parser = lambda raw: parse_strict_best_action_selection(raw, candidate_records)
                best_correction = 'Return one corrected best_action. candidate_index is zero-based and unified_action must exactly copy candidate_actions[candidate_index].'
            (best_selection, best_raw, best_stats) = await self._chat_json(system_prompt=self.best_prompt, payload=best_payload, screenshot=screenshot, response_format=best_response_format, parser=best_parser, correction=best_correction)
            selected_index = int(best_selection['selected_index'])
            diagnostics.update({'best_action_called': True, 'best_action_raw': best_raw, 'best_action_model_stats': best_stats, 'best_action_reported_index_zero_based': best_selection['reported_index'], 'best_action_index_action_consistent': best_selection['index_action_consistent'], 'best_action_matching_indices_zero_based': best_selection['matching_indices']})
        selected = candidate_records[selected_index]
        diagnostics['selected_candidate_index'] = selected_index + 1
        return ({'selected_action': selected}, json.dumps(diagnostics, ensure_ascii=False), time.monotonic() - started)

class P2DigestAgent(LocalWMAgent):
    """P2 policy: full Local-WM forecasts, per-candidate Digest, then Best."""
    strategy_name = 'p2_digest'
    policy_label = 'P2'
    error_kind_prefix = 'p2'

    def __init__(self, config: RunnerConfig, *, candidate_prompt: str, digest_prompt: str, best_prompt: str, local_base_model: Path, local_adapter: Path | None, local_base_only: bool, local_batch_size: int, local_max_length: int, local_max_new_tokens: int, local_wm_prompt_key: str, local_wm_output_state_mode: str, local_wm_devices: Sequence[str], policy_version: str='p2.1') -> None:
        if policy_version not in {'p2.1', 'p2.4'}:
            raise ValueError(f'unsupported P2 policy version: {policy_version}')
        super().__init__(config, candidate_prompt=candidate_prompt, best_prompt=best_prompt, local_base_model=local_base_model, local_adapter=local_adapter, local_base_only=local_base_only, local_batch_size=local_batch_size, local_max_length=local_max_length, local_max_new_tokens=local_max_new_tokens, local_wm_prompt_key=local_wm_prompt_key, local_wm_output_state_mode=local_wm_output_state_mode, local_wm_devices=local_wm_devices, v3_contract=True)
        self.digest_prompt = digest_prompt
        self.policy_version = policy_version

    async def _digest_one_candidate(self, *, candidate_index: int, goal: str, candidate: dict[str, Any], predicted_next_md: str, screenshot: bytes) -> dict[str, Any]:
        (digest_predicted_next_md, sanitization) = truncate_predicted_md_for_digest(predicted_next_md)
        payload = build_p2_digest_payload(goal=goal, candidate=candidate, predicted_next_md=digest_predicted_next_md)
        policy_version = getattr(self, 'policy_version', 'p2.1')
        is_p24 = policy_version == 'p2.4'
        started = time.monotonic()
        (digest, raw, stats) = await self._chat_json(system_prompt=self.digest_prompt, payload=payload, screenshot=screenshot, response_format=P24_DIGEST_RESPONSE_FORMAT if is_p24 else P2_DIGEST_RESPONSE_FORMAT, parser=lambda text: parse_p24_digest_json(text, digest_predicted_next_md) if is_p24 else parse_p2_digest_json(text, digest_predicted_next_md), correction='Return one corrected Digest JSON object with exactly ' + ('predicted_change, task_progress, and evidence_refs. ' if is_p24 else 'predicted_change, intent_alignment, task_progress, and evidence_refs. ') + 'Copy each evidence reference exactly from predicted_next_md.')
        return {'candidate_index': candidate_index, 'digest': digest, 'raw': raw, 'model_stats': stats, 'latency_seconds': round(time.monotonic() - started, 3), 'input_keys': sorted(payload), 'input_sanitization': sanitization}

    async def _digest_candidates(self, *, goal: str, candidate_records: list[dict[str, Any]], predicted_markdowns: list[str], screenshot: bytes) -> list[dict[str, Any]]:
        if len(candidate_records) != len(predicted_markdowns):
            raise ValueError(f'{self.policy_label} Local-WM predictions must align one-to-one with candidates')
        outcomes = await asyncio.gather(*(self._digest_one_candidate(candidate_index=index, goal=goal, candidate=candidate, predicted_next_md=predicted_markdowns[index], screenshot=screenshot) for (index, candidate) in enumerate(candidate_records)), return_exceptions=True)
        recovered_outcomes: list[dict[str, Any]] = []
        for (index, outcome) in enumerate(outcomes):
            if isinstance(outcome, dict):
                recovered_outcomes.append(outcome)
                continue
            (_, sanitization) = truncate_predicted_md_for_digest(predicted_markdowns[index])
            error_text = f'{type(outcome).__name__}: {outcome}'
            fallback_digest = {'predicted_change': 'Technical anomaly: this candidate forecast could not be digested reliably.', 'task_progress': 'unclear', 'evidence_refs': []}
            if getattr(self, 'policy_version', 'p2.1') != 'p2.4':
                fallback_digest['intent_alignment'] = 'unclear'
            recovered_outcomes.append({'candidate_index': index, 'digest': fallback_digest, 'raw': '', 'model_stats': None, 'latency_seconds': None, 'input_keys': ['candidate_action', 'candidate_action_intent', 'goal_text', 'predicted_next_md'], 'input_sanitization': sanitization, 'technical_anomaly': {'component': 'qwen35b_gui_agent', 'stage': 'digest', 'error_kind': f'{self.error_kind_prefix}_digest_candidate_failure', 'error': error_text[:2000], 'error_sha256': hashlib.sha256(error_text.encode('utf-8')).hexdigest(), 'fallback': 'unclear_digest'}})
        ordered = sorted(recovered_outcomes, key=lambda item: int(item['candidate_index']))
        observed = [int(item['candidate_index']) for item in ordered]
        expected = list(range(len(candidate_records)))
        if observed != expected:
            raise ValueError(f'{self.policy_label} Digest results must restore zero-based candidate order: expected={expected}, observed={observed}')
        return ordered

    def _build_best_action_payload(self, common: dict[str, Any], candidate_records: list[dict[str, Any]], candidate_digests: list[dict[str, Any]], predicted_markdowns: list[str]) -> dict[str, Any]:
        del predicted_markdowns
        return build_p2_best_action_payload(common, candidate_records, candidate_digests, include_candidate_confidence=getattr(self, 'policy_version', 'p2.1') == 'p2.4')

    async def choose(self, goal: str, url: str, history: list[str], screenshot: bytes, markdown: str, *, page: Page | None=None) -> tuple[dict[str, Any], str, float]:
        from mkd2world.actions import UnifiedAction
        from mkd2world.contracts import CandidateAction, GuiState
        started = time.monotonic()
        trace = '\n'.join((f'{index + 1}. {item}' for (index, item) in enumerate(history[-20:]))) or '(none)'
        common = build_local_wm_agent_common_payload(goal=goal, url=url, trace=trace, markdown=markdown, use_current_md=False)
        try:
            (candidates_payload, candidate_raw, candidate_stats) = await self._chat_json(system_prompt=self.candidate_prompt, payload=common, screenshot=screenshot, response_format=V3_LOCAL_CANDIDATE_ACTIONS_RESPONSE_FORMAT, parser=parse_v3_local_candidate_actions_json, correction='Return exactly 1 candidate only when its confidence is 8..10; when confidence is 0..7, return 2 or 3 complementary candidates. Each item must contain exactly unified_action, action_intent, and confidence.')
        except Exception as exc:
            raise ModelAuditError(f'{self.policy_label} Candidate failed: {type(exc).__name__}: {exc}', audit_payload={'component': 'qwen35b_gui_agent', 'stage': 'candidate', 'error_kind': f'{self.error_kind_prefix}_candidate_failure', 'fallback_used': False}) from exc
        candidate_records = list(candidates_payload)
        candidates = [CandidateAction(unified_action=UnifiedAction.from_dict(item['unified_action']), action_intent=item['action_intent'], evidence='', confidence=item['confidence']) for item in candidate_records]
        trigger_reason = wm_trigger_reason(candidate_records)
        diagnostics: dict[str, Any] = {'strategy': self.strategy_name, 'policy_version': getattr(self, 'policy_version', 'p2.1'), 'candidate_count': len(candidate_records), 'candidate_confidences': [item['confidence'] for item in candidate_records], 'wm_trigger_reason': trigger_reason, 'candidate_raw': candidate_raw, 'candidate_model_stats': candidate_stats, 'candidate_input_keys': sorted(common), 'local_wm_called': False, 'digest_called': False, 'best_action_called': False, 'raw_next_md_exposed_to_best': False, 'candidate_confidence_exposed_to_best': False, 'gui_agent_use_current_md': False, 'local_wm_prompt_key': self.local_wm.prompt_key, 'local_wm_output_state_mode': self.local_wm.delta_state_mode}
        if trigger_reason == 'none':
            selected_index = 0
            diagnostics['single_candidate_direct'] = True
        else:
            try:
                (predictions, before_telemetry, after_telemetry, replica_index, replica_device) = await self._predict_with_local_wm_replica(GuiState(screenshot=screenshot, markdown=markdown, url=url), candidates)
            except Exception as exc:
                raise ModelAuditError(f'{self.policy_label} Local WM failed: {type(exc).__name__}: {exc}', audit_payload={'component': 'local_wm', 'stage': 'local_wm', 'error_kind': f'{self.error_kind_prefix}_local_wm_failure', 'fallback_used': False}) from exc
            if len(predictions) != len(candidate_records):
                raise ModelAuditError(f'{self.policy_label} Local WM returned a prediction count that does not match the candidate count', audit_payload={'component': 'local_wm', 'stage': 'local_wm', 'error_kind': f'{self.error_kind_prefix}_local_wm_alignment_failure', 'candidate_count': len(candidate_records), 'prediction_count': len(predictions), 'fallback_used': False})
            diagnostics.update({'local_wm_called': True, 'local_wm_replica_index': replica_index, 'local_wm_replica_device': replica_device, 'local_wm_predictions': [{'candidate_index': index, 'action_analysis': prediction.action_analysis, 'predicted_markdown': prediction.predicted_markdown, 'predicted_delta': prediction.predicted_delta, 'output_state_mode': prediction.output_state_mode, 'raw_output': prediction.raw_output} for (index, prediction) in enumerate(predictions)], 'local_wm_telemetry_before': before_telemetry, 'local_wm_telemetry_after': after_telemetry})
            try:
                digest_results = await self._digest_candidates(goal=goal, candidate_records=candidate_records, predicted_markdowns=[prediction.predicted_markdown for prediction in predictions], screenshot=screenshot)
            except ModelAuditError as exc:
                exc.audit_payload.update({'local_wm_replica_index': replica_index, 'local_wm_replica_device': replica_device, 'candidate_count': len(candidate_records), 'candidate_records': candidate_records, 'candidate_raw': candidate_raw, 'local_wm_prediction_audit': build_p2_prediction_audit(predictions), 'local_wm_telemetry_before': before_telemetry, 'local_wm_telemetry_after': after_telemetry})
                raise
            candidate_digests = [item['digest'] for item in digest_results]
            digest_anomaly_candidate_indices = [int(item['candidate_index']) for item in digest_results if item.get('technical_anomaly') is not None]
            diagnostics.update({'digest_called': True, 'digest_parallel_call_count': len(digest_results), 'digest_results': digest_results, 'digest_anomaly_candidate_indices_zero_based': digest_anomaly_candidate_indices, 'digest_candidate_fallback_used': bool(digest_anomaly_candidate_indices)})
            predicted_markdowns = [truncate_predicted_md_for_digest(prediction.predicted_markdown)[0] for prediction in predictions]
            best_payload = self._build_best_action_payload(common, candidate_records, candidate_digests, predicted_markdowns)
            diagnostics['candidate_confidence_exposed_to_best'] = 'candidate_actions_conf' in best_payload
            positive_candidate_indices = [index for (index, digest) in enumerate(candidate_digests) if digest.get('task_progress') == 'positive']
            raw_next_md_exposed = 'candidate_predicted_next_mds' in best_payload
            if self.strategy_name == 'p3_digest':
                diagnostics.update({'digest_positive_candidate_indices_zero_based': positive_candidate_indices, 'full_prediction_fallback_triggered': raw_next_md_exposed, 'raw_next_md_exposed_to_best': raw_next_md_exposed, 'best_routing': 'full_prediction_fallback' if raw_next_md_exposed else 'digest_only'})
            try:
                (best_selection, best_raw, best_stats) = await self._chat_json(system_prompt=self.best_prompt, payload=best_payload, screenshot=screenshot, response_format=V3_LOCAL_BEST_ACTION_RESPONSE_FORMAT, parser=lambda raw: parse_v3_local_best_action_selection(raw, candidate_records), correction='Return one corrected best_action containing exactly candidate_index, unified_action, and action_intent. The action and intent must exactly copy the indexed candidate.')
            except Exception as exc:
                raise ModelAuditError(f'{self.policy_label} Best failed: {type(exc).__name__}: {exc}', audit_payload={'component': 'qwen35b_gui_agent', 'stage': 'best', 'error_kind': f'{self.error_kind_prefix}_best_failure', 'candidate_count': len(candidate_records), 'candidate_records': candidate_records, 'candidate_raw': candidate_raw, 'local_wm_prediction_audit': build_p2_prediction_audit(predictions), 'candidate_digests': candidate_digests, 'best_input_keys': sorted(best_payload), 'best_payload': best_payload, 'fallback_used': False}) from exc
            selected_index = int(best_selection['selected_index'])
            diagnostics.update({'best_action_called': True, 'best_action_input_keys': sorted(best_payload), 'best_action_raw': best_raw, 'best_action_model_stats': best_stats, 'best_action_reported_index_zero_based': best_selection['reported_index'], 'best_action_index_action_consistent': best_selection['index_action_consistent'], 'best_action_matching_indices_zero_based': best_selection['matching_indices']})
        selected = candidate_records[selected_index]
        diagnostics['selected_candidate_index_zero_based'] = selected_index
        diagnostics['selected_candidate_index'] = selected_index + 1
        return ({'selected_action': selected}, json.dumps(diagnostics, ensure_ascii=False), time.monotonic() - started)

class P3DigestAgent(P2DigestAgent):
    """P3 policy: expose full forecasts to Best only without positive Digest."""
    strategy_name = 'p3_digest'
    policy_label = 'P3'
    error_kind_prefix = 'p3'

    def _build_best_action_payload(self, common: dict[str, Any], candidate_records: list[dict[str, Any]], candidate_digests: list[dict[str, Any]], predicted_markdowns: list[str]) -> dict[str, Any]:
        return build_p3_best_action_payload(common, candidate_records, candidate_digests, predicted_markdowns)

class WMComparisonAgent(VanillaAgent):
    """Screenshot-only GUI Agent with an optional native one-step world model."""

    def __init__(self, config: RunnerConfig, *, candidate_prompt: str, best_prompt: str, world_model_kind: str, world_model_endpoint: str | None, world_model_name: str | None, world_model_prompt: str, world_model_path: Path | None, world_model_context_tokens: int, world_model_max_tokens: int, world_model_concurrency: int, world_model_adapter_path: Path | None=None, world_model_temperature: float=0.0, world_model_top_p: float=1.0, world_model_thinking_policy: str='disabled', wma_input_contract: str='official_contextual', v3_contract: bool=False) -> None:
        super().__init__(config)
        self.candidate_prompt = candidate_prompt
        self.best_prompt = best_prompt
        self.v3_contract = v3_contract
        self.world_model_kind = world_model_kind
        self.wma_input_contract = wma_input_contract
        self.world_model = None
        if world_model_kind != 'none':
            if not world_model_endpoint or not world_model_name:
                raise ValueError('native WM comparison requires endpoint and model name')
            from mkd2world.online_mind2web_wm import NativeWorldModelClient
            self.world_model = NativeWorldModelClient(kind=world_model_kind, endpoint=world_model_endpoint, model=world_model_name, system_prompt=world_model_prompt, max_tokens=world_model_max_tokens, context_tokens=world_model_context_tokens, timeout_seconds=max(config.model_timeout_seconds, 300), tokenizer_path=world_model_path, max_concurrency=world_model_concurrency, temperature=world_model_temperature, top_p=world_model_top_p, thinking_policy=world_model_thinking_policy, wma_input_contract=wma_input_contract)

    async def close(self) -> None:
        if self.world_model is not None:
            await self.world_model.close()
        await super().close()

    async def _chat_json(self, *, system_prompt: str, payload: dict[str, Any], screenshot: bytes, response_format: dict[str, Any], parser: Any, correction: str) -> tuple[Any, str, dict[str, Any]]:
        encoded = base64.b64encode(screenshot).decode('ascii')
        messages: list[dict[str, Any]] = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}, {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}', 'detail': 'high'}}]}]
        last_error: Exception | None = None
        attempts: list[dict[str, Any]] = []
        request_max_tokens = self.config.max_completion_tokens
        for attempt in range(self.config.model_retries + 1):
            try:
                thinking_token_budget = min(self.config.thinking_token_budget, max(1, request_max_tokens - 1))
                request_kwargs: dict[str, Any] = {'model': self.config.model, 'messages': messages, 'temperature': self.config.temperature, 'max_completion_tokens': request_max_tokens, 'response_format': response_format, 'extra_body': {'thinking_token_budget': thinking_token_budget, 'chat_template_kwargs': {'enable_thinking': self.config.enable_thinking}}}
                if self.config.enable_thinking:
                    request_kwargs['reasoning_effort'] = self.config.reasoning_effort
                response = await self.client.chat.completions.create(**request_kwargs)
                choice = response.choices[0]
                raw = choice.message.content or ''
                try:
                    parsed = parser(raw)
                except Exception as exc:
                    last_error = ValueError(f'{exc}; raw_response={raw[:2000]!r}')
                    attempts.append({'attempt': attempt, 'raw_output': raw, 'finish_reason': choice.finish_reason, 'validation_error': f'{type(exc).__name__}: {exc}'})
                    if attempt < self.config.model_retries:
                        messages.extend([{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': f'{correction} Validation error: {exc}'}])
                        await asyncio.sleep(2 ** attempt)
                    continue
                usage = response.usage
                return (parsed, raw, {'finish_reason': choice.finish_reason, 'prompt_tokens': getattr(usage, 'prompt_tokens', None), 'completion_tokens': getattr(usage, 'completion_tokens', None), 'total_tokens': getattr(usage, 'total_tokens', None), 'output_limit_hit': choice.finish_reason == 'length'})
            except Exception as exc:
                last_error = exc
                adjusted = adjust_max_completion_tokens_from_context_error(exc, request_max_tokens, retry_ceiling=512 if response_format is BEST_ACTION_RESPONSE_FORMAT else 1024)
                if not attempts or attempts[-1].get('attempt') != attempt:
                    attempts.append({'attempt': attempt, 'request_max_completion_tokens': request_max_tokens, 'request_error': f'{type(exc).__name__}: {exc}'})
                if adjusted < request_max_tokens:
                    request_max_tokens = adjusted
                if attempt < self.config.model_retries:
                    await asyncio.sleep(2 ** attempt)
        raise ModelAuditError(f'WM comparison GUI Agent failed after {self.config.model_retries + 1} attempts: {last_error}', audit_payload={'component': 'qwen35b_gui_agent', 'error_kind': classify_model_failure(last_error), 'system_prompt_sha256': hashlib.sha256(system_prompt.encode('utf-8')).hexdigest(), 'input_payload_chars': len(json.dumps(payload, ensure_ascii=False)), 'attempts': attempts})

    async def choose(self, goal: str, url: str, history: list[str], screenshot: bytes, markdown: str, *, page: Page | None=None) -> tuple[dict[str, Any], str, float]:
        if markdown:
            raise ValueError('wm_compare forbids project Markdown in GUI Agent requests')
        started = time.monotonic()
        trace = '\n'.join((f'{i + 1}. {item}' for (i, item) in enumerate(history[-20:]))) or '(none)'
        common = {'goal_text': goal, 'current_url': url, 'current_subgoal_trace': trace, 'supported_unified_actions': SUPPORTED_ACTIONS}
        candidate_response_format = V3_LOCAL_CANDIDATE_ACTIONS_RESPONSE_FORMAT if self.v3_contract else CANDIDATE_ACTIONS_RESPONSE_FORMAT
        candidate_parser = parse_v3_local_candidate_actions_json if self.v3_contract else parse_candidate_actions_json
        candidate_correction = 'Return exactly 1 candidate only when its confidence is 8..10; when confidence is 0..7, return 2 or 3 complementary candidates. Each item must contain exactly unified_action, action_intent, and confidence.' if self.v3_contract else CANDIDATE_ACTIONS_CORRECTION
        (candidates_payload, candidate_raw, candidate_stats) = await self._chat_json(system_prompt=self.candidate_prompt, payload=common, screenshot=screenshot, response_format=candidate_response_format, parser=candidate_parser, correction=candidate_correction)
        candidate_records = list(candidates_payload)
        candidate_confidences = [int(candidate['confidence']) for candidate in candidate_records]
        trigger_reason = wm_trigger_reason(candidate_records) if self.v3_contract else 'multiple_candidates' if len(candidate_records) > 1 else 'none'
        diagnostics: dict[str, Any] = {'strategy': 'wm_compare', 'world_model_kind': self.world_model_kind, 'wma_input_contract': self.wma_input_contract if self.world_model_kind == 'wma' else None, 'candidate_count': len(candidate_records), 'candidate_confidences': candidate_confidences, 'wm_trigger_reason': trigger_reason, 'candidate_raw': candidate_raw, 'candidate_model_stats': candidate_stats, 'world_model_called': False, 'best_action_called': False}
        if trigger_reason == 'none':
            selected_index = 0
        else:
            predictions = []
            if self.world_model is not None:
                if page is None:
                    raise ValueError('wm_compare requires the live Playwright page')
                indexed_nonterminal = [(index, candidate) for (index, candidate) in enumerate(candidate_records) if candidate['unified_action']['type'] not in {'finish', 'fail'}]
                if indexed_nonterminal:
                    axtree_action_only = self.world_model_kind == 'wma' and self.wma_input_contract == 'axtree_action_only'
                    predictions = await self.world_model.predict_candidates(page=page, screenshot=b'' if axtree_action_only else screenshot, candidates=[{'unified_action': item[1]['unified_action']} for item in indexed_nonterminal], candidate_indices=[item[0] for item in indexed_nonterminal], width=self.config.width, height=self.config.height, url='' if axtree_action_only else url, objective='' if axtree_action_only else goal, previous_action='' if axtree_action_only else wma_previous_action_from_trace(history[-1] if history else None) if self.world_model_kind == 'wma' else history[-1] if history else 'None')
                    diagnostics['world_model_called'] = any((item.wm_called for item in predictions))
            prediction_payload = [item.to_dict() for item in predictions]
            predicted_indices = {item['candidate_index'] for item in prediction_payload}
            for (index, candidate) in enumerate(candidate_records):
                if index not in predicted_indices:
                    terminal = candidate['unified_action']['type'] in {'finish', 'fail'}
                    prediction_payload.append({'candidate_index': index, 'observation_format': 'terminal' if terminal else 'unavailable', 'predicted_observation': 'Terminal candidate; no browser state transition is simulated.' if terminal else 'No world model is configured for this candidate.', 'wm_called': False})
            prediction_payload.sort(key=lambda item: int(item['candidate_index']))
            diagnostics['world_model_predictions'] = prediction_payload
            if self.v3_contract:
                best_payload = build_v3_other_best_action_payload(common, candidate_records, prediction_payload)
                best_response_format = V3_LOCAL_BEST_ACTION_RESPONSE_FORMAT
                best_parser = lambda raw: parse_v3_local_best_action_selection(raw, candidate_records)
                best_correction = 'Return one corrected best_action containing exactly candidate_index, unified_action, and action_intent. The unified_action and action_intent must exactly copy the indexed candidate.'
            else:
                best_payload = build_aligned_best_action_payload(common, candidate_records, prediction_payload)
                best_response_format = BEST_ACTION_RESPONSE_FORMAT
                best_parser = lambda raw: parse_best_action_selection(raw, candidate_records)
                best_correction = 'Return one corrected best_action. candidate_index is zero-based and unified_action must exactly copy candidate_actions[candidate_index].'
            (best_selection, best_raw, best_stats) = await self._chat_json(system_prompt=self.best_prompt, payload=best_payload, screenshot=screenshot, response_format=best_response_format, parser=best_parser, correction=best_correction)
            selected_index = int(best_selection['selected_index'])
            diagnostics.update({'best_action_called': True, 'best_action_raw': best_raw, 'best_action_model_stats': best_stats, 'best_action_reported_index_zero_based': best_selection['reported_index'], 'best_action_index_action_consistent': best_selection['index_action_consistent'], 'best_action_matching_indices_zero_based': best_selection['matching_indices']})
        selected = candidate_records[selected_index]
        diagnostics['selected_candidate_index_zero_based'] = selected_index
        return ({'selected_action': selected}, json.dumps(diagnostics, ensure_ascii=False), time.monotonic() - started)

async def current_page(context: BrowserContext, fallback: Page) -> Page:
    pages = [page for page in context.pages if not page.is_closed()]
    return pages[-1] if pages else fallback

async def capture(page: Page, path: Path, width: int, height: int) -> bytes:
    try:
        payload = await page.screenshot(path=str(path), type='png', full_page=False, timeout=30000)
        return payload
    except Exception as exc:
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((30, 30), f'Screenshot unavailable: {type(exc).__name__}: {exc}', fill='black')
        image.save(path)
        return path.read_bytes()

async def page_block_reason(page: Page) -> str | None:
    try:
        body = (await page.locator('body').inner_text(timeout=3000))[:20000]
    except Exception:
        body = ''
    match = CAPTCHA_RE.search(f'{page.url}\n{body}')
    return match.group(0) if match else None

def action_thought(selected: dict[str, Any]) -> str:
    return f"Intent: {selected['action_intent']}"

def grid_description(action: dict[str, Any], selected: dict[str, Any], width: int, height: int) -> str:
    kind = action['type']
    intent = ' '.join(selected['action_intent'].split())
    if kind in COORDINATE_ACTIONS or kind in NATIVE_COORDINATE_ACTIONS or (kind in {'select_option', 'clear', 'type_text'} and 'coordinate' in action):
        (x, y) = scale_coordinate(action['coordinate'], width, height)
        return f'{kind.upper()} coords({x}, {y}) -> {intent}'
    if kind == 'drag':
        (x1, y1) = scale_coordinate(action['from_coordinate'], width, height)
        (x2, y2) = scale_coordinate(action['to_coordinate'], width, height)
        return f'DRAG coords({x1}, {y1}) to coords({x2}, {y2}) -> {intent}'
    if kind == 'type_text':
        return f"TYPE {json.dumps(action['text'], ensure_ascii=False)} -> {intent}"
    if kind == 'scroll':
        return f"SCROLL ({action['delta_x']}, {action['delta_y']}) -> {intent}"
    if kind == 'navigate':
        return f'page -> NAVIGATE -> {intent}'
    if kind == 'key_press':
        return f"KEY_PRESS {action['key']} -> {intent}"
    if kind == 'hotkey':
        return f"HOTKEY {'+'.join(action['keys'])} -> {intent}"
    return f'{kind.upper()} -> {intent}'

async def execute_action(page: Page, context: BrowserContext, action: dict[str, Any], config: RunnerConfig) -> tuple[Page, bool, str]:
    kind = action['type']
    try:
        if kind in COORDINATE_ACTIONS or kind in NATIVE_COORDINATE_ACTIONS:
            (x, y) = scale_coordinate(action['coordinate'], config.width, config.height)
            if kind == 'left_click' or kind == 'toggle_check':
                await page.mouse.click(x, y)
            elif kind == 'double_click':
                await page.mouse.dblclick(x, y)
            elif kind == 'right_click':
                await page.mouse.click(x, y, button='right')
            elif kind == 'middle_click':
                await page.mouse.click(x, y, button='middle')
            elif kind == 'triple_click':
                await page.mouse.click(x, y, click_count=3)
            else:
                await page.mouse.move(x, y)
        elif kind == 'type_text':
            if 'coordinate' in action:
                (x, y) = scale_coordinate(action['coordinate'], config.width, config.height)
                await page.mouse.click(x, y)
            if action.get('clear'):
                await page.keyboard.press('Control+A')
                await page.keyboard.press('Backspace')
            await page.keyboard.type(action['text'])
            if action.get('enter'):
                await page.keyboard.press('Enter')
        elif kind == 'scroll':
            dx = round(action['delta_x'] * config.width / 999)
            dy = round(action['delta_y'] * config.height / 999)
            await page.mouse.wheel(dx, dy)
        elif kind == 'key_press':
            await page.keyboard.press(action['key'])
        elif kind == 'hotkey':
            key_alias = {'ctrl': 'Control', 'cmd': 'Meta', 'alt': 'Alt', 'shift': 'Shift'}
            combo = '+'.join((key_alias.get(key.lower(), key) for key in action['keys']))
            await page.keyboard.press(combo)
        elif kind == 'navigate':
            await page.goto(action['url'], wait_until='domcontentloaded', timeout=config.navigation_timeout_ms)
        elif kind == 'go_back':
            await page.go_back(wait_until='domcontentloaded', timeout=config.navigation_timeout_ms)
        elif kind == 'go_forward':
            await page.go_forward(wait_until='domcontentloaded', timeout=config.navigation_timeout_ms)
        elif kind == 'refresh':
            await page.reload(wait_until='domcontentloaded', timeout=config.navigation_timeout_ms)
        elif kind == 'wait':
            await page.wait_for_timeout(max(config.action_wait_ms, 2000))
        elif kind == 'select_option':
            (x, y) = scale_coordinate(action['coordinate'], config.width, config.height)
            await page.locator(f'select').evaluate_all("(nodes, p) => { const e=document.elementFromPoint(p.x,p.y); if (!e || e.tagName !== 'SELECT') throw new Error('target is not select'); }", {'x': x, 'y': y})
            handle = await page.evaluate_handle('([x,y]) => document.elementFromPoint(x,y)', [x, y])
            await handle.as_element().select_option(label=action['option_text'])
        elif kind == 'drag':
            (x1, y1) = scale_coordinate(action['from_coordinate'], config.width, config.height)
            (x2, y2) = scale_coordinate(action['to_coordinate'], config.width, config.height)
            await page.mouse.move(x1, y1)
            await page.mouse.down()
            await page.mouse.move(x2, y2, steps=10)
            await page.mouse.up()
        elif kind == 'clear':
            if 'coordinate' in action:
                (x, y) = scale_coordinate(action['coordinate'], config.width, config.height)
                await page.mouse.click(x, y)
            await page.keyboard.press('Control+A')
            await page.keyboard.press('Backspace')
        else:
            raise ValueError(f'cannot execute action {kind}')
        await page.wait_for_timeout(config.action_wait_ms)
        page = await current_page(context, page)
        return (page, True, '')
    except Exception as exc:
        return (await current_page(context, page), False, f'{type(exc).__name__}: {exc}')

def add_step(history: list[dict[str, Any]], screenshot: str, url: str | None, action: str, thought: str | None, status: str | None=None) -> None:
    item: dict[str, Any] = {'step': len(history), 'screenshot': screenshot, 'url': url if url and urlparse(url).scheme in {'http', 'https'} else None, 'action': action, 'thought': thought}
    if status is not None:
        item['action_status'] = status
    history.append(item)

def navigation_error_is_transient(error: str) -> bool:
    """Return whether an entry-navigation failure is safe to retry unchanged."""
    lowered = error.lower()
    return any((marker in lowered for marker in ('timeout', 'err_http2_protocol_error', 'err_aborted', 'err_connection_', 'err_network_', 'err_name_not_resolved', 'err_internet_disconnected', 'http 408', 'http 425', 'http 429', 'http 500', 'http 502', 'http 503', 'http 504')))

async def navigate_entry(page: Page, url: str, *, timeout_ms: int, retries: int, wait_until: str='domcontentloaded') -> tuple[bool, str, list[dict[str, Any]]]:
    """Open the benchmark-supplied URL with finite, fully logged retries.

    Retries never alter the URL and are limited to transient transport/server
    failures.  Access denial, CAPTCHA, and other HTTP 4xx responses are not
    retried or bypassed.
    """
    attempts: list[dict[str, Any]] = []
    for attempt_index in range(retries + 1):
        started = time.monotonic()
        status: int | None = None
        final_url = page.url
        error = ''
        try:
            response = await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            final_url = page.url
            if response is not None:
                status = response.status
            if status is None or status < 400:
                attempts.append({'attempt': attempt_index + 1, 'ok': True, 'status_code': status, 'final_url': final_url, 'elapsed_seconds': round(time.monotonic() - started, 3), 'error': ''})
                return (True, '', attempts)
            error = f'HTTP {status}'
        except Exception as exc:
            final_url = page.url
            error = f'{type(exc).__name__}: {exc}'
        attempts.append({'attempt': attempt_index + 1, 'ok': False, 'status_code': status, 'final_url': final_url, 'elapsed_seconds': round(time.monotonic() - started, 3), 'error': error})
        if attempt_index >= retries or not navigation_error_is_transient(error):
            return (False, error, attempts)
        await page.wait_for_timeout(min(500 * 2 ** attempt_index, 2000))
    raise AssertionError('navigation retry loop must return')

async def run_task(task: dict[str, Any], browser: Any, agent: OnlineMind2WebAgent, config: RunnerConfig, output_root: Path, schema_path: Path, resume: bool) -> dict[str, Any]:
    task_id = str(task['task_id'])
    task_dir = output_root / task_id
    result_path = task_dir / 'result.json'
    if resume and result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding='utf-8'))
            validate_v2_result(existing, task_dir, schema_path)
            metadata_path = task_dir / 'run_metadata.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
            if should_retry_gui_context_overflow(metadata, current_agent_context_tokens=config.agent_context_tokens):
                attempt = archive_gui_context_overflow(task_dir)
                print(json.dumps({'task_id': task_id, 'status': 'gui_context_expansion_retry_scheduled', 'archived_attempt': attempt, 'agent_context_tokens': config.agent_context_tokens}, ensure_ascii=False), flush=True)
            else:
                return {'task_id': task_id, 'status': 'skipped_valid'}
        except Exception:
            if result_path.is_file():
                result_path.rename(task_dir / f'result.invalid.{int(time.time())}.json')
    trajectory_dir = task_dir / 'trajectory'
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for old in trajectory_dir.glob('*.png'):
        old.unlink()
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    prompt_trace: list[str] = []
    raw_steps: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {'status': 'running', 'difficulty': task.get('level'), 'website': task['website']}
    context_options: dict[str, Any] = {'viewport': {'width': config.width, 'height': config.height}, 'screen': {'width': config.width, 'height': config.height}, 'locale': 'en-US', 'accept_downloads': False}
    if config.browser_user_agent:
        context_options['user_agent'] = config.browser_user_agent
    context = await browser.new_context(**context_options)
    context.set_default_timeout(15000)
    page = await context.new_page()
    final_answer: str | None = None
    terminal_reason = 'unknown'
    recent_pairs: list[tuple[str, str]] = []
    try:
        (nav_ok, nav_error, navigation_attempts) = await navigate_entry(page, str(task['website']), timeout_ms=config.navigation_timeout_ms, retries=config.navigation_retries, wait_until=config.entry_wait_until)
        diagnostics['navigation_attempts'] = navigation_attempts
        if nav_ok and config.entry_settle_ms:
            await page.wait_for_timeout(config.entry_settle_ms)
        page = await current_page(context, page)
        shot = trajectory_dir / '0000.png'
        await capture(page, shot, config.width, config.height)
        add_step(history, shot.name, page.url, f"page -> NAVIGATE -> dataset website | {('SUCCESS' if nav_ok else 'FAILED')}", 'Open the exact website supplied by the benchmark task.', 'SUCCESS' if nav_ok else 'FAILED')
        prompt_trace.append(build_prompt_trace_entry({'type': 'navigate', 'url': str(task['website'])}, 'Open the exact website supplied by the benchmark task.', 'SUCCESS' if nav_ok else 'FAILED'))
        if not nav_ok:
            terminal_reason = 'initial_navigation_error'
            diagnostics['navigation_error'] = nav_error
        blocked = await page_block_reason(page)
        if blocked:
            terminal_reason = 'blocked_page'
            diagnostics['blocked_indicator'] = blocked
        for step in range(1, config.max_steps - 1):
            if terminal_reason != 'unknown':
                break
            page = await current_page(context, page)
            shot = trajectory_dir / f'{step:04d}.png'
            screenshot = await capture(page, shot, config.width, config.height)
            screenshot_hash = hashlib.sha256(screenshot).hexdigest()
            markdown = ''
            try:
                markdown = await agent.render_markdown(screenshot)
            except Exception as exc:
                terminal_reason = 'markdown_error'
                diagnostics['markdown_error'] = f'{type(exc).__name__}: {exc}'
                shot.unlink(missing_ok=True)
                break
            try:
                (payload, raw, latency) = await agent.choose(str(task['confirmed_task']), page.url, prompt_trace, screenshot, markdown, page=page)
            except Exception as exc:
                terminal_reason = classify_model_failure(exc)
                diagnostics['model_error'] = f'{type(exc).__name__}: {exc}'
                audit_payload = getattr(exc, 'audit_payload', None)
                if isinstance(audit_payload, dict):
                    diagnostics['model_error_audit'] = audit_payload
                shot.unlink(missing_ok=True)
                break
            selected = payload['selected_action']
            action = selected['unified_action']
            raw_steps.append({'step': step, 'raw_response': raw, 'parsed': payload, 'model_latency_seconds': latency, 'markdown': markdown if config.mode == 'screenshot_md' else None})
            kind = action['type']
            if kind in {'finish', 'fail'}:
                final_answer = str(action.get('answer') or action.get('reason') or '')
                terminal_reason = 'agent_finish' if kind == 'finish' else 'agent_fail'
                add_step(history, shot.name, page.url, task_complete_action(final_answer), action_thought(selected))
                break
            if BLOCKED_INTENT_RE.search(selected['action_intent']):
                final_answer = 'Blocked an irreversible real-world action.'
                terminal_reason = 'irreversible_action_blocked'
                diagnostics['blocked_action'] = action
                add_step(history, shot.name, page.url, task_complete_action(final_answer), action_thought(selected))
                break
            description = grid_description(action, selected, config.width, config.height)
            (page, ok, error) = await execute_action(page, context, action, config)
            action_string = f"{description} | {('SUCCESS' if ok else 'FAILED')}"
            add_step(history, shot.name, page.url, action_string, action_thought(selected), 'SUCCESS' if ok else 'FAILED')
            prompt_trace.append(build_prompt_trace_entry(action, selected['action_intent'], 'SUCCESS' if ok else 'FAILED'))
            raw_steps[-1]['execution_error'] = error or None
            pair = (screenshot_hash, json.dumps(action, sort_keys=True, ensure_ascii=False))
            recent_pairs.append(pair)
            if len(recent_pairs) >= config.no_progress_limit and len(set(recent_pairs[-config.no_progress_limit:])) == 1:
                terminal_reason = 'repeated_no_progress'
                break
            blocked = await page_block_reason(page)
            if blocked:
                terminal_reason = 'blocked_page'
                diagnostics['blocked_indicator'] = blocked
                break
        if not history[-1]['action'].startswith('TASK_COMPLETE'):
            if terminal_reason == 'unknown':
                terminal_reason = 'max_steps'
            step = len(history)
            shot = trajectory_dir / f'{step:04d}.png'
            await capture(await current_page(context, page), shot, config.width, config.height)
            final_answer = final_answer or f'Task stopped: {terminal_reason}.'
            add_step(history, shot.name, page.url, task_complete_action(final_answer), f'Terminate because {terminal_reason}.')
        result = {'schema_version': 'online-mind2web-v2', 'task': str(task['confirmed_task']), 'task_id': task_id, 'agent_final_answer': final_answer, 'reference_length': int(task['reference_length']), 'action_history': history}
        diagnostics.update({'status': terminal_reason, 'step_count': len(history), 'elapsed_seconds': round(time.monotonic() - started, 3), 'completed_at': datetime.now().astimezone().isoformat(timespec='seconds')})
        if terminal_reason == 'model_error' and isinstance(agent, (LocalWMAgent, WMComparisonAgent)):
            attempt = 1
            while (task_dir / f'result.attempt_{attempt:02d}_model_error.json').exists():
                attempt += 1
            diagnostics['audit_attempt'] = attempt
            atomic_json(task_dir / f'raw_steps.attempt_{attempt:02d}_model_error.json', raw_steps)
            atomic_json(task_dir / f'run_metadata.attempt_{attempt:02d}_model_error.json', diagnostics)
            atomic_json(task_dir / f'result.attempt_{attempt:02d}_model_error.json', result)
            return {'task_id': task_id, 'status': 'model_error', 'steps': len(history), 'elapsed_seconds': diagnostics['elapsed_seconds'], 'audit_attempt': attempt}
        atomic_json(task_dir / 'raw_steps.json', raw_steps)
        atomic_json(task_dir / 'run_metadata.json', diagnostics)
        validate_v2_result(result, task_dir, schema_path)
        atomic_json(result_path, result)
        return {'task_id': task_id, 'status': terminal_reason, 'steps': len(history), 'elapsed_seconds': diagnostics['elapsed_seconds']}
    except Exception as exc:
        diagnostics.update({'status': 'runner_exception', 'exception': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': round(time.monotonic() - started, 3)})
        try:
            page = await current_page(context, page)
            if not history:
                shot = trajectory_dir / '0000.png'
                await capture(page, shot, config.width, config.height)
                add_step(history, shot.name, page.url, 'page -> NAVIGATE -> dataset website | FAILED', 'The runner failed while opening the benchmark website.', 'FAILED')
            if len(history) < config.max_steps and (not history[-1]['action'].startswith('TASK_COMPLETE')):
                shot = trajectory_dir / f'{len(history):04d}.png'
                await capture(page, shot, config.width, config.height)
                add_step(history, shot.name, page.url, task_complete_action('Task stopped because the runner encountered an isolated error.'), 'Terminate and preserve an auditable failure trajectory.')
            fallback = {'schema_version': 'online-mind2web-v2', 'task': str(task['confirmed_task']), 'task_id': task_id, 'agent_final_answer': 'Task stopped because the runner encountered an isolated error.', 'reference_length': int(task['reference_length']), 'action_history': history}
            validate_v2_result(fallback, task_dir, schema_path)
            atomic_json(result_path, fallback)
        except Exception as fallback_exc:
            diagnostics['fallback_exception'] = f'{type(fallback_exc).__name__}: {fallback_exc}'
        atomic_json(task_dir / 'run_metadata.json', diagnostics)
        return {'task_id': task_id, 'status': 'runner_exception', 'error': diagnostics['exception']}
    finally:
        await context.close()

def should_retry_wm_technical_failure(summary: dict[str, Any], *, retries_completed: int, max_retries: int) -> bool:

    return summary.get('status') == 'model_error' and retries_completed < max(0, max_retries)

def completed_wm_model_error_attempts(task_dir: Path) -> list[int]:
    """Return model-error attempt numbers that have all three audit artifacts."""
    attempts: list[int] = []
    for result_path in task_dir.glob('result.attempt_*_model_error.json'):
        match = re.fullmatch('result\\.attempt_(\\d+)_model_error\\.json', result_path.name)
        if match is None:
            continue
        attempt = int(match.group(1))
        if all(((task_dir / f'{stem}.attempt_{attempt:02d}_model_error.json').is_file() for stem in ('result', 'run_metadata', 'raw_steps'))):
            attempts.append(attempt)
    return sorted(set(attempts))

def finalize_exhausted_wm_model_error(task_dir: Path, *, schema_path: Path | None) -> int:

    attempts = completed_wm_model_error_attempts(task_dir)
    if not attempts:
        raise RuntimeError(f'No complete model-error attempt exists in {task_dir}')
    attempt = attempts[-1]
    raw_steps = json.loads((task_dir / f'raw_steps.attempt_{attempt:02d}_model_error.json').read_text(encoding='utf-8'))
    metadata = json.loads((task_dir / f'run_metadata.attempt_{attempt:02d}_model_error.json').read_text(encoding='utf-8'))
    result = json.loads((task_dir / f'result.attempt_{attempt:02d}_model_error.json').read_text(encoding='utf-8'))
    if metadata.get('status') != 'model_error':
        raise ValueError(f'Final WM attempt {attempt} in {task_dir} is not a model_error')
    metadata['technical_retry_exhausted'] = True
    metadata['technical_attempts'] = attempt
    validate_v2_result(result, task_dir, schema_path)
    atomic_json(task_dir / 'raw_steps.json', raw_steps)
    atomic_json(task_dir / 'run_metadata.json', metadata)
    atomic_json(task_dir / 'result.json', result)
    return attempt

@dataclass(frozen=True)
class TaskClaim:
    task_id: str
    path: Path
    token: str
    owner: dict[str, Any]

def scheduled_tasks(selected_tasks: list[dict[str, Any]], *, shard_index: int, num_shards: int, task_offset: int, task_limit: int, task_scheduling: str) -> list[dict[str, Any]]:


    def lane_slice(lane: int) -> list[dict[str, Any]]:
        lane_tasks = selected_tasks[lane::num_shards]
        if task_offset > 0:
            lane_tasks = lane_tasks[task_offset:]
        if task_limit > 0:
            lane_tasks = lane_tasks[:task_limit]
        return lane_tasks
    if task_scheduling == 'static':
        return lane_slice(shard_index)
    included = {str(task['task_id']) for lane in range(num_shards) for task in lane_slice(lane)}
    return [task for task in selected_tasks if str(task['task_id']) in included]

def shared_task_order(tasks: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    """Rotate each lane into a different region while preserving work stealing."""
    if not tasks:
        return []
    offset = len(tasks) * shard_index // num_shards
    return tasks[offset:] + tasks[:offset]

def _process_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks so a recycled PID cannot own a claim."""
    try:
        payload = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
        fields = payload[payload.rfind(')') + 2:].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None

def _task_claim_path(queue_root: Path, task_id: str) -> Path:
    readable = re.sub('[^A-Za-z0-9_.-]', '_', task_id)[:80] or 'task'
    digest = hashlib.sha256(task_id.encode('utf-8')).hexdigest()[:12]
    return queue_root / 'claims' / f'{readable}-{digest}.claim'

def _claim_audit(queue_root: Path, payload: dict[str, Any]) -> None:
    token = str(payload.get('token') or uuid.uuid4().hex)
    event = str(payload.get('event') or 'event')
    atomic_json(queue_root / 'audit' / f'{time.time_ns()}-{token}-{event}.json', payload)

def _claim_is_stale(claim_path: Path, *, stale_seconds: float, write_grace_seconds: float=30.0) -> bool:
    try:
        age = max(0.0, time.time() - claim_path.stat().st_mtime)
    except FileNotFoundError:
        return False
    owner_path = claim_path / 'owner.json'
    if not owner_path.is_file():
        return age >= write_grace_seconds
    try:
        owner = json.loads(owner_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return age >= write_grace_seconds
    if owner.get('hostname') == socket.gethostname():
        pid = owner.get('pid')
        expected_ticks = owner.get('process_start_ticks')
        if isinstance(pid, int) and isinstance(expected_ticks, int):
            if _process_start_ticks(pid) == expected_ticks:
                return False
            return age >= 5.0
    return age >= stale_seconds

def try_acquire_task_claim(queue_root: Path, task_id: str, *, shard_index: int, stale_seconds: float) -> TaskClaim | None:
    """Atomically claim a task, reclaiming only provably dead/stale owners."""
    queue_root.mkdir(parents=True, exist_ok=True)
    (queue_root / 'claims').mkdir(parents=True, exist_ok=True)
    claim_path = _task_claim_path(queue_root, task_id)
    for _ in range(2):
        try:
            claim_path.mkdir()
        except FileExistsError:
            if not _claim_is_stale(claim_path, stale_seconds=stale_seconds):
                return None
            stale_root = queue_root / 'stale'
            stale_root.mkdir(parents=True, exist_ok=True)
            stale_path = stale_root / f'{claim_path.name}.{uuid.uuid4().hex}'
            try:
                os.replace(claim_path, stale_path)
            except (FileNotFoundError, FileExistsError, OSError):
                return None
            owner_path = stale_path / 'owner.json'
            owner: dict[str, Any] = {}
            if owner_path.is_file():
                try:
                    owner = json.loads(owner_path.read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    owner = {}
            _claim_audit(queue_root, {**owner, 'event': 'reclaimed_stale', 'task_id': task_id, 'reclaimed_at': datetime.now().astimezone().isoformat(timespec='seconds')})
            shutil.rmtree(stale_path, ignore_errors=True)
            continue
        token = uuid.uuid4().hex
        owner = {'event': 'claimed', 'task_id': task_id, 'token': token, 'hostname': socket.gethostname(), 'pid': os.getpid(), 'process_start_ticks': _process_start_ticks(os.getpid()), 'shard_index': shard_index, 'claimed_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'claimed_at_epoch': time.time()}
        atomic_json(claim_path / 'owner.json', owner)
        _claim_audit(queue_root, owner)
        return TaskClaim(task_id=task_id, path=claim_path, token=token, owner=owner)
    return None

def release_task_claim(queue_root: Path, claim: TaskClaim, *, status: str) -> None:
    """Release only the caller-owned claim and retain a completion audit."""
    owner_path = claim.path / 'owner.json'
    try:
        owner = json.loads(owner_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return
    if owner.get('token') != claim.token:
        return
    _claim_audit(queue_root, {**owner, 'event': 'released', 'status': status, 'released_at': datetime.now().astimezone().isoformat(timespec='seconds')})
    shutil.rmtree(claim.path)

def task_needs_run(task: dict[str, Any], *, output_root: Path, schema_path: Path, resume: bool, agent_context_tokens: int) -> bool:
    """Use the same durable validity rules as ``run_task`` before claiming."""
    if not resume:
        return True
    task_dir = output_root / str(task['task_id'])
    result_path = task_dir / 'result.json'
    if not result_path.is_file():
        return True
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
        validate_v2_result(result, task_dir, schema_path)
        metadata_path = task_dir / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
        return should_retry_gui_context_overflow(metadata, current_agent_context_tokens=agent_context_tokens)
    except Exception:
        return True

async def async_main(args: argparse.Namespace) -> int:
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    if not isinstance(dataset, list):
        raise ValueError('dataset must be a JSON list')
    task_selection: list[str] | None = None
    if args.task_manifest is not None:
        task_selection = load_task_ids(args.task_manifest)
        dataset_ids = {str(task['task_id']) for task in dataset}
        missing = sorted(set(task_selection) - dataset_ids)
        if missing:
            raise ValueError(f'Task manifest IDs absent from dataset: {missing[:10]}')
        selected_ids = set(task_selection)
        selected_tasks = [task for task in dataset if str(task['task_id']) in selected_ids]
    else:
        selected_tasks = dataset
    tasks = scheduled_tasks(selected_tasks, shard_index=args.shard_index, num_shards=args.num_shards, task_offset=args.task_offset, task_limit=args.task_limit, task_scheduling=args.task_scheduling)
    manifest_selected_tasks = scheduled_tasks(selected_tasks, shard_index=0, num_shards=args.num_shards, task_offset=args.task_offset, task_limit=args.task_limit, task_scheduling='shared') if args.world_model_kind in {'wma', 'webworld32b'} else selected_tasks
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = args.prompt.read_text(encoding='utf-8')
    snapshot = args.output_dir / 'prompt_snapshot.md'
    if snapshot.exists() and snapshot.read_text(encoding='utf-8') != prompt_text:
        raise RuntimeError('output directory contains a different prompt snapshot')
    atomic_text(snapshot, prompt_text)
    best_prompt_text = ''
    if args.agent_strategy in {'local_wm', 'p2_digest', 'p3_digest', 'wm_compare'}:
        best_prompt_text = args.best_prompt.read_text(encoding='utf-8')
        best_snapshot = args.output_dir / 'best_action_prompt_snapshot.md'
        if best_snapshot.exists() and best_snapshot.read_text(encoding='utf-8') != best_prompt_text:
            raise RuntimeError('output directory contains a different best-action prompt snapshot')
        atomic_text(best_snapshot, best_prompt_text)
    digest_prompt_text = ''
    if args.agent_strategy in {'p2_digest', 'p3_digest'}:
        digest_prompt_text = args.digest_prompt.read_text(encoding='utf-8')
        digest_snapshot = args.output_dir / 'digest_prompt_snapshot.md'
        if digest_snapshot.exists() and digest_snapshot.read_text(encoding='utf-8') != digest_prompt_text:
            raise RuntimeError('output directory contains a different Digest prompt snapshot')
        atomic_text(digest_snapshot, digest_prompt_text)
    local_wm_prompt_text = ''
    local_wm_prompt_path: Path | None = None
    if args.agent_strategy in {'local_wm', 'p2_digest', 'p3_digest'}:
        registry = PromptRegistry.from_project(PROJECT_ROOT)
        local_wm_prompt_path = registry.path_for(args.local_wm_prompt_key)
        local_wm_prompt_text = local_wm_prompt_path.read_text(encoding='utf-8')
        local_wm_snapshot = args.output_dir / 'local_wm_prompt_snapshot.md'
        if local_wm_snapshot.exists() and local_wm_snapshot.read_text(encoding='utf-8') != local_wm_prompt_text:
            raise RuntimeError('output directory contains a different Local WM prompt snapshot')
        atomic_text(local_wm_snapshot, local_wm_prompt_text)
    world_model_prompt_text = ''
    if args.agent_strategy == 'wm_compare' and args.world_model_kind != 'none':
        world_model_prompt_text = args.world_model_prompt.read_text(encoding='utf-8')
        world_snapshot = args.output_dir / 'world_model_prompt_snapshot.md'
        if world_snapshot.exists() and world_snapshot.read_text(encoding='utf-8') != world_model_prompt_text:
            raise RuntimeError('output directory contains a different world-model prompt snapshot')
        atomic_text(world_snapshot, world_model_prompt_text)
    manifest_path = args.output_dir / 'manifest.json'
    manifest = {'run_id': args.output_dir.name, 'created_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'dataset': str(args.dataset.resolve()), 'dataset_sha256': sha256_file(args.dataset), 'dataset_task_count': len(dataset), 'task_order': [str(task['task_id']) for task in dataset], 'task_selection': {'manifest': str(args.task_manifest.resolve()) if args.task_manifest else None, 'manifest_sha256': sha256_file(args.task_manifest) if args.task_manifest else None, 'task_ids': [str(task['task_id']) for task in manifest_selected_tasks]} if args.task_manifest or (args.world_model_kind in {'wma', 'webworld32b'} and (args.task_limit > 0 or args.task_offset > 0)) else None, 'difficulty_distribution': {level: sum((str(task.get('level', '')).lower() == level for task in manifest_selected_tasks)) for level in ('easy', 'medium', 'hard')}, 'mode': args.mode, 'agent_strategy': args.agent_strategy, 'model': args.model, 'model_path': str(args.model_path.resolve()), 'model_fingerprint': model_fingerprint(args.model_path), 'prompt': str(args.prompt.resolve()), 'prompt_sha256': hashlib.sha256(prompt_text.encode()).hexdigest(), 'best_prompt': str(args.best_prompt.resolve()) if args.agent_strategy in {'local_wm', 'p2_digest', 'p3_digest', 'wm_compare'} else None, 'best_prompt_sha256': hashlib.sha256(best_prompt_text.encode()).hexdigest() if best_prompt_text else None, 'digest_prompt': str(args.digest_prompt.resolve()) if args.agent_strategy in {'p2_digest', 'p3_digest'} else None, 'digest_prompt_sha256': hashlib.sha256(digest_prompt_text.encode()).hexdigest() if digest_prompt_text else None, 'local_wm': {'base_model': str(args.local_base_model.resolve()), 'base_model_fingerprint': model_fingerprint(args.local_base_model), 'adapter': str(args.local_adapter.resolve()) if args.local_adapter else None, 'adapter_fingerprint': local_adapter_fingerprint(args.local_adapter) if args.local_adapter else None, 'base_only': args.local_base_only, 'thinking': False, 'temperature': 0, 'batch_size': args.local_batch_size, 'max_length': args.local_max_length, 'max_new_tokens': args.local_max_new_tokens, 'prompt_key': args.local_wm_prompt_key, 'prompt': str(local_wm_prompt_path.resolve()), 'prompt_sha256': hashlib.sha256(local_wm_prompt_text.encode()).hexdigest(), 'input_contract': 'current_screenshot+candidate_action' if args.local_wm_prompt_key in {'localwm_next', 'localwm_delta'} else 'current_screenshot+current_md+candidate_action', 'prompt_contract': 'p3_digest' if args.p3_digest_contract else 'p2.4_digest' if args.p2_policy_version == 'p2.4' else 'p2_digest' if args.p2_digest_contract else 'v3_compact' if args.v3_local_wm_contract else 'legacy', 'output_state_mode': args.local_wm_output_state_mode} if args.agent_strategy in {'local_wm', 'p2_digest', 'p3_digest'} else None, 'p2_digest': {'policy_version': args.p2_policy_version, 'prompt': str(args.digest_prompt.resolve()), 'prompt_sha256': hashlib.sha256(digest_prompt_text.encode()).hexdigest(), 'input_contract': 'goal_text+current_screenshot+single_candidate_action+single_candidate_intent+predicted_next_md', 'output_contract': 'predicted_change+task_progress+evidence_refs' if args.p2_policy_version == 'p2.4' else 'predicted_change+intent_alignment+task_progress+evidence_refs', 'dispatch': 'parallel_per_candidate_restore_zero_based_order', 'best_input_contract': 'current_state_without_md+candidate_actions+candidate_intents+candidate_actions_conf+candidate_digests' if args.p2_policy_version == 'p2.4' else 'current_state_without_md+candidate_actions+candidate_intents+candidate_digests', 'single_candidate_policy': 'p1_direct_no_wm_digest_best', 'digest_input_sanitization': {'scope': 'per_line', 'max_line_chars': P2_DIGEST_MAX_PREDICTED_MD_LINE_CHARS, 'replacement': 'audited_truncation_marker'}, 'failure_policy': 'failed_candidate_becomes_unclear_digest;no_raw_next_md_fallback'} if args.agent_strategy == 'p2_digest' else None, 'p3_digest': {'prompt': str(args.digest_prompt.resolve()), 'prompt_sha256': hashlib.sha256(digest_prompt_text.encode()).hexdigest(), 'input_contract': 'goal_text+current_screenshot+single_candidate_action+single_candidate_intent+predicted_next_md', 'output_contract': 'predicted_change+intent_alignment+task_progress+evidence_refs', 'dispatch': 'parallel_per_candidate_restore_zero_based_order', 'best_input_contract': 'current_state_without_md+candidate_actions+candidate_intents+candidate_digests+optional_candidate_predicted_next_mds', 'routing': 'include_complete_predicted_next_mds_iff_no_digest_task_progress_positive', 'single_candidate_policy': 'p1_direct_no_wm_digest_best', 'digest_input_sanitization': {'scope': 'per_line', 'max_line_chars': P2_DIGEST_MAX_PREDICTED_MD_LINE_CHARS, 'replacement': 'audited_truncation_marker'}, 'failure_policy': 'failed_candidate_becomes_unclear_digest;sanitized_full_md_routing_unchanged'} if args.agent_strategy == 'p3_digest' else None, 'world_model': {'kind': args.world_model_kind, 'input_contract': args.wma_input_contract if args.world_model_kind == 'wma' else None, 'input_fields': ['current_observation', 'current_action'] if args.world_model_kind == 'wma' and args.wma_input_contract == 'axtree_action_only' else None, 'endpoint': manifest_world_model_endpoint(args.world_model_endpoint, args.num_shards), 'endpoint_policy': 'single' if args.num_shards == 1 else 'per_shard_local_service', 'name': args.world_model_name, 'path': str(args.world_model_path.resolve()) if args.world_model_path else None, 'model_fingerprint': model_fingerprint(args.world_model_path) if args.world_model_path else None, **({'base_model_fingerprint': model_fingerprint(args.world_model_path), 'adapter': str(args.world_model_adapter_path.resolve()) if args.world_model_adapter_path else None, 'adapter_fingerprint': local_adapter_fingerprint(args.world_model_adapter_path) if args.world_model_adapter_path else None} if args.world_model_kind in {'wma', 'webworld32b'} else {}), 'prompt': str(args.world_model_prompt.resolve()) if args.world_model_prompt else None, 'prompt_sha256': hashlib.sha256(world_model_prompt_text.encode()).hexdigest() if world_model_prompt_text else None, 'gui_agent_prompt_contract': 'v3_compact' if args.v3_other_wm_contract else 'legacy', 'policy_prompt_version': args.comparison_prompt_version, 'context_tokens': args.world_model_context_tokens, 'max_tokens': args.world_model_max_tokens if args.world_model_max_tokens > 0 else None, 'output_policy': 'fixed_cap' if args.world_model_max_tokens > 0 else 'context_remaining_no_independent_cap', 'concurrency': args.world_model_concurrency, 'temperature': args.world_model_temperature, **({'top_p': args.world_model_top_p, 'thinking_policy': args.world_model_thinking_policy, 'thinking': args.world_model_thinking_policy == 'enabled'} if args.world_model_kind in {'wma', 'webworld32b'} else {'thinking': False}), 'input_policy': 'full_current_state_no_truncation', 'context_overflow_policy': 'record_model_failure', 'state_source': 'screenshot' if args.world_model_kind == 'dreamer' else 'viewport_a11y_dom' if args.world_model_kind != 'none' else None} if args.agent_strategy == 'wm_compare' else None, 'official_schema': str(args.schema.resolve()), 'official_schema_sha256': sha256_file(args.schema), 'official_eval_commit': args.official_eval_commit, 'browser': f"playwright/{importlib.metadata.version('playwright')}", 'parameters': {'viewport': [args.viewport_width, args.viewport_height], 'agent_context_tokens': args.agent_context_tokens if args.agent_context_tokens > 0 else None, 'max_steps_including_entry_and_terminal': args.max_steps, 'temperature': args.temperature, 'reasoning_effort': args.reasoning_effort if args.enable_thinking and args.agent_strategy in {'vanilla', 'local_wm', 'p2_digest', 'p3_digest', 'wm_compare'} else 'native_cot' if args.agent_strategy == 'fara' else 'off', 'enable_thinking': args.enable_thinking, 'thinking_token_budget': args.thinking_token_budget, 'max_completion_tokens': args.max_completion_tokens, 'no_progress_limit': args.no_progress_limit, 'model_retries': args.model_retries, 'navigation_timeout_ms': args.navigation_timeout_ms, 'navigation_retries': args.navigation_retries, 'entry_wait_until': args.entry_wait_until, 'entry_settle_ms': args.entry_settle_ms, 'action_wait_ms': args.action_wait_ms, 'browser_executable_path': str(args.browser_executable_path.resolve()) if args.browser_executable_path else None, 'browser_user_agent': args.browser_user_agent, 'browser_disable_http2': args.browser_disable_http2, 'browser_disable_crashpad': True, 'num_shards': args.num_shards, 'concurrency_per_shard': args.concurrency, 'task_scheduling': args.task_scheduling, 'shared_queue_poll_seconds': args.shared_queue_poll_seconds if args.task_scheduling == 'shared' else None, 'task_claim_stale_seconds': args.task_claim_stale_seconds if args.task_scheduling == 'shared' else None, 'execution_layout': args.execution_layout, 'compact_vanilla_input': args.compact_vanilla_input, 'gui_agent_use_current_md': args.gui_agent_use_current_md, 'local_wm_output_state_mode': args.local_wm_output_state_mode, 'local_wm_replica_count': len(args.local_wm_devices), 'local_wm_replica_devices': list(args.local_wm_devices)}}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        reconciled = reconcile_resume_manifest(existing, manifest)
        if reconciled != existing:
            atomic_json(manifest_path, reconciled)
    else:
        atomic_json(manifest_path, manifest)
    config = RunnerConfig(endpoint=args.endpoint, model=args.model, prompt=prompt_text, mode=args.mode, markdown_endpoint=args.markdown_endpoint, width=args.viewport_width, height=args.viewport_height, max_steps=args.max_steps, navigation_timeout_ms=args.navigation_timeout_ms, action_wait_ms=args.action_wait_ms, model_timeout_seconds=args.model_timeout_seconds, model_retries=args.model_retries, reasoning_effort=args.reasoning_effort, thinking_token_budget=args.thinking_token_budget, max_completion_tokens=args.max_completion_tokens, temperature=args.temperature, enable_thinking=args.enable_thinking, agent_context_tokens=args.agent_context_tokens, no_progress_limit=args.no_progress_limit, navigation_retries=args.navigation_retries, browser_user_agent=args.browser_user_agent, entry_wait_until=args.entry_wait_until, entry_settle_ms=args.entry_settle_ms, compact_vanilla_input=args.compact_vanilla_input, gui_agent_use_current_md=args.gui_agent_use_current_md)
    if args.agent_strategy == 'local_wm':
        if args.concurrency > 3:
            raise ValueError('local_wm strategy supports at most 3 concurrent browser tasks per process')
        agent: OnlineMind2WebAgent = LocalWMAgent(config, candidate_prompt=prompt_text, best_prompt=best_prompt_text, local_base_model=args.local_base_model, local_adapter=args.local_adapter, local_base_only=args.local_base_only, local_batch_size=args.local_batch_size, local_max_length=args.local_max_length, local_max_new_tokens=args.local_max_new_tokens, local_wm_prompt_key=args.local_wm_prompt_key, local_wm_output_state_mode=args.local_wm_output_state_mode, local_wm_devices=args.local_wm_devices, v3_contract=args.v3_local_wm_contract)
    elif args.agent_strategy in {'p2_digest', 'p3_digest'}:
        if args.concurrency > 3:
            raise ValueError(f'{args.agent_strategy} strategy supports at most 3 concurrent browser tasks per process')
        digest_agent_class = P3DigestAgent if args.agent_strategy == 'p3_digest' else P2DigestAgent
        digest_agent_kwargs = dict(candidate_prompt=prompt_text, digest_prompt=digest_prompt_text, best_prompt=best_prompt_text, local_base_model=args.local_base_model, local_adapter=args.local_adapter, local_base_only=args.local_base_only, local_batch_size=args.local_batch_size, local_max_length=args.local_max_length, local_max_new_tokens=args.local_max_new_tokens, local_wm_prompt_key=args.local_wm_prompt_key, local_wm_output_state_mode=args.local_wm_output_state_mode, local_wm_devices=args.local_wm_devices)
        if args.agent_strategy == 'p2_digest':
            digest_agent_kwargs['policy_version'] = args.p2_policy_version
        agent = digest_agent_class(config, **digest_agent_kwargs)
    elif args.agent_strategy == 'wm_compare':
        agent = WMComparisonAgent(config, candidate_prompt=prompt_text, best_prompt=best_prompt_text, world_model_kind=args.world_model_kind, world_model_endpoint=args.world_model_endpoint, world_model_name=args.world_model_name, world_model_prompt=world_model_prompt_text, world_model_path=args.world_model_path, world_model_adapter_path=args.world_model_adapter_path, world_model_context_tokens=args.world_model_context_tokens, world_model_max_tokens=args.world_model_max_tokens, world_model_concurrency=args.world_model_concurrency, world_model_temperature=args.world_model_temperature, world_model_top_p=args.world_model_top_p, world_model_thinking_policy=args.world_model_thinking_policy, wma_input_contract=args.wma_input_contract, v3_contract=args.v3_other_wm_contract)
    elif args.agent_strategy in {'fara', 'gui_owl'}:
        agent = NativeComputerUseAgent(config, variant=args.agent_strategy)
    else:
        agent = VanillaAgent(config)
    semaphore = asyncio.Semaphore(args.concurrency)
    summaries: list[dict[str, Any]] = []
    queue_root = args.output_dir.parent / f'.{args.output_dir.name}.task_queue' if args.task_scheduling == 'shared' else None
    if isinstance(agent, LocalWMAgent):
        await agent.preload()
    async with async_playwright() as playwright:
        browser_args = ['--disable-dev-shm-usage', '--disable-crashpad-for-testing']
        if args.browser_disable_http2:
            browser_args.append('--disable-http2')
        browser = await playwright.chromium.launch(headless=True, args=browser_args, executable_path=str(args.browser_executable_path) if args.browser_executable_path else None)
        try:

            async def execute_task(task: dict[str, Any]) -> dict[str, Any]:
                task_id = str(task['task_id'])
                task_dir = args.output_dir / task_id
                archived_attempts = completed_wm_model_error_attempts(task_dir)
                retries_completed = len(archived_attempts)
                if archived_attempts and retries_completed > max(0, args.wm_technical_task_retries):
                    attempt = finalize_exhausted_wm_model_error(task_dir, schema_path=args.schema)
                    return {'task_id': task_id, 'status': 'model_error', 'audit_attempt': attempt, 'technical_retry_exhausted': True}
                while True:
                    summary = await run_task(task, browser, agent, config, args.output_dir, args.schema, args.resume)
                    if not should_retry_wm_technical_failure(summary, retries_completed=retries_completed, max_retries=args.wm_technical_task_retries):
                        if summary.get('status') == 'model_error':
                            attempt = finalize_exhausted_wm_model_error(task_dir, schema_path=args.schema)
                            summary = {**summary, 'audit_attempt': attempt, 'technical_retry_exhausted': True}
                        return summary
                    retries_completed += 1
                    print(json.dumps({'task_id': task_id, 'status': 'technical_retry_scheduled', 'completed_attempts': retries_completed, 'max_retries': args.wm_technical_task_retries}, ensure_ascii=False), flush=True)
                    await asyncio.sleep(min(2 ** retries_completed, 8))

            def record_summary(summary: dict[str, Any]) -> None:
                summaries.append(summary)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            if args.task_scheduling == 'static':

                async def bounded(task: dict[str, Any]) -> None:
                    async with semaphore:
                        record_summary(await execute_task(task))
                await asyncio.gather(*(bounded(task) for task in tasks))
            else:
                assert queue_root is not None
                ordered_tasks = shared_task_order(tasks, args.shard_index, args.num_shards)

                async def shared_worker(worker_index: int) -> None:
                    while True:
                        claim: TaskClaim | None = None
                        claimed_task: dict[str, Any] | None = None
                        for task in ordered_tasks:
                            if not task_needs_run(task, output_root=args.output_dir, schema_path=args.schema, resume=args.resume, agent_context_tokens=config.agent_context_tokens):
                                continue
                            candidate = try_acquire_task_claim(queue_root, str(task['task_id']), shard_index=args.shard_index, stale_seconds=args.task_claim_stale_seconds)
                            if candidate is not None:
                                claim = candidate
                                claimed_task = task
                                break
                        if claim is None or claimed_task is None:
                            if not any((task_needs_run(task, output_root=args.output_dir, schema_path=args.schema, resume=args.resume, agent_context_tokens=config.agent_context_tokens) for task in ordered_tasks)):
                                return
                            await asyncio.sleep(args.shared_queue_poll_seconds)
                            continue
                        status = 'runner_exception'
                        try:
                            summary = await execute_task(claimed_task)
                            status = str(summary.get('status') or 'unknown')
                            record_summary({**summary, 'scheduler_shard': args.shard_index, 'scheduler_worker': worker_index, 'scheduler_claim_token': claim.token})
                        finally:
                            release_task_claim(queue_root, claim, status=status)
                await asyncio.gather(*(shared_worker(index) for index in range(args.concurrency)))
        finally:
            try:
                await browser.close()
            finally:
                await agent.close()
    atomic_json(args.output_dir / f'shard_{args.shard_index:02d}_summary.json', summaries)
    failures = sum((item['status'] == 'runner_exception' for item in summaries))
    return 1 if failures else 0

def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=root / 'eval/online_mind2web/tasks.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--endpoint', required=True, help='OpenAI-compatible endpoint without /v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-path', type=Path, default=root / 'models/Qwen3.5-35B-A3B')
    parser.add_argument('--agent-context-tokens', type=int, default=0, help='Serving-side GUI Agent context window recorded in the run manifest.')
    parser.add_argument('--agent-strategy', choices=('vanilla', 'local_wm', 'p2_digest', 'p3_digest', 'wm_compare', 'fara', 'gui_owl'), default='vanilla')
    parser.add_argument('--prompt', type=Path, default=root / 'mkd2world/prompts/templatesV3/vanilla/gui_agent.md')
    parser.add_argument('--best-prompt', type=Path, default=root / 'mkd2world/prompts/p2.1/best.md')
    parser.add_argument('--digest-prompt', type=Path, default=None, help='P2/P3 single-candidate Digest prompt.')
    parser.add_argument('--digest-contract-prompt-dir', type=Path, default=None, help='Canonical candidate/digest/best prompt directory used only for the immutable Digest contract hash gate.')
    parser.add_argument('--local-base-model', type=Path, default=root / 'models/Qwen3.5-9B')
    parser.add_argument('--local-adapter', type=Path, default=None)
    parser.add_argument('--local-base-only', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--local-batch-size', type=int, default=4)
    parser.add_argument('--local-wm-devices', type=parse_local_wm_devices, default=('auto',), help='Comma-separated logical CUDA devices for independent Local-WM replicas, for example cuda:0,cuda:1. Each replica serves one request at a time through a shared async pool.')
    parser.add_argument('--local-max-length', type=int, default=60000)
    parser.add_argument('--local-max-new-tokens', type=int, default=WORLD_MODEL_MAX_OUTPUT_TOKENS, help='Project-wide Local-WM generation ceiling; fixed to 6000 for evaluation runs.')
    parser.add_argument('--gui-agent-use-current-md', action=argparse.BooleanOptionalAction, default=True, help='Include the rendered current Markdown in GUI Agent candidate and best-action payloads. Disable only with matching gui_agent_omd prompts.')
    parser.add_argument('--local-wm-prompt-key', choices=('world_local', 'localwm_next', 'localwm_delta'), default='world_local', help='Local WM input contract: world_local consumes current_md; localwm_next/localwm_delta consume only the current screenshot and action.')
    parser.add_argument('--local-wm-output-state-mode', choices=('raw_delta', 'replayed_next'), default='replayed_next', help='Representation passed from a delta Local WM to the GUI selector: the strict raw Delta or the strictly replayed full next Markdown.')
    parser.add_argument('--world-model-kind', choices=('none',), default='none')
    parser.add_argument('--world-model-endpoint', default=None)
    parser.add_argument('--world-model-name', default=None)
    parser.add_argument('--world-model-path', type=Path, default=None)
    parser.add_argument('--world-model-adapter-path', type=Path, default=None)
    parser.add_argument('--world-model-prompt', type=Path, default=None)
    parser.add_argument('--wma-input-contract', choices=('official_contextual', 'axtree_action_only'), default='official_contextual')
    parser.add_argument('--comparison-prompt-version', choices=('P2.1', 'P2.3'), default=None, help='Versioned GUI Policy/Prompt contract for comparison World Models.')
    parser.add_argument('--world-model-context-tokens', type=int, default=0)
    parser.add_argument('--world-model-max-tokens', type=int, default=0)
    parser.add_argument('--world-model-concurrency', type=int, default=4)
    parser.add_argument('--world-model-temperature', type=float, default=None)
    parser.add_argument('--world-model-top-p', type=float, default=None)
    parser.add_argument('--world-model-thinking-policy', choices=('disabled', 'enabled'), default=None)
    parser.add_argument('--mode', choices=('screenshot', 'screenshot_md'), default='screenshot')
    parser.add_argument('--markdown-endpoint', default=None)
    parser.add_argument('--schema', type=Path, default=root / 'eval/online_mind2web/submission_schema.json')
    parser.add_argument('--official-eval-commit', default='f0d805ee0e9e0b3ea70911e45e5264b72968f3dc')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--task-scheduling', choices=('static', 'shared'), default='static', help='static keeps fixed task shards; shared lets identical lanes claim unfinished tasks atomically and take over work from an idle lane')
    parser.add_argument('--shared-queue-poll-seconds', type=float, default=2.0, help='Polling interval while all unfinished shared tasks are claimed')
    parser.add_argument('--task-claim-stale-seconds', type=float, default=900.0, help='Cross-host/no-owner age before a shared claim is reclaimed; dead same-host runner claims are recovered after a short grace')
    parser.add_argument('--execution-layout', default='unspecified')
    parser.add_argument('--task-limit', type=int, default=0)
    parser.add_argument('--task-offset', type=int, default=0)
    parser.add_argument('--task-manifest', type=Path, default=None)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--no-progress-limit', type=int, default=4)
    parser.add_argument('--viewport-width', type=int, default=1280)
    parser.add_argument('--viewport-height', type=int, default=720)
    parser.add_argument('--navigation-timeout-ms', type=int, default=45000)
    parser.add_argument('--navigation-retries', type=int, default=0)
    parser.add_argument('--entry-wait-until', choices=('commit', 'domcontentloaded'), default='domcontentloaded')
    parser.add_argument('--entry-settle-ms', type=int, default=0)
    parser.add_argument('--browser-executable-path', type=Path, default=None)
    parser.add_argument('--browser-user-agent', default=None)
    parser.add_argument('--browser-disable-http2', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--action-wait-ms', type=int, default=1500)
    parser.add_argument('--model-timeout-seconds', type=int, default=180)
    parser.add_argument('--model-retries', type=int, default=2)
    parser.add_argument('--reasoning-effort', choices=('minimal', 'low', 'medium', 'high'), default='medium')
    parser.add_argument('--thinking-token-budget', type=int, default=4096)
    parser.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-completion-tokens', type=int, default=5000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--compact-vanilla-input', action=argparse.BooleanOptionalAction, default=None, help='For the vanilla strategy, omit the always-empty subgoal_history and current_subgoal_text fields. Enabled by default; pass --no-compact-vanilla-input only for legacy reproduction.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wm-technical-task-retries', type=int, default=2, help='Fresh-browser retries for audited wm_compare model_error tasks. Context overflow and completed benchmark failures are never retried.')
    return parser

def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.agent_strategy in {'p2_digest', 'p3_digest'}:
        policy_dir = root / ('mkd2world/prompts/p2.1' if args.agent_strategy == 'p2_digest' else 'mkd2world/prompts/p3')
        if args.prompt.name in {'gui_agent_vanilla.md', 'gui_agent.md'}:
            args.prompt = policy_dir / 'candidate.md'
        if args.best_prompt.name == 'gui_agent_action_best.md':
            args.best_prompt = policy_dir / 'best.md'
        if args.digest_prompt is None:
            args.digest_prompt = policy_dir / 'digest.md'
        args.gui_agent_use_current_md = False
        if args.local_wm_prompt_key == 'world_local':
            args.local_wm_prompt_key = 'localwm_next'
    if args.agent_strategy == 'wm_compare':
        if args.prompt.name == 'gui_agent_vanilla.md':
            args.prompt = root / 'mkd2world/prompts/p2.1/otherwm/gui_agent_candidate.md'
        if args.best_prompt.name == 'gui_agent_action_best.md':
            args.best_prompt = root / 'mkd2world/prompts/p2.1/otherwm/gui_agent_best.md'
        if args.world_model_kind != 'none':
            from mkd2world.online_mind2web_wm import profile_defaults
            prompt_names = {'agentworld': 'world_agentworld_web.md', 'dreamer': 'world_dreamer.md', 'webworld': 'world_webworld.md', 'webworld32b': 'world_webworld.md', 'wma': 'world_wma.md'}
            if args.world_model_kind == 'wma' and args.wma_input_contract == 'axtree_action_only':
                prompt_names['wma'] = 'world_wma_v2.md'
            if args.world_model_prompt is None:
                args.world_model_prompt = root / 'mkd2world/prompts/p2.1/otherwm' / prompt_names[args.world_model_kind]
            (default_context, _) = profile_defaults(args.world_model_kind)
            if args.world_model_context_tokens == 0:
                args.world_model_context_tokens = default_context
            if args.world_model_max_tokens == 0:
                args.world_model_max_tokens = WORLD_MODEL_MAX_OUTPUT_TOKENS
            profile_sampling = {'wma': (1.0, 0.95, 'disabled'), 'webworld32b': (0.0, 1.0, 'enabled')}
            (default_temperature, default_top_p, default_thinking) = profile_sampling.get(args.world_model_kind, (0.0, 1.0, 'disabled'))
            if args.world_model_temperature is None:
                args.world_model_temperature = default_temperature
            if args.world_model_top_p is None:
                args.world_model_top_p = default_top_p
            if args.world_model_thinking_policy is None:
                args.world_model_thinking_policy = default_thinking
    if args.world_model_temperature is None:
        args.world_model_temperature = 0.0
    if args.world_model_top_p is None:
        args.world_model_top_p = 1.0
    if args.world_model_thinking_policy is None:
        args.world_model_thinking_policy = 'disabled'
    apply_prompt_contract_defaults(args)
    enforce_world_model_output_limit(args)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('shard-index must be in [0, num-shards)')
    if args.task_scheduling == 'shared' and (not args.resume):
        raise ValueError('shared task scheduling requires --resume')
    if args.shared_queue_poll_seconds < 0.1:
        raise ValueError('shared-queue-poll-seconds must be at least 0.1')
    if args.task_claim_stale_seconds < 60:
        raise ValueError('task-claim-stale-seconds must be at least 60')
    if args.max_steps < 3 or args.max_steps > 50:
        raise ValueError('max-steps must be in [3, 50]')
    if args.temperature < 0:
        raise ValueError('temperature must be non-negative')
    if args.thinking_token_budget <= 0 or args.thinking_token_budget >= args.max_completion_tokens:
        raise ValueError('thinking-token-budget must be positive and lower than max-completion-tokens')
    if args.no_progress_limit < 2:
        raise ValueError('no-progress-limit must be at least 2')
    if args.navigation_retries < 0 or args.navigation_retries > 5:
        raise ValueError('navigation-retries must be in [0, 5]')
    if args.entry_settle_ms < 0 or args.entry_settle_ms > 30000:
        raise ValueError('entry-settle-ms must be in [0, 30000]')
    if args.browser_executable_path is not None and (not args.browser_executable_path.is_file()):
        raise ValueError(f'browser executable does not exist: {args.browser_executable_path}')
    if args.agent_strategy in {'local_wm', 'p2_digest', 'p3_digest'}:
        needs_internal_md = args.local_wm_prompt_key == 'localwm_delta'
        required_mode = 'screenshot_md' if args.gui_agent_use_current_md or needs_internal_md else 'screenshot'
        if args.mode != required_mode:
            raise ValueError(f'{args.agent_strategy} input contract requires --mode {required_mode}')
        if required_mode == 'screenshot_md' and (not args.markdown_endpoint):
            raise ValueError(f'{args.agent_strategy} observation pipeline requires --markdown-endpoint')
        if required_mode == 'screenshot' and args.markdown_endpoint:
            raise ValueError('this local_wm observation pipeline forbids --markdown-endpoint')
        if args.local_base_only and args.local_adapter is not None:
            raise ValueError('--local-base-only and --local-adapter are mutually exclusive')
        if not args.local_base_only and args.local_adapter is None:
            raise ValueError(f'{args.agent_strategy} requires --local-adapter unless --local-base-only is set')
        if args.local_batch_size < 1:
            raise ValueError('local-batch-size must be at least 1')
    if args.agent_strategy == 'wm_compare':
        if args.mode != 'screenshot':
            raise ValueError('wm_compare requires screenshot mode and forbids project Markdown')
        if args.world_model_concurrency < 1:
            raise ValueError('world-model-concurrency must be at least 1')
        if args.world_model_temperature < 0:
            raise ValueError('world-model-temperature must be non-negative')
        if not 0 < args.world_model_top_p <= 1:
            raise ValueError('world-model-top-p must be in (0, 1]')
        if args.world_model_kind != 'wma' and args.wma_input_contract != 'official_contextual':
            raise ValueError('--wma-input-contract axtree_action_only requires --world-model-kind wma')
        if args.world_model_kind == 'none':
            if args.world_model_endpoint or args.world_model_name or args.world_model_path:
                raise ValueError('world-model kind none must not configure a model endpoint or path')
        else:
            if not args.world_model_endpoint or not args.world_model_name:
                raise ValueError('wm_compare requires --world-model-endpoint and --world-model-name')
            if args.world_model_path is None or not (args.world_model_path / 'config.json').is_file():
                raise ValueError('wm_compare requires a complete --world-model-path')
            if args.world_model_kind == 'wma':
                if args.world_model_adapter_path is None:
                    raise ValueError('WMA requires --world-model-adapter-path')
                for filename in ('adapter_config.json', 'adapter_model.safetensors'):
                    if not (args.world_model_adapter_path / filename).is_file():
                        raise ValueError(f'WMA adapter is incomplete: {args.world_model_adapter_path / filename}')
            elif args.world_model_adapter_path is not None:
                raise ValueError('only WMA accepts --world-model-adapter-path')
            if args.world_model_prompt is None or not args.world_model_prompt.is_file():
                raise ValueError('wm_compare world-model prompt is missing')
            if args.world_model_max_tokens > 0 and args.world_model_context_tokens <= args.world_model_max_tokens + 2048:
                raise ValueError('world-model context is too small for configured output')
    if args.compact_vanilla_input and args.agent_strategy != 'vanilla':
        raise ValueError('--compact-vanilla-input is supported only by the vanilla strategy')
    if args.mode == 'screenshot_md' and args.prompt.name == 'gui_agent_vanilla.md':
        raise ValueError('screenshot_md mode requires the GUIAgent_Vanilla+md prompt')
    return asyncio.run(async_main(args))
if __name__ == '__main__':
    raise SystemExit(main())
