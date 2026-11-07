from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
try:
    from mkd2world.training.qwen35_35b_current_tsm import CHECKPOINT_STEPS, ContractError, FORMAL_EVALUATION_FILES, atomic_write_json, build_exposure_plan, canonical_json_sha256, load_json, recursive_file_receipt, resolve_project_path, sha256_file, validate_dcp_tree, validate_manifest_contract, validate_transformers_5121_token_scaling_source
except ModuleNotFoundError:
    _trainer_path = Path(__file__).with_name('qwen35_35b_current_tsm.py')
    _trainer_spec = importlib.util.spec_from_file_location('qwen35_35b_current_tsm', _trainer_path)
    if _trainer_spec is None or _trainer_spec.loader is None:
        raise
    _trainer = importlib.util.module_from_spec(_trainer_spec)
    sys.modules[_trainer_spec.name] = _trainer
    _trainer_spec.loader.exec_module(_trainer)
    CHECKPOINT_STEPS = _trainer.CHECKPOINT_STEPS
    ContractError = _trainer.ContractError
    FORMAL_EVALUATION_FILES = _trainer.FORMAL_EVALUATION_FILES
    atomic_write_json = _trainer.atomic_write_json
    build_exposure_plan = _trainer.build_exposure_plan
    canonical_json_sha256 = _trainer.canonical_json_sha256
    load_json = _trainer.load_json
    recursive_file_receipt = _trainer.recursive_file_receipt
    resolve_project_path = _trainer.resolve_project_path
    sha256_file = _trainer.sha256_file
    validate_dcp_tree = _trainer.validate_dcp_tree
    validate_manifest_contract = _trainer.validate_manifest_contract
    validate_transformers_5121_token_scaling_source = _trainer.validate_transformers_5121_token_scaling_source
PUBLIC_ACTIONS = ('preflight', 'certify', 'train', 'launch', 'recover', 'status')
PERSISTENT_TMUX_PATH = Path('MKD2WORLD_EXTERNAL_ROOT/envs/bin/tmux')
TERMINAL_PHASES = {'COMPLETE', 'HALTED', 'FAILED'}
WATCHDOG_STOP_PHASES = TERMINAL_PHASES | {'EVAL_BLOCKED'}
ALLOWED_TRANSITIONS = {'NEW': {'PREFLIGHT_PASSED', 'FAILED'}, 'PREFLIGHT_PASSED': {'CERTIFY_STEP1_RUNNING', 'CERTIFIED', 'FAILED'}, 'CERTIFY_STEP1_RUNNING': {'CERTIFY_STEP1_COMMITTED', 'FAILED'}, 'CERTIFY_STEP1_COMMITTED': {'CERTIFY_RESUME_RUNNING', 'FAILED'}, 'CERTIFY_RESUME_RUNNING': {'CERTIFY_STEP2_COMMITTED', 'FAILED'}, 'CERTIFY_STEP2_COMMITTED': {'CERTIFIED', 'FAILED'}, 'CERTIFIED': {'BASE_EVAL_RUNNING', 'FAILED'}, 'BASE_EVAL_RUNNING': {'BASE_EVAL_PASSED', 'EVAL_BLOCKED', 'HALTED', 'FAILED'}, 'BASE_EVAL_PASSED': {'FORMAL_SEGMENT_RUNNING', 'FAILED'}, 'RESUME_READY': {'FORMAL_SEGMENT_RUNNING', 'COMPLETE', 'FAILED'}, 'FORMAL_SEGMENT_RUNNING': {'CHECKPOINT_COMMITTED', 'FAILED'}, 'CHECKPOINT_COMMITTED': {'TRAIN_EXITED', 'FAILED'}, 'TRAIN_EXITED': {'POSTHOC_EVAL_RUNNING', 'FAILED'}, 'POSTHOC_EVAL_RUNNING': {'EVAL_PASSED', 'EVAL_BLOCKED', 'HALTED', 'FAILED'}, 'EVAL_BLOCKED': {'BASE_EVAL_RUNNING', 'POSTHOC_EVAL_RUNNING', 'RECOVERING', 'FAILED'}, 'EVAL_PASSED': {'RESUME_READY', 'COMPLETE', 'FAILED'}, 'RECOVERING': {'NEW', 'PREFLIGHT_PASSED', 'CERTIFIED', 'BASE_EVAL_PASSED', 'RESUME_READY', 'FAILED'}, 'HALTED': {'RECOVERING'}, 'FAILED': {'RECOVERING'}, 'COMPLETE': set()}

class BlockedError(ContractError):
    """Expected pause while an external validation gate is not configured."""

class GateFailedError(ContractError):
    """A configured blocking validation gate completed but did not pass."""

class LockBusyError(ContractError):
    pass

@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    staging_root: Path
    manifest_path: Path
    resource_path: Path
    output_root: Path
    log_root: Path
    prepared_jsonl: Path
    prepared_receipt: Path
    image_list: Path
    exposure_plan: Path
    preflight_receipt: Path
    certification_root: Path
    certification_receipt: Path
    state_path: Path
    ledger_path: Path
    lock_path: Path
    owner_path: Path
    runner_path: Path
    trainer_path: Path
    runtime_path: Path
    prepare_path: Path

def default_staging_root() -> Path:
    return Path(__file__).resolve().parents[3]

def default_manifest_path() -> Path:
    return default_staging_root() / 'configs/experiments/qwen35b_a3b_v3_cur_current50k_lr2e6_s834_20260901.json'

def runtime_paths(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], RuntimePaths]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    project_root = Path(manifest['identity']['project_root'])
    staging_root = default_staging_root()
    resource_from_project = resolve_project_path(project_root, manifest['resource_profile']['path'])
    resource_from_staging = staging_root / manifest['resource_profile']['path']
    resource_path = resource_from_project if resource_from_project.is_file() else resource_from_staging
    resource = load_json(resource_path)
    output_root = resolve_project_path(project_root, manifest['artifacts']['output_root'])
    log_root = resolve_project_path(project_root, manifest['artifacts']['log_root'])
    paths = RuntimePaths(project_root=project_root, staging_root=staging_root, manifest_path=manifest_path, resource_path=resource_path, output_root=output_root, log_root=log_root, prepared_jsonl=resolve_project_path(project_root, manifest['data']['current50k']['path']), prepared_receipt=resolve_project_path(project_root, manifest['data']['current50k']['receipt_path']), image_list=resolve_project_path(project_root, manifest['data']['current50k']['image_list_path']), exposure_plan=resolve_project_path(project_root, manifest['data']['exposure_plan']['path']), preflight_receipt=log_root / 'preflight_receipt.json', certification_root=output_root / 'certification', certification_receipt=log_root / 'certification_receipt.json', state_path=log_root / 'latest_state.json', ledger_path=log_root / 'transition_ledger.jsonl', lock_path=log_root / 'formal.lock', owner_path=log_root / 'owner.json', runner_path=staging_root / 'scripts/training/run_qwen35b_v3_cur_2gpu.sh', trainer_path=staging_root / 'mkd2world/training/qwen35_35b_current_tsm.py', runtime_path=Path(__file__).resolve(), prepare_path=staging_root / 'scripts/data/prepare_qwen35b_v3_cur_50k.py')
    return (manifest, resource, paths)

def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

def read_state(paths: RuntimePaths) -> dict[str, Any]:
    if not paths.state_path.is_file():
        return {'phase': 'NEW', 'revision': 0, 'step': 0}
    return load_json(paths.state_path)

def transition(paths: RuntimePaths, phase: str, **fields: Any) -> dict[str, Any]:
    current = read_state(paths)
    old_phase = str(current.get('phase', 'NEW'))
    if phase not in ALLOWED_TRANSITIONS.get(old_phase, set()):
        raise ContractError(f'illegal runtime transition: {old_phase} -> {phase}')
    value = {**current, **fields, 'phase': phase, 'revision': int(current.get('revision', 0)) + 1, 'updated_at_epoch': time.time()}
    append_jsonl(paths.ledger_path, {'from': old_phase, **value})
    atomic_write_json(paths.state_path, value)
    return value

@contextlib.contextmanager
def nonblocking_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockBusyError(f'another managed process owns {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

@contextlib.contextmanager
def bounded_lock(path: Path, *, wait_seconds: float=15.0) -> Iterator[None]:
    """Bridge the short launch-to-tmux race without hiding duplicate trainers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    with path.open('a+') as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise LockBusyError(f'another managed process owns {path}') from exc
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ContractError(f'missing {label}: {path}')

def parse_sha256sums(path: Path) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for (line_number, line) in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch('([0-9a-f]{64})\\s+\\*?(.+)', line.strip())
        if not match:
            raise ContractError(f'invalid SHA256SUMS line {line_number}')
        (digest, raw_name) = match.groups()
        relative = Path(raw_name)
        if relative.is_absolute() or '..' in relative.parts or raw_name in seen:
            raise ContractError(f'unsafe or duplicate model receipt path: {raw_name}')
        seen.add(raw_name)
        values.append((raw_name, digest))
    if not values:
        raise ContractError('model SHA256SUMS is empty')
    return values

def canonicalize_legacy_model_sha256sums(*, original_path: Path, original_sha256: str, canonical_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Convert a hash-bound legacy absolute-path list to safe basenames.

    Absolute paths are accepted only by this dedicated converter. The normal
    parser remains strict and will only consume the generated relative list.
    """
    verify_hash(original_path, original_sha256, 'original model SHA256SUMS')
    entries: list[dict[str, str]] = []
    seen_basenames: set[str] = set()
    for (line_number, line) in enumerate(original_path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch('([0-9a-f]{64})\\s+\\*?(.+)', line.strip())
        if not match:
            raise ContractError(f'invalid original SHA256SUMS line {line_number}')
        (digest, raw_name) = match.groups()
        source_path = Path(raw_name)
        if not source_path.is_absolute():
            raise ContractError('legacy canonicalization expects every original model path to be absolute')
        basename = source_path.name
        if not basename or basename in {'.', '..'} or Path(basename).name != basename:
            raise ContractError(f'unsafe model basename on original line {line_number}')
        if basename in seen_basenames:
            raise ContractError(f'duplicate model basename in original SHA256SUMS: {basename}')
        seen_basenames.add(basename)
        entries.append({'source_absolute_path': raw_name, 'basename': basename, 'sha256': digest})
    if not entries:
        raise ContractError('original model SHA256SUMS is empty')
    canonical_text = ''.join((f"{item['sha256']}  {item['basename']}\n" for item in entries))
    canonical_digest = hashlib.sha256(canonical_text.encode('utf-8')).hexdigest()
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    if canonical_path.is_file():
        if canonical_path.read_text(encoding='utf-8') != canonical_text:
            raise ContractError('existing canonical model SHA256SUMS differs from the bound original')
    else:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=canonical_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, canonical_path)
    parsed = parse_sha256sums(canonical_path)
    receipt = {'schema_version': 'guiagent-model-sha256sums-canonicalization-v1', 'status': 'passed', 'original_path': str(original_path), 'original_sha256': original_sha256, 'canonical_path': str(canonical_path), 'canonical_sha256': canonical_digest, 'entry_count': len(entries), 'unique_basename_count': len(seen_basenames), 'entries_sha256': canonical_json_sha256(entries), 'entries': entries}
    if receipt_path.is_file():
        existing = load_json(receipt_path)
        if existing != receipt:
            raise ContractError('existing model canonicalization receipt drifted')
    else:
        atomic_write_json(receipt_path, receipt)
    return receipt

def verify_model_directory(model_root: Path, sha256sums_path: Path) -> dict[str, Any]:
    entries = parse_sha256sums(sha256sums_path)
    files: list[dict[str, Any]] = []
    for (relative, expected) in entries:
        path = model_root / relative
        if not path.is_file() or path.is_symlink():
            raise ContractError(f'model file is missing or a symlink: {relative}')
        actual = sha256_file(path)
        if actual != expected:
            raise ContractError(f'model file hash mismatch: {relative}')
        files.append({'path': relative, 'sha256': actual, 'size_bytes': path.stat().st_size})
    return {'model_root': str(model_root), 'sha256sums_path': str(sha256sums_path), 'files': files, 'canonical_receipt_sha256': canonical_json_sha256(files)}

def persistent_tmux_receipt(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the project-local tmux executable and its two persistent libraries."""
    executable = expected.get('executable', {})
    libraries = expected.get('libraries', [])
    if not isinstance(executable, Mapping) or not isinstance(libraries, list):
        raise ContractError('persistent tmux software contract is malformed')
    declared = [executable, *libraries]
    if len(libraries) != 3 or any((not isinstance(item, Mapping) for item in declared)):
        raise ContractError('persistent tmux must declare one executable and exactly three libraries')
    files: list[dict[str, Any]] = []
    for item in declared:
        path = Path(str(item.get('path', '')))
        expected_sha256 = str(item.get('sha256', ''))
        if not path.is_absolute():
            raise ContractError('persistent tmux paths and SHA256 values must be absolute/frozen')
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise ContractError(f'persistent tmux dependency is missing: {path}') from exc
        if path.is_symlink() or not stat.S_ISREG(mode):
            raise ContractError(f'persistent tmux dependency is not a regular non-symlink: {path}')
        actual_sha256 = sha256_file(path)
        files.append({'path': str(path), 'sha256': actual_sha256, 'size_bytes': path.stat().st_size, 'executable': os.access(path, os.X_OK)})
    executable_path = Path(str(executable['path']))
    if not files[0]['executable']:
        raise ContractError(f'persistent tmux binary is not executable: {executable_path}')
    environment = os.environ.copy()
    library_directory = str(executable_path.parent)
    inherited = environment.get('LD_LIBRARY_PATH', '')
    environment['LD_LIBRARY_PATH'] = f'{library_directory}:{inherited}' if inherited else library_directory
    try:
        completed = subprocess.run([str(executable_path), '-V'], capture_output=True, check=True, env=environment, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError('cannot execute the frozen persistent tmux binary') from exc
    actual_version = completed.stdout.strip()
    expected_version = str(executable.get('version', ''))
    if actual_version != expected_version:
        raise ContractError(f'persistent tmux version {actual_version!r} != certified {expected_version!r}')
    return {'status': 'passed', 'version': actual_version, 'files': files}

def _persistent_toolchain_inventory(root: Path, *, label: str) -> dict[str, Any]:
    """Fingerprint a closed toolchain tree, including safe internal symlinks."""
    if not root.is_absolute():
        raise ContractError(f'persistent {label} toolchain root must be absolute')
    try:
        root_metadata = root.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f'persistent {label} toolchain is missing: {root}') from exc
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise ContractError(f'persistent {label} toolchain root must be a non-symlink directory')
    canonical_root = root.resolve(strict=True)
    entries: list[list[Any]] = []
    regular_files = 0
    symlinks = 0
    total_bytes = 0
    for path in sorted(root.rglob('*'), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raw_target = os.readlink(path)
            target = Path(raw_target)
            unresolved = target if target.is_absolute() else path.parent / target
            try:
                resolved = unresolved.resolve(strict=True)
                resolved.relative_to(canonical_root)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise ContractError(f'{label} toolchain symlink is dangling or escapes its root: {relative}') from exc
            entries.append([relative, 'symlink', raw_target])
            symlinks += 1
        elif stat.S_ISDIR(metadata.st_mode):
            entries.append([relative, 'dir'])
        elif stat.S_ISREG(metadata.st_mode):
            entries.append([relative, 'file', sha256_file(path), int(metadata.st_size), oct(metadata.st_mode & 511)])
            regular_files += 1
            total_bytes += int(metadata.st_size)
        else:
            raise ContractError(f'{label} toolchain contains a non-file entry: {relative}')
    return {'algorithm': "sha256(canonical_json(entries)); dir=[path,'dir']; file=[path,'file',sha256,size_bytes,oct(mode&0777)]; symlink=[path,'symlink',raw_target]", 'entry_count': len(entries), 'regular_files': regular_files, 'symlinks': symlinks, 'unsafe_symlinks': 0, 'dangling_symlinks': 0, 'total_regular_file_bytes': total_bytes, 'sha256': canonical_json_sha256(entries)}

def persistent_cpp_toolchain_inventory(root: Path) -> dict[str, Any]:
    return _persistent_toolchain_inventory(root, label='C++')

def persistent_cuda_toolchain_inventory(root: Path) -> dict[str, Any]:
    return _persistent_toolchain_inventory(root, label='CUDA')

def _run_contract_command(command: Sequence[str], *, env: Mapping[str, str]) -> str:
    try:
        completed = subprocess.run(list(command), capture_output=True, check=False, env=dict(env), text=True)
    except OSError as exc:
        raise ContractError(f'cannot execute software-contract command: {command[0]}') from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ContractError(f'software-contract command failed rc={completed.returncode}: {detail}')
    return completed.stdout.strip()

def _critical_toolchain_receipts(root: Path, expected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    canonical_root = root.resolve(strict=True)
    for item in expected:
        relative = Path(str(item.get('path', '')))
        if relative.is_absolute() or not relative.parts or '..' in relative.parts:
            raise ContractError('critical toolchain paths must be safe and relative')
        path = root / relative
        if 'raw_symlink_target' in item:
            if not path.is_symlink() or os.readlink(path) != item['raw_symlink_target']:
                raise ContractError(f'critical toolchain symlink drifted: {relative}')
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(canonical_root)
            except ValueError as exc:
                raise ContractError(f'critical toolchain symlink escapes root: {relative}') from exc
            expected_resolved = root / str(item['resolved_path'])
            if resolved != expected_resolved.resolve(strict=True):
                raise ContractError(f'critical toolchain symlink target drifted: {relative}')
            actual_sha256 = sha256_file(resolved)
            actual_size = resolved.stat().st_size
            if actual_size != int(item['resolved_size_bytes']):
                raise ContractError(f'critical toolchain resolved file drifted: {relative}')
            values.append({'path': relative.as_posix(), 'kind': 'symlink', 'raw_target': os.readlink(path), 'resolved_path': resolved.relative_to(canonical_root).as_posix(), 'resolved_sha256': actual_sha256, 'resolved_size_bytes': actual_size})
            continue
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise ContractError(f'critical toolchain file is missing: {relative}') from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f'critical toolchain file is not regular: {relative}')
        actual_sha256 = sha256_file(path)
        if metadata.st_size != int(item['size_bytes']) or os.access(path, os.X_OK) is not bool(item['executable']):
            raise ContractError(f'critical toolchain file drifted: {relative}')
        values.append({'path': relative.as_posix(), 'kind': 'file', 'sha256': actual_sha256, 'size_bytes': int(metadata.st_size), 'executable': os.access(path, os.X_OK)})
    return values

def jit_environment_receipt(expected: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    if expected.get('schema_version') != 'guiagent-persistent-jit-cache-v1':
        raise ContractError('persistent JIT cache schema drifted')
    root = Path(str(expected.get('root', '')))
    project_root = project_root.resolve(strict=True)
    if not root.is_absolute():
        raise ContractError('persistent JIT cache root must be absolute')
    try:
        root.resolve(strict=True).relative_to(project_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ContractError('persistent JIT cache must exist under the project root') from exc
    if root.is_symlink() or not root.is_dir():
        raise ContractError('persistent JIT cache root must be a non-symlink directory')
    directories = expected.get('directories', {})
    fixed = expected.get('fixed_environment', {})
    if not isinstance(directories, Mapping) or not isinstance(fixed, Mapping):
        raise ContractError('persistent JIT cache environment is malformed')
    receipts: list[dict[str, Any]] = []
    for (variable, value) in sorted(directories.items()):
        path = Path(str(value))
        if os.environ.get(str(variable)) != str(path):
            raise ContractError(f'JIT cache environment variable drifted: {variable}')
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
            metadata = path.lstat()
        except (FileNotFoundError, ValueError) as exc:
            raise ContractError(f'JIT cache directory is missing/unsafe: {path}') from exc
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ContractError(f'JIT cache directory is not regular: {path}')
        probe = path / f'.qwen35b_write_probe_{os.getpid()}_{uuid.uuid4().hex}'
        descriptor = -1
        try:
            descriptor = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 384)
            os.write(descriptor, b'persistent-jit-cache-write-probe\n')
            os.fsync(descriptor)
        except OSError as exc:
            raise ContractError(f'JIT cache directory is not writable: {path}') from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if probe.exists():
                probe.unlink()
        receipts.append({'environment_variable': str(variable), 'path': str(path)})
    for (variable, value) in sorted(fixed.items()):
        if os.environ.get(str(variable)) != str(value):
            raise ContractError(f'fixed JIT environment variable drifted: {variable}')
    return {'status': 'passed', 'root': str(root), 'directories': receipts, 'fixed_environment': dict(sorted(((str(k), str(v)) for (k, v) in fixed.items()))), 'writable': True}

def persistent_cpp_toolchain_receipt(expected: Mapping[str, Any], *, smoke_directory: Path) -> dict[str, Any]:
    if expected.get('schema_version') != 'guiagent-persistent-cpp-toolchain-v1':
        raise ContractError('persistent C++ toolchain schema drifted')
    root = Path(str(expected.get('root', '')))
    inventory = persistent_cpp_toolchain_inventory(root)
    if inventory != expected.get('inventory'):
        raise ContractError('persistent C++ toolchain inventory drifted')
    compiler_contract = expected.get('compiler', {})
    compiler = Path(str(compiler_contract.get('path', '')))
    if not compiler.is_symlink() or os.readlink(compiler) != compiler_contract.get('raw_symlink_target'):
        raise ContractError('persistent g++ symlink drifted')
    resolved_compiler = compiler.resolve(strict=True)
    if resolved_compiler != Path(str(compiler_contract['resolved_path'])).resolve(strict=True) or resolved_compiler.stat().st_size != int(compiler_contract['resolved_size_bytes']) or (not os.access(resolved_compiler, os.X_OK)):
        raise ContractError('persistent g++ driver drifted')
    c_compiler_contract = expected.get('c_compiler', {})
    c_compiler = Path(str(c_compiler_contract.get('path', '')))
    if not c_compiler.is_symlink() or os.readlink(c_compiler) != c_compiler_contract.get('raw_symlink_target'):
        raise ContractError('persistent gcc symlink drifted')
    resolved_c_compiler = c_compiler.resolve(strict=True)
    if resolved_c_compiler != Path(str(c_compiler_contract['resolved_path'])).resolve(strict=True) or resolved_c_compiler.stat().st_size != int(c_compiler_contract['resolved_size_bytes']) or (not os.access(resolved_c_compiler, os.X_OK)):
        raise ContractError('persistent gcc driver drifted')
    critical = _critical_toolchain_receipts(root, expected.get('critical_files', ()))
    environment_contract = expected.get('environment', {})
    for variable in ('CC', 'CXX', 'NVCC_CCBIN'):
        if os.environ.get(variable) != environment_contract.get(variable):
            raise ContractError(f'toolchain environment variable drifted: {variable}')
    for variable in ('COMPILER_PATH', 'CPLUS_INCLUDE_PATH'):
        prefix = [str(value) for value in environment_contract.get(f'{variable}_prefix', ())]
        actual = os.environ.get(variable, '').split(os.pathsep)
        if not prefix or actual[:len(prefix)] != prefix:
            raise ContractError(f'toolchain environment prefix drifted: {variable}')
    if any((str(root) in value for value in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep))):
        raise ContractError('persistent toolchain libraries must not enter LD_LIBRARY_PATH')
    environment = os.environ.copy()
    c_version_output = _run_contract_command([str(c_compiler), '--version'], env=environment)
    c_version_line = c_version_output.splitlines()[0] if c_version_output else ''
    c_dumpfullversion = _run_contract_command([str(c_compiler), '-dumpfullversion', '-dumpversion'], env=environment)
    c_dumpmachine = _run_contract_command([str(c_compiler), '-dumpmachine'], env=environment)
    if c_version_line != c_compiler_contract['version_line'] or c_dumpfullversion != c_compiler_contract['dumpfullversion'] or c_dumpmachine != c_compiler_contract['dumpmachine']:
        raise ContractError('persistent gcc version/target drifted')
    cc1 = Path(_run_contract_command([str(c_compiler), '-print-prog-name=cc1'], env=environment))
    if not cc1.is_absolute() or cc1.resolve(strict=True) != (root / 'usr/lib/gcc/x86_64-linux-gnu/11/cc1').resolve(strict=True):
        raise ContractError('gcc does not resolve the certified cc1')
    version_output = _run_contract_command([str(compiler), '--version'], env=environment)
    version_line = version_output.splitlines()[0] if version_output else ''
    dumpfullversion = _run_contract_command([str(compiler), '-dumpfullversion', '-dumpversion'], env=environment)
    dumpmachine = _run_contract_command([str(compiler), '-dumpmachine'], env=environment)
    if version_line != compiler_contract['version_line'] or dumpfullversion != compiler_contract['dumpfullversion'] or dumpmachine != compiler_contract['dumpmachine']:
        raise ContractError('persistent g++ version/target drifted')
    cc1plus = Path(_run_contract_command([str(compiler), '-print-prog-name=cc1plus'], env=environment))
    libstdcxx = Path(_run_contract_command([str(compiler), '-print-file-name=libstdc++.so'], env=environment))
    if not cc1plus.is_absolute() or cc1plus.resolve(strict=True) != (root / 'usr/lib/gcc/x86_64-linux-gnu/11/cc1plus').resolve(strict=True):
        raise ContractError('g++ does not resolve the certified cc1plus')
    if not libstdcxx.is_absolute() or libstdcxx.resolve(strict=True) != (root / 'usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30').resolve(strict=True):
        raise ContractError('g++ does not resolve the certified libstdc++')
    search_dirs = _run_contract_command([str(compiler), '-print-search-dirs'], env=environment)
    if str(root) not in search_dirs:
        raise ContractError('g++ search directories do not include the certified root')
    with tempfile.TemporaryDirectory(prefix='qwen35b_cpp_smoke_', dir=smoke_directory) as temporary:
        temporary_root = Path(temporary)
        c_source = temporary_root / 'smoke.c'
        c_binary = temporary_root / 'c_smoke'
        c_source.write_text('#include <stdio.h>\nint main(void){puts("gcc11-c-ok");return 0;}\n', encoding='utf-8')
        _run_contract_command([str(c_compiler), '-std=c11', '-O0', str(c_source), '-o', str(c_binary)], env=environment)
        c_stdout = _run_contract_command([str(c_binary)], env=environment)
        if c_stdout != 'gcc11-c-ok':
            raise ContractError('persistent gcc C11 compile/link/execute smoke failed')
        cxx_source = temporary_root / 'smoke.cpp'
        cxx_binary = temporary_root / 'smoke'
        cxx_source.write_text('#include <iostream>\n#include <optional>\nint main(){static_assert(__cplusplus>=201703L);std::optional<int> v=11;std::cout<<"gxx11-cxx17-ok";return *v==11?0:1;}\n', encoding='utf-8')
        _run_contract_command([str(compiler), '-std=c++17', '-O0', str(cxx_source), '-o', str(cxx_binary)], env=environment)
        cxx_stdout = _run_contract_command([str(cxx_binary)], env=environment)
        if cxx_stdout != 'gxx11-cxx17-ok':
            raise ContractError('persistent g++ C++17 compile/link/execute smoke failed')
    return {'status': 'passed', 'profile_id': expected['profile_id'], 'root': str(root), 'inventory': inventory, 'compiler': {'path': str(compiler), 'resolved_path': str(resolved_compiler), 'resolved_sha256': sha256_file(resolved_compiler), 'version_line': version_line, 'dumpfullversion': dumpfullversion, 'dumpmachine': dumpmachine, 'cc1plus': str(cc1plus.resolve(strict=True)), 'libstdcxx': str(libstdcxx.resolve(strict=True)), 'search_dirs_sha256': hashlib.sha256(search_dirs.encode('utf-8')).hexdigest()}, 'c_compiler': {'path': str(c_compiler), 'resolved_path': str(resolved_c_compiler), 'resolved_sha256': sha256_file(resolved_c_compiler), 'version_line': c_version_line, 'dumpfullversion': c_dumpfullversion, 'dumpmachine': c_dumpmachine, 'cc1': str(cc1.resolve(strict=True))}, 'critical_files': critical, 'c11_smoke': {'status': 'passed', 'stdout': 'gcc11-c-ok'}, 'cxx17_smoke': {'status': 'passed', 'stdout': 'gxx11-cxx17-ok'}}

def _frozen_regular_file(path: Path, *, expected_sha256: str, expected_size: int, expected_executable: bool, label: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise ContractError(f'{label} path must be absolute: {path}')
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f'{label} is missing: {path}') from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f'{label} is not a regular non-symlink: {path}')
    actual_sha256 = sha256_file(path)
    executable = os.access(path, os.X_OK)
    if int(metadata.st_size) != int(expected_size) or executable is not expected_executable:
        raise ContractError(f'{label} drifted: {path}')
    return {'path': str(path), 'sha256': actual_sha256, 'size_bytes': int(metadata.st_size), 'executable': executable}

def _distribution_name_version(metadata_path: Path) -> tuple[str, str]:
    name = ''
    version = ''
    for line in metadata_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('Name: '):
            name = line.removeprefix('Name: ').strip()
        elif line.startswith('Version: '):
            version = line.removeprefix('Version: ').strip()
        if name and version:
            break
    if not name or not version:
        raise ContractError(f'distribution METADATA is malformed: {metadata_path}')
    return (name, version)

def _macro_value(path: Path, macro: str) -> int:
    pattern = re.compile(f'^\\s*#\\s*define\\s+{re.escape(macro)}\\s+(\\d+)\\b', re.MULTILINE)
    match = pattern.search(path.read_text(encoding='utf-8'))
    if match is None:
        raise ContractError(f'cannot read {macro} from {path}')
    return int(match.group(1))

def persistent_cuda_toolchain_receipt(expected: Mapping[str, Any], *, smoke_directory: Path, package_versions: Mapping[str, str]) -> dict[str, Any]:
    """Verify the isolated CUDA 13.0 compiler closure and a real TileLang cubin."""
    if expected.get('schema_version') != 'guiagent-persistent-cuda-toolchain-v1':
        raise ContractError('persistent CUDA toolchain schema drifted')
    root = Path(str(expected.get('root', '')))
    cuda_home = Path(str(expected.get('cuda_home', '')))
    inventory = persistent_cuda_toolchain_inventory(root)
    if inventory != expected.get('inventory'):
        raise ContractError('persistent CUDA toolchain inventory drifted')
    try:
        cuda_home.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ContractError('CUDA_HOME is missing or outside the persistent CUDA root') from exc
    if cuda_home.is_symlink() or not cuda_home.is_dir():
        raise ContractError('CUDA_HOME must be a regular directory inside the bundle')
    environment_contract = expected.get('environment', {})
    if os.environ.get('CUDA_HOME') != str(cuda_home):
        raise ContractError('CUDA_HOME does not match the persistent CUDA toolchain')
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path and cuda_path != str(cuda_home):
        raise ContractError('CUDA_PATH points at a competing CUDA toolchain')
    path_prefix = [str(value) for value in environment_contract.get('PATH_prefix', ())]
    actual_path = os.environ.get('PATH', '').split(os.pathsep)
    if not path_prefix or actual_path[:len(path_prefix)] != path_prefix:
        raise ContractError('persistent CUDA PATH prefix drifted')
    if environment_contract.get('CUDA_HOME') != str(cuda_home):
        raise ContractError('persistent CUDA environment contract drifted')
    if environment_contract.get('forbid_bundle_in_LD_LIBRARY_PATH') is not True:
        raise ContractError('persistent CUDA LD_LIBRARY_PATH policy drifted')
    for entry in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep):
        if entry and (entry == str(root) or entry.startswith(f'{root}{os.sep}')):
            raise ContractError('persistent CUDA bundle must not enter LD_LIBRARY_PATH')
    for variable in ('CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH'):
        for entry in os.environ.get(variable, '').split(os.pathsep):
            lowered = entry.lower()
            if entry and ('/cuda' in lowered or '/nvidia/cu' in lowered):
                raise ContractError(f'competing CUDA include path in {variable}: {entry}')
    bypass_variables = ('CPPFLAGS', 'CFLAGS', 'CXXFLAGS', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')
    if any(('CCCL_DISABLE_CTK_COMPATIBILITY_CHECK' in os.environ.get(variable, '') for variable in bypass_variables)):
        raise ContractError('CCCL compatibility-check bypass is forbidden')
    if os.environ.get('FLA_TILELANG') == '0':
        raise ContractError('FLA TileLang backend may not be disabled')
    if os.environ.get('FLA_DISABLE_BACKEND_DISPATCH') not in (None, '', '0'):
        raise ContractError('FLA backend dispatch may not be disabled')
    declared_wheels = expected.get('wheel_provenance', ())
    required_wheels = {'nvidia-cuda-cccl': '13.0.50', 'nvidia-cuda-crt': '13.0.88', 'nvidia-cuda-nvcc': '13.0.88', 'nvidia-cuda-runtime': '13.0.96', 'nvidia-nvvm': '13.0.88'}
    if not isinstance(declared_wheels, list) or {str(item.get('distribution')): str(item.get('version')) for item in declared_wheels} != required_wheels:
        raise ContractError('persistent CUDA wheel closure drifted')
    wheel_receipts: list[dict[str, Any]] = []
    canonical_root = root.resolve(strict=True)
    for item in declared_wheels:
        relative_dist_info = Path(str(item.get('dist_info_path', '')))
        if relative_dist_info.is_absolute() or not relative_dist_info.parts or '..' in relative_dist_info.parts:
            raise ContractError('CUDA wheel dist-info paths must be safe and relative')
        dist_info = root / relative_dist_info
        try:
            dist_info.resolve(strict=True).relative_to(canonical_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ContractError('CUDA wheel dist-info is missing or escapes the bundle') from exc
        if dist_info.is_symlink() or not dist_info.is_dir():
            raise ContractError('CUDA wheel dist-info must be a regular directory')
        metadata_path = dist_info / 'METADATA'
        direct_url_path = dist_info / 'direct_url.json'
        record_path = dist_info / 'RECORD'
        for path in (metadata_path, direct_url_path, record_path):
            if not path.is_file() or path.is_symlink():
                raise ContractError(f'CUDA wheel provenance file is missing/unsafe: {path}')
        (actual_name, actual_version) = _distribution_name_version(metadata_path)
        if actual_name != item['distribution'] or actual_version != item['version']:
            raise ContractError(f'CUDA wheel METADATA drifted: {dist_info}')
        wheel_path = Path(str(item.get('wheel_path', '')))
        wheel_receipt = _frozen_regular_file(wheel_path, expected_sha256=str(item.get('wheel_sha256', '')), expected_size=int(item.get('wheel_size_bytes', -1)), expected_executable=False, label='CUDA provenance wheel')
        direct_url = load_json(direct_url_path)
        archive_hashes = direct_url.get('archive_info', {}).get('hashes', {})
        if direct_url.get('url') != wheel_path.as_uri():
            raise ContractError(f'CUDA wheel direct_url provenance drifted: {dist_info}')
        wheel_receipts.append({'distribution': actual_name, 'version': actual_version, 'dist_info_path': relative_dist_info.as_posix(), 'metadata_sha256': sha256_file(metadata_path), 'record_sha256': sha256_file(record_path), 'direct_url_sha256': sha256_file(direct_url_path), 'wheel': wheel_receipt})
    critical_expected = expected.get('critical_files', ())
    critical = _critical_toolchain_receipts(root, critical_expected)
    required_critical = {'site-packages/nvidia/cu13/bin/nvcc', 'site-packages/nvidia/cu13/nvvm/libdevice/libdevice.10.bc', 'site-packages/nvidia/cu13/include/cuda_runtime_api.h', 'site-packages/nvidia/cu13/include/crt/host_config.h', 'site-packages/nvidia/cu13/include/cccl/cuda/std/__cccl/cuda_toolkit.h', 'site-packages/nvidia/cu13/include/cccl/cuda/std/__cccl/version.h'}
    if not required_critical <= {str(item.get('path')) for item in critical_expected}:
        raise ContractError('persistent CUDA critical-file set is incomplete')
    python_critical_expected = expected.get('python_jit_critical_files', ())
    if not isinstance(python_critical_expected, list) or not python_critical_expected:
        raise ContractError('Python JIT critical-file set is missing')
    python_critical = [_frozen_regular_file(Path(str(item.get('path', ''))), expected_sha256=str(item.get('sha256', '')), expected_size=int(item.get('size_bytes', -1)), expected_executable=bool(item.get('executable')), label='Python JIT critical file') for item in python_critical_expected]
    expected_jit_versions = {str(item.get('distribution')): str(item.get('version')) for item in expected.get('python_jit_packages', ())}
    required_jit_versions = {'tilelang': '0.1.9', 'flash-linear-attention': '0.5.1'}
    if expected_jit_versions != required_jit_versions:
        raise ContractError('Python JIT package contract drifted')
    if {name: package_versions.get(name) for name in required_jit_versions} != required_jit_versions:
        raise ContractError('Python JIT package versions drifted')
    versions = expected.get('versions', {})
    expected_versions = {'cuda_version': 13000, 'cudart_version': 13000, 'cccl_version': 3000001}
    if versions != expected_versions:
        raise ContractError('persistent CUDA header-version contract drifted')
    cuda_version = _macro_value(cuda_home / 'include/cuda.h', 'CUDA_VERSION')
    cudart_version = _macro_value(cuda_home / 'include/cuda_runtime_api.h', 'CUDART_VERSION')
    cccl_version = _macro_value(cuda_home / 'include/cccl/cuda/std/__cccl/version.h', 'CCCL_VERSION')
    if {'cuda_version': cuda_version, 'cudart_version': cudart_version, 'cccl_version': cccl_version} != expected_versions:
        raise ContractError('persistent CUDA compiler/header versions are incompatible')
    nvcc_contract = expected.get('nvcc', {})
    nvcc = Path(str(nvcc_contract.get('path', '')))
    if nvcc != cuda_home / 'bin/nvcc':
        raise ContractError('certified nvcc must be CUDA_HOME/bin/nvcc')
    nvcc_file = _frozen_regular_file(nvcc, expected_sha256=str(nvcc_contract.get('sha256', '')), expected_size=int(nvcc_contract.get('size_bytes', -1)), expected_executable=True, label='certified nvcc')
    environment = os.environ.copy()
    nvcc_version = _run_contract_command([str(nvcc), '--version'], env=environment)
    if str(nvcc_contract.get('version_line', '')) not in nvcc_version.splitlines() or str(nvcc_contract.get('build_line', '')) not in nvcc_version.splitlines():
        raise ContractError('nvcc release/build drifted')
    smoke = expected.get('smoke', {})
    tilelang_nvcc_path = next((Path(item['path']) for item in python_critical_expected if str(item.get('path', '')).endswith('/tilelang/contrib/nvcc.py')), None)
    if tilelang_nvcc_path is None:
        raise ContractError('TileLang nvcc.py is not bound by the CUDA contract')
    tilelang_root = tilelang_nvcc_path.parent.parent
    expected_flags = ['--cubin', '-O3', '-lineinfo', '-arch=sm_80', '-std=c++17', f'-I{tilelang_root}/3rdparty/../src', f'-I{tilelang_root}/3rdparty/cutlass/include', '--use_fast_math']
    if smoke.get('flags') != expected_flags or smoke.get('source_includes') != ['tl_templates/cuda/atomic.h', 'cuda/std/__cccl/version.h'] or smoke.get('explicit_ccbin') is not False or (smoke.get('gpu_execution') is not False) or (nvcc_contract.get('smoke_arch') != 'sm_80') or (nvcc_contract.get('gpu_execution') is not False):
        raise ContractError('TileLang CUDA cubin smoke contract drifted')
    if not os.environ.get('NVCC_CCBIN'):
        raise ContractError('NVCC_CCBIN must select the certified host compiler')
    source_text = '#include <tl_templates/cuda/atomic.h>\n#include <cuda/std/__cccl/version.h>\n#if CCCL_VERSION != 3000001\n#error CCCL version drifted\n#endif\n#if __CUDACC_VER_MAJOR__ != 13 || __CUDACC_VER_MINOR__ != 0 || __CUDACC_VER_BUILD__ != 88\n#error nvcc version drifted\n#endif\n#if !defined(__GNUC__) || __GNUC__ != 11\n#error NVCC_CCBIN did not select GCC 11\n#endif\nextern "C" __global__ void qwen35b_tilelang_cccl_smoke(int* value) { if (threadIdx.x == 0) *value = 1; }\n'
    with tempfile.TemporaryDirectory(prefix='qwen35b_cuda_smoke_', dir=smoke_directory) as temporary:
        temporary_root = Path(temporary)
        source_path = temporary_root / 'tilelang_cccl_smoke.cu'
        cubin_path = temporary_root / 'tilelang_cccl_smoke.cubin'
        source_path.write_text(source_text, encoding='utf-8')
        command = [str(nvcc), *expected_flags, '-o', str(cubin_path), str(source_path)]
        if any((value == '-ccbin' or value.startswith('-ccbin=') or value.startswith('--compiler-bindir') for value in command)):
            raise ContractError('TileLang cubin smoke must exercise NVCC_CCBIN')
        _run_contract_command(command, env=environment)
        try:
            cubin_metadata = cubin_path.lstat()
        except FileNotFoundError as exc:
            raise ContractError('TileLang CUDA cubin smoke produced no output') from exc
        if cubin_path.is_symlink() or not stat.S_ISREG(cubin_metadata.st_mode) or cubin_metadata.st_size <= 0:
            raise ContractError('TileLang CUDA cubin smoke output is empty/unsafe')
    return {'status': 'passed', 'profile_id': expected.get('profile_id'), 'root': str(root), 'cuda_home': str(cuda_home), 'inventory': inventory, 'wheel_provenance': wheel_receipts, 'critical_files': critical, 'python_jit_packages': required_jit_versions, 'python_jit_critical_files': python_critical, 'versions': expected_versions, 'nvcc': {**nvcc_file, 'version_line': nvcc_contract['version_line'], 'build_line': nvcc_contract['build_line']}, 'tilelang_cccl_cubin_smoke': {'status': 'passed', 'flags': expected_flags, 'source_sha256': hashlib.sha256(source_text.encode('utf-8')).hexdigest(), 'host_compiler': os.environ['NVCC_CCBIN'], 'output_nonempty': True, 'explicit_ccbin': False, 'gpu_execution': False}}

def software_receipt(resource: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    expected = resource['software']
    cuda_contract = expected['persistent_cuda_toolchain']
    packages = {}
    required_distributions = ['torch', 'transformers', 'accelerate', 'peft']
    required_distributions.extend((str(item['distribution']) for item in cuda_contract.get('python_jit_packages', ())))
    for name in required_distributions:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ContractError(f'required package is not installed: {name}') from exc
    expected_python = Path(expected['python_executable'])
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise ContractError(f'runtime Python {sys.executable} != certified {expected_python}')
    if f'{sys.version_info.major}.{sys.version_info.minor}' != str(expected['python']):
        raise ContractError('Python major/minor does not match the certified resource profile')
    for name in ('transformers', 'accelerate', 'peft'):
        if packages[name] != str(expected[name]):
            raise ContractError(f'{name}={packages[name]} != certified {expected[name]}')
    if packages['torch'].split('+', 1)[0] != str(expected['torch']):
        raise ContractError(f"torch={packages['torch']} != certified {expected['torch']}")
    try:
        from transformers import Trainer
        from transformers.trainer import get_fsdp_ckpt_kwargs
        fsdp_kwargs = get_fsdp_ckpt_kwargs()
    except Exception as exc:
        raise ContractError('cannot verify Transformers adapter-only FSDP checkpoint support') from exc
    if fsdp_kwargs != {'adapter_only': True}:
        raise ContractError(f'Transformers FSDP checkpoint is not adapter-only: {fsdp_kwargs}')
    get_num_items_source = inspect.getsource(Trainer._get_num_items_in_batch)
    compute_loss_source = inspect.getsource(Trainer.compute_loss)
    accelerator_setup_source = inspect.getsource(Trainer.create_accelerator_and_postprocess)
    token_scaling = validate_transformers_5121_token_scaling_source(get_num_items_source, compute_loss_source, accelerator_setup_source)
    persistent_tmux = persistent_tmux_receipt(expected['persistent_tmux'])
    jit_cache = jit_environment_receipt(expected['jit_cache'], project_root=project_root)
    toolchain = persistent_cpp_toolchain_receipt(expected['persistent_cpp_toolchain'], smoke_directory=Path(os.environ['TILELANG_TMP_DIR']))
    cuda_toolchain = persistent_cuda_toolchain_receipt(cuda_contract, smoke_directory=Path(os.environ['TILELANG_TMP_DIR']), package_versions=packages)
    return {'python': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}', 'python_executable': str(Path(sys.executable).resolve()), 'certified_python_executable': str(expected_python), 'packages': packages, 'persistent_tmux': persistent_tmux, 'persistent_cpp_toolchain': toolchain, 'persistent_cuda_toolchain': cuda_toolchain, 'jit_cache': jit_cache, 'get_fsdp_ckpt_kwargs': fsdp_kwargs, 'token_scaling': token_scaling, 'trainer_get_num_items_source_sha256': hashlib.sha256(get_num_items_source.encode('utf-8')).hexdigest(), 'trainer_compute_loss_source_sha256': hashlib.sha256(compute_loss_source.encode('utf-8')).hexdigest(), 'trainer_accelerator_setup_source_sha256': hashlib.sha256(accelerator_setup_source.encode('utf-8')).hexdigest()}

def gpu_receipt() -> dict[str, Any]:
    command = ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ContractError(f'nvidia-smi failed: {result.stderr.strip()}')
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 2:
        raise ContractError(f'container must expose exactly two GPUs, found {len(rows)}')
    parsed = []
    for (expected_index, row) in enumerate(rows):
        fields = [value.strip() for value in row.split(',')]
        if len(fields) != 4 or int(fields[0]) != expected_index:
            raise ContractError(f'unexpected GPU row: {row}')
        if 'A800' not in fields[1] or int(fields[2]) < 81000:
            raise ContractError(f'GPU {expected_index} is not an A800 80GB: {row}')
        parsed.append({'index': int(fields[0]), 'name': fields[1], 'memory_total_mib': int(fields[2]), 'memory_free_mib': int(fields[3])})
    return {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpus': parsed}

def _load_prepare_module(path: Path):
    spec = importlib.util.spec_from_file_location('prepare_qwen35b_v3_cur_50k', path)
    if spec is None or spec.loader is None:
        raise ContractError(f'cannot import prepare script: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def validate_data_receipt(receipt: Mapping[str, Any], manifest: Mapping[str, Any], paths: RuntimePaths) -> dict[str, Any]:
    expected = manifest['data']['current50k']
    if receipt.get('status') != 'passed':
        raise ContractError('current50k receipt did not pass')
    checks = {'source.sha256': (receipt.get('source', {}).get('sha256'), manifest['data']['source']['sha256']), 'source.rows': (receipt.get('source', {}).get('rows'), manifest['data']['source']['rows']), 'output.rows': (receipt.get('output', {}).get('rows'), expected['rows']), 'output.bytes': (receipt.get('output', {}).get('bytes'), expected['jsonl_bytes']), 'audit.state_side_counts': (receipt.get('audit', {}).get('state_side_counts'), expected['state_side_counts']), 'audit.unique_domains': (receipt.get('audit', {}).get('unique_domains'), expected['unique_domains']), 'audit.unique_images': (receipt.get('audit', {}).get('unique_images'), expected['unique_images']), 'audit.unique_md_hashes': (receipt.get('audit', {}).get('unique_md_hashes'), expected['unique_md_hashes']), 'audit.images_bytes': (receipt.get('audit', {}).get('images_bytes'), expected['images_bytes']), 'audit.ordered_ids_sha256': (receipt.get('audit', {}).get('ordered_ids_sha256'), expected['ordered_ids_sha256']), 'audit.ordered_image_hashes_sha256': (receipt.get('audit', {}).get('ordered_image_hashes_sha256'), expected['ordered_image_hashes_sha256']), 'audit.ordered_md_hashes_sha256': (receipt.get('audit', {}).get('ordered_md_hashes_sha256'), expected['ordered_md_hashes_sha256'])}
    drift = [name for (name, (actual, frozen)) in checks.items() if actual != frozen]
    if drift:
        raise ContractError(f'current50k receipt drifted: {drift}')
    if not paths.prepared_jsonl.is_file():
        raise ContractError('current50k JSONL does not match its receipt')
    image_list = receipt.get('image_list') or {}
    if not paths.image_list.is_file() or int(image_list.get('rows', -1)) != expected['rows']:
        raise ContractError('current50k image-list artifact does not match its receipt')
    return dict(receipt)

def prepare_data(manifest: Mapping[str, Any], paths: RuntimePaths) -> dict[str, Any]:
    source = resolve_project_path(paths.project_root, manifest['data']['source']['path'])
    prepare = _load_prepare_module(paths.prepare_path)
    if paths.prepared_jsonl.is_file() and paths.prepared_receipt.is_file():
        receipt = load_json(paths.prepared_receipt)
        return validate_data_receipt(receipt, manifest, paths)
    receipt = prepare.materialize_current_50k(source, paths.prepared_jsonl, paths.prepared_receipt, image_list=paths.image_list, verify_image_bytes=True, project_root=paths.project_root)
    return validate_data_receipt(receipt, manifest, paths)

def implementation_hashes(paths: RuntimePaths) -> dict[str, str]:
    values = {'trainer': paths.trainer_path, 'runtime': paths.runtime_path, 'runner': paths.runner_path, 'prepare': paths.prepare_path, **{relative: paths.staging_root / relative for relative in FORMAL_EVALUATION_FILES}}
    missing = [name for (name, path) in values.items() if not path.is_file()]
    if missing:
        raise ContractError(f'missing staged implementation files: {missing}')
    return {name: sha256_file(path) for (name, path) in values.items()}

def verify_image_inventory(*, image_list_path: Path, project_root: Path, expected_rows: int, expected_total_bytes: int) -> dict[str, Any]:
    if not image_list_path.is_file() or image_list_path.is_symlink():
        raise ContractError(f'image inventory list is missing or a symlink: {image_list_path}')
    project_root = project_root.resolve()
    ordered: list[str] = []
    roots: set[str] = set()
    total_bytes = 0
    with image_list_path.open(encoding='utf-8') as handle:
        for (line_number, raw_line) in enumerate(handle, start=1):
            value = raw_line.rstrip('\n')
            if not value or value != value.strip():
                raise ContractError(f'image inventory line {line_number} is empty or padded')
            path = Path(value)
            if not path.is_absolute() or '..' in path.parts:
                raise ContractError(f'image inventory line {line_number} is not an absolute safe path')
            if path.is_symlink():
                raise ContractError(f'image inventory line {line_number} is a symlink: {path}')
            try:
                canonical_path = path.resolve(strict=True)
                relative = canonical_path.relative_to(project_root)
            except (FileNotFoundError, ValueError) as exc:
                raise ContractError(f'image inventory line {line_number} is missing or escapes project root: {path}') from exc
            parts = relative.parts
            markers = [index for index in range(len(parts) - 1) if parts[index:index + 2] == ('render', 'images')]
            if len(markers) != 1 or markers[0] + 2 >= len(parts):
                raise ContractError(f'image inventory line {line_number} violates the project render/images contract')
            try:
                metadata = path.lstat()
            except FileNotFoundError as exc:
                raise ContractError(f'image inventory line {line_number} is missing: {path}') from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f'image inventory line {line_number} is not a regular file: {path}')
            total_bytes += int(metadata.st_size)
            ordered.append(str(canonical_path))
            root_parts = parts[:markers[0] + 2]
            roots.add(str(project_root.joinpath(*root_parts)))
    if len(ordered) != expected_rows or len(set(ordered)) != expected_rows:
        raise ContractError(f'image inventory has rows={len(ordered)}, unique={len(set(ordered))}; expected exactly {expected_rows} unique paths')
    if total_bytes != expected_total_bytes:
        raise ContractError(f'image inventory bytes={total_bytes}, expected {expected_total_bytes}')
    digest = hashlib.sha256()
    for value in ordered:
        digest.update(value.encode('utf-8'))
        digest.update(b'\n')
    return {'schema_version': 'guiagent-current50k-image-inventory-v1', 'status': 'passed', 'image_list_path': str(image_list_path), 'image_list_sha256': sha256_file(image_list_path), 'rows': len(ordered), 'unique_paths': len(set(ordered)), 'regular_files': len(ordered), 'symlinks': 0, 'total_bytes': total_bytes, 'path_contract': 'absolute_project_path_containing_render/images', 'inventory_roots': sorted(roots), 'ordered_paths_sha256': digest.hexdigest()}

def run_preflight(manifest_path: Path, *, require_gpu: bool=True) -> dict[str, Any]:
    (manifest, resource, paths) = runtime_paths(manifest_path)
    validate_manifest_contract(manifest, resource)
    source = resolve_project_path(paths.project_root, manifest['data']['source']['path'])
    validation = resolve_project_path(paths.project_root, manifest['data']['validation']['path'])
    prompt = resolve_project_path(paths.project_root, manifest['data']['prompt']['path'])
    verify_hash(source, manifest['data']['source']['sha256'], 'frozen source100k')
    verify_hash(validation, manifest['data']['validation']['sha256'], 'Validation500')
    verify_hash(prompt, manifest['data']['prompt']['sha256'], 'localwm_cur prompt')
    data_receipt = prepare_data(manifest, paths)
    image_inventory = verify_image_inventory(image_list_path=paths.image_list, project_root=paths.project_root, expected_rows=int(manifest['data']['current50k']['rows']), expected_total_bytes=int(manifest['data']['current50k']['images_bytes']))
    plan = build_exposure_plan()
    atomic_write_json(paths.exposure_plan, plan)
    model_root = Path(manifest['model']['base_model_path'])
    canonical_sums_path = resolve_project_path(paths.project_root, manifest['model']['sha256sums_path'])
    canonicalization_receipt_path = resolve_project_path(paths.project_root, manifest['model']['sha256sums_canonicalization_receipt_path'])
    canonicalization = canonicalize_legacy_model_sha256sums(original_path=Path(manifest['model']['original_sha256sums_path']), original_sha256=manifest['model']['original_sha256sums_sha256'], canonical_path=canonical_sums_path, receipt_path=canonicalization_receipt_path)
    model_receipt = verify_model_directory(model_root, canonical_sums_path)
    model_receipt['canonicalization'] = canonicalization
    software = software_receipt(resource, project_root=paths.project_root)
    gpu = gpu_receipt() if require_gpu else {'skipped_for_cpu_test': True}
    receipt = {'schema_version': 'guiagent-v3-cur-preflight-v1', 'status': 'passed', 'gate_tier': 2, 'manifest_path': str(paths.manifest_path), 'manifest_sha256': sha256_file(paths.manifest_path), 'resource_profile_sha256': sha256_file(paths.resource_path), 'implementation_sha256s': implementation_hashes(paths), 'data_receipt_sha256': sha256_file(paths.prepared_receipt), 'data': data_receipt, 'image_inventory': image_inventory, 'exposure_plan_sha256': sha256_file(paths.exposure_plan), 'model': model_receipt, 'software': software, 'gpu': gpu}
    atomic_write_json(paths.preflight_receipt, receipt)
    state = read_state(paths)
    if state['phase'] == 'NEW':
        transition(paths, 'PREFLIGHT_PASSED', step=0)
    return receipt

def validate_preflight_receipt(manifest: Mapping[str, Any], paths: RuntimePaths) -> dict[str, Any]:
    if not paths.preflight_receipt.is_file():
        raise ContractError('preflight receipt is missing')
    receipt = load_json(paths.preflight_receipt)
    if receipt.get('status') != 'passed':
        raise ContractError('preflight receipt did not pass')
    resource = load_json(paths.resource_path)
    current_software = software_receipt(resource, project_root=paths.project_root)
    if receipt.get('software') != current_software:
        raise ContractError('certified software/toolchain changed after preflight')
    if not paths.prepared_receipt.is_file():
        raise ContractError('prepared-data receipt changed after preflight')
    validate_data_receipt(load_json(paths.prepared_receipt), manifest, paths)
    current_inventory = verify_image_inventory(image_list_path=paths.image_list, project_root=paths.project_root, expected_rows=int(manifest['data']['current50k']['rows']), expected_total_bytes=int(manifest['data']['current50k']['images_bytes']))
    if receipt.get('image_inventory') != current_inventory:
        raise ContractError('current50k image inventory changed after preflight')
    if not paths.exposure_plan.is_file():
        raise ContractError('exposure plan changed after preflight')
    canonical_sums = resolve_project_path(paths.project_root, manifest['model']['sha256sums_path'])
    model_receipt = receipt.get('model', {})
    if not canonical_sums.is_file():
        raise ContractError('canonical model SHA256SUMS changed after preflight')
    for item in model_receipt.get('files', []):
        path = Path(model_receipt['model_root']) / item['path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(item['size_bytes']):
            raise ContractError(f"model file presence/size changed after preflight: {item['path']}")
    return receipt

def build_torchrun_command(paths: RuntimePaths, *, output_root: Path, start_step: int, end_step: int, resume_from: Path | None, preflight: Mapping[str, Any], certification_boundary: bool=False) -> list[str]:
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', str(paths.trainer_path), '--manifest', str(paths.manifest_path), '--output-root', str(output_root), '--segment-start-step', str(start_step), '--segment-end-step', str(end_step)]
    if resume_from is not None:
        command.extend(['--resume-from-checkpoint', str(resume_from)])
    if certification_boundary:
        command.append('--certification-boundary')
    for (name, digest) in sorted(preflight['implementation_sha256s'].items()):
        if name != 'trainer':
            command.extend(['--implementation-sha256', f'{name}={digest}'])
    return command

def run_checked(command: Sequence[str], *, cwd: Path, log_path: Path, env: Mapping[str, str] | None=None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('COMMAND ' + shlex.join(command) + '\n')
        log.flush()
        result = subprocess.run(list(command), cwd=cwd, env=dict(os.environ) | dict(env or {}), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise ContractError(f'command failed with exit {result.returncode}; see {log_path}')

def validate_checkpoint(checkpoint_dir: Path, *, step: int, preflight: Mapping[str, Any], resource_sha256: str) -> dict[str, Any]:
    binding_path = checkpoint_dir / 'checkpoint_binding.json'
    integrity_path = checkpoint_dir / 'checkpoint_integrity.json'
    if not binding_path.is_file() or not integrity_path.is_file():
        raise ContractError(f'checkpoint-{step} is not committed')
    validate_dcp_tree(checkpoint_dir)
    binding = load_json(binding_path)
    if binding.get('status') != 'committed' or int(binding.get('global_step', -1)) != step:
        raise ContractError(f'checkpoint-{step} binding is invalid')
    expected = {'manifest_sha256': preflight['manifest_sha256'], 'resource_profile_sha256': resource_sha256, 'implementation_sha256s': preflight['implementation_sha256s'], 'integrity_sha256': sha256_file(integrity_path)}
    for (name, value) in expected.items():
        if binding.get(name) != value:
            raise ContractError(f'checkpoint-{step} binding field drifted: {name}')
    integrity = load_json(integrity_path)
    actual_files = recursive_file_receipt(checkpoint_dir, excluded=('checkpoint_integrity.json', 'checkpoint_binding.json'))
    if integrity.get('files') != actual_files:
        raise ContractError(f'checkpoint-{step} file tree no longer matches its integrity receipt')
    return {'binding': binding, 'integrity': integrity}

def _adapter_hash(checkpoint_dir: Path) -> str:
    return sha256_file(checkpoint_dir / 'adapter_model.safetensors')

def _max_reserved_mib(checkpoint: Mapping[str, Any]) -> int:
    memory = checkpoint['integrity'].get('memory', {})
    values = [int(value['max_memory_reserved_mib']) for value in memory.values()]
    if len(values) != 2:
        raise ContractError('checkpoint memory receipt must contain both ranks')
    return max(values)

def validate_certification_checkpoint_health(checkpoint: Mapping[str, Any], *, step: int, memory_limit_mib: int) -> dict[str, Any]:
    memory = checkpoint['integrity'].get('memory', {})
    if set(memory) != {'rank0', 'rank1'}:
        raise ContractError(f'certification checkpoint-{step} lacks both rank health receipts')
    peaks: dict[str, int] = {}
    for (rank_name, receipt) in memory.items():
        peak = int(receipt.get('max_memory_reserved_mib', -1))
        if peak < 0 or peak > memory_limit_mib:
            raise ContractError(f'checkpoint-{step} {rank_name} reserved {peak} MiB exceeds {memory_limit_mib} MiB')
        if int(receipt.get('finite_logged_losses', 0)) <= 0:
            raise ContractError(f'checkpoint-{step} {rank_name} has no finite-loss evidence')
        if receipt.get('all_trainable_gradients_finite') is not True:
            raise ContractError(f'checkpoint-{step} {rank_name} gradient finite gate failed')
        if int(receipt.get('checked_trainable_gradient_tensors', -1)) != 840:
            raise ContractError(f'checkpoint-{step} {rank_name} did not check all 840 LoRA gradients')
        if int(receipt.get('finite_gradient_checks', 0)) <= 0:
            raise ContractError(f'checkpoint-{step} {rank_name} has no pre-optimizer gradient check')
        norm = float(receipt.get('global_gradient_norm_after_clipping', float('nan')))
        if not (norm >= 0.0 and norm < float('inf')):
            raise ContractError(f'checkpoint-{step} {rank_name} gradient norm is non-finite')
        peaks[rank_name] = peak
    boundary = validate_certification_boundary_receipts(checkpoint, step=step)
    return {'step': step, 'rank_max_reserved_mib': peaks, 'finite_loss_and_grad': True, 'boundary_sample': boundary}

def validate_certification_boundary_receipts(checkpoint: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    """Fail closed unless both ranks exercised the same worst-case collator path."""
    boundary = checkpoint['integrity'].get('certification_boundary', {})
    if set(boundary) != {'rank0', 'rank1'}:
        raise ContractError(f'certification checkpoint-{step} lacks both boundary receipts')
    image_hashes: set[str] = set()
    target_hashes: set[str] = set()
    summaries: dict[str, Any] = {}
    for (rank_index, rank_name) in enumerate(('rank0', 'rank1')):
        value = boundary[rank_name]
        expected = {'schema_version': 'guiagent-tier2-boundary-sample-v1', 'status': 'passed', 'rank': rank_index, 'injection_policy': 'first_microbatch_of_every_rank_local_ga_window', 'segment_start_step': step - 1, 'segment_end_step': step, 'max_length_config': 20000, 'max_output_tokens_config': 6000, 'max_pixels_config': 802816, 'raw_image_width': 896, 'raw_image_height': 896, 'raw_image_pixels': 802816, 'effective_processor_pixels': 802816, 'gradient_accumulation_steps': 30}
        for (field, expected_value) in expected.items():
            if value.get(field) != expected_value:
                raise ContractError(f'checkpoint-{step} {rank_name} boundary {field}={value.get(field)!r}, expected {expected_value!r}')
        supervised = int(value.get('actual_supervised_tokens', -1))
        target_tokens = int(value.get('target_tokenizer_tokens', -1))
        input_tokens = int(value.get('actual_input_tokens', -1))
        if not 5900 <= supervised <= 6000:
            raise ContractError(f'checkpoint-{step} {rank_name} supervised tokens={supervised}, expected 5900..6000')
        if not 5940 <= target_tokens <= 5980:
            raise ContractError(f'checkpoint-{step} {rank_name} target tokens={target_tokens}, expected 5940..5980')
        if not supervised <= input_tokens <= 20000:
            raise ContractError(f'checkpoint-{step} {rank_name} input tokens={input_tokens} do not cover the {supervised}-token target within 20K')
        pixel_shape = value.get('pixel_values_shape')
        image_grid = value.get('image_grid_thw')
        if not isinstance(pixel_shape, list) or not pixel_shape or any((not isinstance(item, int) or item <= 0 for item in pixel_shape)) or (not isinstance(image_grid, list)) or (len(image_grid) != 3) or any((not isinstance(item, int) or item <= 0 for item in image_grid)):
            raise ContractError(f'checkpoint-{step} {rank_name} image tensor geometry is invalid')
        microbatches = int(value.get('rank_local_microbatches', -1))
        ga_windows = int(value.get('ga_windows', -1))
        injected = value.get('injected_local_positions')
        if microbatches <= 0 or microbatches % 30 != 0 or ga_windows != microbatches // 30:
            raise ContractError(f'checkpoint-{step} {rank_name} GA window geometry is invalid')
        expected_positions = list(range(0, microbatches, 30))
        if injected != expected_positions or len(expected_positions) != ga_windows:
            raise ContractError(f'checkpoint-{step} {rank_name} did not inject the boundary in every GA window')
        image_hash = str(value.get('image_sha256', ''))
        target_hash = str(value.get('target_sha256', ''))
        if not re.fullmatch('[0-9a-f]{64}', image_hash) or not re.fullmatch('[0-9a-f]{64}', target_hash):
            raise ContractError(f'checkpoint-{step} {rank_name} boundary hashes are invalid')
        image_hashes.add(image_hash)
        target_hashes.add(target_hash)
        summaries[rank_name] = {'actual_input_tokens': input_tokens, 'actual_supervised_tokens': supervised, 'pixel_values_shape': pixel_shape, 'image_grid_thw': image_grid, 'raw_image_size': [896, 896], 'ga_windows': ga_windows}
    if len(image_hashes) != 1 or len(target_hashes) != 1:
        raise ContractError(f'checkpoint-{step} ranks did not use the same boundary sample')
    return {'status': 'passed', 'rank_receipts': summaries, 'image_sha256': next(iter(image_hashes)), 'target_sha256': next(iter(target_hashes))}

def run_certification(manifest_path: Path) -> dict[str, Any]:
    (manifest, _resource, paths) = runtime_paths(manifest_path)
    preflight = validate_preflight_receipt(manifest, paths) if paths.preflight_receipt.is_file() else run_preflight(manifest_path, require_gpu=True)
    resource_sha = sha256_file(paths.resource_path)
    state = read_state(paths)
    if paths.certification_receipt.is_file():
        receipt = load_json(paths.certification_receipt)
        if receipt.get('status') == 'passed':
            if state['phase'] == 'PREFLIGHT_PASSED':
                transition(paths, 'CERTIFIED', step=0, certification_reused=True)
            return receipt
        raise ContractError('existing certification receipt is stale')
    if state['phase'] == 'PREFLIGHT_PASSED':
        transition(paths, 'CERTIFY_STEP1_RUNNING', step=0)
    command1 = build_torchrun_command(paths, output_root=paths.certification_root, start_step=0, end_step=1, resume_from=None, preflight=preflight, certification_boundary=True)
    run_checked(command1, cwd=paths.project_root, log_path=paths.log_root / 'certify_step1.log', env={'CUDA_VISIBLE_DEVICES': '0,1', 'PYTHONPATH': str(paths.project_root / 'src')})
    checkpoint1_dir = paths.certification_root / 'checkpoints/checkpoint-1'
    checkpoint1 = validate_checkpoint(checkpoint1_dir, step=1, preflight=preflight, resource_sha256=resource_sha)
    transition(paths, 'CERTIFY_STEP1_COMMITTED', step=1)
    transition(paths, 'CERTIFY_RESUME_RUNNING', step=1)
    command2 = build_torchrun_command(paths, output_root=paths.certification_root, start_step=1, end_step=2, resume_from=checkpoint1_dir, preflight=preflight, certification_boundary=True)
    run_checked(command2, cwd=paths.project_root, log_path=paths.log_root / 'certify_step2_resume.log', env={'CUDA_VISIBLE_DEVICES': '0,1', 'PYTHONPATH': str(paths.project_root / 'src')})
    checkpoint2_dir = paths.certification_root / 'checkpoints/checkpoint-2'
    checkpoint2 = validate_checkpoint(checkpoint2_dir, step=2, preflight=preflight, resource_sha256=resource_sha)
    transition(paths, 'CERTIFY_STEP2_COMMITTED', step=2)
    if _adapter_hash(checkpoint1_dir) == _adapter_hash(checkpoint2_dir):
        raise ContractError('adapter did not change between certification steps')
    memory_limit = int(manifest['certification']['maximum_reserved_memory_mib_each'])
    step1_health = validate_certification_checkpoint_health(checkpoint1, step=1, memory_limit_mib=memory_limit)
    step2_health = validate_certification_checkpoint_health(checkpoint2, step=2, memory_limit_mib=memory_limit)
    observed_peak = max(_max_reserved_mib(checkpoint1), _max_reserved_mib(checkpoint2))
    resume_files = sorted(checkpoint2_dir.glob('resume_state_rank*.json'))
    if len(resume_files) != 2:
        raise ContractError('certification resume-state receipts for both ranks are missing')
    for path in resume_files:
        value = load_json(path)
        if value.get('global_step') != 1 or value.get('segment_start_step') != 1 or value.get('segment_end_step') != 2 or (value.get('optimizer_state_entries', 0) <= 0) or (value.get('scheduler_last_epoch') != 1):
            raise ContractError(f'optimizer/scheduler resume was not proven: {path}')
    receipt = {'schema_version': 'guiagent-v3-cur-fsdp2-certification-v1', 'status': 'passed', 'manifest_sha256': preflight['manifest_sha256'], 'implementation_sha256s': preflight['implementation_sha256s'], 'scheduler_horizon': 834, 'step1_binding_sha256': sha256_file(checkpoint1_dir / 'checkpoint_binding.json'), 'step2_binding_sha256': sha256_file(checkpoint2_dir / 'checkpoint_binding.json'), 'step1_adapter_sha256': _adapter_hash(checkpoint1_dir), 'step2_adapter_sha256': _adapter_hash(checkpoint2_dir), 'max_reserved_memory_mib': observed_peak, 'memory_limit_mib': memory_limit, 'checkpoint_health': [step1_health, step2_health], 'resume_state_receipts': [str(path) for path in resume_files]}
    atomic_write_json(paths.certification_receipt, receipt)
    transition(paths, 'CERTIFIED', step=0)
    return receipt

def validate_certification(preflight: Mapping[str, Any], paths: RuntimePaths) -> dict[str, Any]:
    if not paths.certification_receipt.is_file():
        raise ContractError('Tier2 certification receipt is missing')
    receipt = load_json(paths.certification_receipt)
    if receipt.get('status') != 'passed':
        raise ContractError('Tier2 certification did not pass')
    return receipt

def evaluation_receipt_path(paths: RuntimePaths, step: int) -> Path:
    return paths.output_root / f'validation/step_{step:06d}/evaluation_receipt.json'

def description_scopes_for_step(step: int) -> tuple[str, ...]:
    if step not in (0, *CHECKPOINT_STEPS):
        raise ContractError(f'Description evaluation is not permitted at step {step}')
    return ('core100', 'final500') if step == 834 else ('core100',)

def _safe_relative_child(root: Path, relative: str, *, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or '..' in value.parts:
        raise ContractError(f'{label} must be a safe relative path')
    return root / value

def posthoc_command_plan(manifest: Mapping[str, Any], paths: RuntimePaths, checkpoint_dir: Path | None, step: int) -> dict[str, list[str]]:
    """Construct the entire native evaluator CLI contract without shell indirection."""
    if step not in (0, *CHECKPOINT_STEPS):
        raise ContractError(f'posthoc evaluation is not permitted at step {step}')
    if (step == 0) != (checkpoint_dir is None):
        raise ContractError('only bare step-0 evaluation may omit the adapter checkpoint')
    evaluation = manifest['evaluation']
    description = evaluation['description_current']
    action = evaluation['action_retention']
    python_bin = str(Path(evaluation['python_executable']))
    base_model = str(Path(manifest['model']['base_model_path']))
    description_script = str(paths.staging_root / description['runner'])
    description_assets = str(paths.staging_root / description['asset_contract'])
    action_builder = str(paths.staging_root / action['builder'])
    action_runner = str(paths.staging_root / action['runner'])
    probe_dir = _safe_relative_child(paths.output_root, action['bundle_relative_to_output'], label='action-retention bundle')
    source_run = _safe_relative_child(paths.project_root, action['source_run'], label='action-retention source run')
    output_dir = evaluation_receipt_path(paths, step).parent
    description_run = output_dir / ('description_bare' if step == 0 else 'description_adapter')
    final500_run = output_dir / 'description_final500_adapter'
    action_run = output_dir / ('action_bare' if step == 0 else 'action_adapter')
    plan = {'action_build': [python_bin, action_builder, '--project-root', str(paths.project_root), '--source-run-root', str(source_run), '--output-dir', str(probe_dir)], 'action_score': [python_bin, action_runner, 'score-model', '--probe-manifest', str(probe_dir / 'manifest.json'), '--project-root', str(paths.project_root), '--production-contract-root', str(_safe_relative_child(probe_dir, action['production_contract_bundle_path'], label='portable production-contract bundle')), '--model-path', base_model, '--model-fingerprint', action['base_model_fingerprint'], '--role', 'bare' if step == 0 else 'adapter', '--output-dir', str(action_run), '--max-context-tokens', str(action['max_context_tokens']), '--max-memory-gib', str(action['max_memory_gib_each']), '--resume'], 'description_run': [python_bin, description_script, 'run', '--project-root', str(paths.project_root), '--asset-contract', description_assets, '--python-bin', python_bin, '--base-model', base_model, '--output-dir', str(description_run), '--scope', 'core100'], 'description_verify': [python_bin, description_script, 'verify-output', '--project-root', str(paths.project_root), '--asset-contract', description_assets, '--base-model', base_model, '--output-dir', str(description_run), '--scope', 'core100']}
    if 'final500' in description_scopes_for_step(step):
        plan['description_final500_run'] = [python_bin, description_script, 'run', '--project-root', str(paths.project_root), '--asset-contract', description_assets, '--python-bin', python_bin, '--base-model', base_model, '--output-dir', str(final500_run), '--scope', 'final500']
        plan['description_final500_verify'] = [python_bin, description_script, 'verify-output', '--project-root', str(paths.project_root), '--asset-contract', description_assets, '--base-model', base_model, '--output-dir', str(final500_run), '--scope', 'final500']
    if step > 0:
        if checkpoint_dir is None:
            raise AssertionError('adapter checkpoint disappeared')
        adapter = str(checkpoint_dir)
        plan['action_score'].extend(['--adapter-path', adapter])
        plan['description_run'].extend(['--adapter-path', adapter])
        plan['description_verify'].extend(['--adapter-path', adapter])
        if 'final500' in description_scopes_for_step(step):
            plan['description_final500_run'].extend(['--adapter-path', adapter])
            plan['description_final500_verify'].extend(['--adapter-path', adapter])
        base_dir = evaluation_receipt_path(paths, 0).parent
        plan['action_compare'] = [python_bin, action_runner, 'compare', '--bare-run', str(base_dir / 'action_bare'), '--adapter-run', str(action_run), '--output', str(output_dir / 'action_comparison.json')]
        plan['description_compare'] = [python_bin, description_script, 'compare', '--bare-receipt', str(base_dir / 'description_bare/evaluation_receipt.json'), '--checkpoint-receipt', str(description_run / 'evaluation_receipt.json'), '--output', str(output_dir / 'description_comparison.json'), '--min-elef1-improvement', str(description['min_ele_f1_improvement']), '--max-format-success-drop', str(description['max_format_success_drop'])]
    return plan

def posthoc_environment(paths: RuntimePaths) -> dict[str, str]:
    return {'CUDA_VISIBLE_DEVICES': '0,1', 'PYTHONPATH': str(paths.project_root / 'src'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        raise ContractError(f'missing JSONL: {path}')
    return sum((1 for line in path.open(encoding='utf-8') if line.strip()))

def verify_portable_production_contract(probe_manifest: Mapping[str, Any], contract_root: Path) -> dict[str, Any]:
    production = probe_manifest.get('source', {}).get('production_contract', {})
    if not isinstance(production, Mapping) or len(production) != 4:
        raise ContractError('portable production contract must declare exactly four files')
    entries: list[list[Any]] = []
    expected_paths: set[str] = set()
    for (label, item) in sorted(production.items()):
        if not isinstance(item, Mapping):
            raise ContractError(f'portable production contract entry is invalid: {label}')
        relative_text = str(item.get('path', ''))
        relative = Path(relative_text)
        if not relative_text or relative.is_absolute() or '..' in relative.parts or (relative.as_posix() != relative_text) or (relative_text in expected_paths):
            raise ContractError(f'portable production contract path is unsafe: {relative_text!r}')
        expected_paths.add(relative_text)
        path = contract_root / relative
        if not path.is_file() or path.is_symlink():
            raise ContractError(f'portable production contract file is missing/non-regular: {path}')
        digest = sha256_file(path)
        entries.append([relative_text, digest, path.stat().st_size])
    actual_paths: set[str] = set()
    if not contract_root.is_dir() or contract_root.is_symlink():
        raise ContractError(f'portable production contract root is invalid: {contract_root}')
    for path in contract_root.rglob('*'):
        if path.is_symlink():
            raise ContractError(f'symlink is forbidden in portable production contract: {path}')
        if path.is_file():
            actual_paths.add(path.relative_to(contract_root).as_posix())
    if actual_paths != expected_paths:
        raise ContractError(f'portable production contract file set drifted: missing={sorted(expected_paths - actual_paths)}, unexpected={sorted(actual_paths - expected_paths)}')
    entries.sort(key=lambda item: str(item[0]))
    aggregate = {'algorithm': 'sha256(canonical_json([[relative_path,file_sha256,size_bytes],...]))', 'entry_count': len(entries), 'total_bytes': sum((int(item[2]) for item in entries)), 'sha256': canonical_json_sha256(entries)}
    expected = probe_manifest.get('artifacts', {}).get('production_contract', {})
    for (key, value) in aggregate.items():
        if expected.get(key) != value:
            raise ContractError(f'portable production contract aggregate drifted: {key}')
    return aggregate

def _validate_native_description_scope(manifest: Mapping[str, Any], paths: RuntimePaths, checkpoint_dir: Path | None, step: int, scope: str) -> dict[str, Any]:
    output_dir = evaluation_receipt_path(paths, step).parent
    if scope == 'core100':
        run_dir = output_dir / ('description_bare' if step == 0 else 'description_adapter')
    elif scope == 'final500' and step == 834:
        run_dir = output_dir / 'description_final500_adapter'
    else:
        raise ContractError(f'step-{step} does not permit Description scope {scope}')
    receipt_path = run_dir / 'evaluation_receipt.json'
    receipt = load_json(receipt_path)
    expected_role = 'bare' if step == 0 else 'adapter'
    expected_adapter = str(checkpoint_dir) if checkpoint_dir is not None else None
    description = manifest['evaluation']['description_current']
    scope_contract = description[scope]
    expected_rows = int(scope_contract['rows'])
    if receipt.get('schema_version') != 'guiagent-qwen35b-description-current-receipt-v2' or receipt.get('status') != 'passed' or receipt.get('evaluation_scope') != scope or (receipt.get('role') != expected_role) or (receipt.get('base_model') != manifest['model']['base_model_path']) or (receipt.get('adapter_path') != expected_adapter) or (int(receipt.get('metrics', {}).get('evaluated_rows', -1)) != expected_rows) or (receipt.get('gpu_contract') != {'cuda_visible_devices': '0,1', 'model_device_map': 'balanced', 'embedding_device': 'cuda:0', 'cpu_fallback': False}):
        raise ContractError(f'step-{step} native Description {scope} receipt is invalid')
    selection = receipt.get('selection', {})
    if scope == 'core100':
        for name in ('ordered_sample_ids_sha256', 'ordered_current_image_paths_sha256', 'selection_contract_sha256'):
            if selection.get(name) != scope_contract[name]:
                raise ContractError(f'step-{step} core100 selection drifted: {name}')
    expected_outputs = {'selection_receipt': 'selection_receipt.json', 'evaluation_subset': 'evaluation_subset.jsonl', 'predictions': 'predictions.jsonl', 'metrics': 'metrics.json', 'run_config': 'run_config.json', 'length_validation': 'length_validation.json'}
    for (name, filename) in expected_outputs.items():
        path = run_dir / filename
        item = receipt.get('outputs', {}).get(name, {})
        if item.get('path') != str(path):
            raise ContractError(f'step-{step} Description {scope} output binding drifted: {name}')
    if _count_jsonl(run_dir / 'predictions.jsonl') != expected_rows:
        raise ContractError(f'step-{step} Description {scope} predictions are not exactly {expected_rows} rows')
    if scope == 'core100' and step > 0:
        comparison_path = output_dir / 'description_comparison.json'
        comparison = load_json(comparison_path)
        if comparison.get('schema_version') != 'guiagent-qwen35b-description-current-core100-comparison-v2' or comparison.get('status') != 'passed':
            raise ContractError(f'step-{step} Description core100 comparison failed')
        bare_receipt = evaluation_receipt_path(paths, 0).parent / 'description_bare/evaluation_receipt.json'
    return receipt

def validate_native_description_run(manifest: Mapping[str, Any], paths: RuntimePaths, checkpoint_dir: Path | None, step: int) -> dict[str, dict[str, Any]]:
    receipts = {'core100': _validate_native_description_scope(manifest, paths, checkpoint_dir, step, 'core100')}
    if 'final500' in description_scopes_for_step(step):
        receipts['final500'] = _validate_native_description_scope(manifest, paths, checkpoint_dir, step, 'final500')
    return receipts

def expected_description_wrapper(manifest: Mapping[str, Any], paths: RuntimePaths, checkpoint_dir: Path | None, step: int, native: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output_dir = evaluation_receipt_path(paths, step).parent
    core_dir = output_dir / ('description_bare' if step == 0 else 'description_adapter')
    core = native['core100']
    wrapper: dict[str, Any] = {'schema_version': 'guiagent-description-current-gate-wrapper-v2', 'status': 'passed', 'global_step': step, 'mode': 'base_baseline' if step == 0 else 'checkpoint_vs_base', 'core100': {'status': 'passed', 'rows': 100, 'paired_comparison': 'baseline_recorded' if step == 0 else 'passed', 'native_evaluation_receipt_path': str(core_dir / 'evaluation_receipt.json'), 'native_evaluation_receipt_sha256': sha256_file(core_dir / 'evaluation_receipt.json'), 'selection_contract_sha256': core['selection']['selection_contract_sha256'], 'EleF1': core['metrics']['EleF1'], 'format_success_rate': core['metrics']['format_success_rate']}, 'final500': {'status': 'not_required' if step != 834 else 'passed', 'absolute_report_only': True}}
    if step > 0:
        if checkpoint_dir is None:
            raise ContractError('adapter Description wrapper requires a checkpoint')
        base_receipt = evaluation_receipt_path(paths, 0).parent / 'description_bare/evaluation_receipt.json'
        wrapper['core100'].update({'baseline_native_receipt_sha256': sha256_file(base_receipt), 'comparison_sha256': sha256_file(output_dir / 'description_comparison.json')})
        wrapper.update({'checkpoint_binding_sha256': sha256_file(checkpoint_dir / 'checkpoint_binding.json'), 'adapter_sha256': _adapter_hash(checkpoint_dir)})
    if 'final500' in description_scopes_for_step(step):
        final_dir = output_dir / 'description_final500_adapter'
        final = native['final500']
        wrapper['final500'].update({'rows': 500, 'native_evaluation_receipt_path': str(final_dir / 'evaluation_receipt.json'), 'native_evaluation_receipt_sha256': sha256_file(final_dir / 'evaluation_receipt.json'), 'ordered_source_indices_sha256': final['selection']['ordered_source_indices_sha256'], 'EleF1': final['metrics']['EleF1'], 'format_success_rate': final['metrics']['format_success_rate'], 'by_action_bucket': final['metrics']['by_action_bucket']})
    return wrapper

def validate_native_action_run(manifest: Mapping[str, Any], paths: RuntimePaths, checkpoint_dir: Path | None, step: int) -> dict[str, Any]:
    output_dir = evaluation_receipt_path(paths, step).parent
    run_dir = output_dir / ('action_bare' if step == 0 else 'action_adapter')
    run_manifest_path = run_dir / 'run_manifest.json'
    records_path = run_dir / 'records.jsonl'
    placement_path = run_dir / 'gpu_placement.json'
    summary_path = run_dir / 'summary.json'
    run_manifest = load_json(run_manifest_path)
    summary = load_json(summary_path)
    placement = load_json(placement_path)
    action = manifest['evaluation']['action_retention']
    probe_manifest_path = _safe_relative_child(paths.output_root, action['bundle_relative_to_output'], label='action-retention bundle') / 'manifest.json'
    probe_manifest = load_json(probe_manifest_path)
    contract_root = _safe_relative_child(probe_manifest_path.parent, action['production_contract_bundle_path'], label='portable production-contract bundle')
    expected_contract_aggregate = verify_portable_production_contract(probe_manifest, contract_root)
    expected_adapter = str(checkpoint_dir) if checkpoint_dir is not None else None
    expected_inference = {'enable_thinking': False, 'do_sample': False, 'temperature': 0, 'retries': 0, 'max_new_tokens': 512, 'max_context_tokens': 262144, 'min_pixels': 200704, 'max_pixels': 802816, 'teacher_forced_loss_scope': 'unified_action_json_span_only'}
    if run_manifest.get('schema_version') != 'guiagent-qwen35b-action-retention-model-run-v1' or run_manifest.get('role') != ('bare' if step == 0 else 'adapter') or run_manifest.get('base_model') != manifest['model']['base_model_path'] or (run_manifest.get('base_model_fingerprint') != action['base_model_fingerprint']) or (run_manifest.get('production_contract_root') != str(contract_root)) or (run_manifest.get('production_contract_aggregate') != expected_contract_aggregate) or (run_manifest.get('adapter_path') != expected_adapter) or (run_manifest.get('inference_contract') != expected_inference) or (summary.get('status') != 'complete') or (int(summary.get('records', -1)) != 128) or (_count_jsonl(records_path) != 128):
        raise ContractError(f'step-{step} native action-retention run is invalid')
    devices = set((str(item) for item in placement.get('hf_device_map_devices', ())))
    if placement.get('visible_devices') != '0,1' or int(placement.get('torch_device_count', -1)) != 2 or (not {'cuda:0', 'cuda:1'} <= devices) or ('cpu' in devices) or ('disk' in devices):
        raise ContractError(f'step-{step} action scorer did not use balanced two-GPU placement')
    if step > 0:
        comparison_path = output_dir / 'action_comparison.json'
        comparison = load_json(comparison_path)
        base_dir = evaluation_receipt_path(paths, 0).parent / 'action_bare'
        if comparison.get('schema_version') != 'guiagent-qwen35b-action-retention-comparison-v1' or comparison.get('status') != 'passed':
            raise ContractError(f'step-{step} action-retention comparison is invalid')
    return {'run_manifest': run_manifest, 'summary': summary, 'placement': placement}

def validate_posthoc_wrapper_receipts(manifest: Mapping[str, Any], paths: RuntimePaths, checkpoint_dir: Path | None, step: int) -> tuple[dict[str, Any], dict[str, Any]]:
    native_description = validate_native_description_run(manifest, paths, checkpoint_dir, step)
    native_action = validate_native_action_run(manifest, paths, checkpoint_dir, step)
    output_dir = evaluation_receipt_path(paths, step).parent
    description_path = output_dir / 'description_receipt.json'
    action_path = output_dir / 'action_retention_receipt.json'
    description = load_json(description_path)
    action = load_json(action_path)
    action_run_dir = output_dir / ('action_bare' if step == 0 else 'action_adapter')
    expected_mode = 'base_baseline' if step == 0 else 'checkpoint_vs_base'
    expected_description = expected_description_wrapper(manifest, paths, checkpoint_dir, step, native_description)
    action_config = manifest['evaluation']['action_retention']
    probe_manifest = _safe_relative_child(paths.output_root, action_config['bundle_relative_to_output'], label='action-retention bundle') / 'manifest.json'
    expected_action = {'schema_version': 'guiagent-action-retention-gate-wrapper-v1', 'status': 'passed', 'global_step': step, 'mode': expected_mode, 'forgetting_gate': 'baseline_recorded' if step == 0 else 'passed', 'probe_manifest_path': str(probe_manifest), 'probe_manifest_sha256': sha256_file(probe_manifest), 'native_run_manifest_sha256': sha256_file(action_run_dir / 'run_manifest.json'), 'native_records_sha256': sha256_file(action_run_dir / 'records.jsonl'), 'native_summary_sha256': sha256_file(action_run_dir / 'summary.json'), 'native_gpu_placement_sha256': sha256_file(action_run_dir / 'gpu_placement.json'), 'records': native_action['summary']['records'], 'inference_contract': native_action['run_manifest']['inference_contract']}
    if step > 0:
        if checkpoint_dir is None:
            raise ContractError('adapter wrapper validation requires a checkpoint')
        base_output = evaluation_receipt_path(paths, 0).parent
        expected_action.update({'baseline_receipt_sha256': sha256_file(base_output / 'action_retention_receipt.json'), 'comparison_sha256': sha256_file(output_dir / 'action_comparison.json'), 'checkpoint_binding_sha256': sha256_file(checkpoint_dir / 'checkpoint_binding.json'), 'adapter_sha256': _adapter_hash(checkpoint_dir)})
    if description != expected_description:
        raise ContractError(f'step-{step} Description wrapper drifted')
    for (name, expected_value) in expected_action.items():
        if action.get(name) != expected_value:
            raise ContractError(f'step-{step} action-retention wrapper field {name} drifted')
    return (description, action)

def validate_evaluation_receipt(paths: RuntimePaths, checkpoint_dir: Path | None, step: int, preflight: Mapping[str, Any]) -> dict[str, Any]:
    path = evaluation_receipt_path(paths, step)
    if not path.is_file():
        raise ContractError(f'step-{step} evaluation receipt is missing')
    receipt = load_json(path)
    manifest = load_json(paths.manifest_path)
    (description_wrapper, action_wrapper) = validate_posthoc_wrapper_receipts(manifest, paths, checkpoint_dir, step)
    expected = {'status': 'passed', 'global_step': step, 'preflight_receipt_sha256': sha256_file(paths.preflight_receipt), 'description_core100': 'passed', 'description_final500': 'passed' if step == 834 else 'not_required', 'forgetting_gate': 'passed', 'description_receipt_sha256': sha256_file(path.parent / 'description_receipt.json'), 'action_retention_receipt_sha256': sha256_file(path.parent / 'action_retention_receipt.json')}
    if step == 0:
        expected.update({'evaluation_mode': 'bare_base', 'adapter_enabled': False, 'base_model_receipt_sha256': canonical_json_sha256(preflight['model'])})
    else:
        if checkpoint_dir is None:
            raise ContractError('checkpoint evaluation requires a checkpoint directory')
        expected.update({'evaluation_mode': 'adapter_checkpoint_vs_base', 'adapter_enabled': True, 'checkpoint_binding_sha256': sha256_file(checkpoint_dir / 'checkpoint_binding.json'), 'adapter_sha256': _adapter_hash(checkpoint_dir), 'base_evaluation_receipt_sha256': sha256_file(evaluation_receipt_path(paths, 0))})
    for (name, value) in expected.items():
        if receipt.get(name) != value:
            raise ContractError(f'evaluation receipt field {name}={receipt.get(name)!r}, expected {value!r}')
    return receipt

def run_posthoc_gates(manifest: Mapping[str, Any], paths: RuntimePaths, checkpoint_dir: Path | None, step: int, preflight: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = evaluation_receipt_path(paths, step).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = posthoc_command_plan(manifest, paths, checkpoint_dir, step)
    environment = posthoc_environment(paths)
    action = manifest['evaluation']['action_retention']
    probe_dir = _safe_relative_child(paths.output_root, action['bundle_relative_to_output'], label='action-retention bundle')
    probe_manifest = probe_dir / 'manifest.json'
    base_output_dir = evaluation_receipt_path(paths, 0).parent
    base_action_receipt = base_output_dir / 'action_retention_receipt.json'
    if step > 0 and (not evaluation_receipt_path(paths, 0).is_file() or not base_action_receipt.is_file()):
        raise GateFailedError('checkpoint gates require the immutable passed step-0 baseline')
    try:
        if step == 0 and (not probe_manifest.is_file()):
            run_checked(plan['action_build'], cwd=paths.project_root, log_path=output_dir / 'action_probe_builder.log', env=environment)
        if not probe_manifest.is_file():
            raise ContractError('the immutable action-retention probe bundle is missing')
        run_checked(plan['action_score'], cwd=paths.project_root, log_path=output_dir / 'action_score.log', env=environment)
        if step > 0:
            run_checked(plan['action_compare'], cwd=paths.project_root, log_path=output_dir / 'action_compare.log', env=environment)
        description_outputs = ('selection_receipt.json', 'evaluation_subset.jsonl', 'predictions.jsonl', 'metrics.json', 'run_config.json', 'length_validation.json')
        description_dir = output_dir / ('description_bare' if step == 0 else 'description_adapter')
        if description_dir.is_dir() and any(description_dir.iterdir()):
            missing = [name for name in description_outputs if not (description_dir / name).is_file()]
            if missing:
                raise ContractError(f'partial Description core100 output cannot be resumed safely: {missing}')
        else:
            run_checked(plan['description_run'], cwd=paths.project_root, log_path=output_dir / 'description_run.log', env=environment)
        run_checked(plan['description_verify'], cwd=paths.project_root, log_path=output_dir / 'description_verify.log', env=environment)
        if step > 0:
            run_checked(plan['description_compare'], cwd=paths.project_root, log_path=output_dir / 'description_compare.log', env=environment)
        if 'final500' in description_scopes_for_step(step):
            final_dir = output_dir / 'description_final500_adapter'
            if final_dir.is_dir() and any(final_dir.iterdir()):
                missing = [name for name in description_outputs if not (final_dir / name).is_file()]
                if missing:
                    raise ContractError(f'partial Description final500 output cannot be resumed safely: {missing}')
            else:
                run_checked(plan['description_final500_run'], cwd=paths.project_root, log_path=output_dir / 'description_final500_run.log', env=environment)
            run_checked(plan['description_final500_verify'], cwd=paths.project_root, log_path=output_dir / 'description_final500_verify.log', env=environment)
    except ContractError as exc:
        raise GateFailedError(str(exc)) from exc
    try:
        native_action = validate_native_action_run(manifest, paths, checkpoint_dir, step)
        native_description = validate_native_description_run(manifest, paths, checkpoint_dir, step)
    except ContractError as exc:
        raise GateFailedError(str(exc)) from exc
    action_run_dir = output_dir / ('action_bare' if step == 0 else 'action_adapter')
    description_wrapper = expected_description_wrapper(manifest, paths, checkpoint_dir, step, native_description)
    action_wrapper = {'schema_version': 'guiagent-action-retention-gate-wrapper-v1', 'status': 'passed', 'global_step': step, 'mode': 'base_baseline' if step == 0 else 'checkpoint_vs_base', 'forgetting_gate': 'baseline_recorded' if step == 0 else 'passed', 'probe_manifest_path': str(probe_manifest), 'probe_manifest_sha256': sha256_file(probe_manifest), 'native_run_manifest_sha256': sha256_file(action_run_dir / 'run_manifest.json'), 'native_records_sha256': sha256_file(action_run_dir / 'records.jsonl'), 'native_summary_sha256': sha256_file(action_run_dir / 'summary.json'), 'native_gpu_placement_sha256': sha256_file(action_run_dir / 'gpu_placement.json'), 'records': native_action['summary']['records'], 'inference_contract': native_action['run_manifest']['inference_contract']}
    if step > 0:
        if checkpoint_dir is None:
            raise AssertionError('adapter checkpoint disappeared')
        action_wrapper.update({'baseline_receipt_sha256': sha256_file(base_action_receipt), 'comparison_sha256': sha256_file(output_dir / 'action_comparison.json'), 'checkpoint_binding_sha256': sha256_file(checkpoint_dir / 'checkpoint_binding.json'), 'adapter_sha256': _adapter_hash(checkpoint_dir)})
    atomic_write_json(output_dir / 'description_receipt.json', description_wrapper)
    atomic_write_json(output_dir / 'action_retention_receipt.json', action_wrapper)
    receipt = {'schema_version': 'guiagent-posthoc-evaluation-receipt-v1', 'status': 'passed', 'global_step': step, 'preflight_receipt_sha256': sha256_file(paths.preflight_receipt), 'evaluation_mode': 'bare_base' if step == 0 else 'adapter_checkpoint_vs_base', 'adapter_enabled': step > 0, 'description_core100': 'passed', 'description_final500': 'passed' if step == 834 else 'not_required', 'forgetting_gate': 'passed', 'description_receipt_sha256': sha256_file(output_dir / 'description_receipt.json'), 'action_retention_receipt_sha256': sha256_file(output_dir / 'action_retention_receipt.json')}
    if step == 0:
        receipt['base_model_receipt_sha256'] = canonical_json_sha256(preflight['model'])
    else:
        if checkpoint_dir is None:
            raise AssertionError('checkpoint directory disappeared')
        receipt.update({'checkpoint_binding_sha256': sha256_file(checkpoint_dir / 'checkpoint_binding.json'), 'adapter_sha256': _adapter_hash(checkpoint_dir), 'base_evaluation_receipt_sha256': sha256_file(evaluation_receipt_path(paths, 0))})
    atomic_write_json(evaluation_receipt_path(paths, step), receipt)
    return receipt

def newest_contiguous_step(paths: RuntimePaths, preflight: Mapping[str, Any], resource_sha: str) -> int:
    current = 0
    for step in CHECKPOINT_STEPS:
        checkpoint_dir = paths.output_root / f'checkpoints/checkpoint-{step}'
        if not checkpoint_dir.is_dir():
            break
        validate_checkpoint(checkpoint_dir, step=step, preflight=preflight, resource_sha256=resource_sha)
        if not evaluation_receipt_path(paths, step).is_file():
            break
        validate_evaluation_receipt(paths, checkpoint_dir, step, preflight)
        current = step
    return current

def next_gate_after(step: int) -> int:
    for candidate in CHECKPOINT_STEPS:
        if candidate > step:
            return candidate
    raise ContractError(f'no checkpoint gate follows step {step}')

def finalize_formal_run(paths: RuntimePaths, preflight: Mapping[str, Any]) -> dict[str, Any]:
    target_dir = paths.output_root / 'checkpoints/checkpoint-834'
    final_dir = paths.output_root / 'final_adapter'
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        source = target_dir / name
        destination = final_dir / name
        if destination.exists():
            raise ContractError(f'final adapter artifact already exists with different content: {destination}')
        if not destination.exists():
            os.link(source, destination)
    completion = {'schema_version': 'guiagent-v3-cur-formal-receipt-v1', 'status': 'complete', 'global_step': 834, 'manifest_sha256': preflight['manifest_sha256'], 'certification_receipt_sha256': sha256_file(paths.certification_receipt), 'checkpoint_binding_sha256s': {str(step): sha256_file(paths.output_root / f'checkpoints/checkpoint-{step}/checkpoint_binding.json') for step in CHECKPOINT_STEPS}, 'evaluation_receipt_sha256s': {str(step): sha256_file(evaluation_receipt_path(paths, step)) for step in (0, *CHECKPOINT_STEPS)}, 'final_adapter_sha256': sha256_file(final_dir / 'adapter_model.safetensors')}
    atomic_write_json(paths.output_root / 'formal_receipt.json', completion)
    transition(paths, 'COMPLETE', step=834)
    return completion

def run_formal_training(manifest_path: Path) -> dict[str, Any]:
    (manifest, _resource, paths) = runtime_paths(manifest_path)
    with bounded_lock(paths.lock_path):
        preflight = validate_preflight_receipt(manifest, paths)
        validate_certification(preflight, paths)
        resource_sha = sha256_file(paths.resource_path)
        base_receipt_path = evaluation_receipt_path(paths, 0)
        state = read_state(paths)
        if state['phase'] in {'CERTIFIED', 'EVAL_BLOCKED'} and (not base_receipt_path.is_file()):
            transition(paths, 'BASE_EVAL_RUNNING', step=0)
        elif state['phase'] == 'CERTIFIED' and base_receipt_path.is_file():
            transition(paths, 'BASE_EVAL_RUNNING', step=0, receipt_reused=True)
        elif state['phase'] == 'EVAL_BLOCKED' and int(state.get('step', -1)) == 0 and base_receipt_path.is_file():
            transition(paths, 'BASE_EVAL_RUNNING', step=0, receipt_reused=True)
        if not base_receipt_path.is_file():
            try:
                run_posthoc_gates(manifest, paths, None, 0, preflight)
            except BlockedError:
                transition(paths, 'EVAL_BLOCKED', step=0, reason='base_gate_commands_missing')
                raise
            except GateFailedError:
                transition(paths, 'EVAL_BLOCKED', step=0, reason='base_validation_failed')
                raise
        try:
            validate_evaluation_receipt(paths, None, 0, preflight)
        except ContractError:
            if read_state(paths)['phase'] == 'BASE_EVAL_RUNNING':
                transition(paths, 'EVAL_BLOCKED', step=0, reason='base_receipt_validation_failed')
            raise
        if read_state(paths)['phase'] == 'BASE_EVAL_RUNNING':
            transition(paths, 'BASE_EVAL_PASSED', step=0)
        current = newest_contiguous_step(paths, preflight, resource_sha)
        state = read_state(paths)
        if state['phase'] not in {'BASE_EVAL_PASSED', 'RESUME_READY', 'EVAL_PASSED', 'FORMAL_SEGMENT_RUNNING', 'CHECKPOINT_COMMITTED', 'TRAIN_EXITED', 'EVAL_BLOCKED'}:
            raise ContractError(f"formal training cannot start from phase {state['phase']}")
        if current == 834:
            if state['phase'] == 'EVAL_PASSED':
                transition(paths, 'RESUME_READY', step=834, finalization_recovery=True)
            if read_state(paths)['phase'] != 'RESUME_READY':
                raise ContractError('step-834 finalization recovery requires RESUME_READY')
            return finalize_formal_run(paths, preflight)
        while current < 834:
            target = next_gate_after(current)
            target_dir = paths.output_root / f'checkpoints/checkpoint-{target}'
            committed = target_dir.is_dir() and (target_dir / 'checkpoint_binding.json').is_file()
            if committed:
                validate_checkpoint(target_dir, step=target, preflight=preflight, resource_sha256=resource_sha)
            else:
                phase = read_state(paths)['phase']
                if phase == 'EVAL_PASSED':
                    transition(paths, 'RESUME_READY', step=current)
                    phase = 'RESUME_READY'
                if phase in {'BASE_EVAL_PASSED', 'RESUME_READY'}:
                    transition(paths, 'FORMAL_SEGMENT_RUNNING', step=current, segment_end_step=target)
                elif phase != 'FORMAL_SEGMENT_RUNNING':
                    raise ContractError(f'cannot run segment {current}->{target} from phase {phase}')
                resume_from = paths.output_root / f'checkpoints/checkpoint-{current}' if current else None
                command = build_torchrun_command(paths, output_root=paths.output_root, start_step=current, end_step=target, resume_from=resume_from, preflight=preflight)
                run_checked(command, cwd=paths.project_root, log_path=paths.log_root / f'formal_segment_{current:06d}_{target:06d}.log', env={'CUDA_VISIBLE_DEVICES': '0,1', 'PYTHONPATH': str(paths.project_root / 'src')})
                validate_checkpoint(target_dir, step=target, preflight=preflight, resource_sha256=resource_sha)
                transition(paths, 'CHECKPOINT_COMMITTED', step=target)
            phase = read_state(paths)['phase']
            if committed and phase in {'BASE_EVAL_PASSED', 'RESUME_READY'}:
                transition(paths, 'FORMAL_SEGMENT_RUNNING', step=current, segment_end_step=target, receipt_reused=True)
                transition(paths, 'CHECKPOINT_COMMITTED', step=target, receipt_reused=True)
                phase = 'CHECKPOINT_COMMITTED'
            elif committed and phase == 'FORMAL_SEGMENT_RUNNING':
                transition(paths, 'CHECKPOINT_COMMITTED', step=target, receipt_reused=True)
                phase = 'CHECKPOINT_COMMITTED'
            if phase == 'CHECKPOINT_COMMITTED':
                transition(paths, 'TRAIN_EXITED', step=target)
            if evaluation_receipt_path(paths, target).is_file():
                phase = read_state(paths)['phase']
                if phase in {'TRAIN_EXITED', 'EVAL_BLOCKED'}:
                    transition(paths, 'POSTHOC_EVAL_RUNNING', step=target, receipt_reused=True)
                try:
                    validate_evaluation_receipt(paths, target_dir, target, preflight)
                except ContractError:
                    if read_state(paths)['phase'] == 'POSTHOC_EVAL_RUNNING':
                        transition(paths, 'EVAL_BLOCKED', step=target, reason='checkpoint_receipt_validation_failed')
                    raise
                if read_state(paths)['phase'] == 'POSTHOC_EVAL_RUNNING':
                    transition(paths, 'EVAL_PASSED', step=target, receipt_reused=True)
            else:
                transition(paths, 'POSTHOC_EVAL_RUNNING', step=target)
                try:
                    run_posthoc_gates(manifest, paths, target_dir, target, preflight)
                except BlockedError:
                    transition(paths, 'EVAL_BLOCKED', step=target, reason='checkpoint_gate_commands_missing')
                    raise
                except GateFailedError:
                    transition(paths, 'EVAL_BLOCKED', step=target, reason='checkpoint_validation_failed')
                    raise
                transition(paths, 'EVAL_PASSED', step=target)
            current = target
            if current == 834:
                return finalize_formal_run(paths, preflight)
            transition(paths, 'RESUME_READY', step=current)
        raise AssertionError('formal loop exited without completion')

def tmux_has_session(name: str) -> bool:
    try:
        completed = subprocess.run([str(PERSISTENT_TMUX_PATH), 'has-session', '-t', name], capture_output=True, check=False, text=True)
    except OSError as exc:
        raise ContractError(f'cannot execute persistent tmux: {PERSISTENT_TMUX_PATH}') from exc
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = (completed.stderr or completed.stdout).strip()
    raise ContractError(f'persistent tmux has-session failed with rc={completed.returncode}: {detail}')

def tmux_names(manifest: Mapping[str, Any]) -> tuple[str, str]:
    token = re.sub('[^A-Za-z0-9_-]+', '_', manifest['identity']['run_id'])
    return (f'{token}_formal', f'{token}_watchdog')

def quarantine_incomplete_description_runs(*, output_root: Path, recovery_root: Path, steps: Sequence[int]) -> list[dict[str, Any]]:
    required = ('selection_receipt.json', 'evaluation_subset.jsonl', 'predictions.jsonl', 'metrics.json', 'run_config.json', 'length_validation.json')
    quarantined: list[dict[str, Any]] = []
    for step in steps:
        validation_dir = output_root / f'validation/step_{step:06d}'
        outer_receipt = validation_dir / 'evaluation_receipt.json'
        if outer_receipt.is_file() and load_json(outer_receipt).get('status') == 'passed':
            continue
        runs = [('core100', 'description_bare' if step == 0 else 'description_adapter')]
        if 'final500' in description_scopes_for_step(step):
            runs.append(('final500', 'description_final500_adapter'))
        for (scope, name) in runs:
            source = validation_dir / name
            if not source.is_dir() or not any(source.iterdir()):
                continue
            missing = [filename for filename in required if not (source / filename).is_file()]
            if not missing:
                continue
            destination = recovery_root / 'description_runs' / f'step_{step:06d}' / name
            if destination.exists():
                raise ContractError(f'description recovery quarantine already exists: {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            quarantined.append({'step': step, 'scope': scope, 'source': str(source), 'destination': str(destination), 'missing_core_outputs': missing})
    return quarantined

def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()

def formal_artifact_paths(paths: RuntimePaths) -> list[str]:
    artifacts: list[Path] = []
    checkpoints = paths.output_root / 'checkpoints'
    if _path_present(checkpoints):
        if checkpoints.is_symlink() or not checkpoints.is_dir():
            artifacts.append(checkpoints)
        else:
            artifacts.extend(sorted(checkpoints.iterdir()))
    validation = paths.output_root / 'validation'
    if _path_present(validation):
        if validation.is_symlink() or not validation.is_dir():
            artifacts.append(validation)
        else:
            artifacts.extend((child for child in sorted(validation.iterdir()) if child.name != '_shared'))
    for path in (paths.output_root / 'formal_receipt.json', paths.output_root / 'final_adapter'):
        if _path_present(path):
            artifacts.append(path)
    return [str(path) for path in artifacts]

def require_no_formal_artifacts(paths: RuntimePaths, *, mode: str) -> None:
    artifacts = formal_artifact_paths(paths)
    if artifacts:
        raise ContractError(f'{mode} recovery requires no formal checkpoint/evaluation artifacts: {artifacts}')

def certification_quarantine_plan(certification_root: Path, recovery_root: Path) -> dict[str, Any] | None:
    """Validate and fingerprint a disposable Tier2 tree before any move."""
    if not _path_present(certification_root):
        return None
    if certification_root.is_symlink() or not certification_root.is_dir():
        raise ContractError(f'certification root is not a regular directory: {certification_root}')
    destination = recovery_root / 'certification'
    if _path_present(destination):
        raise ContractError(f'certification recovery quarantine already exists: {destination}')
    files = recursive_file_receipt(certification_root)
    tree_receipt = {'algorithm': 'sha256(canonical_json(recursive_file_receipt))', 'file_count': len(files), 'total_bytes': sum((int(item['size_bytes']) for item in files)), 'sha256': canonical_json_sha256(files)}
    return {'source': str(certification_root), 'destination': str(destination), 'tree': tree_receipt}

def execute_certification_quarantine(plan: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    source = Path(str(plan['source']))
    destination = Path(str(plan['destination']))
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    return dict(plan)

def quarantine_certification_root(certification_root: Path, recovery_root: Path) -> dict[str, Any] | None:
    return execute_certification_quarantine(certification_quarantine_plan(certification_root, recovery_root))

def quarantine_preflight_receipt(preflight_receipt: Path, recovery_root: Path) -> dict[str, Any]:
    """Atomically archive one obsolete, previously passed preflight receipt."""
    try:
        metadata = preflight_receipt.lstat()
    except FileNotFoundError as exc:
        raise ContractError('stale-preflight recovery requires the old receipt') from exc
    if preflight_receipt.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ContractError('stale preflight receipt must be a regular non-symlink file')
    destination = recovery_root / 'preflight_receipt.json'
    if _path_present(destination):
        raise ContractError(f'preflight recovery quarantine already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(preflight_receipt)
    size_bytes = int(metadata.st_size)
    os.replace(preflight_receipt, destination)
    return {'source': str(preflight_receipt), 'destination': str(destination), 'sha256': digest, 'size_bytes': size_bytes}

def should_transition_failure(action: str, exc: BaseException, phase: str) -> bool:
    """Keep diagnostic/preflight failures read-only and fail only active managed work."""
    if isinstance(exc, LockBusyError) or phase in TERMINAL_PHASES | {'EVAL_BLOCKED'}:
        return False
    if action in {'status', 'preflight'}:
        return False
    if action == 'certify' and phase in {'NEW', 'PREFLIGHT_PASSED'}:
        return False
    return 'FAILED' in ALLOWED_TRANSITIONS.get(phase, set())

def _recovery_original(state: Mapping[str, Any]) -> dict[str, Any]:
    return {'phase': state.get('phase'), 'revision': int(state.get('revision', 0)), 'reason': state.get('reason'), 'step': int(state.get('step', 0))}

def _recovery_receipt_path(paths: RuntimePaths, original: Mapping[str, Any]) -> Path:
    return paths.log_root / 'recoveries' / f"recovery_from_revision_{int(original['revision']):06d}.json"

def _ledger_binding(paths: RuntimePaths) -> dict[str, Any]:
    if not paths.ledger_path.is_file() or paths.ledger_path.is_symlink():
        return {'path': str(paths.ledger_path), 'exists': False}
    return {'path': str(paths.ledger_path), 'exists': True, 'sha256_before_recovery': sha256_file(paths.ledger_path), 'size_bytes_before_recovery': paths.ledger_path.stat().st_size}

def _complete_preformal_recovery(*, manifest: Mapping[str, Any], paths: RuntimePaths, original: Mapping[str, Any], formal_name: str, watchdog_name: str, mode: str, target_phase: str, implementation: Mapping[str, str], preflight_sha256: str | None, quarantined_certification: Mapping[str, Any] | None, quarantined_preflight: Mapping[str, Any] | None=None, stale_preflight_binding: Mapping[str, Any] | None=None, preformal_config_revision_allowed: bool=False) -> dict[str, Any]:
    receipt_path = _recovery_receipt_path(paths, original)
    receipt = {'schema_version': 'guiagent-v3-cur-recovery-v2', 'status': 'validated', 'mode': mode, 'run_id': manifest['identity']['run_id'], 'recovered_from': dict(original), 'manifest_sha256': sha256_file(paths.manifest_path), 'resource_profile_sha256': sha256_file(paths.resource_path), 'preflight_receipt_sha256': preflight_sha256, 'certification_receipt_sha256': None, 'implementation_sha256s': dict(implementation), 'sessions_confirmed_absent': [formal_name, watchdog_name], 'ledger_preserved': _ledger_binding(paths), 'quarantined_preflight_receipt': dict(quarantined_preflight) if quarantined_preflight is not None else None, 'quarantined_certification_root': quarantined_certification, 'stale_preflight_binding': dict(stale_preflight_binding) if stale_preflight_binding is not None else None, 'recovery_target_phase': target_phase, 'auto_launch': False, 'preformal_config_revision_allowed': preformal_config_revision_allowed, 'config_change_allowed': False, 'quality_gate_bypass_allowed': False}
    atomic_write_json(receipt_path, receipt)
    transition(paths, 'RECOVERING', step=0, recovered_from_revision=original['revision'], recovery_receipt=str(receipt_path))
    recovered_state = transition(paths, target_phase, step=0, reason=None, recovered_from_phase=original['phase'], recovered_from_reason=original['reason'])
    receipt.update({'status': 'recovered', 'recovered_state_revision': recovered_state['revision']})
    atomic_write_json(receipt_path, receipt)
    return {'status': 'recovered_to_new' if target_phase == 'NEW' else 'recovered_to_preflight_passed', 'recovery_receipt': str(receipt_path), 'auto_launch': False}

def recover(manifest_path: Path) -> dict[str, Any]:
    (manifest, resource, paths) = runtime_paths(manifest_path)
    (formal_name, watchdog_name) = tmux_names(manifest)
    if tmux_has_session(formal_name) or tmux_has_session(watchdog_name):
        raise ContractError('recover requires both exact managed tmux sessions to be absent')
    state = read_state(paths)
    if state.get('phase') not in {'HALTED', 'FAILED', 'EVAL_BLOCKED'}:
        raise ContractError(f"recover is allowed only from HALTED/FAILED/EVAL_BLOCKED, not {state.get('phase')!r}")
    original = _recovery_original(state)
    recovery_root = paths.output_root / 'recovery_quarantine' / f"from_revision_{original['revision']:06d}"
    for step in (0, *CHECKPOINT_STEPS):
        validation_dir = evaluation_receipt_path(paths, step).parent
        for name in ('action_comparison.json', 'description_comparison.json'):
            comparison = validation_dir / name
            if comparison.is_file():
                report = load_json(comparison)
                if report.get('status') == 'failed':
                    raise ContractError(f'recover refuses to bypass a failed quality gate: {comparison}')
                if report.get('status') != 'passed':
                    raise ContractError(f'recover refuses a malformed quality comparison: {comparison}')
    preflight_present = _path_present(paths.preflight_receipt)
    certification_present = _path_present(paths.certification_receipt)
    if state.get('phase') == 'EVAL_BLOCKED' and (not preflight_present or not certification_present):
        raise ContractError('EVAL_BLOCKED recovery requires passed preflight and certification receipts')
    if not preflight_present:
        require_no_formal_artifacts(paths, mode='bootstrap')
        if certification_present or _path_present(paths.certification_root):
            raise ContractError('bootstrap recovery requires no preflight or certification artifacts')
        validate_manifest_contract(manifest, resource)
        return _complete_preformal_recovery(manifest=manifest, paths=paths, original=original, formal_name=formal_name, watchdog_name=watchdog_name, mode='bootstrap_without_preflight', target_phase='NEW', implementation=implementation_hashes(paths), preflight_sha256=None, quarantined_certification=None)
    if state.get('phase') == 'FAILED' and (not certification_present):
        if paths.preflight_receipt.is_symlink() or not paths.preflight_receipt.is_file():
            raise ContractError('stale preflight receipt must be a regular non-symlink file')
        stale_preflight = load_json(paths.preflight_receipt)
        stale_implementation = stale_preflight.get('implementation_sha256s')
        valid_old_binding = stale_preflight.get('status') == 'passed' and re.fullmatch('[0-9a-f]{64}', str(stale_preflight.get('manifest_sha256', ''))) is not None and (re.fullmatch('[0-9a-f]{64}', str(stale_preflight.get('resource_profile_sha256', ''))) is not None) and isinstance(stale_implementation, Mapping) and bool(stale_implementation) and all((isinstance(name, str) and re.fullmatch('[0-9a-f]{64}', str(digest)) is not None for (name, digest) in stale_implementation.items()))
        if valid_old_binding:
            validate_manifest_contract(manifest, resource)
            current_resource_sha256 = sha256_file(paths.resource_path)
            current_implementation = implementation_hashes(paths)
            current_manifest_sha256 = sha256_file(paths.manifest_path)
            config_binding_changed = stale_preflight['manifest_sha256'] != current_manifest_sha256 or stale_preflight['resource_profile_sha256'] != current_resource_sha256
            if config_binding_changed:
                require_no_formal_artifacts(paths, mode='stale-preflight-config-revision')
                certification_plan = certification_quarantine_plan(paths.certification_root, recovery_root)
                quarantined_preflight = quarantine_preflight_receipt(paths.preflight_receipt, recovery_root)
                try:
                    quarantined_certification = execute_certification_quarantine(certification_plan)
                except OSError as exc:
                    archived_preflight = Path(str(quarantined_preflight['destination']))
                    try:
                        os.replace(archived_preflight, paths.preflight_receipt)
                    except OSError as rollback_exc:
                        raise ContractError('certification quarantine failed and preflight rollback failed') from rollback_exc
                    raise ContractError('certification quarantine failed; old preflight was restored') from exc
                stale_binding = {'manifest_sha256': stale_preflight['manifest_sha256'], 'resource_profile_sha256': stale_preflight['resource_profile_sha256'], 'implementation_sha256s': dict(stale_implementation)}
                return _complete_preformal_recovery(manifest=manifest, paths=paths, original=original, formal_name=formal_name, watchdog_name=watchdog_name, mode='preformal_stale_preflight_config_revision', target_phase='NEW', implementation=current_implementation, preflight_sha256=quarantined_preflight['sha256'], quarantined_certification=quarantined_certification, quarantined_preflight=quarantined_preflight, stale_preflight_binding=stale_binding, preformal_config_revision_allowed=True)
    preflight = validate_preflight_receipt(manifest, paths)
    if not certification_present:
        require_no_formal_artifacts(paths, mode='certification')
        quarantined_certification = quarantine_certification_root(paths.certification_root, recovery_root)
        return _complete_preformal_recovery(manifest=manifest, paths=paths, original=original, formal_name=formal_name, watchdog_name=watchdog_name, mode='certification_retry', target_phase='PREFLIGHT_PASSED', implementation=preflight['implementation_sha256s'], preflight_sha256=sha256_file(paths.preflight_receipt), quarantined_certification=quarantined_certification)
    validate_certification(preflight, paths)
    resource_sha = sha256_file(paths.resource_path)
    committed_steps: list[int] = []
    for step in CHECKPOINT_STEPS:
        checkpoint = paths.output_root / f'checkpoints/checkpoint-{step}'
        if (checkpoint / 'checkpoint_binding.json').is_file():
            validate_checkpoint(checkpoint, step=step, preflight=preflight, resource_sha256=resource_sha)
            committed_steps.append(step)
    if committed_steps != list(CHECKPOINT_STEPS[:len(committed_steps)]):
        raise ContractError(f'committed checkpoint gates are not a contiguous prefix: {committed_steps}')
    quarantined_descriptions = quarantine_incomplete_description_runs(output_root=paths.output_root, recovery_root=recovery_root, steps=(0, *committed_steps))
    base_exists = evaluation_receipt_path(paths, 0).is_file()
    if base_exists:
        validate_evaluation_receipt(paths, None, 0, preflight)
    if committed_steps and (not base_exists):
        raise ContractError('committed formal checkpoints exist without the passed step-0 baseline')
    evaluated_steps: list[int] = []
    for step in CHECKPOINT_STEPS:
        receipt_path = evaluation_receipt_path(paths, step)
        if receipt_path.is_file():
            checkpoint = paths.output_root / f'checkpoints/checkpoint-{step}'
            if step not in committed_steps:
                raise ContractError(f'step-{step} evaluation exists without a committed checkpoint')
            validate_evaluation_receipt(paths, checkpoint, step, preflight)
            evaluated_steps.append(step)
    if evaluated_steps != committed_steps[:len(evaluated_steps)]:
        raise ContractError(f'evaluated checkpoint gates are not a contiguous prefix: {evaluated_steps}')
    if len(committed_steps) - len(evaluated_steps) > 1:
        raise ContractError('more than one committed checkpoint is awaiting its blocking evaluation')
    current = evaluated_steps[-1] if evaluated_steps else 0
    quarantined: list[dict[str, str]] = []
    for step in CHECKPOINT_STEPS:
        checkpoint = paths.output_root / f'checkpoints/checkpoint-{step}'
        if checkpoint.is_dir() and (not (checkpoint / 'checkpoint_binding.json').is_file()):
            destination = recovery_root / checkpoint.name
            if destination.exists():
                raise ContractError(f'recovery quarantine target already exists: {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(checkpoint, destination)
            quarantined.append({'source': str(checkpoint), 'destination': str(destination)})
    if not base_exists:
        target_phase = 'CERTIFIED'
    elif current == 0 and (not committed_steps):
        target_phase = 'BASE_EVAL_PASSED'
    else:
        target_phase = 'RESUME_READY'
    receipt_path = _recovery_receipt_path(paths, original)
    receipt = {'schema_version': 'guiagent-v3-cur-recovery-v2', 'status': 'validated', 'mode': 'formal_certified_prefix', 'run_id': manifest['identity']['run_id'], 'recovered_from': original, 'manifest_sha256': preflight['manifest_sha256'], 'preflight_receipt_sha256': sha256_file(paths.preflight_receipt), 'certification_receipt_sha256': sha256_file(paths.certification_receipt), 'implementation_sha256s': preflight['implementation_sha256s'], 'sessions_confirmed_absent': [formal_name, watchdog_name], 'latest_contiguous_evaluated_step': current, 'committed_steps': committed_steps, 'evaluated_steps': evaluated_steps, 'quarantined_uncommitted_checkpoints': quarantined, 'quarantined_incomplete_description_runs': quarantined_descriptions, 'recovery_target_phase': target_phase, 'auto_launch': True, 'config_change_allowed': False, 'quality_gate_bypass_allowed': False}
    atomic_write_json(receipt_path, receipt)
    transition(paths, 'RECOVERING', step=current, recovered_from_revision=original['revision'], recovery_receipt=str(receipt_path))
    transition(paths, target_phase, step=current, reason=None, recovered_from_phase=original['phase'], recovered_from_reason=original['reason'])
    launched = launch(manifest_path)
    receipt.update({'status': 'relaunched', 'managed_token': launched['managed_token'], 'formal_tmux': launched['formal_tmux'], 'watchdog_tmux': launched['watchdog_tmux']})
    atomic_write_json(receipt_path, receipt)
    return {**launched, 'status': 'recovered_and_relaunched', 'recovery_receipt': str(receipt_path)}

def launch(manifest_path: Path) -> dict[str, Any]:
    (manifest, _resource, paths) = runtime_paths(manifest_path)
    preflight = validate_preflight_receipt(manifest, paths)
    validate_certification(preflight, paths)
    gpu = gpu_receipt()
    (formal_name, watchdog_name) = tmux_names(manifest)
    if tmux_has_session(formal_name) or tmux_has_session(watchdog_name):
        raise ContractError('formal or watchdog tmux session already exists')
    owner = {'run_id': manifest['identity']['run_id'], 'managed_token': str(uuid.uuid4()), 'created_at_epoch': time.time(), 'manifest_sha256': sha256_file(paths.manifest_path)}
    with nonblocking_lock(paths.lock_path):
        atomic_write_json(paths.owner_path, owner)
        train_command = shlex.join([str(paths.runner_path), 'train', '--manifest', str(paths.manifest_path)])
        watchdog_command = shlex.join([sys.executable, str(paths.runtime_path), '__watchdog', '--manifest', str(paths.manifest_path), '--formal-session', formal_name])
        subprocess.run([str(PERSISTENT_TMUX_PATH), 'new-session', '-d', '-s', formal_name, train_command], check=True)
        try:
            subprocess.run([str(PERSISTENT_TMUX_PATH), 'new-session', '-d', '-s', watchdog_name, watchdog_command], check=True)
        except BaseException:
            subprocess.run([str(PERSISTENT_TMUX_PATH), 'kill-session', '-t', formal_name], check=False)
            raise
    return {'status': 'launched', 'formal_tmux': formal_name, 'watchdog_tmux': watchdog_name, 'managed_token': owner['managed_token'], 'gpu': gpu}

def watchdog(manifest_path: Path, formal_session: str) -> None:
    (manifest, _resource, paths) = runtime_paths(manifest_path)
    owner = load_json(paths.owner_path)
    expected_run = manifest['identity']['run_id']
    if owner.get('run_id') != expected_run:
        raise ContractError('watchdog owner token does not match the immutable run')
    while True:
        state = read_state(paths)
        if state['phase'] in WATCHDOG_STOP_PHASES:
            return
        if not tmux_has_session(formal_session):
            command = shlex.join([str(paths.runner_path), 'train', '--manifest', str(paths.manifest_path)])
            subprocess.run([str(PERSISTENT_TMUX_PATH), 'new-session', '-d', '-s', formal_session, command], check=True)
            append_jsonl(paths.ledger_path, {'event': 'watchdog_restarted_missing_formal_session', 'phase': state['phase'], 'step': state.get('step', 0), 'managed_token': owner['managed_token'], 'updated_at_epoch': time.time()})
        time.sleep(60)

def status(manifest_path: Path) -> dict[str, Any]:
    (manifest, _resource, paths) = runtime_paths(manifest_path)
    (formal_name, watchdog_name) = tmux_names(manifest)
    state = read_state(paths)
    checkpoints = []
    for step in CHECKPOINT_STEPS:
        checkpoint = paths.output_root / f'checkpoints/checkpoint-{step}'
        checkpoints.append({'step': step, 'committed': (checkpoint / 'checkpoint_binding.json').is_file(), 'evaluation_passed': evaluation_receipt_path(paths, step).is_file()})
    return {'run_id': manifest['identity']['run_id'], 'phase': state.get('phase'), 'step': state.get('step', 0), 'revision': state.get('revision', 0), 'preflight_passed': paths.preflight_receipt.is_file(), 'certified': paths.certification_receipt.is_file(), 'formal_tmux_alive': tmux_has_session(formal_name), 'watchdog_tmux_alive': tmux_has_session(watchdog_name), 'base_step_0_evaluation_passed': evaluation_receipt_path(paths, 0).is_file(), 'checkpoints': checkpoints, 'formal_receipt': str(paths.output_root / 'formal_receipt.json')}

def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=(*PUBLIC_ACTIONS, '__watchdog'))
    parser.add_argument('--manifest', type=Path, default=default_manifest_path())
    parser.add_argument('--formal-session')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> None:
    args = parse_args(argv)
    try:
        if args.action == 'preflight':
            value = run_preflight(args.manifest, require_gpu=True)
        elif args.action == 'certify':
            value = run_certification(args.manifest)
        elif args.action == 'train':
            value = run_formal_training(args.manifest)
        elif args.action == 'launch':
            value = launch(args.manifest)
        elif args.action == 'recover':
            value = recover(args.manifest)
        elif args.action == 'status':
            value = status(args.manifest)
        else:
            if not args.formal_session:
                raise ContractError('__watchdog requires --formal-session')
            watchdog(args.manifest, args.formal_session)
            value = {'status': 'watchdog_complete'}
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    except BaseException as exc:
        try:
            (_manifest, _resource, paths) = runtime_paths(args.manifest)
            current = read_state(paths)
            if should_transition_failure(args.action, exc, str(current.get('phase', 'NEW'))):
                transition(paths, 'FAILED', reason=f'{type(exc).__name__}: {exc}')
        except BaseException:
            pass
        raise
if __name__ == '__main__':
    main()
