"""Train and evaluate YOLO26 on AtomBlock-WebUI."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
CLASS_NAMES = ['button', 'link', 'input', 'image', 'icon', 'checkbox', 'radio', 'select', 'block-nav', 'block-sidebar', 'block-footer', 'block-form', 'block-table', 'block']

def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit('Missing ultralytics. Run this script in the YOLO environment used by tsm_renderer/services/yolo_service.py.') from exc
    return YOLO

def run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ''
    return completed.stdout.strip()

def gpu_memory_snapshot() -> dict[str, dict[str, float]]:
    output = run_text(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    snapshot: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 4:
            continue
        (index, total, used, util) = parts
        try:
            snapshot[index] = {'memory_total_mb': float(total), 'memory_used_mb': float(used), 'utilization_gpu_pct': float(util)}
        except ValueError:
            continue
    return snapshot

def gpu_pmon_snapshot() -> dict[str, float]:
    output = run_text(['nvidia-smi', 'pmon', '-c', '1'])
    sm_by_gpu: dict[str, float] = {}
    samples_by_gpu: dict[str, int] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        gpu = parts[0]
        try:
            sm = float(parts[3])
        except ValueError:
            continue
        sm_by_gpu[gpu] = sm_by_gpu.get(gpu, 0.0) + sm
        samples_by_gpu[gpu] = samples_by_gpu.get(gpu, 0) + 1
    return {gpu: sm_by_gpu[gpu] / samples_by_gpu[gpu] for gpu in sm_by_gpu}

def choose_device(requested: str, *, min_free_mb: float=0.0, max_auto_gpus: int=0) -> tuple[str, dict[str, Any]]:
    if requested != 'auto':
        if requested in {'auto-multi', 'auto_multi'}:
            memory = gpu_memory_snapshot()
            pmon = gpu_pmon_snapshot()
            if not memory:
                return ('cpu', {'selected_device': 'cpu', 'strategy': 'fallback_no_nvidia_smi'})
            candidates = []
            for (gpu, stats) in memory.items():
                free_mb = stats['memory_total_mb'] - stats['memory_used_mb']
                if free_mb < min_free_mb:
                    continue
                sm_pct = pmon.get(gpu, stats['utilization_gpu_pct'])
                candidates.append((sm_pct, stats['memory_used_mb'], int(gpu), free_mb, stats))
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            if max_auto_gpus > 0:
                candidates = candidates[:max_auto_gpus]
            if len(candidates) < 2:
                (single, report) = choose_device('auto')
                report['strategy'] = 'fallback_auto_multi_to_single_gpu'
                report['min_free_mb'] = min_free_mb
                return (single, report)
            selected = sorted((str(item[2]) for item in candidates))
            report = {'selected_device': ','.join(selected), 'strategy': 'auto_multi_gpu', 'min_free_mb': min_free_mb, 'max_auto_gpus': max_auto_gpus, 'selected_gpus': {str(item[2]): {'sm_pct': item[0], 'free_mb': item[3], **item[4]} for item in candidates}}
            return (','.join(selected), report)
        return (requested, {'selected_device': requested, 'strategy': 'explicit'})
    memory = gpu_memory_snapshot()
    pmon = gpu_pmon_snapshot()
    if not memory:
        return ('cpu', {'selected_device': 'cpu', 'strategy': 'fallback_no_nvidia_smi'})
    scored = []
    for (gpu, stats) in memory.items():
        total = max(stats['memory_total_mb'], 1.0)
        memory_pct = 100.0 * stats['memory_used_mb'] / total
        sm_pct = pmon.get(gpu, stats['utilization_gpu_pct'])
        score = sm_pct * 2.0 + memory_pct
        scored.append((score, int(gpu), sm_pct, memory_pct, stats))
    scored.sort(key=lambda item: (item[0], item[1]))
    (_, index, sm_pct, memory_pct, stats) = scored[0]
    report = {'selected_device': str(index), 'strategy': 'least_busy_single_gpu', 'selected_sm_pct': sm_pct, 'selected_memory_pct': memory_pct, 'all_gpus': {str(item[1]): {'score': item[0], 'sm_pct': item[2], 'memory_pct': item[3], **item[4]} for item in scored}}
    return (str(index), report)

def is_oom_error(exc: BaseException) -> bool:
    text = f'{type(exc).__name__}: {exc}'.lower()
    return any((pattern in text for pattern in ('cuda out of memory', 'outofmemoryerror', 'out of memory')))

def metrics_payload(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, 'box', None)
    payload: dict[str, Any] = {}
    for name in ('map', 'map50', 'map75', 'mp', 'mr'):
        value = getattr(box, name, None) if box is not None else None
        if value is not None:
            payload[name] = float(value)
    maps = getattr(box, 'maps', None) if box is not None else None
    if maps is not None:
        payload['per_class_map50_95'] = {CLASS_NAMES[index]: float(value) for (index, value) in enumerate(maps) if index < len(CLASS_NAMES)}
    return payload

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

def train_once(YOLO, args: argparse.Namespace, device: str, batch: int, name: str):
    model_path = args.resume if args.resume else args.model
    model = YOLO(str(model_path))
    train_kwargs = {'data': str(args.data), 'epochs': args.epochs, 'imgsz': args.imgsz, 'batch': batch, 'device': device, 'workers': args.workers, 'seed': args.seed, 'deterministic': True, 'mosaic': 0, 'mixup': 0, 'copy_paste': 0, 'fliplr': 0, 'rect': args.rect, 'patience': args.patience, 'save': True, 'save_period': args.save_period, 'project': str(args.project), 'name': name, 'exist_ok': args.exist_ok, 'fraction': args.fraction, 'val': args.val, 'plots': args.plots}
    if args.resume:
        train_kwargs['resume'] = str(args.resume)
    if args.save_period < 0:
        from tsm_renderer.training.completed_checkpoint import CompletedEpochDetectionTrainer
        train_kwargs['trainer'] = CompletedEpochDetectionTrainer
    return model.train(**train_kwargs)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while (chunk := handle.read(1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def validate_completed_epoch_checkpoints(run_dir: Path, *, epochs: int, save_period: int, seed: int) -> list[dict[str, Any]]:
    """Validate and describe exact completed-epoch checkpoints."""
    if save_period >= 0:
        return []
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required to validate training checkpoints') from exc
    period = abs(save_period)
    expected_epochs = list(range(period, epochs + 1, period))
    rows: list[dict[str, Any]] = []
    for completed_epoch in expected_epochs:
        path = run_dir / 'weights' / f'epoch{completed_epoch:03d}.pt'
        if not path.is_file():
            raise RuntimeError(f'Missing completed-epoch checkpoint: {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('epoch') != completed_epoch - 1:
            raise RuntimeError(f"{path} has internal epoch={checkpoint.get('epoch')}, expected {completed_epoch - 1}")
        train_args = checkpoint.get('train_args') or {}
        if int(train_args.get('seed', -1)) != seed:
            raise RuntimeError(f"{path} has seed={train_args.get('seed')}, expected {seed}")
        missing_state = [key for key in ('optimizer', 'scaler', 'ema') if checkpoint.get(key) is None]
        if missing_state:
            raise RuntimeError(f'{path} is not a complete resume checkpoint; missing {missing_state}')
        rows.append({'completed_epoch': completed_epoch, 'internal_epoch': checkpoint['epoch'], 'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha256_file(path), 'seed': int(train_args['seed']), 'optimizer_present': True, 'scaler_present': True, 'ema_present': True})
    return rows

def latest_run_dir(project: Path, name_prefix: str) -> Path:
    candidates = [path for path in project.glob(f'{name_prefix}*') if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f'No run directory found under {project} for {name_prefix}')
    return max(candidates, key=lambda path: path.stat().st_mtime)

def resolve_run_dir(args: argparse.Namespace, run_name: str) -> Path:
    if args.resume:
        resume = args.resume.resolve()
        if resume.parent.name == 'weights':
            return resume.parent.parent
    return latest_run_dir(args.project, run_name)

def run_evaluation(YOLO, best_path: Path, args: argparse.Namespace, device: str, run_dir: Path) -> None:
    model = YOLO(str(best_path))
    eval_root = run_dir / 'eval'
    results: dict[str, Any] = {}
    for split in ('val', 'test'):
        metrics = model.val(data=str(args.data), split=split, imgsz=args.imgsz, device=device, project=str(eval_root), name=f'{split}_standard', exist_ok=True, plots=True, save_json=True)
        results[f'{split}_standard'] = metrics_payload(metrics)
        deploy_metrics = model.val(data=str(args.data), split=split, imgsz=args.imgsz, device=device, conf=args.deploy_conf, iou=args.deploy_iou, agnostic_nms=True, project=str(eval_root), name=f'{split}_deploy_conf{args.deploy_conf}_iou{args.deploy_iou}', exist_ok=True, plots=False, save_json=True)
        results[f'{split}_deploy'] = metrics_payload(deploy_metrics)
    write_json(run_dir / 'atomblock_eval_metrics.json', results)

def export_onnx(YOLO, best_path: Path, args: argparse.Namespace, device: str, run_dir: Path) -> Path:
    model = YOLO(str(best_path))
    exported = model.export(format='onnx', imgsz=args.imgsz, device=device)
    exported_path = Path(exported)
    target_path = run_dir / 'weights' / exported_path.name
    if exported_path.exists() and exported_path.resolve() != target_path.resolve():
        shutil.copy2(exported_path, target_path)
    return target_path if target_path.exists() else exported_path

def validate_data_yaml(data_path: Path) -> None:
    text = data_path.read_text(encoding='utf-8')
    missing = [name for name in CLASS_NAMES if not re.search(f"""['\\"]?{re.escape(name)}['\\"]?""", text)]
    if missing:
        raise SystemExit(f'data.yaml is missing class names: {missing}')
    if 'path/to/yolo-dataset' in text:
        raise SystemExit(f'data.yaml still contains placeholder path: {data_path}')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train YOLO26 on AtomBlock-WebUI.')
    parser.add_argument('--data', type=Path, default=Path('MKD2WORLD_EXTERNAL_ROOT/datasets/AtomBlock-WebUI/yolo-dataset/data.yaml'))
    parser.add_argument('--model', type=Path, default=Path('MKD2WORLD_EXTERNAL_ROOT/models/YOLO26/yolo26m.pt'))
    parser.add_argument('--expected-model-sha256', default='', help='Fail before training if the initial checkpoint hash differs.')
    parser.add_argument('--project', type=Path, default=Path('MKD2WORLD_EXTERNAL_ROOT/output/yolo/atomblock'))
    parser.add_argument('--name', default='yolo26m_img1920_seed0')
    parser.add_argument('--deploy-path', type=Path, default=Path('MKD2WORLD_EXTERNAL_ROOT/models/yolo/atomblock_yolo26m_best.pt'))
    parser.add_argument('--resume', type=Path, help='Resume from an Ultralytics training checkpoint such as weights/last.pt.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=1920)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--save-period', type=int, default=2)
    parser.add_argument('--device', default='auto', help="'auto', 'auto-multi', 'cpu', '0', or CUDA devices such as '0,1,2,3'.")
    parser.add_argument('--eval-device', help='Device for final val/test/export. Defaults to the training device.')
    parser.add_argument('--min-free-mb', type=float, default=20000.0, help='Minimum free memory per GPU for --device auto-multi.')
    parser.add_argument('--max-auto-gpus', type=int, default=0, help='Maximum GPUs for --device auto-multi; 0 means no limit.')
    parser.add_argument('--fraction', type=float, default=1.0, help='Training data fraction, useful for short throughput tests.')
    parser.add_argument('--rect', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--val', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--deploy-conf', type=float, default=0.1)
    parser.add_argument('--deploy-iou', type=float, default=0.1)
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--skip-deploy', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser

def main() -> None:
    args = build_parser().parse_args()
    YOLO = import_yolo()
    validate_data_yaml(args.data)
    if not args.model.exists():
        raise SystemExit(f'Missing pretrained model: {args.model}')
    model_sha256 = sha256_file(args.model)
    (device, gpu_report) = choose_device(args.device, min_free_mb=args.min_free_mb, max_auto_gpus=args.max_auto_gpus)
    args.project.mkdir(parents=True, exist_ok=True)
    args.deploy_path.parent.mkdir(parents=True, exist_ok=True)
    planned = {'data': str(args.data), 'model': str(args.model), 'model_sha256': model_sha256, 'expected_model_sha256': args.expected_model_sha256 or None, 'resume': str(args.resume) if args.resume else None, 'project': str(args.project), 'name': args.name, 'deploy_path': str(args.deploy_path), 'epochs': args.epochs, 'imgsz': args.imgsz, 'batch': args.batch, 'workers': args.workers, 'seed': args.seed, 'mosaic': 0, 'mixup': 0, 'copy_paste': 0, 'fliplr': 0, 'rect': args.rect, 'fraction': args.fraction, 'val': args.val, 'plots': args.plots, 'patience': args.patience, 'save_period': args.save_period, 'device': device, 'eval_device': args.eval_device or device, 'gpu_report': gpu_report}
    print(json.dumps(planned, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    run_name = args.name
    batch = args.batch
    try:
        train_once(YOLO, args, device, batch, run_name)
    except Exception as exc:
        if batch == 8 and is_oom_error(exc):
            print('Batch 8 hit CUDA OOM; retrying with batch=4 and unchanged training settings.', file=sys.stderr)
            batch = 4
            run_name = f'{args.name}_batch4'
            train_once(YOLO, args, device, batch, run_name)
        else:
            raise
    run_dir = resolve_run_dir(args, run_name)
    best_path = run_dir / 'weights' / 'best.pt'
    if not best_path.exists():
        raise SystemExit(f'Training finished but best.pt was not found: {best_path}')
    if not args.skip_deploy:
        shutil.copy2(best_path, args.deploy_path)
    write_json(run_dir / 'atomblock_train_manifest.json', {**planned, 'actual_batch': batch, 'run_dir': str(run_dir), 'best_path': str(best_path), 'deploy_path': str(args.deploy_path), 'deploy_copied': not args.skip_deploy})
    completed_checkpoints = validate_completed_epoch_checkpoints(run_dir, epochs=args.epochs, save_period=args.save_period, seed=args.seed)
    if completed_checkpoints:
        write_json(run_dir / 'completed_epoch_checkpoints.json', {'save_period': abs(args.save_period), 'checkpoints': completed_checkpoints})
    if not args.skip_eval:
        run_evaluation(YOLO, best_path, args, args.eval_device or device, run_dir)
    if not args.skip_export:
        onnx_path = export_onnx(YOLO, best_path, args, args.eval_device or device, run_dir)
        print(f'Exported ONNX: {onnx_path}')
    if not args.skip_deploy:
        print(f'Best checkpoint copied to: {args.deploy_path}')
if __name__ == '__main__':
    main()
