#!/usr/bin/env python3
"""Evaluate every saved AtomBlock YOLO checkpoint with per-class metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_NAMES = [
    "button",
    "link",
    "input",
    "image",
    "icon",
    "checkbox",
    "radio",
    "select",
    "block-nav",
    "block-sidebar",
    "block-footer",
    "block-form",
    "block-table",
    "block",
]

EVAL_CONFIGS = [
    {
        "name": "val_standard",
        "split": "val",
        "conf": None,
        "iou": None,
        "agnostic_nms": False,
        "plots": True,
    },
    {
        "name": "val_deploy_conf0.1_iou0.1",
        "split": "val",
        "conf": 0.1,
        "iou": 0.1,
        "agnostic_nms": True,
        "plots": False,
    },
    {
        "name": "test_standard",
        "split": "test",
        "conf": None,
        "iou": None,
        "agnostic_nms": False,
        "plots": True,
    },
    {
        "name": "test_deploy_conf0.1_iou0.1",
        "split": "test",
        "conf": 0.1,
        "iou": 0.1,
        "agnostic_nms": True,
        "plots": False,
    },
]


@dataclass(frozen=True)
class WorkerJob:
    device: str
    weights: list[str]
    data: str
    output_dir: str
    imgsz: int
    batch: int
    workers: int
    save_json: bool
    overwrite: bool


def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("Run this script inside the yolo conda environment.") from exc
    return YOLO


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"Expected epoch checkpoint name, got: {path}")
    return int(match.group(1))


def checkpoint_id(path: Path) -> str:
    return f"epoch{checkpoint_epoch(path):03d}"


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def metrics_payload(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    fitness_value = getattr(metrics, "fitness", None)
    if callable(fitness_value):
        fitness_value = fitness_value()
    payload: dict[str, Any] = {
        "map": as_float(getattr(box, "map", None)) if box is not None else None,
        "map50": as_float(getattr(box, "map50", None)) if box is not None else None,
        "map75": as_float(getattr(box, "map75", None)) if box is not None else None,
        "mp": as_float(getattr(box, "mp", None)) if box is not None else None,
        "mr": as_float(getattr(box, "mr", None)) if box is not None else None,
        "fitness": as_float(fitness_value),
        "speed": getattr(metrics, "speed", {}),
    }

    per_class = {name: {"precision": None, "recall": None, "ap50": None, "ap50_95": None, "targets": None} for name in CLASS_NAMES}
    if box is not None:
        ap_class_index = as_list(getattr(box, "ap_class_index", None))
        p_values = as_list(getattr(box, "p", None))
        r_values = as_list(getattr(box, "r", None))
        ap50_values = as_list(getattr(box, "ap50", None))
        ap_values = as_list(getattr(box, "ap", None))
        for position, class_index in enumerate(ap_class_index):
            class_index = int(class_index)
            if class_index >= len(CLASS_NAMES):
                continue
            per_class[CLASS_NAMES[class_index]].update(
                {
                    "precision": as_float(p_values[position]) if position < len(p_values) else None,
                    "recall": as_float(r_values[position]) if position < len(r_values) else None,
                    "ap50": as_float(ap50_values[position]) if position < len(ap50_values) else None,
                    "ap50_95": as_float(ap_values[position]) if position < len(ap_values) else None,
                }
            )

    nt_per_class = getattr(metrics, "nt_per_class", None)
    if nt_per_class is not None:
        for index, value in enumerate(as_list(nt_per_class)):
            if index < len(CLASS_NAMES):
                per_class[CLASS_NAMES[index]]["targets"] = int(value)

    payload["per_class"] = per_class
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def evaluate_checkpoint(
    YOLO,
    weights: Path,
    data: str,
    output_dir: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    save_json: bool,
) -> dict[str, Any]:
    ckpt_id = checkpoint_id(weights)
    ckpt_dir = output_dir / "per_checkpoint" / ckpt_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    result: dict[str, Any] = {
        "checkpoint": weights.name,
        "checkpoint_id": ckpt_id,
        "epoch": checkpoint_epoch(weights),
        "weights": str(weights),
        "device": device,
        "imgsz": imgsz,
        "evaluations": {},
    }

    for config in EVAL_CONFIGS:
        kwargs: dict[str, Any] = {
            "data": data,
            "split": config["split"],
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "device": device,
            "project": str(ckpt_dir),
            "name": config["name"],
            "exist_ok": True,
            "plots": config["plots"],
            "save_json": save_json,
        }
        if config["conf"] is not None:
            kwargs["conf"] = config["conf"]
        if config["iou"] is not None:
            kwargs["iou"] = config["iou"]
        if config["agnostic_nms"]:
            kwargs["agnostic_nms"] = True
        metrics = model.val(**kwargs)
        result["evaluations"][config["name"]] = metrics_payload(metrics)

    write_json(ckpt_dir / "metrics.json", result)
    return result


def run_worker(job: WorkerJob) -> list[dict[str, Any]]:
    YOLO = import_yolo()
    output_dir = Path(job.output_dir)
    worker_results: list[dict[str, Any]] = []
    for weight_text in job.weights:
        weights = Path(weight_text)
        ckpt_dir = output_dir / "per_checkpoint" / checkpoint_id(weights)
        metrics_path = ckpt_dir / "metrics.json"
        if metrics_path.exists() and not job.overwrite:
            worker_results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            print(f"[SKIP] {weights.name} device={job.device} metrics={metrics_path}", flush=True)
            continue
        started = time.time()
        print(f"[START] {weights.name} device={job.device}", flush=True)
        result = evaluate_checkpoint(
            YOLO=YOLO,
            weights=weights,
            data=job.data,
            output_dir=output_dir,
            imgsz=job.imgsz,
            batch=job.batch,
            workers=job.workers,
            device=job.device,
            save_json=job.save_json,
        )
        elapsed = time.time() - started
        result["elapsed_seconds"] = elapsed
        write_json(metrics_path, result)
        worker_results.append(result)
        print(f"[DONE] {weights.name} device={job.device} elapsed={elapsed:.1f}s", flush=True)
    return worker_results


def flatten_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item["epoch"]):
        for eval_name, metrics in result["evaluations"].items():
            summary_rows.append(
                {
                    "epoch": result["epoch"],
                    "checkpoint": result["checkpoint"],
                    "eval": eval_name,
                    "map50_95": metrics.get("map"),
                    "map50": metrics.get("map50"),
                    "map75": metrics.get("map75"),
                    "precision": metrics.get("mp"),
                    "recall": metrics.get("mr"),
                    "fitness": metrics.get("fitness"),
                    "weights": result["weights"],
                }
            )
            for class_name, class_metrics in metrics.get("per_class", {}).items():
                per_class_rows.append(
                    {
                        "epoch": result["epoch"],
                        "checkpoint": result["checkpoint"],
                        "eval": eval_name,
                        "class": class_name,
                        "targets": class_metrics.get("targets"),
                        "precision": class_metrics.get("precision"),
                        "recall": class_metrics.get("recall"),
                        "ap50": class_metrics.get("ap50"),
                        "ap50_95": class_metrics.get("ap50_95"),
                        "weights": result["weights"],
                    }
                )
    return summary_rows, per_class_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output_dir: Path) -> list[dict[str, Any]]:
    metrics_paths = sorted((output_dir / "per_checkpoint").glob("epoch*/metrics.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in metrics_paths]
    summary_rows, per_class_rows = flatten_rows(results)
    write_json(output_dir / "summary.json", {"results": results})
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_class.csv", per_class_rows)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate all AtomBlock YOLO checkpoint weights.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("MKD2WORLD_EXTERNAL_ROOT/output/yolo/atomblock/yolo26m_img1920_seed0_8gpu_b64/weights"),
    )
    parser.add_argument(
        "--data",
        default="MKD2WORLD_EXTERNAL_ROOT/datasets/AtomBlock-WebUI/yolo-dataset/data.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "MKD2WORLD_EXTERNAL_ROOT/output/yolo/atomblock/yolo26m_img1920_seed0_8gpu_b64/eval_all_checkpoints"
        ),
    )
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = sorted(args.weights_dir.glob("epoch*.pt"), key=checkpoint_epoch)
    if not weights:
        raise SystemExit(f"No epoch checkpoints found under {args.weights_dir}")

    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one device")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "weights_dir": str(args.weights_dir),
            "checkpoints": [weight.name for weight in weights],
            "checkpoint_count": len(weights),
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "devices": devices,
            "save_json": args.save_json,
            "eval_configs": EVAL_CONFIGS,
            "output_layout": {
                "per_checkpoint_metrics": "per_checkpoint/epochXXX/metrics.json",
                "ultralytics_outputs": "per_checkpoint/epochXXX/{val_standard,val_deploy_conf0.1_iou0.1,test_standard,test_deploy_conf0.1_iou0.1}",
                "summary": "summary.csv and summary.json",
                "per_class": "per_class.csv",
            },
        },
    )

    if not args.aggregate_only:
        partitions = [[] for _ in devices]
        for index, weight in enumerate(weights):
            partitions[index % len(devices)].append(str(weight))
        jobs = [
            WorkerJob(
                device=device,
                weights=partitions[index],
                data=args.data,
                output_dir=str(args.output_dir),
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                save_json=args.save_json,
                overwrite=args.overwrite,
            )
            for index, device in enumerate(devices)
            if partitions[index]
        ]
        print(json.dumps({"jobs": [job.__dict__ for job in jobs]}, ensure_ascii=False, indent=2), flush=True)
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as executor:
            futures = [executor.submit(run_worker, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    results = aggregate(args.output_dir)
    print(
        json.dumps(
            {
                "completed_checkpoints": len(results),
                "summary_csv": str(args.output_dir / "summary.csv"),
                "per_class_csv": str(args.output_dir / "per_class.csv"),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
