#!/usr/bin/env python3
"""Encode CC3M captions on several GPUs, then merge one FAISS index."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np

DIM = 384
SCRIPT = Path(__file__).with_name("build_caption_index.py")


def run_worker(gpu_id: int, num_workers: int, force_rebuild: bool, cc3m_dir: Path, index_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, str(SCRIPT),
        "--cc3m-dir", str(cc3m_dir),
        "--index-dir", str(index_dir),
        "--worker-id", str(gpu_id),
        "--num-workers", str(num_workers),
    ]
    if force_rebuild:
        cmd.append("--force-rebuild")
    print(f"[launcher] starting worker {gpu_id}/{num_workers}: CUDA_VISIBLE_DEVICES={gpu_id}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def merge_results(num_workers: int, index_dir: Path) -> None:
    print("[launcher] merging partial results")
    caption_path = index_dir / "captions.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    faiss_path = index_dir / "index.faiss"
    with caption_path.open("w", encoding="utf-8") as output:
        for worker_id in range(num_workers):
            part = index_dir / f"captions.jsonl.part_{worker_id}"
            if not part.exists():
                continue
            output.write(part.read_text(encoding="utf-8"))

    parts = []
    for worker_id in range(num_workers):
        part = index_dir / f"embeddings.part_{worker_id}.npy"
        if part.exists():
            parts.append(np.load(part).astype(np.float32))
    if not parts:
        print("[launcher] warning: no embedding parts found")
        return
    combined = np.vstack(parts)
    np.save(embeddings_path, combined)
    print(f"[launcher] merged embeddings: {combined.shape[0]} vectors")

    print("[launcher] building faiss index")
    index = faiss.IndexFlatIP(DIM)
    index.add(combined)
    faiss.write_index(index, str(faiss_path))
    print(f"[launcher] faiss index saved: {index.ntotal} vectors")
    for worker_id in range(num_workers):
        for part in (
            index_dir / f"captions.jsonl.part_{worker_id}",
            index_dir / f"embeddings.part_{worker_id}.npy",
        ):
            if part.exists():
                part.unlink()
    print("[launcher] cleanup done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CC3M caption index on multiple GPUs.")
    parser.add_argument("--cc3m-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--ngpus", type=int, default=8)
    args = parser.parse_args()

    print(f"[launcher] using {args.ngpus} GPUs for parallel encoding")
    procs = []
    for gpu_id in range(args.ngpus):
        procs.append((gpu_id, run_worker(gpu_id, args.ngpus, args.force_rebuild, args.cc3m_dir, args.index_dir)))

    for gpu_id, process in procs:
        print(f"\n[launcher] ===== worker {gpu_id} output =====")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"[launcher] ERROR: worker {gpu_id} exited with code {process.returncode}")

    failed = [gpu_id for gpu_id, process in procs if process.returncode != 0]
    if failed:
        print(f"[launcher] FATAL: workers {failed} failed. Aborting merge.")
        sys.exit(1)
    merge_results(args.ngpus, args.index_dir)
    print("[launcher] all done")


if __name__ == "__main__":
    main()
