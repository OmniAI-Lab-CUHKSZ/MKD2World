#!/usr/bin/env python3
"""Split reference screenshots across processes that call generate_html.py."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def split_files(input_dir: Path, workers: int, image_list: Path | None = None) -> list[list[Path]]:
    if image_list:
        image_files = []
        for line in image_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = input_dir / path
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                image_files.append(path)
        image_files = sorted(set(image_files))
    else:
        image_files = sorted(
            path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTS
        )
    if not image_files:
        print(f"[run-html] no images in {input_dir}")
        return []
    chunks: list[list[Path]] = [[] for _ in range(workers)]
    for index, path in enumerate(image_files):
        chunks[index % workers].append(path)
    return chunks


def run_worker(
    worker_id: int,
    files: list[Path],
    concurrency: int,
    model: str,
    api_script: Path,
    input_dir: Path,
    output_dir: Path,
    use_image_list: bool,
) -> subprocess.Popen:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"atomblock_html_{worker_id}_"))
    for src in files:
        try:
            rel = src.relative_to(input_dir)
        except ValueError:
            rel = Path(src.name)
        dst = tmp_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    cmd = [
        sys.executable,
        str(api_script),
        "--input-dir", str(tmp_dir),
        "--output-dir", str(output_dir),
        "--concurrency", str(concurrency),
        "--model", model,
    ]
    if use_image_list:
        worker_list = tmp_dir / "worker_image_list.txt"
        lines = []
        for src in files:
            try:
                lines.append(str(src.relative_to(input_dir)))
            except ValueError:
                lines.append(src.name)
        worker_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd.extend(["--image-list", str(worker_list)])
    print(f"[run-html] worker {worker_id}: {len(files)} images, concurrency={concurrency}")
    return subprocess.Popen(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HTML generation in several processes.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--api-script", type=Path, default=Path(__file__).with_name("generate_html.py"))
    parser.add_argument("--image-list", type=Path, default=None)
    args = parser.parse_args()

    chunks = split_files(args.input_dir, args.workers, args.image_list)
    if not chunks:
        return
    total = sum(len(chunk) for chunk in chunks)
    print(f"[run-html] {total} images, {args.workers} processes x {args.concurrency} concurrent calls")
    processes = []
    for worker_id, files in enumerate(chunks):
        if not files:
            continue
        processes.append(run_worker(
            worker_id,
            files,
            args.concurrency,
            args.model,
            args.api_script,
            args.input_dir,
            args.output_dir,
            use_image_list=args.image_list is not None,
        ))
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\n[run-html] interrupted; stopping workers")
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()
    print("[run-html] workers finished")


if __name__ == "__main__":
    main()
