#!/usr/bin/env python3
"""Unpack CC3M Arrow streams into ``{shard}/{key}.jpg`` and ``.txt`` files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
from tqdm import tqdm


def extract_single_file(args: tuple[str, str, str]) -> tuple[str, int, int, int]:
    arrow_path, out_dir, shard_name = args
    out_path = Path(out_dir) / shard_name
    out_path.mkdir(parents=True, exist_ok=True)

    stream = pa.ipc.open_stream(str(arrow_path))
    table = stream.read_all()
    rows = table.to_pylist()
    written_img = 0
    written_txt = 0
    for row in rows:
        key = row["__key__"]
        caption = row["txt"]
        jpg_bytes = row["jpg"]["bytes"]
        img_file = out_path / f"{key}.jpg"
        txt_file = out_path / f"{key}.txt"
        if not img_file.exists():
            img_file.write_bytes(jpg_bytes)
            written_img += 1
        if not txt_file.exists():
            txt_file.write_text(caption, encoding="utf-8")
            written_txt += 1
    return shard_name, len(rows), written_img, written_txt


def build_tasks(src_dir: str, out_dir: str, limit: int | None = None) -> list[tuple[Path, str, str]]:
    arrow_files = sorted(Path(src_dir).glob("cc3m-wds-train-*.arrow"))
    if limit:
        arrow_files = arrow_files[:limit]
    tasks = []
    for arrow_file in arrow_files:
        # cc3m-wds-train-00000-of-00521 -> shard_00000
        shard_idx = arrow_file.stem.split("-")[-3]
        tasks.append((arrow_file, out_dir, f"shard_{shard_idx}"))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CC3M Arrow files to images and captions.")
    parser.add_argument("--src", type=Path, required=True, help="Directory of cc3m-wds-train-*.arrow files.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N Arrow files.")
    args = parser.parse_args()

    tasks = build_tasks(str(args.src), str(args.out), args.limit)
    print(f"[extract] total arrow files to process: {len(tasks)}")
    print(f"[extract] workers: {args.workers}")
    print(f"[extract] output: {args.out}")

    total_rows = 0
    total_written_img = 0
    total_written_txt = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_single_file, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="extracting"):
            _shard, n_rows, written_img, written_txt = future.result()
            total_rows += n_rows
            total_written_img += written_img
            total_written_txt += written_txt

    print(f"[extract] done. total rows processed: {total_rows}")
    print(f"[extract] images written: {total_written_img}")
    print(f"[extract] captions written: {total_written_txt}")


if __name__ == "__main__":
    main()
