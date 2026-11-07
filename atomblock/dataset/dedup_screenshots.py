#!/usr/bin/env python3
"""Drop near-duplicate reference screenshots with a 256-bit dHash.

Images are compared only inside the same immediate subdirectory. A pair is
linked when the Hamming distance is at most 5. Each component keeps the
largest file, preferring a parent directory whose name starts with ``train-``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

HASH_SIZE = 16
HAMMING_THRESHOLD = 5


def dhash(image_path: Path) -> int | None:
    """Return a 256-bit difference hash, or None if the file cannot be read."""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
            pixels = list(gray.getdata())
            diff = 0
            for row in range(HASH_SIZE):
                for col in range(HASH_SIZE):
                    left = pixels[row * (HASH_SIZE + 1) + col]
                    right = pixels[row * (HASH_SIZE + 1) + col + 1]
                    diff = (diff << 1) | (1 if left > right else 0)
            return diff
    except Exception:
        return None


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def is_train_shard(folder_name: str) -> bool:
    return folder_name.startswith("train-")


def choose_representative(paths: list[Path]) -> Path:
    """Keep the last file after sorting by train-shard flag, then size.

Inside one folder the flag is constant, so the largest file is kept.
"""
    ranked = sorted(paths, key=lambda p: (-int(is_train_shard(p.parent.name)), p.stat().st_size))
    return ranked[-1]


def dedup_folder(img_paths: list[Path]) -> tuple[set[Path], dict[str, str]]:
    hashes: dict[Path, int] = {}
    failed = []
    for img_path in img_paths:
        digest = dhash(img_path)
        if digest is None:
            failed.append(str(img_path))
            continue
        hashes[img_path] = digest
    if failed:
        print(f"    Warning: {len(failed)} images failed hash")

    paths = list(hashes.keys())
    hash_vals = [hashes[p] for p in paths]
    parent = list(range(len(paths)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hamming_distance(hash_vals[i], hash_vals[j]) <= HAMMING_THRESHOLD:
                union(i, j)

    clusters: dict[int, list[Path]] = defaultdict(list)
    for idx, path in enumerate(paths):
        clusters[find(idx)].append(path)

    kept: set[Path] = set()
    removed_map: dict[str, str] = {}
    for group in clusters.values():
        representative = choose_representative(group)
        kept.add(representative)
        for path in group:
            if path != representative:
                removed_map[str(path)] = str(representative)
    return kept, removed_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter near-duplicate screenshots with dHash.")
    parser.add_argument("--src", type=Path, required=True, help="Directory of shard folders containing jpg files.")
    parser.add_argument("--dst", type=Path, required=True, help="Directory that receives the kept copies.")
    parser.add_argument("--log", type=Path, required=True, help="JSON log of removed paths.")
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source directory not found: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    folders = sorted(path for path in args.src.iterdir() if path.is_dir())
    print(f"[dedup] Found {len(folders)} folders in {args.src}")

    total_images = 0
    total_kept = 0
    total_removed = 0
    all_removed: dict[str, str] = {}
    for folder in folders:
        img_paths = sorted(folder.glob("*.jpg"))
        if not img_paths:
            continue
        total_images += len(img_paths)
        print(f"[dedup] Processing {folder.name}: {len(img_paths)} images")
        kept, removed_map = dedup_folder(img_paths)
        total_kept += len(kept)
        total_removed += len(removed_map)
        all_removed.update(removed_map)
        for src_path in kept:
            rel = src_path.relative_to(args.src)
            dst_path = args.dst / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

    log = {
        "total_images": total_images,
        "kept_images": total_kept,
        "removed_images": total_removed,
        "threshold": HAMMING_THRESHOLD,
        "hash_size": HASH_SIZE,
        "scope": "per-folder",
        "removed_map": all_removed,
    }
    args.log.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[dedup] Done.")
    print(f"  Original : {total_images}")
    print(f"  Kept     : {total_kept}")
    print(f"  Removed  : {total_removed}")
    print(f"  Output   : {args.dst}")
    print(f"  Log      : {args.log}")


if __name__ == "__main__":
    main()
