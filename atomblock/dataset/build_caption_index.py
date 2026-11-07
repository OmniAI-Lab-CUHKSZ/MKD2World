#!/usr/bin/env python3
"""Build a FAISS caption index for CC3M images.

``captions.jsonl``, ``embeddings.npy``, and ``index.faiss`` are replaced
together. With more than one worker, each worker writes a part file and
``run_caption_index.py`` merges them.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512
DIM = 384
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

CC3M_DIR: Path
INDEX_DIR: Path
CAPTION_JSONL: Path
EMBEDDINGS_NPY: Path
FAISS_INDEX: Path
CAPTION_JSONL_TMP: Path
EMBEDDINGS_NPY_TMP: Path
FAISS_INDEX_TMP: Path


def configure(cc3m_dir: Path, index_dir: Path) -> None:
    global CC3M_DIR, INDEX_DIR, CAPTION_JSONL, EMBEDDINGS_NPY, FAISS_INDEX
    global CAPTION_JSONL_TMP, EMBEDDINGS_NPY_TMP, FAISS_INDEX_TMP
    CC3M_DIR = cc3m_dir
    INDEX_DIR = index_dir
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_JSONL = INDEX_DIR / "captions.jsonl"
    EMBEDDINGS_NPY = INDEX_DIR / "embeddings.npy"
    FAISS_INDEX = INDEX_DIR / "index.faiss"
    CAPTION_JSONL_TMP = INDEX_DIR / "captions.jsonl.tmp"
    # NumPy writes this name as-is because it already ends in .npy.
    EMBEDDINGS_NPY_TMP = INDEX_DIR / "embeddings.tmp.npy"
    FAISS_INDEX_TMP = INDEX_DIR / "index.faiss.tmp"


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-10, None)


def _read_caption(text: str, ext: str) -> str:
    text = text.strip()
    if ext == ".json":
        try:
            text = json.loads(text).get("caption", text)
        except Exception:
            pass
    return text


def gather_sources(root: Path):
    """Yield (source_type, source_path, image_id, caption)."""
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        for image_file in subdir.rglob("*"):
            if image_file.suffix.lower() not in IMG_EXTS:
                continue
            caption = None
            for ext in (".txt", ".json", ".caption"):
                cap_file = image_file.with_suffix(ext)
                if cap_file.exists():
                    caption = _read_caption(cap_file.read_text(encoding="utf-8"), ext)
                    break
            if caption:
                yield ("file", str(image_file), str(image_file.relative_to(root)), caption)

    for tar_path in sorted(root.rglob("*.tar")):
        try:
            with tarfile.open(tar_path, "r:*") as handle:
                members = {member.name: member for member in handle.getmembers()}
                for name in sorted(members):
                    if not any(name.lower().endswith(ext) for ext in IMG_EXTS):
                        continue
                    base = os.path.splitext(name)[0]
                    caption = None
                    for cap_name in (base + ".txt", base + ".json", base + ".caption"):
                        if cap_name in members:
                            raw = handle.extractfile(members[cap_name]).read().decode("utf-8")
                            caption = _read_caption(raw, os.path.splitext(cap_name)[1])
                            break
                    if caption:
                        yield ("tar", str(tar_path), f"{tar_path}::{name}", caption)
        except Exception as exc:
            print(f"[warn] failed to read tar {tar_path}: {exc}")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def get_faiss_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return faiss.read_index(str(path)).ntotal
    except Exception:
        return 0


def get_embeddings_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(np.load(path).shape[0])
    except Exception:
        return 0


def validate_index(strict: bool = True) -> int:
    n_jsonl = count_jsonl_lines(CAPTION_JSONL)
    n_emb = get_embeddings_count(EMBEDDINGS_NPY)
    n_faiss = get_faiss_count(FAISS_INDEX)
    print(f"[validate] captions.jsonl={n_jsonl}, embeddings.npy={n_emb}, index.faiss={n_faiss}")
    if n_jsonl == 0 and n_emb == 0 and n_faiss == 0:
        return 0
    if n_jsonl != n_emb or n_jsonl != n_faiss:
        message = (
            f"Index inconsistency detected: jsonl={n_jsonl}, emb={n_emb}, faiss={n_faiss}. "
            f"Delete {INDEX_DIR} and rerun, or pass --force-rebuild."
        )
        if strict:
            raise RuntimeError(message)
        print(f"[warn] {message}")
    return n_jsonl


def load_existing_index() -> set[str]:
    existing_ids: set[str] = set()
    if CAPTION_JSONL.exists():
        with CAPTION_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_ids.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    return existing_ids


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    os.replace(tmp_path, final_path)


def build_index(force_rebuild: bool = False, worker_id: int = 0, num_workers: int = 1) -> None:
    if force_rebuild and any(path.exists() for path in (CAPTION_JSONL, EMBEDDINGS_NPY, FAISS_INDEX)):
        print("[build-index] force rebuild: removing old index files")
        for path in (CAPTION_JSONL, EMBEDDINGS_NPY, FAISS_INDEX):
            if path.exists():
                path.unlink()

    validate_index(strict=True)
    print(f"[build-index] using model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    existing_ids = load_existing_index()
    print(f"[build-index] existing records: {len(existing_ids)}")

    new_records = []
    print("[build-index] scanning image captions")
    for src_type, src_path, image_id, caption in tqdm(gather_sources(CC3M_DIR), desc="scanning"):
        if image_id in existing_ids:
            continue
        new_records.append({
            "source_type": src_type,
            "source_path": src_path,
            "image_id": image_id,
            "caption": caption,
        })

    if num_workers > 1:
        total_new = len(new_records)
        new_records = [record for index, record in enumerate(new_records) if index % num_workers == worker_id]
        print(f"[build-index] worker {worker_id}/{num_workers}: {len(new_records)}/{total_new} records")

    if not new_records:
        print(f"[build-index] worker {worker_id}: no new images to index.")
        return

    captions = [record["caption"] for record in new_records]
    chunks = []
    for start in tqdm(range(0, len(captions), BATCH_SIZE), desc="embedding"):
        batch = captions[start:start + BATCH_SIZE]
        chunks.append(model.encode(batch, convert_to_numpy=True, show_progress_bar=False))
    new_embeddings = normalize(np.vstack(chunks).astype(np.float32))

    if num_workers > 1:
        part_jsonl = INDEX_DIR / f"captions.jsonl.part_{worker_id}"
        part_npy = INDEX_DIR / f"embeddings.part_{worker_id}.npy"
        with part_jsonl.open("w", encoding="utf-8") as handle:
            for record in new_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        np.save(part_npy, new_embeddings)
        print(f"[build-index] worker {worker_id}: saved {len(new_records)} records")
        return

    if EMBEDDINGS_NPY.exists():
        combined = np.vstack([np.load(EMBEDDINGS_NPY).astype(np.float32), new_embeddings])
    else:
        combined = new_embeddings

    if CAPTION_JSONL.exists():
        CAPTION_JSONL_TMP.write_bytes(CAPTION_JSONL.read_bytes())
        mode = "a"
    else:
        mode = "w"
    with CAPTION_JSONL_TMP.open(mode, encoding="utf-8") as handle:
        for record in new_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    np.save(EMBEDDINGS_NPY_TMP, combined)
    index = faiss.IndexFlatIP(DIM)
    index.add(combined)
    faiss.write_index(index, str(FAISS_INDEX_TMP))
    atomic_replace(CAPTION_JSONL_TMP, CAPTION_JSONL)
    atomic_replace(EMBEDDINGS_NPY_TMP, EMBEDDINGS_NPY)
    atomic_replace(FAISS_INDEX_TMP, FAISS_INDEX)
    for path in (CAPTION_JSONL_TMP, EMBEDDINGS_NPY_TMP, FAISS_INDEX_TMP):
        if path.exists():
            path.unlink()
    print(f"[build-index] done. total records: {combined.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MiniLM caption index for CC3M.")
    parser.add_argument("--cc3m-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    configure(args.cc3m_dir, args.index_dir)
    build_index(force_rebuild=args.force_rebuild, worker_id=args.worker_id, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
