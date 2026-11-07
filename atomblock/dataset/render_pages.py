#!/usr/bin/env python3
"""Inject CC3M images, render full-page screenshots, and write YOLO labels.

Released layout under ``--output-dir``:

- ``html_with_images/`` image-injected HTML
- ``images/`` resized CC3M files
- ``screenshot/`` full-page captures
- ``annotations_viz/`` boxes drawn on those captures
- ``labels/`` YOLO text, one file per page

Retrieval encodes ``data-caption`` with MiniLM-L6-v2, takes the FAISS top 50,
reranks by aspect-ratio mismatch, and samples one of the top 5. Nodes without
a ``yolo-*`` class fall back to the tag name, matching the generator prompt.
Workers use the fork start method so they inherit this process configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tarfile
from multiprocessing import get_context
from pathlib import Path

import faiss
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

VIEWPORT_W = 1920
VIEWPORT_H = 1080
WAIT_MS = 2000

CLASS_NAMES = [
    "yolo-button", "yolo-link", "yolo-input", "yolo-image",
    "yolo-icon", "yolo-checkbox", "yolo-radio", "yolo-select",
    "yolo-block-nav", "yolo-block-sidebar", "yolo-block-footer",
    "yolo-block-form", "yolo-block-table", "yolo-block",
]
CLASS_ID_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
COLOR_MAP = {
    "button": (255, 59, 48),
    "link": (52, 199, 89),
    "input": (0, 122, 255),
    "image": (255, 204, 0),
    "icon": (175, 82, 222),
    "checkbox": (90, 200, 250),
    "radio": (255, 149, 0),
    "select": (255, 0, 128),
    "block-nav": (0, 255, 255),
    "block-sidebar": (255, 105, 180),
    "block-footer": (220, 20, 60),
    "block-form": (255, 165, 0),
    "block-table": (30, 144, 255),
    "block": (128, 128, 128),
}

HTML_DIR: Path
SCREENSHOT_DIR: Path
VIZ_DIR: Path
LABELS_DIR: Path
IMAGES_DIR: Path
HTML_WITH_IMAGES_DIR: Path
INDEX_FAISS: Path
INDEX_JSONL: Path

random.seed(42)


def configure(html_dir: Path, output_dir: Path, index_dir: Path) -> None:
    global HTML_DIR, SCREENSHOT_DIR, VIZ_DIR, LABELS_DIR, IMAGES_DIR, HTML_WITH_IMAGES_DIR
    global INDEX_FAISS, INDEX_JSONL
    HTML_DIR = html_dir
    SCREENSHOT_DIR = output_dir / "screenshot"
    VIZ_DIR = output_dir / "annotations_viz"
    LABELS_DIR = output_dir / "labels"
    IMAGES_DIR = output_dir / "images"
    HTML_WITH_IMAGES_DIR = output_dir / "html_with_images"
    INDEX_FAISS = index_dir / "index.faiss"
    INDEX_JSONL = index_dir / "captions.jsonl"
    for path in (SCREENSHOT_DIR, VIZ_DIR, LABELS_DIR, IMAGES_DIR, HTML_WITH_IMAGES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_color(yolo_class: str) -> tuple[int, int, int]:
    stripped = yolo_class.replace("yolo-", "")
    if stripped in COLOR_MAP:
        return COLOR_MAP[stripped]
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def write_yolo_labels(bboxes: list, img_w: int, img_h: int, out_path: Path) -> None:
    lines = []
    for box in bboxes:
        classes = box["classes"]
        label = classes[0] if classes else None
        if label is None or label not in CLASS_ID_MAP:
            continue
        cls_id = CLASS_ID_MAP[label]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        xc = max(0.0, min(1.0, (x + w / 2) / img_w))
        yc = max(0.0, min(1.0, (y + h / 2) / img_h))
        nw = max(0.0, min(1.0, w / img_w))
        nh = max(0.0, min(1.0, h / img_h))
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def extract_yolo_bboxes(page):
    """Document coordinates for a full-page screenshot, with the tag fallback."""
    return page.evaluate(
        """
        () => {
            const data = [];
            const scrollX = window.pageXOffset;
            const scrollY = window.pageYOffset;
            const elements = document.querySelectorAll('[class*="yolo-"], [class*="dropdown"], a, button, img, input, select, textarea');
            for (const el of elements) {
                const rect = el.getBoundingClientRect();
                if (rect.width < 1 || rect.height < 1) continue;
                let classes = Array.from(el.classList).filter(c => c.startsWith('yolo-'));
                if (classes.length === 0) {
                    if (el.classList.contains('dropdown')) {
                        classes = ['yolo-select'];
                    } else if (el.tagName === 'A') {
                        classes = ['yolo-link'];
                    } else if (el.tagName === 'BUTTON') {
                        classes = ['yolo-button'];
                    } else if (el.tagName === 'IMG') {
                        classes = ['yolo-image'];
                    } else if (el.tagName === 'SELECT') {
                        classes = ['yolo-select'];
                    } else if (el.tagName === 'TEXTAREA') {
                        classes = ['yolo-input'];
                    } else if (el.tagName === 'INPUT') {
                        const type = el.getAttribute('type') || 'text';
                        if (type === 'checkbox') classes = ['yolo-checkbox'];
                        else if (type === 'radio') classes = ['yolo-radio'];
                        else classes = ['yolo-input'];
                    }
                }
                if (classes.length === 0) continue;
                data.push({
                    x: rect.left + scrollX,
                    y: rect.top + scrollY,
                    w: rect.width,
                    h: rect.height,
                    classes: classes
                });
            }
            return data;
        }
        """
    )


def draw_bboxes(image_path: Path, bboxes: list, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in bboxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        classes = box["classes"]
        label = classes[0] if classes else "yolo-?"
        color = get_color(label)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        text = label.replace("yolo-", "") if label.startswith("yolo-") else label
        text_w = len(text) * 6 + 4
        text_h = 12
        draw.rectangle([x, y - text_h, x + text_w, y], fill=color)
        draw.text((x + 2, y - text_h), text, fill=(255, 255, 255))
    img.save(out_path, "PNG")


class ImageInjector:
    """Retrieve a CC3M image for each ``data-caption``."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(INDEX_FAISS))
        self.records = []
        with INDEX_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self._dim_cache: dict[str, tuple[int, int]] = {}

    def search(self, caption: str, top_k: int = 50):
        emb = self.model.encode([caption], convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb.astype(np.float32)
        scores, indices = self.index.search(emb, top_k)
        return scores[0], indices[0]

    def get_image_dim(self, rec: dict):
        image_id = rec["image_id"]
        if image_id in self._dim_cache:
            return self._dim_cache[image_id]
        try:
            if rec["source_type"] == "file":
                with Image.open(rec["source_path"]) as img:
                    dim = img.size
            else:
                tar_path, member_name = rec["source_path"], image_id.split("::")[-1]
                with tarfile.open(tar_path, "r:*") as handle:
                    member = handle.getmember(member_name)
                    with Image.open(handle.extractfile(member)) as img:
                        dim = img.size
            self._dim_cache[image_id] = dim
            return dim
        except Exception:
            return None


def resize_to_target(src, dst_path: Path, target_w: int, target_h: int) -> None:
    with Image.open(src) as img:
        img = img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "PNG")


def parse_target_size(img_tag) -> tuple[int | None, int | None]:
    src = img_tag.get("src", "")
    match = re.search(r"placehold\.co/(\d+)x(\d+)", src, re.I)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"picsum\.photos/(\d+)/(\d+)", src, re.I)
    if match:
        return int(match.group(1)), int(match.group(2))
    width, height = img_tag.get("width"), img_tag.get("height")
    if width and height:
        try:
            return int(width), int(height)
        except ValueError:
            pass
    style = img_tag.get("style", "")
    width_match = re.search(r"width\s*:\s*(\d+)px", style, re.I)
    height_match = re.search(r"height\s*:\s*(\d+)px", style, re.I)
    if width_match and height_match:
        return int(width_match.group(1)), int(height_match.group(1))
    return None, None


def inject_images(html_path: Path, injector: ImageInjector, out_html_path: Path):
    html_text = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_text, "html.parser")
    img_tags = soup.find_all("img")
    if not img_tags:
        return False, html_text

    images_dir = IMAGES_DIR / html_path.stem
    images_dir.mkdir(parents=True, exist_ok=True)
    modified = False
    for img_tag in img_tags:
        caption = img_tag.get("data-caption", "").strip()
        target_w, target_h = parse_target_size(img_tag)
        if not target_w or not target_h:
            continue
        if caption and INDEX_FAISS.exists():
            _scores, indices = injector.search(caption, top_k=50)
        else:
            indices = np.array([])

        candidates = []
        for idx in indices:
            rec = injector.records[int(idx)]
            dim = injector.get_image_dim(rec)
            if dim is None:
                continue
            iw, ih = dim
            if iw <= 0 or ih <= 0:
                continue
            ratio_diff = abs(np.log(target_w / target_h) - np.log(iw / ih))
            candidates.append((ratio_diff, rec))

        if not candidates:
            pool = injector.records
            if len(pool) > 100:
                pool = random.sample(pool, 100)
            for rec in pool:
                dim = injector.get_image_dim(rec)
                if dim is None:
                    continue
                iw, ih = dim
                if iw <= 0 or ih <= 0:
                    continue
                ratio_diff = abs(np.log(target_w / target_h) - np.log(iw / ih))
                candidates.append((ratio_diff, rec))
        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0])
        _ratio, chosen = random.choice(candidates[:5])
        ext = Path(chosen["image_id"]).suffix or ".jpg"
        dst_path = images_dir / f"img_{random.randint(100000, 999999)}{ext}"
        try:
            if chosen["source_type"] == "file":
                resize_to_target(chosen["source_path"], dst_path, target_w, target_h)
            else:
                tar_path, member_name = chosen["source_path"], chosen["image_id"].split("::")[-1]
                with tarfile.open(tar_path, "r:*") as handle:
                    member = handle.getmember(member_name)
                    resize_to_target(handle.extractfile(member), dst_path, target_w, target_h)
        except Exception as exc:
            print(f"[inject] failed to resize image for {html_path.name}: {exc}")
            continue

        img_tag["src"] = os.path.relpath(dst_path, start=out_html_path.parent).replace("\\", "/")
        style = re.sub(r"object-fit\s*:\s*[^;]+;?", "", img_tag.get("style", "")).strip(" ;")
        if style:
            img_tag["style"] = style
        elif "style" in img_tag.attrs:
            del img_tag.attrs["style"]
        modified = True
    return modified, str(soup)


def _worker_capture(working_html_path: str) -> dict:
    html_path = Path(working_html_path)
    name = html_path.stem
    screenshot_path = SCREENSHOT_DIR / f"{name}.png"
    viz_path = VIZ_DIR / f"{name}.png"
    label_path = LABELS_DIR / f"{name}.txt"
    if screenshot_path.exists() and viz_path.exists() and label_path.exists():
        return {"name": name, "boxes_count": "skipped"}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--single-process",
                ],
            )
            page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
            page.goto(html_path.as_uri())
            page.wait_for_timeout(WAIT_MS)
            page.screenshot(path=str(screenshot_path), full_page=True, timeout=120000)
            bboxes = extract_yolo_bboxes(page)
            browser.close()
        draw_bboxes(screenshot_path, bboxes, viz_path)
        with Image.open(screenshot_path) as image:
            img_w, img_h = image.size
        write_yolo_labels(bboxes, img_w, img_h, label_path)
        return {"name": name, "boxes_count": len(bboxes)}
    except Exception as exc:
        print(f"[render] error processing {html_path.name}: {exc}")
        return {"name": name, "boxes_count": f"error: {exc}"}


def _worker_inject(html_path_strs: list[str]) -> list[str]:
    injector = ImageInjector() if INDEX_FAISS.exists() and INDEX_JSONL.exists() else None
    working_paths = []
    for html_path_str in html_path_strs:
        html_path = Path(html_path_str)
        working = html_path
        if injector is not None:
            out_path = HTML_WITH_IMAGES_DIR / html_path.name
            modified, new_html = inject_images(html_path, injector, out_path)
            if modified:
                out_path.write_text(new_html, encoding="utf-8")
                working = out_path
        working_paths.append(str(working))
    return working_paths


def run(single_html: str | None = None, workers: int = 1) -> None:
    if single_html:
        html_path = Path(single_html)
        if not html_path.is_absolute():
            html_path = HTML_DIR / html_path
        if not html_path.exists() and not html_path.suffix:
            html_path = html_path.with_suffix(".html")
        if not html_path.exists():
            print(f"[render] file not found: {html_path}")
            return
        html_files = [html_path]
        print(f"[render] single file: {html_path.name}")
    else:
        html_files = sorted(set(HTML_DIR.glob("*.html")) | set(HTML_DIR.glob("*.txt")))
        if not html_files:
            print(f"[render] no HTML files found in {HTML_DIR}")
            return

    has_index = INDEX_FAISS.exists() and INDEX_JSONL.exists()
    print("[render] caption index found" if has_index else "[render] caption index not found, skipping image injection")

    if has_index and len(html_files) > 1 and workers > 1:
        print(f"[render] starting {workers} workers for image injection")
        chunks: list[list[str]] = [[] for _ in range(workers)]
        for index, html_path in enumerate(html_files):
            chunks[index % workers].append(str(html_path))
        with get_context("fork").Pool(workers) as pool:
            phase1 = list(tqdm(pool.imap(_worker_inject, chunks), total=len(chunks), desc="[render] injecting"))
        working_paths = [path for chunk in phase1 for path in chunk]
    else:
        injector = ImageInjector() if has_index else None
        if injector is not None:
            print(f"[render] index loaded: {len(injector.records)} records")
        working_paths = []
        for html_path in tqdm(html_files, desc="[render] injecting"):
            working = html_path
            if injector is not None:
                out_path = HTML_WITH_IMAGES_DIR / html_path.name
                modified, new_html = inject_images(html_path, injector, out_path)
                if modified:
                    out_path.write_text(new_html, encoding="utf-8")
                    print(f"[render] injected real images for {html_path.name}")
                    working = out_path
            working_paths.append(str(working))

    if workers > 1:
        print(f"[render] starting {workers} workers for screenshots and labels")
        with get_context("fork").Pool(workers) as pool:
            results = list(tqdm(pool.imap(_worker_capture, working_paths), total=len(working_paths), desc="[render] capturing"))
    else:
        results = [_worker_capture(path) for path in tqdm(working_paths, desc="[render] capturing")]

    print(f"\n[render] finished {len(results)}/{len(html_files)} files")
    for result in results:
        print(f"  - {result['name']}: {result['boxes_count']} boxes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject CC3M images and render AtomBlock-WebUI pages.")
    parser.add_argument("--html-dir", type=Path, required=True, help="Generated HTML, one page per file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset root for screenshots, labels, and injected HTML.")
    parser.add_argument("--index-dir", type=Path, required=True, help="Directory with index.faiss and captions.jsonl.")
    parser.add_argument("single_html", nargs="?", help="One HTML file, absolute or relative to --html-dir.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    configure(args.html_dir, args.output_dir, args.index_dir)
    run(single_html=args.single_html, workers=args.workers)


if __name__ == "__main__":
    main()
