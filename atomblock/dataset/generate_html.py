#!/usr/bin/env python3
"""Generate ontology-labeled HTML from reference screenshots.

The prompt is ``prompts/html_generation.md``. The client is an OpenAI-compatible
endpoint. Set ``ATOMBLOCK_API_KEY`` and ``ATOMBLOCK_BASE_URL`` before running.
The construction run used ``qwen3.6-plus``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import os
import random
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm

MAX_IMAGE_BYTES = 7 * 1024 * 1024
DEFAULT_PROMPT = Path(__file__).resolve().parent / "prompts" / "html_generation.md"

client: AsyncOpenAI | None = None
semaphore: asyncio.Semaphore | None = None
output_dir = ""


def _compress_image(image_path: str) -> str:
    """Return a JPEG under the request size cap, or the original path."""
    if os.path.getsize(image_path) <= MAX_IMAGE_BYTES:
        return image_path

    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    for quality in (85, 70, 50, 30):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        if buffer.tell() <= MAX_IMAGE_BYTES:
            tmp_path = f"/tmp/atomblock_html_{Path(image_path).name}.jpg"
            Path(tmp_path).write_bytes(buffer.getvalue())
            return tmp_path

    scale = 0.9
    while scale > 0.1:
        resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=70)
        if buffer.tell() <= MAX_IMAGE_BYTES:
            tmp_path = f"/tmp/atomblock_html_{Path(image_path).name}.jpg"
            Path(tmp_path).write_bytes(buffer.getvalue())
            return tmp_path
        scale -= 0.1
    return image_path


def encode_image(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


async def chat_with_images(text: str, image_paths: list[str] | None = None, model: str = "qwen3.6-plus", max_retries: int = 5):
    content: list[dict] = [{"type": "text", "text": text}]
    for path in image_paths or []:
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp", "gif") else "image/jpeg"
        if mime == "image/jpg":
            mime = "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encode_image(path)}"},
        })
    messages = [{"role": "user", "content": content}]
    assert client is not None and semaphore is not None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(model=model, messages=messages)
            return completion.choices[0].message.content, getattr(completion, "usage", None)
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.random()
                print(f"  [retry] {exc}; backoff {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise


def clean_response(response_text: str) -> str:
    response_text = response_text.strip()
    if response_text.startswith("```html"):
        response_text = response_text[len("```html"):].strip()
    elif response_text.startswith("```"):
        response_text = response_text[len("```"):].strip()
    if response_text.endswith("```"):
        response_text = response_text[:-len("```")].strip()
    return response_text


async def generate_file(text: str, image_paths: list[str] | None = None, model: str = "qwen3.6-plus", output_filename: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    response_text, usage = await chat_with_images(text, image_paths=image_paths, model=model)
    response_text = clean_response(response_text or "")
    if output_filename is None:
        output_filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    output_path = os.path.join(output_dir, output_filename)
    Path(output_path).write_text(response_text, encoding="utf-8")
    return output_path, usage


async def process_single(prompt_text: str, img_path: Path, input_dir: Path, model: str) -> str:
    rel_path = img_path.relative_to(input_dir)
    prefix = str(rel_path.with_suffix("")).replace(os.sep, "_")
    if len(prefix) > 200:
        prefix = prefix[:180] + "_" + hashlib.md5(prefix.encode()).hexdigest()[:8]
    output_name = f"{prefix}.html"
    output_path = os.path.join(output_dir, output_name)
    if os.path.exists(output_path):
        return f"[html] skip {rel_path} -> {output_name} (exists)"

    prepared_path = _compress_image(str(img_path))
    try:
        file_path, usage = await generate_file(
            prompt_text,
            image_paths=[prepared_path],
            model=model,
            output_filename=output_name,
        )
        result = f"[html] wrote {rel_path} -> {output_name}\n  saved: {file_path}"
        if usage:
            result += f"\n  tokens: {usage.prompt_tokens}/{usage.completion_tokens}"
        return result
    except Exception as exc:
        return f"[html] failed {rel_path} -> {output_name}\n  error: {exc}"


def collect_images(input_dir: Path, image_list: list[str] | None) -> list[Path]:
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    if image_list:
        image_files = []
        for line in image_list:
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = input_dir / path
            if path.is_file() and path.suffix.lower() in valid_exts:
                image_files.append(path)
        return sorted(set(image_files))
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in valid_exts)


async def generate_for_directory(
    input_dir: Path,
    model: str = "qwen3.6-plus",
    concurrency: int = 8,
    image_list: list[str] | None = None,
    prompt_path: Path = DEFAULT_PROMPT,
) -> None:
    global semaphore
    semaphore = asyncio.Semaphore(concurrency)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    image_files = collect_images(input_dir, image_list)
    if not image_files:
        print(f"[html] no images in {input_dir}")
        return

    print(f"[html] {len(image_files)} images, concurrency={concurrency}")
    tasks = [asyncio.create_task(process_single(prompt_text, img_path, input_dir, model)) for img_path in image_files]
    progress = tqdm(total=len(tasks), desc="[html]")
    for task in asyncio.as_completed(tasks):
        try:
            print(await task)
        except Exception as exc:
            print(f"[html] exception: {exc}")
        finally:
            progress.update(1)
    progress.close()


def main() -> None:
    global client, output_dir
    parser = argparse.ArgumentParser(description="Generate yolo-* HTML from reference screenshots.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--image-list", type=Path, default=None, help="One image path per line.")
    args = parser.parse_args()

    api_key = os.environ.get("ATOMBLOCK_API_KEY", "").strip()
    base_url = os.environ.get("ATOMBLOCK_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise SystemExit("Set ATOMBLOCK_API_KEY and ATOMBLOCK_BASE_URL.")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    output_dir = str(args.output_dir)
    image_list = None
    if args.image_list and args.image_list.is_file():
        image_list = args.image_list.read_text(encoding="utf-8").splitlines()
    asyncio.run(generate_for_directory(
        args.input_dir,
        model=args.model,
        concurrency=args.concurrency,
        image_list=image_list,
        prompt_path=args.prompt,
    ))


if __name__ == "__main__":
    main()
