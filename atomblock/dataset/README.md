# AtomBlock-WebUI construction

These scripts build the dataset used to train the UI detector. Paths are arguments. Set `ATOMBLOCK_API_KEY` and `ATOMBLOCK_BASE_URL` for HTML generation. Do not commit either value.

The HTML prompt is `prompts/html_generation.md`. It is the text sent to the generator and is left in its original wording. The construction run called `qwen3.6-plus`.

## Order

1. Keep one screenshot from each near-duplicate group. Comparison is inside each shard folder. The hash is a 256-bit dHash (grayscale, Lanczos, 17 by 16). Hamming distance at most 5 links two images. The kept file is the largest in that folder.

```bash
python atomblock/dataset/dedup_screenshots.py \
  --src /path/to/screenshots \
  --dst /path/to/screenshots_kept \
  --log /path/to/dedup_log.json
```

2. Generate HTML. The prompt asks for `yolo-*` classes and an English `data-caption` on every image. Viewport width in the prompt is 1920.

```bash
export ATOMBLOCK_API_KEY=...
export ATOMBLOCK_BASE_URL=https://your-openai-compatible-endpoint/v1
python atomblock/dataset/run_generate_html.py \
  --input-dir /path/to/screenshots_kept \
  --output-dir /path/to/AtomBlock-WebUI/html \
  --model qwen3.6-plus \
  --workers 4 \
  --concurrency 50
```

3. Download CC3M, unpack Arrow shards, and build the caption index with `all-MiniLM-L6-v2` (inner-product FAISS, 384 dimensions).

```bash
python atomblock/dataset/download_cc3m.py --cache-dir /path/to/cc3m-cache
python atomblock/dataset/extract_cc3m.py --src /path/to/arrow --out /path/to/cc3m
python atomblock/dataset/run_caption_index.py \
  --cc3m-dir /path/to/cc3m \
  --index-dir /path/to/caption_index \
  --ngpus 8
```

4. Inject images and render. Each caption searches the top 50, reranks by log aspect-ratio gap, and samples one of the top 5. The image is resized to the placeholder size, not cropped. Playwright uses a 1920 by 1080 viewport, waits 2 seconds, and saves a full-page screenshot. Boxes come from `yolo-*` classes, with a tag fallback for `a`, `button`, `img`, `input`, `select`, `textarea`, and `dropdown`.

```bash
python atomblock/dataset/render_pages.py \
  --html-dir /path/to/AtomBlock-WebUI/html \
  --output-dir /path/to/AtomBlock-WebUI \
  --index-dir /path/to/caption_index \
  --workers 32
```

`--output-dir` receives `html_with_images/`, `images/`, `screenshot/`, `annotations_viz/`, and `labels/`.

`atomblock/read_yolo_boxes.py` only reads nodes that already have a `yolo-*` class. This renderer also applies the tag fallback used when the pages were built.
