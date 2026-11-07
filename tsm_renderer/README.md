# TSM renderer

`rendering` in the `mkd2world` package calls `tsm_renderer/atomblock_builder.py`. The builder asks three HTTP services for evidence, then builds a UI tree:

1. YOLO26 type and box (`services/yolo_service.py`, default port 8001).
2. PaddleOCR text (`services/ocr_vl_service.py`, default port 8002). Set `use_ocr=False` to skip.
3. Qwen3.5-0.8B visual semantics for icons, images, and control state (URL passed as `qwen_url`). Set `use_vl=False` to skip.

```bash
PYTHONPATH=. python -c "import tsm_renderer.atomblock_builder as builder; print(builder.BLOCK_CLASSES)"
```
