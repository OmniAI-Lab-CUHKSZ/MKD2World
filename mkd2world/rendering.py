from __future__ import annotations

import io
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image


DEFAULT_RENDER_CONFIG = {
    "conf": 0.1,
    "iou": 0.1,
    "imgsz": 1920,
    "use_ocr": True,
    "use_vl": True,
    "yolo_url": "http://localhost:8001",
    "ocr_url": "http://localhost:8002",
    "qwen_url": "http://localhost:8000",
    "yolo_model": "output/yolo/atomblock/yolo26m_img1920_seed0_8gpu_b64/weights/best.pt",
}


def render_markdown_from_screenshot(screenshot_bytes: bytes, config: dict | None = None) -> str:
    render_config = DEFAULT_RENDER_CONFIG | (config or {})
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        image_path = tmp_dir / "screenshot.png"
        image.save(image_path)

        tsm_renderer_dir = Path(__file__).resolve().parents[1] / "tsm_renderer"
        sys.path.insert(0, str(tsm_renderer_dir))
        try:
            from atomblock_builder import build_atomblock_tree
            from md_atomblock_render import render_nodes
        finally:
            if str(tsm_renderer_dir) in sys.path:
                sys.path.remove(str(tsm_renderer_dir))

        tree = build_atomblock_tree(
            image_path=str(image_path),
            model_path=render_config.get("yolo_model", ""),
            conf=render_config["conf"],
            iou=render_config["iou"],
            imgsz=render_config.get("yolo_imgsz", render_config["imgsz"]),
            agnostic_nms=True,
            use_ocr=bool(render_config.get("use_ocr", True)),
            use_vl=render_config["use_vl"],
            device=render_config.get("device", 0),
            yolo_url=render_config["yolo_url"],
            ocr_url=render_config["ocr_url"],
            qwen_url=render_config["qwen_url"],
        )
        width = tree.attributes.get("width", image.width)
        height = tree.attributes.get("height", image.height)
        lines = [f"[Environment: {render_config.get('environment', 'Web')}]"]
        lines.extend(render_nodes(tree.children, width, height, indent=0))
        return "\n".join(lines)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
