#!/usr/bin/env python3

import argparse
import json
import os
import uuid
from pathlib import Path

from atomblock_builder import build_atomblock_tree, DEFAULT_YOLO_URL, DEFAULT_OCR_URL, DEFAULT_QWEN_URL
from md_atomblock_render import render_nodes
import requests

HTTP = requests.Session()
HTTP.trust_env = False


def check_service_health(name: str, url: str, health_path: str = "/health") -> bool:
    try:
        r = HTTP.get(f"{url}{health_path}", timeout=5)
        if r.status_code == 200:
            print(f"   ✅ {name} ready at {url}")
            return True
    except Exception:
        pass
    try:
        r = HTTP.get(f"{url}/v1/models", timeout=5)
        if r.status_code == 200:
            print(f"   ✅ {name} ready at {url}")
            return True
    except Exception:
        pass
    print(f"   ❌ {name} not responding at {url}")
    return False


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="AtomBlock-WebUI → Markdown Pipeline (Remote Services)")
    parser.add_argument("--image_dir", type=str, required=True, help="\u8f93\u5165\u56fe\u7247\u76ee\u5f55")
    parser.add_argument("--model", type=str, required=True, help="YOLO v4 \u6743\u91cd\u8def\u5f84")
    parser.add_argument("--output", type=str, required=True, help="\u8f93\u51fa Markdown \u76ee\u5f55")
    parser.add_argument(
        "--output-map",
        type=Path,
        help="\u53ef\u9009 JSON \u5bf9\u8c61：\u56fe\u7247 stem \u5230\u6700\u7ec8 Markdown \u8def\u5f84；\u9010\u56fe\u539f\u5b50\u53d1\u5e03",
    )
    parser.add_argument("--conf", type=float, default=0.1, help="\u7f6e\u4fe1\u5ea6\u9608\u503c")
    parser.add_argument("--iou", type=float, default=0.1, help="IoU \u9608\u503c")
    parser.add_argument("--imgsz", "--yolo-imgsz", dest="imgsz", type=int, default=1920, help="YOLO \u63a8\u7406\u8f93\u5165\u5c3a\u5bf8")
    parser.add_argument("--agnostic-nms", action=argparse.BooleanOptionalAction, default=True, help="\u8de8\u7c7b\u522b NMS")
    parser.add_argument("--device", type=int, default=0, help="CUDA \u8bbe\u5907（YOLO service \u5185\u90e8\u4f7f\u7528，CLI \u900f\u4f20）")
    parser.add_argument("--no-ocr", action="store_true", help="\u8df3\u8fc7 OCR（\u4ec5 YOLO \u7ed3\u6784）")
    parser.add_argument("--use-vl", action="store_true", help="\u542f\u7528 Qwen VL \u5bf9 image/icon \u505a\u8bed\u4e49\u5206\u6790")
    parser.add_argument("--yolo-url", type=str, default=DEFAULT_YOLO_URL, help="YOLO Service URL")
    parser.add_argument("--ocr-url", type=str, default=DEFAULT_OCR_URL, help="OCR Service URL")
    parser.add_argument("--qwen-url", type=str, default=DEFAULT_QWEN_URL, help="Qwen Service URL")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_map = {}
    if args.output_map:
        output_map = json.loads(args.output_map.read_text(encoding="utf-8"))
        if not isinstance(output_map, dict):
            raise ValueError("--output-map must contain a JSON object")

    images = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    ])

    if not images:
        print(f"❌ \u672a\u627e\u5230\u56fe\u7247: {image_dir}")
        return 2

    print(f"🚀 AtomBlock-WebUI → Markdown (Remote Services)")
    print(f"   \u6a21\u578b: {args.model}")
    print(f"   conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, agnostic_nms={args.agnostic_nms}")
    print(f"   \u56fe\u7247\u6570: {len(images)}")
    print(f"   OCR={not args.no_ocr}, VL={args.use_vl}")
    print("⏳ \u68c0\u67e5\u670d\u52a1\u5065\u5eb7\u72b6\u6001...")
    if not check_service_health("YOLO", args.yolo_url):
        return 2
    if not args.no_ocr and not check_service_health("OCR", args.ocr_url):
        return 2
    if args.use_vl and not check_service_health("Qwen", args.qwen_url):
        return 2

    failed = []
    for idx, img_path in enumerate(images, 1):
        print(f"  [{idx}/{len(images)}] {img_path.name}")
        try:
            tree = build_atomblock_tree(
                image_path=str(img_path.absolute()),
                model_path=str(Path(args.model).absolute()),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                agnostic_nms=args.agnostic_nms,
                use_ocr=not args.no_ocr,
                use_vl=args.use_vl,
                device=args.device,
                yolo_url=args.yolo_url,
                ocr_url=args.ocr_url,
                qwen_url=args.qwen_url,
            )
            w = tree.attributes.get('width', 1920)
            h = tree.attributes.get('height', 1080)

            lines = ["[Environment: Web]\n"]
            lines.extend(render_nodes(tree.children, w, h, indent=0))

            md_content = "\n".join(lines).rstrip() + "\n"
            mapped_path = output_map.get(img_path.stem)
            out_path = Path(mapped_path) if mapped_path else output_dir / f"{img_path.stem}.md"
            _atomic_write_text(out_path, md_content)
            print(f"      → {out_path}")
        except Exception as e:
            failed.append(img_path.name)
            print(f"    ❌ \u9519\u8bef: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ \u5168\u90e8\u5b8c\u6210，\u8f93\u51fa\u76ee\u5f55: {output_dir.absolute()}")
    if failed:
        print(f"❌ \u5931\u8d25 {len(failed)}/{len(images)}: {failed[:20]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
