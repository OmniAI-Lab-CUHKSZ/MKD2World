#!/usr/bin/env python3
"""Render one HTML page and read boxes from pre-injected yolo-* nodes.

This is the annotation step of AtomBlock-WebUI: element types are already
written into the HTML, and geometry comes from the laid-out DOM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_ORDER = [
    "button", "link", "input", "image", "icon", "checkbox", "radio", "select",
    "block-nav", "block-sidebar", "block-footer", "block-form", "block-table", "block",
]


def extract_live_boxes(page) -> list[dict]:
    return page.evaluate(
        """() => {
          const out = [];
          const seen = new Set();
          for (const el of document.querySelectorAll('[class*="yolo-"]')) {
            const cls = Array.from(el.classList).filter(c => c.startsWith('yolo-'));
            if (!cls.length) continue;
            const r = el.getBoundingClientRect();
            const x = r.x + window.scrollX;
            const y = r.y + window.scrollY;
            if (r.width < 1 || r.height < 1) continue;
            const key = cls.join(',') + ':' + [x, y, r.width, r.height].map(v => Math.round(v)).join(',');
            if (seen.has(key)) continue;
            seen.add(key);
            out.push({classes: cls, x, y, w: r.width, h: r.height, tag: el.tagName.toLowerCase()});
          }
          return out;
        }"""
    )


def render_html(html: str, *, width: int = 1280, height: int = 800) -> list[dict]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html, wait_until="load")
        boxes = extract_live_boxes(page)
        browser.close()
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Read yolo-* boxes from an HTML file.")
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()
    boxes = render_html(args.html.read_text(encoding="utf-8"), width=args.width, height=args.height)
    print(json.dumps({"count": len(boxes), "classes": CLASS_ORDER, "boxes": boxes}, ensure_ascii=True))


if __name__ == "__main__":
    main()
