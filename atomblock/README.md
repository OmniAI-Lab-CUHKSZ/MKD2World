# AtomBlock-WebUI


`dataset/` builds the pages: dHash filtering, HTML generation, CC3M retrieval, and Playwright rendering. Commands are in `dataset/README.md`.

`read_yolo_boxes.py` opens HTML in headless Chromium and returns boxes for nodes whose class starts with `yolo-`. Types are the classes injected before rendering. Coordinates are the laid-out DOM boxes. The dataset builder in `dataset/render_pages.py` also applies the tag fallback used for the released labels.

```bash
python atomblock/read_yolo_boxes.py --html page.html
```

`train_yolo.sh` wraps `tsm_renderer/training/train_atomblock_yolo.py`. Point it at a YOLO-format dataset you prepared from those boxes.
