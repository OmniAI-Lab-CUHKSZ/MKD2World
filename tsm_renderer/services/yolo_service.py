"""YOLO FastAPI Service for UI element detection."""

import gc
import os
import threading
from contextlib import nullcontext
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from ultralytics import YOLO
import uvicorn

app = FastAPI()
_models: Dict[str, YOLO] = {}
_model_inference_lock = threading.Lock()
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "")
_shutdown_requested = False

YOLO_V4_CLASS_NAMES = [
    'button', 'link', 'input', 'image', 'icon', 'checkbox', 'radio', 'select',
    'block-nav', 'block-sidebar', 'block-footer', 'block-form', 'block-table', 'block'
]


class DetectReq(BaseModel):
    image_path: str
    model_path: str
    conf: float = 0.1
    iou: float = 0.1
    imgsz: int = 1920
    agnostic_nms: bool = True


@app.get("/health")
def health():
    return {"status": "ok" if not _shutdown_requested else "stopping", "service": "yolo"}


def _cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[YOLO Service] torch cuda synchronize failed: {exc}", flush=True)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as exc:
        print(f"[YOLO Service] torch cuda cleanup skipped: {exc}", flush=True)


def _cleanup_models() -> None:
    global _shutdown_requested
    _shutdown_requested = True
    with _model_inference_lock:
        print(f"[YOLO Service] Releasing {len(_models)} model(s).", flush=True)
        for model_path, model in list(_models.items()):
            try:
                inner_model = getattr(model, "model", None)
                if inner_model is not None and hasattr(inner_model, "to"):
                    inner_model.to("cpu")
            except Exception as exc:
                print(f"[YOLO Service] failed to move model to CPU ({model_path}): {exc}", flush=True)
        _models.clear()
    _cuda_cleanup()


@app.on_event("shutdown")
def shutdown_event():
    _cleanup_models()


@app.post("/detect")
def detect(req: DetectReq):
    if _shutdown_requested:
        return {"error": "YOLO service is shutting down"}
    results = None
    try:

        with _model_inference_lock:
            if req.model_path not in _models:
                print(f"[YOLO Service] Loading model: {req.model_path}")
                _models[req.model_path] = YOLO(req.model_path)
            model = _models[req.model_path]
            predict_kwargs = {
                "verbose": False,
                "conf": req.conf,
                "iou": req.iou,
                "imgsz": req.imgsz,
                "end2end": False,
                "agnostic_nms": req.agnostic_nms,
            }
            if YOLO_DEVICE:
                predict_kwargs["device"] = YOLO_DEVICE
            try:
                import torch

                infer_context = torch.inference_mode()
            except Exception:
                infer_context = nullcontext()
            with infer_context:
                results = model(req.image_path, **predict_kwargs)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box, cid, conf_val in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
            ):
                detections.append({
                    "bbox": [float(v) for v in box],
                    "class_id": int(cid),
                    "class_name": YOLO_V4_CLASS_NAMES[int(cid)],
                    "confidence": float(conf_val),
                })
    finally:
        if results is not None:
            del results
        if os.environ.get("YOLO_EMPTY_CACHE_EACH_REQUEST", "0") == "1":
            _cuda_cleanup()
    return {"detections": detections}


if __name__ == "__main__":
    port = int(os.environ.get("YOLO_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
