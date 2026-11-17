"""OCR Service: Standard PaddleOCR via http.server.

Run:  conda run -n gui_ocr python ocr_vl_service.py
Port: 8002
"""

import gc
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import json
import queue
import signal
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import List, Dict

# Globals set in __main__
_req_queue: multiprocessing.Queue = None
_res_queue: multiprocessing.Queue = None
_workers: List[multiprocessing.Process] = []
_ocr_model = None
_ocr_lock = threading.Lock()
_worker_ready = False
_pending_results: Dict[str, dict] = {}
_pending_lock = threading.Lock()
_shutdown_requested = False
_server = None
_process_mode = "inline"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _env_int(name: str):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _paddleocr_kwargs() -> Dict:
    """GUI screenshots are not document scans; keep geometry unchanged by default."""
    kwargs = {
        "lang": os.environ.get("OCR_LANG", "ch"),
        "use_doc_orientation_classify": _env_bool("OCR_USE_DOC_ORIENTATION_CLASSIFY", False),
        "use_doc_unwarping": _env_bool("OCR_USE_DOC_UNWARPING", False),
        "use_textline_orientation": _env_bool("OCR_USE_TEXTLINE_ORIENTATION", False),
    }
    optional_float_params = {
        "text_det_thresh": "OCR_TEXT_DET_THRESH",
        "text_det_box_thresh": "OCR_TEXT_DET_BOX_THRESH",
        "text_det_unclip_ratio": "OCR_TEXT_DET_UNCLIP_RATIO",
        "text_rec_score_thresh": "OCR_TEXT_REC_SCORE_THRESH",
    }
    optional_int_params = {
        "text_det_limit_side_len": "OCR_TEXT_DET_LIMIT_SIDE_LEN",
    }
    optional_str_params = {
        "text_det_limit_type": "OCR_TEXT_DET_LIMIT_TYPE",
    }
    for param, env_name in optional_float_params.items():
        value = _env_float(env_name)
        if value is not None:
            kwargs[param] = value
    for param, env_name in optional_int_params.items():
        value = _env_int(env_name)
        if value is not None:
            kwargs[param] = value
    for param, env_name in optional_str_params.items():
        value = os.environ.get(env_name)
        if value:
            kwargs[param] = value
    return kwargs


def _ocr_worker(request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
    """Persistent worker: load PaddleOCR once, serve many requests."""
    from paddleocr import PaddleOCR

    kwargs = _paddleocr_kwargs()
    print(f"[OCR Worker] Loading PaddleOCR with kwargs={kwargs}")
    ocr = None
    try:
        ocr = PaddleOCR(**kwargs)
        print("[OCR Worker] Model loaded.")
        response_queue.put({"_ready": True})

        while True:
            req = request_queue.get()
            if req is None:
                print("[OCR Worker] Shutdown.")
                break

            request_id = req["request_id"]
            image_path = req["image_path"]
            try:
                response_queue.put({"request_id": request_id, "texts": _predict_ocr(ocr, image_path)})
            except Exception as e:
                response_queue.put({"request_id": request_id, "error": str(e)})
    finally:
        if ocr is not None:
            del ocr
        _accelerator_cleanup()


def _accelerator_cleanup() -> None:
    gc.collect()
    try:
        import paddle

        cuda_api = getattr(getattr(paddle, "device", None), "cuda", None)
        if cuda_api is not None:
            for method_name in ("synchronize", "empty_cache"):
                method = getattr(cuda_api, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:
                        print(f"[OCR Service] paddle cuda {method_name} failed: {exc}", flush=True)
    except Exception as exc:
        print(f"[OCR Service] paddle cleanup skipped: {exc}", flush=True)
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[OCR Service] torch cuda synchronize failed: {exc}", flush=True)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _predict_ocr(ocr, image_path: str) -> List[Dict]:
    result = None
    try:
        result = ocr.predict(image_path)
        r = result[0].json["res"]
        texts = r["rec_texts"]
        boxes = r["rec_boxes"]
        scores = r["rec_scores"]

        data: List[Dict] = []
        for text, box, score in zip(texts, boxes, scores):
            x1, y1, x2, y2 = box
            data.append({
                "text": text,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(score),
            })
        return data
    finally:
        if result is not None:
            del result


def _start_inline_ocr() -> None:
    global _ocr_model, _worker_ready
    from paddleocr import PaddleOCR

    kwargs = _paddleocr_kwargs()
    print(f"[OCR Service] Loading inline PaddleOCR with kwargs={kwargs}", flush=True)
    _ocr_model = PaddleOCR(**kwargs)
    _worker_ready = True
    print("[OCR Service] Inline model loaded.", flush=True)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = False
    block_on_close = True


def _workers_alive() -> bool:
    if _process_mode == "inline":
        return _worker_ready and _ocr_model is not None
    return bool(_workers) and any(worker.is_alive() for worker in _workers)


def _request_shutdown(signum, frame):
    global _shutdown_requested
    if not _shutdown_requested:
        print(f"[OCR Service] Received signal {signum}; requesting shutdown.", flush=True)
    _shutdown_requested = True
    if _server is not None:
        threading.Thread(target=_server.shutdown, daemon=True).start()


def _close_queue(q):
    if q is None:
        return
    try:
        q.close()
    except Exception:
        pass
    try:
        q.join_thread()
    except Exception:
        pass


def _shutdown_workers():
    global _worker_ready
    if _req_queue is not None:
        for _ in _workers:
            try:
                _req_queue.put(None)
            except Exception:
                break
    for worker in _workers:
        worker.join(timeout=10)
        if worker.is_alive():
            print(f"[OCR Service] Worker {worker.pid} still alive; terminate.", flush=True)
            worker.terminate()
            worker.join(timeout=10)
        if worker.is_alive():
            print(f"[OCR Service] Worker {worker.pid} still alive; kill.", flush=True)
            worker.kill()
            worker.join(timeout=5)
    _close_queue(_req_queue)
    _close_queue(_res_queue)
    _worker_ready = False


def _shutdown_inline_ocr():
    global _ocr_model, _worker_ready
    if _ocr_model is not None:
        print("[OCR Service] Releasing inline OCR model.", flush=True)
        model = _ocr_model
        _ocr_model = None
        del model
    _worker_ready = False
    _accelerator_cleanup()


def _get_result_for_request(request_id: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _pending_lock:
            cached = _pending_results.pop(request_id, None)
        if cached is not None:
            return cached
        remaining = max(0.1, deadline - time.time())
        try:
            result = _res_queue.get(timeout=min(1.0, remaining))
        except queue.Empty:
            continue
        result_id = result.get("request_id")
        if result_id == request_id:
            return result
        if result_id is not None:
            with _pending_lock:
                _pending_results[result_id] = result
    raise TimeoutError("OCR timeout")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            alive = _workers_alive()
            self._send_json({
                "status": "ok" if alive else "error",
                "service": "ocr",
                "mode": _process_mode,
                "ready": alive and _worker_ready,
                "workers": len(_workers),
                "alive_workers": sum(1 for worker in _workers if worker.is_alive()),
            })
        elif self.path == "/shutdown":
            # Paddle can delay SIGTERM after CUDA init; this endpoint exits cleanly.
            self._send_json({"status": "stopping", "service": "ocr"})
            _request_shutdown(0, None)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/ocr":
            if _shutdown_requested:
                self._send_json({"error": "OCR service is shutting down"}, 503)
                return
            if not _workers_alive():
                self._send_json({"error": "OCR worker not alive"}, 503)
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                req = json.loads(body)
            except Exception as e:
                self._send_json({"error": f"bad request: {e}"}, 400)
                return

            image_path = req.get("image_path")
            if not image_path:
                self._send_json({"error": "missing image_path"}, 400)
                return

            if _process_mode == "inline":
                try:
                    with _ocr_lock:
                        texts = _predict_ocr(_ocr_model, image_path)
                    result = {"texts": texts}
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
                    return
            else:
                request_id = f"{time.time_ns()}-{id(self)}"
                _req_queue.put({"request_id": request_id, "image_path": image_path})
                try:
                    result = _get_result_for_request(request_id, timeout=300)
                except Exception:
                    self._send_json({"error": "OCR timeout"}, 504)
                    return

            if "error" in result:
                self._send_json({"error": result["error"]}, 500)
            else:
                self._send_json(result)
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        print(f"[HTTP] {format % args}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    _process_mode = os.environ.get("OCR_PROCESS_MODE", "inline").strip().lower()
    if _process_mode in {"inline", "inprocess", "single"}:
        _process_mode = "inline"
        try:
            _start_inline_ocr()
        except Exception as e:
            print(f"[OCR Service] Inline OCR failed to start: {e}", flush=True)
            _shutdown_inline_ocr()
            exit(1)
    elif _process_mode in {"multiprocess", "worker", "workers"}:
        _process_mode = "multiprocess"
        worker_count = int(os.environ.get("OCR_WORKERS", "1"))
        print(f"[OCR Service] Starting {worker_count} worker process(es)...")
        _req_queue = multiprocessing.Queue()
        _res_queue = multiprocessing.Queue()
        for _ in range(worker_count):
            worker = multiprocessing.Process(
                target=_ocr_worker, args=(_req_queue, _res_queue)
            )
            worker.start()
            _workers.append(worker)

        ready_timeout = int(os.environ.get("OCR_READY_TIMEOUT", "900"))
        print(f"[OCR Service] Waiting for model load (timeout={ready_timeout}s)...")
        try:
            ready = 0
            deadline = time.time() + ready_timeout
            while ready < worker_count and time.time() < deadline:
                msg = _res_queue.get(timeout=max(0.1, deadline - time.time()))
                if msg.get("_ready"):
                    ready += 1
            if ready == worker_count:
                _worker_ready = True
                print(f"[OCR Service] Workers ready: {ready}/{worker_count}.")
            else:
                raise TimeoutError(f"only {ready}/{worker_count} OCR workers became ready")
        except Exception as e:
            print(f"[OCR Service] Workers failed to start: {e}")
            _shutdown_workers()
            exit(1)
    else:
        print(f"[OCR Service] Unknown OCR_PROCESS_MODE={_process_mode!r}", flush=True)
        exit(1)

    port = int(os.environ.get("OCR_PORT", "8002"))
    _server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[OCR Service] HTTP server on port {port}")
    try:
        _server.serve_forever()
    finally:
        print("[OCR Service] Shutting down...")
        if _server is not None:
            _server.server_close()
        if _process_mode == "inline":
            _shutdown_inline_ocr()
        else:
            _shutdown_workers()
