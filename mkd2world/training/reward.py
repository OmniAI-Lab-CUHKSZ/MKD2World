"""Semantic-layout reward for Local World Model Markdown predictions."""

from __future__ import annotations

import math
from pathlib import Path

from mkd2world.training.markdown import MdElement, extract_predicted_markdown, parse_md_elements


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBEDDING_MODEL = PROJECT_ROOT / "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2"
GRID_MAX = 999.0
CENTER_SIGMA = 0.05
ELEMENT_COUNT_REWARD_VERSION = "elef1_excess_exp_v1"
DEFAULT_ELEMENT_COUNT_FREE_RATIO = 0.1
DEFAULT_ELEMENT_COUNT_ANCHOR_RATIO = 0.3
DEFAULT_ELEMENT_COUNT_ANCHOR_MULTIPLIER = 0.8
TYPE_COMPATIBILITY = {
    "text": {"text": 1.0, "link": 0.5},
    "link": {"text": 0.5, "link": 1.0, "button": 0.5, "image": 0.5, "icon": 0.5},
    "button": {"link": 0.5, "button": 1.0, "image": 0.5, "icon": 0.5},
    "image": {"link": 0.5, "button": 0.5, "image": 1.0, "icon": 0.5},
    "icon": {"link": 0.5, "button": 0.5, "image": 0.5, "icon": 1.0},
    "toggle": {"toggle": 1.0},
    "select": {"select": 1.0},
    "input": {"input": 1.0},
}


def finite_float(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, finite_float(value)))


def element_count_decay_rate(
    *,
    free_ratio: float = DEFAULT_ELEMENT_COUNT_FREE_RATIO,
    anchor_ratio: float = DEFAULT_ELEMENT_COUNT_ANCHOR_RATIO,
    anchor_multiplier: float = DEFAULT_ELEMENT_COUNT_ANCHOR_MULTIPLIER,
) -> float:
    """Derive the exponential decay rate from the configured anchor."""
    values = (free_ratio, anchor_ratio, anchor_multiplier)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("element-count reward parameters must be finite")
    if free_ratio < 0.0:
        raise ValueError("element_count_free_ratio must be non-negative")
    if anchor_ratio <= free_ratio:
        raise ValueError("element_count_anchor_ratio must exceed element_count_free_ratio")
    if not 0.0 < anchor_multiplier < 1.0:
        raise ValueError("element_count_anchor_multiplier must be in (0, 1)")
    return -math.log(anchor_multiplier) / (anchor_ratio - free_ratio)


def element_count_training_reward(
    metrics: dict[str, float],
    *,
    free_ratio: float = DEFAULT_ELEMENT_COUNT_FREE_RATIO,
    anchor_ratio: float = DEFAULT_ELEMENT_COUNT_ANCHOR_RATIO,
    anchor_multiplier: float = DEFAULT_ELEMENT_COUNT_ANCHOR_MULTIPLIER,
) -> dict[str, float | str]:
    """Apply the V4 count penalty without changing the raw EleF1 evaluator."""
    decay_rate = element_count_decay_rate(
        free_ratio=free_ratio,
        anchor_ratio=anchor_ratio,
        anchor_multiplier=anchor_multiplier,
    )
    pred_count = max(0.0, finite_float(metrics.get("pred_elements")))
    target_count = max(0.0, finite_float(metrics.get("target_elements")))
    overage_ratio = max(0.0, (pred_count - target_count) / max(target_count, 1.0))
    multiplier = (
        1.0
        if overage_ratio <= free_ratio
        else math.exp(-decay_rate * (overage_ratio - free_ratio))
    )
    raw_reward = clamp01(finite_float(metrics.get("EleF1", metrics.get("reward", 0.0))))
    training_reward = raw_reward * multiplier
    if not all(math.isfinite(value) for value in (overage_ratio, multiplier, training_reward)):
        raise FloatingPointError("element-count training reward produced a non-finite value")
    return {
        "training_reward": training_reward,
        "raw_ele_f1_reward": raw_reward,
        "element_count_overage_ratio": overage_ratio,
        "element_count_multiplier": multiplier,
        "element_count_decay_rate": decay_rate,
        "element_count_reward_version": ELEMENT_COUNT_REWARD_VERSION,
    }


def bbox_xyxy(element: MdElement) -> tuple[float, float, float, float]:
    return (
        element.cx - element.w / 2,
        element.cy - element.h / 2,
        element.cx + element.w / 2,
        element.cy + element.h / 2,
    )


def iou(a: MdElement, b: MdElement) -> float:
    ax1, ay1, ax2, ay2 = bbox_xyxy(a)
    bx1, by1, bx2, by2 = bbox_xyxy(b)
    if not all(math.isfinite(value) for value in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def has_finite_bbox(element: MdElement) -> bool:
    return all(math.isfinite(value) for value in bbox_xyxy(element))


def semantic_text(element: MdElement) -> str:
    text = element.content.strip()
    for prefix in ("block:", "icon:", "img:", "image:"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def semantic_soft_score(cosine: float, soft_threshold: float = 0.55, accept_threshold: float = 0.65) -> float:
    cosine = clamp01(cosine)
    if cosine < soft_threshold:
        return 0.0
    if cosine < accept_threshold:
        return clamp01(0.5 * (cosine - soft_threshold) / (accept_threshold - soft_threshold))
    return clamp01(0.5 + 0.5 * (cosine - accept_threshold) / (1.0 - accept_threshold))


def _matrix_value(matrix, row: int, col: int) -> float:
    try:
        return finite_float(matrix[row, col])
    except Exception:
        return finite_float(matrix[row][col])


def maximum_weight_match_sum(score_matrix) -> float:
    rows = len(score_matrix)
    cols = len(score_matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return 0.0
    if rows > cols:
        transposed = [[_matrix_value(score_matrix, row, col) for row in range(rows)] for col in range(cols)]
        return maximum_weight_match_sum(transposed)

    max_score = max(_matrix_value(score_matrix, row, col) for row in range(rows) for col in range(cols))
    cost = [[max_score - _matrix_value(score_matrix, row, col) for col in range(cols)] for row in range(rows)]
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [(p[j] - 1, j - 1) for j in range(1, cols + 1) if p[j] > 0]
    return sum(_matrix_value(score_matrix, row, col) for row, col in assignment)


class SemanticEmbedder:
    def __init__(self, model_name_or_path: str | Path | None = None, *, device: str | None = None):
        resolved_model = str(resolve_embedding_model(model_name_or_path))
        self.device = device
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": device} if device else {}
            self.model = SentenceTransformer(resolved_model, **kwargs)
            self.backend = "sentence_transformers"
        except ModuleNotFoundError:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.backend = "transformers"
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_model)
            self.model = AutoModel.from_pretrained(resolved_model).to(device or "cpu")
            self.model.eval()
        self._embedding_cache: dict[str, object] = {}

    def _encode_uncached(self, texts: list[str]):
        if self.backend == "sentence_transformers":
            return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return self._transformers_encode(texts)

    def prime(self, texts: list[str]) -> None:
        """Batch-encode unique texts so a rollout group does not re-encode targets."""
        missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        if not missing:
            return
        embeddings = self._encode_uncached(missing)
        for text, embedding in zip(missing, embeddings):
            self._embedding_cache[text] = embedding

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def cosine_matrix(self, pred: list[str], target: list[str]):
        import numpy as np

        if not pred or not target:
            return np.zeros((len(pred), len(target)), dtype=float)
        self.prime(pred + target)
        pred_emb = np.stack([self._embedding_cache[text] for text in pred])
        target_emb = np.stack([self._embedding_cache[text] for text in target])
        return np.matmul(pred_emb, target_emb.T)

    def _transformers_encode(self, texts: list[str]):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        device = self.device or "cpu"
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()


def type_compatibility(pred_type: str, target_type: str) -> float:
    """Return C_type(pred_type, target_type) from the EleF1 specification."""
    if pred_type == "block" or target_type == "block":
        return 1.0 if pred_type == target_type else 0.5
    return TYPE_COMPATIBILITY.get(pred_type, {}).get(target_type, 1.0 if pred_type == target_type else 0.0)


def normalized_center_distance(pred: MdElement, target: MdElement) -> float:
    diagonal = math.hypot(GRID_MAX, GRID_MAX)
    if diagonal <= 0 or not all(math.isfinite(value) for value in (pred.cx, pred.cy, target.cx, target.cy)):
        return 0.0
    return clamp01(math.hypot(pred.cx - target.cx, pred.cy - target.cy) / diagonal)


def center_similarity(pred: MdElement, target: MdElement, *, sigma: float = CENTER_SIGMA) -> float:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("center similarity sigma must be positive and finite")
    distance = normalized_center_distance(pred, target)
    return clamp01(math.exp(-(distance * distance) / (2 * sigma * sigma)))


def bbox_score(pred: MdElement, target: MdElement) -> float:
    """Compute the documented centre-distance/IoU position score."""
    return clamp01(0.5 * math.sqrt(clamp01(iou(pred, target))) + 0.5 * center_similarity(pred, target))


def element_match_metrics(pred: list[MdElement], target: list[MdElement], embedder: SemanticEmbedder) -> dict[str, float]:
    """Compute the maximum-match score and EleF1 for one Markdown prediction."""
    pred_count = len(pred)
    target_count = len(target)
    if not pred_count and not target_count:
        return {
            "matched_score": 0.0,
            "pred_elements": 0.0,
            "target_elements": 0.0,
            "ElePrecision": 0.0,
            "EleRecall": 0.0,
            "EleF1": 0.0,
            "metric_eligible": 0.0,
        }
    if not pred_count or not target_count:
        return {
            "matched_score": 0.0,
            "pred_elements": float(pred_count),
            "target_elements": float(target_count),
            "ElePrecision": 0.0,
            "EleRecall": 0.0,
            "EleF1": 0.0,
            "metric_eligible": 1.0,
        }
    pred_texts = [semantic_text(item) for item in pred]
    target_texts = [semantic_text(item) for item in target]
    cosines = embedder.cosine_matrix(pred_texts, target_texts)
    scores = [[0.0 for _ in target] for _ in pred]
    for row, pred_element in enumerate(pred):
        for col, target_element in enumerate(target):
            if not has_finite_bbox(pred_element) or not has_finite_bbox(target_element):
                continue
            try:
                cosine = cosines[row, col]
            except Exception:
                cosine = cosines[row][col]
            compatibility = type_compatibility(pred_element.element_type, target_element.element_type)
            scores[row][col] = (
                semantic_soft_score(cosine)
                * bbox_score(pred_element, target_element)
                * compatibility
            )
    matched = maximum_weight_match_sum(scores)
    return {
        "matched_score": matched,
        "pred_elements": float(pred_count),
        "target_elements": float(target_count),
        "ElePrecision": matched / pred_count,
        "EleRecall": matched / target_count,
        "EleF1": 2 * matched / (pred_count + target_count),
        "metric_eligible": 1.0,
    }


def semantic_bbox_score(pred: list[MdElement], target: list[MdElement], embedder: SemanticEmbedder) -> float:
    """Compatibility wrapper for callers that need only the EleF1 scalar."""
    return element_match_metrics(pred, target, embedder)["EleF1"]


def reward_from_markdown(model_output: str, target_markdown: str, embedder: SemanticEmbedder) -> dict[str, float]:
    predicted = extract_predicted_markdown(model_output)
    pred_elements = parse_md_elements(predicted)
    target_elements = parse_md_elements(target_markdown)
    metrics = element_match_metrics(pred_elements, target_elements, embedder)
    return {
        "reward": metrics["EleF1"],
        **metrics,
    }


def resolve_embedding_model(path_or_name: str | Path | None) -> str | Path:
    if path_or_name:
        return path_or_name
    if LOCAL_EMBEDDING_MODEL.exists():
        return LOCAL_EMBEDDING_MODEL
    return DEFAULT_EMBEDDING_MODEL
