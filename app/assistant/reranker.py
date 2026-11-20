from __future__ import annotations

import math
import threading
from typing import Any

from app.assistant.config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, enabled: bool | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.enabled = settings.reranker_enabled if enabled is None else enabled
        self._model = None
        self.last_used = False

    def rerank(self, question: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        self.last_used = False
        if not self.enabled or not rows:
            return rows[:limit]

        pairs: list[tuple[str, str]] = []
        candidates: list[dict[str, Any]] = []
        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            candidates.append(row)
            pairs.append((question, text))

        if not pairs:
            return rows[:limit]

        try:
            import torch

            model = self._load_model()
            # Lấy logit thô rồi tự áp sigmoid: có model (bge-reranker) đã cấu hình sẵn Sigmoid,
            # để mặc định thì điểm bị sigmoid hai lần và dồn hết về quanh 0,5.
            raw_scores = model.predict(
                pairs, batch_size=16, show_progress_bar=False, activation_fn=torch.nn.Identity()
            )
        except Exception:
            self.enabled = False
            return rows[:limit]

        scored: list[dict[str, Any]] = []
        for row, score in zip(candidates, raw_scores):
            value = _to_float(score)
            scored.append({
                **row,
                "reranker_score": _sigmoid(value),
                "reranker_raw_score": value,
            })
        scored.sort(key=lambda item: item.get("reranker_score", 0.0), reverse=True)
        self.last_used = True
        return scored[:limit]

    def _load_model(self):
        if self._model is None:
            self._model = load_shared_reranker_model(self.model_name)
        return self._model


_SHARED_MODELS: dict[tuple[str, str], Any] = {}
_SHARED_LOCK = threading.Lock()


def load_shared_reranker_model(model_name: str):
    """Một bản model cho cả tiến trình; luồng nạp sẵn và request đầu tiên không nạp trùng."""
    device = resolve_device(settings.reranker_device)
    key = (model_name, device)
    with _SHARED_LOCK:
        model = _SHARED_MODELS.get(key)
        if model is None:
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {"device": device, "max_length": settings.reranker_max_length}
            if device.startswith("cuda"):
                import torch

                kwargs["model_kwargs"] = {"dtype": torch.float16}
            model = CrossEncoder(model_name, **kwargs)
            _SHARED_MODELS[key] = model
        return model


def resolve_device(setting: str) -> str:
    """'auto' → 'cuda' nếu có GPU, không thì 'cpu'; giá trị khác giữ nguyên."""
    setting = (setting or "auto").strip().lower()
    if setting != "auto":
        return setting
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _to_float(value: Any) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)
