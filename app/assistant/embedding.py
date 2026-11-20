from __future__ import annotations

import threading

from app.assistant.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.model_name = settings.embedding_model
        self._model = None
        self._dimension: int | None = None
        self.enabled = bool(self.model_name)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            vector = self.embed_text("dimension probe")
            self._dimension = len(vector)
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean = [str(text or "").strip() for text in texts if str(text or "").strip()]
        if not clean or not self.enabled:
            return []
        try:
            model = self._load_model()
            vectors = model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
            return [vector.astype(float).tolist() for vector in vectors]
        except Exception:
            self.enabled = False
            return []

    def _load_model(self):
        if self._model is None:
            self._model = load_shared_embedding_model(self.model_name)
        return self._model


# Dashboard (ghi vector khi phân tích) và trợ lý (tìm kiếm) mỗi bên tạo một EmbeddingService;
# dùng chung một bản model để không nạp hai lần (bge-m3 ~2,3 GB, nạp mất hơn một phút).
_SHARED_MODELS: dict[tuple[str, str], object] = {}
_SHARED_LOCK = threading.Lock()


def load_shared_embedding_model(model_name: str):
    from app.assistant.reranker import resolve_device

    device = resolve_device(settings.embedding_device)
    key = (model_name, device)
    with _SHARED_LOCK:  # luồng nạp sẵn lúc khởi động và request đầu tiên không nạp trùng
        model = _SHARED_MODELS.get(key)
        if model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {"device": device}
            if device.startswith("cuda"):
                import torch

                kwargs["model_kwargs"] = {"dtype": torch.float16}
            model = SentenceTransformer(model_name, **kwargs)
            _SHARED_MODELS[key] = model
        return model
