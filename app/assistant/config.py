from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class V2Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    reranker_enabled: bool = _get_bool("RERANKER_ENABLED", False)
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "40"))
    reranker_min_score: float = float(os.getenv("RERANKER_MIN_SCORE", "0.40"))
    # Cross-encoder mmarco chấm review tiếng Việt rất thấp (dòng đúng thường < 0,1), nên
    # ngưỡng tuyệt đối loại gần hết. Giữ dòng đạt RERANKER_MIN_SCORE hoặc đạt tỷ lệ này
    # so với dòng tốt nhất.
    reranker_relative_cutoff: float = float(os.getenv("RERANKER_RELATIVE_CUTOFF", "0.30"))
    # "auto": GPU nếu có (nạp fp16 để model lớn như bge-reranker-v2-m3 vừa GPU 2 GB), không thì CPU.
    reranker_device: str = os.getenv("RERANKER_DEVICE", "auto")
    reranker_max_length: int = int(os.getenv("RERANKER_MAX_LENGTH", "256"))
    # Mặc định CPU: mỗi câu hỏi chỉ nhúng một câu, còn GPU nhỏ để dành cho reranker.
    # Đặt EMBEDDING_DEVICE=cuda khi chạy scripts/reindex_vectors.py cho nhanh.
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    # Mỗi mô hình embedding một bảng (số chiều khác nhau), vd. opinion_chunks_bge_m3 cho BAAI/bge-m3.
    vector_table: str = os.getenv("VECTOR_TABLE", "opinion_chunks")
    retrieval_candidate_limit: int = int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "40"))
    retrieval_final_limit: int = int(os.getenv("RETRIEVAL_FINAL_LIMIT", "6"))
    semantic_min_score: float = float(os.getenv("SEMANTIC_MIN_SCORE", "0.30"))
    hybrid_retrieval_enabled: bool = _get_bool("HYBRID_RETRIEVAL_ENABLED", True)
    retrieval_cache_ttl_seconds: int = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "1800"))
    db_auto_create: bool = _get_bool("DB_AUTO_CREATE", True)
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    groq_fallback_model: str = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm_max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
    # Gemini 2.5 tính token "suy nghĩ" vào maxOutputTokens; với 500 token, phần suy nghĩ
    # chiếm gần hết và JSON câu trả lời bị cắt. 0 = tắt suy nghĩ (nhanh hơn, đủ cho tác vụ này);
    # -1 = để Gemini tự quyết.
    gemini_thinking_budget: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    # Số lần thử lại khi Gemini báo 429 (vượt giới hạn theo phút). Web để 0 cho người dùng khỏi
    # phải chờ; bộ đo Ragas đặt cao hơn để câu trả lời đúng là do LLM viết.
    llm_retry_on_429: int = int(os.getenv("LLM_RETRY_ON_429", "0"))
    phobert_model_path: str = os.getenv("PHOBERT_MODEL_PATH", "models/phobert/best_model.pt")


settings = V2Settings()
