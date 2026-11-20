from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.assistant.cache import RedisCache
from app.assistant.config import settings
from app.assistant.query_analyzer import QueryAnalysis
from app.assistant.reranker import CrossEncoderReranker
from app.assistant.vector_store import PgVectorOpinionStore


STATISTICAL_INTENTS = {
    "SUMMARY",
    "COMPLAINT_SUMMARY",
    "PRICE_VALUE",
    "DELIVERY_SERVICE",
}

SEMANTIC_INTENTS = {
    "ASPECT_QA",
    "PRODUCT_FIT",
    "PRODUCT_QA",
    "RISK_CHECK",
    "RECOMMENDATION",
    "UNKNOWN",
}


class HybridRetrievalService:
    def __init__(
        self,
        vector_store: PgVectorOpinionStore | None = None,
        cache: RedisCache | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.vector_store = vector_store or PgVectorOpinionStore()
        self.cache = cache or RedisCache()
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(
        self,
        product_id: str,
        question: str,
        query: QueryAnalysis,
        structured_evidence: list[dict[str, Any]],
        limit: int = 6,
        allowed_review_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """allowed_review_ids: khi câu hỏi có khoảng thời gian, chỉ giữ ý kiến của các
        review này. Bảng vector chưa có cột ngày nên lọc sau khi tìm."""
        if not settings.hybrid_retrieval_enabled:
            return {
                "strategy": "postgres_only",
                "evidence": structured_evidence[:limit],
                "vector_used": False,
                "cache_hit": False,
            }

        cache_key = _cache_key(product_id, question, query)
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, dict):
            cached["cache_hit"] = True
            return cached

        strategy = "postgres_first" if query.intent in STATISTICAL_INTENTS else "vector_first"
        final_limit = min(limit, max(1, settings.retrieval_final_limit))
        vector_limit = max(final_limit, settings.retrieval_candidate_limit)
        if settings.reranker_enabled:
            vector_limit = max(vector_limit, settings.reranker_top_k)
        vector_rows = self._search_vector(product_id, question, query, limit=vector_limit)
        if allowed_review_ids is not None:
            vector_rows = [row for row in vector_rows if str(row.get("review_id") or "") in allowed_review_ids]
        vector_rows = self.reranker.rerank(_build_rerank_query(question, query), vector_rows, limit=vector_limit)
        if self.reranker.last_used and not query.has_filters():
            # Có lọc cứng thì mọi dòng đều đúng điều kiện: reranker chỉ dùng để xếp thứ tự.
            vector_rows = _keep_relevant(vector_rows)

        if vector_rows and query.intent in SEMANTIC_INTENTS and not query.has_filters():
            # Câu hỏi tra cứu không có điều kiện lọc: ý kiến SQL lúc này chỉ là các ý kiến
            # nổi bật chung của sản phẩm, không liên quan câu hỏi, nên không dùng để lấp chỗ.
            strategy = "vector_only"
            merged = _dedupe_by_review(vector_rows)
        elif strategy == "postgres_first":
            merged = _hybrid_rank(
                structured_evidence + vector_rows,
                query=query,
                qdrant_weight=0.35,
                structured_weight=0.65,
            )
        else:
            merged = _hybrid_rank(
                vector_rows + structured_evidence,
                query=query,
                qdrant_weight=0.65,
                structured_weight=0.35,
            )

        if self.reranker.last_used and _is_topic_lookup(query):
            # Hỏi tra cứu theo chủ đề ("thấm hút ra sao"): có hàng chục ý kiến cùng chủ đề, câu nào
            # trả lời đúng câu hỏi do reranker quyết định, không phải độ tự tin của nhãn chủ đề.
            # Câu thống kê (lọc khen/chê hoặc thời gian) giữ cách xếp theo hybrid_score.
            merged = sorted(merged, key=lambda row: float(row.get("reranker_score") or -1.0), reverse=True)
        if query.has_filters():
            # Mọi dòng đều đúng điều kiện; reranker lại ưu tiên câu ngắn chung chung
            # ("Không hài lòng.") nên đưa câu có nội dung cụ thể lên trước.
            merged = _informative_first(merged)

        result = {
            "strategy": strategy,
            "evidence": _select_diverse(merged, query, final_limit),
            "vector_used": bool(vector_rows),
            "cache_hit": False,
        }
        self.cache.set_json(cache_key, result, ttl_seconds=settings.retrieval_cache_ttl_seconds)
        return result

    def _search_vector(
        self,
        product_id: str,
        question: str,
        query: QueryAnalysis,
        limit: int,
    ) -> list[dict[str, Any]]:
        sentiment = getattr(query, "sentiment", None)
        sentiments = [sentiment] if sentiment else _sentiments_for_intent(query.intent)
        aspects = query.target_aspects or None
        # Đã lọc cứng theo chủ đề/cảm xúc thì mọi dòng đều đúng điều kiện; ngưỡng cosine
        # chỉ làm mất dòng đúng (câu hỏi và câu review khác dạng nên cosine thấp).
        min_score = 0.0 if (aspects or sentiment) else None
        if query.intent == "PRODUCT_FIT" and aspects and len(aspects) > 1:
            per_aspect = max(2, limit // len(aspects))
            groups = [
                self.vector_store.search(
                    product_id=product_id,
                    query=question,
                    limit=per_aspect,
                    sentiments=sentiments,
                    aspects=[aspect],
                    min_score=min_score,
                )
                for aspect in aspects
            ]
            rows = _merge_unique(groups, limit=limit)
            if len(rows) >= limit:
                return rows
            broad = self.vector_store.search(
                product_id=product_id,
                query=question,
                limit=limit,
                sentiments=sentiments,
                aspects=aspects,
                min_score=min_score,
            )
            return _merge_unique([rows, broad], limit=limit)
        return self.vector_store.search(
            product_id=product_id,
            query=question,
            limit=limit,
            sentiments=sentiments,
            aspects=aspects,
            min_score=min_score,
        )


def _is_topic_lookup(query: QueryAnalysis) -> bool:
    return bool(
        query.target_aspects
        and query.intent in SEMANTIC_INTENTS
        and not getattr(query, "sentiment", None)
        and not getattr(query, "date_from", None)
    )


def _keep_relevant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows đã sắp theo reranker_score giảm dần. Giữ dòng đạt ngưỡng tuyệt đối, hoặc
    đạt tỷ lệ RERANKER_RELATIVE_CUTOFF so với dòng tốt nhất."""
    if not rows:
        return rows
    top = float(rows[0].get("reranker_score") or 0.0)
    cutoff = min(settings.reranker_min_score, top * settings.reranker_relative_cutoff)
    return [row for row in rows if float(row.get("reranker_score") or 0.0) >= cutoff]


def _is_informative(text: str) -> bool:
    text = " ".join(str(text or "").split())
    return len(text) >= 28 and len(text.split()) >= 5


def _informative_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Giữ nguyên thứ tự xếp hạng trong từng nhóm; câu quá ngắn chỉ dùng khi thiếu."""
    return [row for row in rows if _is_informative(row.get("text"))] + [
        row for row in rows if not _is_informative(row.get("text"))
    ]


def _dedupe_by_review(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Một câu review được lưu một dòng cho mỗi chủ đề; giữ dòng đầu, theo thứ tự xếp hạng."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("review_id") or "") or _text_key(str(row.get("text") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _sentiments_for_intent(intent: str) -> list[str] | None:
    if intent in {"COMPLAINT_SUMMARY", "RISK_CHECK"}:
        return ["negative"]
    if intent in {"SUMMARY", "RECOMMENDATION", "PRODUCT_FIT", "PRODUCT_QA", "ASPECT_QA"}:
        return ["negative", "positive", "neutral"]
    return None


def _build_rerank_query(question: str, query: QueryAnalysis) -> str:
    parts = [question.strip()]
    if query.target_aspects:
        parts.append("Khía cạnh: " + ", ".join(query.target_aspects))
    # QueryAnalysis currently stores sentiment as a constraint rather than a
    # dedicated field. Keep this compatible with both representations.
    sentiment = getattr(query, "sentiment", None)
    if sentiment:
        parts.append("Cảm xúc: " + str(sentiment).lower())
    return " | ".join(part for part in parts if part)


def _merge_unique(groups: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            key = str(row.get("review_id") or "") + "|" + " ".join(str(row.get("text") or "").lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= limit:
                return out
    return out


def _select_diverse(rows: list[dict[str, Any]], query: QueryAnalysis, limit: int) -> list[dict[str, Any]]:
    del query
    selected: list[dict[str, Any]] = []
    selected_tokens: list[set[str]] = []
    for row in rows:
        tokens = set(re.findall(r"\w+", str(row.get("text") or "").lower()))
        if any(_jaccard(tokens, previous) >= 0.85 for previous in selected_tokens):
            continue
        selected.append(row)
        selected_tokens.append(tokens)
        if len(selected) >= limit:
            break
    return selected or rows[:limit]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _hybrid_rank(
    rows: list[dict[str, Any]],
    query: QueryAnalysis,
    qdrant_weight: float,
    structured_weight: float,
) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        key = str(row.get("review_id") or "") or _text_key(text)
        current = deduped.get(key)
        scored = {**row, "hybrid_score": _score_row(row, query, qdrant_weight, structured_weight)}
        # Cùng một review: giữ câu có nội dung cụ thể trước, rồi mới xét điểm.
        rank = (_is_informative(text), scored["hybrid_score"])
        if current is None or rank > (_is_informative(current.get("text")), current.get("hybrid_score", 0)):
            deduped[key] = scored
    out = list(deduped.values())
    out.sort(key=lambda row: row.get("hybrid_score", 0), reverse=True)
    return out


def _score_row(row: dict[str, Any], query: QueryAnalysis, qdrant_weight: float, structured_weight: float) -> float:
    semantic_score = float(row.get("semantic_score") or 0.0)
    reranker_score = row.get("reranker_score")
    confidence = float(row.get("confidence") or 0.0)
    source = str(row.get("source") or "postgres")
    aspect = str(row.get("aspect_code") or row.get("aspect") or "")
    sentiment = str(row.get("sentiment") or "neutral")

    aspect_bonus = 0.0
    if query.target_aspects and aspect in query.target_aspects:
        aspect_bonus = 0.2

    sentiment_bonus = 0.0
    if query.intent in {"COMPLAINT_SUMMARY", "RISK_CHECK"} and sentiment == "negative":
        sentiment_bonus = 0.25
    elif query.intent == "ASPECT_QA" and sentiment in {"negative", "positive"}:
        sentiment_bonus = 0.15
    elif query.intent in {"SUMMARY", "RECOMMENDATION"} and sentiment in {"positive", "negative"}:
        sentiment_bonus = 0.1

    source_weight = qdrant_weight if source == "pgvector" else structured_weight
    relevance_score = float(reranker_score) if reranker_score is not None else semantic_score
    return round(source_weight + relevance_score * 0.35 + confidence * 0.25 + aspect_bonus + sentiment_bonus, 4)


def _cache_key(product_id: str, question: str, query: QueryAnalysis) -> str:
    raw = json.dumps({
        "product_id": str(product_id),
        "question": str(question).strip().lower(),
        "intent": query.intent,
        "aspects": query.target_aspects,
        "sentiment": getattr(query, "sentiment", None),
        "date_from": getattr(query, "date_from", None),
        "date_to": getattr(query, "date_to", None),
    }, ensure_ascii=False, sort_keys=True)
    return "retrieval:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _text_key(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
