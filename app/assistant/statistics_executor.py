from __future__ import annotations

from typing import Any

from sqlalchemy import desc, func

from app.assistant.db import session_scope
from app.assistant.models import AspectSentiment, Product, Review, RiskFlag
from app.assistant.statistics_planner import StatisticsQueryPlan
from app.web.seller_summary import parse_review_date

# Đếm theo review, cùng quy tắc với dashboard: một review nói một ý nhiều lần
# vẫn chỉ tính một lần cho mỗi (chủ đề, cảm xúc).
REVIEW_COUNT = func.count(func.distinct(AspectSentiment.review_id))


class StatisticsQueryExecutor:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def execute(self, tiki_product_id: str, plan: StatisticsQueryPlan | None) -> dict[str, Any]:
        if plan is None or not self.repository.enabled:
            return {}

        if plan.metric == "top_aspects_by_sentiment":
            rows = self._top_aspects_by_sentiment(tiki_product_id, plan)
        elif plan.metric == "aspect_sentiment_distribution":
            rows = self._aspect_sentiment_distribution(tiki_product_id, plan)
        elif plan.metric in {"negative_ratio_by_aspect", "positive_ratio_by_aspect"}:
            rows = self._ratio_by_aspect(tiki_product_id, plan)
        elif plan.metric == "review_statistics":
            data = (
                self._review_statistics_in_period(tiki_product_id, plan)
                if plan.date_from
                else self.repository.get_review_statistics(tiki_product_id)
            )
            return {"plan": plan.to_prompt_context(), "metric": plan.metric, "data": data}
        elif plan.metric == "risk_summary":
            rows = self._risk_summary(tiki_product_id, plan)
        elif plan.metric == "evidence_by_aspect":
            rows = self._evidence_by_aspect(tiki_product_id, plan)
        elif plan.metric == "rating_sentiment_summary":
            rows = self._rating_sentiment_summary(tiki_product_id, plan)
        else:
            rows = []

        return {
            "plan": plan.to_prompt_context(),
            "metric": plan.metric,
            "rows": rows,
        }

    def _get_product_db_id(self, session: Any, tiki_product_id: str) -> int | None:
        product_id = (
            session.query(Product.id)
            .filter(Product.tiki_product_id == str(tiki_product_id))
            .scalar()
        )
        return int(product_id) if product_id is not None else None

    def _ratio_by_aspect(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        key = "negative" if plan.metric == "negative_ratio_by_aspect" else "positive"
        if not plan.date_from:
            fetch = (
                self.repository.get_negative_ratio_by_aspect
                if key == "negative"
                else self.repository.get_positive_ratio_by_aspect
            )
            rows = _filter_rows_by_aspect(fetch(tiki_product_id, limit=1000), plan.target_aspects)
            return rows[:plan.limit]

        # Có khoảng ngày: tính tỷ lệ từ phân bố đã lọc theo ngày (bỏ lọc cảm xúc để có mẫu số đúng).
        distribution = self._aspect_sentiment_distribution(
            tiki_product_id,
            StatisticsQueryPlan(
                metric="aspect_sentiment_distribution",
                target_aspects=plan.target_aspects,
                limit=1000,
                date_from=plan.date_from,
                date_to=plan.date_to,
            ),
        )
        grouped: dict[str, dict[str, Any]] = {}
        for row in distribution:
            bucket = grouped.setdefault(row["aspect"], {"aspect": row["aspect"], "positive": 0, "neutral": 0, "negative": 0, "total": 0})
            sentiment = row["sentiment"] if row["sentiment"] in ("positive", "neutral", "negative") else "neutral"
            bucket[sentiment] += row["total"]
            bucket["total"] += row["total"]
        out = []
        for row in grouped.values():
            row["positive_ratio"] = round(row["positive"] / row["total"], 4) if row["total"] else 0.0
            row["negative_ratio"] = round(row["negative"] / row["total"], 4) if row["total"] else 0.0
            out.append(row)
        out.sort(key=lambda r: (r[f"{key}_ratio"], r[key], r["total"]), reverse=True)
        return out[:plan.limit]

    def _top_aspects_by_sentiment(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    AspectSentiment.category.label("aspect"),
                    REVIEW_COUNT.label("total"),
                    func.avg(AspectSentiment.confidence).label("avg_confidence"),
                )
                .filter(AspectSentiment.product_id == product_db_id)
            )
            query = _apply_aspect_sentiment_filters(query, plan, _review_ids_in_period(session, product_db_id, plan))

            rows = (
                query.group_by(AspectSentiment.category)
                .order_by(desc("total"))
                .limit(plan.limit)
                .all()
            )

        return [
            {
                "aspect": row.aspect,
                "sentiment": plan.sentiment or "",
                "total": int(row.total or 0),
                "avg_confidence": round(float(row.avg_confidence or 0.0), 4),
            }
            for row in rows
        ]

    def _aspect_sentiment_distribution(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    AspectSentiment.category.label("aspect"),
                    AspectSentiment.polarity.label("sentiment"),
                    REVIEW_COUNT.label("total"),
                    func.avg(AspectSentiment.confidence).label("avg_confidence"),
                )
                .filter(AspectSentiment.product_id == product_db_id)
            )
            query = _apply_aspect_sentiment_filters(query, plan, _review_ids_in_period(session, product_db_id, plan))

            rows = (
                query.group_by(AspectSentiment.category, AspectSentiment.polarity)
                .order_by(desc("total"))
                .limit(plan.limit * 3)
                .all()
            )

        return [
            {
                "aspect": row.aspect,
                "sentiment": row.sentiment,
                "total": int(row.total or 0),
                "avg_confidence": round(float(row.avg_confidence or 0.0), 4),
            }
            for row in rows
        ]

    def _review_statistics_in_period(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> dict[str, Any]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return {}
            review_ids = _review_ids_in_period(session, product_db_id, plan) or []
            if not review_ids:
                return {"stored_review_count": 0, "aspect_record_count": 0, "avg_review_rating": None}
            avg_rating = (
                session.query(func.avg(Review.rating))
                .filter(Review.id.in_(review_ids), Review.rating.isnot(None))
                .scalar()
            )
            opinion_count = (
                session.query(func.count())
                .select_from(
                    session.query(AspectSentiment.review_id, AspectSentiment.category, AspectSentiment.polarity)
                    .filter(AspectSentiment.review_id.in_(review_ids))
                    .distinct()
                    .subquery()
                )
                .scalar()
            )
        return {
            "stored_review_count": len(review_ids),
            "aspect_record_count": int(opinion_count or 0),
            "avg_review_rating": round(float(avg_rating), 4) if avg_rating is not None else None,
        }

    def _risk_summary(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    RiskFlag.risk_type,
                    RiskFlag.severity,
                    func.count(RiskFlag.id).label("total"),
                    func.avg(RiskFlag.confidence).label("avg_confidence"),
                )
                .filter(RiskFlag.product_id == product_db_id)
            )
            review_ids = _review_ids_in_period(session, product_db_id, plan)
            if review_ids is not None:
                query = query.filter(RiskFlag.review_id.in_(review_ids))
            rows = (
                query.group_by(RiskFlag.risk_type, RiskFlag.severity)
                .order_by(desc("total"))
                .limit(plan.limit)
                .all()
            )

        return [
            {
                "risk_type": row.risk_type,
                "severity": row.severity,
                "total": int(row.total or 0),
                "avg_confidence": round(float(row.avg_confidence or 0.0), 4),
            }
            for row in rows
        ]

    def _evidence_by_aspect(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    AspectSentiment.category.label("aspect"),
                    AspectSentiment.polarity.label("sentiment"),
                    AspectSentiment.confidence,
                    AspectSentiment.sentence,
                    Review.tiki_review_id,
                    Review.rating,
                )
                .outerjoin(Review, Review.id == AspectSentiment.review_id)
                .filter(AspectSentiment.product_id == product_db_id)
            )
            query = _apply_aspect_sentiment_filters(query, plan, _review_ids_in_period(session, product_db_id, plan))

            rows = (
                query.order_by(AspectSentiment.confidence.desc())
                .limit(plan.limit)
                .all()
            )

        return [
            {
                "aspect": row.aspect,
                "sentiment": row.sentiment,
                "confidence": round(float(row.confidence or 0.0), 4),
                "text": row.sentence,
                "review_id": row.tiki_review_id or "",
                "rating": row.rating,
            }
            for row in rows
        ]

    def _rating_sentiment_summary(self, tiki_product_id: str, plan: StatisticsQueryPlan) -> list[dict[str, Any]]:
        with session_scope() as session:
            product_db_id = self._get_product_db_id(session, tiki_product_id)
            if product_db_id is None:
                return []

            query = (
                session.query(
                    Review.rating,
                    AspectSentiment.polarity.label("sentiment"),
                    REVIEW_COUNT.label("total"),
                )
                .join(Review, Review.id == AspectSentiment.review_id)
                .filter(
                    AspectSentiment.product_id == product_db_id,
                    Review.rating.isnot(None),
                )
            )
            query = _apply_aspect_sentiment_filters(query, plan, _review_ids_in_period(session, product_db_id, plan))

            rows = (
                query.group_by(Review.rating, AspectSentiment.polarity)
                .order_by(Review.rating.desc(), desc("total"))
                .limit(plan.limit * 3)
                .all()
            )

        return [
            {
                "rating": int(row.rating or 0),
                "sentiment": row.sentiment,
                "total": int(row.total or 0),
            }
            for row in rows
        ]


def _review_ids_in_period(session: Any, product_db_id: int, plan: StatisticsQueryPlan) -> list[int] | None:
    """Id các review trong khoảng ngày của kế hoạch; None nghĩa là không lọc theo ngày.

    Ngày được đọc bằng đúng hàm của dashboard (giờ Việt Nam), nên hai bên luôn
    đếm trên cùng một tập review.
    """
    if not plan.date_from:
        return None
    rows = (
        session.query(Review.id, Review.created_at_tiki)
        .filter(Review.product_id == product_db_id)
        .all()
    )
    ids = []
    for review_id, created_at in rows:
        parsed = parse_review_date(created_at)
        if parsed and plan.date_from <= parsed.date().isoformat() <= str(plan.date_to):
            ids.append(int(review_id))
    return ids


def _apply_aspect_sentiment_filters(query: Any, plan: StatisticsQueryPlan, review_ids: list[int] | None = None) -> Any:
    if plan.sentiment:
        query = query.filter(AspectSentiment.polarity == plan.sentiment)
    if plan.target_aspects:
        query = query.filter(AspectSentiment.category.in_(plan.target_aspects))
    if review_ids is not None:
        query = query.filter(AspectSentiment.review_id.in_(review_ids))
    return query


def _filter_rows_by_aspect(rows: list[dict[str, Any]], target_aspects: list[str]) -> list[dict[str, Any]]:
    if not target_aspects:
        return rows
    allowed = set(target_aspects)
    return [row for row in rows if str(row.get("aspect") or "") in allowed]
