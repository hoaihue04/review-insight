"""Số liệu cho dashboard chủ shop: chỉ đếm kết quả PhoBERT, không tính điểm.

Đơn vị đếm là "ý kiến" = một bộ (review, chủ đề, cảm xúc). Một câu có thể cho
nhiều ý kiến; cùng một bộ lặp lại ở nhiều câu của một review chỉ tính một lần.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from app.web.constants import ASPECT_DISPLAY_NAMES

SENTIMENTS = ("positive", "neutral", "negative")
LOW_DATA_THRESHOLD = 5

# Ngày của review tính theo giờ Việt Nam, không phải UTC. Giao diện tự gom theo
# năm / tháng / ngày / thứ từ trường `date` của từng ý kiến, nên ngày phải đúng ở đây.
VN_TZ = timezone(timedelta(hours=7))

def parse_review_date(value: Any) -> Optional[datetime]:
    """Tiki trả `created_at` dạng unix timestamp (giây); chấp nhận cả ms và ISO."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    try:
        number = float(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(VN_TZ) if parsed.tzinfo else parsed
    if number != number or number <= 0:  # NaN hoặc không hợp lệ
        return None
    if number > 1e12:
        number /= 1000.0
    try:
        return datetime.fromtimestamp(number, tz=VN_TZ)
    except (OverflowError, OSError, ValueError):
        return None


def _display_name(aspect: str) -> str:
    return ASPECT_DISPLAY_NAMES.get(aspect) or aspect.split("#")[-1].capitalize()


def dedupe_opinions(tuples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Giữ một dòng cho mỗi (review, chủ đề, cảm xúc), chọn câu PhoBERT tự tin nhất."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for item in tuples:
        sentiment = str(item.get("sentiment") or "")
        if sentiment not in SENTIMENTS:
            continue
        aspect = str(item.get("aspect") or "")
        if not aspect:
            continue
        key = (str(item.get("review_id") or ""), aspect, sentiment)
        current = best.get(key)
        if current is None or float(item.get("confidence") or 0) > float(current.get("confidence") or 0):
            best[key] = item
    return list(best.values())


def _empty_counts() -> Dict[str, int]:
    return {s: 0 for s in SENTIMENTS}


def build_seller_summary(
    tuples: Iterable[Dict[str, Any]],
    reviews: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    review_meta: Dict[str, Dict[str, Any]] = {}
    for rv in reviews:
        rid = str(rv.get("review_id") or "")
        date = parse_review_date(rv.get("created_at_ts"))
        rating = rv.get("rating")
        try:
            rating = int(float(rating)) if rating is not None else None
        except (TypeError, ValueError):
            rating = None
        review_meta[rid] = {"date": date, "rating": rating}

    totals = _empty_counts()
    reviews_with: Dict[str, set] = {s: set() for s in SENTIMENTS}
    by_topic: Dict[str, Dict[str, int]] = defaultdict(_empty_counts)
    rows: List[Dict[str, Any]] = []

    for item in dedupe_opinions(tuples):
        rid = str(item.get("review_id") or "")
        aspect = str(item["aspect"])
        sentiment = str(item["sentiment"])
        meta = review_meta.get(rid, {})
        date: Optional[datetime] = meta.get("date")

        totals[sentiment] += 1
        reviews_with[sentiment].add(rid)
        by_topic[aspect][sentiment] += 1

        rows.append({
            "review_id": rid,
            "aspect": aspect,
            "display_name": _display_name(aspect),
            "sentiment": sentiment,
            "sentence": str(item.get("sentence") or "").strip(),
            "rating": meta.get("rating"),
            "date": date.strftime("%Y-%m-%d") if date else "",
        })

    total_opinions = sum(totals.values())
    sentiments = {
        s: {
            "opinions": totals[s],
            "ratio": round(totals[s] / total_opinions, 4) if total_opinions else 0.0,
            "reviews": len(reviews_with[s]),
        }
        for s in SENTIMENTS
    }

    topics = [
        {
            "aspect": aspect,
            "display_name": _display_name(aspect),
            **counts,
            "total": sum(counts.values()),
            "low_data": sum(counts.values()) < LOW_DATA_THRESHOLD,
        }
        for aspect, counts in by_topic.items()
    ]
    topics.sort(key=lambda t: (-t["total"], t["aspect"]))

    rows.sort(key=lambda r: (r["date"], r["review_id"], r["aspect"], r["sentiment"]), reverse=True)

    return {
        "total_reviews": len(review_meta),
        "total_opinions": total_opinions,
        "sentiments": sentiments,
        "topics": topics,
        "opinions": rows,
    }
