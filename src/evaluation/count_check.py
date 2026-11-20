"""Kiểm tra số liệu của chatbot (không cần LLM).

Ragas Faithfulness coi số liệu SQL (answer_facts) là sự thật, nên không phát hiện được khi số
liệu sai. Module này đếm lại số ý kiến từ đáp án chuẩn (bảng aspect_sentiments, lọc theo điều
kiện gold của câu hỏi) rồi so với số liệu chatbot dùng:

- count_accuracy: 1 nếu số ý kiến, số review và phân bố khen / trung lập / chê khớp đáp án.
  Đây là phép đo đầu-cuối: sai có thể do SQL đếm sai hoặc do chatbot hiểu sai điều kiện câu
  hỏi (chủ đề, thời gian); xem count_diff để biết lệch ở đâu.
- answer_number_accuracy: tỉ lệ con số trong câu trả lời (ngoài phần trích dẫn) có trong số
  liệu đáp án. Bắt cả lỗi số liệu lẫn lỗi LLM chép sai số.

Cách đếm ở đây viết riêng, không gọi build_answer_facts, để hai bên kiểm tra lẫn nhau. Chỉ
dùng chung dedupe_opinions (loại trùng theo review, chủ đề, cảm xúc) vì đó là định nghĩa
"ý kiến" của cả dự án.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from src.evaluation.rag_filter_eval import ANY, GoldFilters, opinion_matches

NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d{3})*")
QUOTED_PATTERN = re.compile(r"\"[^\"]*\"|“[^”]*”|«[^»]*»")
SENTIMENT_KEYS = {"positive": "khen", "neutral": "trung_lap", "negative": "che"}


def countable(gold: GoldFilters) -> bool:
    """Câu hỏi có đáp án đếm được chính xác.

    Bỏ câu không chấm chủ đề/cảm xúc (tra cứu) và câu thời gian mơ hồ ("overlap"): với câu mơ
    hồ, khoảng ngày chatbot chọn chỉ cần trùng phần lớn với đáp án nên số đếm có thể lệch hợp lệ.
    """
    return gold.aspects is not None and gold.sentiment != ANY and gold.date_match == "exact"


def _breakdown(opinions: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(SENTIMENT_KEYS.get(o["sentiment"], "trung_lap") for o in opinions)
    return {"khen": counts["khen"], "trung_lap": counts["trung_lap"], "che": counts["che"], "tong": len(opinions)}


def gold_counts(opinions: list[dict[str, Any]], gold: GoldFilters) -> dict[str, Any] | None:
    """Số liệu đáp án cho một câu hỏi. `opinions` là ý kiến đã loại trùng, có khóa
    review_id, aspect, sentiment, day (như _load_truth trả về)."""
    if not countable(gold):
        return None
    scope = replace(gold, sentiment=None)  # phạm vi: đúng chủ đề và thời gian, mọi cảm xúc
    in_scope = [o for o in opinions if not opinion_matches(o["aspect"], o["sentiment"], o.get("day"), scope)]
    matching = [o for o in in_scope if gold.sentiment is None or o["sentiment"] == gold.sentiment]

    by_aspect: dict[str, list] = defaultdict(list)
    by_year: dict[str, list] = defaultdict(list)
    for o in in_scope:
        by_aspect[o["aspect"]].append(o)
        if o.get("day"):
            by_year[str(o["day"].year)].append(o)
    return {
        "y_kien": len(matching),
        "review": len({str(o["review_id"]) for o in matching}),
        "theo_cam_xuc": _breakdown(in_scope),
        "theo_chu_de": {a: _breakdown(rows) for a, rows in sorted(by_aspect.items())},
        "theo_nam": {y: _breakdown(rows) for y, rows in sorted(by_year.items())},
    }


def _numbers(text: str) -> list[int]:
    out = []
    for token in NUMBER_PATTERN.findall(text or ""):
        digits = re.sub(r"[.,]", "", token)
        if digits.isdigit():
            out.append(int(digits))
    return out


def _all_ints(value: Any) -> set[int]:
    """Mọi số nguyên trong một cấu trúc lồng nhau (kể cả khóa năm như "2022")."""
    found: set[int] = set()
    if isinstance(value, bool):
        return found
    if isinstance(value, int):
        found.add(value)
    elif isinstance(value, str):
        found.update(_numbers(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            found |= _all_ints(k) | _all_ints(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            found |= _all_ints(v)
    return found


def count_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    """count_accuracy, answer_number_accuracy và count_diff cho một mẫu trong dataset."""
    empty = {"count_accuracy": None, "answer_number_accuracy": None, "count_diff": ""}
    gold = sample.get("gold_counts")
    if not gold:
        return empty
    facts = sample.get("answer_facts")

    if facts:
        by_sentiment = facts.get("so_y_kien_theo_cam_xuc") or {}
        pairs = [
            ("số ý kiến đúng điều kiện", facts.get("so_y_kien_dung_dieu_kien"), gold["y_kien"]),
            ("số review đúng điều kiện", facts.get("so_review_dung_dieu_kien"), gold["review"]),
            *[(f"số ý kiến {k}", by_sentiment.get(k), gold["theo_cam_xuc"][k])
              for k in ("khen", "trung_lap", "che", "tong")],
        ]
    else:
        # Chatbot trả lời "không có dữ liệu" (không tính số liệu): đúng khi đáp án cũng là 0.
        pairs = [("số ý kiến đúng điều kiện", 0, gold["y_kien"])]
    diffs = [f"{name}: chatbot {got} ≠ đáp án {want}" for name, got, want in pairs if got != want]

    allowed = _all_ints(gold) | set(_numbers(sample.get("user_input") or "")) | set(range(0, 13)) | {100}
    if facts:
        # Mốc ngày của phạm vi và khoảng ngày có dữ liệu (dễ kiểm chứng, không phải số đếm).
        allowed |= _all_ints(facts.get("pham_vi")) | _all_ints(facts.get("khoang_ngay_co_du_lieu"))
    bases = sorted(n for n in allowed if 0 < n < 100000)
    percents = {round(100 * a / b) for b in bases for a in bases if a <= b}
    numbers = _numbers(QUOTED_PATTERN.sub(" ", sample.get("response") or ""))
    number_accuracy = (
        sum(1 for n in numbers if n in allowed or n in percents) / len(numbers) if numbers else None
    )
    return {
        "count_accuracy": 0.0 if diffs else 1.0,
        "answer_number_accuracy": number_accuracy,
        "count_diff": "; ".join(diffs),
    }
