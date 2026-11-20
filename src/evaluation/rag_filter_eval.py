"""Bộ đo cho trợ lý review: câu hỏi có điều kiện (chủ đề, cảm xúc, thời gian).

Mỗi câu trong bộ chuẩn chỉ ghi *cách hiểu đúng* do người gán (gold filters).
Đáp án (số ý kiến, các ý kiến hợp lệ) được tính lại từ database mỗi lần chạy,
theo đúng quy tắc đếm của dashboard: loại trùng theo (review, chủ đề, cảm xúc).

Chạy (từ thư mục gốc dự án):
    python -m src.evaluation.rag_filter_eval             # không gọi LLM, kết quả lặp lại được
    python -m src.evaluation.rag_filter_eval --llm on    # dùng LLM đã cấu hình trong .env
    python -m src.evaluation.rag_filter_eval --ids B01,B18

Biến môi trường RAG_REFERENCE_DATE được đặt theo từng câu hỏi để các cách nói
tương đối ("năm ngoái", "tháng trước") có một mốc cố định khi đo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = ROOT / "data" / "eval" / "rag_filter_questions.jsonl"
DEFAULT_OUT_DIR = ROOT / "results" / "rag_eval"
DEFAULT_REFERENCE_DATE = "2026-11-17"

SENTIMENTS = {"positive", "neutral", "negative"}
ANY = "*"  # giá trị gold "không chấm" cho cảm xúc
NO_DATA_PHRASES = ("không có", "chưa có", "không tìm thấy", "không ghi nhận")
OVERLAP_MIN_IOU = 0.5


# ── Bộ câu hỏi chuẩn ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoldFilters:
    aspects: tuple[str, ...] | None  # None: không chấm chủ đề; (): câu hỏi không giới hạn chủ đề
    sentiment: str | None            # None: không giới hạn cảm xúc; "*": không chấm
    date_from: date | None
    date_to: date | None
    date_match: str = "exact"        # "exact" cho mốc tường minh, "overlap" cho cách nói mơ hồ


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    product_id: str
    category: str
    question: str
    gold: GoldFilters
    reference_date: date


def _parse_day(value: Any) -> date | None:
    if value in (None, ""):
        return None
    return date.fromisoformat(str(value)[:10])


def parse_question(raw: dict[str, Any]) -> EvalQuestion:
    gold_raw = raw["gold"]
    aspects = gold_raw.get("aspects")
    sentiment = gold_raw.get("sentiment")
    if sentiment is not None and sentiment != ANY and sentiment not in SENTIMENTS:
        raise ValueError(f"{raw['id']}: cảm xúc không hợp lệ {sentiment!r}")
    date_from, date_to = _parse_day(gold_raw.get("date_from")), _parse_day(gold_raw.get("date_to"))
    if (date_from is None) != (date_to is None):
        raise ValueError(f"{raw['id']}: phải có cả date_from và date_to, hoặc không có cả hai")
    if date_from and date_from > date_to:
        raise ValueError(f"{raw['id']}: date_from sau date_to")
    date_match = gold_raw.get("date_match", "exact")
    if date_match not in {"exact", "overlap"}:
        raise ValueError(f"{raw['id']}: date_match không hợp lệ {date_match!r}")
    return EvalQuestion(
        id=str(raw["id"]),
        product_id=str(raw["product_id"]),
        category=str(raw.get("category") or "khac"),
        question=str(raw["question"]),
        gold=GoldFilters(
            aspects=None if aspects is None else tuple(aspects),
            sentiment=sentiment,
            date_from=date_from,
            date_to=date_to,
            date_match=date_match,
        ),
        reference_date=_parse_day(raw.get("reference_date") or DEFAULT_REFERENCE_DATE),
    )


def load_questions(path: Path) -> list[EvalQuestion]:
    questions = [
        parse_question(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Trùng id câu hỏi: {sorted(duplicates)}")
    return questions


# ── Đáp án tính từ dữ liệu ────────────────────────────────────────────────────

def opinion_matches(aspect: str, sentiment: str, day: date | None, gold: GoldFilters) -> list[str]:
    """Trả về danh sách lý do không khớp; rỗng nghĩa là ý kiến thỏa mọi điều kiện gold."""
    reasons = []
    if gold.aspects and aspect not in gold.aspects:
        reasons.append("sai chủ đề")
    if gold.sentiment not in (None, ANY) and sentiment != gold.sentiment:
        reasons.append("sai cảm xúc")
    if gold.date_from is not None:
        if day is None:
            reasons.append("không có ngày")
        elif not (gold.date_from <= day <= gold.date_to):
            reasons.append("ngoài khoảng ngày")
    return reasons


def count_matching(opinions: Iterable[dict[str, Any]], gold: GoldFilters) -> int:
    return sum(
        1 for o in opinions
        if not opinion_matches(o["aspect"], o["sentiment"], o.get("day"), gold)
    )


# ── Chấm điểm một câu trả lời ─────────────────────────────────────────────────

def predicted_filters(result: dict[str, Any]) -> dict[str, Any]:
    """Đọc cách hiểu của chatbot từ kết quả answer_chat.

    Đọc các trường sẽ có sau khi cải tiến (sentiment, date_from, date_to trong
    query_analysis); với phiên bản hiện tại, suy cảm xúc từ kế hoạch thống kê
    hoặc intent, đúng như cách pipeline đang dùng cảm xúc khi truy xuất.
    """
    qa = result.get("query_analysis") or {}
    plan = result.get("statistics_plan") or {}
    sentiment = qa.get("sentiment") or plan.get("sentiment")
    if not sentiment and qa.get("intent") in {"COMPLAINT_SUMMARY", "RISK_CHECK"}:
        sentiment = "negative"
    return {
        "aspects": sorted(set(qa.get("target_aspects") or [])),
        "sentiment": sentiment or None,
        "date_from": _safe_day(qa.get("date_from")),
        "date_to": _safe_day(qa.get("date_to")),
    }


def _safe_day(value: Any) -> date | None:
    try:
        return _parse_day(value)
    except ValueError:
        return None


def aspect_ok(predicted: list[str], gold: GoldFilters) -> bool | None:
    if gold.aspects is None:
        return None
    return set(predicted) == set(gold.aspects)


def sentiment_ok(predicted: str | None, gold: GoldFilters) -> bool | None:
    if gold.sentiment == ANY:
        return None
    return predicted == gold.sentiment


def date_ok(pred_from: date | None, pred_to: date | None, gold: GoldFilters) -> bool:
    if gold.date_from is None:
        return pred_from is None and pred_to is None
    if pred_from is None or pred_to is None:
        return False
    if gold.date_match == "exact":
        return pred_from == gold.date_from and pred_to == gold.date_to
    return day_range_iou(pred_from, pred_to, gold.date_from, gold.date_to) >= OVERLAP_MIN_IOU


def day_range_iou(a_from: date, a_to: date, b_from: date, b_to: date) -> float:
    inter = (min(a_to, b_to) - max(a_from, b_from)).days + 1
    if inter <= 0:
        return 0.0
    union = (max(a_to, b_to) - min(a_from, b_from)).days + 1
    return inter / union


def check_evidence(
    evidence: list[dict[str, Any]],
    gold: GoldFilters,
    day_of_review: dict[str, date | None],
) -> list[dict[str, Any]]:
    checked = []
    for row in evidence:
        aspect = str(row.get("aspect_code") or row.get("aspect") or "")
        sentiment = str(row.get("sentiment") or "")
        review_id = str(row.get("review_id") or "")
        day = day_of_review.get(review_id)
        reasons = opinion_matches(aspect, sentiment, day, gold)
        checked.append({
            "review_id": review_id,
            "aspect": aspect,
            "sentiment": sentiment,
            "date": day.isoformat() if day else None,
            "text": str(row.get("text") or "")[:120],
            "ok": not reasons,
            "reasons": reasons,
        })
    return checked


def stated_count(result: dict[str, Any], gold: GoldFilters) -> int | None:
    """Con số chatbot báo cho cặp (chủ đề, cảm xúc). Chỉ chấm khi câu hỏi nhắm
    đúng một chủ đề và một cảm xúc cụ thể, vì khi đó con số mới có nghĩa rõ."""
    if not gold.aspects or len(gold.aspects) != 1 or gold.sentiment in (None, ANY):
        return None
    rows = (result.get("statistics_result") or {}).get("rows") or []
    for row in rows:
        if row.get("aspect") == gold.aspects[0] and row.get("sentiment") in (gold.sentiment, "", None):
            return int(row.get("total") or 0)
    match = re.search(r"\bCó (\d+)\b", result.get("answer") or "")
    return int(match.group(1)) if match else None


def mentions_no_data(answer: str) -> bool:
    text = (answer or "").lower()
    return any(phrase in text for phrase in NO_DATA_PHRASES)


def score_result(
    q: EvalQuestion,
    result: dict[str, Any],
    opinions: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    day_of_review: dict[str, date | None],
) -> dict[str, Any]:
    gold = q.gold
    pred = predicted_filters(result)
    evidence = check_evidence(result.get("evidence") or [], gold, day_of_review)
    gold_opinions = count_matching(opinions, gold)
    gold_raw = count_matching(raw_rows, gold)
    count = stated_count(result, gold)
    n_ok = sum(1 for e in evidence if e["ok"])
    no_data_ok = None
    if gold_opinions == 0:
        no_data_ok = not evidence and mentions_no_data(result.get("answer") or "")
    return {
        "id": q.id,
        "category": q.category,
        "product_id": q.product_id,
        "question": q.question,
        "gold": _gold_to_json(gold),
        "predicted": {**pred, "date_from": _iso(pred["date_from"]), "date_to": _iso(pred["date_to"])},
        "gold_opinions": gold_opinions,
        "gold_raw_rows": gold_raw,
        "aspect_ok": aspect_ok(pred["aspects"], gold),
        "sentiment_ok": sentiment_ok(pred["sentiment"], gold),
        "date_ok": date_ok(pred["date_from"], pred["date_to"], gold),
        "evidence": evidence,
        "evidence_ok": n_ok,
        "evidence_total": len(evidence),
        "missed_all": gold_opinions > 0 and n_ok == 0,
        "stated_count": count,
        "count_ok": None if count is None else count == gold_opinions,
        "count_is_raw_rows": None if count is None else (count == gold_raw and gold_raw != gold_opinions),
        "no_data_ok": no_data_ok,
        "answer": (result.get("answer") or "")[:400],
        "llm_used": bool(result.get("llm_used")),
    }


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _gold_to_json(gold: GoldFilters) -> dict[str, Any]:
    data = asdict(gold)
    data["aspects"] = None if gold.aspects is None else list(gold.aspects)
    data["date_from"], data["date_to"] = _iso(gold.date_from), _iso(gold.date_to)
    return data


# ── Tổng hợp ──────────────────────────────────────────────────────────────────

def _rate(values: Iterable[bool | None]) -> dict[str, Any]:
    scored = [v for v in values if v is not None]
    return {"ok": sum(scored), "n": len(scored), "rate": round(sum(scored) / len(scored), 3) if scored else None}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_evidence = [r for r in rows if r["evidence_total"]]
    ev_ok = sum(r["evidence_ok"] for r in rows)
    ev_total = sum(r["evidence_total"] for r in rows)
    time_rows = [r for r in rows if r["gold"]["date_from"] is not None]
    summary = {
        "questions": len(rows),
        "hieu_chu_de": _rate(r["aspect_ok"] for r in rows),
        "hieu_cam_xuc": _rate(r["sentiment_ok"] for r in rows),
        "hieu_thoi_gian": _rate(r["date_ok"] for r in time_rows),
        "khong_bia_thoi_gian": _rate(r["date_ok"] for r in rows if r["gold"]["date_from"] is None),
        "bang_chung_dung_dieu_kien": {
            "ok": ev_ok, "n": ev_total, "rate": round(ev_ok / ev_total, 3) if ev_total else None,
        },
        "cau_co_bang_chung_sai": sum(1 for r in with_evidence if r["evidence_ok"] < r["evidence_total"]),
        "bo_sot_hoan_toan": _rate(not r["missed_all"] for r in rows if r["gold_opinions"] > 0),
        "so_lieu_dung": _rate(r["count_ok"] for r in rows),
        "so_lieu_dem_dong_tho": sum(1 for r in rows if r["count_is_raw_rows"]),
        "khong_du_lieu_tra_loi_dung": _rate(r["no_data_ok"] for r in rows),
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    summary["theo_nhom"] = {
        cat: {
            "n": len(items),
            "hieu_thoi_gian": _rate(r["date_ok"] for r in items),
            "bang_chung_dung": _rate(e["ok"] for r in items for e in r["evidence"]),
        }
        for cat, items in sorted(by_category.items())
    }
    return summary


# ── Chạy với chatbot thật ─────────────────────────────────────────────────────

def _prepare_env(llm_on: bool, keep_db_host: bool) -> None:
    """Phải chạy trước khi import app.assistant.*, vì cấu hình được đọc lúc import."""
    from dotenv import dotenv_values

    file_env = dotenv_values(ROOT / ".env")
    db_url = os.environ.get("DATABASE_URL") or file_env.get("DATABASE_URL") or ""
    if db_url and not keep_db_host:
        # .env trỏ tới tên máy của Docker; khi chạy trên máy thì dùng localhost.
        db_url = re.sub(r"@postgres([:/])", r"@localhost\1", db_url)
    os.environ["DATABASE_URL"] = db_url
    os.environ["REDIS_URL"] = ""  # tắt bộ nhớ đệm để mỗi lần đo đều chạy lại toàn bộ pipeline
    if not llm_on:
        os.environ["GEMINI_API_KEY"] = ""
        os.environ["GROQ_API_KEY"] = ""


def _load_truth(product_ids: set[str]) -> dict[str, dict[str, Any]]:
    from sqlalchemy import text

    from app.assistant.db import get_engine
    from app.web.seller_summary import dedupe_opinions, parse_review_date

    engine = get_engine()
    if engine is None:
        raise RuntimeError("Không kết nối được database (DATABASE_URL trống hoặc sai).")
    truth = {}
    with engine.connect() as conn:
        for pid in sorted(product_ids):
            rows = [dict(r._mapping) for r in conn.execute(text("""
                SELECT r.tiki_review_id AS review_id, a.category AS aspect, a.polarity AS sentiment,
                       a.confidence, a.sentence, r.created_at_tiki
                FROM aspect_sentiments a
                JOIN reviews r ON r.id = a.review_id
                JOIN products p ON p.id = a.product_id
                WHERE p.tiki_product_id = :pid
            """), {"pid": pid})]
            day_of_review: dict[str, date | None] = {}
            for row in rows:
                parsed = parse_review_date(row["created_at_tiki"])
                row["day"] = parsed.date() if parsed else None
                day_of_review[str(row["review_id"])] = row["day"]
            truth[pid] = {"raw_rows": rows, "opinions": dedupe_opinions(rows), "day_of_review": day_of_review}
    return truth


def run(args: argparse.Namespace) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    llm_on = args.llm == "on"
    _prepare_env(llm_on, args.keep_db_host)

    from app.assistant.assistant_service import ParentAssistantService

    questions = load_questions(Path(args.questions))
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q.id in wanted]
    truth = _load_truth({q.product_id for q in questions})
    missing = [pid for pid, t in truth.items() if not t["raw_rows"]]
    if missing:
        print(f"Cảnh báo: chưa có dữ liệu phân tích cho sản phẩm {missing}. Hãy phân tích sản phẩm trên web trước.")

    service = ParentAssistantService()
    rows = []
    for q in questions:
        os.environ["RAG_REFERENCE_DATE"] = q.reference_date.isoformat()
        try:
            result = service.answer_chat(q.product_id, q.question, use_llm=llm_on) or {}
        except Exception as exc:  # một câu lỗi không được làm dừng cả bộ đo
            result = {"answer": f"[LỖI] {exc}", "evidence": []}
        t = truth[q.product_id]
        rows.append(score_result(q, result, t["opinions"], t["raw_rows"], t["day_of_review"]))

    summary = summarize(rows)
    _print_report(rows, summary, llm_on)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"rag_eval_{stamp}_llm-{args.llm}.json"
    out_path.write_text(
        json.dumps({"llm": args.llm, "summary": summary, "questions": rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nĐã lưu chi tiết: {out_path}")
    return 0


def _mark(value: bool | None) -> str:
    return "–" if value is None else ("✓" if value else "✗")


def _print_report(rows: list[dict[str, Any]], summary: dict[str, Any], llm_on: bool) -> None:
    print(f"\nBỘ ĐO RAG THEO ĐIỀU KIỆN  (LLM: {'bật' if llm_on else 'tắt'})")
    header = f"{'ID':<4} {'Nhóm':<20} {'Đáp án':>6} {'Chủ đề':^6} {'C.xúc':^6} {'T.gian':^6} {'BC đúng':>8} {'Số báo':>7} {'Không DL':^8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ev = f"{r['evidence_ok']}/{r['evidence_total']}"
        count = "–" if r["stated_count"] is None else f"{r['stated_count']}{'' if r['count_ok'] else '✗'}"
        print(
            f"{r['id']:<4} {r['category']:<20} {r['gold_opinions']:>6} {_mark(r['aspect_ok']):^6} "
            f"{_mark(r['sentiment_ok']):^6} {_mark(r['date_ok']):^6} {ev:>8} {count:>7} {_mark(r['no_data_ok']):^8}"
        )

    def line(label: str, item: dict[str, Any]) -> None:
        rate = "–" if item["rate"] is None else f"{item['rate'] * 100:.0f}%"
        print(f"  {label:<44} {item['ok']:>4}/{item['n']:<4} {rate:>5}")

    s = summary
    print("\nTỔNG HỢP")
    line("Hiểu đúng chủ đề", s["hieu_chu_de"])
    line("Hiểu đúng cảm xúc", s["hieu_cam_xuc"])
    line("Hiểu đúng khoảng thời gian", s["hieu_thoi_gian"])
    line("Không tự thêm thời gian khi câu hỏi không có", s["khong_bia_thoi_gian"])
    line("Bằng chứng thỏa mọi điều kiện (theo dòng)", s["bang_chung_dung_dieu_kien"])
    line("Có ít nhất 1 bằng chứng đúng khi có dữ liệu", s["bo_sot_hoan_toan"])
    line("Số liệu khớp dashboard", s["so_lieu_dung"])
    line("Trả lời đúng khi không có dữ liệu", s["khong_du_lieu_tra_loi_dung"])
    print(f"  {'Câu có ít nhất 1 bằng chứng sai':<44} {s['cau_co_bang_chung_sai']:>4}")
    print(f"  {'Số liệu đếm dòng thô thay vì ý kiến':<44} {s['so_lieu_dem_dong_tho']:>4}")
    print("\nTHEO NHÓM CÂU HỎI (hiểu thời gian | bằng chứng đúng)")
    for cat, item in s["theo_nhom"].items():
        t, e = item["hieu_thoi_gian"], item["bang_chung_dung"]
        tr = "–" if t["rate"] is None else f"{t['rate'] * 100:.0f}%"
        er = "–" if e["rate"] is None else f"{e['rate'] * 100:.0f}%"
        print(f"  {cat:<22} n={item['n']:<3} {tr:>5} | {er:>5} ({e['ok']}/{e['n']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo trợ lý review với câu hỏi có điều kiện.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--llm", choices=["off", "on"], default="off")
    parser.add_argument("--ids", default="", help="Chỉ chạy các câu này, ví dụ B01,B18")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--keep-db-host", action="store_true", help="Không đổi host 'postgres' sang 'localhost'")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
