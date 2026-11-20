"""Viết câu trả lời của chatbot cho nhân viên và chủ shop.

Văn phong: báo cáo khách đã nói gì, bằng số liệu và trích dẫn ngắn; không đưa lời khuyên.

- build_answer_facts: số liệu trong đúng phạm vi câu hỏi, đếm theo ý kiến đã loại trùng
  (review, chủ đề, cảm xúc) như dashboard. Đây là nguồn số duy nhất cho cả LLM lẫn mẫu soạn sẵn.
- template_answer: câu trả lời soạn sẵn khi không có LLM hoặc LLM trả lời không đạt.
- build_llm_prompt / SYSTEM_PROMPT: prompt cho LLM.
- check_llm_answer: kiểm tra câu trả lời của LLM (không khuyên, không có số lạ).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from app.assistant.query_analyzer import QueryAnalysis
from app.assistant.time_parser import fold
from app.web.constants import ASPECT_DISPLAY_NAMES
from app.web.seller_summary import dedupe_opinions

SENTIMENT_WORD = {"positive": "khen", "neutral": "trung lập", "negative": "chê"}
OVERVIEW_INTENTS = {"SUMMARY", "RECOMMENDATION", "COMPLAINT_SUMMARY", "RISK_CHECK"}
MAX_QUOTE_WORDS = 15

# Chỉ bắt câu mang tính khuyên thật ("nên/hãy + hành động", "cần lưu ý", gọi người đọc là "bạn").
# Không chặn từng chữ riêng lẻ: "đóng gói không cẩn thận", "khách rất yên tâm", "vì giao chậm nên
# khách không hài lòng" là thuật lại lời khách, không phải lời khuyên.
_ADVICE_ACTIONS = (
    "mua|chọn|cân nhắc|cẩn thận|kiểm tra|lưu ý|chú ý|xem|tránh|cải thiện|đọc|liên hệ|đổi|thử|"
    "ưu tiên|yên tâm|theo dõi|tham khảo|đặt|dùng|sử dụng|khắc phục|bổ sung|tăng cường|giảm"
)
ADVICE_PATTERN = re.compile(
    rf"\b(?:(?:nên|hãy)\s+(?:\w+\s+)?(?:{_ADVICE_ACTIONS})"
    r"|cần\s+(?:cẩn thận|lưu ý|chú ý|kiểm tra|cân nhắc|cải thiện|theo dõi|khắc phục)"
    r"|khuyên|khuyến nghị|khuyến cáo|cân nhắc|đáng mua|có thể yên tâm|bạn)\b",
    re.IGNORECASE,
)
QUOTED_PATTERN = re.compile(r"\"[^\"]*\"|“[^”]*”|«[^»]*»")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d{3})*")


def aspect_name(code: str) -> str:
    return ASPECT_DISPLAY_NAMES.get(code, code)


# ── Số liệu ──────────────────────────────────────────────────────────────────

def _format_day(iso_day: str | None) -> str:
    if not iso_day:
        return ""
    year, month, day = str(iso_day)[:10].split("-")
    return f"{day}/{month}/{year}"


def _percent(part: int, total: int) -> int:
    return round(100 * part / total) if total else 0


def build_answer_facts(aspect_rows: list[dict[str, Any]], query: QueryAnalysis) -> dict[str, Any]:
    opinions = dedupe_opinions(aspect_rows)
    days = sorted(str(o["review_date"])[:10] for o in opinions if o.get("review_date"))
    targets = [a for a in (query.target_aspects or [])]

    in_scope = [
        o for o in opinions
        if query.in_period(o.get("review_date")) and (not targets or o.get("aspect") in targets)
    ]
    matching = [o for o in in_scope if not query.sentiment or o.get("sentiment") == query.sentiment]
    counts = Counter(o["sentiment"] for o in in_scope)
    total = sum(counts.values())

    by_aspect: dict[str, Counter] = defaultdict(Counter)
    by_year: dict[str, Counter] = defaultdict(Counter)
    for o in in_scope:
        by_aspect[o["aspect"]][o["sentiment"]] += 1
        if o.get("review_date"):
            by_year[str(o["review_date"])[:4]][o["sentiment"]] += 1

    focus = query.sentiment or "negative"
    aspect_rows_out = sorted(by_aspect.items(), key=lambda kv: (-kv[1][focus], -sum(kv[1].values())))
    negatives = [o for o in in_scope if o["sentiment"] == "negative"]
    negative_high_star = [o for o in negatives if (o.get("rating") or 0) >= 4]

    return {
        "pham_vi": {
            "chu_de": [aspect_name(a) for a in targets] or "tất cả chủ đề",
            "cam_xuc_duoc_hoi": SENTIMENT_WORD.get(query.sentiment or "", None),
            "thoi_gian": query.period_text() if query.date_from else "toàn bộ dữ liệu",
        },
        "khoang_ngay_co_du_lieu": f"{_format_day(days[0])} - {_format_day(days[-1])}" if days else "",
        "so_y_kien_dung_dieu_kien": len(matching),
        "so_review_dung_dieu_kien": len({o.get("review_id") for o in matching}),
        "so_y_kien_theo_cam_xuc": {
            "khen": counts["positive"], "trung_lap": counts["neutral"], "che": counts["negative"], "tong": total,
            "phan_tram": {
                "khen": _percent(counts["positive"], total),
                "trung_lap": _percent(counts["neutral"], total),
                "che": _percent(counts["negative"], total),
            },
        },
        "theo_chu_de": [
            {"chu_de": aspect_name(a), "khen": c["positive"], "trung_lap": c["neutral"],
             "che": c["negative"], "tong": sum(c.values())}
            for a, c in aspect_rows_out[:6]
        ] if len(targets) != 1 else [],
        "theo_nam": {
            year: {"khen": c["positive"], "trung_lap": c["neutral"], "che": c["negative"]}
            for year, c in sorted(by_year.items())
        },
        "loi_che_trong_review_4_5_sao": {
            "so_loi_che": len(negatives), "nam_trong_review_4_5_sao": len(negative_high_star),
        },
    }


def evidence_for_prompt(evidence: list[dict[str, Any]], review_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i, row in enumerate(evidence, 1):
        info = review_info.get(str(row.get("review_id") or ""), {})
        code = str(row.get("aspect_code") or row.get("aspect") or "")
        rows.append({
            "stt": i,
            "chu_de": aspect_name(code) if code in ASPECT_DISPLAY_NAMES else (row.get("aspect_label") or code),
            "cam_xuc": SENTIMENT_WORD.get(str(row.get("sentiment") or ""), "trung lập"),
            "ngay": _format_day(info.get("review_date")),
            "so_sao": info.get("rating"),
            "cau": " ".join(str(row.get("text") or "").split()),
        })
    return rows


# ── Ngữ cảnh cho bằng chứng ngắn ─────────────────────────────────────────────

MIN_INFORMATIVE_CHARS, MIN_INFORMATIVE_WORDS = 28, 5
CONTEXT_TARGET_CHARS, CONTEXT_MAX_CHARS = 80, 240
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def is_informative(text: str) -> bool:
    text = " ".join(str(text or "").split())
    return len(text) >= MIN_INFORMATIVE_CHARS and len(text.split()) >= MIN_INFORMATIVE_WORDS


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split()).strip(" .!?…")


def add_review_context(evidence: list[dict[str, Any]], review_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Câu bằng chứng quá ngắn ("Rất không hài lòng.") được ghép thêm các câu liền trước
    trong cùng review để người đọc (và LLM) biết khách không hài lòng về điều gì.
    Câu gốc giữ ở trường "sentence" để trích dẫn."""
    out = []
    for row in evidence:
        text = str(row.get("text") or "")
        content = str((review_info.get(str(row.get("review_id") or "")) or {}).get("content") or "")
        if is_informative(text) or not content:
            out.append(row)
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(content) if s.strip()]
        target = _norm(text)
        index = next((i for i, s in enumerate(sentences) if target and target in _norm(s)), None)
        if index is None and target and _norm(content) != target:
            # Câu không nằm trong nội dung (thường là tiêu đề review): nối phần đầu nội dung.
            body = " ".join(content.split())
            if len(body) > CONTEXT_MAX_CHARS:
                body = body[:CONTEXT_MAX_CHARS].rsplit(" ", 1)[0] + "…"
            out.append({**row, "sentence": text, "text": f"{text.strip().rstrip('.')}. {body}",
                        "context_added": True, "context_after": True})
            continue
        if index is None or index == 0:
            out.append(row)
            continue
        window = [sentences[index]]
        i = index - 1
        while i >= 0 and len(" ".join(window)) < CONTEXT_TARGET_CHARS:
            window.insert(0, sentences[i])
            i -= 1
        joined = " ".join(window)
        if len(joined) > CONTEXT_MAX_CHARS:
            joined = "…" + joined[-CONTEXT_MAX_CHARS:].split(" ", 1)[-1]
        elif i >= 0:
            joined = "… " + joined
        out.append({**row, "sentence": text, "text": joined, "context_added": True})
    return out


# ── Câu trả lời soạn sẵn ─────────────────────────────────────────────────────

def _short_quote(text: str, from_end: bool = False) -> str:
    words = " ".join(str(text or "").split()).strip(" .!…").split(" ")
    if len(words) <= MAX_QUOTE_WORDS:
        return " ".join(words)
    if from_end:  # đoạn có ngữ cảnh: phần quan trọng là câu gốc ở cuối
        return "…" + " ".join(words[-MAX_QUOTE_WORDS:])
    return " ".join(words[:MAX_QUOTE_WORDS]) + "…"


def _quote_of(row: dict[str, Any]) -> str:
    from_end = bool(row.get("context_added")) and not row.get("context_after")
    return _short_quote(str(row.get("text") or ""), from_end=from_end)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _join_vi(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " và " + items[-1]


def _is_lookup(query: QueryAnalysis) -> bool:
    return not (query.target_aspects or query.sentiment or query.date_from) and query.intent not in OVERVIEW_INTENTS


# Từ khung câu hỏi, không mang nội dung cần tìm trong review.
_FRAME_WORDS = {
    "khach", "khach hang", "co", "khong", "ai", "nao", "nhac", "den", "toi", "viec", "phai", "san", "pham",
    "nay", "gi", "ve", "the", "bi", "la", "duoc", "cua", "nguoi", "mua", "hay", "chua", "nhu", "thi", "voi",
}
_CONTENT_TOKEN = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> list[str]:
    return [t for t in _CONTENT_TOKEN.findall(fold(text or "")) if t not in _FRAME_WORDS]


def is_yes_no_question(question: str) -> bool:
    folded = fold(question or "").strip()
    return bool(re.search(r"(?<![a-z0-9])(khong|chua)\s*\??\s*$", folded))


def is_detail_question(question: str, query: QueryAnalysis) -> bool:
    """Câu hỏi có/không về một chi tiết cụ thể ("có ai nhắc chờ 2 tuần không", "khăn có bị
    khô không"), khác câu hỏi đánh giá chung theo chủ đề ("dịch vụ của shop có tốt không")."""
    if not is_yes_no_question(question) or query.sentiment or query.date_from:
        return False
    if asks_for_advice(question) or query.intent in OVERVIEW_INTENTS:
        return False
    folded = fold(question)
    specific = bool(re.search(r"\d", folded)) or bool(re.search(r"(?<![a-z])(co ai|nhac den|nhac toi)(?![a-z])", folded))
    return specific or not query.target_aspects


def mentions_directly(row: dict[str, Any], question: str, threshold: float = 0.6) -> bool:
    """Câu review chứa phần lớn từ nội dung của câu hỏi (sau khi bỏ dấu và bỏ từ khung)."""
    wanted = set(_content_tokens(question))
    if not wanted:
        return False
    have = set(_CONTENT_TOKEN.findall(fold(str(row.get("sentence") or row.get("text") or ""))))
    return len(wanted & have) / len(wanted) >= threshold


def _detail_answer(evidence: list[dict[str, Any]], question: str) -> str:
    direct = [row for row in evidence if mentions_directly(row, question)]
    if direct:
        quotes = [f"khách {SENTIMENT_WORD.get(str(row.get('sentiment')), 'nhắc')} \"{_quote_of(row)}\"" for row in direct[:3]]
        return f"Có {len(direct)} ý kiến nhắc trực tiếp đến điều này: " + "; ".join(quotes) + "."
    if not evidence:
        return "Không thấy ý kiến nào của khách nhắc đến điều này."
    # So khớp theo từ không đủ để khẳng định "không có" ("độ đặc" và "lỏng hơn" không trùng chữ
    # nhưng cùng ý), nên chỉ đưa ra các ý kiến liên quan nhất, không kết luận.
    parts = [f"khách {SENTIMENT_WORD.get(str(row.get('sentiment')), 'nhắc')} \"{_quote_of(row)}\"" for row in evidence[:3]]
    return "Các ý kiến liên quan nhất đến câu hỏi này: " + "; ".join(parts) + "."


_MAPPING_PHRASES: list[tuple[str, list[str]]] | None = None


def _mapping_phrases() -> list[tuple[str, list[str]]]:
    global _MAPPING_PHRASES
    if _MAPPING_PHRASES is None:
        from app.assistant.query_analyzer import QueryAnalyzer

        mapping = QueryAnalyzer().mapping
        _MAPPING_PHRASES = sorted(
            ((phrase, list(cfg.get("aspects") or [])) for phrase, cfg in mapping.items()),
            key=lambda item: -len(item[0]),
        )
    return _MAPPING_PHRASES


def user_terms(question: str, aspects: list[str]) -> list[str]:
    """Cụm từ người hỏi dùng cho các chủ đề trong câu hỏi, giữ nguyên chữ gốc ("mẫu mã", "giao hàng").

    Khớp cụm dài nhất trước như bộ phân tích câu hỏi; fold() giữ nguyên độ dài nên cắt được chữ gốc.
    """
    folded = fold(question or "")
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for phrase, phrase_aspects in _mapping_phrases():
        if not set(phrase_aspects) & set(aspects):
            continue
        pattern = r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?![a-z0-9])"
        for m in re.finditer(pattern, folded):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), question[m.start():m.end()]))
    terms: list[str] = []
    for _, term in sorted(found):
        term = term.strip()
        if term.lower() not in (t.lower() for t in terms):
            terms.append(term[0].lower() + term[1:] if term[:1].isupper() and not term[1:2].isupper() else term)
    return terms


def _question_form(question: str) -> str:
    folded = fold(question or "")
    if re.search(r"(?<![a-z])bao nhieu(?![a-z])", folded):
        return "count"
    if is_yes_no_question(question):
        return "yes_no"
    return "what"


def asks_for_advice(question: str) -> bool:
    return bool(re.search(r"\b(có nên|nên mua|đáng mua|có đáng|nên chọn)\b", question or "", re.IGNORECASE))


def template_answer(
    facts: dict[str, Any], evidence: list[dict[str, Any]], query: QueryAnalysis, question: str = ""
) -> str:
    quotes_pool = evidence
    if query.sentiment:
        quotes_pool = [row for row in evidence if row.get("sentiment") == query.sentiment] or evidence

    if is_detail_question(question, query):
        return _detail_answer(evidence, question)
    if _is_lookup(query):
        if not evidence:
            return "Không thấy ý kiến nào của khách liên quan đến câu hỏi này."
        parts = [f"khách {SENTIMENT_WORD.get(str(row.get('sentiment')), 'nhắc')} \"{_quote_of(row)}\""
                 for row in evidence[:3]]
        return "Các ý kiến liên quan nhất tìm được trong review: " + "; ".join(parts) + "."

    counts = facts["so_y_kien_theo_cam_xuc"]
    total = counts["tong"]
    period = facts["pham_vi"]["thoi_gian"]
    if not query.date_from:
        period_phrase = ""
    elif period.startswith("từ"):
        period_phrase = f" {period}"
    else:
        period_phrase = f" trong {period}"
    targets = query.target_aspects or []
    # Gọi chủ đề bằng đúng từ người hỏi dùng ("mẫu mã", không đổi thành "thiết kế & kiểu dáng").
    subject = _join_vi(user_terms(question, targets)) or _join_vi([aspect_name(a).lower() for a in targets])
    form = _question_form(question)
    pct = counts["phan_tram"]
    mix = _join_vi([f"{counts[key]} {label} ({pct[key]}%)"
                    for key, label in (("khen", "khen"), ("trung_lap", "trung lập"), ("che", "chê")) if counts[key]])
    scope = f"Về {subject}{period_phrase}" if subject else (
        _capitalize(period_phrase.strip()) if query.date_from else "")
    sentences: list[str] = []
    quotes_in_lead = False
    top_line_in_lead = False

    def top_aspects(key: str, limit: int = 3) -> str:
        col = {"positive": "khen", "neutral": "trung_lap", "negative": "che"}[key]
        rows = [r for r in sorted(facts["theo_chu_de"], key=lambda r: -r[col]) if r[col]][:limit]
        return _join_vi([f"{r['chu_de'].lower()} ({r[col]})" for r in rows])

    def lead_quotes(rows: list[dict[str, Any]], limit: int = 2) -> list[str]:
        return [f"\"{_quote_of(row)}\"" for row in rows
                if row.get("text") and not ADVICE_PATTERN.search(str(row.get("text")))][:limit]

    if query.sentiment:
        word = SENTIMENT_WORD[query.sentiment]
        n = facts["so_y_kien_dung_dieu_kien"]
        about = f" về {subject}" if subject else ""
        of_total = f", trên tổng {total} ý kiến" + (f" về {subject}" if subject else "") if total > n else ""
        quotes = lead_quotes(quotes_pool)
        if form == "count":
            sentences.append(f"Có {n} ý kiến {word}{about}{period_phrase}{of_total}.")
        elif form == "yes_no":
            sentences.append(f"Có. {n} ý kiến {word}{about}{period_phrase}{of_total}.")
        elif subject and quotes:
            # "Khách chê gì về mẫu mã?" -> nêu ngay khách chê điều gì.
            sentences.append(f"{scope}, khách {word} ở những điểm như " + _join_vi(quotes) + f" ({n} ý kiến{of_total}).")
            quotes_in_lead = True
        elif facts["theo_chu_de"]:
            lead = f"{scope}, khách" if scope else "Khách"
            sentences.append(f"{lead} {word} nhiều nhất về {top_aspects(query.sentiment)} ({n} ý kiến{of_total}).")
            top_line_in_lead = True
        else:
            sentences.append(f"Có {n} ý kiến {word}{about}{period_phrase}{of_total}.")
    else:
        if subject and form == "yes_no":
            sentences.append(f"{scope}, có {counts['khen']} ý kiến khen và {counts['che']} ý kiến chê "
                             f"(trên {total} ý kiến).")
        elif subject:
            sentences.append(f"{scope}, khách có {total} ý kiến: {mix}.")
        elif re.search(r"(?<![a-z])tom tat(?![a-z])", fold(question)):
            sentences.append(f"Tóm tắt review{period_phrase}: có {total} ý kiến, gồm {mix}.")
        elif scope:
            sentences.append(f"{scope}, khách có {total} ý kiến về sản phẩm: {mix}.")
        else:
            sentences.append(f"Nhận xét chung của khách trong dữ liệu đã phân tích: {total} ý kiến, gồm {mix}.")

    details: list[str] = []
    if facts["theo_chu_de"] and not top_line_in_lead:
        focus_keys = [query.sentiment] if query.sentiment else ["negative", "positive"]
        for key in focus_keys:
            word = SENTIMENT_WORD[key]
            col = {"positive": "khen", "neutral": "trung_lap", "negative": "che"}[key]
            top = [row for row in sorted(facts["theo_chu_de"], key=lambda r: -r[col]) if row[col]][:3]
            if top:
                details.append(f"Khách {word} nhiều nhất về " + _join_vi([f"{r['chu_de'].lower()} ({r[col]})" for r in top]) + ".")

    focus = query.sentiment or "negative"
    col = {"positive": "khen", "neutral": "trung_lap", "negative": "che"}[focus]
    years = {y: c[col] for y, c in facts["theo_nam"].items() if c[col]}
    focus_total = sum(years.values())
    if len(years) >= 2 and focus_total >= 3:
        top_year, top_count = max(years.items(), key=lambda kv: kv[1])
        if top_count * 2 >= focus_total:
            details.append(f"Phần lớn ý kiến {SENTIMENT_WORD[focus]} rơi vào năm {top_year} ({top_count}/{focus_total} ý kiến).")

    if not quotes_in_lead:
        if not query.sentiment and subject:
            # Câu hỏi chung về một chủ đề: một lời khen và một lời chê để thấy cả hai phía.
            pos = [r for r in evidence if r.get("sentiment") == "positive"]
            neg = [r for r in evidence if r.get("sentiment") == "negative"]
            sides = []
            if pos and lead_quotes(pos, 1):
                sides.append("khách khen " + lead_quotes(pos, 1)[0])
            if neg and lead_quotes(neg, 1):
                sides.append("khách chê " + lead_quotes(neg, 1)[0])
            if sides:
                details.append("Ví dụ, " + "; ".join(sides) + ".")
        else:
            quotes = lead_quotes(quotes_pool)
            if quotes:
                details.append("Ví dụ, khách viết: " + _join_vi(quotes) + ".")

    high = facts["loi_che_trong_review_4_5_sao"]
    if focus == "negative" and high["so_loi_che"] >= 3 and high["nam_trong_review_4_5_sao"]:
        details.append(f"Có {high['nam_trong_review_4_5_sao']}/{high['so_loi_che']} lời chê nằm trong review 4–5 sao.")

    if asks_for_advice(question):
        sentences.insert(0, "Trợ lý không đánh giá nên hay không nên mua, mà tóm tắt khách đã nói gì.")
    answer = " ".join(sentences)
    if details:
        answer += "\n\n" + " ".join(details)
    return answer


# ── Prompt cho LLM ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Bạn là trợ lý phân tích review cho nhân viên và chủ shop bán hàng. Người đọc là người bán, "
    "không phải người mua. Nhiệm vụ: thuật lại khách hàng đã nói gì về sản phẩm, bằng số liệu và "
    "trích dẫn có trong dữ liệu được cung cấp. Luôn viết tiếng Việt có dấu, văn xuôi tự nhiên. "
    "Chỉ trả về JSON hợp lệ."
)

ANSWER_RULES = """QUY TẮC VIẾT CÂU TRẢ LỜI:
1. Chỉ trả lời đúng điều được hỏi. Không nêu tên sản phẩm, phần trăm, phân bố theo năm, số lời chê
   trong review 4-5 sao hay chủ đề khác nếu câu hỏi không hỏi đến. Bỏ qua ý kiến minh họa không liên
   quan, không viết câu kiểu "các ý kiến khác không liên quan đến ...".
2. Câu đầu tiên trả lời thẳng câu hỏi, đúng dạng câu hỏi, dùng lại cụm từ của câu hỏi (hỏi "khách
   hài lòng điều gì" thì mở đầu "Khách hài lòng nhất về ..."):
   - hỏi "gì / những gì / ra sao / thế nào": nêu ngay khách khen/chê/nói điều gì (nội dung), số
     lượng để trong ngoặc hoặc câu sau;
   - hỏi "bao nhiêu": nêu con số trong SO_LIEU;
   - hỏi "có ... không": mở đầu bằng "Có" (kèm số ý kiến) hoặc nói rõ số ý kiến khen và chê.
   Gọi chủ đề bằng đúng các từ trong CACH_GOI_CHU_DE (hỏi "bao bì" thì viết "bao bì", không đổi
   thành "chất lượng đóng gói"); gọi thời gian theo cách người hỏi nói (hỏi "năm ngoái" thì viết
   "năm ngoái (năm 2025)").
   Nếu câu hỏi về một chi tiết cụ thể (ví dụ khăn có bị khô không) thì dựa vào Y_KIEN_MINH_HOA để
   nói khách nhắc đến điều đó thế nào. Chỉ thuật lại khách nói gì, không tự kết luận thay khách
   (review viết "không bị khô" thì không được trả lời "khăn bị khô"). Chỉ nói "không có ý kiến
   nào" khi Y_KIEN_MINH_HOA thật sự không có câu nào liên quan.
3. Sau đó 1 đoạn ngắn: khách khen/chê điều gì cụ thể và lý do chính khách nêu ra, kèm 1-2 trích
   dẫn ngắn (tối đa 15 từ) trong ngoặc kép từ Y_KIEN_MINH_HOA. Chỉ nói lời khen/chê tập trung vào
   năm nào (theo_nam) khi câu hỏi hỏi về thời gian hay xu hướng.
4. Chỉ dùng con số có trong SO_LIEU hoặc Y_KIEN_MINH_HOA. Được tính phần trăm từ các số đó.
   Không bịa số, không suy diễn nguyên nhân mà khách không nói.
5. KHÔNG đưa lời khuyên hay khuyến cáo cho người mua hoặc cho shop. Không dùng các từ: nên, hãy,
   cẩn thận, yên tâm, đáng mua, khuyên, khuyến nghị, cân nhắc, quyết định. Không gọi người đọc là
   "bạn". Không tự đánh giá sản phẩm thay khách (không viết "sản phẩm tốt", "phù hợp với bé").
   Nếu câu hỏi xin lời khuyên (ví dụ "có nên mua không") thì nói trợ lý không đánh giá nên hay không
   và tóm tắt khách đã khen, chê gì.
6. PHAM_VI là bộ lọc đã áp dụng cho mọi số liệu và ý kiến. Chỉ nói trong phạm vi đó và nêu rõ
   khoảng thời gian nếu câu hỏi có hỏi về thời gian. KHÔNG mở đầu bằng khoảng ngày có dữ liệu
   (khoang_ngay_co_du_lieu) khi câu hỏi không hỏi về thời gian. Gọi số đếm là "ý kiến", không gọi
   là "đánh giá" hay "review" (một review có thể chứa nhiều ý kiến).
7. Tối đa khoảng 90 từ. Không tiêu đề, không gạch đầu dòng, không mã chủ đề kỹ thuật.

Trả về đúng JSON: {"answer": "<câu trả lời>"}"""


def build_llm_prompt(question: str, product_name: str, facts: dict[str, Any], evidence_rows: list[dict[str, Any]]) -> str:
    no_period = facts["pham_vi"].get("thoi_gian") == "toàn bộ dữ liệu"
    # Câu hỏi không hỏi về thời gian thì không đưa khoảng ngày có dữ liệu, để LLM không mở đầu
    # bằng "Trong khoảng thời gian từ ... đến ...".
    skip = {"pham_vi", "khoang_ngay_co_du_lieu"} if no_period else {"pham_vi"}
    # Số liệu phụ không được hỏi làm câu trả lời lan man (answer_relevancy thấp), nên không gửi.
    skip |= {"loi_che_trong_review_4_5_sao"}
    if not no_period:
        skip |= {"theo_nam"}  # đã lọc theo thời gian thì phân bố theo năm là thừa
    context = {
        "SAN_PHAM": product_name,
        "CAU_HOI": question,
        "PHAM_VI": facts["pham_vi"],
        # Đúng chữ người hỏi dùng cho chủ đề; câu trả lời phải gọi chủ đề bằng các từ này.
        "CACH_GOI_CHU_DE": user_terms(question, list(ASPECT_DISPLAY_NAMES)),
        "SO_LIEU": {k: v for k, v in facts.items() if k not in skip},
        "Y_KIEN_MINH_HOA": evidence_rows,
    }
    return json.dumps(context, ensure_ascii=False, indent=2) + "\n\n" + ANSWER_RULES


# ── Kiểm tra câu trả lời của LLM ─────────────────────────────────────────────

def _numbers(text: str) -> set[int]:
    out = set()
    for token in NUMBER_PATTERN.findall(text):
        digits = re.sub(r"[.,]", "", token)
        if digits.isdigit():
            out.add(int(digits))
    return out


MAX_ANSWER_WORDS = 140  # prompt yêu cầu ~90 từ; dài hơn nhiều thì dùng câu trả lời mẫu


def check_llm_answer(answer: str, facts: dict[str, Any], evidence_rows: list[dict[str, Any]], question: str) -> str | None:
    """Trả về lý do không đạt, hoặc None nếu câu trả lời dùng được."""
    if not answer or len(answer.split()) < 4:
        return "câu trả lời rỗng"
    if len(answer.split()) > MAX_ANSWER_WORDS:
        return f"quá dài ({len(answer.split())} từ)"
    outside_quotes = QUOTED_PATTERN.sub(" ", answer)
    # Câu từ chối đúng mẫu ("không đánh giá nên hay không nên mua") không phải lời khuyên.
    checked = re.sub(r"(có )?nên hay không( nên)?( mua)?", " ", outside_quotes, flags=re.IGNORECASE)
    advice = ADVICE_PATTERN.search(checked)
    if advice:
        return f"có từ mang tính khuyên: '{advice.group(0)}'"

    # 0-12: số sao, số tháng.
    allowed = _numbers(json.dumps(facts, ensure_ascii=False)) | _numbers(question) | set(range(0, 13))
    for row in evidence_rows:
        allowed |= _numbers(json.dumps(row, ensure_ascii=False))
    allowed |= {100}
    # Cho phép phần trăm tính từ hai số bất kỳ trong số liệu.
    bases = sorted(n for n in allowed if 0 < n < 100000)
    percents = {round(100 * a / b) for b in bases for a in bases if a <= b}
    unknown = sorted(n for n in _numbers(outside_quotes) if n not in allowed and n not in percents)
    if unknown:
        return f"có số không có trong dữ liệu: {unknown}"
    return None
