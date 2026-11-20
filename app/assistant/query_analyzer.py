from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from app.assistant.llm import LLMError
from app.web.constants import ASPECT_DISPLAY_NAMES
from app.assistant.time_parser import FUZZY_MARKERS, describe_period, extract_time_range, fold


ALLOWED_INTENTS = {
    "SUMMARY",
    "PRODUCT_QA",
    "ASPECT_QA",
    "COMPLAINT_SUMMARY",
    "RISK_CHECK",
    "PRODUCT_FIT",
    "RECOMMENDATION",
    "PRICE_VALUE",
    "DELIVERY_SERVICE",
    "UNKNOWN",
}

# Đúng 17 chủ đề mà PhoBERT gán nhãn; chủ đề ngoài danh sách này không có dữ liệu để lọc.
ALLOWED_ASPECTS = set(ASPECT_DISPLAY_NAMES)

# Cụm chứa từ khóa nhưng không mang nghĩa chủ đề ("đánh giá" chứa "giá").
BLOCKING_PHRASES = ["danh gia"]


# Cụm từ cho biết người hỏi muốn lọc theo cảm xúc. Chỉ dùng cụm rõ nghĩa;
# câu hỏi kiểu "có tốt không" là hỏi đánh giá chứ không phải lọc ý kiến tích cực.
NEGATIVE_FILTER_PHRASES = [
    "che", "phan nan", "khong hai long", "tieu cuc", "diem yeu", "phan anh", "than phien", "khieu nai",
    "chua tot", "khong tot", "chua hai long", "nhuoc diem", "han che", "diem tru",
]
POSITIVE_FILTER_PHRASES = ["khen", "tich cuc", "uu diem", "diem manh", "hai long"]
NEUTRAL_FILTER_PHRASES = ["trung lap"]


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str = "UNKNOWN"
    target_aspects: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    source: str = "rule"
    # Bộ lọc dùng chung cho thống kê, bằng chứng và tìm vector.
    sentiment: str | None = None
    date_from: str | None = None  # ISO, tính cả hai đầu
    date_to: str | None = None
    time_expression: str = ""

    def to_prompt_context(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "target_aspects": self.target_aspects,
            "constraints": self.constraints,
            "source": self.source,
            "sentiment": self.sentiment,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "time_expression": self.time_expression,
        }

    def has_filters(self) -> bool:
        return bool(self.target_aspects or self.sentiment or self.date_from)

    def in_period(self, day: date | str | None) -> bool:
        """True nếu không lọc thời gian, hoặc ngày nằm trong khoảng đã lọc."""
        if not self.date_from:
            return True
        if not day:
            return False
        value = str(day)[:10]
        return self.date_from <= value <= str(self.date_to)

    def period_text(self) -> str:
        period = describe_period(self.date_from, self.date_to)
        expression = self.time_expression.strip()
        if period and expression and (FUZZY_MARKERS.search(fold(expression)) or not re.search(r"\d", expression)):
            # "đầu năm 2023", "năm ngoái" được quy thành một khoảng ngày: giữ cách nói của người hỏi
            # và nêu khoảng ngày đã lọc, ví dụ "năm ngoái (năm 2025)".
            return f"{expression[0].lower()}{expression[1:]} ({period})"
        return period


class QueryAnalyzer:
    def __init__(self, mapping_path: str | Path | None = None, llm: Any | None = None) -> None:
        self.mapping_path = Path(mapping_path or Path(__file__).with_name("aspect_mapping.json"))
        self.mapping = _load_mapping(self.mapping_path)
        self.llm = llm

    def analyze(self, question: str) -> QueryAnalysis:
        normalized = normalize_text(question)
        rule = self._analyze_by_rule(normalized)
        llm = self._analyze_by_llm(question)
        # Chủ đề luôn lấy theo luật: mỗi chủ đề phải có một cụm trong câu hỏi làm căn cứ.
        # LLM chỉ góp phần nhận diện loại câu hỏi (intent); để LLM tự gán chủ đề từng biến
        # "khăn có bị khô không" thành câu hỏi về chất lượng chung.
        result = replace(llm, target_aspects=rule.target_aspects) if llm else rule
        return attach_filters(question, result)

    def match_aspects(self, normalized: str) -> list[str]:
        """Khớp cụm dài nhất trước; vùng chữ đã khớp không được dùng lại cho cụm ngắn hơn.

        Nhờ vậy "tốc độ giao hàng" chỉ là DELIVERY#SPEED, còn "giao hàng" (nói chung)
        là cả nhóm giao hàng.
        """
        candidates: list[tuple[int, int, list[str]]] = []
        for phrase in [*self.mapping, *BLOCKING_PHRASES]:
            aspects = (self.mapping.get(phrase) or {}).get("aspects") or []
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized):
                candidates.append((m.start(), m.end(), aspects))
        candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
        taken: list[tuple[int, int]] = []
        matched: list[tuple[int, list[str]]] = []
        for start, end, aspects in candidates:
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            taken.append((start, end))
            matched.append((start, aspects))
        out: list[str] = []
        for _, aspects in sorted(matched, key=lambda m: m[0]):
            for aspect in aspects:
                if aspect in ALLOWED_ASPECTS and aspect not in out:
                    out.append(aspect)
        return out

    def _analyze_by_rule(self, normalized: str) -> QueryAnalysis:
        intent = _detect_intent(normalized)
        aspects = self.match_aspects(normalized)
        if aspects and intent in {"PRODUCT_QA", "COMPLAINT_SUMMARY", "PRICE_VALUE", "DELIVERY_SERVICE"}:
            intent = "ASPECT_QA"
        return QueryAnalysis(intent=intent, target_aspects=aspects, source="rule")

    def _analyze_by_llm(self, question: str) -> QueryAnalysis | None:
        if self.llm is None or not self.llm.is_configured():
            return None
        system_prompt = (
            "Ban la query analyzer cho chatbot review san pham Me & Be. "
            "Map cau hoi nguoi dung sang intent va aspect ABSA hop le. "
            "Chi tra ve JSON, khong giai thich, khong xuat <think>."
        )
        user_prompt = (
            f"Allowed intents: {sorted(ALLOWED_INTENTS)}\n"
            f"Allowed aspects: {sorted(ALLOWED_ASPECTS)}\n"
            "Schema: {\"intent\":\"\", \"target_aspects\":[], \"constraints\":[]}\n"
            f"Question: {question}"
        )
        try:
            raw = self.llm.generate_json(system_prompt, user_prompt)
        except LLMError:
            return None
        return _coerce_llm_result(raw)


def attach_filters(question: str, query: QueryAnalysis) -> QueryAnalysis:
    """Gắn cảm xúc và khoảng thời gian trích bằng luật vào kết quả phân tích.

    Chạy cho cả nhánh LLM lẫn nhánh luật, để bộ lọc không phụ thuộc việc LLM
    có đang hoạt động hay không. Mốc thời gian tường minh luôn lấy theo luật.
    """
    time_range = extract_time_range(question)
    return replace(
        query,
        sentiment=query.sentiment or detect_sentiment_filter(question),
        date_from=time_range.date_from.isoformat() if time_range else None,
        date_to=time_range.date_to.isoformat() if time_range else None,
        time_expression=time_range.expression if time_range else "",
    )


def detect_sentiment_filter(question: str) -> str | None:
    normalized = normalize_text(question)
    negative = _has_any_phrase(normalized, NEGATIVE_FILTER_PHRASES)
    # "không hài lòng" chứa "hài lòng": bỏ cụm phủ định trước khi tìm ý tích cực.
    positive = _has_any_phrase(re.sub(r"(khong|chua) hai long", " ", normalized), POSITIVE_FILTER_PHRASES)
    if negative and positive:
        return None  # hỏi cả khen lẫn chê: không lọc theo cảm xúc
    if negative:
        return "negative"
    if positive:
        return "positive"
    if _has_any_phrase(normalized, NEUTRAL_FILTER_PHRASES):
        return "neutral"
    return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9#\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {normalize_text(key): value for key, value in raw.items() if isinstance(value, dict)}


def _detect_intent(q: str) -> str:
    if _has_any_phrase(q, ["tom tat", "tong quan", "review", "danh gia chung"]):
        return "SUMMARY"
    if _has_any_phrase(q, ["co bi", "co gay", "co mui", "co mem", "co tham", "bi tran", "gay ham"]):
        return "ASPECT_QA"
    if _has_any_phrase(q, ["phan nan", "che", "khong hai long", "diem yeu", "te nhat"]):
        return "COMPLAINT_SUMMARY"
    if _has_any_phrase(q, ["luu y", "rui ro", "canh bao", "co van de", "nguy hiem"]):
        return "RISK_CHECK"
    if _has_any_phrase(q, ["co nen mua", "nen mua", "mua khong", "dang mua"]):
        return "RECOMMENDATION"
    if _has_any_phrase(q, ["phu hop", "hop voi", "dung cho", "da nhay cam", "tre so sinh", "em be"]):
        return "PRODUCT_FIT"
    if _has_any_phrase(q, ["dang tien", "gia ca", "gia tien", "muc gia", "re", "dat", "ngan sach"]) or _price_word_in_text(q):
        return "PRICE_VALUE"
    if _has_any_phrase(q, ["giao hang", "ship", "dong goi", "van chuyen"]):
        return "DELIVERY_SERVICE"
    if q:
        return "PRODUCT_QA"
    return "UNKNOWN"


def _phrase_in_text(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _price_word_in_text(text: str) -> bool:
    without_review_phrase = re.sub(r"(?<![a-z0-9])danh\s+gia(?![a-z0-9])", " ", text)
    return _phrase_in_text("gia", without_review_phrase)


def _has_any_phrase(text: str, phrases: list[str]) -> bool:
    return any(_phrase_in_text(phrase, text) for phrase in phrases)


def _coerce_llm_result(raw: dict[str, Any]) -> QueryAnalysis | None:
    intent = str(raw.get("intent") or "UNKNOWN").strip().upper()
    if intent not in ALLOWED_INTENTS:
        intent = "UNKNOWN"
    aspects = [
        str(aspect)
        for aspect in raw.get("target_aspects") or []
        if str(aspect) in ALLOWED_ASPECTS
    ]
    constraints_raw = raw.get("constraints") or []
    if isinstance(constraints_raw, dict):
        constraints = [f"{key}:{value}" for key, value in constraints_raw.items() if value]
    else:
        constraints = [str(item) for item in constraints_raw if item]
    return QueryAnalysis(
        intent=intent,
        target_aspects=list(dict.fromkeys(aspects)),
        constraints=list(dict.fromkeys(constraints)),
        source="llm",
    )
