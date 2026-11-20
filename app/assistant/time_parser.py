"""Nhận diện khoảng thời gian trong câu hỏi tiếng Việt bằng luật.

Xử lý mốc tường minh (ngày, tháng, quý, năm, khoảng "từ ... đến ..."), vài cách nói
tương đối cố định ("năm ngoái", "tháng trước") và cách nói mơ hồ có kèm mốc ("đầu năm
2023", "cuối tháng 3/2022", "dịp Tết 2022"). Cách nói không có mốc ("gần đây", "dạo này",
"đầu năm" không kèm năm) trả về None: thà không lọc còn hơn lọc sai.
"""
from __future__ import annotations

import calendar
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

VN_TZ = timezone(timedelta(hours=7))
MIN_YEAR, MAX_YEAR = 1990, 2100

# Có các cụm này thì câu hỏi nói về một khoảng mơ hồ; để LLM (hoặc không lọc) xử lý.
FUZZY_MARKERS = re.compile(
    r"(?<![0-9a-z])(dau nam|cuoi nam|giua nam|dau thang|cuoi thang|giua thang|dip|tet|gan day|dao nay|hoi truoc)(?![0-9a-z])"
)


@dataclass(frozen=True)
class TimeRange:
    date_from: date
    date_to: date
    expression: str  # đoạn nguyên văn trong câu hỏi

    def contains(self, day: date | None) -> bool:
        return day is not None and self.date_from <= day <= self.date_to


def reference_today() -> date:
    """Ngày hiện tại theo giờ Việt Nam; RAG_REFERENCE_DATE cố định mốc khi đo."""
    raw = os.getenv("RAG_REFERENCE_DATE", "").strip()
    if raw:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            pass
    return datetime.now(VN_TZ).date()


def fold(text: str) -> str:
    """Chữ thường, bỏ dấu, giữ nguyên độ dài để vị trí khớp được với câu gốc."""
    out = []
    for ch in text:
        if ch in "đĐ":
            out.append("d")
            continue
        base = unicodedata.normalize("NFD", ch)[0].lower()
        out.append(base if len(base) == 1 else ch)
    return "".join(out)


def _year(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = int(raw)
    if len(raw) == 2:
        value += 2000
    return value if MIN_YEAR <= value <= MAX_YEAR else None


def _month_range(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


_MONTH = r"(\d{1,2})"
_YEAR = r"(\d{4}|\d{2})"
_SEP = r"\s*(?:nam\s+|/|-|\s)\s*"
_B = r"(?<![0-9a-z])"   # đầu từ
_E = r"(?![0-9a-z])"    # cuối từ

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("month_range", re.compile(
        rf"{_B}tu\s+thang\s+{_MONTH}(?:{_SEP}{_YEAR})?\s+(?:den|toi)\s+thang\s+{_MONTH}(?:{_SEP}{_YEAR})?{_E}")),
    ("year_range", re.compile(rf"{_B}tu\s+nam\s+(\d{{4}})\s+(?:den|toi)\s+(?:nam\s+)?(\d{{4}}){_E}")),
    ("day", re.compile(r"(?<![0-9/])(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?![0-9/])")),
    ("month", re.compile(rf"{_B}(?:thang|t)\s*{_MONTH}{_SEP}{_YEAR}{_E}")),
    ("month_slash", re.compile(r"(?<![0-9/])(\d{1,2})/(\d{4})(?![0-9/])")),
    ("quarter", re.compile(rf"{_B}(?:quy|q)\s*([1-4]){_SEP}(\d{{4}}){_E}")),
    ("year", re.compile(rf"{_B}nam\s+(\d{{4}}){_E}")),
    ("relative", re.compile(rf"{_B}(nam ngoai|nam truoc|nam roi|nam nay|thang truoc|thang roi|thang nay|hom nay|hom qua){_E}")),
]


# Mùng 1 Tết Nguyên Đán theo dương lịch.
TET_DAYS = {
    2016: date(2016, 2, 8), 2017: date(2017, 1, 28), 2018: date(2018, 2, 16), 2019: date(2019, 2, 5),
    2020: date(2020, 1, 25), 2021: date(2021, 2, 12), 2022: date(2022, 2, 1), 2023: date(2023, 1, 22),
    2024: date(2024, 2, 10), 2025: date(2025, 1, 29), 2026: date(2026, 2, 17), 2027: date(2027, 2, 6),
    2028: date(2028, 1, 26), 2029: date(2029, 2, 13), 2030: date(2030, 2, 3),
}
TET_BEFORE, TET_AFTER = 15, 14  # "dịp Tết": nửa tháng trước đến hai tuần sau mùng 1

_REL_YEAR = r"(\d{4}|nay|ngoai|truoc|roi)"
FUZZY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("tet", re.compile(rf"{_B}(?:dip\s+)?tet(?:\s+nguyen\s+dan)?\s+(?:nam\s+)?{_REL_YEAR}{_E}")),
    ("part_year", re.compile(rf"{_B}(dau|giua|cuoi)\s+nam\s+{_REL_YEAR}{_E}")),
    ("part_month", re.compile(rf"{_B}(dau|giua|cuoi)\s+thang\s+{_MONTH}{_SEP}{_YEAR}{_E}")),
]


def _relative_year(raw: str, today: date) -> int | None:
    if raw == "nay":
        return today.year
    if raw in {"ngoai", "truoc", "roi"}:
        return today.year - 1
    return _year(raw)


def _resolve_fuzzy(kind: str, m: re.Match[str], today: date) -> tuple[date, date] | None:
    try:
        if kind == "tet":
            year = _relative_year(m.group(1), today)
            tet = TET_DAYS.get(year or 0)
            return (tet - timedelta(days=TET_BEFORE), tet + timedelta(days=TET_AFTER)) if tet else None
        if kind == "part_year":
            part, year = m.group(1), _relative_year(m.group(2), today)
            if year is None:
                return None
            months = {"dau": (1, 3), "giua": (5, 8), "cuoi": (10, 12)}[part]
            return _month_range(year, months[0])[0], _month_range(year, months[1])[1]
        if kind == "part_month":
            part, month, year = m.group(1), int(m.group(2)), _year(m.group(3))
            if year is None or not 1 <= month <= 12:
                return None
            first, last = _month_range(year, month)
            days = {"dau": (1, 10), "giua": (11, 20), "cuoi": (21, last.day)}[part]
            return first.replace(day=days[0]), first.replace(day=days[1])
    except ValueError:
        return None
    return None


def extract_time_range(question: str, today: date | None = None) -> TimeRange | None:
    folded = fold(question or "")
    today = today or reference_today()
    # Cách nói mơ hồ nhưng phổ biến ("đầu năm 2023", "dịp Tết 2022") được quy về khoảng cố định.
    for kind, pattern in FUZZY_PATTERNS:
        for match in pattern.finditer(folded):
            span = _resolve_fuzzy(kind, match, today)
            if span is not None:
                return TimeRange(span[0], span[1], question[match.start():match.end()].strip())
    if FUZZY_MARKERS.search(folded):
        return None  # còn lại ("gần đây", "đầu năm" không kèm năm...): không đoán
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(folded):
            span = _resolve(kind, match, today)
            if span is not None:
                start, end = span
                return TimeRange(start, end, question[match.start():match.end()].strip())
    return None


def _resolve(kind: str, m: re.Match[str], today: date) -> tuple[date, date] | None:
    try:
        if kind == "month_range":
            m1, y1, m2, y2 = int(m.group(1)), _year(m.group(2)), int(m.group(3)), _year(m.group(4))
            year_to = y2 or y1
            year_from = y1 or year_to
            if year_to is None or not (1 <= m1 <= 12 and 1 <= m2 <= 12):
                return None
            start, end = _month_range(year_from, m1)[0], _month_range(year_to, m2)[1]
            return (start, end) if start <= end else None
        if kind == "year_range":
            y1, y2 = _year(m.group(1)), _year(m.group(2))
            if y1 is None or y2 is None or y1 > y2:
                return None
            return date(y1, 1, 1), date(y2, 12, 31)
        if kind == "day":
            day = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            return (day, day) if MIN_YEAR <= day.year <= MAX_YEAR else None
        if kind in {"month", "month_slash"}:
            month, year = int(m.group(1)), _year(m.group(2))
            if year is None or not 1 <= month <= 12:
                return None
            return _month_range(year, month)
        if kind == "quarter":
            quarter, year = int(m.group(1)), _year(m.group(2))
            if year is None:
                return None
            return _month_range(year, 3 * quarter - 2)[0], _month_range(year, 3 * quarter)[1]
        if kind == "year":
            year = _year(m.group(1))
            return (date(year, 1, 1), date(year, 12, 31)) if year else None
        if kind == "relative":
            phrase = m.group(1)
            if phrase in {"nam ngoai", "nam truoc", "nam roi"}:
                return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
            if phrase == "nam nay":
                return date(today.year, 1, 1), date(today.year, 12, 31)
            if phrase in {"thang truoc", "thang roi"}:
                last = today.replace(day=1) - timedelta(days=1)
                return _month_range(last.year, last.month)
            if phrase == "thang nay":
                return _month_range(today.year, today.month)
            if phrase == "hom nay":
                return today, today
            if phrase == "hom qua":
                day = today - timedelta(days=1)
                return day, day
    except ValueError:  # ví dụ 31/02
        return None
    return None


def describe_period(date_from: date | str | None, date_to: date | str | None) -> str:
    """Mô tả khoảng ngày bằng tiếng Việt, ví dụ 'tháng 12/2023', 'năm 2022'."""
    if not date_from or not date_to:
        return ""
    start = date.fromisoformat(str(date_from)[:10])
    end = date.fromisoformat(str(date_to)[:10])
    if start == end:
        return f"ngày {start:%d/%m/%Y}"
    full_months = start.day == 1 and end.day == calendar.monthrange(end.year, end.month)[1]
    if full_months and start.year == end.year:
        if start.month == 1 and end.month == 12:
            return f"năm {start.year}"
        if start.month == end.month:
            return f"tháng {start.month}/{start.year}"
        if start.month % 3 == 1 and end.month == start.month + 2:
            return f"quý {(start.month + 2) // 3}/{start.year}"
        return f"từ tháng {start.month} đến tháng {end.month}/{start.year}"
    if full_months and start.month == 1 and end.month == 12:
        return f"từ năm {start.year} đến năm {end.year}"
    return f"từ {start:%d/%m/%Y} đến {end:%d/%m/%Y}"
