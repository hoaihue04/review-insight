"""Sinh bộ câu hỏi đánh giá RAG cho mọi sản phẩm đã phân tích, đáp án lấy từ database.

Mỗi sản phẩm có khoảng 10 loại câu hỏi, chọn theo số liệu thật của chính sản phẩm đó
(chủ đề bị chê nhiều nhất, năm có nhiều lời chê nhất, tháng không có dữ liệu...). Mỗi loại có
nhiều cách hỏi, trong đó có cách hỏi mà luật hiện tại có thể chưa nhận ra, để bộ đo phản ánh
câu hỏi thật của người dùng chứ không chỉ khớp với luật.

    .venv\\Scripts\\python.exe -X utf8 -m src.evaluation.build_ragas_questions
    ... --lookup-llm 2     # thêm 2 câu tra cứu/sản phẩm do LLM viết từ một câu review (tốn API)
    ... --only-lookup      # chỉ viết lại câu tra cứu trong file có sẵn, giữ nguyên ID và câu khác

Câu tra cứu phải hỏi về một chi tiết riêng của review gốc (âm tiết hiếm trong sản phẩm, LLM chép
nguyên văn chi tiết đó); câu chỉ có nhận xét chung như "giao nhanh, đóng gói kỹ" bị bỏ, vì nhiều
review khác nói y hệt nên không thể là đáp án duy nhất.

Kết quả: data/eval/ragas_questions_full.jsonl (cùng định dạng rag_filter_questions.jsonl, thêm
question_type và, với câu tra cứu, gold_review_ids).
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from src.evaluation.rag_filter_eval import DEFAULT_QUESTIONS, ROOT, _load_truth, _prepare_env

OUT_PATH = ROOT / "data" / "eval" / "ragas_questions_full.jsonl"

# Cách gọi tự nhiên của từng chủ đề trong câu hỏi.
ASPECT_PHRASES = {
    "PRODUCT#QUALITY": ["chất lượng", "chất lượng sản phẩm"],
    "PRODUCT#MATERIAL": ["chất liệu"],
    "PRODUCT#COMFORT": ["cảm giác sử dụng"],
    "PRODUCT#SAFETY": ["độ an toàn", "an toàn cho bé"],
    "PRODUCT#FUNCTION": ["công dụng", "tính năng"],
    "PRODUCT#SIZE": ["kích thước", "size"],
    "PRODUCT#DURABILITY": ["độ bền"],
    "PRODUCT#DESIGN": ["thiết kế", "mẫu mã"],
    "PRODUCT#VALUE": ["độ đáng tiền"],
    "PRICE#AFFORDABILITY": ["giá", "giá cả"],
    "PRICE#DISCOUNT": ["khuyến mãi", "ưu đãi"],
    "DELIVERY#SPEED": ["tốc độ giao hàng", "thời gian giao hàng"],
    "DELIVERY#PACKAGING": ["đóng gói", "bao bì"],
    "DELIVERY#ACCURACY": ["giao thiếu hoặc giao nhầm"],
    "SELLER#SERVICE": ["dịch vụ của shop", "thái độ của shop"],
    "SELLER#RESPONSIVENESS": ["hỗ trợ của shop"],
    "SELLER#AUTHENTICITY": ["hàng chính hãng", "nguồn gốc sản phẩm"],
}

TEMPLATES = {
    "neg_aspect": ["Khách chê gì về {a}?", "Khách phàn nàn những gì về {a}?", "Điểm chưa tốt về {a} mà khách nhắc là gì?"],
    "pos_aspect": ["Khách khen gì về {a}?", "Khách hài lòng điều gì về {a}?", "Ưu điểm về {a} theo khách là gì?"],
    "count_neg": ["Có bao nhiêu ý kiến tiêu cực về {a}?", "Bao nhiêu ý kiến chê {a}?"],
    "aspect_any": ["Khách nói gì về {a}?", "Khách nhận xét thế nào về {a}?", "{A} được khách đánh giá ra sao?"],
    "year_neg": ["Năm {y} khách chê gì?", "Trong năm {y} khách phàn nàn những gì?"],
    "month_aspect": ["Tháng {m}/{y} khách nói gì về {a}?", "Khách nhận xét gì về {a} trong tháng {m} năm {y}?"],
    "fuzzy": ["{f} khách chê gì?", "{F} có những lời chê nào?"],
    "fuzzy_any": ["{f} khách nói gì về sản phẩm?"],
    "relative_pos": ["Năm ngoái khách khen gì?", "Năm ngoái khách hài lòng điều gì?"],
    "no_data": ["Tháng {m}/{y} khách chê gì về {a}?", "Trong tháng {m} năm {y} có ai phàn nàn về {a} không?"],
    "overview": ["Tóm tắt review của sản phẩm này", "Khách nhận xét chung về sản phẩm thế nào?"],
}
CATEGORY = {
    "neg_aspect": "khong_thoi_gian", "pos_aspect": "khong_thoi_gian", "count_neg": "khong_thoi_gian",
    "aspect_any": "khong_thoi_gian", "year_neg": "thoi_gian_ro", "month_aspect": "thoi_gian_ro",
    "fuzzy": "thoi_gian_mo_ho", "fuzzy_any": "thoi_gian_mo_ho", "relative_pos": "thoi_gian_tuong_doi",
    "no_data": "khong_du_lieu", "overview": "tong_quat", "lookup": "tra_cuu",
}
FUZZY_PARTS = {"đầu năm": (1, 3), "giữa năm": (5, 8), "cuối năm": (10, 12)}


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _gold(aspects, sentiment, d_from=None, d_to=None, match="exact") -> dict[str, Any]:
    return {"aspects": aspects, "sentiment": sentiment,
            "date_from": d_from.isoformat() if d_from else None,
            "date_to": d_to.isoformat() if d_to else None, "date_match": match}


def questions_for_product(pid: str, opinions: list[dict], rng: random.Random) -> list[dict]:
    by_aspect: dict[str, Counter] = defaultdict(Counter)
    for o in opinions:
        by_aspect[o["aspect"]][o["sentiment"]] += 1
    dated = [o for o in opinions if o.get("day")]
    if not by_aspect or not dated:
        return []

    def phrase(aspect: str) -> str:
        return rng.choice(ASPECT_PHRASES.get(aspect, [aspect]))

    def ask(kind: str, **kw) -> str:
        text = rng.choice(TEMPLATES[kind]).format(**kw)
        return text[0].upper() + text[1:]

    out: list[dict] = []

    def add(kind: str, question: str, gold: dict, **extra) -> None:
        out.append({"product_id": pid, "question_type": kind, "category": CATEGORY[kind],
                    "question": question, "gold": gold, **extra})

    ranked_neg = sorted(by_aspect, key=lambda a: -by_aspect[a]["negative"])
    ranked_pos = sorted(by_aspect, key=lambda a: -by_aspect[a]["positive"])
    ranked_all = sorted(by_aspect, key=lambda a: -sum(by_aspect[a].values()))

    if by_aspect[ranked_neg[0]]["negative"] >= 1:
        a = ranked_neg[0]
        add("neg_aspect", ask("neg_aspect", a=phrase(a)), _gold([a], "negative"))
    pos = [a for a in ranked_pos if by_aspect[a]["positive"] >= 2]
    if pos:
        a = pos[1] if len(pos) > 1 and rng.random() < 0.5 else pos[0]
        add("pos_aspect", ask("pos_aspect", a=phrase(a)), _gold([a], "positive"))
    neg = [a for a in ranked_neg if by_aspect[a]["negative"] >= 1]
    if neg:
        a = neg[1] if len(neg) > 1 else neg[0]
        add("count_neg", ask("count_neg", a=phrase(a)), _gold([a], "negative"))
    if len(ranked_all) > 2:
        a = rng.choice(ranked_all[1:4])
        p = phrase(a)
        add("aspect_any", ask("aspect_any", a=p, A=p[0].upper() + p[1:]), _gold([a], None))

    neg_years = Counter(o["day"].year for o in dated if o["sentiment"] == "negative")
    if neg_years:
        y = neg_years.most_common(1)[0][0]
        add("year_neg", ask("year_neg", y=y), _gold([], "negative", date(y, 1, 1), date(y, 12, 31)))

    top = ranked_all[0]
    months = Counter((o["day"].year, o["day"].month) for o in dated if o["aspect"] == top)
    if months:
        (y, m), _ = rng.choice(months.most_common(5))
        add("month_aspect", ask("month_aspect", m=m, y=y, a=phrase(top)),
            _gold([top], None, date(y, m, 1), _month_end(y, m)))

    years = sorted({o["day"].year for o in dated})
    part, (m1, m2) = rng.choice(list(FUZZY_PARTS.items()))
    fuzzy_options = [
        (y, sum(1 for o in dated if o["day"].year == y and m1 <= o["day"].month <= m2 and o["sentiment"] == "negative"))
        for y in years
    ]
    fuzzy_options = [(y, n) for y, n in fuzzy_options if n]
    if fuzzy_options:
        y = max(fuzzy_options, key=lambda t: t[1])[0]
        f = f"{part} {y}"
        add("fuzzy", ask("fuzzy", f=f[0].upper() + f[1:], F=f[0].upper() + f[1:]),
            _gold([], "negative", date(y, m1, 1), _month_end(y, m2), "overlap"))
    else:
        y = years[-1]
        f = f"cuối năm {y}"
        add("fuzzy_any", ask("fuzzy_any", f=f[0].upper() + f[1:]),
            _gold([], None, date(y, 10, 1), date(y, 12, 31), "overlap"))

    pos_years = sorted({o["day"].year for o in dated if o["sentiment"] == "positive"})
    if pos_years:
        y = rng.choice(pos_years)
        add("relative_pos", ask("relative_pos"), _gold([], "positive", date(y, 1, 1), date(y, 12, 31)),
            reference_date=date(y + 1, 6, 15).isoformat())

    first, last = min(o["day"] for o in dated), max(o["day"] for o in dated)
    a = ranked_neg[0]
    busy = {(o["day"].year, o["day"].month) for o in dated if o["aspect"] == a and o["sentiment"] == "negative"}
    empty = []
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        if (y, m) not in busy:
            empty.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if empty:
        y, m = rng.choice(empty)
        add("no_data", ask("no_data", m=m, y=y, a=phrase(a)), _gold([a], "negative", date(y, m, 1), _month_end(y, m)))

    add("overview", ask("overview"), _gold([], None))
    return out


LOOKUP_PROMPT = """Câu dưới đây trích từ một review sản phẩm "{name}":
"{sentence}"

Bước 1. Tìm trong câu một CHI TIẾT RIÊNG mà ít review khác có: một sự việc, con số, bộ phận,
so sánh với lần mua trước, quà tặng, cách dùng cụ thể... Không tính các nhận xét chung như "giao
hàng nhanh", "đóng gói kỹ", "chất lượng tốt", "giá rẻ", "hài lòng", "sẽ ủng hộ". Chép chi tiết
đó NGUYÊN VĂN từ câu (vài chữ). Nếu câu chỉ có nhận xét chung thì để chi_tiet là "".

Bước 2. Nếu có chi tiết, viết MỘT câu hỏi tiếng Việt mà nhân viên shop có thể hỏi trợ lý phân
tích review về đúng chi tiết đó, sao cho câu review trên trả lời được. Câu hỏi phải nhắc tới
chi tiết (có thể dùng từ đồng nghĩa), không hỏi chung chung kiểu "giao hàng ra sao", không nhắc
tên khách, không quá 18 từ, kết thúc bằng dấu hỏi.

Trả về JSON: {{"chi_tiet": "...", "question": "..."}}"""

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def _nfc(text: str) -> str:
    # Review và câu LLM trả về có thể mã hóa dấu khác nhau (dựng sẵn / tổ hợp).
    return unicodedata.normalize("NFC", text or "")


def _syllables(text: str) -> set[str]:
    return {w for w in WORD_PATTERN.findall(_nfc(text).lower()) if len(w) >= 2}


def rare_syllables(opinions: list[dict]) -> set[str]:
    """Âm tiết xuất hiện trong rất ít review của sản phẩm (dấu hiệu của chi tiết riêng)."""
    by_review: dict[str, set[str]] = defaultdict(set)
    for o in opinions:
        by_review[str(o["review_id"])] |= _syllables(o.get("sentence") or "")
    df = Counter(w for words in by_review.values() for w in words)
    limit = max(2, len(by_review) // 100)  # ≤ 1% số review, tối thiểu 2
    return {w for w, n in df.items() if n <= limit}


def is_specific_detail(detail: str, sentence: str, rare: set[str]) -> bool:
    """Chi tiết LLM chọn phải nằm trong câu gốc và chứa ít nhất một âm tiết hiếm."""
    from app.assistant.time_parser import fold

    detail = _nfc(detail).strip()
    if not detail or fold(detail) not in fold(_nfc(sentence)):
        return False
    return bool(_syllables(detail) & rare)


def lookup_questions(pid: str, name: str, opinions: list[dict], per_product: int, seed: int, llm) -> list[dict]:
    from app.assistant.answer_writer import is_informative

    # Bộ sinh số riêng cho câu tra cứu, để thêm/bớt câu tra cứu không làm đổi các câu hỏi khác.
    rng = random.Random(f"{seed}-lookup-{pid}")
    rare = rare_syllables(opinions)
    # Chỉ gửi LLM những câu có âm tiết hiếm: câu toàn nhận xét chung ("giao nhanh, đóng gói kỹ")
    # có hàng chục review khác nói y hệt, không làm đáp án duy nhất được.
    pool = [
        o for o in opinions
        if is_informative(o.get("sentence") or "") and len(o["sentence"]) <= 220
        and _syllables(o["sentence"]) & rare
    ]
    rng.shuffle(pool)
    out, seen, calls = [], set(), 0
    for o in pool:
        if len(out) >= per_product or calls >= per_product * 4:  # giới hạn số lượt gọi LLM
            break
        if o["review_id"] in seen:
            continue
        seen.add(o["review_id"])
        calls += 1
        try:
            data = llm.generate_json("Chỉ trả về JSON hợp lệ.", LOOKUP_PROMPT.format(name=name, sentence=o["sentence"]))
        except Exception as exc:
            print(f"  [LLM lỗi] {exc}")
            break
        detail = str(data.get("chi_tiet") or "")
        question = str(data.get("question") or "").strip()
        if not is_specific_detail(detail, o["sentence"], rare):
            continue
        if not question.endswith("?") or len(question.split()) > 22:
            continue
        out.append({"product_id": pid, "question_type": "lookup", "category": CATEGORY["lookup"],
                    "question": question, "gold": _gold(None, "*"),
                    "gold_review_ids": [str(o["review_id"])], "source_sentence": o["sentence"],
                    "detail": detail.strip()})
    return out


def refresh_lookup(path: Path, seed: int, llm, products: dict[str, str], truth) -> int:
    """Viết lại các câu tra cứu trong file có sẵn, giữ nguyên ID và mọi câu khác."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    slots: dict[str, list[int]] = defaultdict(list)  # sản phẩm → vị trí các câu tra cứu cũ
    for i, row in enumerate(rows):
        if row.get("question_type") == "lookup":
            slots[str(row["product_id"])].append(i)
    if not slots:
        print(f"{path} chưa có câu tra cứu nào. Sinh lại bằng --lookup-llm 2.")
        return 0
    replaced = 0
    for pid, positions in slots.items():
        if pid not in products:
            print(f"{pid}: không còn trong database, giữ câu cũ")
            continue
        new = lookup_questions(pid, products[pid], truth[pid]["opinions"], len(positions), seed, llm)
        for pos, q in zip(positions, new):
            q["id"] = rows[pos]["id"]
            rows[pos] = q
            replaced += 1
        if len(new) < len(positions):
            print(f"{pid}: chỉ viết được {len(new)}/{len(positions)} câu, giữ câu cũ cho phần còn lại")
        for q in new:
            print(f"  {q['id']}: {q['question']}  [chi tiết: {q['detail']}]")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"\nĐã thay {replaced} câu tra cứu trong {path}")
    return replaced


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lookup-llm", type=int, default=0, help="Số câu tra cứu do LLM viết cho mỗi sản phẩm")
    # gemini-flash-lite-latest hay hết hạn mức 500 lượt/ngày; 3.1-flash-lite có hạn mức riêng.
    parser.add_argument("--lookup-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--include-handwritten", action="store_true", default=True,
                        help="Gộp cả 36 câu viết tay trong rag_filter_questions.jsonl")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--only-lookup", action="store_true",
                        help="Chỉ viết lại câu tra cứu trong file --out có sẵn, giữ nguyên ID và câu khác")
    args = parser.parse_args(argv)

    if args.only_lookup and not Path(args.out).exists():
        raise SystemExit(f"Không có {args.out} để viết lại câu tra cứu. Sinh bộ câu hỏi trước.")
    use_llm = args.lookup_llm > 0 or args.only_lookup
    _prepare_env(use_llm, keep_db_host=False)
    if use_llm:
        # gemini-2.5-flash (mặc định của chatbot) chỉ có 20 lượt/ngày ở gói miễn phí.
        os.environ["GEMINI_MODEL"] = args.lookup_model
    from sqlalchemy import text

    from app.assistant.db import get_engine

    with get_engine().connect() as conn:
        products = [(str(r[0]), str(r[1])) for r in conn.execute(
            text("SELECT tiki_product_id, name FROM products ORDER BY id"))]
    truth = _load_truth({pid for pid, _ in products})
    llm = None
    if use_llm:
        from app.assistant.llm import ChatLLMClient
        llm = ChatLLMClient()
    if args.only_lookup:
        refresh_lookup(Path(args.out), args.seed, llm, dict(products), truth)
        return 0

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for index, (pid, name) in enumerate(products, 1):
        opinions = truth[pid]["opinions"]
        generated = questions_for_product(pid, opinions, rng)
        if llm:
            generated += lookup_questions(pid, name, opinions, args.lookup_llm, args.seed, llm)
        for j, q in enumerate(generated, 1):
            q["id"] = f"P{index:02d}-{j:02d}"
        rows += generated
        print(f"{pid}: {len(generated)} câu ({len(opinions)} ý kiến) — {name[:60]}")

    if args.include_handwritten:
        for line in Path(DEFAULT_QUESTIONS).read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                q.setdefault("question_type", "handwritten")
                rows.append(q)

    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    kinds = Counter(r.get("question_type") for r in rows)
    print(f"\nĐã lưu {len(rows)} câu hỏi vào {args.out}")
    print("Theo loại:", dict(kinds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
