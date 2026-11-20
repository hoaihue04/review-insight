"""Mở rộng đáp án của câu tra cứu: thêm mọi review trả lời được câu hỏi, không chỉ review gốc.

Mỗi câu tra cứu được sinh từ một câu review, nên đáp án ban đầu chỉ có một ID; chatbot lấy
được review khác nói cùng ý vẫn bị tính là trượt. Script này:
  1. gom ứng viên từ nhiều nguồn (MiniLM và bge-m3, mỗi mô hình theo câu hỏi và theo câu
     review gốc, cộng trùng âm tiết với câu hỏi) để đáp án không thiên vị mô hình embedding nào
     đang được so sánh;
  2. nhờ LLM giám khảo chọn các câu trả lời được câu hỏi;
  3. ghi vào trường gold_review_ids_extra của file câu hỏi (giữ nguyên các trường khác).

    .venv\\Scripts\\python.exe -X utf8 -m src.evaluation.label_lookup_gold
    .venv\\Scripts\\python.exe -X utf8 -m src.evaluation.label_lookup_gold --ids P09-11,P20-12 --force

Kết quả LLM được lưu đệm trong cache/ragas/lookup_gold, chạy lại không tốn lượt gọi.
Chạy lại script sau khi sinh lại câu tra cứu (build_ragas_questions --only-lookup).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.rag_filter_eval import _load_truth, _prepare_env  # noqa: E402
from src.evaluation.ragas_eval import CACHE_DIR, FULL_QUESTIONS, _judge_config  # noqa: E402

CANDIDATE_MODELS = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-m3")
PER_SOURCE = 25
PROMPT_VERSION = "v1"
PROMPT = """Bạn gán nhãn dữ liệu đánh giá hệ thống hỏi đáp về review sản phẩm trên sàn thương mại điện tử.

Câu hỏi của người mua: "{question}"

Dưới đây là các câu trích từ review của sản phẩm, mỗi câu có số thứ tự.
Chọn MỌI câu chứa thông tin giúp trả lời trực tiếp câu hỏi trên, dù ý kiến là khen hay chê
(ví dụ hỏi "giao hàng có chậm không" thì cả "giao nhanh" lẫn "giao lâu" đều trả lời được).
Không chọn câu chỉ khen/chê chung chung ("sản phẩm tốt", "hài lòng") nếu câu hỏi hỏi về
một khía cạnh cụ thể mà câu đó không nhắc tới.

{sentences}

Chỉ trả về JSON dạng {{"ids": [các số thứ tự được chọn]}}."""


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")
    return re.sub(r"[^a-z0-9\s]+", " ", text)


STOP = set("co khong ko k cua san pham nay thi la va voi cho nhu the nao ra sao gi khach hang nguoi mua minh".split())


def _syllables(text: str) -> set[str]:
    return {s for s in _fold(text).split() if s not in STOP and len(s) > 1}


def candidates(question: str, source: str, opinions: list[dict], models) -> list[dict]:
    """Câu ứng viên (mỗi nguồn PER_SOURCE câu điểm cao nhất), luôn gồm câu review gốc."""
    sentences = [o for o in opinions if str(o.get("sentence") or "").strip()]
    if not sentences:
        return []
    texts = [str(o["sentence"]) for o in sentences]
    sources: list[list[float]] = []
    for model in models:
        emb = model.encode([question, source, *texts], normalize_embeddings=True,
                           batch_size=32, show_progress_bar=False)
        q_vec, s_vec, s_mat = emb[0], emb[1], emb[2:]
        sources += [(s_mat @ q_vec).tolist(), (s_mat @ s_vec).tolist()]
    q_syl = _syllables(question)
    sources.append([len(q_syl & _syllables(t)) / (len(q_syl) or 1) for t in texts])

    picked: dict[str, dict] = {}
    for scores in sources:
        order = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        n = 0
        for i in order:
            key = _fold(texts[i]).strip()
            if key in picked:
                continue
            picked[key] = sentences[i]
            n += 1
            if n >= PER_SOURCE:
                break
    source_key = _fold(source).strip()
    if source_key not in picked:
        match = next((o for o in sentences if _fold(o["sentence"]).strip() == source_key), None)
        if match:
            picked[source_key] = match
    return list(picked.values())


def label(client, judge_model: str, question: str, cands: list[dict]) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {c['sentence']}" for i, c in enumerate(cands))
    prompt = PROMPT.format(question=question, sentences=numbered)
    key = hashlib.sha1(f"{PROMPT_VERSION}|{judge_model}|{prompt}".encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / "lookup_gold" / f"{key}.json"
    if cache_file.exists():
        ids = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        reply = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = json.loads(reply.choices[0].message.content or "{}")
        ids = [int(i) for i in raw.get("ids") or [] if str(i).isdigit()]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(ids), encoding="utf-8")
    out: list[str] = []
    for i in ids:
        if 1 <= i <= len(cands):
            rid = str(cands[i - 1]["review_id"])
            if rid not in out:
                out.append(rid)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(FULL_QUESTIONS))
    parser.add_argument("--ids", default="", help="Chỉ gán nhãn các câu này (phân cách bằng dấu phẩy)")
    parser.add_argument("--force", action="store_true", help="Gán lại cả câu đã có gold_review_ids_extra")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    _prepare_env(False, keep_db_host=False)
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    _, judge_model, api_key, base_url = _judge_config()
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    path = Path(args.questions)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
    todo = [
        r for r in rows
        if r.get("question_type") == "lookup"
        and (not wanted or r["id"] in wanted)
        and (args.force or "gold_review_ids_extra" not in r)
    ]
    if not todo:
        print("Không có câu tra cứu nào cần gán nhãn.")
        return 0
    truth = _load_truth({r["product_id"] for r in todo})
    from app.assistant.reranker import resolve_device

    device = resolve_device("auto")
    models = [SentenceTransformer(name, device=device) for name in CANDIDATE_MODELS]

    for r in todo:
        # Mọi câu ý kiến (chưa gộp theo chủ đề), vì chatbot có thể trích bất kỳ câu nào.
        opinions: list[dict[str, Any]] = truth[r["product_id"]]["raw_rows"]
        cands = candidates(r["question"], r.get("source_sentence") or "", opinions, models)
        extra = label(client, judge_model, r["question"], cands)
        original = [str(i) for i in r.get("gold_review_ids") or []]
        r["gold_review_ids_extra"] = [i for i in extra if i not in original]
        r["gold_label_info"] = {"judge": judge_model, "prompt": PROMPT_VERSION, "candidates": len(cands)}
        print(f"{r['id']}: {len(cands)} ứng viên -> {len(extra)} review trả lời được "
              f"(review gốc {'được' if set(original) & set(extra) else 'KHÔNG được'} chọn) | {r['question']}")

    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"Đã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
