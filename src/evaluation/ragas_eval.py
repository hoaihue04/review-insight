"""Đánh giá RAG của trợ lý review bằng Ragas (0.4.x).

Chạy trọn quy trình bằng một lệnh (sinh câu hỏi nếu chưa có → thu thập → chấm đủ metric):

       .venv\\Scripts\\python.exe -X utf8 -m src.evaluation.ragas_eval run

  Mặc định: chatbot dùng LLM, mẫu 60 câu, chấm đủ 9 metric (4 không cần LLM + 5 cần giám khảo).
  Thêm --refresh-lookup để viết lại câu tra cứu trước khi chạy. Xem tùy chọn: ... run --help

Hoặc chạy từng bước:

  1. collect: cho chatbot trả lời bộ câu hỏi, lưu dataset theo trường của Ragas
       .venv\\Scripts\\python.exe -X utf8 -m src.evaluation.ragas_eval collect
  2. score: chấm điểm dataset
       .venv\\Scripts\\python.exe -X utf8 -m src.evaluation.ragas_eval score results\\ragas\\dataset_....jsonl
       ... score <file> --llm-metrics all   # hoặc liệt kê: context_relevance,faithfulness

Metric:
  - Không cần LLM (luôn chạy):
      id_context_precision, id_context_recall: đáp án là các review thỏa điều kiện gold (chủ đề,
        cảm xúc, thời gian), lấy từ database, không cần gán nhãn tay.
      id_context_recall_at_k: như id_context_recall nhưng chia cho min(k, |đáp án|), k = số bằng
        chứng tối đa (RETRIEVAL_FINAL_LIMIT); tính cho mọi câu có đáp án, kể cả đáp án rộng.
      count_accuracy, answer_number_accuracy: số liệu chatbot dùng và con số trong câu trả lời
        có khớp số đếm lại từ đáp án không (xem src/evaluation/count_check.py).
  - Cần LLM giám khảo (tùy chọn): context_relevance, faithfulness, response_groundedness,
    answer_relevancy, lookup_context_recall (chỉ câu tra cứu: bằng chứng có chứa ý của câu
    review gốc không, nên review khác nói cùng ý cũng được tính). Cấu hình bằng biến môi trường:
        RAGAS_JUDGE_PROVIDER  = gemini | groq | openai   (mặc định gemini)
        RAGAS_JUDGE_MODEL     = tên mô hình   (mặc định: gemini-flash-lite-latest với gemini,
                                               gpt-4.1-mini-2025-04-14 với openai)
        RAGAS_EMBEDDING_MODEL = mô hình embedding cho answer_relevancy (mặc định MiniLM;
                                vd. BAAI/bge-m3). Đổi mô hình là đổi thước đo: chấm lại cả
                                dataset cũ bằng mô hình mới trước khi so sánh.
    Mỗi lượt gọi lỗi (429, 5xx...) được ghi lại; metric của câu đó bị đánh dấu lỗi và không
    được tính trung bình, vì Ragas vẫn có thể trả về một con số khi lượt gọi thất bại.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from src.evaluation.rag_filter_eval import (
    ANY,
    DEFAULT_QUESTIONS,
    ROOT,
    _load_truth,
    _prepare_env,
    load_questions,
    opinion_matches,
)
from src.evaluation.count_check import count_metrics, gold_counts

OUT_DIR = ROOT / "results" / "ragas"
FULL_QUESTIONS = ROOT / "data" / "eval" / "ragas_questions_full.jsonl"
CACHE_DIR = ROOT / "cache" / "ragas"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NON_LLM_METRICS = (
    "id_context_precision", "id_context_recall", "id_context_recall_at_k",
    "count_accuracy", "answer_number_accuracy",
)
# Đáp án quá rộng (vd. 369 ý kiến khen) thì recall trên 6 bằng chứng không có ý nghĩa.
MAX_GOLD_FOR_RECALL = 10
# Số bằng chứng tối đa chatbot trả về (RETRIEVAL_FINAL_LIMIT). recall@k chia cho min(k, |đáp án|):
# đáp án 10 review mà chỉ được lấy 6 bằng chứng thì 6/6 đúng là đạt tối đa, không phải 0.6.
RECALL_K = int(os.environ.get("RETRIEVAL_FINAL_LIMIT") or 6)

JUDGE_ENDPOINTS = {
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "openai": ("OPENAI_API_KEY", None),
}
# Giám khảo mặc định theo nhà cung cấp; ghi đè bằng RAGAS_JUDGE_MODEL.
DEFAULT_JUDGE_MODELS = {
    "gemini": "gemini-flash-lite-latest",
    # Bản có ngày cố định để điểm giữa các lần chạy so sánh được; không phải mô hình suy luận nên
    # nhận temperature=0.
    "openai": "gpt-4.1-mini-2025-04-14",
}
LLM_METRICS = ("context_relevance", "faithfulness", "response_groundedness", "answer_relevancy",
               "lookup_context_recall")


def ragas_embedding_model() -> str:
    """Đọc lúc chấm (sau load_dotenv), không đọc lúc import."""
    return os.environ.get("RAGAS_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# ── Bước 1: thu thập ─────────────────────────────────────────────────────────

def _facts_text(facts: dict[str, Any] | None) -> str:
    if not facts:
        return ""
    return "Số liệu đếm từ database (đã lọc theo câu hỏi): " + json.dumps(facts, ensure_ascii=False)


def _gold_review_ids(q, truth) -> list[str] | None:
    """Review có ít nhất một ý kiến thỏa mọi điều kiện gold. None nếu câu hỏi không chấm được."""
    if q.gold.aspects is None or q.gold.sentiment == ANY:
        return None
    ids = {
        str(o["review_id"]) for o in truth["opinions"]
        if not opinion_matches(o["aspect"], o["sentiment"], o.get("day"), q.gold)
    }
    return sorted(ids)


def _meta_path(dataset: Path) -> Path:
    return dataset.with_name(dataset.stem + ".meta.json")


def collect(args: argparse.Namespace) -> Path:
    _prepare_env(args.chat_llm == "on", keep_db_host=False)
    if args.chat_model:
        os.environ["GEMINI_MODEL"] = args.chat_model  # phải đặt trước khi import cấu hình
    if args.chat_llm == "on":
        # Gói miễn phí giới hạn số lượt mỗi phút: chờ và thử lại thay vì rơi về câu soạn sẵn.
        os.environ.setdefault("LLM_RETRY_ON_429", "3")
    from app.assistant.answer_writer import asks_for_advice
    from app.assistant.assistant_service import ParentAssistantService

    questions = load_questions(Path(args.questions))
    # Trường bổ sung của bộ câu hỏi đầy đủ (loại câu hỏi, đáp án ID của câu tra cứu).
    extras = {
        str(raw["id"]): raw
        for raw in (json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip())
    }
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q.id in wanted]
    if args.sample:
        # Cùng cách chọn với lệnh score --sample, để chỉ tốn lượt gọi cho đúng các câu sẽ được chấm.
        picked = {s["id"] for s in stratified_sample(
            [{"id": q.id, "category": q.category, "product_id": q.product_id} for q in questions],
            args.sample, args.seed)}
        questions = [q for q in questions if q.id in picked]
        print(f"Chỉ thu thập {len(questions)} câu theo mẫu (seed {args.seed}).")
    truth = _load_truth({q.product_id for q in questions})
    service = ParentAssistantService()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"dataset_{datetime.now():%Y%m%d-%H%M%S}_chat-{args.chat_llm}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for q in questions:
            os.environ["RAG_REFERENCE_DATE"] = q.reference_date.isoformat()
            try:
                result = service.answer_chat(q.product_id, q.question, use_llm=args.chat_llm == "on") or {}
            except Exception as exc:
                result = {"answer": f"[LỖI] {exc}", "evidence": []}
            evidence = result.get("evidence") or []
            retrieved_ids: list[str] = []
            for row in evidence:
                rid = str(row.get("review_id") or "")
                if rid and rid not in retrieved_ids:
                    retrieved_ids.append(rid)
            day_of_review = truth[q.product_id]["day_of_review"]
            extra = extras.get(q.id, {})
            question_type = extra.get("question_type") or "handwritten"
            reference_ids = (
                # Câu tra cứu: review gốc + các review khác trả lời được câu hỏi (label_lookup_gold).
                list(dict.fromkeys(str(i) for i in [*extra["gold_review_ids"], *(extra.get("gold_review_ids_extra") or [])]))
                if extra.get("gold_review_ids")
                else _gold_review_ids(q, truth[q.product_id])
            )
            sample = {
                "id": q.id,
                "category": q.category,
                "question_type": question_type,
                "product_id": q.product_id,
                "user_input": q.question,
                "response": result.get("answer") or "",
                "retrieved_contexts": [str(row.get("text") or "") for row in evidence],
                "retrieved_context_dates": [
                    (day_of_review.get(str(row.get("review_id") or "")) or "") and
                    day_of_review[str(row.get("review_id"))].strftime("%d/%m/%Y")
                    for row in evidence
                ],
                "retrieved_context_ids": retrieved_ids,
                "reference_context_ids": reference_ids,
                "facts_context": _facts_text(result.get("answer_facts")),
                # Số liệu chatbot dùng và số liệu đếm lại từ đáp án, cho count_accuracy.
                "answer_facts": result.get("answer_facts"),
                "gold_counts": gold_counts(truth[q.product_id]["opinions"], q.gold),
                # Câu review gốc của câu tra cứu, làm đáp án cho lookup_context_recall.
                "reference": extra.get("source_sentence") or "",
                "llm_used": bool(result.get("llm_used")),
                # Đánh dấu để xem riêng khi phân tích kết quả.
                "no_data_answer": not evidence,
                "asks_advice": asks_for_advice(q.question),
            }
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"{q.id}: {len(evidence)} bằng chứng, chatbot dùng LLM={sample['llm_used']}", flush=True)
            if args.chat_llm == "on" and args.delay:
                time.sleep(args.delay)
    from app.assistant.config import settings

    # Ghi cấu hình chatbot cạnh dataset, để bảng điểm biết đã đo với mô hình nào.
    _meta_path(out).write_text(json.dumps({
        "questions": str(args.questions),
        "chat_llm": args.chat_llm,
        "chat_model": settings.gemini_model if args.chat_llm == "on" else "",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "sample": args.sample,
        "seed": args.seed,
        "ids": args.ids,
        "n_questions": len(questions),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    del service
    gc.collect()  # giải phóng mô hình của chatbot trước khi nạp mô hình chấm điểm
    print(f"\nĐã lưu dataset: {out}")
    return out


# ── Bước 2: chấm điểm ────────────────────────────────────────────────────────

def id_metrics(sample: dict[str, Any]) -> dict[str, float | None]:
    from ragas.dataset_schema import SingleTurnSample
    import warnings

    # Ragas 0.4 báo hai metric ID sẽ bị bỏ ở v1.0 (chưa có bản thay thế); vẫn dùng được, nên ẩn
    # cảnh báo để PowerShell không hiện nó như lỗi.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    gold = sample.get("reference_context_ids")
    retrieved = sample.get("retrieved_context_ids") or []
    empty = {"id_context_precision": None, "id_context_recall": None, "id_context_recall_at_k": None}
    if gold is None:
        return empty
    if sample.get("question_type") == "lookup":
        # Chỉ biết chắc review gốc trả lời được câu hỏi; review khác có thể cũng đúng, nên
        # chỉ đo recall (review gốc có nằm trong bằng chứng không), không đo precision.
        # recall: có ít nhất một review trả lời được câu hỏi không; recall@k: bao nhiêu phần
        # bằng chứng là review trả lời được (đáp án mở rộng bằng label_lookup_gold).
        found = len(set(gold) & set(retrieved))
        return {
            **empty,
            "id_context_recall": 1.0 if found else 0.0,
            "id_context_recall_at_k": min(1.0, found / min(RECALL_K, len(gold))) if gold else None,
        }
    if not gold:
        # Không có dữ liệu thỏa điều kiện: đúng là không trả bằng chứng nào.
        return {**empty, "id_context_precision": 1.0 if not retrieved else 0.0}
    found = len(set(gold) & set(retrieved))
    recall_at_k = found / min(RECALL_K, len(gold))
    if not retrieved:
        return {
            "id_context_precision": 0.0,
            "id_context_recall": 0.0 if len(gold) <= MAX_GOLD_FOR_RECALL else None,
            "id_context_recall_at_k": 0.0,
        }
    s = SingleTurnSample(retrieved_context_ids=retrieved, reference_context_ids=gold)
    return {
        "id_context_precision": float(IDBasedContextPrecision().single_turn_score(s)),
        "id_context_recall": (
            float(IDBasedContextRecall().single_turn_score(s)) if len(gold) <= MAX_GOLD_FOR_RECALL else None
        ),
        "id_context_recall_at_k": min(1.0, recall_at_k),
    }


class _ErrorCounter:
    def __init__(self) -> None:
        self.count = 0
        self.last = ""

    async def __call__(self, response) -> None:
        if response.status_code >= 400:
            self.count += 1
            self.last = f"HTTP {response.status_code}"


def _judge_config() -> tuple[str, str, str, str | None]:
    """(provider, model, api_key, base_url) của giám khảo; dừng ngay nếu cấu hình thiếu."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    provider = os.environ.get("RAGAS_JUDGE_PROVIDER", "gemini").lower()
    if provider not in JUDGE_ENDPOINTS:
        raise SystemExit(f"RAGAS_JUDGE_PROVIDER không hợp lệ: {provider}")
    model = os.environ.get("RAGAS_JUDGE_MODEL") or DEFAULT_JUDGE_MODELS.get(provider)
    if not model:
        raise SystemExit(f"Đặt RAGAS_JUDGE_MODEL trong .env cho giám khảo {provider}.")
    key_name, base_url = JUDGE_ENDPOINTS[provider]
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"Thiếu {key_name} trong .env cho giám khảo {provider}.")
    return provider, model, api_key, base_url


def parse_metric_names(text: str) -> list[str]:
    """'all' → mọi metric LLM; nếu không thì danh sách phân cách bằng dấu phẩy."""
    text = (text or "").strip()
    if text.lower() == "all":
        return list(LLM_METRICS)
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in LLM_METRICS]
    if unknown:
        raise SystemExit(f"Metric không hỗ trợ: {', '.join(unknown)}. Chọn trong {', '.join(LLM_METRICS)} hoặc 'all'.")
    return names


def _judge(counter: _ErrorCounter):
    import httpx
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory

    provider, model, api_key, base_url = _judge_config()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,  # tự xử lý chờ giữa các lượt; không để client âm thầm thử lại
        http_client=httpx.AsyncClient(timeout=120, event_hooks={"response": [counter]}),
    )
    # Mỗi giám khảo một thư mục cache, để đổi mô hình không dùng nhầm điểm đã chấm của mô hình khác.
    cache_dir = CACHE_DIR / re.sub(r"[^A-Za-z0-9._-]+", "_", f"{provider}__{model}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Mô hình suy luận của OpenAI (gpt-5*, o*) không nhận temperature/max_tokens.
    is_reasoning = provider == "openai" and model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
    params = {} if is_reasoning else {"max_tokens": 4096, "temperature": 0}
    llm = llm_factory(model, client=client, cache=DiskCacheBackend(str(cache_dir)), **params)
    print(f"Giám khảo: {provider} / {model}  (cache: {cache_dir})")
    return llm


def _build_llm_metrics(names: list[str], llm):
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextRecall,
        ContextRelevance,
        Faithfulness,
        ResponseGroundedness,
    )

    metrics = {}
    for name in names:
        if name == "context_relevance":
            metrics[name] = ContextRelevance(llm=llm)
        elif name == "faithfulness":
            metrics[name] = Faithfulness(llm=llm)
        elif name == "response_groundedness":
            metrics[name] = ResponseGroundedness(llm=llm)
        elif name == "answer_relevancy":
            model = ragas_embedding_model()
            print(f"Embedding cho answer_relevancy: {model}")
            # Batch nhỏ để mô hình lớn (bge-m3) không chiếm quá nhiều RAM.
            metrics[name] = AnswerRelevancy(llm=llm, embeddings=HuggingFaceEmbeddings(model=model, batch_size=8))
        elif name == "lookup_context_recall":
            metrics[name] = ContextRecall(llm=llm, name="lookup_context_recall")
        else:
            raise SystemExit(f"Metric không hỗ trợ: {name}. Chọn trong {LLM_METRICS}")
    return metrics


def _metric_inputs(name: str, sample: dict[str, Any]) -> dict[str, Any] | None:
    # Ghi ngày viết review trước mỗi bằng chứng để giám khảo kiểm tra được điều kiện thời gian.
    dates = sample.get("retrieved_context_dates") or [""] * len(sample["retrieved_contexts"])
    contexts = [f"[Review ngày {d}] {c}" if d else c for c, d in zip(sample["retrieved_contexts"], dates) if c]
    # Câu trả lời dựa trên cả bằng chứng lẫn số liệu SQL, nên Faithfulness/Groundedness được
    # xem cả hai; các metric về tìm kiếm chỉ xem bằng chứng.
    grounded = contexts + ([sample["facts_context"]] if sample.get("facts_context") else [])
    if name == "context_relevance":
        return {"user_input": sample["user_input"], "retrieved_contexts": contexts} if contexts else None
    if name in {"faithfulness", "response_groundedness"} and not grounded:
        return None  # câu trả lời "không có dữ liệu": không có gì để đối chiếu
    if name == "faithfulness":
        return {"user_input": sample["user_input"], "response": sample["response"], "retrieved_contexts": grounded}
    if name == "response_groundedness":
        return {"response": sample["response"], "retrieved_contexts": grounded}
    if name == "answer_relevancy":
        return {"user_input": sample["user_input"], "response": sample["response"]}
    if name == "lookup_context_recall":
        # Chỉ câu tra cứu có câu review gốc. Giám khảo xét bằng chứng có chứa ý của câu gốc
        # không, nên review khác nói cùng ý cũng được tính, khác id_context_recall.
        if sample.get("question_type") != "lookup" or not sample.get("reference") or not contexts:
            return None
        return {"user_input": sample["user_input"], "retrieved_contexts": contexts, "reference": sample["reference"]}
    return None


async def _score_llm(samples, metrics, counter: _ErrorCounter, delay: float) -> None:
    for i, sample in enumerate(samples, 1):
        for name, metric in metrics.items():
            inputs = _metric_inputs(name, sample)
            if inputs is None:
                sample[name] = None
                continue
            before = counter.count
            try:
                result = await metric.ascore(**inputs)
                value = float(result.value) if result.value is not None else None
                error = ""
            except Exception as exc:
                value, error = None, f"{type(exc).__name__}: {str(exc)[:160]}"
            if counter.count > before:
                value, error = None, f"lỗi API ({counter.last}) — không tính điểm"
            sample[name] = value
            if error:
                sample[f"{name}_error"] = error
            shown = "LỖI" if value is None and error else ("–" if value is None else f"{value:.2f}")
            print(f"  [{i}/{len(samples)}] {sample['id']} {name}: {shown} {error}")
            if delay:
                await asyncio.sleep(delay)


def score(args: argparse.Namespace) -> Path:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # lấy khóa API của giám khảo
    if args.dataset == "latest":
        found = sorted(OUT_DIR.glob("dataset_*.jsonl"))
        if not found:
            raise SystemExit("Chưa có dataset nào trong results/ragas. Chạy lệnh collect trước.")
        args.dataset = str(found[-1])
        print(f"Dùng dataset mới nhất: {args.dataset}")
    samples = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        samples = [s for s in samples if s["id"] in wanted]

    for sample in samples:
        sample.update(id_metrics(sample))
        sample.update(count_metrics(sample))

    meta_file = _meta_path(Path(args.dataset))
    config: dict[str, Any] = {
        "chatbot": json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {},
    }
    names = parse_metric_names(args.llm_metrics)
    if names:
        provider, judge_model, _, _ = _judge_config()
        config["judge"] = f"{provider} / {judge_model}"
        if "answer_relevancy" in names:
            config["answer_relevancy_embedding"] = ragas_embedding_model()
        judged = stratified_sample(samples, args.sample, args.seed) if args.sample else samples
        counter = _ErrorCounter()
        metrics = _build_llm_metrics(names, _judge(counter))
        print(f"Chấm {len(judged)}/{len(samples)} câu × {len(names)} metric bằng LLM "
              f"(tuần tự, nghỉ {args.delay}s giữa các lượt)...")
        asyncio.run(_score_llm(judged, metrics, counter, args.delay))

    all_metrics = [*NON_LLM_METRICS, *names]
    summary = {}
    for name in all_metrics:
        values = [s[name] for s in samples if s.get(name) is not None]
        errors = sum(1 for s in samples if s.get(f"{name}_error"))
        summary[name] = {"mean": round(mean(values), 3) if values else None, "n": len(values), "errors": errors}

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    out_json = OUT_DIR / f"ragas_scores_{stamp}.json"
    out_csv = OUT_DIR / f"ragas_scores_{stamp}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"dataset": str(args.dataset), "config": config, "summary": summary,
                                    "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "category", "user_input", *all_metrics])
        for s in samples:
            writer.writerow([s["id"], s["category"], s["user_input"],
                             *["" if s.get(m) is None else round(s[m], 3) for m in all_metrics]])

    print("\nCẤU HÌNH")
    bot = config["chatbot"]
    if bot:
        print(f"  Chatbot: LLM={bot.get('chat_llm')} {bot.get('chat_model') or ''} | embedding={bot.get('embedding_model')}"
              f" | reranker={bot.get('reranker_model')}")
    else:
        print("  Chatbot: không có file .meta.json (dataset thu thập bằng phiên bản cũ)")
    if config.get("judge"):
        print(f"  Giám khảo: {config['judge']}")
    if config.get("answer_relevancy_embedding"):
        print(f"  Embedding chấm answer_relevancy: {config['answer_relevancy_embedding']}")

    print("\nKẾT QUẢ RAGAS")
    print(f"{'Metric':<28} {'Trung bình':>10} {'Số câu':>7} {'Lỗi API':>8}")
    for name, row in summary.items():
        mean_text = "–" if row["mean"] is None else f"{row['mean']:.3f}"
        print(f"{name:<28} {mean_text:>10} {row['n']:>7} {row['errors']:>8}")

    wrong_counts = [s for s in samples if s.get("count_diff")]
    if wrong_counts:
        print("\nSỐ LIỆU LỆCH ĐÁP ÁN (count_accuracy = 0)")
        for s in wrong_counts:
            print(f"  {s['id']} {s['user_input']}\n      {s['count_diff']}")

    print("\nTHEO NHÓM CÂU HỎI (trung bình | số câu)")
    groups = sorted({s.get("category") or "khac" for s in samples})
    print(f"{'Nhóm':<22}" + "".join(f"{m[:15]:>17}" for m in all_metrics))
    by_group = {}
    for group in groups:
        members = [s for s in samples if (s.get("category") or "khac") == group]
        cells = []
        by_group[group] = {}
        for m in all_metrics:
            values = [s[m] for s in members if s.get(m) is not None]
            by_group[group][m] = {"mean": round(mean(values), 3) if values else None, "n": len(values)}
            cells.append("–" if not values else f"{mean(values):.2f} | {len(values)}")
        print(f"{group:<22}" + "".join(f"{c:>17}" for c in cells))
    data = json.loads(out_json.read_text(encoding="utf-8"))
    data["by_category"] = by_group
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu: {out_json}\n        {out_csv}")
    return out_json


def stratified_sample(samples: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Chọn mẫu chia đều theo nhóm câu hỏi, trong mỗi nhóm lần lượt qua các sản phẩm."""
    import random

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        groups.setdefault(s.get("category") or "khac", []).append(s)
    queues = {}
    for name, members in groups.items():
        by_product: dict[str, list[dict[str, Any]]] = {}
        for s in members:
            by_product.setdefault(s["product_id"], []).append(s)
        for lst in by_product.values():
            rng.shuffle(lst)
        order = list(by_product)
        rng.shuffle(order)
        flat = []
        while any(by_product[p] for p in order):  # lần lượt từng sản phẩm
            for p in order:
                if by_product[p]:
                    flat.append(by_product[p].pop())
        queues[name] = flat
    chosen: list[dict[str, Any]] = []
    while len(chosen) < size and any(queues.values()):
        for name in sorted(queues):
            if queues[name] and len(chosen) < size:
                chosen.append(queues[name].pop(0))
    return chosen


def run(args: argparse.Namespace) -> Path:
    """Trọn quy trình: (sinh câu hỏi) → thu thập → chấm đủ metric trên đúng dataset vừa thu thập."""
    names = parse_metric_names(args.llm_metrics)
    if names:
        _judge_config()  # báo thiếu khóa giám khảo trước khi tốn thời gian thu thập

    questions = Path(args.questions)
    if args.rebuild_questions or not questions.exists():
        if questions.resolve() != FULL_QUESTIONS.resolve():
            raise SystemExit(f"Không tìm thấy {questions}. Chỉ sinh tự động được {FULL_QUESTIONS}.")
        from src.evaluation import build_ragas_questions

        print(f"[1/3] Sinh bộ câu hỏi: {questions}")
        build_argv = ["--lookup-llm", str(args.lookup_llm)] if args.lookup_llm else []
        build_ragas_questions.main(build_argv)
    elif args.refresh_lookup:
        from src.evaluation import build_ragas_questions

        # Chỉ viết lại câu tra cứu, giữ nguyên ID và mọi câu khác để so được với lần chạy trước.
        print(f"[1/3] Viết lại câu tra cứu trong {questions}")
        build_ragas_questions.main(["--only-lookup", "--out", str(questions)])
    else:
        print(f"[1/3] Dùng bộ câu hỏi có sẵn: {questions}")

    print("\n[2/3] Thu thập câu trả lời của chatbot")
    dataset = collect(argparse.Namespace(
        questions=str(questions), chat_llm=args.chat_llm, chat_model=args.chat_model,
        delay=args.collect_delay, ids=args.ids, sample=args.sample, seed=args.seed,
    ))

    print(f"\n[3/3] Chấm điểm: {', '.join([*NON_LLM_METRICS, *names])}")
    # Dataset đã được lấy mẫu ở bước thu thập, nên chấm toàn bộ dataset đó.
    return score(argparse.Namespace(
        dataset=str(dataset), llm_metrics=",".join(names), ids="", delay=args.judge_delay, sample=0, seed=args.seed,
    ))


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description="Đánh giá RAG bằng Ragas.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Chạy chatbot và lưu dataset")
    p_collect.add_argument("--questions", default=str(FULL_QUESTIONS if FULL_QUESTIONS.exists() else DEFAULT_QUESTIONS),
                           help="Mặc định: data/eval/ragas_questions_full.jsonl nếu đã sinh, nếu không thì bộ 36 câu")
    p_collect.add_argument("--chat-llm", choices=["off", "on"], default="off",
                           help="Chatbot dùng LLM để viết câu trả lời (tốn hạn mức API)")
    p_collect.add_argument("--chat-model", default="",
                           help="Mô hình Gemini cho chatbot khi thu thập (ghi đè GEMINI_MODEL trong .env)")
    p_collect.add_argument("--delay", type=float, default=4.0,
                           help="Số giây nghỉ giữa các câu khi --chat-llm on (mỗi câu tốn 2 lượt gọi Gemini)")
    p_collect.add_argument("--ids", default="")
    p_collect.add_argument("--sample", type=int, default=0,
                           help="Chỉ thu thập N câu, chọn giống lệnh score --sample (tiết kiệm lượt gọi khi --chat-llm on)")
    p_collect.add_argument("--seed", type=int, default=7)

    p_score = sub.add_parser("score", help="Chấm điểm một dataset đã thu thập")
    p_score.add_argument("dataset", help="Đường dẫn file dataset, hoặc 'latest' để dùng file mới nhất")
    p_score.add_argument("--llm-metrics", default="",
                         help=f"'all' hoặc danh sách phân cách bằng dấu phẩy: {','.join(LLM_METRICS)}")
    p_score.add_argument("--ids", default="")
    p_score.add_argument("--delay", type=float, default=4.0, help="Số giây nghỉ giữa các lượt gọi LLM")
    p_score.add_argument("--sample", type=int, default=0,
                         help="Chỉ cho giám khảo LLM chấm N câu, chọn đều theo nhóm câu hỏi và sản phẩm")
    p_score.add_argument("--seed", type=int, default=7)

    p_run = sub.add_parser("run", help="Chạy trọn quy trình: sinh câu hỏi (nếu chưa có) → thu thập → chấm đủ metric")
    p_run.add_argument("--questions", default=str(FULL_QUESTIONS))
    p_run.add_argument("--rebuild-questions", action="store_true",
                       help="Sinh lại bộ câu hỏi từ database dù file đã có")
    p_run.add_argument("--lookup-llm", type=int, default=0,
                       help="Khi sinh câu hỏi: số câu tra cứu do LLM viết cho mỗi sản phẩm")
    p_run.add_argument("--refresh-lookup", action="store_true",
                       help="Chỉ viết lại các câu tra cứu (giữ chi tiết riêng của review gốc), giữ nguyên câu khác")
    p_run.add_argument("--chat-llm", choices=["off", "on"], default="on")
    p_run.add_argument("--chat-model", default="", help="Mô hình Gemini cho chatbot (ghi đè GEMINI_MODEL)")
    p_run.add_argument("--collect-delay", type=float, default=4.0, help="Số giây nghỉ giữa các câu khi thu thập")
    p_run.add_argument("--sample", type=int, default=60, help="Số câu đánh giá; 0 = toàn bộ bộ câu hỏi")
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--ids", default="", help="Chỉ đánh giá các câu này (phân cách bằng dấu phẩy)")
    p_run.add_argument("--llm-metrics", default="all",
                       help=f"'all' (mặc định), '' để chỉ chấm metric ID, hoặc danh sách: {','.join(LLM_METRICS)}")
    p_run.add_argument("--judge-delay", type=float, default=0.5, help="Số giây nghỉ giữa các lượt gọi giám khảo")

    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.command == "collect":
        collect(args)
    elif args.command == "run":
        run(args)
    else:
        score(args)
    print(f"Thời gian: {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
