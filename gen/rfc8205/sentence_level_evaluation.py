import os
import json
from pathlib import Path
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import argparse

from statistics import mean, median
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory  # kept for compatibility; not used by metrics below
from ragas.metrics.collections import Faithfulness, ContextRecall
from ragas.cache import DiskCacheBackend

from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# Output / cache
# -------------------------
OUT_DIR = Path("eval_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSONL_FILE = OUT_DIR / "ragas_eval_results.jsonl"   # per-QUERY aggregated results
SUMMARY_JSON = OUT_DIR / "overall_summary.json"      # cross-query summary

cache = DiskCacheBackend()
OPENROUTER_API_KEY = "your-api-key-here"

MODEL = "meta-llama/llama-3.3-70b-instruct"
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Judge LLM (no cache to avoid pickling issues with some SDK objects)
judge_llm_strict = llm_factory(MODEL, client=openai_client, cache=None, max_tokens=120000)

# -------------------------
# Metrics: ONLY ContextRecall & Faithfulness
# -------------------------
METRIC_ARG_MAP: Dict[str, List[str]] = {
    "Faithfulness":  ["user_input", "response",  "retrieved_contexts"],
    "ContextRecall": ["user_input", "reference", "retrieved_contexts"],
}

HEAVY_METRICS = [
    Faithfulness(llm=judge_llm_strict),
    ContextRecall(llm=judge_llm_strict),
]

# Retrieved-context chunking stays the same
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=200)

# Async controls
MAX_WORKERS = 8
sem = asyncio.Semaphore(MAX_WORKERS)

# Retry ONLY transient I/O (e.g., timeouts)
RETRYABLE_EXC = (asyncio.TimeoutError,)

# -------------------------
# Sentence split utility (reference/response)
# -------------------------
def split_text_into_sentences(text: str) -> List[str]:
    """
    Try LlamaIndex SentenceWindowNodeParser; fall back to regex if not available.
    """
    try:
        from llama_index.core.node_parser import SentenceWindowNodeParser
        from llama_index.core import Document

        parser = SentenceWindowNodeParser.from_defaults(
            window_size=0,  # atomic sentence
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = parser.build_window_nodes_from_documents([Document(text=text)])
        sents = [n.get_content().strip() for n in nodes if n.get_content().strip()]
        if sents:
            return sents
    except Exception:
        pass

    # Fallback: naive sentence split
    import re
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]

# -------------------------
# Chunk normalizer (to track IDs)
# -------------------------
def normalize_chunks(chunks: List[Any], qid: str) -> List[Dict[str, str]]:
    """
    Accept either:
      - List[str]  -> convert to [{"id": f"{qid}-chunk-{i}", "text": s}, ...]
      - List[dict] -> must contain keys {"id", "text"}; returned as-is
    """
    if not chunks:
        return []
    if isinstance(chunks[0], dict) and "id" in chunks[0] and "text" in chunks[0]:
        return chunks
    return [{"id": f"{qid}-chunk-{i}", "text": c} for i, c in enumerate(chunks)]

# -------------------------
# Metric runner
# -------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=45),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(RETRYABLE_EXC))
async def run_metric(metric, sample: Dict[str, Any]) -> Dict[str, Any]:
    async with sem:
        name = metric.__class__.__name__
        needed = METRIC_ARG_MAP.get(name)
        if not needed:
            raise ValueError(f"Unsupported metric: {name}")
        kwargs = {k: sample[k] for k in needed}
        res = await metric.ascore(**kwargs)
        payload = {"metric": name, "value": float(getattr(res, "value", res))}
        if hasattr(res, "reason") and res.reason is not None:
            payload["reason"] = res.reason
        return payload

async def eval_one_sample(sample: Dict[str, Any], metrics: List[Any]) -> Dict[str, Any]:
    tasks = [run_metric(m, sample) for m in metrics]
    metric_results = await asyncio.gather(*tasks, return_exceptions=True)

    out: Dict[str, Any] = {"metrics": {}, "errors": []}
    for r in metric_results:
        if isinstance(r, Exception):
            out["errors"].append(str(r))
        else:
            out["metrics"][r["metric"]] = {k: v for k, v in r.items() if k != "metric"}
    return out

def append_jsonl(record: Dict[str, Any], path: Path = JSONL_FILE) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# -------------------------
# Utilities for your dataset
# -------------------------
def get_object_by_id(data: List[Dict[str, Any]], search_id: str) -> Optional[Dict[str, Any]]:
    return next((item for item in data if item.get("_id") == search_id), None)

def select_top_k_docs(results: Dict[str, Dict[str, float]],
                      emails: List[Dict[str, Any]],
                      top_k: int = 3) -> Dict[str, List[str]]:
    top_k_dict: Dict[str, List[str]] = {}
    for qid, scores in results.items():
        top_k_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        docs = []
        for docid, score in top_k_docs:
            doc = get_object_by_id(emails, docid)
            if doc is None:
                continue
            docs.append(doc.get("text", ""))
        if len(docs) == top_k:
            top_k_dict[qid] = docs
    return top_k_dict

# -------------------------
# NEW: Evaluate one QUERY
# - split REFERENCE for ContextRecall
# - split RESPONSE for Faithfulness
# - take per-sentence max over chunks
# - average per-sentence maxima
# - track chunk length stats (avg/max/min/median) per query
# -------------------------
async def evaluate_query_by_sentence(
    qid: str,
    user_input_text: str,        # required by RAGAS signatures (not used in calc)
    reference_text: str,         # split → ContextRecall
    response_text: str,          # split → Faithfulness
    chunks: List[Any],           # str or [{"id","text"}]
    write_jsonl: bool = True,
) -> Dict[str, Any]:

    chunk_items = normalize_chunks(chunks, qid)
    chunk_lengths = [len(ci["text"]) for ci in chunk_items]
    chunk_stats = {
        "avg": mean(chunk_lengths) if chunk_lengths else 0.0,
        "max": max(chunk_lengths)  if chunk_lengths else 0,
        "min": min(chunk_lengths)  if chunk_lengths else 0,
        "median": median(chunk_lengths) if chunk_lengths else 0.0,
        "count": len(chunk_lengths),
    }

    # --- ContextRecall over REFERENCE sentences ---
    ref_sentences = split_text_into_sentences(reference_text)
    cr_per_sentence: List[float] = []
    cr_best_ids: List[Optional[str]] = []

    for sent in ref_sentences:
        per_chunk_tasks = []
        for ci in chunk_items:
            sample = {
                "user_input": user_input_text,
                "reference": sent,
                "retrieved_contexts": [ci["text"]],
                "response": sent,  # present but unused for CR
            }
            per_chunk_tasks.append(eval_one_sample(sample, [HEAVY_METRICS[1]]))  # ContextRecall

        results = await asyncio.gather(*per_chunk_tasks, return_exceptions=True)

        pairs: List[Tuple[float, Optional[str]]] = []
        for ci, r in zip(chunk_items, results):
            if isinstance(r, Exception):
                continue
            val = r["metrics"].get("ContextRecall", {}).get("value")
            if isinstance(val, (int, float)):
                pairs.append((float(val), ci["id"]))

        if pairs:
            max_val, max_id = max(pairs, key=lambda x: x[0])
        else:
            max_val, max_id = 0.0, None

        cr_per_sentence.append(max_val)
        cr_best_ids.append(max_id)

    cr_mean = (sum(cr_per_sentence) / len(ref_sentences)) if ref_sentences else 0.0

    # --- Faithfulness over RESPONSE sentences ---
    resp_sentences = split_text_into_sentences(response_text)
    f_per_sentence: List[float] = []
    f_best_ids: List[Optional[str]] = []

    for sent in resp_sentences:
        per_chunk_tasks = []
        for ci in chunk_items:
            sample = {
                "user_input": user_input_text,
                "response": sent,
                "retrieved_contexts": [ci["text"]],
                "reference": sent,  # present but unused for F
            }
            per_chunk_tasks.append(eval_one_sample(sample, [HEAVY_METRICS[0]]))  # Faithfulness

        results = await asyncio.gather(*per_chunk_tasks, return_exceptions=True)

        pairs: List[Tuple[float, Optional[str]]] = []
        for ci, r in zip(chunk_items, results):
            if isinstance(r, Exception):
                continue
            val = r["metrics"].get("Faithfulness", {}).get("value")
            if isinstance(val, (int, float)):
                pairs.append((float(val), ci["id"]))

        if pairs:
            max_val, max_id = max(pairs, key=lambda x: x[0])
        else:
            max_val, max_id = 0.0, None

        f_per_sentence.append(max_val)
        f_best_ids.append(max_id)

    f_mean = (sum(f_per_sentence) / len(resp_sentences)) if resp_sentences else 0.0

    record = {
        "qid": qid,
        "sentence_counts": {
            "reference_sentences": len(ref_sentences),
            "response_sentences":  len(resp_sentences),
        },
        "metrics_per_sentence": {
            "ContextRecall_max_per_reference_sentence": cr_per_sentence,
            "ContextRecall_best_chunk_id_per_reference_sentence": cr_best_ids,
            "Faithfulness_max_per_response_sentence":  f_per_sentence,
            "Faithfulness_best_chunk_id_per_response_sentence":  f_best_ids,
        },
        "aggregates": {
            "ContextRecall_mean_over_reference_sentences": cr_mean,
            "Faithfulness_mean_over_response_sentences":  f_mean,
        },
        "chunk_inventory": [ci["id"] for ci in chunk_items],
        "chunk_length_stats": chunk_stats,
    }
    if write_jsonl:
        append_jsonl(record, JSONL_FILE)
    return record

# -------------------------
# Main orchestration
# - computes per-query results
# - builds cross-query macro averages
# -------------------------
async def main():
    parser = argparse.ArgumentParser(description="Sentence-level evaluation using RAGAS")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., qwen_non, llama_non)")
    args = parser.parse_args()
    
    model_name = args.model
    results_path   = f"../results/stage1/test/c2i/rfc8205_{model_name}_default_predictions.json"
    query_path     = "./c2i/test/queries.jsonl"
    email_path     = "./c2i/test/corpus.386.no.head.jsonl"
    response_path  = f"./qa_data_{model_name}.jsonl"
    reference_path = "./c2i/test/corpus.46.jsonl"

    top_k = 3

    with open(results_path, "r") as file:
        results = json.load(file)
    with open(query_path, "r") as file:
        queries = [json.loads(line) for line in file]
    with open(email_path, "r") as file:
        emails = [json.loads(line) for line in file]
    with open(reference_path, "r") as file:
        references = [json.loads(line) for line in file]
    with open(response_path, "r") as file:
        responses = [json.loads(line) for line in file]

    topk_dict = select_top_k_docs(results, emails, top_k=top_k)

    # Cross-query accumulators (macro averages across queries)
    cr_means: List[float] = []
    f_means:  List[float] = []
    chunk_avg_lengths: List[float] = []

    # Iterate (remove [:N] when ready)
    for query, response, reference in zip(queries, responses, references):
        assert query["_id"] == response["query"], "Query/response ID mismatch"

        # Build retrieved context chunks as before (UNCHANGED)
        joined = " ".join(topk_dict.get(query["_id"], []))
        chunks = text_splitter.split_text(joined)
        chunk_items = [{"id": f"{query['_id']}-chunk-{i}", "text": c} for i, c in enumerate(chunks)]

        result = await evaluate_query_by_sentence(
            qid=query["_id"],
            user_input_text=query["text"],      # required by metric signatures
            reference_text=reference["text"],   # split → ContextRecall
            response_text=response["reply"],    # split → Faithfulness
            chunks=chunk_items,
            write_jsonl=True
        )

        # Collect per-query aggregates for cross-query summary
        cr_means.append(result["aggregates"]["ContextRecall_mean_over_reference_sentences"])
        f_means.append(result["aggregates"]["Faithfulness_mean_over_response_sentences"])
        chunk_avg_lengths.append(result["chunk_length_stats"]["avg"])

    # Macro summary across all processed queries
    overall_summary = {
        "processed_queries": len(cr_means),
        "macro_averages": {
            "ContextRecall_mean_over_queries": mean(cr_means) if cr_means else 0.0,
            "Faithfulness_mean_over_queries":  mean(f_means)  if f_means  else 0.0,
            "ChunkLength_average_over_queries": mean(chunk_avg_lengths) if chunk_avg_lengths else 0.0,
        }
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(overall_summary, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
