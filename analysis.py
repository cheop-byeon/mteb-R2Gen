import os
import json
import sys
import argparse
from typing import List, Dict
from pathlib import Path

def get_object_by_id(data, search_id):
    """Find an object in a list by its _id field."""
    return next((item for item in data if item['_id'] == search_id), None)

def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels (ground truth relevance labels) from TSV file."""
    qrels = {}
    with open(qrels_path, 'r') as f:
        lines = f.readlines()
        # Skip header if first line contains 'query' or 'score'
        start_idx = 1 if lines and ('query' in lines[0].lower() or 'score' in lines[0].lower()) else 0
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
    return qrels

def analyze_errors(model: str, dataset_name: str = "rfc8205", direction: str = "c2i", split: str = "test", top_k: int = 10):
    """
    Analyze queries with HIT@k = 0 (no relevant document retrieved in top-k).
    
    Args:
        model: Retrieval model name (e.g., "jiebi/RFC-DRAlign-LN")
        dataset_name: Dataset name (e.g., "rfc8205", "rfc7657", "rfc8335")
        direction: "c2i" (code-to-issue) or "i2c" (issue-to-code)
        split: "test" or "train"
        top_k: k value for hit@k metric (default 10)
    """
    
    # Setup paths (relative to root workspace directory)
    model_path = model.replace("/", "__").replace(" ", "_")
    results_path = f'results/stage1/{split}/{direction}/{model_path}/{dataset_name}_default_predictions.json'
    query_path = f'ir/{dataset_name}/{direction}/{split}/queries.jsonl'
    corpus_path = f'ir/{dataset_name}/{direction}/{split}/corpus.jsonl'
    qrels_path = f'ir/{dataset_name}/{direction}/{split}/qrels/{split}.tsv'
    
    # Verify all files exist
    missing_files = []
    for path in [results_path, query_path, corpus_path, qrels_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("Error: Missing files:")
        for path in missing_files:
            print(f"  - {path}")
        return None, None
    
    print(f"Loading data for model: {model}")
    print(f"Dataset: {dataset_name}, Direction: {direction}, Split: {split}, Top-k: {top_k}")
    print()
    
    # Load data
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    with open(query_path, 'r') as f:
        queries = [json.loads(line) for line in f]
    
    with open(corpus_path, 'r') as f:
        corpus = [json.loads(line) for line in f]
    
    # Load qrels from TSV file: qid -> {docid -> relevance_score}
    qrels_dict = load_qrels(qrels_path)
    
    # Identify queries with HIT@k = 0
    error_cases = []
    hit_at_k_count = {f"HIT@{k}": 0 for k in [1, 5, 10, 100]}
    total_queries = 0
    
    for qid, scores in results.items():
        total_queries += 1
        
        # Get top-k documents
        top_k_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_k_doc_ids = [doc_id for doc_id, _ in top_k_docs]
        
        # Get relevant documents for this query
        relevant_docs = set(qrels_dict.get(qid, {}).keys())
        
        # Check if any top-k document is relevant
        hit = 0
        for doc_id in top_k_doc_ids:
            if doc_id in relevant_docs:
                hit = 1
                break
        
        if hit == 0:
            # This is an error case
            query_text = get_object_by_id(queries, qid)
            if query_text is None:
                print(f"Warning: Query {qid} not found in queries.jsonl")
                continue
            
            # Get the top-k documents
            retrieved_docs = []
            for doc_id, score in top_k_docs:
                doc = get_object_by_id(corpus, doc_id)
                if doc:
                    retrieved_docs.append({
                        "doc_id": doc_id,
                        "text": doc.get('text', ''),
                        "score": score
                    })
            
            # Get the actual relevant documents (for reference)
            reference_docs = []
            for doc_id in relevant_docs:
                doc = get_object_by_id(corpus, doc_id)
                if doc:
                    reference_docs.append({
                        "doc_id": doc_id,
                        "text": doc.get('text', '')
                    })
            
            error_cases.append({
                "query_id": qid,
                "query_text": query_text.get('text', ''),
                "retrieved_docs_top_k": retrieved_docs,
                "relevant_docs": reference_docs
            })
        
        # Count hits at different k values (for statistics)
        for k in [1, 5, 10, 100]:
            top_k_docs_check = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            hit_check = 0
            for doc_id, _ in top_k_docs_check:
                if doc_id in relevant_docs:
                    hit_check = 1
                    break
            hit_at_k_count[f"HIT@{k}"] += hit_check
    
    # Print statistics
    print(f"Total queries: {total_queries}")
    print(f"Queries with HIT@{top_k} = 0: {len(error_cases)}")
    print()
    print("Hit Statistics across different k values:")
    for k in [1, 5, 10, 100]:
        hit_rate = round(hit_at_k_count[f"HIT@{k}"] / total_queries, 4)
        print(f"  HIT@{k}: {hit_at_k_count[f'HIT@{k}']} / {total_queries} = {hit_rate}")
    print()
    
    # Save error cases to file
    output_file = f'error_analysis_{model_path}_{dataset_name}_{direction}_{split}.jsonl'
    with open(output_file, 'w') as f:
        for case in error_cases:
            f.write(json.dumps(case) + '\n')
    
    print(f"Saved {len(error_cases)} error cases to: {output_file}")
    
    # Also save a summary report
    summary_file = f'error_analysis_summary_{model_path}_{dataset_name}_{direction}_{split}.json'
    summary = {
        "model": model,
        "dataset": dataset_name,
        "direction": direction,
        "split": split,
        "total_queries": total_queries,
        "error_cases_count": len(error_cases),
        "hit_statistics": {k: v / total_queries for k, v in hit_at_k_count.items()},
        "error_cases": error_cases
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {summary_file}")
    
    return error_cases, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error analysis for retrieval results")
    parser.add_argument("--model", type=str, required=True, help="Retrieval Model (e.g., jiebi/RFC-DRAlign-LN, bm25s)")
    parser.add_argument("--dataset", type=str, default="rfc8205", help="Dataset name (e.g., rfc8205, rfc7657, rfc8335)")
    parser.add_argument("--direction", type=str, default="c2i", choices=["c2i", "i2c"], help="Direction: c2i or i2c")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Split: test or train")
    parser.add_argument("--top_k", type=int, default=10, help="k value for HIT@k metric")
    
    args = parser.parse_args()
    
    error_cases, summary = analyze_errors(args.model, args.dataset, args.direction, args.split, args.top_k)
