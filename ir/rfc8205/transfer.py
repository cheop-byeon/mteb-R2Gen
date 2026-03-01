import os
import json
import argparse
import sys
from pathlib import Path

def generate_corpus(dev_or_test_file: str, split: str):
    """Extract unique corpus entries from input file."""
    split_path = Path(split)
    split_path.mkdir(parents=True, exist_ok=True)
    
    corpus_save_path = split_path / 'corpus.jsonl'
    
    corpus_dict = {}  # Use dict for dedup by _id
    with open(dev_or_test_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj['positive_passages'][0]['docid']
            if doc_id not in corpus_dict:
                corpus_dict[doc_id] = {
                    '_id': doc_id,
                    'title': '',
                    'text': obj['positive_passages'][0]['text']
                }
    
    # Write corpus to JSONL directly
    with open(corpus_save_path, 'w', encoding='utf-8') as f:
        for item in corpus_dict.values():
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def find_unique_objects(json_objects, key):
    """Deprecated: kept for compatibility. Use dict-based dedup instead."""
    unique_keys = set()
    unique_objects = []
    for obj in json_objects:
        if key in obj and obj[key] not in unique_keys:
            unique_keys.add(obj[key])
            unique_objects.append(obj)
    return unique_objects

def find_unique_objects_2_keys(json_objects, key1, key2):
    unique_keys = set()
    unique_objects = []

    for obj in json_objects:
        # Assume that both key1 and key2 will always be present in the JSON objects
        # (alternatively, you could add checks to safely handle missing keys).
        key_values = (obj[key1], obj[key2])

        if key_values not in unique_keys:
            unique_keys.add(key_values)
            unique_objects.append(obj)

    return unique_objects

def generate_queries(dev_or_test_file: str, split: str = 'test'):
    """Extract unique query entries from input file."""
    split_path = Path(split)
    split_path.mkdir(parents=True, exist_ok=True)
    
    queries_save_path = split_path / 'queries.jsonl'
    
    query_dict = {}  # Use dict for dedup by _id
    with open(dev_or_test_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query_id = obj['query_id']
            if query_id not in query_dict:
                query_dict[query_id] = {'_id': query_id, 'text': obj['query']}
    
    # Write queries to JSONL directly
    with open(queries_save_path, 'w', encoding='utf-8') as f:
        for item in query_dict.values():
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_qrels(dev_or_test_file: str, split: str):
    """Generate qrels TSV file with dedup by (query_id, corpus_id) pair."""
    split_path = Path(split)
    split_path.mkdir(parents=True, exist_ok=True)
    qrels_dir = split_path / 'qrels'
    qrels_dir.mkdir(parents=True, exist_ok=True)
    
    qrels_save_path = qrels_dir / f'{split}.tsv'
    
    # Use set of tuples for efficient dedup of (query_id, corpus_id) pairs
    qrels_set = set()
    unique_queries = set()
    unique_corpus = set()
    
    with open(dev_or_test_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query_id = obj['query_id']
            corpus_id = obj['positive_passages'][0]['docid']
            
            qrels_set.add((query_id, corpus_id))
            unique_queries.add(query_id)
            unique_corpus.add(corpus_id)
    
    # Print stats
    print(f"Unique queries: {len(unique_queries)}")
    print(f"Unique corpus: {len(unique_corpus)}")
    
    # Write qrels to TSV
    with open(qrels_save_path, 'w', encoding='utf-8') as f:
        f.write('query-id\tcorpus-id\tscore\n')
        for query_id, corpus_id in sorted(qrels_set):
            f.write(f'{query_id}\t{corpus_id}\t1\n')


def main():
    parser = argparse.ArgumentParser(description='Generate corpus, queries, and qrels from retrieval dataset.')
    parser.add_argument('--file', type=str, required=True, help='Input file (JSONL format)')
    parser.add_argument('--split', type=str, required=True, help='Output split name (e.g., test)')
    args = parser.parse_args()
    
    generate_corpus(args.file, args.split)
    generate_queries(args.file, args.split)
    generate_qrels(args.file, args.split)

if __name__ == "__main__":
    sys.exit(main())

