# Error Analysis Script

This document explains how to use the `analysis.py` script to analyze retrieval errors in your MTEB evaluation results.

## Overview

The `analysis.py` script identifies queries where the retrieval model fails to find any relevant document in the top-k results (HIT@k = 0). It collects detailed information about these error cases to help you understand model failures.

## What It Does

1. **Identifies error cases**: Finds all queries where `HIT@k = 0` (no relevant document in top-k retrieved results)
2. **Collects comprehensive data**: For each error case, gathers:
   - Query ID and query text
   - Top-k retrieved documents with their scores
   - Relevant documents (ground truth) for comparison
3. **Generates statistics**: Calculates HIT rates at k=1, 5, 10, and 100
4. **Outputs results**: Creates two files:
   - JSONL file: Individual error cases (one per line)
   - JSON file: Summary report with statistics and all error cases

## Prerequisites

- Completed MTEB evaluation using `RFCAlign_IR_mteb.py`
- Evaluation results should be in: `results/stage1/{split}/{direction}/{model_path}/`
- Dataset files should be in: `ir/{dataset_name}/{direction}/{split}/`
  - `queries.jsonl` - Query data
  - `corpus.jsonl` - Document corpus
  - `qrels/{split}.tsv` - Ground truth relevance labels (TSV format: query_id, doc_id, score)

## Usage

### Basic Command

```bash
python analysis.py --model "model_name" [--dataset rfc8205] [--direction c2i] [--split test] [--top_k 10]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | Model name (e.g., `jiebi/RFC-DRAlign-LN`, `bm25s`) |
| `--dataset` | `rfc8205` | Dataset name (`rfc8205`, `rfc7657`, `rfc8335`, etc.) |
| `--direction` | `c2i` | Retrieval direction: `c2i` (code-to-issue) or `i2c` (issue-to-code) |
| `--split` | `test` | Data split: `test` or `train` |
| `--top_k` | `10` | k value for HIT@k metric |

### Examples

**Analyze a model on rfc8205 (default)**
```bash
python analysis.py --model "jiebi/RFC-DRAlign-LN"
```

**Analyze BM25 on rfc8205 with different parameters**
```bash
python analysis.py --model "bm25s" --direction c2i --split test
```

**Analyze a model on a different dataset**
```bash
python analysis.py --model "jiebi/RFC-DRAlign-QN" --dataset rfc8335 --direction i2c
```

**Analyze with different top-k threshold**
```bash
python analysis.py --model "jiebi/RFC-DRAlign-LN" --top_k 20
```

## Output Files

The script generates two output files in the current directory:

### 1. JSONL File: `error_analysis_{model}_{dataset}_{direction}_{split}.jsonl`

Contains one JSON object per line, each representing an error case:

```json
{
  "query_id": "q123",
  "query_text": "How to handle RFC compliance?",
  "retrieved_docs_top_k": [
    {
      "doc_id": "doc456",
      "text": "Document content here...",
      "score": 0.95
    },
    {
      "doc_id": "doc789",
      "text": "Another document...",
      "score": 0.87
    }
  ],
  "relevant_docs": [
    {
      "doc_id": "doc999",
      "text": "The actual relevant document..."
    }
  ]
}
```

### 2. JSON File: `error_analysis_summary_{model}_{dataset}_{direction}_{split}.json`

Contains overall statistics and all error cases:

```json
{
  "model": "jiebi/RFC-DRAlign-LN",
  "dataset": "rfc8205",
  "direction": "c2i",
  "split": "test",
  "total_queries": 100,
  "error_cases_count": 5,
  "hit_statistics": {
    "HIT@1": 0.75,
    "HIT@5": 0.85,
    "HIT@10": 0.95,
    "HIT@100": 0.98
  },
  "error_cases": [...]
}
```

## Console Output

The script prints statistics to console:

```
Loading data for model: jiebi/RFC-DRAlign-LN
Dataset: rfc8205, Direction: c2i, Split: test, Top-k: 10

Total queries: 100
Queries with HIT@10 = 0: 5

Hit Statistics across different k values:
  HIT@1: 75 / 100 = 0.75
  HIT@5: 85 / 100 = 0.85
  HIT@10: 95 / 100 = 0.95
  HIT@100: 98 / 100 = 0.98

Saved 5 error cases to: error_analysis_jiebi__RFC-DRAlign-LN_rfc8205_c2i_test.jsonl
Saved summary to: error_analysis_summary_jiebi__RFC-DRAlign-LN_rfc8205_c2i_test.json
```

## Interpreting Results

### HIT@k Metric

- `HIT@k = 1` if at least one relevant document is in the top-k retrieved results
- `HIT@k = 0` if no relevant document is in the top-k results (error case)

The script captures all queries where `HIT@k = 0` for analysis.

### Using Error Cases

Each error case contains:
- **Query text**: What the model was searching for
- **Retrieved documents**: What the model actually retrieved (ranked by score)
- **Relevant documents**: What should have been retrieved

This allows you to:
1. **Identify failure patterns**: Do errors cluster around specific query types or document characteristics?
2. **Debug the model**: Why didn't it rank relevant documents higher?
3. **Improve the dataset**: Are there mislabeled examples?
4. **Iterate on training**: Use error cases to improve training data or model architecture

## Common Workflow

```bash
# 1. Run evaluation
python RFCAlign_IR_mteb.py --model "jiebi/RFC-DRAlign-LN" --direction c2i --split test --path "ir/rfc8205/c2i/test" --name rfc8205 --batch_size 8 --topk 1000

# 2. Analyze errors
python analysis.py --model "jiebi/RFC-DRAlign-LN" --dataset rfc8205 --direction c2i

# 3. Review error cases
# Open error_analysis_jiebi__RFC-DRAlign-LN_rfc8205_c2i_test.jsonl
# Analyze patterns in error_analysis_summary_jiebi__RFC-DRAlign-LN_rfc8205_c2i_test.json
```

## Troubleshooting

### "Error: Missing files"

Make sure you have:
1. Run `RFCAlign_IR_mteb.py` to generate evaluation results
2. Have dataset files in the correct location: `ir/{dataset_name}/{direction}/{split}/`

### No errors found

If no error cases are found (all queries have HIT@k > 0), the script will output:
```
Total queries: X
Queries with HIT@k = 0: 0
```

This means the model performed perfectly on this dataset.

### File paths issues

The script expects to be run from the workspace root directory. Make sure your working directory is:
```
/Users/jiebi/Documents/cheop-byeon/mteb-R2Gen
```
