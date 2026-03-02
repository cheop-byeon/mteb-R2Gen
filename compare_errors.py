import os
import json
import sys
import argparse
from typing import List, Dict, Set
from pathlib import Path

def load_error_cases(error_file_path: str) -> tuple[Dict[str, dict], Set[str]]:
    """
    Load error cases from a JSONL file.
    
    Returns:
        Tuple of (error_cases_dict, error_query_ids_set)
        - error_cases_dict: {query_id: error_case_object}
        - error_query_ids_set: set of query IDs with errors
    """
    error_cases = {}
    error_query_ids = set()
    
    if not os.path.exists(error_file_path):
        print(f"Warning: File not found: {error_file_path}")
        return error_cases, error_query_ids
    
    with open(error_file_path, 'r') as f:
        for line in f:
            case = json.loads(line)
            query_id = case['query_id']
            error_cases[query_id] = case
            error_query_ids.add(query_id)
    
    return error_cases, error_query_ids

def compare_errors(model1: str, model2: str, dataset_name: str = "rfc8205", direction: str = "c2i", split: str = "test"):
    """
    Compare error analysis results between two models.
    Find queries where model2 fails but model1 succeeds.
    
    Args:
        model1: Better model name (e.g., "jiebi/RFC-DRAlign-LN")
        model2: Worse model name to compare against (e.g., "bm25s")
        dataset_name: Dataset name (e.g., "rfc8205", "rfc7657", "rfc8335")
        direction: "c2i" (code-to-issue) or "i2c" (issue-to-code)
        split: "test" or "train"
    """
    
    # Convert model names to path-safe format
    model1_path = model1.replace("/", "__").replace(" ", "_")
    model2_path = model2.replace("/", "__").replace(" ", "_")
    
    # Setup paths for error analysis files
    error_file1 = f'error_analysis_{model1_path}_{dataset_name}_{direction}_{split}.jsonl'
    error_file2 = f'error_analysis_{model2_path}_{dataset_name}_{direction}_{split}.jsonl'
    
    print(f"Comparing error analysis results:")
    print(f"  Model 1 (better): {model1}")
    print(f"  Model 2 (worse):  {model2}")
    print(f"  Dataset: {dataset_name}, Direction: {direction}, Split: {split}")
    print()
    
    # Load error cases for both models
    print(f"Loading error cases from:")
    print(f"  {error_file1}")
    print(f"  {error_file2}")
    print()
    
    error_cases1, error_ids1 = load_error_cases(error_file1)
    error_cases2, error_ids2 = load_error_cases(error_file2)
    
    # Check if files were loaded successfully
    if not error_ids1 and not error_ids2:
        print("Error: Both error analysis files are empty or missing.")
        print("Please run analysis.py for both models first.")
        return None
    
    # Find queries that are errors in both models (intersection)
    common_errors = error_ids1 & error_ids2
    
    # Find queries that are errors only in model2 (model2's unique failures)
    model2_unique_errors = error_ids2 - error_ids1
    
    # Find queries that are errors only in model1 (model1's unique failures)
    model1_unique_errors = error_ids1 - error_ids2
    
    # Print statistics
    print("=" * 80)
    print("COMPARISON STATISTICS")
    print("=" * 80)
    print(f"Total error cases in Model 1: {len(error_ids1)}")
    print(f"Total error cases in Model 2: {len(error_ids2)}")
    print()
    print(f"Common errors (both models fail):           {len(common_errors)}")
    print(f"Model 2 unique errors (Model 2 fails only): {len(model2_unique_errors)}")
    print(f"Model 1 unique errors (Model 1 fails only): {len(model1_unique_errors)}")
    print()
    
    # Calculate improvement
    if len(error_ids2) > 0:
        improvement_count = len(model2_unique_errors)
        improvement_pct = (improvement_count / len(error_ids2)) * 100
        print(f"Model 1 fixes {improvement_count} errors from Model 2 ({improvement_pct:.2f}% improvement)")
    
    if len(error_ids1) > 0 and len(model1_unique_errors) > 0:
        regression_count = len(model1_unique_errors)
        print(f"Model 1 introduces {regression_count} new errors compared to Model 2")
    
    print()
    
    # Save model2's unique errors (queries where model2 fails but model1 succeeds)
    if model2_unique_errors:
        output_file = f'error_comparison_{model2_path}_vs_{model1_path}_{dataset_name}_{direction}_{split}.jsonl'
        with open(output_file, 'w') as f:
            for qid in model2_unique_errors:
                error_case = error_cases2[qid]
                # Add comparison metadata
                error_case['comparison_note'] = f"Model 2 ({model2}) fails, Model 1 ({model1}) succeeds"
                f.write(json.dumps(error_case) + '\n')
        
        print(f"Saved {len(model2_unique_errors)} unique Model 2 error cases to: {output_file}")
    else:
        print("No unique errors found in Model 2 (all Model 2 errors also exist in Model 1)")
    
    # Also save a comparison summary
    summary_file = f'error_comparison_summary_{model2_path}_vs_{model1_path}_{dataset_name}_{direction}_{split}.json'
    summary = {
        "model1": model1,
        "model2": model2,
        "dataset": dataset_name,
        "direction": direction,
        "split": split,
        "model1_error_count": len(error_ids1),
        "model2_error_count": len(error_ids2),
        "common_errors_count": len(common_errors),
        "model2_unique_errors_count": len(model2_unique_errors),
        "model1_unique_errors_count": len(model1_unique_errors),
        "improvement_percentage": (len(model2_unique_errors) / len(error_ids2) * 100) if error_ids2 else 0,
        "common_error_query_ids": list(common_errors),
        "model2_unique_error_query_ids": list(model2_unique_errors),
        "model1_unique_error_query_ids": list(model1_unique_errors)
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved comparison summary to: {summary_file}")
    print()
    
    # Print sample query IDs for each category
    if common_errors:
        print(f"Sample common error query IDs (first 5): {list(common_errors)[:5]}")
    if model2_unique_errors:
        print(f"Sample Model 2 unique error query IDs (first 5): {list(model2_unique_errors)[:5]}")
    if model1_unique_errors:
        print(f"Sample Model 1 unique error query IDs (first 5): {list(model1_unique_errors)[:5]}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare error analysis results between two models. "
                    "Finds queries where Model 2 fails but Model 1 succeeds."
    )
    parser.add_argument("--model1", type=str, required=True, 
                        help="Better model name (e.g., jiebi/RFC-DRAlign-LN)")
    parser.add_argument("--model2", type=str, required=True, 
                        help="Worse model name to compare (e.g., bm25s)")
    parser.add_argument("--dataset", type=str, default="rfc8205", 
                        help="Dataset name (e.g., rfc8205, rfc7657, rfc8335)")
    parser.add_argument("--direction", type=str, default="c2i", choices=["c2i", "i2c"], 
                        help="Direction: c2i or i2c")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], 
                        help="Split: test or train")
    
    args = parser.parse_args()
    
    summary = compare_errors(args.model1, args.model2, args.dataset, args.direction, args.split)
