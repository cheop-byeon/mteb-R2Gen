import json
import os
from pathlib import Path

# Define source and destination paths
source_dirs = [
    "rfc7657/i2c/test",
    "rfc8205/i2c/test",
    "rfc8335/i2c/test",
]

dest_dir = "rfcs/i2c/test"

def combine_jsonl_files(source_dirs, dest_file):
    """Combine corpus.jsonl files from multiple directories into one."""
    with open(dest_file, 'w') as outfile:
        for source_dir in source_dirs:
            corpus_file = os.path.join(source_dir, "corpus.jsonl")
            if os.path.exists(corpus_file):
                with open(corpus_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                print(f"Added corpus from {source_dir}")
            else:
                print(f"Warning: {corpus_file} not found")

def combine_queries(source_dirs, dest_file):
    """Combine queries.jsonl files from multiple directories into one."""
    with open(dest_file, 'w') as outfile:
        for source_dir in source_dirs:
            queries_file = os.path.join(source_dir, "queries.jsonl")
            if os.path.exists(queries_file):
                with open(queries_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                print(f"Added queries from {source_dir}")
            else:
                print(f"Warning: {queries_file} not found")

def combine_qrels(source_dirs, dest_file):
    """Combine qrels/test.tsv files from multiple directories into one."""
    # Write header
    with open(dest_file, 'w') as outfile:
        outfile.write("query-id\tcorpus-id\tscore\n")
    
    # Append data from each source
    with open(dest_file, 'a') as outfile:
        for source_dir in source_dirs:
            qrels_file = os.path.join(source_dir, "qrels/test.tsv")
            if os.path.exists(qrels_file):
                with open(qrels_file, 'r') as infile:
                    # Skip header
                    next(infile)
                    for line in infile:
                        outfile.write(line)
                print(f"Added qrels from {source_dir}")
            else:
                print(f"Warning: {qrels_file} not found")

def main():
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.join(dest_dir, "qrels"), exist_ok=True)
    
    # Combine files
    corpus_dest = os.path.join(dest_dir, "corpus.jsonl")
    queries_dest = os.path.join(dest_dir, "queries.jsonl")
    qrels_dest = os.path.join(dest_dir, "qrels", "test.tsv")
    
    print("Combining corpus files...")
    combine_jsonl_files(source_dirs, corpus_dest)
    
    print("\nCombining queries files...")
    combine_queries(source_dirs, queries_dest)
    
    print("\nCombining qrels files...")
    combine_qrels(source_dirs, qrels_dest)
    
    print(f"\nCombined files saved to {dest_dir}/")
    print(f"  - corpus.jsonl")
    print(f"  - queries.jsonl")
    print(f"  - qrels/test.tsv")

if __name__ == "__main__":
    main()
