import argparse
import os
import sys
from typing import Optional
import mteb
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
from pathlib import Path
# Configuration for top_k values by model, direction, dataset, and split


def create_retrieval_task(metadata: TaskMetadata) -> AbsTaskRetrieval:
    """
    Factory function to create a custom retrieval task with dynamic metadata.
    
    This avoids hardcoding metadata in a class definition and allows
    for flexible task creation with runtime-provided configurations.
    
    Args:
        metadata: TaskMetadata object containing task configuration
        
    Returns:
        An instance of AbsTaskRetrieval with the provided metadata
    """
    class DynamicRetrievalTask(AbsTaskRetrieval):
        pass
    
    DynamicRetrievalTask.metadata = metadata
    return DynamicRetrievalTask()


def main() -> int:
    """
    Main entry point for running MTEB evaluation with custom configurations.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Run MTEB evaluation with dynamically configured retrieval tasks"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'bm25s')")
    parser.add_argument("--direction", type=str, required=True, help="Direction (e.g., 'i2c', 'c2i')")
    parser.add_argument("--path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--split", type=str, required=True, help="Data split (e.g., 'dev', 'test')")
    parser.add_argument("--name", type=str, required=True, help="Dataset/task name")
    parser.add_argument("--topk", type=int, default=None, help="Override top K value (optional)")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Create metadata for the task
    metadata = TaskMetadata(
        name=args.name,
        description="Code Retrieval Task",
        reference=None,
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[args.split],
        eval_langs=["eng-Latn"],
        main_score="hit_at_10",
        dataset={
            "path": args.path,
            "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        },
        date=("2012-01-01", "2020-01-01"),
        domains=["Programming"],
        task_subtypes=["Code retrieval", "Reasoning as Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
    )
   
    top_k = args.topk
    # If using bm25s, set top_k to the number of items in corpus.jsonl inside the provided dataset path.
    if args.model == "bm25s":
        try:
            dataset_path = Path(args.path)
            # If args.path is a file and is named corpus.jsonl, use it; otherwise look for corpus.jsonl inside the path
            if dataset_path.is_file() and dataset_path.name == "corpus.jsonl":
                corpus_file = dataset_path
            else:
                corpus_file = dataset_path / "corpus.jsonl"

            if not corpus_file.exists():
                print(f"Warning: corpus.jsonl not found at {corpus_file.resolve()}, keeping top_k={top_k}", file=sys.stderr)
            else:
                # Count non-empty lines to estimate number of items
                with corpus_file.open("r", encoding="utf-8") as cf:
                    count = sum(1 for _ in cf if _.strip())
                top_k = min(int(count), 1000)
                print(f"Model is 'bm25s' — setting top_k to corpus size: {top_k}")
        except Exception as e:
            print(f"Warning: failed to set top_k from corpus.jsonl: {e}; keeping top_k={top_k}", file=sys.stderr)
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Direction: {args.direction}")
    print(f"  Dataset: {args.name}")
    print(f"  Split: {args.split}")
    print(f"  Top K: {top_k}")
    print(f"  Batch Size: {args.batch_size}")
    print()
    
    # Load model and run evaluation
    try:
        model = mteb.get_model(args.model)
        task = create_retrieval_task(metadata)
        evaluation = mteb.MTEB(tasks=[task])
        
        evaluation.run(
            model,
            encode_kwargs={"batch_size": args.batch_size},
            top_k=top_k,
            output_folder=f"results/stage1/{args.split}/{args.direction}",
            save_predictions=True,
            save_predictions_folder=f"{args.model}",
            output_type="all",
            overwrite_results=True,
        )
        
        print("Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())




