import argparse
import csv
import json
import os
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


MetricName = Literal["mrr", "hit", "ndcg", "map", "recall", "precision"]
Alternative = Literal["two-sided", "greater", "less"]


def load_qrels(dataset_path: str, split: str) -> dict[str, dict[str, int]]:
    qrels_file = os.path.join(dataset_path, "qrels", f"{split}.tsv")
    if not os.path.exists(qrels_file):
        raise FileNotFoundError(f"Qrels file not found: {qrels_file}")

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected_cols = {"query-id", "corpus-id", "score"}
        missing = expected_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Qrels file must contain columns {sorted(expected_cols)}, missing {sorted(missing)} in {qrels_file}"
            )

        for row in reader:
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(float(row["score"]))
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = score

    return qrels


def load_predictions(path: str) -> dict[str, dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Prediction file must contain a dict, got {type(raw)}: {path}")

    out: dict[str, dict[str, float]] = {}
    for qid, docs in raw.items():
        if not isinstance(docs, dict):
            continue
        out[str(qid)] = {str(doc_id): float(score) for doc_id, score in docs.items()}
    return out


def reciprocal_rank_at_k(
    ranked_doc_ids: Iterable[str], relevant_doc_ids: set[str], k: int
) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids):
        if rank >= k:
            break
        if doc_id in relevant_doc_ids:
            return 1.0 / (rank + 1)
    return 0.0


def hit_at_k(ranked_doc_ids: Iterable[str], relevant_doc_ids: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids):
        if rank >= k:
            break
        if doc_id in relevant_doc_ids:
            return 1.0
    return 0.0


def precision_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / k


def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    num_rel = len(relevant_doc_ids)
    if num_rel == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / num_rel


def average_precision_at_k(
    ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int
) -> float:
    num_rel = len(relevant_doc_ids)
    if num_rel == 0:
        return 0.0

    hits = 0
    ap_sum = 0.0
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            ap_sum += hits / idx

    denom = min(num_rel, k)
    if denom == 0:
        return 0.0
    return ap_sum / denom


def ndcg_at_k(ranked_doc_ids: list[str], rel_scores: dict[str, int], k: int) -> float:
    def _dcg(rels: list[int]) -> float:
        return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rels))

    top_rels = [int(rel_scores.get(doc_id, 0)) for doc_id in ranked_doc_ids[:k]]
    dcg = _dcg(top_rels)

    ideal_rels = sorted([int(v) for v in rel_scores.values()], reverse=True)[:k]
    idcg = _dcg(ideal_rels)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def per_query_scores(
    metric: MetricName,
    k: int,
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    qids: list[str],
) -> np.ndarray:
    vals: list[float] = []
    for qid in qids:
        rel_scores = qrels[qid]
        rel_docs = {doc_id for doc_id, rel in rel_scores.items() if rel > 0}
        ranked = sorted(run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]

        if metric == "mrr":
            vals.append(reciprocal_rank_at_k(ranked_doc_ids, rel_docs, k))
        elif metric == "hit":
            vals.append(hit_at_k(ranked_doc_ids, rel_docs, k))
        elif metric == "precision":
            vals.append(precision_at_k(ranked_doc_ids, rel_docs, k))
        elif metric == "recall":
            vals.append(recall_at_k(ranked_doc_ids, rel_docs, k))
        elif metric == "map":
            vals.append(average_precision_at_k(ranked_doc_ids, rel_docs, k))
        elif metric == "ndcg":
            vals.append(ndcg_at_k(ranked_doc_ids, rel_scores, k))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    return np.array(vals, dtype=float)


def permutation_p_value(
    diffs: np.ndarray,
    n_resamples: int,
    seed: int,
    alternative: Alternative,
    exact_max_n: int = 20,
) -> float:
    rng = np.random.default_rng(seed)
    obs = float(np.mean(diffs))
    n = diffs.shape[0]

    if n <= exact_max_n:
        test_stats = np.array(
            [float(np.mean(np.array(signs, dtype=float) * diffs)) for signs in product([-1, 1], repeat=n)],
            dtype=float,
        )
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, n), replace=True)
        test_stats = np.mean(signs * diffs[None, :], axis=1)

    if alternative == "two-sided":
        extreme = np.sum(np.abs(test_stats) >= abs(obs))
    elif alternative == "greater":
        extreme = np.sum(test_stats >= obs)
    else:  # less
        extreme = np.sum(test_stats <= obs)

    p = (extreme + 1.0) / (len(test_stats) + 1.0)
    return float(p)


def paired_effect_size(diffs: np.ndarray) -> float:
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diffs) / std)


def resolve_prediction_path(
    explicit_path: str | None,
    model_name: str | None,
    dataset_name: str,
    result_root: str,
) -> str:
    if explicit_path:
        return explicit_path
    if not model_name:
        raise ValueError("You must provide either --pred-a/--pred-b or --model-a/--model-b")

    model_path = model_name.replace("/", "__").replace(" ", "_")
    return str(Path(result_root) / model_path / f"{dataset_name}_default_predictions.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-query retrieval scores for two models and run paired significance tests."
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset directory containing corpus.jsonl, queries.jsonl, qrels/")
    parser.add_argument("--dataset-name", type=str, required=True, help="Task name used in prediction filenames, e.g. rfc8205")
    parser.add_argument("--split", type=str, default="test", help="Split name, e.g. test")

    parser.add_argument("--pred-a", type=str, default=None, help="Prediction file for model A (JSON)")
    parser.add_argument("--pred-b", type=str, default=None, help="Prediction file for model B (JSON)")
    parser.add_argument("--model-a", type=str, default=None, help="Model A name (used to auto-resolve prediction path)")
    parser.add_argument("--model-b", type=str, default=None, help="Model B name (used to auto-resolve prediction path)")
    parser.add_argument(
        "--result-root",
        type=str,
        default="results/stage1/test/i2c",
        help="Folder containing model subfolders with saved predictions",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="mrr",
        choices=["mrr", "hit", "ndcg", "map", "recall", "precision"],
        help="Per-query metric to compare",
    )
    parser.add_argument("--k", type=int, default=10, help="Cutoff k for metric@k")

    parser.add_argument("--alternative", type=str, default="two-sided", choices=["two-sided", "greater", "less"])
    parser.add_argument("--n-resamples", type=int, default=10000, help="Number of Monte-Carlo sign-flip resamples")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-csv", type=str, default="per_query_scores.csv", help="Output CSV for per-query scores")
    parser.add_argument("--output-json", type=str, default="significance_summary.json", help="Output JSON summary")

    args = parser.parse_args()

    pred_a_path = resolve_prediction_path(args.pred_a, args.model_a, args.dataset_name, args.result_root)
    pred_b_path = resolve_prediction_path(args.pred_b, args.model_b, args.dataset_name, args.result_root)

    qrels = load_qrels(args.dataset_path, args.split)
    run_a = load_predictions(pred_a_path)
    run_b = load_predictions(pred_b_path)

    qids = [qid for qid in qrels.keys()]

    scores_a = per_query_scores(args.metric, args.k, qrels, run_a, qids)
    scores_b = per_query_scores(args.metric, args.k, qrels, run_b, qids)

    diffs = scores_b - scores_a  # B - A

    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))
    delta = float(np.mean(diffs))

    p_perm = permutation_p_value(
        diffs,
        n_resamples=args.n_resamples,
        seed=args.seed,
        alternative=args.alternative,
    )

    ttest = ttest_rel(scores_b, scores_a, alternative=args.alternative)
    try:
        wil = wilcoxon(scores_b, scores_a, alternative=args.alternative, zero_method="wilcox")
        p_wilcoxon = float(wil.pvalue)
        wilcoxon_stat = float(wil.statistic)
    except ValueError:
        p_wilcoxon = float("nan")
        wilcoxon_stat = float("nan")

    effect = paired_effect_size(diffs)

    output_rows = []
    for i, qid in enumerate(qids):
        output_rows.append(
            {
                "query_id": qid,
                "score_a": float(scores_a[i]),
                "score_b": float(scores_b[i]),
                "diff_b_minus_a": float(diffs[i]),
            }
        )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["query_id", "score_a", "score_b", "diff_b_minus_a"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "prediction_a": pred_a_path,
        "prediction_b": pred_b_path,
        "metric": f"{args.metric}@{args.k}",
        "num_queries": len(qids),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta_b_minus_a": delta,
        "alternative": args.alternative,
        "p_value_permutation": p_perm,
        "p_value_paired_ttest": float(ttest.pvalue),
        "ttest_statistic": float(ttest.statistic),
        "p_value_wilcoxon": p_wilcoxon,
        "wilcoxon_statistic": wilcoxon_stat,
        "effect_size_cohens_d_paired": effect,
        "output_csv": args.output_csv,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Paired significance test (B vs A)")
    print("=" * 70)
    print(f"Metric: {args.metric}@{args.k}")
    print(f"Queries: {len(qids)}")
    print(f"Model A mean: {mean_a:.6f}")
    print(f"Model B mean: {mean_b:.6f}")
    print(f"Delta (B - A): {delta:.6f}")
    print(f"Permutation p-value ({args.alternative}): {p_perm:.6g}")
    print(f"Paired t-test p-value ({args.alternative}): {float(ttest.pvalue):.6g}")
    if not np.isnan(p_wilcoxon):
        print(f"Wilcoxon p-value ({args.alternative}): {p_wilcoxon:.6g}")
    else:
        print("Wilcoxon p-value: NaN (insufficient non-zero paired differences)")
    print(f"Paired Cohen's d: {effect:.6f}")
    print(f"Per-query scores saved to: {args.output_csv}")
    print(f"Summary saved to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
