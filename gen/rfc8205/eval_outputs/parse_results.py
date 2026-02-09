
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from statistics import mean, median

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load evaluation records from JSONL."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


from typing import Dict, Any, Optional

def macro_micro_overall(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Given the per-qid `summary` from aggregate_by_qid_max_mean_median(records),
    compute, per metric, the macro and micro averages for:
      - max (uses qid-level max.value)
      - median (uses qid-level median)
      - mean (uses qid-level mean)

    Returns a dict:
    {
      metric_name: {
        "macro_max": float|None, "micro_max": float|None,
        "macro_median": float|None, "micro_median": float|None,
        "macro_mean": float|None, "micro_mean": float|None,
        "n_qids_max": int, "n_qids_median": int, "n_qids_mean": int,
        "n_values_max": int, "n_values_median": int, "n_values_mean": int
      },
      ...
    }

    Notes:
      - Macro = unweighted mean across qids (each qid contributes equally).
      - Micro = weighted mean across qids by `count`.
        For 'max', there is only one value per qid, so micro==macro (n_values_max = n_qids_max).
    """
    # Collect per-metric lists for macro and running sums/counts for micro
    macro_lists = {
        "max":    {},   # metric -> [qid_max_value, ...]
        "median": {},
        "mean":   {},
    }
    micro_sums = {
        "max":    {},   # metric -> sum of values * weight
        "median": {},
        "mean":   {},
    }
    micro_counts = {
        "max":    {},   # metric -> total weight (count)
        "median": {},
        "mean":   {},
    }

    for qid, metrics in summary.items():
        for metric, agg in metrics.items():
            # ----- MAX -----
            max_obj = agg.get("max")
            if isinstance(max_obj, dict):
                max_val = max_obj.get("value")
                if isinstance(max_val, (int, float)):
                    macro_lists["max"].setdefault(metric, []).append(float(max_val))
                    # For 'max', each qid contributes exactly one value (weight = 1)
                    micro_sums["max"][metric]   = micro_sums["max"].get(metric, 0.0) + float(max_val) * 1.0
                    micro_counts["max"][metric] = micro_counts["max"].get(metric, 0) + 1

            # ----- MEDIAN -----
            med_val = agg.get("median")
            c = int(agg.get("count", 0) or 0)  # guard against None
            if isinstance(med_val, (int, float)):
                med_val = float(med_val)
                macro_lists["median"].setdefault(metric, []).append(med_val)
                if c > 0:
                    # Micro: weight by the number of chunk-level values behind this qid
                    micro_sums["median"][metric]   = micro_sums["median"].get(metric, 0.0) + med_val * c
                    micro_counts["median"][metric] = micro_counts["median"].get(metric, 0) + c

            # ----- MEAN -----
            mu = agg.get("mean")
            if isinstance(mu, (int, float)):
                mu = float(mu)
                macro_lists["mean"].setdefault(metric, []).append(mu)
                if c > 0:
                    micro_sums["mean"][metric]   = micro_sums["mean"].get(metric, 0.0) + mu * c
                    micro_counts["mean"][metric] = micro_counts["mean"].get(metric, 0) + c

    # Assemble results
    overall: Dict[str, Dict[str, Optional[float]]] = {}
    all_metrics = set(macro_lists["max"]) | set(macro_lists["median"]) | set(macro_lists["mean"]) \
                  | set(micro_sums["max"]) | set(micro_sums["median"]) | set(micro_sums["mean"])

    for metric in all_metrics:
        macro_max_list    = macro_lists["max"].get(metric, [])
        macro_median_list = macro_lists["median"].get(metric, [])
        macro_mean_list   = macro_lists["mean"].get(metric, [])

        micro_max_sum     = micro_sums["max"].get(metric, 0.0)
        micro_median_sum  = micro_sums["median"].get(metric, 0.0)
        micro_mean_sum    = micro_sums["mean"].get(metric, 0.0)

        micro_max_cnt     = micro_counts["max"].get(metric, 0)
        micro_median_cnt  = micro_counts["median"].get(metric, 0)
        micro_mean_cnt    = micro_counts["mean"].get(metric, 0)

        overall[metric] = {
            # Macro: simple average over qids that had that aggregate
            "macro_max":    (sum(macro_max_list)    / len(macro_max_list)    if macro_max_list    else None),
            "macro_median": (sum(macro_median_list) / len(macro_median_list) if macro_median_list else None),
            "macro_mean":   (sum(macro_mean_list)   / len(macro_mean_list)   if macro_mean_list   else None),

            # Micro: weighted by per-qid count (for max, count=1 per qid, so equals macro)
            "micro_max":    (micro_max_sum    / micro_max_cnt    if micro_max_cnt    else None),
            "micro_median": (micro_median_sum / micro_median_cnt if micro_median_cnt else None),
            "micro_mean":   (micro_mean_sum   / micro_mean_cnt   if micro_mean_cnt   else None),

            # Diagnostics
            "n_qids_max":    len(macro_max_list),
            "n_qids_median": len(macro_median_list),
            "n_qids_mean":   len(macro_mean_list),
            "n_values_max":    micro_max_cnt,     # == n_qids_max
            "n_values_median": micro_median_cnt,  # total chunk-values behind medians
            "n_values_mean":   micro_mean_cnt,    # total chunk-values behind means
        }

    return overall

def aggregate_by_qid_max_mean_median(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each qid, compute per-metric:
      - max value (+ index that achieved it, and optional reason)
      - mean value
      - median value
      - count of contributing records

    Returns:
      {
        qid: {
          metric_name: {
            "max":    {"value": float, "index": int|None, "reason": str|None},
            "mean":   float|None,
            "median": float|None,
            "count":  int
          },
          ...
        },
        ...
      }
    """
    # Collect values per (qid, metric)
    per_qid_metric_vals: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for rec in records:
        qid: Optional[str] = rec.get("qid")
        if not qid:
            # skip records missing qid
            continue

        metrics: Dict[str, Any] = rec.get("metrics", {})
        idx: Optional[int] = rec.get("index")

        if qid not in per_qid_metric_vals:
            per_qid_metric_vals[qid] = {}

        for m_name, payload in metrics.items():
            val = payload.get("value")
            if not isinstance(val, (int, float)):
                # skip non-numeric values
                continue
            val = float(val)
            per_qid_metric_vals[qid].setdefault(m_name, []).append({
                "value": val,
                "index": idx,
                "reason": payload.get("reason"),
            })

    # Compute aggregates
    summary: Dict[str, Dict[str, Any]] = {}
    for qid, metric_map in per_qid_metric_vals.items():
        summary[qid] = {}
        for m_name, entries in metric_map.items():
            values = [e["value"] for e in entries]
            if not values:
                agg = {"max": None, "mean": None, "median": None, "count": 0}
            else:
                # max (also capture the associated index/reason from the max entry)
                max_entry = max(entries, key=lambda e: e["value"])
                agg = {
                    "max":    {"value": max_entry["value"],
                               "index": max_entry["index"]},
                    "mean":   float(mean(values)) if values else None,
                    "median": float(median(values)) if values else None,
                    "count":  len(values),
                }
            summary[qid][m_name] = agg

    return summary

def save_summary_json(summary: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """Write the max/mean/median summary to a JSON file."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def save_summary_csv(summary: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """
    Flatten the summary to CSV with columns:
      qid, metric, max_value, max_index, mean, median, count
    """
    rows = []
    for qid, metrics in summary.items():
        for metric, agg in metrics.items():
            max_val   = agg.get("max", {}).get("value")
            max_index = agg.get("max", {}).get("index")
            rows.append({
                "qid": qid,
                "metric": metric,
                "max_value": max_val,
                "max_index": max_index,
                "mean": agg.get("mean"),
                "median": agg.get("median"),
                "count": agg.get("count"),
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qid", "metric", "max_value", "max_index", "mean", "median", "count"]
        )
        writer.writeheader()
        writer.writerows(rows)


from typing import Dict, Any, Optional

def mean_of_max_median_mean(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Given the per-qid summary from `aggregate_by_qid_max_mean_median(records)`,
    compute, for each metric:
       - mean_max:    average of per-qid max.value
       - mean_median: average of per-qid median
       - mean_mean:   average of per-qid mean

    Returns:
      {
        metric_name: {
          "mean_max": <float or None>,
          "mean_median": <float or None>,
          "mean_mean": <float or None>,
          "count_max": <int>,      # number of qids contributing to mean_max
          "count_median": <int>,   # number of qids contributing to mean_median
          "count_mean": <int>,     # number of qids contributing to mean_mean
        },
        ...
      }
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}

    # First, collect per-metric lists of values
    buckets_max: Dict[str, list] = {}
    buckets_median: Dict[str, list] = {}
    buckets_mean: Dict[str, list] = {}

    for qid, metrics in summary.items():
        for metric, agg in metrics.items():
            # max.value (exists only if agg["max"] is not None)
            max_obj = agg.get("max")
            if isinstance(max_obj, dict):
                max_val = max_obj.get("value")
                if isinstance(max_val, (int, float)):
                    buckets_max.setdefault(metric, []).append(float(max_val))

            # median
            med_val = agg.get("median")
            if isinstance(med_val, (int, float)):
                buckets_median.setdefault(metric, []).append(float(med_val))

            # mean
            mu = agg.get("mean")
            if isinstance(mu, (int, float)):
                buckets_mean.setdefault(metric, []).append(float(mu))

    # Compute means per metric
    metrics_all = set(buckets_max) | set(buckets_median) | set(buckets_mean)
    for metric in metrics_all:
        max_vals = buckets_max.get(metric, [])
        med_vals = buckets_median.get(metric, [])
        mean_vals = buckets_mean.get(metric, [])

        out[metric] = {
            "mean_max":    (sum(max_vals) / len(max_vals) if max_vals else None),
            "mean_median": (sum(med_vals) / len(med_vals) if med_vals else None),
            "mean_mean":   (sum(mean_vals) / len(mean_vals) if mean_vals else None),
            "count_max":   len(max_vals),
            "count_median":len(med_vals),
            "count_mean":  len(mean_vals),
        }

    return out


in_path  = Path("bm25/ragas_eval_results.jsonl")
out_json = Path("bm25/agg_max_mean_median_by_qid.json")

records = load_jsonl(in_path)
summary = aggregate_by_qid_max_mean_median(records)

# Inspect one qid (replace with a real qid from your data)
#some_qid = next(iter(summary.keys()), None)
#if some_qid:
#    print(f"Aggregates for {some_qid}:")
#    print(json.dumps(summary[some_qid], indent=2))

# Save outputs
#save_summary_json(summary, out_json)
#print(f"Wrote:\n - {out_json}\n")


# 2) Compute mean of max/median/mean across all qids
#overall_means = mean_of_max_median_mean(summary)

# Inspect
#print(json.dumps(overall_means, indent=2))

# Example: access one metric (e.g., "AnswerRelevancy")
#ar = overall_means.get("AnswerRelevancy", {})
#print("AnswerRelevancy -> mean_max:", ar.get("mean_max"))
#print("AnswerRelevancy -> mean_median:", ar.get("mean_median"))
#print("AnswerRelevancy -> mean_mean:", ar.get("mean_mean"))


# micro
overall = macro_micro_overall(summary)

print(json.dumps(overall, indent=2))

# Example access:
ar = overall.get("AnswerRelevancy", {})
print("AnswerRelevancy macro_mean :", ar.get("macro_mean"))
print("AnswerRelevancy micro_mean :", ar.get("micro_mean"))
print("AnswerRelevancy macro_median:", ar.get("macro_median"))
print("AnswerRelevancy micro_median:", ar.get("micro_median"))
print("AnswerRelevancy macro_max :", ar.get("macro_max"))
print("AnswerRelevancy micro_max :", ar.get("micro_max"))



