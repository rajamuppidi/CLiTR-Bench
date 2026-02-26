"""
Bootstrap 95% Confidence Intervals for CLiTR-Bench CMS125 experiments.
Reads per-patient .jsonl result files and bootstraps F1, Precision, Recall,
Auditability Match Rate with n=10,000 resamples.

Usage:
    python3 experiments/analysis/bootstrap_ci.py

Output:
    experiments/analysis/bootstrap_results.json
    experiments/analysis/bootstrap_summary.md
"""

import os
import json
import random
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR    = os.path.dirname(os.path.abspath(__file__))
N_BOOTSTRAP     = 10_000
RANDOM_SEED     = 42

# Map short label → jsonl filename (publication-scale n=499-500 only)
EXPERIMENT_FILES = {
    "GPT-4o\nguideSupp":
        "results_CMS125_csv_guideline_supplied_openai-gpt-4o_20260226_015346.jsonl",
    "GPT-4o\nzeroShot":
        "results_CMS125_csv_zero_shot_base_openai-gpt-4o_20260226_082257.jsonl",
    "Llama 3.3 70B\nguideSupp":
        "results_CMS125_csv_guideline_supplied_meta-llama-llama-3.3-70b-instruct_20260226_005952.jsonl",
    "Qwen 3 80B\nguideSupp":
        "results_CMS125_csv_guideline_supplied_qwen-qwen3-next-80b-a3b-instruct_20260226_011631.jsonl",
    "Llama 3.3 70B\nzeroShot":
        "results_CMS125_csv_zero_shot_base_llama-3.3-70b_20260226_002453.jsonl",
    "Qwen 3 80B\nzeroShot":
        "results_CMS125_csv_zero_shot_base_qwen-qwen3-next-80b-a3b-instruct_20260226_011620.jsonl",
}


def load_results(jsonl_path: str):
    """Load per-patient results from a .jsonl file."""
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def get_fields(r):
    """Extract gold/pred numerator and auditability from a record."""
    gold_num = bool(r.get("gold_truth", {}).get("numerator", False))
    parsed   = r.get("llm_prediction", {}).get("parsed") or {}
    pred_num = bool(parsed.get("numerator_met", False))

    # Auditability: LLM cited a non-null, non-'None' evidence
    gold_ev   = r.get("gold_truth", {}).get("evidence")          # dict or None
    llm_ev    = parsed.get("audit_evidence", "") or ""           # string or dict
    llm_ev_s  = str(llm_ev).strip().lower()

    # If gold has no evidence (non-compliant) and LLM says None → auditability OK
    if gold_ev is None:
        audit = (llm_ev_s in ("", "none", "null", "n/a"))
    else:
        # Gold has evidence → LLM must cite something meaningful (not None)
        audit = llm_ev_s not in ("", "none", "null", "n/a")

    return gold_num, pred_num, audit


def compute_metrics(records):
    """Compute precision, recall, F1, auditability from a list of records."""
    tp = fp = fn = 0
    audit_match = 0
    total = len(records)

    for r in records:
        gold_num, pred_num, audit = get_fields(r)

        if gold_num and pred_num:       tp += 1
        elif not gold_num and pred_num: fp += 1
        elif gold_num and not pred_num: fn += 1

        if audit:
            audit_match += 1

    precision    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1           = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auditability = audit_match / total if total > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "auditability": auditability}


def bootstrap_ci(records, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Bootstrap confidence intervals for precision, recall, F1, auditability."""
    rng = random.Random(seed)
    n   = len(records)

    metrics_boot = defaultdict(list)
    for _ in range(n_bootstrap):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        m = compute_metrics(sample)
        for k, v in m.items():
            metrics_boot[k].append(v)

    result = {}
    observed = compute_metrics(records)
    for k in ["precision", "recall", "f1", "auditability"]:
        vals = sorted(metrics_boot[k])
        result[k] = {
            "observed": round(observed[k] * 100, 2),
            "ci_lower": round(vals[int(0.025 * n_bootstrap)] * 100, 2),
            "ci_upper": round(vals[int(0.975 * n_bootstrap)] * 100, 2),
        }
    return result


def main():
    random.seed(RANDOM_SEED)
    all_results = {}

    for label, filename in EXPERIMENT_FILES.items():
        path = os.path.join(EXPERIMENTS_DIR, filename)
        if not os.path.exists(path):
            logging.warning(f"File not found: {filename}")
            continue
        logging.info(f"Bootstrapping: {filename}")
        records = load_results(path)
        logging.info(f"  Loaded {len(records)} patient records")
        ci = bootstrap_ci(records)
        all_results[label] = {"n": len(records), "metrics": ci}
        for metric, vals in ci.items():
            logging.info(
                f"  {metric:12s}: {vals['observed']:6.2f}% "
                f"[{vals['ci_lower']:.2f}%, {vals['ci_upper']:.2f}%]"
            )

    # Save JSON
    out_json = os.path.join(ANALYSIS_DIR, "bootstrap_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Saved: {out_json}")

    # Save Markdown summary
    out_md = os.path.join(ANALYSIS_DIR, "bootstrap_summary.md")
    with open(out_md, "w") as f:
        f.write("# Bootstrap 95% Confidence Intervals — CMS125 (n=10,000 resamples)\n\n")
        f.write("All publication-scale experiments (n=499–500 patients).\n\n")

        for label, data in all_results.items():
            clean_label = label.replace("\n", " ")
            f.write(f"## {clean_label} (n={data['n']})\n\n")
            f.write("| Metric | Observed | 95% CI Lower | 95% CI Upper |\n")
            f.write("|---|---|---|---|\n")
            for metric, vals in data["metrics"].items():
                f.write(f"| {metric.title()} | **{vals['observed']}%** | {vals['ci_lower']}% | {vals['ci_upper']}% |\n")
            f.write("\n")

    logging.info(f"Saved: {out_md}")
    logging.info("Bootstrap analysis complete.")


if __name__ == "__main__":
    main()
