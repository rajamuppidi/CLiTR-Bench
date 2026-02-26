"""
McNemar Paired Statistical Comparison for CLiTR-Bench CMS125.
Tests whether two models make *different* errors on the same patients.

H0: The two models make equivalent errors (b ≈ c in the contingency table).
H1: One model is statistically superior.

Usage:
    python3 experiments/analysis/mcnemar_test.py

Output:
    experiments/analysis/mcnemar_results.json
    experiments/analysis/mcnemar_summary.md
"""

import os
import json
import logging
from scipy.stats import chi2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR    = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_FILES = {
    "GPT-4o (guideline)":
        "results_CMS125_csv_guideline_supplied_openai-gpt-4o_20260226_015346.jsonl",
    "GPT-4o (zero-shot)":
        "results_CMS125_csv_zero_shot_base_openai-gpt-4o_20260226_082257.jsonl",
    "Llama 3.3 70B (guideline)":
        "results_CMS125_csv_guideline_supplied_meta-llama-llama-3.3-70b-instruct_20260226_005952.jsonl",
    "Qwen 3 80B (guideline)":
        "results_CMS125_csv_guideline_supplied_qwen-qwen3-next-80b-a3b-instruct_20260226_011631.jsonl",
    "Llama 3.3 70B (zero-shot)":
        "results_CMS125_csv_zero_shot_base_llama-3.3-70b_20260226_002453.jsonl",
    "Qwen 3 80B (zero-shot)":
        "results_CMS125_csv_zero_shot_base_qwen-qwen3-next-80b-a3b-instruct_20260226_011620.jsonl",
}

COMPARISONS = [
    # Guideline-supplied: model vs model
    ("GPT-4o (guideline)", "Llama 3.3 70B (guideline)"),
    ("GPT-4o (guideline)", "Qwen 3 80B (guideline)"),
    ("Llama 3.3 70B (guideline)", "Qwen 3 80B (guideline)"),
    # Prompt effect: guideline vs zero-shot (within model)
    ("GPT-4o (guideline)", "GPT-4o (zero-shot)"),
    ("Llama 3.3 70B (guideline)", "Llama 3.3 70B (zero-shot)"),
    ("Qwen 3 80B (guideline)", "Qwen 3 80B (zero-shot)"),
    # Zero-shot: model vs model
    ("GPT-4o (zero-shot)", "Llama 3.3 70B (zero-shot)"),
    ("GPT-4o (zero-shot)", "Qwen 3 80B (zero-shot)"),
]


def load_results(jsonl_path: str):
    """Returns dict of patient_id -> (gold, pred, correct)."""
    records = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r    = json.loads(line)
                pid  = r.get("patient_id", "")
                gold = bool(r.get("gold_truth", {}).get("numerator", False))
                parsed = r.get("llm_prediction", {}).get("parsed") or {}
                pred = bool(parsed.get("numerator_met", False))
                records[pid] = {
                    "gold":    gold,
                    "pred":    pred,
                    "correct": (gold == pred),
                }
            except json.JSONDecodeError:
                pass
    return records


def mcnemar_test(results_a, results_b, label_a, label_b):
    """
    McNemar test on shared patients.
    b = A correct, B incorrect (A superior cases)
    c = A incorrect, B correct (B superior cases)
    """
    shared_pids = set(results_a) & set(results_b)
    logging.info(f"  Shared patients for {label_a} vs {label_b}: {len(shared_pids)}")

    b = 0   # A correct, B wrong
    c = 0   # A wrong, B correct
    both_correct   = 0
    both_incorrect = 0

    for pid in shared_pids:
        a_ok = results_a[pid]["correct"]
        b_ok = results_b[pid]["correct"]
        if a_ok and not b_ok:     b += 1
        elif not a_ok and b_ok:   c += 1
        elif a_ok and b_ok:       both_correct   += 1
        else:                     both_incorrect += 1

    # McNemar with continuity correction (Yates)
    n_discordant = b + c
    if n_discordant == 0:
        chi2_stat, p_value = 0.0, 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    winner = label_a if b > c else (label_b if c > b else "Tie")
    significance = "p < 0.001" if p_value < 0.001 else (
        "p < 0.01" if p_value < 0.01 else (
        "p < 0.05" if p_value < 0.05 else "not significant"))

    return {
        "shared_patients": len(shared_pids),
        "a_better_b_worse": b,
        "b_better_a_worse": c,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "significance": significance,
        "winner": winner,
    }


def main():
    # Load all files
    loaded = {}
    for label, filename in EXPERIMENT_FILES.items():
        path = os.path.join(EXPERIMENTS_DIR, filename)
        if not os.path.exists(path):
            logging.warning(f"File not found: {filename}")
            continue
        loaded[label] = load_results(path)
        logging.info(f"Loaded {len(loaded[label])} patients for: {label}")

    all_results = {}
    for label_a, label_b in COMPARISONS:
        if label_a not in loaded or label_b not in loaded:
            logging.warning(f"Skipping: {label_a} vs {label_b} — file(s) missing")
            continue
        logging.info(f"\nMcNemar: {label_a} vs {label_b}")
        result = mcnemar_test(loaded[label_a], loaded[label_b], label_a, label_b)
        key = f"{label_a} vs {label_b}"
        all_results[key] = result
        logging.info(f"  χ²={result['chi2_stat']:.4f}  p={result['p_value']:.6f}  ({result['significance']})")
        logging.info(f"  Winner: {result['winner']}")
        logging.info(f"  Discordant pairs: {label_a} better={result['a_better_b_worse']}, {label_b} better={result['b_better_a_worse']}")

    # Save JSON
    out_json = os.path.join(ANALYSIS_DIR, "mcnemar_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nSaved: {out_json}")

    # Save Markdown
    out_md = os.path.join(ANALYSIS_DIR, "mcnemar_summary.md")
    with open(out_md, "w") as f:
        f.write("# McNemar Paired Model Comparison — CMS125\n\n")
        f.write("McNemar test with Yates continuity correction. H0: equivalent error rates.\n\n")

        for comparison, r in all_results.items():
            parts = comparison.split(" vs ")
            f.write(f"## {parts[0]} vs {parts[1]}\n\n")
            f.write(f"| Statistic | Value |\n|---|---|\n")
            f.write(f"| Shared patients | {r['shared_patients']} |\n")
            f.write(f"| {parts[0]} better | {r['a_better_b_worse']} cases |\n")
            f.write(f"| {parts[1]} better | {r['b_better_a_worse']} cases |\n")
            f.write(f"| Both correct | {r['both_correct']} |\n")
            f.write(f"| Both incorrect | {r['both_incorrect']} |\n")
            f.write(f"| χ² statistic | {r['chi2_stat']} |\n")
            f.write(f"| p-value | {r['p_value']} |\n")
            f.write(f"| **Result** | **{r['significance']}** |\n")
            f.write(f"| **Winner** | **{r['winner']}** |\n\n")

    logging.info(f"Saved: {out_md}")
    logging.info("McNemar analysis complete.")


if __name__ == "__main__":
    main()
