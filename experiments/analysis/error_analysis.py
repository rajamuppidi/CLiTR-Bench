"""
Error Analysis — Zero-Shot Hallucinations on CMS125.
Analyses non-auditable (hallucinated) evidence citations across zero-shot runs
to categorise temporal reasoning failure modes.

Models analysed:
  - Qwen 3 80B (zero-shot)
  - GPT-4o (zero-shot)

Categories:
  - OUTSIDE_WINDOW    : Cited a real mammogram but outside the 27-month window
  - WRONG_CONCLUSION  : Correct evidence date but incorrect compliance decision
  - NO_EVIDENCE_CITED : Prediction made without citing any evidence
  - FABRICATED_DETAILS: Evidence details don't parse to a recognisable date

Usage:
    python3 experiments/analysis/error_analysis.py

Output:
    experiments/analysis/<model>_zero_shot_error_analysis.json / .md
    experiments/analysis/zero_shot_error_comparison.md
"""

import os
import json
import logging
from datetime import datetime, date
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR    = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "qwen_zero_shot": {
        "label": "Qwen 3 80B",
        "file":  "results_CMS125_csv_zero_shot_base_qwen-qwen3-next-80b-a3b-instruct_20260226_011620.jsonl",
    },
    "gpt4o_zero_shot": {
        "label": "GPT-4o",
        "file":  "results_CMS125_csv_zero_shot_base_openai-gpt-4o_20260226_082257.jsonl",
    },
}

DESCRIPTIONS = {
    "OUTSIDE_WINDOW":     "Cited a real mammogram outside the 27-month window",
    "FABRICATED_DETAILS": "Cited event details that do not resolve to a recognisable date",
    "WRONG_CONCLUSION":   "Evidence date parses within the window; compliance decision is still wrong",
    "FUTURE_DATE":        "Cited a mammogram date after the measurement index date",
    "NO_EVIDENCE_CITED":  "Compliance prediction made without citing any supporting evidence",
}

MEASUREMENT_END     = date(2025, 12, 31)
LOOKBACK_DAYS       = 821   # 27 months
MAMMOGRAPHY_CODE    = "71651007"  # SNOMED — Synthea's only code


def parse_date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def parse_record(r):
    """Extract fields from a JSONL record."""
    pid      = r.get("patient_id", "")
    gold     = r.get("gold_truth", {})
    gold_num = bool(gold.get("numerator", False))
    gold_ev  = gold.get("evidence")   # dict or None

    parsed   = r.get("llm_prediction", {}).get("parsed") or {}
    pred_num = bool(parsed.get("numerator_met", False))
    llm_ev   = parsed.get("audit_evidence", "")
    llm_ev_s = str(llm_ev).strip().lower() if llm_ev else ""

    # Auditability: evidence claim matches ground truth
    if gold_ev is None:
        audit = (llm_ev_s in ("", "none", "null", "n/a"))
    else:
        audit = llm_ev_s not in ("", "none", "null", "n/a")

    return pid, gold_num, pred_num, gold_ev, llm_ev, audit


def classify_hallucination(gold_num, pred_num, gold_ev, llm_ev, audit):
    """Determine error type for a non-auditable case."""
    if audit:
        return "CORRECT"

    llm_ev_s = str(llm_ev).strip().lower() if llm_ev else ""

    if not llm_ev or llm_ev_s in ("", "none", "null", "n/a"):
        return "NO_EVIDENCE_CITED"

    # Try to parse as date from the string value
    llm_date = parse_date(str(llm_ev)) if isinstance(llm_ev, str) else None
    if not llm_date and isinstance(llm_ev, dict):
        llm_date = parse_date(llm_ev.get("event_date") or llm_ev.get("date", ""))

    if llm_date:
        days_diff = (MEASUREMENT_END - llm_date).days
        if days_diff < 0:
            return "FUTURE_DATE"
        elif days_diff > LOOKBACK_DAYS:
            return "OUTSIDE_WINDOW"
        else:
            return "WRONG_CONCLUSION"

    return "FABRICATED_DETAILS"




def analyse_model(path, label, model_key):
    """Run error analysis for one model's zero-shot JSONL file."""
    logging.info(f"\n{'='*60}\nAnalysing: {label}\n{'='*60}")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    parsed_records = [parse_record(r) for r in records]
    hallucinated   = [(pid, gn, pn, ge, le, a) for pid, gn, pn, ge, le, a in parsed_records if not a]
    n_hall  = len(hallucinated)
    n_total = len(records)

    logging.info(f"  Total: {n_total} | Non-auditable: {n_hall} ({n_hall/n_total*100:.1f}%)")

    error_counts = Counter()
    categorised  = []
    for pid, gn, pn, ge, le, a in hallucinated:
        cat = classify_hallucination(gn, pn, ge, le, a)
        error_counts[cat] += 1
        categorised.append({"patient_id": pid, "gold_numerator": gn,
                            "llm_numerator": pn, "error_type": cat,
                            "llm_evidence": str(le)})

    fp_hall = sum(1 for _, gn, pn, _, _, _ in hallucinated if not gn and pn)
    fn_hall = sum(1 for _, gn, pn, _, _, _ in hallucinated if gn and not pn)

    logging.info("  Error types:")
    for cat, count in error_counts.most_common():
        logging.info(f"    {cat:30s}: {count:4d} ({count/n_hall*100:.1f}%)")

    result = {
        "model": label, "prompt": "zero_shot_base",
        "n_total": n_total, "n_hallucinated": n_hall,
        "hallucination_rate_pct": round(n_hall / n_total * 100, 2),
        "error_type_counts": dict(error_counts.most_common()),
        "decision_breakdown": {
            "false_positive_wrong_compliant_call": fp_hall,
            "false_negative_wrong_noncompliant_call": fn_hall,
        },
        "cases": categorised,
    }

    out_json = os.path.join(ANALYSIS_DIR, f"{model_key}_error_analysis.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    out_md = os.path.join(ANALYSIS_DIR, f"{model_key}_error_analysis.md")
    with open(out_md, "w") as f:
        f.write(f"# {label} Zero-Shot Error Analysis — CMS125\n\n")
        f.write(f"**Total patients:** {n_total} | **Non-auditable:** {n_hall} ({result['hallucination_rate_pct']}%)\n\n")
        f.write("## Error Type Breakdown\n\n")
        f.write("| Error Type | Count | % of Non-auditable | Description |\n|---|---|---|---|\n")
        for cat, count in error_counts.most_common():
            pct  = round(count / n_hall * 100, 1)
            desc = DESCRIPTIONS.get(cat, "—")
            f.write(f"| {cat} | {count} | {pct}% | {desc} |\n")
        f.write(f"\n**False positives (missed care gaps):** {fp_hall}/{n_hall} ({fp_hall/n_hall*100:.1f}%)\n")
        f.write(f"**False negatives (unnecessary outreach):** {fn_hall}/{n_hall} ({fn_hall/n_hall*100:.1f}%)\n")

    logging.info(f"  Saved: {out_json} | {out_md}")
    return result


def main():
    all_results = {}
    for model_key, cfg in MODELS.items():
        path = os.path.join(EXPERIMENTS_DIR, cfg["file"])
        if not os.path.exists(path):
            logging.warning(f"File not found: {cfg['file']}")
            continue
        all_results[model_key] = analyse_model(path, cfg["label"], model_key)

    # Cross-model comparison
    out_cmp = os.path.join(ANALYSIS_DIR, "zero_shot_error_comparison.md")
    with open(out_cmp, "w") as f:
        f.write("# Zero-Shot Error Comparison — CMS125\n\n")
        f.write("| Metric | " + " | ".join(r["model"] for r in all_results.values()) + " |\n")
        f.write("|---|" + "---|" * len(all_results) + "\n")
        f.write("| N total | " + " | ".join(str(r["n_total"]) for r in all_results.values()) + " |\n")
        f.write("| Non-auditable | " + " | ".join(f"{r['n_hallucinated']} ({r['hallucination_rate_pct']}%)" for r in all_results.values()) + " |\n")
        f.write("| False positives | " + " | ".join(str(r["decision_breakdown"]["false_positive_wrong_compliant_call"]) for r in all_results.values()) + " |\n")
        f.write("| False negatives | " + " | ".join(str(r["decision_breakdown"]["false_negative_wrong_noncompliant_call"]) for r in all_results.values()) + " |\n\n")

        all_cats = sorted(set(k for r in all_results.values() for k in r["error_type_counts"]))
        f.write("## Error Type Counts\n\n| Error Type | " + " | ".join(r["model"] for r in all_results.values()) + " |\n")
        f.write("|---|" + "---|" * len(all_results) + "\n")
        for cat in all_cats:
            counts = [str(r["error_type_counts"].get(cat, 0)) for r in all_results.values()]
            f.write(f"| {cat} | " + " | ".join(counts) + " |\n")

    logging.info(f"\nComparison saved: {out_cmp}")
    logging.info("Error analysis complete.")


if __name__ == "__main__":
    main()



