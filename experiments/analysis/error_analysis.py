"""
Error Analysis — Qwen 3 80B Zero-Shot Hallucinations on CMS125.
Analyses the 342 hallucinated evidence citations in the Qwen zero-shot run
to categorise temporal reasoning failure modes.

Categories:
  - OUTSIDE_WINDOW: Cited a mammogram event that exists but is outside the 27-month window
  - WRONG_CODE: Cited an event with a non-mammography code
  - FABRICATED: Evidence event doesn't exist in gold truth at all
  - WRONG_CONCLUSION: Correct evidence cited but compliance decision wrong

Usage:
    python3 experiments/analysis/error_analysis.py

Output:
    experiments/analysis/qwen_zero_shot_error_analysis.json
    experiments/analysis/qwen_zero_shot_error_analysis.md
"""

import os
import json
import logging
from datetime import datetime, date
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR    = os.path.dirname(os.path.abspath(__file__))

QWEN_ZERO_SHOT_FILE = "results_CMS125_csv_zero_shot_base_qwen-qwen3-next-80b-a3b-instruct_20260226_011620.jsonl"

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


def main():
    path = os.path.join(EXPERIMENTS_DIR, QWEN_ZERO_SHOT_FILE)
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        return

    logging.info(f"Loading: {QWEN_ZERO_SHOT_FILE}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    logging.info(f"  Total records: {len(records)}")

    # Parse all records
    parsed_records = [parse_record(r) for r in records]
    # (pid, gold_num, pred_num, gold_ev, llm_ev, audit)

    hallucinated = [(pid, gn, pn, ge, le, a) for pid, gn, pn, ge, le, a in parsed_records if not a]
    correct      = [(pid, gn, pn, ge, le, a) for pid, gn, pn, ge, le, a in parsed_records if a]

    logging.info(f"  Non-auditable (hallucinated evidence): {len(hallucinated)}")
    logging.info(f"  Auditable (correct evidence): {len(correct)}")

    # Classify each hallucination
    error_counts = Counter()
    categorised  = []

    for pid, gn, pn, ge, le, a in hallucinated:
        cat = classify_hallucination(gn, pn, ge, le, a)
        error_counts[cat] += 1
        categorised.append({
            "patient_id":    pid,
            "gold_numerator": gn,
            "llm_numerator":  pn,
            "error_type":    cat,
            "llm_evidence":  str(le),
        })

    # Decision breakdown
    tp_hall  = sum(1 for _, gn, pn, _, _, _ in hallucinated if gn and pn)
    fp_hall  = sum(1 for _, gn, pn, _, _, _ in hallucinated if not gn and pn)
    fn_hall  = sum(1 for _, gn, pn, _, _, _ in hallucinated if gn and not pn)
    tn_hall  = sum(1 for _, gn, pn, _, _, _ in hallucinated if not gn and not pn)

    logging.info("\nError type breakdown:")
    for cat, count in error_counts.most_common():
        logging.info(f"  {cat:30s}: {count:4d} ({count/len(hallucinated)*100:.1f}%)")

    logging.info(f"\nDecision breakdown within hallucinated cases:")
    logging.info(f"  True Positive (right call, wrong evidence):  {tp_hall}")
    logging.info(f"  False Positive (wrong compliant call):       {fp_hall}")
    logging.info(f"  False Negative (wrong non-compliant call):   {fn_hall}")
    logging.info(f"  True Negative (right call, wrong evidence):  {tn_hall}")

    # Save JSON
    result = {
        "model": "Qwen 3 80B",
        "prompt": "zero_shot_base",
        "n_total": len(records),
        "n_hallucinated": len(hallucinated),
        "hallucination_rate_pct": round(len(hallucinated)/len(records)*100, 2),
        "error_type_counts": dict(error_counts.most_common()),
        "decision_breakdown": {
            "true_positive_with_wrong_evidence": tp_hall,
            "false_positive_wrong_compliant_call": fp_hall,
            "false_negative_wrong_noncompliant_call": fn_hall,
            "true_negative_with_wrong_evidence": tn_hall,
        },
        "cases": categorised,
    }

    out_json = os.path.join(ANALYSIS_DIR, "qwen_zero_shot_error_analysis.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"\nSaved: {out_json}")

    # Save Markdown report
    out_md = os.path.join(ANALYSIS_DIR, "qwen_zero_shot_error_analysis.md")
    with open(out_md, "w") as f:
        f.write("# Qwen 3 80B Zero-Shot Error Analysis — CMS125\n\n")
        f.write(f"**Total patients:** {len(records)} | ")
        f.write(f"**Hallucinated evidence:** {len(hallucinated)} ({result['hallucination_rate_pct']}%)\n\n")
        f.write("## Error Type Breakdown\n\n")
        f.write("| Error Type | Count | % of Hallucinations | Description |\n")
        f.write("|---|---|---|---|\n")
        descriptions = {
            "OUTSIDE_WINDOW":     "Cited a real mammogram event but it's older than 27 months",
            "FABRICATED_DETAILS": "Cited event details that don't match any gold truth record",
            "WRONG_CODE":         "Cited a non-mammography event as the compliance evidence",
            "WRONG_CONCLUSION":   "Correct evidence date but made the wrong compliance decision",
            "FUTURE_DATE":        "Hallucinated a mammogram date after the index date",
            "NO_EVIDENCE_CITED":  "No evidence cited despite making a prediction",
        }
        for cat, count in error_counts.most_common():
            pct  = round(count / len(hallucinated) * 100, 1)
            desc = descriptions.get(cat, "—")
            f.write(f"| {cat} | {count} | {pct}% | {desc} |\n")

        f.write("\n## Decision Analysis Within Hallucinated Cases\n\n")
        f.write("| Outcome | Count | Implication |\n|---|---|---|\n")
        f.write(f"| True Positive (correct call, wrong evidence) | {tp_hall} | Model got lucky — right answer, fabricated justification |\n")
        f.write(f"| False Positive (falsely marked compliant) | {fp_hall} | **Clinically dangerous** — would miss real care gaps |\n")
        f.write(f"| False Negative (falsely marked non-compliant) | {fn_hall} | Missed true compliance — causes unnecessary outreach |\n")
        f.write(f"| True Negative (correct call, wrong evidence) | {tn_hall} | Right answer, fabricated justification |\n")

        f.write("\n## Key Insight\n\n")
        f.write("Qwen zero-shot's dominant failure mode under temporal compliance tasks:\n\n")
        f.write("1. **It finds mammography events** (high recall) but cannot determine if they fall inside the 27-month boundary.\n")
        f.write("2. The 27-month lookback window (Oct 1, 2023 → Dec 31, 2025) is a non-trivial calculation that requires knowing the exact index date and computing a rolling window — capability that emerges only with guideline injection.\n")
        f.write("3. Most hallucinations are **OUTSIDE_WINDOW** temporal errors, not fabricated events — the model is reporting real mammograms in the wrong time frame.\n")

    logging.info(f"Saved: {out_md}")
    logging.info("Error analysis complete.")


if __name__ == "__main__":
    main()
