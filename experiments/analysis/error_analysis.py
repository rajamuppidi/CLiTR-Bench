"""
Error Analysis — Evidence Grounding on CMS125.

Classifies every AMR-failing citation by checking it against the patient's REAL
mammography event history (from data_generation/output/canonical/events.csv),
so that fabricated citations are distinguished from real events cited with the
wrong window logic.

Auditability definition is IDENTICAL to evaluation/metrics_engine.py:
  - Gold-compliant patient: citation must contain the gold evidence event's
    date or code (the most recent in-window qualifying event).
  - Gold-non-compliant patient: the model must cite nothing.

On top of that, each AMR failure is grounded against the record:
  NO_EVIDENCE_CITED           : gold-compliant, but model cited nothing (missed evidence)
  NO_PARSEABLE_DATE           : citation contains no ISO date to check
  GROUNDED_ALTERNATE_IN_WINDOW: cited a REAL mammogram inside the window
                                (legitimate evidence, but not the gold most-recent event)
  GROUNDED_OUT_OF_WINDOW      : cited a REAL mammogram outside the 27-month window
  GROUNDED_POST_INDEX         : cited a REAL mammogram dated after the index date
  FABRICATED_EVENT            : cited date(s) exist nowhere in the patient's record

Usage:
    python3 experiments/analysis/error_analysis.py

Output:
    experiments/analysis/cohort_mammo_index.json   (cached record index, built once)
    experiments/analysis/<model>_error_analysis.json / .md
    experiments/analysis/grounding_comparison.md
"""

import os
import re
import csv
import json
import logging
from datetime import datetime, date
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR    = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR      = os.path.dirname(EXPERIMENTS_DIR)
CANONICAL_DIR   = os.path.join(TARGET_DIR, "data_generation", "output", "canonical")
COHORT_FILE     = os.path.join(EXPERIMENTS_DIR, "cohort_2025_cms125_500_final.txt")
MAMMO_INDEX     = os.path.join(ANALYSIS_DIR, "cohort_mammo_index.json")

MEASUREMENT_END = date(2025, 12, 31)
WINDOW_START    = date(2023, 10, 1)   # CMS125v13: Oct 1 two years prior to the measurement period

MAMMO_CODES = {"24606-6", "71651007", "77061", "77062", "77063", "77065", "77066", "77067"}
NULL_VALS   = ("", "none", "null", "n/a", "false")
DATE_RE     = re.compile(r"\d{4}-\d{2}-\d{2}")

MODELS = {
    "gpt4o_zero_shot": {
        "label": "GPT-4o (zero-shot)",
        "file":  "results_CMS125_csv_zero_shot_base_openai-gpt-4o_20260605_103217.jsonl",
    },
    "gpt4o_guideline": {
        "label": "GPT-4o (guideline)",
        "file":  "results_CMS125_csv_guideline_supplied_openai-gpt-4o_20260604_221440.jsonl",
    },
    "claude_zero_shot": {
        "label": "Claude Sonnet 4.6 (zero-shot)",
        "file":  "results_CMS125_csv_zero_shot_base_anthropic-claude-sonnet-4.6_20260605_113755.jsonl",
    },
    "claude_guideline": {
        "label": "Claude Sonnet 4.6 (guideline)",
        "file":  "results_CMS125_csv_guideline_supplied_anthropic-claude-sonnet-4.6_20260605_135639.jsonl",
    },
    "llama_zero_shot": {
        "label": "Llama 3.3 70B (zero-shot)",
        "file":  "results_CMS125_csv_zero_shot_base_meta-llama-llama-3.3-70b-instruct_20260605_154816.jsonl",
    },
    "llama_guideline": {
        "label": "Llama 3.3 70B (guideline)",
        "file":  "results_CMS125_csv_guideline_supplied_meta-llama-llama-3.3-70b-instruct_20260605_163208.jsonl",
    },
}

DESCRIPTIONS = {
    "NO_EVIDENCE_CITED":            "Gold-compliant, but the model cited nothing (missed evidence)",
    "NO_PARSEABLE_DATE":            "Citation contains no ISO date to check against the record",
    "GROUNDED_ALTERNATE_IN_WINDOW": "Cited a real in-window mammogram other than the gold most-recent event",
    "GROUNDED_OUT_OF_WINDOW":       "Cited a real mammogram dated outside the 27-month window",
    "GROUNDED_POST_INDEX":          "Cited a real mammogram dated after the measurement index date",
    "FABRICATED_EVENT":             "Cited date(s) that exist nowhere in the patient's record",
}


def parse_date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def build_mammo_index():
    """One pass over events.csv -> {patient_id: [mammo dates]} for the cohort. Cached."""
    if os.path.exists(MAMMO_INDEX):
        with open(MAMMO_INDEX) as f:
            return {k: set(v) for k, v in json.load(f).items()}

    cohort = set(x.strip() for x in open(COHORT_FILE) if x.strip())
    logging.info("Building mammography record index from events.csv (one-time, ~2 min)...")
    index = {}
    with open(os.path.join(CANONICAL_DIR, "events.csv")) as f:
        for row in csv.DictReader(f):
            if row["patient_id"] in cohort and row["code"] in MAMMO_CODES:
                index.setdefault(row["patient_id"], set()).add(row["event_date"][:10])

    with open(MAMMO_INDEX, "w") as f:
        json.dump({k: sorted(v) for k, v in index.items()}, f, indent=1)
    logging.info(f"Cached record index: {MAMMO_INDEX} ({len(index)} patients with mammograms)")
    return index


def amr_match(gold_ev, llm_ev_s):
    """Auditability match — identical to evaluation/metrics_engine.py."""
    if gold_ev is None:
        return llm_ev_s in NULL_VALS
    target_date = str(gold_ev.get("event_date", "")).lower()
    target_code = str(gold_ev.get("code", "")).lower()
    return bool((target_date and target_date in llm_ev_s) or
                (target_code and target_code in llm_ev_s))


def classify_failure(gold_ev, llm_ev_s, real_dates):
    """Ground an AMR-failing citation against the patient's real mammography record."""
    if llm_ev_s in NULL_VALS:
        # Only reachable when gold_ev is not None (gold-compliant patient).
        return "NO_EVIDENCE_CITED"

    cited = DATE_RE.findall(llm_ev_s)
    if not cited:
        return "NO_PARSEABLE_DATE"

    real_hits = [parse_date(c) for c in cited if c in real_dates]
    if not real_hits:
        return "FABRICATED_EVENT"
    if any(WINDOW_START <= d <= MEASUREMENT_END for d in real_hits):
        return "GROUNDED_ALTERNATE_IN_WINDOW"
    if any(d > MEASUREMENT_END for d in real_hits):
        return "GROUNDED_POST_INDEX"
    return "GROUNDED_OUT_OF_WINDOW"


def analyse_model(path, label, model_key, record_index):
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

    n_total = len(records)
    n_fail  = 0
    n_grounded_ok = 0  # citation correctly none, or contains >=1 real record event
    error_counts = Counter()
    cases = []
    fp_fail = fn_fail = 0

    for r in records:
        pid      = r.get("patient_id", "")
        gold     = r.get("gold_truth", {})
        gold_ev  = gold.get("evidence")
        gold_num = bool(gold.get("numerator", False))
        parsed   = r.get("llm_prediction", {}).get("parsed") or {}
        pred_num = bool(parsed.get("numerator_met", False))
        llm_ev_s = str(parsed.get("audit_evidence", "") or "").strip().lower()
        real     = record_index.get(pid, set())

        # Evidence Grounding Rate: is the citation consistent with the record at all?
        cited = DATE_RE.findall(llm_ev_s)
        if llm_ev_s in NULL_VALS:
            # A "none" citation is grounded only when there is genuinely nothing to cite.
            grounded = gold_ev is None
        else:
            grounded = any(c in real for c in cited)
        if grounded:
            n_grounded_ok += 1

        if amr_match(gold_ev, llm_ev_s):
            continue

        n_fail += 1
        cat = classify_failure(gold_ev, llm_ev_s, real)
        error_counts[cat] += 1
        if not gold_num and pred_num: fp_fail += 1
        if gold_num and not pred_num: fn_fail += 1
        cases.append({"patient_id": pid, "gold_numerator": gold_num,
                      "llm_numerator": pred_num, "error_type": cat,
                      "llm_evidence": llm_ev_s})

    amr = (n_total - n_fail) / n_total * 100
    grounding_rate = n_grounded_ok / n_total * 100
    logging.info(f"  Total: {n_total} | AMR failures: {n_fail} ({n_fail/n_total*100:.1f}%) "
                 f"| AMR: {amr:.1f}% | Grounding rate: {grounding_rate:.1f}%")
    for cat, count in error_counts.most_common():
        logging.info(f"    {cat:30s}: {count:4d} ({count/max(n_fail,1)*100:.1f}%)")

    result = {
        "model": label,
        "n_total": n_total,
        "amr_pct": round(amr, 2),
        "evidence_grounding_rate_pct": round(grounding_rate, 2),
        "n_amr_failures": n_fail,
        "error_type_counts": dict(error_counts.most_common()),
        "decision_breakdown": {
            "false_positive_wrong_compliant_call": fp_fail,
            "false_negative_wrong_noncompliant_call": fn_fail,
        },
        "cases": cases,
    }

    with open(os.path.join(ANALYSIS_DIR, f"{model_key}_error_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(ANALYSIS_DIR, f"{model_key}_error_analysis.md"), "w") as f:
        f.write(f"# {label} Error Analysis — CMS125\n\n")
        f.write(f"**Total patients:** {n_total} | **AMR:** {result['amr_pct']}% | "
                f"**Evidence Grounding Rate:** {result['evidence_grounding_rate_pct']}% | "
                f"**AMR failures:** {n_fail}\n\n")
        f.write("## Failure Type Breakdown (grounded against the real record)\n\n")
        f.write("| Error Type | Count | % of failures | Description |\n|---|---|---|---|\n")
        for cat, count in error_counts.most_common():
            pct  = round(count / max(n_fail, 1) * 100, 1)
            f.write(f"| {cat} | {count} | {pct}% | {DESCRIPTIONS.get(cat, '—')} |\n")

    return result


def main():
    record_index = build_mammo_index()

    all_results = {}
    for model_key, cfg in MODELS.items():
        path = os.path.join(EXPERIMENTS_DIR, cfg["file"])
        if not os.path.exists(path):
            logging.warning(f"File not found: {cfg['file']}")
            continue
        all_results[model_key] = analyse_model(path, cfg["label"], model_key, record_index)

    out_cmp = os.path.join(ANALYSIS_DIR, "grounding_comparison.md")
    with open(out_cmp, "w") as f:
        f.write("# Evidence Grounding Comparison — CMS125\n\n")
        f.write("AMR failures classified against each patient's real mammography record.\n\n")
        f.write("| Metric | " + " | ".join(r["model"] for r in all_results.values()) + " |\n")
        f.write("|---|" + "---|" * len(all_results) + "\n")
        f.write("| AMR | " + " | ".join(f"{r['amr_pct']}%" for r in all_results.values()) + " |\n")
        f.write("| Evidence Grounding Rate | " + " | ".join(f"{r['evidence_grounding_rate_pct']}%" for r in all_results.values()) + " |\n")
        f.write("| AMR failures | " + " | ".join(str(r["n_amr_failures"]) for r in all_results.values()) + " |\n")

        all_cats = sorted(set(k for r in all_results.values() for k in r["error_type_counts"]))
        f.write("\n## Failure Type Counts\n\n| Error Type | " + " | ".join(r["model"] for r in all_results.values()) + " |\n")
        f.write("|---|" + "---|" * len(all_results) + "\n")
        for cat in all_cats:
            counts = [str(r["error_type_counts"].get(cat, 0)) for r in all_results.values()]
            f.write(f"| {cat} | " + " | ".join(counts) + " |\n")

    logging.info(f"\nComparison saved: {out_cmp}")
    logging.info("Error analysis complete.")


if __name__ == "__main__":
    main()
