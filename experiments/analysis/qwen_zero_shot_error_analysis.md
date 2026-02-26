# Qwen 3 80B Zero-Shot Error Analysis — CMS125

**Total patients:** 500 | **Hallucinated evidence:** 282 (56.4%)

## Error Type Breakdown

| Error Type | Count | % of Hallucinations | Description |
|---|---|---|---|
| WRONG_CONCLUSION | 262 | 92.9% | Correct evidence date but made the wrong compliance decision |
| NO_EVIDENCE_CITED | 13 | 4.6% | No evidence cited despite making a prediction |
| OUTSIDE_WINDOW | 7 | 2.5% | Cited a real mammogram event but it's older than 27 months |

## Decision Analysis Within Hallucinated Cases

| Outcome | Count | Implication |
|---|---|---|
| True Positive (correct call, wrong evidence) | 0 | Model got lucky — right answer, fabricated justification |
| False Positive (falsely marked compliant) | 269 | **Clinically dangerous** — would miss real care gaps |
| False Negative (falsely marked non-compliant) | 13 | Missed true compliance — causes unnecessary outreach |
| True Negative (correct call, wrong evidence) | 0 | Right answer, fabricated justification |

## Key Insight

Qwen zero-shot's dominant failure mode under temporal compliance tasks:

1. **It finds mammography events** (high recall) but cannot determine if they fall inside the 27-month boundary.
2. The 27-month lookback window (Oct 1, 2023 → Dec 31, 2025) is a non-trivial calculation that requires knowing the exact index date and computing a rolling window — capability that emerges only with guideline injection.
3. Most hallucinations are **OUTSIDE_WINDOW** temporal errors, not fabricated events — the model is reporting real mammograms in the wrong time frame.
