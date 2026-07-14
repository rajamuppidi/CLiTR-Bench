# GPT-4o (zero-shot) Error Analysis — CMS125

**Total patients:** 500 | **AMR:** 34.4% | **Evidence Grounding Rate:** 35.4% | **AMR failures:** 328

## Failure Type Breakdown (grounded against the real record)

| Error Type | Count | % of failures | Description |
|---|---|---|---|
| FABRICATED_EVENT | 288 | 87.8% | Cited date(s) that exist nowhere in the patient's record |
| NO_EVIDENCE_CITED | 35 | 10.7% | Gold-compliant, but the model cited nothing (missed evidence) |
| GROUNDED_ALTERNATE_IN_WINDOW | 3 | 0.9% | Cited a real in-window mammogram other than the gold most-recent event |
| GROUNDED_OUT_OF_WINDOW | 2 | 0.6% | Cited a real mammogram dated outside the 27-month window |
