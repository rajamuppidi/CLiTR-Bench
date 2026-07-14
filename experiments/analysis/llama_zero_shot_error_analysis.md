# Llama 3.3 70B (zero-shot) Error Analysis — CMS125

**Total patients:** 500 | **AMR:** 60.6% | **Evidence Grounding Rate:** 60.8% | **AMR failures:** 197

## Failure Type Breakdown (grounded against the real record)

| Error Type | Count | % of failures | Description |
|---|---|---|---|
| FABRICATED_EVENT | 120 | 60.9% | Cited date(s) that exist nowhere in the patient's record |
| NO_EVIDENCE_CITED | 76 | 38.6% | Gold-compliant, but the model cited nothing (missed evidence) |
| GROUNDED_ALTERNATE_IN_WINDOW | 1 | 0.5% | Cited a real in-window mammogram other than the gold most-recent event |
