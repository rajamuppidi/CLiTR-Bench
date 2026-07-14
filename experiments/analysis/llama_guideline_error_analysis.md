# Llama 3.3 70B (guideline) Error Analysis — CMS125

**Total patients:** 500 | **AMR:** 89.8% | **Evidence Grounding Rate:** 89.4% | **AMR failures:** 51

## Failure Type Breakdown (grounded against the real record)

| Error Type | Count | % of failures | Description |
|---|---|---|---|
| FABRICATED_EVENT | 39 | 76.5% | Cited date(s) that exist nowhere in the patient's record |
| NO_EVIDENCE_CITED | 12 | 23.5% | Gold-compliant, but the model cited nothing (missed evidence) |
