# Claude Sonnet 4.6 (guideline) Error Analysis — CMS125

**Total patients:** 500 | **AMR:** 94.0% | **Evidence Grounding Rate:** 94.0% | **AMR failures:** 30

## Failure Type Breakdown (grounded against the real record)

| Error Type | Count | % of failures | Description |
|---|---|---|---|
| FABRICATED_EVENT | 21 | 70.0% | Cited date(s) that exist nowhere in the patient's record |
| NO_EVIDENCE_CITED | 9 | 30.0% | Gold-compliant, but the model cited nothing (missed evidence) |
