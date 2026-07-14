# Claude Sonnet 4.6 (zero-shot) Error Analysis — CMS125

**Total patients:** 500 | **AMR:** 65.4% | **Evidence Grounding Rate:** 65.4% | **AMR failures:** 173

## Failure Type Breakdown (grounded against the real record)

| Error Type | Count | % of failures | Description |
|---|---|---|---|
| FABRICATED_EVENT | 154 | 89.0% | Cited date(s) that exist nowhere in the patient's record |
| NO_EVIDENCE_CITED | 19 | 11.0% | Gold-compliant, but the model cited nothing (missed evidence) |
