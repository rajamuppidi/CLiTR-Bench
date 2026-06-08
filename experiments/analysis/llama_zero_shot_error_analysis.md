# Llama 3.3 70B Zero-Shot Error Analysis — CMS125

**Total patients:** 500 | **Non-auditable:** 148 (29.6%)

## Error Type Breakdown

| Error Type | Count | % of Non-auditable | Description |
|---|---|---|---|
| NO_EVIDENCE_CITED | 76 | 51.4% | Compliance prediction made without citing any supporting evidence |
| WRONG_CONCLUSION | 59 | 39.9% | Evidence date parses within the window; compliance decision is still wrong |
| OUTSIDE_WINDOW | 13 | 8.8% | Cited a real mammogram outside the 27-month window |

**False positives (missed care gaps):** 72/148 (48.6%)
**False negatives (unnecessary outreach):** 76/148 (51.4%)
