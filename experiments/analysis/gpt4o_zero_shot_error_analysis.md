# GPT-4o Zero-Shot Error Analysis — CMS125

**Total patients:** 500 | **Non-auditable:** 237 (47.4%)

## Error Type Breakdown

| Error Type | Count | % of Non-auditable | Description |
|---|---|---|---|
| WRONG_CONCLUSION | 121 | 51.1% | Evidence date parses within the window; compliance decision is still wrong |
| OUTSIDE_WINDOW | 86 | 36.3% | Cited a real mammogram outside the 27-month window |
| NO_EVIDENCE_CITED | 30 | 12.7% | Compliance prediction made without citing any supporting evidence |

**False positives (missed care gaps):** 207/237 (87.3%)
**False negatives (unnecessary outreach):** 30/237 (12.7%)
