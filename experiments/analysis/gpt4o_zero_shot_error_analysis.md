# GPT-4o Zero-Shot Error Analysis — CMS125

**Total patients:** 500 | **Non-auditable:** 246 (49.2%)

## Error Type Breakdown

| Error Type | Count | % of Non-auditable | Description |
|---|---|---|---|
| WRONG_CONCLUSION | 125 | 50.8% | Evidence date parses within the window; compliance decision is still wrong |
| OUTSIDE_WINDOW | 85 | 34.6% | Cited a real mammogram outside the 27-month window |
| NO_EVIDENCE_CITED | 35 | 14.2% | Compliance prediction made without citing any supporting evidence |
| FUTURE_DATE | 1 | 0.4% | Cited a mammogram date after the measurement index date |

**False positives (missed care gaps):** 211/246 (85.8%)
**False negatives (unnecessary outreach):** 35/246 (14.2%)
