# Claude Sonnet 4.6 Zero-Shot Error Analysis — CMS125

**Total patients:** 500 | **Non-auditable:** 166 (33.2%)

## Error Type Breakdown

| Error Type | Count | % of Non-auditable | Description |
|---|---|---|---|
| WRONG_CONCLUSION | 129 | 77.7% | Evidence date parses within the window; compliance decision is still wrong |
| NO_EVIDENCE_CITED | 19 | 11.4% | Compliance prediction made without citing any supporting evidence |
| FABRICATED_DETAILS | 14 | 8.4% | Cited event details that do not resolve to a recognisable date |
| OUTSIDE_WINDOW | 4 | 2.4% | Cited a real mammogram outside the 27-month window |

**False positives (missed care gaps):** 134/166 (80.7%)
**False negatives (unnecessary outreach):** 19/166 (11.4%)
