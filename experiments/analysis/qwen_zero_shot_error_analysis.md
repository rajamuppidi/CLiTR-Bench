# Qwen 3 80B Zero-Shot Error Analysis — CMS125

**Total patients:** 500 | **Non-auditable:** 282 (56.4%)

## Error Type Breakdown

| Error Type | Count | % of Non-auditable | Description |
|---|---|---|---|
| WRONG_CONCLUSION | 262 | 92.9% | Evidence date parses within the window; compliance decision is still wrong |
| NO_EVIDENCE_CITED | 13 | 4.6% | Compliance prediction made without citing any supporting evidence |
| OUTSIDE_WINDOW | 7 | 2.5% | Cited a real mammogram outside the 27-month window |

**False positives (missed care gaps):** 269/282 (95.4%)
**False negatives (unnecessary outreach):** 13/282 (4.6%)
