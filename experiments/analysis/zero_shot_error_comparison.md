# Zero-Shot Error Comparison — CMS125

| Metric | Qwen 3 80B | GPT-4o |
|---|---|---|
| N total | 500 | 500 |
| Non-auditable | 282 (56.4%) | 237 (47.4%) |
| False positives | 269 | 207 |
| False negatives | 13 | 30 |

## Error Type Counts

| Error Type | Qwen 3 80B | GPT-4o |
|---|---|---|
| NO_EVIDENCE_CITED | 13 | 30 |
| OUTSIDE_WINDOW | 7 | 86 |
| WRONG_CONCLUSION | 262 | 121 |
