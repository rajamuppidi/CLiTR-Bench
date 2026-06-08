# Zero-Shot Error Comparison — CMS125

| Metric | GPT-4o | Claude Sonnet 4.6 | Llama 3.3 70B |
|---|---|---|---|
| N total | 500 | 500 | 500 |
| Non-auditable | 246 (49.2%) | 166 (33.2%) | 148 (29.6%) |
| False positives | 211 | 134 | 72 |
| False negatives | 35 | 19 | 76 |

## Error Type Counts

| Error Type | GPT-4o | Claude Sonnet 4.6 | Llama 3.3 70B |
|---|---|---|---|
| FABRICATED_DETAILS | 0 | 14 | 0 |
| FUTURE_DATE | 1 | 0 | 0 |
| NO_EVIDENCE_CITED | 35 | 19 | 76 |
| OUTSIDE_WINDOW | 85 | 4 | 13 |
| WRONG_CONCLUSION | 125 | 129 | 59 |
