# Evidence Grounding Comparison — CMS125

AMR failures classified against each patient's real mammography record.

| Metric | GPT-4o (zero-shot) | GPT-4o (guideline) | Claude Sonnet 4.6 (zero-shot) | Claude Sonnet 4.6 (guideline) | Llama 3.3 70B (zero-shot) | Llama 3.3 70B (guideline) |
|---|---|---|---|---|---|---|
| AMR | 34.4% | 98.4% | 65.4% | 94.0% | 60.6% | 89.8% |
| Evidence Grounding Rate | 35.4% | 98.4% | 65.4% | 94.0% | 60.8% | 89.4% |
| AMR failures | 328 | 8 | 173 | 30 | 197 | 51 |

## Failure Type Counts

| Error Type | GPT-4o (zero-shot) | GPT-4o (guideline) | Claude Sonnet 4.6 (zero-shot) | Claude Sonnet 4.6 (guideline) | Llama 3.3 70B (zero-shot) | Llama 3.3 70B (guideline) |
|---|---|---|---|---|---|---|
| FABRICATED_EVENT | 288 | 0 | 154 | 21 | 120 | 39 |
| GROUNDED_ALTERNATE_IN_WINDOW | 3 | 0 | 0 | 0 | 1 | 0 |
| GROUNDED_OUT_OF_WINDOW | 2 | 0 | 0 | 0 | 0 | 0 |
| NO_EVIDENCE_CITED | 35 | 8 | 19 | 9 | 76 | 12 |
