# CLiTR-Bench

**CLiTR-Bench** (Clinical Temporal Reasoning Benchmark) is a fully reproducible benchmark for evaluating large language model (LLM) performance on electronic clinical quality measure (eCQM) temporal compliance reasoning.

The benchmark is instantiated on **CMS125v13 Breast Cancer Screening**, targeting the 27-month bilateral mammography lookback window, a representative eCQM that requires rolling-window date arithmetic, longitudinal event disambiguation, and auditable evidence citation.

> **Paper:** *CLiTR-Bench: A Benchmark for Large Language Model Temporal Reasoning on Electronic Clinical Quality Measures* (manuscript in preparation).

---

## What this benchmark measures

Deciding whether a patient meets an eCQM is not a knowledge-recall task. A system must confirm eligibility, find a qualifying event inside a fixed time window, apply exclusion logic, and cite the exact record event that justifies its decision. CLiTR-Bench evaluates an LLM against a **deterministic gold truth engine** built directly from the CMS125v13 specification, and introduces the **Auditability Match Rate (AMR)**, a metric that checks whether the evidence a model cites is actually present in the patient record, separately from whether its final yes-or-no answer is correct.

---

## Key Results (CMS125v13, n = 500 patients)

Three frontier models were evaluated under two prompt conditions: zero-shot (no specification) and guideline-supplied (full CMS125v13 specification in the prompt). Every configuration produced a valid prediction for all 500 patients (100% format compliance).

| Model | Prompt | F1 [95% CI] | Precision | Recall | AMR | Hallucinations |
|---|---|---|---|---|---|---|
| **GPT-4o** | Guideline-supplied | **97.20** [95.07, 98.94] | **100.0** | 94.56 | **98.4** | **8** |
| Claude Sonnet 4.6 | Guideline-supplied | 90.79 [87.18, 93.94] | 87.90 | 93.88 | 94.0 | 30 |
| Llama 3.3 70B | Guideline-supplied | 84.11 [79.54, 88.07] | 77.59 | 91.84 | 89.8 | 51 |
| Claude Sonnet 4.6 | Zero-shot | 59.85 [53.98, 65.20] | 47.24 | 81.63 | 66.8 | 173 |
| Llama 3.3 70B | Zero-shot | 48.97 [41.77, 55.91] | 49.65 | 48.30 | 70.4 | 197 |
| GPT-4o | Zero-shot | 47.66 [41.96, 53.00] | 34.67 | 76.19 | 50.8 | 328 |

All values are percentages except the hallucination count. F1 confidence intervals are from 10,000 bootstrap resamples.

**Headline findings:**

1. **Guideline supplementation is decisive.** Supplying the specification improved every model, and the gain was largest for the weakest zero-shot performer: GPT-4o +49.5 F1, Llama 3.3 70B +35.1, Claude Sonnet 4.6 +30.9 (all p < 0.0001, Bonferroni-corrected).
2. **The ranking reverses between conditions.** Claude Sonnet 4.6 is the best zero-shot model (F1 59.9), significantly ahead of GPT-4o (47.7), yet GPT-4o is the clear leader once the specification is supplied (97.2). Reasoning about a measure and applying its written rules are separable abilities.
3. **Failure modes differ by model.** Without the guideline, GPT-4o miscalculates the temporal window (34.6% OUTSIDE_WINDOW errors), while Llama frequently cites no supporting event at all (51.4% NO_EVIDENCE_CITED).

---

## Repository Structure

```
CLiTR-Bench/
├── gold_truth_engine/          # Deterministic CMS125v13 gold truth implementation
│   ├── gold_truth_engine.py    # Core engine (initial pop, denominator, numerator, exclusions)
│   └── test_gold_truth_engine.py
│
├── llm_runner/                 # LLM inference via OpenRouter API
│   └── run_inference.py        # Hardened: JSON mode, token cap, degenerate-output retry
│
├── prompts/                    # Prompt templates (zero-shot base & guideline-supplied)
│
├── terminology/                # NCQA value sets (CMS125 mammography codes, SNOMED)
│
├── representations/            # Patient data serialization (CSV event table)
│
├── experiments/                # Experiment runner and frozen results
│   ├── run_experiment.py       # Main experiment entrypoint
│   ├── build_cms125_cohort.py  # Cohort builder (500-patient publication cohort)
│   ├── cohort_2025_cms125_500_final.txt   # Publication cohort (n=500, seed=99)
│   ├── final_scores_*.json     # Aggregate benchmark scores
│   ├── results_*.jsonl         # Per-patient inference logs
│   │
│   └── analysis/               # Statistical analysis scripts
│       ├── bootstrap_ci.py     # Bootstrap 95% confidence intervals (B=10,000)
│       ├── mcnemar_test.py     # McNemar paired significance tests
│       ├── error_analysis.py   # Zero-shot hallucination error taxonomy
│       ├── generate_figures.py # Publication figures (300 DPI)
│       ├── bootstrap_summary.md
│       └── mcnemar_summary.md
│
└── data_generation/            # Synthea setup and FHIR-to-events parsing
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install "httpx<0.28" openai scipy numpy matplotlib python-dotenv pyyaml
```

### 2. Configure API key

```bash
cp .env.example .env
# Add your OpenRouter API key to .env:
# OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Generate Synthea patient data

Download [Synthea™](https://github.com/synthetichealth/synthea) and generate patients:

```bash
# Generate 25,000 California patients
java -jar synthea-with-dependencies.jar -p 25000 California
```

### 4. Build the cohort

```bash
cd experiments
python3 build_cms125_cohort.py --size 500 --seed 99
```

### 5. Run an experiment

```bash
cd experiments
python3 run_experiment.py \
  --measure CMS125 \
  --measure-name "Breast Cancer Screening" \
  --format csv \
  --prompt guideline_supplied \
  --model "openai/gpt-4o" \
  --cohort "cohort_2025_cms125_500_final.txt"
```

Supported prompt styles: `zero_shot_base`, `guideline_supplied`. The same command works for `anthropic/claude-sonnet-4.6` and `meta-llama/llama-3.3-70b-instruct`.

### 6. Reproduce statistical analysis and figures

```bash
cd experiments
python3 analysis/bootstrap_ci.py      # Bootstrap 95% CIs
python3 analysis/mcnemar_test.py      # Paired significance tests
python3 analysis/error_analysis.py    # Zero-shot error taxonomy
python3 analysis/generate_figures.py  # Publication figures
```

You can also recompute aggregate scores from any saved results file without re-calling the API:

```bash
python3 evaluation/metrics_engine.py --rescore experiments/results_<run>.jsonl
```

---

## Benchmark Design

### Measure: CMS125v13 Breast Cancer Screening

- **Index date:** December 31, 2025
- **Initial population:** women aged 52 to 74 at the index date with a qualifying 2025 encounter (the computable age gate; the measure description summarizes the clinical target as 50 to 74)
- **Lookback window:** October 1, 2023 to December 31, 2025 (27 months; 0 to 821 days before the index date)
- **Numerator event:** bilateral mammography (SNOMED CT `71651007`)
- **Exclusion implemented:** bilateral mastectomy history (other specification exclusions, hospice / palliative / frailty / long-term care, do not occur in the synthetic cohort)

### Prompt Strategies

| Strategy | Description |
|---|---|
| `zero_shot_base` | Measure name and required output schema only |
| `guideline_supplied` | Full CMS125v13 specification injected into the prompt (single-document RAG) |

### Novel Metric: Auditability Match Rate (AMR)

AMR scores whether the model's cited evidence is consistent with the gold record:

- Gold-compliant patient: the model must cite a supporting event.
- Gold-non-compliant patient: the model must cite no evidence.
- Any other combination is an AMR failure (a hallucination), even when the final yes-or-no decision is correct.

---

## Models Evaluated

| Model | Provider | Type |
|---|---|---|
| GPT-4o | OpenAI (via OpenRouter) | Proprietary |
| Claude Sonnet 4.6 | Anthropic (via OpenRouter) | Proprietary |
| Llama 3.3 70B Instruct | Meta (via OpenRouter) | Open-weight |

All models were accessed through a single provider (OpenRouter) so that serving infrastructure does not confound the comparison.

---

## Cohort

The publication cohort (`cohort_2025_cms125_500_final.txt`) contains 500 patient IDs:

- **147 compliant** (all gold-truth compliant patients from the 25,000-patient Synthea simulation)
- **353 non-compliant** (random sample, seed = 99)
- **4.9% natural compliance rate** in Synthea; **29.4%** in the publication cohort (positive class oversampled for statistical power; not a real-world rate)

---

## Citation

If you use CLiTR-Bench in your research, please cite:

```bibtex
@misc{muppidi2026clitr,
  title  = {CLiTR-Bench: A Benchmark for Large Language Model Temporal Reasoning
            on Electronic Clinical Quality Measures},
  author = {Muppidi, Raja},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

---

## License

MIT License. Synthetic patient data generated using [Synthea™](https://synthetichealth.github.io/synthea/) (The MITRE Corporation). No real patient data were used. No IRB approval required.

---

## Contact

Raja Muppidi — open an [issue](https://github.com/rajamuppidi/CLiTR-Bench/issues) for questions or collaboration.
