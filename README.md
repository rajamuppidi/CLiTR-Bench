# CLiTR-Bench

**CLiTR-Bench** (Clinical Temporal Reasoning Benchmark) is a fully reproducible benchmark for evaluating large language model (LLM) performance on electronic clinical quality measure (eCQM) temporal compliance reasoning.

The benchmark is instantiated on **CMS125 Breast Cancer Screening** (HEDIS 2025 BCS-E), targeting the 27-month bilateral mammography lookback window—a representative eCQM requiring rolling-window date arithmetic, longitudinal event disambiguation, and auditable evidence citation.

> **Paper:** *CLiTR-Bench: Benchmarking Large Language Model Temporal Compliance Reasoning on Electronic Clinical Quality Measures* — *manuscript in preparation*

---

## Repository Structure

```
CLiTR-Bench/
├── gold_truth_engine/          # Deterministic CMS125 gold truth implementation
│   ├── gold_truth_engine.py    # Core engine (initial pop, denominator, numerator, exclusions)
│   └── test_gold_truth_engine.py
│
├── llm_runner/                 # LLM inference via OpenRouter API
│   └── run_inference.py
│
├── prompts/                    # Prompt templates
│   └── cms125_prompts.py       # Zero-shot base & guideline-supplied prompts
│
├── terminology/                # NCQA value sets
│   └── minimal_value_sets.json # CMS125 mammography codes (SNOMED)
│
├── representations/            # Patient data serialization
│   └── csv_representation.py
│
├── experiments/                # Experiment runner and results
│   ├── run_experiment.py       # Main experiment entrypoint
│   ├── build_cms125_cohort.py  # Cohort builder (500-patient publication cohort)
│   ├── verify_synthea_data.py  # Synthea data validation utility
│   │
│   ├── cohort_2025_cms125_500_meta-llama-llama-3.3-70b-instruct.txt  # Publication cohort (n=499, seed=99)
│   │
│   ├── final_scores_*.json     # Aggregate benchmark scores (publication-scale)
│   ├── results_*.jsonl         # Per-patient inference logs (publication-scale)
│   │
│   └── analysis/               # Statistical analysis scripts
│       ├── bootstrap_ci.py     # Bootstrap 95% confidence intervals (B=10,000)
│       ├── mcnemar_test.py     # McNemar paired significance tests
│       ├── error_analysis.py   # Hallucination error taxonomy
│       ├── bootstrap_summary.md
│       ├── mcnemar_summary.md
│       └── qwen_zero_shot_error_analysis.md
│
└── manuscript/
    └── CLiTR_Bench_Paper_Draft.md   # Full AMIA 2026 paper draft
```

---

## Key Results (CMS125, n=499–500 patients)

| Model | Prompt | F1 | Precision | Recall | Auditability | Hallucinations |
|---|---|---|---|---|---|---|
| **GPT-4o** | Guideline-Supplied | **96.55%** [94.16–98.50] | **97.9%** | 95.24% | **98.0%** | **10/500** |
| Llama 3.3 70B | Guideline-Supplied | 87.66% [83.44–91.30] | 83.85% | 92.47% | 92.4% | 38/499 |
| Qwen 3 80B | Guideline-Supplied | 76.92% [71.88–81.44] | 64.52% | 95.24% | 82.6% | 89/500 |
| Qwen 3 80B | Zero-Shot | 48.73% [43.49–53.70] | 33.25% | 91.16% | 43.6% | 282/500 |
| Llama 3.3 70B | Zero-Shot | 47.02% [39.60–53.87] | 48.55% | 45.58% | 69.8% | 196/499 |

*All guideline-supplied pairwise differences are statistically significant (McNemar, p < 0.001, Bonferroni-corrected).*

---

## Quick Start

### 1. Install dependencies

```bash
pip install httpx scipy numpy
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
# Place output CSV files in data_generation/synthea_output/
```

### 4. Build the cohort

```bash
cd experiments
python3 build_cms125_cohort.py \
  --data-dir ../data_generation/synthea_output \
  --size 500 \
  --seed 99
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
  --cohort "cohort_2025_cms125_500_meta-llama-llama-3.3-70b-instruct.txt"
```

### 6. Reproduce statistical analysis

```bash
cd experiments
python3 analysis/bootstrap_ci.py    # Bootstrap 95% CIs
python3 analysis/mcnemar_test.py    # Paired significance tests
python3 analysis/error_analysis.py  # Qwen zero-shot error taxonomy
```

---

## Benchmark Design

### Measure: CMS125 BCS-E (HEDIS 2025)
- **Index date:** December 31, 2025
- **Lookback window:** October 1, 2023 – December 31, 2025 (27 months)
- **Numerator event:** Bilateral mammography (SNOMED `71651007`)
- **Exclusion:** Bilateral mastectomy history

### Prompt Strategies
| Strategy | Description |
|---|---|
| `zero_shot_base` | Measure name + output format only |
| `guideline_supplied` | Full HEDIS BCS-E specification injected in context |

### Novel Metric: Auditability Match Rate (AMR)
AMR measures whether the LLM's cited evidence is grounded in the actual patient record:
- For non-compliant patients: model must cite no evidence
- For compliant patients: model must cite a non-null mammography event
- Hallucination = AMR failure

---

## Models Evaluated

| Model | Provider | Parameters |
|---|---|---|
| GPT-4o | OpenAI (via OpenRouter) | ~200B |
| Llama 3.3 70B Instruct | Meta (via OpenRouter/DeepInfra) | 70B |
| Qwen 3 80B | Alibaba (via OpenRouter/DeepInfra) | 80B |

---

## Cohort

The publication cohort (`cohort_2025_cms125_500_meta-llama-llama-3.3-70b-instruct.txt`) contains 499 patient IDs:
- **147 compliant** (all compliant patients from 25,000-patient Synthea simulation)
- **352 non-compliant** (random sample, seed=99)
- **4.9% natural compliance rate** in Synthea; 29.4% in publication cohort (documented)

---

## Citation

If you use CLiTR-Bench in your research, please cite:

```bibtex
@misc{muppidi2026clitr,
  title  = {CLiTR-Bench: Benchmarking Large Language Model Temporal Compliance Reasoning
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
