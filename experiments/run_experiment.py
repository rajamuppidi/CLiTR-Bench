import os
import sys
import json
import logging
import time
from datetime import datetime

# Import project modules
TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TARGET_DIR)

from gold_truth_engine.gold_truth_engine import GoldTruthEngine
from llm_runner.run_inference import LLMRunner
from representations.renderers import RepresentationRenderer
from evaluation.metrics_engine import MetricsEngine

from dotenv import load_dotenv
load_dotenv(os.path.join(TARGET_DIR, '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiment(measure_id: str = "CMS125", 
                   measure_name: str = "Breast Cancer Screening",
                   cohort_file: str = "cohort_500.txt",
                   prompt_style: str = "zero_shot_base",
                   format_type: str = "csv",
                   model_name: str = "gpt-4-turbo"):
                   
    cohort_path = os.path.join(TARGET_DIR, "experiments", cohort_file)
    if not os.path.exists(cohort_path):
        logging.error(f"Cohort file {cohort_path} not found.")
        return
        
    with open(cohort_path, 'r') as f:
        patient_ids = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(patient_ids)} patients for evaluation.")
    
    # Initialize engines
    renderer = RepresentationRenderer()
    gold_engine = GoldTruthEngine(index_date_str="2025-12-31")
    llm_runner = LLMRunner(model_name=model_name)
    metrics_engine = MetricsEngine()
    
    # Check API Key based on provider
    if ("/" in model_name or "openrouter" in model_name.lower()) and not os.environ.get("OPENROUTER_API_KEY"):
        logging.warning("OPENROUTER_API_KEY not found in environment. Proceeding with dummy API for now.")
    elif "gpt" in model_name and not os.environ.get("OPENAI_API_KEY") and "openrouter" not in model_name.lower():
        logging.warning("OPENAI_API_KEY not found in environment. Proceeding with dummy API for now.")
    elif "gemini" in model_name.lower() and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        logging.warning("GOOGLE_API_KEY not found in environment for Gemini. Proceeding with dummy API for now.")
        
    # Read Guideline logic if needed
    guideline_logic = ""
    if prompt_style == "guideline_supplied":
        guideline_path = os.path.join(TARGET_DIR, "documentation", "02_gold_truth_engine_logic.md")
        with open(guideline_path, 'r') as f:
            guideline_logic = f.read()

    gold_truths = []
    llm_predictions = []
    
    # Output file names with model slug for traceability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import re
    model_slug = re.sub(r'[/:\s]+', '-', model_name).strip('-')
    results_file = os.path.join(TARGET_DIR, "experiments", f"results_{measure_id}_{format_type}_{prompt_style}_{model_slug}_{timestamp}.jsonl")
    
    # Ultra-fast RAM caching for just the cohort (skips 99% of 3GB CSV files)
    logging.info(f"Preloading 3GB datasets into optimized memory indices for N={len(patient_ids)} cohort...")
    gold_engine.preload_cohort(patient_ids)
    renderer.preload_cohort(patient_ids)
    logging.info("Memory initialization complete.")
    
    logging.info(f"Starting execution loop over {len(patient_ids)} patients. Tracking to {results_file}")
    
    for idx, pid in enumerate(patient_ids):
        # 1. Generate Gold Truth
        is_denom, is_num, evidence = gold_engine.evaluate_patient(pid, measure_id)
        gt_record = {
            "patient_id": pid,
            "measure": measure_id,
            "denominator": is_denom,
            "numerator": is_num,
            "evidence": evidence
        }
        gold_truths.append(gt_record)
        
        # 2. Render formatting for LLM
        representation = ""
        if format_type == "csv" or format_type == "json":
            representation = renderer.render_structured(pid, format_type=format_type)
        else:
            representation = renderer.render_note(pid)
            
        # 3. Call LLM Pipeline
        llm_response = llm_runner.evaluate_patient(
            patient_representation=representation,
            measure_id=measure_id,
            measure_name=measure_name,
            prompt_style=prompt_style,
            format_type=format_type,
            guideline_logic=guideline_logic
        )
        
        # Merge patient metadata
        llm_response["patient_id"] = pid
        llm_response["measure"] = measure_id
        llm_predictions.append(llm_response)
        
        # Save lines instantly for crash recovery
        with open(results_file, 'a') as f:
            log_line = {"patient_id": pid, "gold_truth": gt_record, "llm_prediction": llm_response}
            f.write(json.dumps(log_line) + "\n")
            
        if (idx + 1) % 10 == 0:
            logging.info(f"Processed {idx + 1}/{len(patient_ids)}...")
        
        # Rate limit delays for free-tier providers
        if "gemini" in model_name.lower():
            time.sleep(4)
        elif ":free" in model_name.lower() and "/" in model_name:
            # OpenRouter free tier: ~20 req/min → 3s gap is safe
            time.sleep(3)

    # 4. Final Evaluation
    logging.info("All patients inferenced. Computing benchmark metrics...")
    final_scores = metrics_engine.evaluate_batch(gold_truths, llm_predictions)
    
    scores_file = os.path.join(TARGET_DIR, "experiments", f"final_scores_{measure_id}_{format_type}_{prompt_style}_{model_slug}_{timestamp}.json")
    with open(scores_file, 'w') as f:
        json.dump(final_scores, f, indent=2)
        
    logging.info("====================================")
    logging.info("EXPERIMENT COMPLETE. FINAL SCORES:")
    logging.info("====================================")
    print(json.dumps(final_scores, indent=2))
    logging.info(f"Full transaction logs saved to {results_file}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", default="CMS125")
    parser.add_argument("--measure-name", default="Breast Cancer Screening")
    parser.add_argument("--format", choices=["csv", "json", "clinical_note"], default="csv")
    parser.add_argument("--prompt", choices=["zero_shot_base", "zero_shot_cot", "guideline_supplied"], default="zero_shot_base")
    parser.add_argument("--model", default="gemini-1.5-flash-8b")
    parser.add_argument("--cohort", default="cohort_2025_cms125.txt")
    
    args = parser.parse_args()
    
    run_experiment(
        measure_id=args.measure,
        measure_name=args.measure_name,
        format_type=args.format,
        prompt_style=args.prompt,
        model_name=args.model,
        cohort_file=args.cohort
    )
