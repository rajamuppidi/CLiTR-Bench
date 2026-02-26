import os
import csv
import random
import logging

logging.basicConfig(level=logging.INFO)

TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_DIR = os.path.join(TARGET_DIR, "data_generation", "output", "canonical")
OUTPUT_FILE = os.path.join(TARGET_DIR, "experiments", "cohort_500.txt")

def sample_patients(sample_size: int = 500, measure_id: str = "CMS125", seed: int = 42):
    random.seed(seed)
    
    patient_ids = []
    patient_csv = os.path.join(CANONICAL_DIR, "patients.csv")
    
    if not os.path.exists(patient_csv):
        logging.error(f"Patient CSV not found at {patient_csv}")
        return
        
    logging.info(f"Reading {patient_csv}...")
    with open(patient_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_ids.append(row["patient_id"])
            
    # To avoid loading all 25k patients at once, we will query them in random batches
    # until we find exactly 400 denominator-positive and 100 denominator-negative patterns.
    random.shuffle(patient_ids)
    
    # Import the engine dynamically so we get the exact denominator math
    import sys
    sys.path.append(TARGET_DIR)
    from gold_truth_engine.gold_truth_engine import GoldTruthEngine
    
    engine = GoldTruthEngine()
    
    target_pos = int(sample_size * 0.8)
    target_neg = sample_size - target_pos
    
    pos_cohort = []
    neg_cohort = []
    
    # We will grab patients in chunks of 1000 to keep memory extremely low but process fast
    chunk_size = 1000
    for i in range(0, len(patient_ids), chunk_size):
        chunk = patient_ids[i:i+chunk_size]
        logging.info(f"Preloading and evaluating chunk of {len(chunk)} patients...")
        
        # Preload the 3GB files just for this small chunk
        engine.preload_cohort(set(chunk))
        
        for pid in chunk:
            is_denom, is_num, ev = engine.evaluate_patient(pid, measure_id)
            
            if is_denom and len(pos_cohort) < target_pos:
                pos_cohort.append(pid)
            elif not is_denom and len(neg_cohort) < target_neg:
                neg_cohort.append(pid)
                
            if len(pos_cohort) == target_pos and len(neg_cohort) == target_neg:
                break
                
        if len(pos_cohort) == target_pos and len(neg_cohort) == target_neg:
            break
            
    final_cohort = pos_cohort + neg_cohort
    random.shuffle(final_cohort)
    
    with open(OUTPUT_FILE, 'w') as f:
        for pid in final_cohort:
            f.write(f"{pid}\n")
            
    logging.info(f"Successfully sampled {len(final_cohort)} patients.")
    logging.info(f"  - Built with {len(pos_cohort)} Denominator YES (80%)")
    logging.info(f"  - Built with {len(neg_cohort)} Denominator NO (20%)")
    logging.info(f"Saved optimized cohort list to {OUTPUT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=500, help="Number of patients to sample")
    parser.add_argument("--measure", type=str, default="CMS125", help="Target measure to bias towards")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    sample_patients(sample_size=args.size, measure_id=args.measure, seed=args.seed)
