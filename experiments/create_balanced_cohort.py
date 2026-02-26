"""
Create a balanced cohort with equal representation of numerator-compliant cases.

This addresses severe class imbalance in the default Synthea-generated cohort
where only ~6% of denominator-eligible patients have documented mammography.

Balanced cohort distribution:
- 200 Numerator YES (40%)  - True positives to test recall
- 200 Numerator NO, Denominator YES (40%) - True negatives to test precision
- 100 Denominator NO (20%) - Test exclusion logic
"""

import os
import csv
import random
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_DIR = os.path.join(TARGET_DIR, "data_generation", "output", "canonical")
DEFAULT_OUTPUT_FILE = os.path.join(TARGET_DIR, "experiments", "cohort_2025_cms165.txt")

def create_balanced_cohort(
    sample_size: int = 100,
    measure_id: str = "CMS165",
    seed: int = 42,
    output_file: str = None,
    index_date: str = "2025-12-31"
):
    if output_file is None:
        output_file = DEFAULT_OUTPUT_FILE
    random.seed(seed)

    patient_csv = os.path.join(CANONICAL_DIR, "patients.csv")
    encounters_csv = os.path.join(CANONICAL_DIR, "encounters.csv")
    events_csv = os.path.join(CANONICAL_DIR, "events.csv")

    if not os.path.exists(patient_csv):
        logging.error(f"Patient CSV not found at {patient_csv}")
        return

    logging.info("Optimized Cohort Generation: Single-Pass Streaming")
    
    # 1. Load all patients
    logging.info(f"Loading {patient_csv}...")
    patients = {}
    with open(patient_csv, 'r') as f:
        for row in csv.DictReader(f):
            patients[row["patient_id"]] = row
            
    # 2. Load all encounters
    logging.info(f"Loading {encounters_csv}...")
    encounters_by_patient = defaultdict(list)
    with open(encounters_csv, 'r') as f:
        for row in csv.DictReader(f):
            encounters_by_patient[row["patient_id"]].append(row)
            
    # 3. Stream 2.6GB events.csv ONCE, storing only target patient events
    logging.info(f"Streaming {events_csv} ONCE (this takes ~30 seconds)...")
    events_by_patient = defaultdict(list)
    
    # Pre-select a random target pool so we only store their events to save RAM
    all_pids = list(patients.keys())
    random.shuffle(all_pids)
    target_pool = set(all_pids[:10000]) # We only need to check ~10k patients to find 100
    
    with open(events_csv, 'r') as f:
        for row in csv.DictReader(f):
            if row["patient_id"] in target_pool:
                events_by_patient[row["patient_id"]].append(row)

    # 4. Evaluate using the pre-loaded dictionary
    import sys
    sys.path.append(TARGET_DIR)
    from gold_truth_engine.gold_truth_engine import GoldTruthEngine

    engine = GoldTruthEngine(index_date_str=index_date)
    
    # Directly inject our manually loaded data into the engine's cache
    engine._cache_patients = patients
    engine._cache_encounters = encounters_by_patient
    engine._cache_events = events_by_patient
    engine._target_cohort = target_pool

    target_num_yes = int(sample_size * 0.6)
    target_num_no = sample_size - target_num_yes

    num_yes_cohort = []
    num_no_cohort = []

    logging.info(f"Searching for balanced cohort (index_date={index_date}):")
    logging.info(f"  - {target_num_yes} Numerator YES (screened, compliant)")
    logging.info(f"  - {target_num_no} Numerator NO (overdue, non-compliant)")

    debug_stats = {"total_checked": 0, "denom_yes": 0, "num_yes": 0, "failed_initial": 0, "failed_exclusion": 0}
    debug_reasons = defaultdict(int)

    for pid in target_pool:
        res = engine.evaluate_all(pid)
        if "error" in res or measure_id not in res: continue
        
        eval_dict = res[measure_id]
        is_denom = eval_dict["denominator"]
        is_num = eval_dict["numerator"]

        debug_stats["total_checked"] += 1
        if not eval_dict["initial_population"]: 
            debug_stats["failed_initial"] += 1
            if "debug_reason" in eval_dict:
                debug_reasons[eval_dict["debug_reason"]] += 1
        elif eval_dict["exclusion"]: 
            debug_stats["failed_exclusion"] += 1
        
        if is_denom: debug_stats["denom_yes"] += 1
        if is_num: debug_stats["num_yes"] += 1

        if is_num and len(num_yes_cohort) < target_num_yes:
            num_yes_cohort.append(pid)
        elif is_denom and not is_num and len(num_no_cohort) < target_num_no:
            num_no_cohort.append(pid)

        if len(num_yes_cohort) == target_num_yes and len(num_no_cohort) == target_num_no:
            break
            
    logging.info(f"Debug stats: {debug_stats}")
    logging.info(f"Failure Reasons: {dict(debug_reasons)}")

    # Combine and shuffle
    final_cohort = num_yes_cohort + num_no_cohort
    random.shuffle(final_cohort)

    # Save
    with open(output_file, 'w') as f:
        for pid in final_cohort:
            f.write(f"{pid}\n")

    logging.info(f"\n{'='*80}")
    logging.info(f"BALANCED COHORT CREATED — {measure_id} ({index_date})")
    logging.info(f"{'='*80}")
    logging.info(f"Total: {len(final_cohort)} patients")
    if len(final_cohort) > 0:
        logging.info(f"  - {len(num_yes_cohort)} Numerator YES / Compliant ({len(num_yes_cohort)/len(final_cohort)*100:.1f}%)")
        logging.info(f"  - {len(num_no_cohort)} Numerator NO / Overdue ({len(num_no_cohort)/len(final_cohort)*100:.1f}%)")
    logging.info(f"Saved to: {output_file}")
    logging.info(f"{'='*80}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a balanced cohort for CLiTR-Bench")
    parser.add_argument("--size", type=int, default=100, help="Total cohort size")
    parser.add_argument("--measure", type=str, default="CMS165", help="Target measure")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--index-date", type=str, default="2025-12-31", help="Measurement period end date")
    args = parser.parse_args()

    create_balanced_cohort(
        sample_size=args.size,
        measure_id=args.measure,
        seed=args.seed,
        output_file=args.output,
        index_date=args.index_date
    )
