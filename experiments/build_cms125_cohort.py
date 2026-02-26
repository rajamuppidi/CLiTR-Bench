"""
Fast CMS125 2025 Cohort Builder
=================================
Single-pass scan of all canonical CSVs to build a balanced cohort for a given
model and size. Much faster than the chunk-by-chunk approach (1 scan vs 25+).

Usage:
    python3 experiments/build_cms125_cohort.py --size 500 --seed 99 --model "meta-llama/llama-3.3-70b-instruct"
"""

import os
import csv
import json
import random
import logging
from datetime import datetime, date
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_DIR = os.path.join(TARGET_DIR, "data_generation", "output", "canonical")
TERMINOLOGY_PATH = os.path.join(TARGET_DIR, "terminology", "minimal_value_sets.json")

MEASUREMENT_YEAR_END = date(2025, 12, 31)
MEASUREMENT_YEAR_START = date(2025, 1, 1)
MAMMOGRAPHY_WINDOW_START = date(2023, 10, 1)  # 27 months before Dec 31, 2025


def parse_date(d_str):
    try:
        return datetime.strptime(d_str[:10], "%Y-%m-%d").date()
    except:
        return None


def calculate_age(dob_str, index_date):
    dob = parse_date(dob_str)
    if not dob:
        return None
    return index_date.year - dob.year - ((index_date.month, index_date.day) < (dob.month, dob.day))


def build_cohort(sample_size: int = 100, seed: int = 42, output_file: str = None, model_name: str = ""):
    random.seed(seed)

    if output_file is None:
        # Sanitize model name for use in filename (replace / : spaces with -)
        import re
        model_slug = re.sub(r'[/:\s]+', '-', model_name).strip('-') if model_name else ""
        suffix = f"_{model_slug}" if model_slug else ""
        output_file = os.path.join(TARGET_DIR, "experiments", f"cohort_2025_cms125_{sample_size}{suffix}.txt")

    # Load value sets
    with open(TERMINOLOGY_PATH, 'r') as f:
        value_sets = json.load(f)

    cms125_vs = value_sets["CMS125"]["value_sets"]
    mammo_codes = {c["code"] for c in cms125_vs["Mammography"]}
    bilateral_codes = {c["code"] for c in cms125_vs["Bilateral Mastectomy"]}
    left_codes = {c["code"] for c in cms125_vs["Unilateral Mastectomy Left"] + cms125_vs["Absence of Left Breast"]}
    right_codes = {c["code"] for c in cms125_vs["Unilateral Mastectomy Right"] + cms125_vs["Absence of Right Breast"]}

    # ─── Step 1: Load patients ─────────────────────────────────────────────────
    logging.info("Step 1: Loading patients.csv ...")
    female_52_74 = {}
    with open(os.path.join(CANONICAL_DIR, "patients.csv"), 'r') as f:
        for row in csv.DictReader(f):
            if row["sex"] == "F":
                age = calculate_age(row["dob"], MEASUREMENT_YEAR_END)
                if age is not None and 52 <= age <= 74:
                    female_52_74[row["patient_id"]] = row

    logging.info(f"  Female patients aged 52-74: {len(female_52_74):,}")

    # ─── Step 2: Scan encounters for 2025 visits ───────────────────────────────
    logging.info("Step 2: Scanning encounters.csv for 2025 visits ...")
    has_2025_visit = set()
    with open(os.path.join(CANONICAL_DIR, "encounters.csv"), 'r') as f:
        for row in csv.DictReader(f):
            if row["patient_id"] not in female_52_74:
                continue
            edate = parse_date(row["encounter_date"])
            if edate and MEASUREMENT_YEAR_START <= edate <= MEASUREMENT_YEAR_END:
                has_2025_visit.add(row["patient_id"])

    initial_population = set(female_52_74.keys()) & has_2025_visit
    logging.info(f"  Initial Population (female 52-74 with 2025 visit): {len(initial_population):,}")

    # ─── Step 3: Single-pass scan of events.csv ────────────────────────────────
    logging.info("Step 3: Scanning events.csv (single pass) ...")
    patients_with_mammo_in_window = set()
    patients_with_bilateral = set()
    patients_with_left_mast = set()
    patients_with_right_mast = set()

    rows = 0
    with open(os.path.join(CANONICAL_DIR, "events.csv"), 'r') as f:
        for row in csv.DictReader(f):
            rows += 1
            if rows % 2_000_000 == 0:
                logging.info(f"  ... {rows:,} rows scanned")

            pid = row["patient_id"]
            if pid not in initial_population:
                continue

            code = row["code"]
            edate = parse_date(row["event_date"])
            if not edate or edate > MEASUREMENT_YEAR_END:
                continue

            # Mammography in 27-month compliance window
            if code in mammo_codes and edate >= MAMMOGRAPHY_WINDOW_START:
                patients_with_mammo_in_window.add(pid)

            # Mastectomy codes
            if code in bilateral_codes:
                patients_with_bilateral.add(pid)
            if code in left_codes:
                patients_with_left_mast.add(pid)
            if code in right_codes:
                patients_with_right_mast.add(pid)

    logging.info(f"  Total rows scanned: {rows:,}")

    # ─── Step 4: Apply measure logic ──────────────────────────────────────────
    logging.info("Step 4: Applying CMS125 measure logic ...")

    # Exclusions: bilateral OR left+right unilateral
    excluded = patients_with_bilateral | (patients_with_left_mast & patients_with_right_mast)
    denominator = initial_population - excluded

    # Numerator: mammogram in compliance window (denominator-eligible only)
    numerator = denominator & patients_with_mammo_in_window
    non_compliant = denominator - numerator

    logging.info(f"  Denominator: {len(denominator):,}")
    logging.info(f"  Numerator (compliant): {len(numerator):,}")
    logging.info(f"  Non-compliant (overdue): {len(non_compliant):,}")
    logging.info(f"  Excluded: {len(excluded):,}")
    if len(denominator) > 0:
        logging.info(f"  Compliance rate: {len(numerator)/len(denominator)*100:.1f}%")

    # ─── Step 5: Sample balanced cohort ───────────────────────────────────────
    logging.info("Step 5: Sampling balanced cohort ...")

    target_yes = int(sample_size * 0.6)   # 60% compliant (ideal)
    target_no = sample_size - target_yes  # 40% non-compliant (ideal)

    num_yes_list = random.sample(sorted(numerator), min(target_yes, len(numerator)))

    # If YES pool is smaller than target, fill remaining slots with extra NO patients
    shortfall = target_yes - len(num_yes_list)
    if shortfall > 0:
        logging.warning(f"  Only {len(num_yes_list)} Numerator YES available (wanted {target_yes}). Filling {shortfall} extra slots with Numerator NO patients.")
        target_no += shortfall

    num_no_list = random.sample(sorted(non_compliant), min(target_no, len(non_compliant)))

    if len(num_no_list) < target_no:
        logging.warning(f"  Only {len(num_no_list)} Numerator NO available (wanted {target_no})")

    final_cohort = num_yes_list + num_no_list
    random.shuffle(final_cohort)

    # ─── Step 6: Write cohort file ─────────────────────────────────────────────
    with open(output_file, 'w') as f:
        for pid in final_cohort:
            f.write(f"{pid}\n")

    logging.info("=" * 80)
    logging.info("COHORT BUILT — CMS125 2025 (Measurement Year: 2025-01-01 → 2025-12-31)")
    logging.info("=" * 80)
    logging.info(f"  Total patients in cohort: {len(final_cohort)}")
    logging.info(f"  Numerator YES (screened):   {len(num_yes_list)} ({len(num_yes_list)/len(final_cohort)*100:.1f}%)")
    logging.info(f"  Numerator NO (overdue):     {len(num_no_list)} ({len(num_no_list)/len(final_cohort)*100:.1f}%)")
    logging.info(f"  Cohort saved to: {output_file}")
    logging.info("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast single-scan CMS125 2025 cohort builder")
    parser.add_argument("--size", type=int, default=100, help="Total cohort size (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="", help="Model name to embed in output filename (e.g. 'meta-llama/llama-3.3-70b-instruct')")
    parser.add_argument("--output", type=str, default=None, help="Override output file path (skips auto-naming)")
    args = parser.parse_args()

    build_cohort(sample_size=args.size, seed=args.seed, output_file=args.output, model_name=args.model)
