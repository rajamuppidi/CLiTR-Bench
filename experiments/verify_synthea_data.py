"""
CMS125 Synthea Data Verification
=================================
Before building a cohort, verify that the existing Synthea canonical data
contains the necessary clinical codes for CMS125 (Breast Cancer Screening):
  - Female patients aged 52-74 (measurement year end: 2025-12-31)
  - Visits in the measurement period (2025)
  - Mammography codes (LOINC 24606-6, SNOMED 71651007, CPT 77061-77067)
  - Mastectomy codes (bilateral / unilateral left / unilateral right)

Run from: clitr-bench/
"""

import os
import csv
import json
import sys
from datetime import datetime, date
from collections import defaultdict, Counter

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
    if not dob: return None
    return index_date.year - dob.year - ((index_date.month, index_date.day) < (dob.month, dob.day))

def main():
    print("=" * 70)
    print("CLiTR-Bench: CMS125 Synthea Data Verification")
    print(f"Measurement Period: {MEASUREMENT_YEAR_START} → {MEASUREMENT_YEAR_END}")
    print(f"Mammography Lookback Window: {MAMMOGRAPHY_WINDOW_START} → {MEASUREMENT_YEAR_END}")
    print("=" * 70)

    # --- Load value sets ---
    with open(TERMINOLOGY_PATH, 'r') as f:
        value_sets = json.load(f)

    cms125_vs = value_sets["CMS125"]["value_sets"]
    mammo_codes = {c["code"] for cs in [cms125_vs["Mammography"]] for c in cs}
    bilateral_codes = {c["code"] for c in cms125_vs["Bilateral Mastectomy"]}
    left_codes = {c["code"] for c in cms125_vs["Unilateral Mastectomy Left"] + cms125_vs["Absence of Left Breast"]}
    right_codes = {c["code"] for c in cms125_vs["Unilateral Mastectomy Right"] + cms125_vs["Absence of Right Breast"]}
    unspecified_codes = {c["code"] for c in cms125_vs["Unilateral Mastectomy Unspecified"]}
    all_mastectomy_codes = bilateral_codes | left_codes | right_codes | unspecified_codes

    print(f"\n[Value Sets Loaded]")
    print(f"  Mammography codes: {sorted(mammo_codes)}")
    print(f"  Bilateral mastectomy codes: {sorted(bilateral_codes)}")
    print(f"  Left mastectomy codes: {sorted(left_codes)}")
    print(f"  Right mastectomy codes: {sorted(right_codes)}")
    print(f"  Unspecified mastectomy codes: {sorted(unspecified_codes)}")

    # --- Step 1: Scan patients.csv ---
    print(f"\n[Step 1] Scanning patients.csv...")
    all_patients = {}
    total_patients = 0
    female_patients = set()
    eligible_age_patients = set()  # Female AND 52-74

    with open(os.path.join(CANONICAL_DIR, "patients.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_patients += 1
            pid = row["patient_id"]
            all_patients[pid] = row
            if row["sex"] == "F":
                female_patients.add(pid)
                age = calculate_age(row["dob"], MEASUREMENT_YEAR_END)
                if age is not None and 52 <= age <= 74:
                    eligible_age_patients.add(pid)

    print(f"  Total patients in canonical data: {total_patients:,}")
    print(f"  Female patients: {len(female_patients):,}")
    print(f"  Female patients aged 52-74 at end of 2025: {len(eligible_age_patients):,}")

    # --- Step 2: Scan encounters.csv ---
    print(f"\n[Step 2] Scanning encounters.csv for 2025 visits...")
    patients_with_2025_visit = set()
    encounter_counts_by_pid = defaultdict(int)

    with open(os.path.join(CANONICAL_DIR, "encounters.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            edate = parse_date(row["encounter_date"])
            if edate and MEASUREMENT_YEAR_START <= edate <= MEASUREMENT_YEAR_END:
                patients_with_2025_visit.add(pid)
                encounter_counts_by_pid[pid] += 1

    # Filter eligible_age to those with 2025 visit (approximate initial population)
    initial_population = eligible_age_patients & patients_with_2025_visit
    print(f"  Patients with at least one encounter in 2025: {len(patients_with_2025_visit):,}")
    print(f"  Female 52-74 WITH 2025 visit (Initial Population ≈): {len(initial_population):,}")

    # --- Step 3: Scan events.csv ---
    print(f"\n[Step 3] Scanning events.csv for CMS125 codes (this may take 30-60s)...")
    
    patients_with_mammo_ever = set()
    patients_with_mammo_in_window = set()  # Oct 2023 – Dec 2025
    patients_with_bilateral_mastectomy = set()
    patients_with_left_mastectomy = set()
    patients_with_right_mastectomy = set()
    
    mammo_dates_by_pid = defaultdict(list)
    code_frequency = Counter()

    rows_scanned = 0
    with open(os.path.join(CANONICAL_DIR, "events.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_scanned += 1
            if rows_scanned % 1_000_000 == 0:
                print(f"    ... {rows_scanned:,} event rows scanned")
            
            pid = row["patient_id"]
            code = row["code"]
            edate = parse_date(row["event_date"])
            
            if not edate or edate > MEASUREMENT_YEAR_END:
                continue
            
            # Mammography
            if code in mammo_codes:
                patients_with_mammo_ever.add(pid)
                mammo_dates_by_pid[pid].append(edate)
                code_frequency[f"mammo:{code}"] += 1
                if edate >= MAMMOGRAPHY_WINDOW_START:
                    patients_with_mammo_in_window.add(pid)
            
            # Mastectomy
            if code in bilateral_codes:
                patients_with_bilateral_mastectomy.add(pid)
                code_frequency[f"bilateral_mast:{code}"] += 1
            if code in left_codes:
                patients_with_left_mastectomy.add(pid)
                code_frequency[f"left_mast:{code}"] += 1
            if code in right_codes:
                patients_with_right_mastectomy.add(pid)
                code_frequency[f"right_mast:{code}"] += 1

    print(f"  Total event rows scanned: {rows_scanned:,}")

    # --- Step 4: Compute CMS125 pipeline stats ---
    print(f"\n[Step 4] Computing CMS125 Measure Statistics...")

    # Denominator exclusions (bilateral OR left+right)
    bilateral_excluded = initial_population & patients_with_bilateral_mastectomy
    combined_unilateral_excluded = initial_population & (patients_with_left_mastectomy & patients_with_right_mastectomy)
    all_excluded = bilateral_excluded | combined_unilateral_excluded

    denominator = initial_population - all_excluded
    
    # Numerator: mammography in compliance window, denominator-eligible
    numerator = denominator & patients_with_mammo_in_window
    non_compliant_denom = denominator - numerator

    print(f"\n{'='*70}")
    print(f"CMS125 POPULATION BREAKDOWN (2025 Measurement Year)")
    print(f"{'='*70}")
    print(f"  Total patients in data:           {total_patients:,}")
    print(f"  Female patients:                  {len(female_patients):,}")
    print(f"  Female aged 52-74:                {len(eligible_age_patients):,}")
    print(f"  Initial Population (+ 2025 visit):{len(initial_population):,}")
    print(f"  ─── Exclusions ───────────────────")
    print(f"  Bilateral mastectomy excluded:    {len(bilateral_excluded):,}")
    print(f"  Bilateral unilateral excluded:    {len(combined_unilateral_excluded):,}")
    print(f"  Total excluded:                   {len(all_excluded):,}")
    print(f"  ─── Denominator (eligible) ───────")
    print(f"  Denominator:                      {len(denominator):,}")
    print(f"  ─── Numerator ────────────────────")
    print(f"  Mammo EVER (any patient):         {len(patients_with_mammo_ever):,}")
    print(f"  Mammo in compliance window:       {len(patients_with_mammo_in_window):,}")
    print(f"  Numerator (Denominator + window): {len(numerator):,}")
    print(f"  Non-compliant (Denom, no mammo):  {len(non_compliant_denom):,}")
    print(f"{'='*70}")
    
    # --- CMS125 compliance rate ---
    if len(denominator) > 0:
        compliance_rate = len(numerator) / len(denominator) * 100
        print(f"\n  ✅ CMS125 Compliance Rate: {compliance_rate:.1f}%")
    
    # --- Balanced cohort feasibility ---
    print(f"\n[Step 5] Balanced Cohort Feasibility (100-patient target):")
    target_total = 100
    target_num_yes = 40   # 40%
    target_num_no = 40    # 40%
    target_denom_no = 20  # 20%
    
    feasible = (len(numerator) >= target_num_yes and 
                len(non_compliant_denom) >= target_num_no and 
                len(initial_population) - len(denominator) + len(all_excluded) >= target_denom_no)
    
    print(f"  Need {target_num_yes} Numerator YES  → Available: {len(numerator):,}  {'✅ OK' if len(numerator) >= target_num_yes else '❌ INSUFFICIENT'}")
    print(f"  Need {target_num_no} Numerator NO   → Available: {len(non_compliant_denom):,}  {'✅ OK' if len(non_compliant_denom) >= target_num_no else '❌ INSUFFICIENT'}")
    denom_no_pool = len(eligible_age_patients) - len(initial_population)
    print(f"  Need {target_denom_no} Denominator NO → Available: {denom_no_pool:,} (age-eligible but no 2025 visit)  {'✅ OK' if denom_no_pool >= target_denom_no else '❌ INSUFFICIENT'}")
    
    if feasible:
        print(f"\n  ✅ Balanced 100-patient cohort is FEASIBLE with existing Synthea data.")
    else:
        print(f"\n  ⚠️  Balanced cohort may require adjustment — check insufficient groups above.")

    # --- Code frequency ---
    print(f"\n[Step 6] Most Frequent CMS125 Codes Found:")
    for code, count in sorted(code_frequency.items(), key=lambda x: -x[1])[:20]:
        print(f"  {code}: {count} occurrences")

    # --- Sample mammogram dates ---
    print(f"\n[Step 7] Sample Most Recent Mammogram Dates (first 10 compliant patients):")
    count = 0
    for pid in sorted(numerator)[:10]:
        dates_in_window = [d for d in mammo_dates_by_pid.get(pid, []) if d >= MAMMOGRAPHY_WINDOW_START]
        if dates_in_window:
            most_recent = max(dates_in_window)
            patient = all_patients.get(pid, {})
            age = calculate_age(patient.get("dob", ""), MEASUREMENT_YEAR_END)
            print(f"  Patient {pid[:8]}... | Age: {age} | Most recent mammo: {most_recent}")
            count += 1

    print(f"\n{'='*70}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
