# Gold Truth Engine Rules & Exclusions

This document defines the deterministic reasoning rules programmed into the Gold Truth Engine. This serves as the ground truth that the clinical LLMs must reproduce. 

## 1. CMS125: Breast Cancer Screening (BCS-E) — HEDIS 2025

Based on the National Committee for Quality Assurance (NCQA) HEDIS® 2025 specifications.

**Measurement Period**: January 1, 2025 – December 31, 2025 (index date: 2025-12-31)

### Patient Cohort (Initial Population)
- **Sex Restrictions**: Restricted strictly to Female (`F`) identifying patients.
- **Age Criteria**: **52 – 74** years of age at the index date (December 31, 2025).
- **Visit Requirement**: Must have had at least one encounter during the measurement year (2025).

### Exclusions
Before placing a patient into the final Denominator, we scan the entire longitudinal history for exclusion codes up to and including the measurement year.
1. **Absolute History of Bilateral Mastectomy**: Any instance of `0HTV0ZZ`, `OHTV0ZZ`, or `Z90.13` removes the patient.
2. **Combination of Unilateral Mastectomies**: If a patient has a combination of *both* a Left Breast removal AND a Right Breast removal, they are excluded.

### Denominator
- Defined as: `Initial Population` - `Exclusions`.

### Numerator (Screening Evidence)
- The model searches for valid Mammography procedure codes (LOINC `24606-6`, SNOMED `71651007`, CPT `77061`–`77067`).
- The evidence date must fall within **27 months prior to December 31, 2025** — i.e., on or after **October 1, 2023**.
- Any mammogram dated October 1, 2023 through December 31, 2025 counts as compliant.

---

## 2. CMS130: Colorectal Cancer Screening (COL)

### Patient Cohort (Initial Population)
- **Age Criteria**: 45 – 75 years of age at the index date.

### Exclusions
1. **Colorectal Cancer**: Any history of Colorectal Cancer (e.g., ICD-10 `C18.9`).
2. **Total Colectomy**: Any history of a Total Colectomy (e.g., CPT `44150`).

### Denominator
- Defined as: `Initial Population` - `Exclusions`.

### Numerator (Screening Evidence)
- Valid evidence includes:
  - **Colonoscopy** within 10 years prior.
  - **FIT test** during the measurement year (1 year lookback).

---

## 3. CMS165: Controlling High Blood Pressure (CBP)

### Patient Cohort (Initial Population)
- **Age Criteria**: `AgeInYearsAt(date from end of "Measurement Period")` in Interval [18, 85]
- **Diagnosis**: exists "Essential Hypertension Diagnosis" overlapping Interval [start of "Measurement Period", start of "Measurement Period" + 6 months)
- **Encounters**: exists AdultOutpatientEncounters."Qualifying Encounters" during day of "Measurement Period"

### Denominator Exclusions
Exclude patients if ANY of the following are met:
1. Hospice."Has Hospice Services" (overlaps Measurement Period)
2. exists ("Pregnancy or Renal Diagnosis" overlapping Measurement Period)
3. exists ("End Stage Renal Disease Procedures" on or before end of Measurement Period)
4. exists ("End Stage Renal Disease Encounter" on or before end of Measurement Period)
5. AIFrailLTCF."Is Age 66 to 80 with Advanced Illness and Frailty or Is Age 81 or Older with Frailty"
   - Age 66 to 80 AND Frailty Criteria AND (Advanced Illness in Year Before/During OR Dementia Meds in Year Before/During)
   - OR Age >= 81 AND Frailty Criteria
6. AIFrailLTCF."Is Age 66 or Older Living Long Term in a Nursing Home" (Housing status "Lives in nursing home" on or before end of Measurement Period)
7. PalliativeCare."Has Palliative Care in the Measurement Period"

### Numerator
- "Has Systolic Blood Pressure Less Than 140" AND "Has Diastolic Blood Pressure Less Than 90"
- Both readings must come from the **"Most Recent Blood Pressure Day"**.
- "Most Recent Blood Pressure Day" is the latest date in the Measurement Period that has *both* a qualifying systolic reading and a qualifying diastolic reading.
- "Lowest Diastolic Reading on Most Recent Blood Pressure Day" < 90 mmHg
- "Lowest Systolic Reading on Most Recent Blood Pressure Day" < 140 mmHg

### Numerator Exclusions & Denominator Exceptions
- None


## 4. CMS122: Diabetes: Hemoglobin A1c Poor Control (> 9%) (HBD)

### Patient Cohort (Initial Population)
- **Age Criteria**: 18 – 75 years of age at the index date.
- **Diagnosis**: Must have an active diagnosis of Diabetes (e.g., ICD-10 `E11.9`) by the end of the measurement year.

### Exclusions
- General advanced illness/frailty.
- Steroid, Gestational, or PCOS induced diabetes without true base diabetes diagnosis. 

### Denominator
- Defined as: `Initial Population` - `Exclusions`.

### Numerator ("Poor Control")
- **Control defined as Poor if:**
  - The most recent HbA1c test result in the measurement year is **> 9.0%**.
  - **OR** if *no* HbA1c test was performed at all during the measurement year (defaults to Poor control/Numerator compliant).
