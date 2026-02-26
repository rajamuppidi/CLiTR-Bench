import os
import csv
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Tuple

TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_DIR = os.path.join(TARGET_DIR, "data_generation", "output", "canonical")
TERMINOLOGY_PATH = os.path.join(TARGET_DIR, "terminology", "minimal_value_sets.json")

def parse_date(d_str: str) -> date:
    return datetime.strptime(d_str, "%Y-%m-%d").date()

def calculate_age(dob: date, index_date: date) -> int:
    age = index_date.year - dob.year - ((index_date.month, index_date.day) < (dob.month, dob.day))
    return age

class GoldTruthEngine:
    def __init__(self, index_date_str="2025-12-31"):
        self.index_date = parse_date(index_date_str)
        
        with open(TERMINOLOGY_PATH, 'r') as f:
            self.value_sets = json.load(f)
            
        self.code_maps = {}
        for measure, md in self.value_sets.items():
            self.code_maps[measure] = {}
            for vs_name, codes_list in md['value_sets'].items():
                self.code_maps[measure][vs_name+"_CODES"] = {c.get('code', '') for c in codes_list}
                
        # Cache to prevent massive file I/O overhead
        self._cache_patients = {}
        self._cache_encounters = {}
        self._cache_events = {}
        self._loaded = False
        self._target_cohort = set()
        
    def _ensure_loaded(self, patient_id: str):
        if patient_id in self._cache_patients or patient_id in self._target_cohort: return
        with open(os.path.join(CANONICAL_DIR, "patients.csv"), 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] == patient_id:
                    self._cache_patients[row['patient_id']] = row
                    break
        with open(os.path.join(CANONICAL_DIR, "encounters.csv"), 'r') as f:
            for row in csv.DictReader(f): 
                if row['patient_id'] == patient_id: self._cache_encounters.setdefault(row['patient_id'], []).append(row)
        with open(os.path.join(CANONICAL_DIR, "events.csv"), 'r') as f:
            for row in csv.DictReader(f): 
                if row['patient_id'] == patient_id: self._cache_events.setdefault(row['patient_id'], []).append(row)
        self._target_cohort.add(patient_id)

    def preload_cohort(self, cohort_ids: set):
        missing = set(cohort_ids) - set(self._cache_patients.keys())
        if not missing: return
        
        logging.info("  -> Loading patients.csv...")
        with open(os.path.join(CANONICAL_DIR, "patients.csv"), 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] in missing: self._cache_patients[row['patient_id']] = row
                
        logging.info("  -> Loading encounters.csv...")
        with open(os.path.join(CANONICAL_DIR, "encounters.csv"), 'r') as f:
            for row in csv.DictReader(f): 
                if row['patient_id'] in missing: self._cache_encounters.setdefault(row['patient_id'], []).append(row)
                
        logging.info("  -> Loading 2.6GB events.csv (this takes 30-40 seconds)...")
        with open(os.path.join(CANONICAL_DIR, "events.csv"), 'r') as f:
            for row in csv.DictReader(f): 
                if row['patient_id'] in missing: self._cache_events.setdefault(row['patient_id'], []).append(row)
                
        self._target_cohort.update(missing)
    
    def load_patient_data(self, patient_id: str) -> Tuple[Dict, List[Dict], List[Dict]]:
        self._ensure_loaded(patient_id)
        return (
            self._cache_patients.get(patient_id),
            self._cache_encounters.get(patient_id, []),
            self._cache_events.get(patient_id, [])
        )

    def evaluate_cms125(self, patient: Dict, encounters: List[Dict], events: List[Dict]) -> Dict:
        """Breast Cancer Screening (BCS-E)"""
        result = {"initial_population": False, "denominator": False, "numerator": False, "exclusion": False, "exclusion_reason": None, "evidence": None}
        if not patient or patient['sex'] != 'F': return result
        
        age = calculate_age(parse_date(patient['dob']), self.index_date)
        if 52 <= age <= 74: result['initial_population'] = True  # HEDIS 2025 BCS-E: 52-74
        
        has_encounter = any(parse_date(e['encounter_date']).year == self.index_date.year for e in encounters)
        if not result['initial_population'] or not has_encounter: return result

        map_bcs = self.code_maps["CMS125"]
        has_bilateral, has_left, has_right = False, False, False
        for ev in events:
            cd = ev['code']
            edate = parse_date(ev['event_date'])
            if edate.year > self.index_date.year: continue
            if cd in map_bcs["Bilateral Mastectomy_CODES"]: has_bilateral = True
            if cd in map_bcs["Absence of Left Breast_CODES"] or cd in map_bcs["Unilateral Mastectomy Left_CODES"]: has_left = True
            if cd in map_bcs["Absence of Right Breast_CODES"] or cd in map_bcs["Unilateral Mastectomy Right_CODES"]: has_right = True
                
        if has_bilateral or (has_left and has_right):
            result['exclusion'] = True
            result['exclusion_reason'] = "Bilateral mastectomy or equivalent"
            
        if not result['exclusion']: result['denominator'] = True
        if result['denominator']:
            lookback_days = 821  # 27 months prior (Oct 1 two years prior → end of measurement period)
            valid_mammos = []
            for ev in events:
                if ev['code'] in map_bcs["Mammography_CODES"]:
                    days_diff = (self.index_date - parse_date(ev['event_date'])).days
                    if 0 <= days_diff <= lookback_days:
                        valid_mammos.append(ev)
            if valid_mammos:
                valid_mammos.sort(key=lambda x: parse_date(x['event_date']), reverse=True)
                result['numerator'] = True
                result['evidence'] = valid_mammos[0]
        return result

    def evaluate_cms130(self, patient: Dict, encounters: List[Dict], events: List[Dict]) -> Dict:
        """Colorectal Cancer Screening (COL)"""
        result = {"initial_population": False, "denominator": False, "numerator": False, "exclusion": False, "exclusion_reason": None, "evidence": None}
        if not patient: return result
        
        age = calculate_age(parse_date(patient['dob']), self.index_date)
        if 45 <= age <= 75: result['initial_population'] = True
        
        has_encounter = any(parse_date(e['encounter_date']).year == self.index_date.year for e in encounters)
        if not result['initial_population'] or not has_encounter: return result

        map_col = self.code_maps["CMS130"]
        for ev in events:
            cd = ev['code']
            edate = parse_date(ev['event_date'])
            if edate.year > self.index_date.year: continue
            if cd in map_col["Colorectal Cancer Exclusion_CODES"]:
                result['exclusion'] = True
                result['exclusion_reason'] = "Colorectal Cancer"
            elif cd in map_col["Total Colectomy Exclusion_CODES"]:
                result['exclusion'] = True
                result['exclusion_reason'] = "Total Colectomy"
                
        if not result['exclusion']: result['denominator'] = True
        if result['denominator']:
            valid_screenings = []
            for ev in events:
                cd = ev['code']
                days_diff = (self.index_date - parse_date(ev['event_date'])).days
                if days_diff < 0: continue
                # Colonoscopy - 10 years (3650 days)
                if cd in map_col["Colonoscopy_CODES"] and days_diff <= 3650:
                    valid_screenings.append(ev)
                # FIT - measurement year (say, 365 days for simplicity)
                if cd in map_col["FIT_CODES"] and days_diff <= 365:
                    valid_screenings.append(ev)
            if valid_screenings:
                valid_screenings.sort(key=lambda x: parse_date(x['event_date']), reverse=True)
                result['numerator'] = True
                result['evidence'] = valid_screenings[0]
        return result

    def evaluate_cms165(self, patient: Dict, encounters: List[Dict], events: List[Dict]) -> Dict:
        """Controlling High Blood Pressure (CBP)"""
        result = {"initial_population": False, "denominator": False, "numerator": False, "exclusion": False, "exclusion_reason": None, "evidence": None}
        if not patient: return result
        
        age = calculate_age(parse_date(patient['dob']), self.index_date)
        age_pass = 18 <= age <= 85
        if age_pass: result['initial_population'] = True
        
        map_cbp = self.code_maps["CMS165"]
        
        # Encounter must be a "Qualifying Encounter"
        # Since Synthea's canonical dataset does not reliably attach encounter CPT codes to the events.csv
        # we will use the standard metric fallback: did they have an encounter in the measurement year?
        has_encounter = any(parse_date(e['encounter_date']).year == self.index_date.year for e in encounters)
                
        if not age_pass or not has_encounter: 
            result['initial_population'] = False
            result['debug_reason'] = f"Failed IP: age_pass={age_pass}, has_encounter={has_encounter}"
            return result

        has_htn = False
        has_esrd = False
        has_pregnancy = False
        has_hospice_or_pallas = False
        has_frailty = False
        has_advanced_illness = False
        has_ltc = False
        
        for ev in events:
            cd = ev['code']
            edate = parse_date(ev['event_date'])
            
            # Essential Hypertension must overlap first 6 months
            # Simplified: Diagnosed on or before June 30 of measurement year
            if cd in map_cbp["Essential Hypertension_CODES"] and edate <= date(self.index_date.year, 6, 30):
                has_htn = True
                
            if edate.year > self.index_date.year: continue
            
            if cd in map_cbp["ESRD Exclusion_CODES"]: has_esrd = True
            if cd in map_cbp["Pregnancy Exclusion_CODES"] and edate.year == self.index_date.year: has_pregnancy = True
            
            if cd in map_cbp["Hospice Exclusion_CODES"] and edate.year == self.index_date.year: has_hospice_or_pallas = True
            if cd in map_cbp["Palliative Care Exclusion_CODES"] and edate.year == self.index_date.year: has_hospice_or_pallas = True
            
            if cd in map_cbp["Frailty Exclusion_CODES"] and edate.year == self.index_date.year: has_frailty = True
            
            # Advanced illness/dementia meds: year prior or during measurement year
            if (cd in map_cbp["Advanced Illness Exclusion_CODES"] or cd in map_cbp["Dementia Medications Exclusion_CODES"]) and self.index_date.year - 1 <= edate.year <= self.index_date.year:
                has_advanced_illness = True
                
            if cd in map_cbp["LTC Exclusion_CODES"]: has_ltc = True
                
        if not has_htn: 
            result['initial_population'] = False
            result['debug_reason'] = "Failed IP: No valid hypertension diagnosis found"
            
        if result['initial_population']:
            # Evaluate Exclusions
            if has_hospice_or_pallas:
                result['exclusion'], result['exclusion_reason'] = True, "Hospice or Palliative Care"
            elif has_esrd:
                result['exclusion'], result['exclusion_reason'] = True, "ESRD Diagnosis"
            elif has_pregnancy:
                result['exclusion'], result['exclusion_reason'] = True, "Pregnancy"
            elif age >= 66 and has_ltc:
                result['exclusion'], result['exclusion_reason'] = True, "Age >= 66 in LTC"
            elif 66 <= age <= 80 and has_frailty and has_advanced_illness:
                result['exclusion'], result['exclusion_reason'] = True, "Age 66-80 with Frailty and Advanced Illness"
            elif age >= 81 and has_frailty:
                result['exclusion'], result['exclusion_reason'] = True, "Age >= 81 with Frailty"
                
            if not result['exclusion']:
                result['denominator'] = True
                
        if result['denominator']:
            # Find closest date with BOTH readings
            readings_by_date = {}
            for ev in events:
                cd = ev['code']
                edate = parse_date(ev['event_date'])
                if edate.year == self.index_date.year and ev['value_num']:
                    try:
                        v = float(ev['value_num'])
                        if cd in map_cbp["Systolic Blood Pressure_CODES"]:
                            readings_by_date.setdefault(ev['event_date'], {'sys': [], 'sys_ev': [], 'dia': [], 'dia_ev': []})['sys'].append(v)
                            readings_by_date[ev['event_date']]['sys_ev'].append(ev)
                        elif cd in map_cbp["Diastolic Blood Pressure_CODES"]:
                            readings_by_date.setdefault(ev['event_date'], {'sys': [], 'sys_ev': [], 'dia': [], 'dia_ev': []})['dia'].append(v)
                            readings_by_date[ev['event_date']]['dia_ev'].append(ev)
                    except ValueError:
                        pass
                        
            valid_days = [d for d, r in readings_by_date.items() if len(r['sys']) > 0 and len(r['dia']) > 0]
            
            if valid_days:
                valid_days.sort(key=lambda x: parse_date(x), reverse=True)
                most_recent = valid_days[0]
                reads = readings_by_date[most_recent]
                
                min_sys = min(reads['sys'])
                min_dia = min(reads['dia'])
                
                # We return the event corresponding to the lowest systolic for auditability
                sys_idx = reads['sys'].index(min_sys)
                result['evidence'] = reads['sys_ev'][sys_idx]
                
                if min_sys < 140 and min_dia < 90:
                    result['numerator'] = True
                    
        return result

    def evaluate_cms122(self, patient: Dict, encounters: List[Dict], events: List[Dict]) -> Dict:
        """Diabetes: Hemoglobin A1c (HbA1c) Poor Control (> 9%)"""
        result = {"initial_population": False, "denominator": False, "numerator": False, "evidence": None}
        if not patient: return result
        
        age = calculate_age(parse_date(patient['dob']), self.index_date)
        if 18 <= age <= 75: result['initial_population'] = True
        
        has_encounter = any(parse_date(e['encounter_date']).year == self.index_date.year for e in encounters)
        if not result['initial_population'] or not has_encounter: return result

        map_hbd = self.code_maps["CMS122"]
        has_diabetes = False
        for ev in events:
            cd = ev['code']
            edate = parse_date(ev['event_date'])
            if edate.year > self.index_date.year: continue
            if cd in map_hbd["Diabetes_CODES"]:
                has_diabetes = True
                
        if not has_diabetes: result['initial_population'] = False
        if result['initial_population']:
            result['denominator'] = True
            
        if result['denominator']:
            a1c_readings = []
            for ev in events:
                cd = ev['code']
                edate = parse_date(ev['event_date'])
                if edate.year == self.index_date.year and cd in map_hbd["HbA1c Test_CODES"]:
                    a1c_readings.append(ev)
                    
            if not a1c_readings:
                # No reading means poor control automatically
                result['numerator'] = True
                result['evidence'] = None
            else:
                a1c_readings.sort(key=lambda x: parse_date(x['event_date']), reverse=True)
                latest = a1c_readings[0]
                if latest['value_num']:
                    try:
                        v = float(latest['value_num'])
                        if v > 9.0:
                            result['numerator'] = True
                    except: pass
                result['evidence'] = latest
        return result

    def evaluate_all(self, patient_id: str) -> Dict:
        patient, encounters, events = self.load_patient_data(patient_id)
        if not patient:
            return {"error": "Patient not found"}
            
        return {
            "CMS125": self.evaluate_cms125(patient, encounters, events),
            "CMS130": self.evaluate_cms130(patient, encounters, events),
            "CMS165": self.evaluate_cms165(patient, encounters, events),
            "CMS122": self.evaluate_cms122(patient, encounters, events)
        }
        
    def evaluate_patient(self, patient_id: str, measure_id: str) -> Tuple[bool, bool, Dict]:
        # Helper wrapper for LLM orchestration
        res = self.evaluate_all(patient_id)
        if "error" in res or measure_id not in res:
            return False, False, None
        
        eval_dict = res[measure_id]
        return eval_dict["denominator"], eval_dict["numerator"], eval_dict["evidence"]


if __name__ == "__main__":
    engine = GoldTruthEngine(index_date_str="2025-12-31")
    with open(os.path.join(CANONICAL_DIR, "patients.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            res = engine.evaluate_all(pid)
            print(f"Eval {pid[:8]}: CMS125 Denom={res['CMS125']['denominator']} Num={res['CMS125']['numerator']} | CMS130 Denom={res['CMS130']['denominator']} Num={res['CMS130']['numerator']}")
