import os
import csv
import json
from typing import Dict, List, Tuple

TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_DIR = os.path.join(TARGET_DIR, "data_generation", "output", "canonical")

class RepresentationRenderer:
    def __init__(self):
        self._cache_patients = {}
        self._cache_encounters = {}
        self._cache_events = {}
        self._loaded = False
        self._target_cohort = set()
        
    def _ensure_loaded(self, patient_id: str):
        if patient_id in self._cache_patients or patient_id in self._target_cohort:
            return
            
        patient_csv = os.path.join(CANONICAL_DIR, "patients.csv")
        encounter_csv = os.path.join(CANONICAL_DIR, "encounters.csv")
        events_csv = os.path.join(CANONICAL_DIR, "events.csv")
        if not os.path.exists(patient_csv): return
        
        # When cache misses, doing a full file scan is too expensive for single missing patients.
        # But if we must load, we selectively cache the missing pid.
        with open(patient_csv, 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] == patient_id:
                    self._cache_patients[row['patient_id']] = row
                    break
        with open(encounter_csv, 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] == patient_id:
                    self._cache_encounters.setdefault(row['patient_id'], []).append(row)
        with open(events_csv, 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] == patient_id:
                    self._cache_events.setdefault(row['patient_id'], []).append(row)
        
        if patient_id in self._cache_events:
            self._cache_events[patient_id].sort(key=lambda x: x['event_date'])
            
        self._target_cohort.add(patient_id)

    def preload_cohort(self, cohort_ids: List[str]):
        """Preload 3GB CSV datasets cleanly in memory for just the matching N=500 patients."""
        cohort_set = set(cohort_ids)
        missing = [p for p in cohort_set if p not in self._cache_patients]
        if not missing: return
        missing_set = set(missing)
        
        patient_csv = os.path.join(CANONICAL_DIR, "patients.csv")
        encounter_csv = os.path.join(CANONICAL_DIR, "encounters.csv")
        events_csv = os.path.join(CANONICAL_DIR, "events.csv")
        
        import logging
        logging.info("  [Renderer] Loading patients.csv...")
        with open(patient_csv, 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] in missing_set:
                    self._cache_patients[row['patient_id']] = row
                    
        logging.info("  [Renderer] Loading encounters.csv...")
        with open(encounter_csv, 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] in missing_set:
                    self._cache_encounters.setdefault(row['patient_id'], []).append(row)
                    
        logging.info("  [Renderer] Loading 2.6GB events.csv...")
        with open(events_csv, 'r') as f:
            for row in csv.DictReader(f):
                if row['patient_id'] in missing_set:
                    self._cache_events.setdefault(row['patient_id'], []).append(row)
                    
        for p in missing_set:
            if p in self._cache_events:
                self._cache_events[p].sort(key=lambda x: x['event_date'])
        
        self._target_cohort.update(missing_set)
        
    def get_patient_data(self, patient_id: str) -> Tuple[Dict, List[Dict], List[Dict]]:
        self._ensure_loaded(patient_id)
        return (
            self._cache_patients.get(patient_id),
            self._cache_encounters.get(patient_id, []),
            self._cache_events.get(patient_id, [])
        )

    def render_structured(self, patient_id: str, format_type='csv') -> str:
        """
        Renders the entire patient's longitudinal history into a clean, 
        machine-readable structured format (CSV string or JSON string).
        """
        patient, encounters, events = self.get_patient_data(patient_id)
        if not patient:
            return "Patient not found."
            
        # Truncate longitudinal data for token limits (12K for Groq, higher for OpenAI)
        # Use 4-year lookback to cover ALL HEDIS measure windows safely:
        # - CMS125: 27-month mammography lookback (needs Oct 2021+)
        # - CMS130: 10-year colonoscopy lookback (needs 2015+ but recent matters most)
        # - CMS165: Recent BP readings (1-2 years)
        # - CMS122: Recent HbA1c (1-2 years)
        events = [e for e in events if e.get('event_date', '') >= '2020-01-01']
        if len(events) > 400:
            events = events[-400:]  # Keep most recent 400 events
            
        if format_type == 'json':
            structured_doc = {
                "patient_demographics": patient,
                "encounters": encounters,
                "clinical_events": events
            }
            return json.dumps(structured_doc, indent=2)
            
        elif format_type == 'csv':
            # Create a combined unified list format
            lines = [f"PATIENT DATA:\nID: {patient['patient_id']} | DOB: {patient['dob']} | Sex: {patient['sex']}\n"]
            lines.append("CLINICAL EVENTS (Chronological):")
            lines.append("Date, Event Type, Code System, Code, Value")
            for e in events:
                val = f"{e['value_num']} {e.get('unit', '')}" if e['value_num'] else "N/A"
                lines.append(f"{e['event_date']}, {e.get('event_type', 'UNKNOWN')}, {e['code_system']}, {e['code']}, {val}")
                
            return "\n".join(lines)
        else:
            raise ValueError("Supported structured formats are 'csv' or 'json'")

    def render_note(self, patient_id: str) -> str:
        """
        Renders the patient's data into a synthetic, semi-structured clinical note summary.
        This tests the LLM's capacity for unstructured NLP extraction over tabular reasoning.
        """
        patient, encounters, events = self.get_patient_data(patient_id)
        if not patient:
            return "Patient not found."
            
        # Truncate longitudinal data for token limits (12K for Groq, higher for OpenAI)
        # Use 4-year lookback to cover ALL HEDIS measure windows safely:
        # - CMS125: 27-month mammography lookback (needs Oct 2021+)
        # - CMS130: 10-year colonoscopy lookback (needs 2015+ but recent matters most)
        # - CMS165: Recent BP readings (1-2 years)
        # - CMS122: Recent HbA1c (1-2 years)
        events = [e for e in events if e.get('event_date', '') >= '2020-01-01']
        if len(events) > 400:
            events = events[-400:]  # Keep most recent 400 events
            
        note_sections = []
        note_sections.append(f"### CLINICAL SUMMARY NOTE ###")
        note_sections.append(f"PATIENT_ID: {patient['patient_id'][:8]}-XXX")
        note_sections.append(f"DOB: {patient['dob']}")
        note_sections.append(f"SEX: {patient['sex']}")
        note_sections.append("-" * 40)
        
        # Group events logically
        conditions = [e for e in events if e.get('event_type') == 'CONDITION']
        procedures = [e for e in events if e.get('event_type') in ['PROCEDURE', 'OBSERVATION'] and not e['value_num']]
        labs_vitals = [e for e in events if e.get('event_type') == 'OBSERVATION' and e['value_num']]
        medications = [e for e in events if e.get('event_type') == 'MEDICATION']

        note_sections.append("ACTIVE PROBLEM LIST:")
        if conditions:
            for c in conditions:
                note_sections.append(f"- [{c['code_system']}: {c['code']}] (Diagnosed: {c['event_date']})")
        else:
            note_sections.append("None known.")
        
        note_sections.append("\nSURGICAL & MEDICAL HISTORY (Procedures/Imaging):")
        if procedures:
            for p in procedures:
                note_sections.append(f"- {p['event_date']}: [{p['code_system']}: {p['code']}]")
        else:
            note_sections.append("None recorded.")
            
        note_sections.append("\nRECENT VITALS & LABS:")
        if labs_vitals:
            # Maybe just grab the last 5 years to prevent massive notes, or dump all
            recent_labs = labs_vitals[-25:] # Limit to most recent 25 for prompt overflow protection
            for lab in recent_labs:
                note_sections.append(f"- {lab['event_date']}: [{lab['code_system']}: {lab['code']}] = {lab['value_num']} {lab.get('unit', '')}")
            if len(labs_vitals) > 25:
                note_sections.append(f"  ... [and {len(labs_vitals)-25} older lab results omitted for brevity]")
        else:
            note_sections.append("None available.")
            
        note_sections.append("\nMEDICATIONS:")
        if medications:
            for m in medications:
                 note_sections.append(f"- Dispensed {m['event_date']}: [{m['code_system']}: {m['code']}]")
        else:
            note_sections.append("No active medications.")
            
        note_sections.append("\n--- END OF RECORD ---")
        return "\n".join(note_sections)

if __name__ == "__main__":
    r = RepresentationRenderer()
    
    # Just render the first patient as an example debug output
    with open(os.path.join(CANONICAL_DIR, "patients.csv"), 'r') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row:
            pid = first_row["patient_id"]
            
            print("====================================")
            print("STRUCTURED (CSV) PREVIEW:")
            print("====================================")
            # Take just first few lines
            print("\n".join(r.render_structured(pid, 'csv').split('\n')[:15]))
            
            print("\n====================================")
            print("CLINICAL NOTE PREVIEW:")
            print("====================================")
            print(r.render_note(pid)[:1000]) # Cap output
