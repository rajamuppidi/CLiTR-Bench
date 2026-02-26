import unittest
from datetime import date
from gold_truth_engine import GoldTruthEngine

class TestGoldTruthEngine(unittest.TestCase):
    
    def setUp(self):
        # Initialize engine with an arbitrary index date matching typical end-of-year logic
        self.engine = GoldTruthEngine(index_date_str="2024-12-31")

    def test_cms125_breast_cancer_screening_eligibility(self):
        # Patient exactly 40 years old at index date, female
        patient_40F = {"sex": "F", "dob": "1984-06-15"}
        # Patient 74 years old at index date, female
        patient_74F = {"sex": "F", "dob": "1950-01-01"}
        # Patient 39 years old at index date, female (Out of bounds)
        patient_39F = {"sex": "F", "dob": "1985-06-15"}
        # Patient 50 years old at index date, male (Out of bounds)
        patient_50M = {"sex": "M", "dob": "1974-06-15"}

        encounters_active = [{"encounter_date": "2024-05-10"}]
        
        res_40F = self.engine.evaluate_cms125(patient_40F, encounters_active, [])
        self.assertTrue(res_40F["initial_population"])
        self.assertTrue(res_40F["denominator"])

        res_74F = self.engine.evaluate_cms125(patient_74F, encounters_active, [])
        self.assertTrue(res_74F["initial_population"])
        
        res_39F = self.engine.evaluate_cms125(patient_39F, encounters_active, [])
        self.assertFalse(res_39F["initial_population"])
        
        res_50M = self.engine.evaluate_cms125(patient_50M, encounters_active, [])
        self.assertFalse(res_50M["initial_population"])

    def test_cms125_exclusions(self):
        patient = {"sex": "F", "dob": "1970-01-01"} # 54 years old
        enc_active = [{"encounter_date": "2024-05-10"}]

        # 1. Bilateral Mastectomy directly
        events_bilateral = [{"code": "Z90.13", "event_date": "2020-01-01"}]
        res_bl = self.engine.evaluate_cms125(patient, enc_active, events_bilateral)
        self.assertTrue(res_bl["exclusion"])
        self.assertFalse(res_bl["denominator"])

        # 2. Right and Left Mastectomies on different dates
        events_combined = [
            {"code": "Z90.11", "event_date": "2018-05-10"}, # Right
            {"code": "0HTU0ZZ", "event_date": "2021-08-15"} # Left
        ]
        res_combo = self.engine.evaluate_cms125(patient, enc_active, events_combined)
        self.assertTrue(res_combo["exclusion"])
        self.assertFalse(res_combo["denominator"])

        # 3. Only one side removed (Should NOT exclude from screening the other side)
        events_single = [{"code": "Z90.11", "event_date": "2018-05-10"}] # Right only
        res_single = self.engine.evaluate_cms125(patient, enc_active, events_single)
        self.assertFalse(res_single["exclusion"])
        self.assertTrue(res_single["denominator"])

    def test_cms125_numerator_lookback(self):
        patient = {"sex": "F", "dob": "1970-01-01"} 
        enc_active = [{"encounter_date": "2024-05-10"}]

        # Mammogram inside 27 month window (e.g. 1 year prior)
        events_valid = [{"code": "77067", "event_date": "2023-12-01"}]
        res_valid = self.engine.evaluate_cms125(patient, enc_active, events_valid)
        self.assertTrue(res_valid["numerator"])

        # Mammogram exactly 27 months prior (Approx Oct 2, 2022 due to Leap Year 2024)
        events_boundary = [{"code": "77067", "event_date": "2022-10-02"}]
        res_boundary = self.engine.evaluate_cms125(patient, enc_active, events_boundary)
        self.assertTrue(res_boundary["numerator"])

        # Mammogram older than 27 months (e.g., 3 years prior)
        events_expired = [{"code": "77067", "event_date": "2021-01-01"}]
        res_expired = self.engine.evaluate_cms125(patient, enc_active, events_expired)
        self.assertFalse(res_expired["numerator"])

    def test_cms165_blood_pressure(self):
        patient = {"sex": "M", "dob": "1960-01-01"}
        enc_active = [{"encounter_date": "2024-05-10"}]
        
        # Missing hypertension diagnosis -> should fail initial pop
        res_no_htn = self.engine.evaluate_cms165(patient, enc_active, [])
        self.assertFalse(res_no_htn["initial_population"])

        htn_event = {"code": "I10", "event_date": "2020-01-01"}

        # HTN, but BP is HIGH (145/95)
        bp_high = [
            htn_event,
            {"code": "8480-6", "event_date": "2024-11-01", "value_num": "145"}, # Sys
            {"code": "8462-4", "event_date": "2024-11-01", "value_num": "95"}   # Dia
        ]
        res_high = self.engine.evaluate_cms165(patient, enc_active, bp_high)
        self.assertTrue(res_high["initial_population"])
        self.assertTrue(res_high["denominator"])
        self.assertFalse(res_high["numerator"])

        # HTN, and BP is CONTROLLED (135/85)
        bp_controlled = [
            htn_event,
            {"code": "8480-6", "event_date": "2024-11-01", "value_num": "135"},
            {"code": "8462-4", "event_date": "2024-11-01", "value_num": "85"}
        ]
        res_controlled = self.engine.evaluate_cms165(patient, enc_active, bp_controlled)
        self.assertTrue(res_controlled["numerator"])
        
    def test_cms122_hba1c_poor_control(self):
        patient = {"sex": "M", "dob": "1960-01-01"}
        enc_active = [{"encounter_date": "2024-05-10"}]
        
        # Missing diabetes -> should fail initial pop
        res_no_diab = self.engine.evaluate_cms122(patient, enc_active, [])
        self.assertFalse(res_no_diab["initial_population"])

        diabetes_event = {"code": "E11.9", "event_date": "2020-01-01"}

        # Patient has Diabetes, but NO test performed in the year. 
        # HEDIS Rule: No test = Poor Control (>9%) = TRUE
        res_notest = self.engine.evaluate_cms122(patient, enc_active, [diabetes_event])
        self.assertTrue(res_notest["denominator"])
        self.assertTrue(res_notest["numerator"]) # Poor control met

        # Patient has Diabetes, Test shows 10.5 (Poor control)
        a1c_bad = [diabetes_event, {"code": "4548-4", "event_date": "2024-06-01", "value_num": "10.5"}]
        res_bad = self.engine.evaluate_cms122(patient, enc_active, a1c_bad)
        self.assertTrue(res_bad["numerator"])

        # Patient has Diabetes, Test shows 7.5 (Good control)
        # Therefore, "Poor control > 9%" logic should be FALSE
        a1c_good = [diabetes_event, {"code": "4548-4", "event_date": "2024-06-01", "value_num": "7.5"}]
        res_good = self.engine.evaluate_cms122(patient, enc_active, a1c_good)
        self.assertFalse(res_good["numerator"])

if __name__ == '__main__':
    unittest.main()
