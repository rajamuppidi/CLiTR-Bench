import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

class MetricsEngine:
    """
    Evaluates LLM predictions against the Deterministic Gold Truth Engine outputs
    across multiple dimensions of clinical reasoning.
    """
    def __init__(self):
        # Tracking metrics for detailed parity
        self.metrics = {
            "total_attempted": 0,   # every patient row seen (incl. parse failures)
            "total_evaluated": 0,   # rows with a successfully parsed prediction
            "parse_failures": 0,    # rows dropped due to unparseable/degenerate output

            # Decision Accuracy
            "denominator_accuracy": 0,
            "numerator_accuracy": 0,
            "overall_decision_accuracy": 0,
            
            # Guideline Logic Fidelity (True Postives/Negatives for exclusions)
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            
            # Auditability
            "audit_evidence_matched": 0,
            "audit_evidence_hallucinated": 0
        }

    def evaluate_batch(self, gold_truths: List[Dict[str, Any]], llm_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes a list of gold truth evaluations and a parallel list of LLM predictions
        and computes the core benchmark metrics.
        
        Assumed structure of single gold truth element (from GoldTruthEngine):
        {
          "patient_id": "123",
          "measure": "CMS125",
          "denominator": True/False,
          "numerator": True/False,
          "evidence": {"date": "...", "code": "..."} # Or None
        }
        
        Assumed structure of single LLM prediction (from LLMRunner):
        {
          "patient_id": "123",
          "measure": "CMS125",
          "success": True, # Did JSON parsing succeed?
          "parsed": {
              "denominator_met": True/False,
              "numerator_met": True/False,
              "audit_evidence": "String description..."
          }
        }
        """
        
        if len(gold_truths) != len(llm_predictions):
            logging.warning("Mismatched length between gold truths and LLM predictions!")
            
        failed_ids = []
        for gt, pred in zip(gold_truths, llm_predictions):
            self.metrics["total_attempted"] += 1
            # Record (do not silently drop) invalid API responses so a lost row is auditable.
            if not pred.get("success", False) or "parsed" not in pred:
                self.metrics["parse_failures"] += 1
                failed_ids.append(pred.get("patient_id", gt.get("patient_id", "?")))
                continue

            self.metrics["total_evaluated"] += 1
            
            llm_res = pred["parsed"]
            
            # ===============================
            # 1. Decision Accuracy Core
            # ===============================
            denom_match = (gt["denominator"] == llm_res.get("denominator_met"))
            num_match = (gt["numerator"] == llm_res.get("numerator_met"))
            
            if denom_match: self.metrics["denominator_accuracy"] += 1
            if num_match: self.metrics["numerator_accuracy"] += 1
            if denom_match and num_match: self.metrics["overall_decision_accuracy"] += 1
                
            # ===============================
            # 2. Logic Fidelity (using Numerator as standard target)
            # ===============================
            if gt["numerator"] and llm_res.get("numerator_met"):
                self.metrics["true_positives"] += 1
            elif not gt["numerator"] and not llm_res.get("numerator_met"):
                self.metrics["true_negatives"] += 1
            elif not gt["numerator"] and llm_res.get("numerator_met"):
                 self.metrics["false_positives"] += 1
            elif gt["numerator"] and not llm_res.get("numerator_met"):
                 self.metrics["false_negatives"] += 1
                 
            # ===============================
            # 3. Auditability Match Rate (AMR)
            # ===============================
            # Definition: for gold-compliant patients, the model's citation must contain the
            # gold evidence event's date or code (the most recent in-window qualifying event);
            # for gold-non-compliant patients, the model must cite nothing. AMR therefore
            # measures agreement with the gold evidence event, not merely whether the cited
            # event exists somewhere in the record — record-level grounding is computed
            # separately in experiments/analysis/error_analysis.py.
            gt_evidence = gt.get("evidence")
            llm_evidence = str(llm_res.get("audit_evidence", "")).lower()
            
            if gt_evidence is not None:
                # Naive text match against date or code to see if LLM extracted the right event
                target_date = gt_evidence.get('event_date', '')
                target_code = gt_evidence.get('code', '')
                
                if target_date in llm_evidence or target_code in llm_evidence:
                    self.metrics["audit_evidence_matched"] += 1
                else:
                    self.metrics["audit_evidence_hallucinated"] += 1
            else:
                # Gold truth found NO evidence (num is false).
                # If LLM claims to have evidence (and not "none", "n/a"), it's a hallucination
                if llm_evidence and llm_evidence not in ["none", "n/a", "null", "false"]:
                    self.metrics["audit_evidence_hallucinated"] += 1
                else:
                    self.metrics["audit_evidence_matched"] += 1 # Correctly identified lack thereof

        if self.metrics["parse_failures"] > 0:
            logging.warning(
                "%d of %d rows had UNPARSEABLE model output and were excluded from accuracy. "
                "Patient IDs: %s",
                self.metrics["parse_failures"], self.metrics["total_attempted"], failed_ids
            )

        return self.calculate_final_scores()

    def calculate_final_scores(self) -> Dict[str, float]:
        """Calculates final percentages and F1 scores based on raw counts."""
        total = max(self.metrics["total_evaluated"], 1)
        attempted = max(self.metrics["total_attempted"], 1)

        # Accuracy Rates
        denom_acc_rate = self.metrics["denominator_accuracy"] / total
        num_acc_rate = self.metrics["numerator_accuracy"] / total
        overall_acc_rate = self.metrics["overall_decision_accuracy"] / total
        audit_match_rate = self.metrics["audit_evidence_matched"] / total

        # Format compliance: fraction of attempted patients that produced parseable output.
        compliance_rate = self.metrics["total_evaluated"] / attempted

        # F1 Score Components
        tp = self.metrics["true_positives"]
        fp = self.metrics["false_positives"]
        fn = self.metrics["false_negatives"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results = {
            # --- Coverage / reproducibility (proves no row was silently lost) ---
            "Total_Attempted": self.metrics["total_attempted"],
            "Total_Evaluated": self.metrics["total_evaluated"],
            "Parse_Failures": self.metrics["parse_failures"],
            "Format_Compliance_Rate": round(compliance_rate * 100, 2),

            # --- Primary signal: the numerator decision (denominator is non-discriminative) ---
            "Numerator_Accuracy": round(num_acc_rate * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall_Sensitivity": round(recall * 100, 2),
            "F1_Score": round(f1_score * 100, 2),
            "Overall_Exact_Match_Accuracy": round(overall_acc_rate * 100, 2),

            # --- Secondary / context metrics ---
            "Denominator_Accuracy": round(denom_acc_rate * 100, 2),
            "Denominator_Note": "Cohort is 100% denominator-positive; this metric is non-discriminative (a model that always answers True scores 100). Lead with Numerator/F1.",
            "Auditability_Match_Rate": round(audit_match_rate * 100, 2),
            "Hallucinated_Evidence_Count": self.metrics["audit_evidence_hallucinated"]
        }

        return results


def rescore_jsonl(results_path: str) -> Dict[str, Any]:
    """
    Re-computes final scores from a saved per-patient results .jsonl (the format written by
    run_experiment.py), with NO API calls. Each line holds {"gold_truth": ..., "llm_prediction": ...}.
    Lets us regenerate corrected metrics for an existing run without re-paying for inference.
    """
    gold_truths, llm_predictions = [], []
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold_truths.append(rec["gold_truth"])
            llm_predictions.append(rec["llm_prediction"])

    engine = MetricsEngine()
    scores = engine.evaluate_batch(gold_truths, llm_predictions)
    print(json.dumps(scores, indent=2))
    return scores


if __name__ == "__main__":
    import sys
    # Re-score an existing run: python metrics_engine.py --rescore path/to/results_*.jsonl
    if len(sys.argv) >= 3 and sys.argv[1] == "--rescore":
        rescore_jsonl(sys.argv[2])
        sys.exit(0)

    # Pre-flight Mock Test
    engine = MetricsEngine()
    
    mock_gold_truths = [
        {"patient_id": "1", "measure": "CMS125", "denominator": True, "numerator": True, "evidence": {"event_date": "2024-05-10", "code": "77067"}},
        {"patient_id": "2", "measure": "CMS125", "denominator": False, "numerator": False, "evidence": None}
    ]
    
    mock_llm_preds = [
        {"patient_id": "1", "measure": "CMS125", "success": True, "parsed": {"denominator_met": True, "numerator_met": True, "audit_evidence": "Mammogram performed on 2024-05-10 with CPT 77067"}}, # Perfect
        {"patient_id": "2", "measure": "CMS125", "success": True, "parsed": {"denominator_met": True, "numerator_met": False, "audit_evidence": "None"}} # Denominator Hallucination (False Positive on Denom)
    ]
    
    print("Pre-flight testing Metrics Engine...")
    scores = engine.evaluate_batch(mock_gold_truths, mock_llm_preds)
    print(json.dumps(scores, indent=2))
