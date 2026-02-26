# CLiTR-Bench Standardized Prompt Templates

This directory houses the frozen YAML prompts that will be piped into the target LLM API engine.

```yaml
version: 1.0

prompts:
  # Base prompt providing zero-shot instructions on predicting measure compliance
  zero_shot_base: |
    You are an expert Clinical Quality Measurement Assistant. Your task is to analyze the following longitudinal patient record and determine if the patient is compliant with the specified HEDIS measure for the measurement year ending strictly on 2025-12-31.
    
    Target Measure: {measure_name} ({measure_id})
    Task: Use your knowledge of the NCQA HEDIS criteria to evaluate this measure. 
    1. Determine if the patient belongs in the Denominator (Age/Sex eligible AND free of Exclusions).
    2. Determine if the patient belongs in the Numerator (Received the necessary care within the correct lookback timeframe).
    
    Output strictly a JSON block containing the following keys:
    - `denominator_met`: (boolean true/false)
    - `numerator_met`: (boolean true/false)
    - `audit_evidence`: (string identifying the Date and exact code/description of the procedure that justifies the Numerator, or "None" if failed).

    === PATIENT RECORD (Format: {record_format}) ===
    {patient_data}
    ================================================

  # Guideline supplied prompt providing the specific deterministic logic as context to the LLM (In-Context Learning/RAG style)
  guideline_supplied: |
    You are an expert Clinical Quality Measurement Assistant. Your task is to analyze the patient record and determine compliance against the following provided Guideline Logic. The measurement year ends strictly on 2025-12-31.

    Target Measure: {measure_name} ({measure_id})

    === REQUIRED HEDIS GUIDELINE LOGIC ===
    {guideline_text}
    ======================================

    Read the above guidelines carefully. Pay strict attention to exclusion codes and looking back exact timeframes (e.g. 27 months, 10 years, or measurement-year).

    Output strictly a JSON block containing the following keys:
    - `denominator_met`: (boolean true/false)
    - `numerator_met`: (boolean true/false)
    - `audit_evidence`: (string identifying the Date and exact code/description of the procedure that justifies the Numerator, or "None" if failed).

    === PATIENT RECORD (Format: {record_format}) ===
    {patient_data}
    ================================================

  # Chain-of-thought prompt encouraging explicit temporal reasoning step-by-step
  zero_shot_cot: |
    You are an expert Clinical Quality Measurement Assistant. Your task is to analyze the following longitudinal patient record and determine if the patient is compliant with the specified HEDIS measure for the measurement year ending strictly on 2025-12-31.

    Target Measure: {measure_name} ({measure_id})

    IMPORTANT: Before providing your final answer, think step-by-step through your reasoning:

    Step 1: DENOMINATOR ELIGIBILITY
    - What are the age/sex requirements for this measure?
    - Does this patient meet those requirements on 2025-12-31?
    - Are there any exclusion criteria? Does the patient have any exclusions?

    Step 2: NUMERATOR COMPLIANCE (only if denominator is met)
    - What is the required screening/procedure for this measure?
    - What is the exact lookback timeframe? (Calculate the valid date range carefully)
    - Search the patient record for relevant procedures within that timeframe
    - Identify the specific date and code if found

    Step 3: FINAL DETERMINATION
    - Based on steps 1 and 2, provide your final answer

    After your reasoning, output strictly a JSON block containing the following keys:
    - `denominator_met`: (boolean true/false)
    - `numerator_met`: (boolean true/false)
    - `audit_evidence`: (string identifying the Date and exact code/description of the procedure that justifies the Numerator, or "None" if failed).

    === PATIENT RECORD (Format: {record_format}) ===
    {patient_data}
    ================================================

```
