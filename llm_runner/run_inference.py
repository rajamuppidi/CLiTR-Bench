import os
import json
import yaml
import time
from typing import Dict, List, Any
import openai

TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_FILE = os.path.join(TARGET_DIR, "prompts", "templates.md")

class LLMRunner:
    """
    Orchestrates the connection between the compiled patient representations
    (from renderers.py) and external LLM APIs (e.g., OpenAI, Claude, Gemini).
    """
    def __init__(self, model_name: str = "gpt-4-turbo", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.prompts = self._load_prompts()

        # Initialize client (OpenAI, OpenRouter, or Groq)
        self.client = None
        self.provider = None

        # Check if we should use Groq (FREE, FAST!)
        if self.model_name.startswith("groq/"):
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                self.provider = "groq"
                # Strip groq/ prefix for actual model name
                self.model_name = self.model_name.replace("groq/", "")

        # Check Gemini BEFORE generic slash check (gemini/ prefix would match slash rule)
        elif "gemini" in self.model_name.lower():
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                try:
                    from google import genai
                    # Strip gemini/ prefix if present
                    model_name_clean = self.model_name.replace("gemini/", "")
                    self.client = genai.Client(api_key=api_key)
                    self.model_name = model_name_clean
                    self.provider = "gemini"
                except ImportError:
                    print("[WARNING] google-genai not installed. Run: pip install google-genai")
            else:
                print("[WARNING] GOOGLE_API_KEY not found in environment for Gemini.")

        # Check if we should use OpenRouter (generic slash-separated model names)
        elif "/" in self.model_name or "openrouter" in self.model_name.lower():
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                self.provider = "openrouter"
        # Otherwise, check if we should use default OpenAI
        elif "gpt" in self.model_name.lower() or "o1" in self.model_name.lower() or "o3" in self.model_name.lower():
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                self.provider = "openai"


    def _load_prompts(self) -> Dict[str, str]:
        """Parses the YAML blocks from the templates.md file"""
        prompts = {}
        if not os.path.exists(PROMPTS_FILE):
             return prompts
             
        with open(PROMPTS_FILE, 'r') as f:
            content = f.read()
            # Extract YAML block
            if "```yaml" in content:
                yaml_content = content.split("```yaml")[1].split("```")[0]
                data = yaml.safe_load(yaml_content)
                prompts = data.get('prompts', {})
        return prompts

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Routes inference payload to the dynamically selected foundation model.
        """
        if self.client and self.provider != "gemini":  # OpenAI-compatible providers only
            # Call actual API endpoint
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Build base kwargs
                    kwargs = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": self.temperature,
                    }

                    # Provider-specific configurations
                    if self.provider == "openai" and "gpt" in self.model_name:
                        # OpenAI supports JSON mode
                        kwargs["response_format"] = {"type": "json_object"}

                    elif self.provider == "groq":
                        # Groq supports JSON mode on most models
                        kwargs["max_tokens"] = 2048
                        kwargs["top_p"] = 1

                    elif self.provider == "openrouter":
                        # OpenRouter: Ban Venice (BYOK required on free tier)
                        kwargs["extra_body"] = {"provider": {"ignore": ["Venice", "venice"]}}

                    response = self.client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content

                except Exception as e:
                    # Catch 402, 429 rate limits, or 401 authentication upstream failures safely
                    print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(3)  # Exponential backoff
                    else:
                        return json.dumps({"error": str(e)})

        elif self.provider == "gemini" and self.client:
            # Gemini API via google.genai (new SDK)
            # Free tier: gemini-1.5-flash-8b → 1500 req/day, 1M TPM
            from google.genai import types as genai_types
            max_retries = 5
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt,
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=self.temperature,
                        )
                    )
                    return response.text
                except Exception as e:
                    err_str = str(e)
                    print(f"Gemini API Error (Attempt {attempt+1}/{max_retries}): {err_str[:200]}")
                    # Parse retryDelay from 429 response if present
                    retry_delay = 15  # default backoff
                    if "retryDelay" in err_str:
                        import re
                        match = re.search(r'retryDelay.*?(\d+)', err_str)
                        if match:
                            retry_delay = int(match.group(1)) + 2
                    elif "429" in err_str:
                        retry_delay = 15 * (attempt + 1)  # linear backoff for quota
                    if attempt < max_retries - 1:
                        print(f"  Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                    else:
                        return json.dumps({"error": err_str[:300]})
        else:
            # Fallback mock for local testing
            time.sleep(0.5)
            return json.dumps({
                "denominator_met": True,
                "numerator_met": False,
                "audit_evidence": "None"
            })

    def evaluate_patient(self, 
                         patient_representation: str, 
                         measure_id: str, 
                         measure_name: str, 
                         prompt_style: str = "zero_shot_base",
                         format_type: str = "clinical_note",
                         guideline_logic: str = "") -> Dict[str, Any]:
        """
        Executes a structured query against the LLM to determine measure compliance.
        Format Type indicates what strings are provided (csv, json, clinical_note).
        """
        
        template = self.prompts.get(prompt_style)
        if not template:
            raise ValueError(f"Prompt style {prompt_style} not found in templates.")
            
        # Format the template with the injected data
        if prompt_style == "zero_shot_base" or prompt_style == "zero_shot_cot":
            prompt = template.format(
                measure_id=measure_id,
                measure_name=measure_name,
                record_format=format_type,
                patient_data=patient_representation
            )
        elif prompt_style == "guideline_supplied":
            prompt = template.format(
                measure_id=measure_id,
                measure_name=measure_name,
                guideline_text=guideline_logic,
                record_format=format_type,
                patient_data=patient_representation
            )
        else:
            prompt = f"Executing {prompt_style} over \n {patient_representation}"

        # In a real environment, we'd have a system prompt instructing the model to act as the JSON evaluator
        system_msg = "You are a Clinical Intelligence system returning only parseable JSON."
        
        # Dispatch to the model inference layer
        raw_response = self._call_llm_api(system_prompt=system_msg, user_prompt=prompt)

        
        # Attempt to parse json
        try:
            # DeepSeek R1 and Free models sometimes output markdown wrappers like ```json
            import re
            clean_response = raw_response.strip()
            # Remove <think>...</think> blocks from R1
            clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL).strip()
            
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[-1].split("```")[0].strip()
            elif clean_response.startswith("```"):
                lines = clean_response.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                clean_response = "\n".join(lines).strip()
                
            parsed = json.loads(clean_response)
            return {
                "success": True,
                "raw_output": raw_response,
                "parsed": parsed
            }
        except Exception as e:
            return {
                "success": False,
                "raw_output": raw_response,
                "error": f"Failed to parse LLM valid JSON: {str(e)}"
            }

if __name__ == "__main__":
    runner = LLMRunner()
    print("Pre-flight testing LLM orchestrator...")
    res = runner.evaluate_patient(
        patient_representation="Patient is a 55 year old female. She had a mammogram on 2024-05-10.",
        measure_id="CMS125",
        measure_name="Breast Cancer Screening",
        prompt_style="zero_shot_base",
        format_type="clinical_note"
    )
    print("Mock Result:", json.dumps(res, indent=2))
