"""
Groq API Provider - Free tier with fast inference
Sign up at: https://console.groq.com/
Free tier: 1,000 requests/day on Llama 3.3 70B
"""

import os
import json
from groq import Groq

def call_groq_api(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_retries: int = 3
) -> str:
    """
    Call Groq API for free, fast inference

    Available models (free tier):
    - llama-3.3-70b-versatile (best for reasoning)
    - llama-3.3-8b-text (faster, smaller)
    - qwen2.5-32b-instruct (good for structured tasks)
    - deepseek-r1-distill-llama-70b (reasoning specialist)

    Args:
        system_prompt: System message
        user_prompt: User message with patient data
        model: Groq model name
        temperature: Sampling temperature (0.0 for deterministic)
        max_retries: Number of retry attempts

    Returns:
        str: Model response text
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Get free key at https://console.groq.com/")

    client = Groq(api_key=api_key)

    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model,
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                stream=False
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Groq API failed after {max_retries} attempts: {str(e)}")
            print(f"Groq API attempt {attempt + 1} failed, retrying...")
            continue

    return ""


# Quick test
if __name__ == "__main__":
    test_response = call_groq_api(
        system_prompt="You are a helpful medical AI assistant.",
        user_prompt="Explain what CMS125 breast cancer screening measures.",
        model="llama-3.3-70b-versatile"
    )
    print(test_response)
