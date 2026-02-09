"""LLM API client with retry logic for OpenAI and OpenRouter."""
import os
import time
import json
import openai
import httpx
from config import MODELS, OPENAI_API_KEY, OPENROUTER_API_KEY


def _call_openai(model_id, prompt, temperature=0, max_tokens=256):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _call_openrouter(model_id, prompt, temperature=0, max_tokens=256):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def call_llm(model_name, prompt, max_retries=3):
    """Call an LLM with retry logic. Returns the response text."""
    cfg = MODELS[model_name]
    api = cfg["api"]
    model_id = cfg["model_id"]
    temperature = cfg["temperature"]

    for attempt in range(max_retries):
        try:
            if api == "openai":
                return _call_openai(model_id, prompt, temperature)
            elif api == "openrouter":
                return _call_openrouter(model_id, prompt, temperature)
            else:
                raise ValueError(f"Unknown API: {api}")
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [Retry {attempt+1}/{max_retries}] {model_name}: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [FAILED] {model_name}: {e}")
                return None


def call_llm_for_judge(model_name, system_prompt, user_prompt, max_retries=3):
    """Call an LLM with system prompt for judge tasks."""
    cfg = MODELS[model_name]
    api = cfg["api"]
    model_id = cfg["model_id"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            if api == "openai":
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0,
                    max_tokens=256,
                )
                return resp.choices[0].message.content.strip()
            elif api == "openrouter":
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 256,
                }
                with httpx.Client(timeout=60) as client:
                    resp = client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                return None


if __name__ == "__main__":
    # Quick test
    result = call_llm("gpt-4.1", "Say hello in 5 words.")
    print(f"GPT-4.1: {result}")
    result = call_llm("claude-sonnet-4.5", "Say hello in 5 words.")
    print(f"Claude: {result}")
