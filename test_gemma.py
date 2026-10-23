#!/usr/bin/env python3
"""
test_gemma.py - Dedicated Diagnostic and Verification Tool for Gemma Integration.

Checks:
1. Local Ollama server status (http://localhost:11434)
2. Model availability ('gemma2:2b' or other configured tag)
3. Structured JSON extraction validation
4. Latency and token generation test
"""

import json
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:2b")


def check_server():
    print(f"[*] Step 1: Checking local Ollama endpoint at {OLLAMA_BASE_URL}...")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            print(f"[+] Ollama server is ONLINE.")
            print(f"[+] Available local models: {', '.join(models) if models else 'None installed yet'}")
            return models
        else:
            print(f"[-] Ollama returned HTTP status {resp.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        print("[-] FAILED: Ollama server is offline or unreachable.")
        print("    Run 'ollama serve' in your terminal or launch Ollama.app.")
        return None
    except Exception as e:
        print(f"[-] ERROR: {e}")
        return None


def test_generation(models):
    print(f"\n[*] Step 2: Testing Gemma generation with model '{LLM_MODEL}'...")
    matching = any(LLM_MODEL in m for m in models) if models else False
    if not matching:
        print(f"[!] Warning: Model '{LLM_MODEL}' not found in local models list.")
        print(f"    You can download it with: ollama pull {LLM_MODEL}")

    prompt_payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a factual extractor. Return ONLY valid JSON: {\"status\": \"ok\", \"model\": \"<name>\", \"message\": \"<brief test summary>\"}"
            },
            {
                "role": "user",
                "content": "Confirm that Gemma 2 is operational and ready to process local notifications."
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    start_time = time.time()
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=prompt_payload, timeout=60)
        duration = round(time.time() - start_time, 2)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        print(f"[+] Response received in {duration}s:")
        print(f"    Raw Output: {content}")
        
        parsed = json.loads(content)
        print(f"[+] JSON Validation: SUCCESS!")
        print(f"[+] Structured Output: {parsed}")
        print("\n=======================================================")
        print("🎉 ALL GEMMA INTEGRATION CHECKS PASSED!")
        print("=======================================================")
        return True
    except Exception as e:
        print(f"[-] Generation test failed: {e}")
        return False


def main():
    print("=======================================================")
    print("      InfoTamer (Sloth Agent) - Gemma Test Suite       ")
    print("=======================================================")
    models = check_server()
    if models is None:
        sys.exit(1)
    
    success = test_generation(models)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
