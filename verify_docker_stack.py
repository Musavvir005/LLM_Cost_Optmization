#!/usr/bin/env python3
"""End-to-end health checks for AD-BoN when run inside its Docker image."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import requests

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
ROOT = Path("/app") if Path("/app").is_dir() else Path.cwd()
ROUTER_DIR = ROOT / "models" / "saved_router"
CACHE_FILE = ROOT / "semantic_cache.json"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")


def status(name: str, check: Callable[[], str]) -> bool:
    try:
        detail = check()
    except Exception as exc:  # Keep all checks running so failures are actionable.
        print(f"{RED}[FAIL]{RESET} {name}: {exc}")
        return False
    print(f"{GREEN}[PASS]{RESET} {name}: {detail}")
    return True


def check_router() -> str:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not ROUTER_DIR.is_dir():
        raise FileNotFoundError(f"router directory not mounted: {ROUTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(ROUTER_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(ROUTER_DIR, local_files_only=True)
    model.to("cpu").eval()
    with torch.no_grad():
        output = model(**tokenizer("AD-BoN container health check", return_tensors="pt"))
    return f"PyTorch {torch.__version__}; CPU inference logits {tuple(output.logits.shape)}"


def check_ollama() -> str:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    names = [m.get("name", "") for m in response.json().get("models", [])]
    if not any(name == "phi3" or name.startswith("phi3:") for name in names):
        raise RuntimeError(f"Ollama is reachable but phi3 is absent (available: {names or 'none'})")
    return f"phi3 available via {OLLAMA_URL}"


def required_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_") or value.startswith("your-"):
        raise RuntimeError(f"{name} is missing or still contains the example value")
    return value


def check_gemini() -> str:
    import google.generativeai as genai

    genai.configure(api_key=required_key("GEMINI_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    response = genai.GenerativeModel(model_name).generate_content(
        "Reply with OK.", generation_config={"max_output_tokens": 1}, request_options={"timeout": 20}
    )
    if not getattr(response, "text", "").strip():
        raise RuntimeError("Gemini returned no text")
    return f"{model_name} responded (minimal request)"


def check_groq() -> str:
    key = required_key("GROQ_API_KEY")
    model = os.getenv("GROQ_VERIFY_MODEL", "llama3-70b-8192")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 1, "temperature": 0},
        timeout=20,
    )
    response.raise_for_status()
    if not response.json().get("choices"):
        raise RuntimeError("Groq response contained no choices")
    return f"{model} responded (minimal request)"


def check_cache() -> str:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    original = CACHE_FILE.read_bytes() if CACHE_FILE.exists() else None
    marker = {"adbon_docker_verification": True, "timestamp": time.time()}
    try:
        # Validate write/read of JSON while preserving the cache exactly afterwards.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CACHE_FILE.parent, delete=False) as temp:
            json.dump([marker], temp)
            temporary_path = Path(temp.name)
        temporary_path.replace(CACHE_FILE)
        loaded = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if loaded != [marker]:
            raise RuntimeError("cache content did not round-trip")
    finally:
        if original is None:
            CACHE_FILE.unlink(missing_ok=True)
        else:
            CACHE_FILE.write_bytes(original)
    return f"read/write succeeded at {CACHE_FILE} (original data restored)"


def main() -> int:
    print(f"{YELLOW}AD-BoN Docker stack verification{RESET}")
    results = [
        status("1. PyTorch and trained Hugging Face router", check_router),
        status("2. Host Ollama phi3 connectivity", check_ollama),
        status("3a. Gemini credentials and egress", check_gemini),
        status("3b. Groq credentials and egress", check_groq),
        status("4. Semantic-cache volume persistence", check_cache),
    ]
    passed = sum(results)
    print(f"\n{GREEN if all(results) else RED}{passed}/{len(results)} checks passed{RESET}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
