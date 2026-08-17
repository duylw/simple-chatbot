from __future__ import annotations

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", API_BASE_URL)
APP_TITLE = "Agentic RAG Assistant"
DEFAULT_PORT = 7860
MAX_PORT_ATTEMPTS = 10
REQUEST_TIMEOUT = 120.0
GRADIO_PORT = os.getenv("GRADIO_PORT")

DEFAULT_MODEL = "gemini-3.5-flash-lite"

SUPPORTED_MODEL_CHOICES = [
    # Google Gemini & Gemma
    ("gemini-3.5-flash-lite", "[Google] Gemini 3.5 Flash Lite"),
    ("gemini-3.5-flash", "[Google] Gemini 3.5 Flash"),
    ("gemma-4-26b-a4b-it", "[Google] Gemma 4 26B"),
    ("gemma-4-31b-it", "[Google] Gemma 4 31B"),
    
    # Groq (Ultra-Fast Inference - Free Tier)
    ("groq/llama-3.3-70b-versatile", "[Groq] Llama 3.3 70B Versatile"),
    ("groq/deepseek-r1-distill-llama-70b", "[Groq] DeepSeek R1 Distill 70B"),
    ("groq/llama-3.1-8b-instant", "[Groq] Llama 3.1 8B Instant"),
    
    # OpenRouter (Free Tier Models)
    ("openrouter/deepseek/deepseek-r1:free", "[OpenRouter] DeepSeek R1 Full 671B"),
    ("openrouter/meta-llama/llama-3.3-70b-instruct:free", "[OpenRouter] Meta Llama 3.3 70B"),
    ("openrouter/qwen/qwen-2.5-72b-instruct:free", "[OpenRouter] Qwen 2.5 72B Instruct"),
    
    # OpenAI & Compatible
    ("openai/gpt-4o-mini", "[OpenAI] GPT-4o Mini"),
    ("openai/gpt-4o", "[OpenAI] GPT-4o"),
]