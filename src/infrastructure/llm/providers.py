"""LLM Provider Strategy implementations for multi-provider support."""
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

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


class BaseLLMProvider(ABC):
    """Abstract Strategy Provider for LLM instances."""

    @abstractmethod
    def create_chat_model(self, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        pass


class GoogleGeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini / Gemma models."""

    def create_chat_model(self, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI
        clean_name = model_name.removeprefix("google/")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model=clean_name,
            temperature=temperature,
            google_api_key=api_key,
            **kwargs,
        )


class GroqProvider(BaseLLMProvider):
    """Provider for Groq Cloud LPU models."""

    def create_chat_model(self, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        real_model = model_name.removeprefix("groq/")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment.")

        effective_temp = 0.1 if temperature == 0.0 else temperature
        return ChatOpenAI(
            model=real_model,
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=effective_temp,
            max_tokens=4096,
            **kwargs,
        )


class OpenRouterProvider(BaseLLMProvider):
    """Provider for OpenRouter multi-model gateway."""

    def create_chat_model(self, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        real_model = model_name.removeprefix("openrouter/")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment.")

        return ChatOpenAI(
            model=real_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Temporal RAG QA System",
            },
            **kwargs,
        )


class OpenAIProvider(BaseLLMProvider):
    """Provider for official OpenAI models."""

    def create_chat_model(self, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        real_model = model_name.removeprefix("openai/")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment.")

        return ChatOpenAI(
            model=real_model,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )


class OllamaProvider(BaseLLMProvider):
    """Provider for local Ollama instances."""

    def create_chat_model(self, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        real_model = model_name.removeprefix("ollama/")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return ChatOpenAI(
            model=real_model,
            api_key="ollama",
            base_url=ollama_url,
            temperature=temperature,
            **kwargs,
        )
