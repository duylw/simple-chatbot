import os
from unittest.mock import patch, MagicMock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from src.infrastructure.llm.gateway import (
    get_chat_model,
    get_structured_chat_model,
)
from src.infrastructure.llm.providers import (
    SUPPORTED_MODEL_CHOICES,
    DEFAULT_MODEL,
)
from src.domain.entities.agent import GuardrailEvaluation


def test_supported_model_choices():
    """Verify curated models contain Gemini, Gemma, Groq, OpenRouter, and OpenAI."""
    assert len(SUPPORTED_MODEL_CHOICES) >= 8
    model_ids = [m[0] for m in SUPPORTED_MODEL_CHOICES]
    assert "gemini-3.5-flash-lite" in model_ids
    assert "gemma-4-26b-a4b-it" in model_ids
    assert "groq/llama-3.3-70b-versatile" in model_ids
    assert "openrouter/deepseek/deepseek-r1:free" in model_ids
    assert "openai/gpt-4o-mini" in model_ids
    assert DEFAULT_MODEL == "gemini-3.5-flash-lite"


def test_get_chat_model_gemini():
    """Verify Google Gemini initialization."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyTestKey"}):
        model = get_chat_model("gemini-3.5-flash-lite", temperature=0.1)
        assert isinstance(model, ChatGoogleGenerativeAI)
        assert model.model == "gemini-3.5-flash-lite"
        assert model.temperature == 0.1


def test_get_chat_model_gemma():
    """Verify Google Gemma initialization."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyTestKey"}):
        model = get_chat_model("gemma-4-26b-a4b-it", temperature=0.2)
        assert isinstance(model, ChatGoogleGenerativeAI)
        assert model.model == "gemma-4-26b-a4b-it"


def test_get_chat_model_groq():
    """Verify Groq Cloud initialization with ChatOpenAI base_url."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_mock"}):
        model = get_chat_model("groq/llama-3.3-70b-versatile", temperature=0.0)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "llama-3.3-70b-versatile"
        assert "api.groq.com" in str(model.openai_api_base)
        # Verify effective temperature is set to 0.1 to avoid repetition loops
        assert model.temperature == 0.1


def test_get_chat_model_openrouter():
    """Verify OpenRouter initialization with custom headers."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}):
        model = get_chat_model("openrouter/deepseek/deepseek-r1:free", temperature=0.3)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "deepseek/deepseek-r1:free"
        assert "openrouter.ai" in str(model.openai_api_base)


def test_get_chat_model_missing_key_fallback():
    """Verify graceful fallback to default Gemini when provider key is missing."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyTestKey"}, clear=False):
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
        model = get_chat_model("groq/llama-3.3-70b-versatile")
        assert isinstance(model, ChatGoogleGenerativeAI)
        assert model.model == DEFAULT_MODEL


def test_get_structured_chat_model():
    """Verify structured output wrapper returns Runnable with schema bound."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyTestKey"}):
        structured_model = get_structured_chat_model(
            schema=GuardrailEvaluation,
            model_name="gemini-3.5-flash-lite",
            temperature=0.0,
        )
        assert structured_model is not None
