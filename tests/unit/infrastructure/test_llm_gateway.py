"""Unit tests for MultiProviderLLMGateway and Strategy Providers."""
import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.llm.gateway import MultiProviderLLMGateway, llm_gateway
from src.domain.entities.agent import GuardrailEvaluation


def test_llm_gateway_default_model_resolution():
    """Verify LLM Gateway resolves default Google Gemini model without prefix."""
    gateway = MultiProviderLLMGateway(default_model="gemini-3.5-flash-lite")
    with patch.object(gateway.providers["google"], "create_chat_model") as mock_create:
        mock_create.return_value = MagicMock()
        model = gateway.get_chat_model("gemini-3.5-flash-lite")
        assert model is not None
        mock_create.assert_called_once_with("gemini-3.5-flash-lite", 0.0)


def test_llm_gateway_groq_resolution_and_fallback():
    """Verify LLM Gateway calls Groq and gracefully falls back to Gemini when error occurs."""
    gateway = MultiProviderLLMGateway(default_model="gemini-3.5-flash-lite")

    with patch.object(gateway.providers["groq"], "create_chat_model", side_effect=ValueError("GROQ_API_KEY missing")):
        with patch.object(gateway.providers["google"], "create_chat_model") as mock_fallback:
            mock_fallback.return_value = MagicMock()
            model = gateway.get_chat_model("groq/llama-3.3-70b-versatile")
            assert model is not None
            mock_fallback.assert_called_once_with("gemini-3.5-flash-lite", 0.0)


def test_llm_gateway_structured_output():
    """Verify LLM Gateway configures structured output schema properly."""
    gateway = MultiProviderLLMGateway(default_model="gemini-3.5-flash-lite")
    mock_base_llm = MagicMock()
    mock_structured = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_structured

    with patch.object(gateway, "get_chat_model", return_value=mock_base_llm):
        result = gateway.get_structured_chat_model(GuardrailEvaluation, "gemini-3.5-flash-lite")
        assert result == mock_structured
        mock_base_llm.with_structured_output.assert_called_once_with(GuardrailEvaluation)
