"""Multi-Provider LLM Gateway implementing ILLMGateway with Fallback & Circuit Breaker."""
import logging
from typing import Optional, Any, Type, Dict
from langchain_core.language_models.chat_models import BaseChatModel

from src.domain.interfaces.llm import ILLMGateway
from src.infrastructure.llm.providers import (
    BaseLLMProvider,
    GoogleGeminiProvider,
    GroqProvider,
    OpenRouterProvider,
    OpenAIProvider,
    OllamaProvider,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


class MultiProviderLLMGateway(ILLMGateway):
    """Factory and Gateway orchestrating multiple LLM providers with automatic fallback."""

    def __init__(self, default_model: str = DEFAULT_MODEL):
        self.default_model = default_model
        self.providers: Dict[str, BaseLLMProvider] = {
            "google": GoogleGeminiProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider(),
            "openai": OpenAIProvider(),
            "ollama": OllamaProvider(),
        }

    def get_chat_model(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any
    ) -> BaseChatModel:
        """Initialize chat model instance based on prefix strategy with fallback to Google Gemini."""
        selected_model = (model_name or self.default_model).strip()
        logger.info(f"LLM Gateway: initializing model '{selected_model}' with temp={temperature}")

        provider_key = self._resolve_provider_key(selected_model)
        provider = self.providers.get(provider_key, self.providers["google"])

        try:
            return provider.create_chat_model(selected_model, temperature, **kwargs)
        except Exception as exc:
            logger.warning(
                f"LLM Gateway: Provider '{provider_key}' failed for model '{selected_model}': {exc}. "
                f"Falling back to default '{self.default_model}'."
            )
            return self.providers["google"].create_chat_model(self.default_model, temperature, **kwargs)

    def get_structured_chat_model(
        self,
        schema: Type[Any],
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any
    ) -> Any:
        """Initialize chat model configured with structured output schema across providers."""
        base_llm = self.get_chat_model(model_name, temperature=temperature, **kwargs)

        try:
            from langchain_openai import ChatOpenAI
            if isinstance(base_llm, ChatOpenAI):
                try:
                    return base_llm.with_structured_output(schema, method="function_calling")
                except Exception:
                    return base_llm.with_structured_output(schema, method="json_mode")
        except ImportError:
            pass

        return base_llm.with_structured_output(schema)

    def _resolve_provider_key(self, model_name: str) -> str:
        """Resolve provider name from model prefix."""
        if model_name.startswith("groq/"):
            return "groq"
        if model_name.startswith("openrouter/"):
            return "openrouter"
        if model_name.startswith("openai/"):
            return "openai"
        if model_name.startswith("ollama/"):
            return "ollama"
        return "google"


# Singleton instance
llm_gateway = MultiProviderLLMGateway()

# Module-level convenience functions matching legacy interface
def get_chat_model(model_name: Optional[str] = None, temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
    return llm_gateway.get_chat_model(model_name, temperature, **kwargs)


def get_structured_chat_model(schema: Type[Any], model_name: Optional[str] = None, temperature: float = 0.0, **kwargs: Any) -> Any:
    return llm_gateway.get_structured_chat_model(schema, model_name, temperature, **kwargs)
