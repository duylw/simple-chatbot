"""LLM Gateway Interface (Port) for Domain Layer."""
from abc import ABC, abstractmethod
from typing import Optional, Any, Type


class ILLMGateway(ABC):
    """Port for Multi-Provider LLM Gateway (Gemini, Groq, OpenAI, OpenRouter, Ollama)."""

    @abstractmethod
    def get_chat_model(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any
    ) -> Any:
        """Get standard chat model instance for the given provider/model name."""
        pass

    @abstractmethod
    def get_structured_chat_model(
        self,
        schema: Type[Any],
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any
    ) -> Any:
        """Get chat model instance configured with structured output schema."""
        pass
