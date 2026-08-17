import os
import logging
from typing import Optional, Dict, Any, List
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# Curated list of popular and free models
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


def get_chat_model(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any
) -> BaseChatModel:
    """Factory function to initialize and return a BaseChatModel instance based on model name prefix.

    Supported prefixes:
    - 'groq/<model>' -> ChatOpenAI targeting Groq API
    - 'openrouter/<model>' -> ChatOpenAI targeting OpenRouter API
    - 'openai/<model>' -> ChatOpenAI targeting OpenAI API
    - 'ollama/<model>' -> ChatOpenAI targeting local Ollama
    - 'gemini-...' or default -> ChatGoogleGenerativeAI
    """
    selected_model = (model_name or DEFAULT_MODEL).strip()
    logger.info(f"Initializing Chat Model: '{selected_model}' with temperature={temperature}")

    # 1. Groq Cloud (Ultra Fast LPU)
    if selected_model.startswith("groq/"):
        real_model = selected_model.removeprefix("groq/")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY is not set in environment! Fallback to Google Gemini.")
            return _get_gemini_model(DEFAULT_MODEL, temperature)

        try:
            from langchain_openai import ChatOpenAI
            # Use small temperature (0.1) if 0.0 to prevent greedy repetitive looping in Llama-3 models
            effective_temp = 0.1 if temperature == 0.0 else temperature
            return ChatOpenAI(
                model=real_model,
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=effective_temp,
                max_tokens=4096,
                **kwargs
            )
        except ImportError:
            logger.error("langchain-openai is not installed, falling back to Google Gemini")
            return _get_gemini_model(DEFAULT_MODEL, temperature)

    # 2. OpenRouter (Multi-model free tier)
    elif selected_model.startswith("openrouter/"):
        real_model = selected_model.removeprefix("openrouter/")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY is not set in environment! Fallback to Google Gemini.")
            return _get_gemini_model(DEFAULT_MODEL, temperature)

        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=real_model,
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Temporal RAG QA System"
                },
                **kwargs
            )
        except ImportError:
            logger.error("langchain-openai is not installed, falling back to Google Gemini")
            return _get_gemini_model(DEFAULT_MODEL, temperature)

    # 3. OpenAI
    elif selected_model.startswith("openai/"):
        real_model = selected_model.removeprefix("openai/")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY is not set in environment! Fallback to Google Gemini.")
            return _get_gemini_model(DEFAULT_MODEL, temperature)

        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=real_model,
                api_key=openai_api_key,
                temperature=temperature,
                **kwargs
            )
        except ImportError:
            logger.error("langchain-openai is not installed, falling back to Google Gemini")
            return _get_gemini_model(DEFAULT_MODEL, temperature)

    # 4. Local Ollama
    elif selected_model.startswith("ollama/"):
        real_model = selected_model.removeprefix("ollama/")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=real_model,
                api_key="ollama",
                base_url=ollama_url,
                temperature=temperature,
                **kwargs
            )
        except ImportError:
            logger.error("langchain-openai is not installed, falling back to Google Gemini")
            return _get_gemini_model(DEFAULT_MODEL, temperature)

    # 5. Default Google Gemini
    else:
        clean_name = selected_model.removeprefix("google/")
        return _get_gemini_model(clean_name, temperature)


def _get_gemini_model(model_name: str, temperature: float) -> BaseChatModel:
    """Internal helper to initialize Google Gemini / Gemma chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=google_api_key
    )


def get_structured_chat_model(
    schema: type,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any
):
    """Factory function to initialize a chat model with structured output compatibility across all providers (Groq, OpenAI, Gemini, OpenRouter).

    Uses method='function_calling' for ChatOpenAI / Groq to avoid 400 errors with strict json_schema.
    """
    base_llm = get_chat_model(model_name, temperature=temperature, **kwargs)

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

