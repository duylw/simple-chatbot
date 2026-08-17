from pydantic import BaseModel, Field
from src.core.config import Settings, get_settings

class GraphConfig(BaseModel):
    """Configuration for the entire graph execution.

    """
    llm_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-2-preview"
    temperature: float = 0.0
    retriever_top_k: int = 10
    n_iterations: int = 3
    use_hybrid: bool = True
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3

    settings: Settings = Field(default_factory=get_settings)