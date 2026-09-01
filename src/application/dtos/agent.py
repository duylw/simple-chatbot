"""Agent DTO schemas for API request and response."""
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequestDTO(BaseModel):
    """Payload for asking a question to the Agent."""
    question: str = Field(..., min_length=1, max_length=2000, description="The user's academic question.")
    model: Optional[str] = Field(None, description="Optional LLM model override.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Cơ chế Self-Attention trong Transformer hoạt động như thế nào?",
                "model": "gemini-3.5-flash-lite"
            }
        }
    }


class SourceCitationDTO(BaseModel):
    """DTO representing retrieved and merged video transcript source."""
    video_name: str
    timestamp: float
    time_range: str
    content: str
    slide_number: Optional[int] = None
    similarity_score: Optional[float] = None


class AgentResponseDTO(BaseModel):
    """Structured response from the Agent RAG pipeline."""
    query: str
    rewritten_query: str
    answer: str
    sources: List[SourceCitationDTO]
    n_iterations: int
    n_llm_calls: int
    execution_time: float
    guardrail_result: Optional[str] = None


class ModelChoiceDTO(BaseModel):
    """Supported LLM choice item."""
    id: str
    label: str
