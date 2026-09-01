"""Agent Domain Entities & Evaluation Schemas."""
from dataclasses import dataclass, field
from typing import Optional, List, Any
from pydantic import BaseModel, Field


class GuardrailEvaluation(BaseModel):
    """Evaluation result from academic and security guardrail."""
    is_lecture_related: bool = Field(
        description="Boolean flag indicating whether the query is strictly relevant to academic lecture concepts."
    )
    reasoning: str = Field(
        description="Analytical justification for the decision in Vietnamese."
    )
    feedback: str = Field(
        description="Polite, student-facing message in Vietnamese."
    )


class AnswerGrade(BaseModel):
    """Quality and relevance grade of generated answer."""
    is_relevant: bool = Field(
        description="Is the answer relevant and helpful to the user query?"
    )
    suggestion: str = Field(
        default="",
        description="Short suggestion or keywords in Vietnamese/English for rewriting if not relevant."
    )
    reasoning: str = Field(
        default="",
        description="Short analytical reasoning for the grade in Vietnamese."
    )


@dataclass
class TemporalCitation:
    """Domain model representing a temporal citation in a lecture video."""
    video_name: str
    time_range: str
    start_seconds: int = 0
    end_seconds: int = 0
    slide_number: Optional[int] = None
    content_snippet: str = ""
    video_url: Optional[str] = None


@dataclass
class AgentResponseDomain:
    """Full domain response returned by Agent Orchestrator."""
    query: str
    rewritten_query: str = ""
    answer: str = ""
    sources: List[Any] = field(default_factory=list)
    citations: List[TemporalCitation] = field(default_factory=list)
    n_iterations: int = 0
    n_llm_calls: int = 0
    execution_time: float = 0.0
    guardrail_result: Optional[str] = None
