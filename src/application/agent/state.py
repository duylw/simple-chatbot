"""LangGraph ThreadState & Runtime Context definitions."""
from typing import TypedDict, Annotated, List, Optional
from dataclasses import dataclass
import operator

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from src.domain.entities.agent import GuardrailEvaluation, AnswerGrade


@dataclass
class Context:
    """Immutable runtime configuration context for Agent execution."""
    llm_model: str = "gemini-3.5-flash-lite"
    model_provider: str = "google-genai"
    temperature: float = 0.0
    retriever_top_k: int = 10
    n_iterations: int = 3


@dataclass
class GraphConfig:
    """Configuration parameters for compiling and executing the Agent workflow."""
    llm_model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    retriever_top_k: int = 10
    n_iterations: int = 2
    use_hybrid: bool = True
    semantic_weight: float = 1.0
    bm25_weight: float = 1.0


class ThreadState(TypedDict):
    """LangGraph execution thread state."""
    messages: Annotated[list[AnyMessage], operator.add]

    original_query: Optional[str]
    rewritten_query: Annotated[List[str], operator.add]
    guardrail_result: Optional[GuardrailEvaluation]

    sources: List[Document]
    answer: Optional[str]
    answer_grade: Annotated[List[AnswerGrade], operator.add]
    routing_decision: Optional[str]

    n_iterations: int
    n_llm_calls: int
