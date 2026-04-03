from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
import operator


from langchain_core.documents import Document
from langchain_core.messages import AnyMessage

class QueryEvaluation(BaseModel):
    is_lecture_related: bool = Field(
            description="Boolean flag indicating whether the user's query is strictly relevant to the academic content, concepts, or logistics of the specific lecture."
    )

    reasoning: str = Field(
        description="A short analytical explanation justifying the relevance decision, highlighting specific academic keywords or themes identified in the query, written in Vietnamese."
    )

    feedback: str = Field(
        description="A short, polite, student-facing message providing guidance in Vietnamese. If irrelevant, it explains the assistant's scope; if relevant, it confirms the intent to search the lecture materials. Must be in Vietnamese."
    )

class AnswerGrade(BaseModel):
    is_relevant: bool = Field(description="Is the answer relevant to the query?")
    suggestion: str = Field(description="Short suggestion for improving the answer in Vietnamese")
    reasoning: str = Field(description="Short reasoning for the answer in Vietnamese")


class ThreadState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] # Chat messages

    original_query: Optional[str]
    rewritten_query: Annotated[List[str], operator.add]
    user_query_grade: QueryEvaluation

    source: List[Document]
    answer: Optional[str]
    answer_grade: Annotated[List[AnswerGrade], operator.add]
    routing_decision: Optional[str]

    n_iterations: int
    n_llm_calls: int