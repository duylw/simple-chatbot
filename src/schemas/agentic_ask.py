from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from langchain_core.documents import Document

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu hỏi thắc mắc của sinh viên về bài giảng"
    )
    model: Optional[str] = Field(
        None,
        description="Tùy chọn mô hình LLM (ví dụ: gemini-2.5-flash-lite)"
    )

class AgenticAskResponse(BaseModel):
    query: str
    rewritten_query: str
    answer: str
    sources: list[Document]
    n_iterations: int
    execution_time: float
    guardrail_result: Optional[str] = None
    n_llm_calls: int

    model_config = ConfigDict(from_attributes=True)