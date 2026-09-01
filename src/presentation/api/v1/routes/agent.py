"""Agent Question-Answering & Reasoning API Routes."""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import StreamingResponse

from src.application.dtos.agent import (
    QueryRequestDTO,
    AgentResponseDTO,
    ModelChoiceDTO,
)
from src.application.use_cases.ask_question import AskQuestionUseCase
from src.application.use_cases.stream_question import StreamQuestionUseCase
from src.domain.entities.user import User
from src.presentation.di.agent import (
    get_ask_question_use_case,
    get_stream_question_use_case,
    get_agent_orchestrator,
)
from src.presentation.di.auth import get_current_user
from src.infrastructure.llm.providers import SUPPORTED_MODEL_CHOICES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent & RAG"])


@router.post("/ask", response_model=AgentResponseDTO, summary="Ask question to Agent (Batch)")
async def ask_agent(
    request: QueryRequestDTO,
    use_case: AskQuestionUseCase = Depends(get_ask_question_use_case),
    current_user: Optional[User] = Depends(get_current_user),
) -> AgentResponseDTO:
    """Submit a question to the Temporal RAG Agent and receive a comprehensive synthesized answer."""
    user_id = current_user.email if current_user else "anonymous"
    return await use_case.execute(request, user_id=user_id)


@router.post("/ask/stream", summary="Ask question to Agent (SSE Stream)")
async def ask_agent_stream(
    request: QueryRequestDTO,
    use_case: StreamQuestionUseCase = Depends(get_stream_question_use_case),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Stream synthesized answer tokens in real-time via Server-Sent Events."""
    user_id = current_user.email if current_user else "anonymous"

    async def event_generator():
        async for chunk in use_case.execute(request, user_id=user_id):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/models", response_model=List[ModelChoiceDTO], summary="List supported LLM models")
async def list_models() -> List[ModelChoiceDTO]:
    """List all supported Multi-Provider LLM models."""
    return [ModelChoiceDTO(id=model_id, label=label) for model_id, label in SUPPORTED_MODEL_CHOICES]


@router.get("/graph", summary="Export LangGraph State Diagram PNG")
async def get_graph_diagram():
    """Retrieve Mermaid visualization diagram of the underlying LangGraph workflow."""
    orchestrator = get_agent_orchestrator()
    img_bytes = orchestrator.get_graph_visualization()
    return Response(content=img_bytes, media_type="image/png")
