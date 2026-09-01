"""Ask Question Use Case."""
import logging
from typing import Optional

from src.application.agent.orchestrator import AgentOrchestrator
from src.application.agent.context_manager import ContextManager
from src.application.dtos.agent import QueryRequestDTO, AgentResponseDTO, SourceCitationDTO

logger = logging.getLogger(__name__)


class AskQuestionUseCase:
    """Orchestrates query submission to the Agent and mapping to response DTO."""

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator

    async def execute(self, request: QueryRequestDTO, user_id: Optional[str] = None) -> AgentResponseDTO:
        """Execute the question-answering workflow."""
        logger.info(f"AskQuestionUseCase: Executing query '{request.question[:60]}...'")
        result = await self.orchestrator.ask(
            query=request.question,
            model=request.model,
            trace_user_id=user_id,
        )

        sources_dto = []
        raw_sources = result.get("sources", [])
        for doc in raw_sources:
            meta = doc.metadata if hasattr(doc, "metadata") else (doc if isinstance(doc, dict) else {})
            content = doc.page_content if hasattr(doc, "page_content") else str(meta.get("content", ""))

            start_t = float(meta.get("start_time", meta.get("timestamp", 0) or 0))
            time_range = meta.get("time_range")
            if not time_range:
                dur = float(meta.get("duration", 15) or 15)
                time_range = f"{ContextManager.format_timestamp(start_t)} - {ContextManager.format_timestamp(start_t + dur)}"

            sources_dto.append(
                SourceCitationDTO(
                    video_name=meta.get("video_name") or meta.get("source") or "Video Bài Giảng",
                    timestamp=start_t,
                    time_range=time_range,
                    content=content.strip(),
                    slide_number=meta.get("slide_number"),
                    similarity_score=meta.get("score"),
                )
            )

        return AgentResponseDTO(
            query=result.get("query", request.question),
            rewritten_query=result.get("rewritten_query", ""),
            answer=result.get("answer", ""),
            sources=sources_dto,
            n_iterations=result.get("n_iterations", 1),
            n_llm_calls=result.get("n_llm_calls", 0),
            execution_time=result.get("execution_time", 0.0),
            guardrail_result=result.get("guardrail_result"),
        )
