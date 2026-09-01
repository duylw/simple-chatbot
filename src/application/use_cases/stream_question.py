"""Stream Question Use Case."""
from collections.abc import AsyncGenerator
from typing import Optional

from src.application.agent.orchestrator import AgentOrchestrator
from src.application.dtos.agent import QueryRequestDTO


class StreamQuestionUseCase:
    """Orchestrates real-time SSE token streaming from AgentOrchestrator."""

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator

    async def execute(self, request: QueryRequestDTO, user_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream chunks of answer."""
        async for chunk in self.orchestrator.ask_streaming(
            query=request.question,
            model=request.model,
            trace_user_id=user_id
        ):
            yield chunk
