"""Agent and RAG Dependency Providers."""
from fastapi import Depends

from src.application.agent.orchestrator import AgentOrchestrator
from src.application.agent.state import GraphConfig
from src.application.use_cases.ask_question import AskQuestionUseCase
from src.application.use_cases.stream_question import StreamQuestionUseCase
from src.infrastructure.vectorstore.chroma_adapter import make_vector_db_retriever
from src.infrastructure.search.bm25_adapter import make_bm25_retriever

# Global cached orchestrator instance
_orchestrator_instance = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Provide singleton AgentOrchestrator instance with initialized retrievers."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        try:
            bm25_ret = make_bm25_retriever()
            vector_ret = make_vector_db_retriever()
            _orchestrator_instance = AgentOrchestrator(
                bm25_retriever=bm25_ret,
                vectordb_retriever=vector_ret,
                graph_config=GraphConfig(),
            )
        except Exception:
            # Fallback if DB/Vector services are not yet populated
            _orchestrator_instance = AgentOrchestrator(
                bm25_retriever=None,
                vectordb_retriever=None,
                graph_config=GraphConfig(),
            )
    return _orchestrator_instance


def get_ask_question_use_case(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> AskQuestionUseCase:
    """Provide AskQuestionUseCase."""
    return AskQuestionUseCase(orchestrator)


def get_stream_question_use_case(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> StreamQuestionUseCase:
    """Provide StreamQuestionUseCase."""
    return StreamQuestionUseCase(orchestrator)
