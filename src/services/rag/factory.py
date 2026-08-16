from .agent_graph import AgenticRagService
from .config import GraphConfig


def make_agentic_rag_service(
    bm25_retriever,
    vectordb_retriever,
    retriever_top_k: int = 5,
    use_hybrid: bool = True,
) -> AgenticRagService:
    """Factory function to create an instance of AgenticRagService with the provided retrievers and configuration."""
    
    # Create graph configuration (Intialize with default values or provided parameters)
    graph_config = GraphConfig(
        retriever_top_k=retriever_top_k,
        use_hybrid=use_hybrid
    )
    
    return AgenticRagService(
        bm25_retriever=bm25_retriever,
        vectordb_retriever=vectordb_retriever,
        graph_config=graph_config
    )