from src.services.rag.nodes import (
    continue_after_guardrail,
    invoke_query_guardrail,
    invoke_out_of_scope_response,
    invoke_query_rewrite,
    invoke_get_relevant_documents,
    invoke_rerank,
    invoke_generate_answer,
    invoke_grade_answer,
    invoke_response
)

from src.services.rag.config import GraphConfig

from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever

from typing import Optional

class AgenticRagService:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        embedding_retriever: VectorStoreRetriever,
        graph_config: Optional[GraphConfig] = None
    ):
        self.bm25_retriever = bm25_retriever
        self.embedding_retriever = embedding_retriever
        self.graph_config = graph_config or GraphConfig()

    pass