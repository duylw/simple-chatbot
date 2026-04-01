from services.rag.tools import create_retriever_tool
from .nodes import (
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
from src.services.rag.state import ThreadState
from src.services.rag.context import Context

from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode

from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AgenticRagService:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vectordb_retriever: VectorStoreRetriever,
        graph_config: Optional[GraphConfig] = None
    ):
        self.bm25_retriever = bm25_retriever
        self.vectordb_retriever = vectordb_retriever
        self.graph_config = graph_config or GraphConfig()

    def _build_graph(self):
        # Build the graph using the provided nodes and configuration
        logging.info("Building the agentic RAG graph!")

        hybrid_search = create_retriever_tool(
            vectordb_retriever=self.vectordb_retriever,
            bm25_retriever=self.bm25_retriever,
            top_k=self.graph_config.retriever_top_k,
            use_hybrid=self.graph_config.use_hybrid,
            semantic_weight=self.graph_config.semantic_weight,
            bm25_weight=self.graph_config.bm25_weight
        )
        tools = [hybrid_search]

        workflow = StateGraph(ThreadState, context_schema=Context)
        workflow.add_node("query_guardrail", invoke_query_guardrail)
        workflow.add_node("out_of_scope_response", invoke_out_of_scope_response)
        workflow.add_node("query_rewrite", invoke_query_rewrite)
        workflow.add_node("get_relevant_documents", invoke_get_relevant_documents)
        workflow.add_node("search_tool", ToolNode(tools))
        workflow.add_node("generate_answer", invoke_generate_answer)
        workflow.add_node("grade_answer", invoke_grade_answer)
        workflow.add_node("response", invoke_response)

        workflow.set_entry_point("query_guardrail")

        workflow.add_conditional_edges(
            "query_guardrail",
            continue_after_guardrail,
            {"continue": "query_rewrite", "out_of_scope": "out_of_scope_response"}
        )

        workflow.add_edge("out_of_scope_response", END)
        workflow.add_edge("query_rewrite", "get_relevant_documents")

        workflow.add_conditional_edges(
            "get_relevant_documents",
            tools_condition,
            {"tools": "search_tool"}
        )
        workflow.add_edge("search_tool", "generate_answer")
        workflow.add_edge("generate_answer", "grade_answer")

        workflow.add_conditional_edges(
            "grade_answer",
            lambda state: state.get("routing_decision", "response"),
            {"response": "response", "rewrite_query": "query_rewrite"}
        )

        workflow.add_edge("response", END)

        completed_graph = workflow.compile()

        return completed_graph

    async def ask(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> dict:
        # Main method to handle user query and return answer
        return await self._execute_graph(query, model)

    async def _execute_graph(self, query: str, model: Optional[str]) -> dict:
        pass

    def _extract_answer(self, result: dict) -> str:
        pass

    def _extract_sources(self, result: dict) -> list:
        pass

    def _extract_reasoning(self, result: dict) -> str:
        pass