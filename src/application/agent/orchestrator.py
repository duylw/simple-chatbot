"""AgentOrchestrator: LangGraph State Machine, Routing and Execution Engine."""
import time
import uuid
import logging
from typing import Optional, Dict, Any, List
from collections.abc import AsyncGenerator

from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode

from src.application.agent.state import ThreadState, Context, GraphConfig
from src.application.agent.tools import create_retriever_tool
from src.application.agent.context_manager import ContextManager
from src.infrastructure.observability.tracer import get_tracer_callback
from src.infrastructure.observability.tracer import trace_agent_execution
from src.core.config import get_settings

from .nodes import (
    continue_after_guardrail,
    invoke_query_guardrail,
    invoke_out_of_scope_response,
    invoke_query_rewrite,
    invoke_get_relevant_documents,
    invoke_generate_answer,
    invoke_grade_answer,
    invoke_response,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Enterprise Agent Orchestrator compiling and executing the RAG State Machine."""

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        vectordb_retriever: Optional[VectorStoreRetriever] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        self.bm25_retriever = bm25_retriever
        self.vectordb_retriever = vectordb_retriever
        self.graph_config = graph_config or GraphConfig()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build and compile the LangGraph state machine workflow."""
        logger.info("Building AgentOrchestrator LangGraph workflow...")

        tools = create_retriever_tool(
            vectordb_retriever=self.vectordb_retriever,
            bm25_retriever=self.bm25_retriever,
            top_k=self.graph_config.retriever_top_k,
            use_hybrid=self.graph_config.use_hybrid,
            semantic_weight=self.graph_config.semantic_weight,
            bm25_weight=self.graph_config.bm25_weight,
        )
        if not isinstance(tools, list):
            tools = [tools]

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

        logger.info("Compiling AgentOrchestrator graph...")
        compiled = workflow.compile()
        logger.info("AgentOrchestrator graph compiled successfully!")
        return compiled

    async def ask(
        self,
        query: str,
        model: Optional[str] = None,
        trace_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow for a single user query."""
        return await self._execute_graph(query, model, trace_user_id)

    async def ask_streaming(
        self,
        query: str,
        model: Optional[str] = None,
        trace_user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream answer chunks in real-time."""
        result = await self._execute_graph(query, model, trace_user_id)
        answer = result.get("answer", "")
        if not answer:
            return

        chunk_size = 256
        for start in range(0, len(answer), chunk_size):
            yield answer[start : start + chunk_size]

    async def _execute_graph(
        self,
        query: str,
        model: Optional[str],
        trace_user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal execution engine with tracing and metrics calculation."""
        start_time = time.time()
        model_to_use = model or self.graph_config.llm_model

        initial_state: ThreadState = {
            "messages": [HumanMessage(content=query)],
            "n_iterations": 1,
            "n_llm_calls": 0,
            "original_query": None,
            "rewritten_query": [],
            "guardrail_result": None,
            "sources": [],
            "answer": None,
            "answer_grade": [],
            "routing_decision": None,
        }

        runtime_context = Context(
            llm_model=model_to_use,
            model_provider="google-genai",
            temperature=self.graph_config.temperature,
            retriever_top_k=self.graph_config.retriever_top_k,
            n_iterations=self.graph_config.n_iterations,
        )

        handler = get_tracer_callback()
        user_id_for_trace = trace_user_id or "anonymous"

        with trace_agent_execution(user_id=user_id_for_trace, session_id=str(uuid.uuid4())):
            result = await self.graph.ainvoke(
                initial_state,
                context=runtime_context,
                config={"callbacks": [handler]}
            )

        execution_time = time.time() - start_time
        logger.info(f"Agent execution finished in {execution_time:.2f}s")

        answer = ContextManager.extract_text_content(self._extract_answer(result))
        sources = self._extract_sources(result)
        n_iterations = result.get("n_iterations", 0)
        n_llm_calls = result.get("n_llm_calls", 0)

        raw_rewritten = result.get("rewritten_query", [])
        rewritten_query = ContextManager.extract_text_content(raw_rewritten[-1]) if raw_rewritten else ""

        guardrail_obj = result.get("guardrail_result")
        guardrail_result = guardrail_obj.reasoning if guardrail_obj and hasattr(guardrail_obj, "reasoning") else ""

        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources,
            "n_iterations": n_iterations,
            "n_llm_calls": n_llm_calls,
            "execution_time": execution_time,
            "guardrail_result": guardrail_result,
        }

    def _extract_answer(self, result: Dict[str, Any]) -> str:
        """Extract answer content from state dictionary."""
        if result.get("answer"):
            return ContextManager.extract_text_content(result["answer"])

        messages = result.get("messages", [])
        if not messages:
            return "No answer generated"

        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        return ContextManager.extract_text_content(content)

    def _extract_sources(self, result: Dict[str, Any]) -> List[Any]:
        """Extract sources from state or tool messages."""
        sources = result.get("sources", [])
        if not sources:
            messages = result.get("messages", [])
            sources = ContextManager.extract_sources_from_tool_messages(messages)
        return sources

    def get_graph_visualization(self) -> bytes:
        """Export LangGraph workflow PNG image bytes."""
        try:
            return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            raise
