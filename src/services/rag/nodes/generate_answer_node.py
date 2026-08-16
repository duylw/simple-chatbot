from src.services.rag.state import (
    ThreadState
)
from src.services.rag.prompts import (
    answer_generation_prompt,
)
from src.services.rag.nodes.utils import (
    get_latest_query,
    get_latest_context,
    format_context,
    merge_temporal_chunks,
    extract_sources_from_tool_messages,
)
from src.services.rag.context import Context

from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler

from typing import Dict
import logging

logger = logging.getLogger(__name__)

async def invoke_generate_answer(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    logger.info("NODE: generate_answer")
    
    updates = {}

    query = state.get("original_query") or get_latest_query(state.get("messages", []))
    
    raw_sources = state.get("sources", [])
    if not raw_sources:
        raw_sources = extract_sources_from_tool_messages(state.get("messages", []))
    
    # Merge contiguous/overlapping chunks for clean citation boundaries
    merged_sources = merge_temporal_chunks(raw_sources)
    updates["sources"] = merged_sources
    
    formated_context = format_context(merged_sources)
    prompt = answer_generation_prompt.format(query=query, context=formated_context)
    
    llm = ChatGoogleGenerativeAI(model=runtime.context.llm_model, temperature=runtime.context.temperature)
    res = await llm.ainvoke(prompt)
    
    return {
        **updates,
        "answer": res.content,
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }