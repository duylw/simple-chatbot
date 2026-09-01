"""Generate Answer Node: Synthesize pedagogic answer from lecture slides and transcripts."""
import logging
from typing import Dict
from langgraph.runtime import Runtime

from src.application.agent.state import ThreadState, Context
from src.application.agent.prompts import answer_generation_prompt
from src.application.agent.context_manager import ContextManager
from src.infrastructure.llm.gateway import get_chat_model

logger = logging.getLogger(__name__)


async def invoke_generate_answer(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    """Generate final answer connecting slides and transcripts."""
    logger.info("NODE: generate_answer")
    updates = {}

    query = state.get("original_query") or ContextManager.get_latest_query(state.get("messages", []))

    raw_sources = state.get("sources", [])
    if not raw_sources:
        raw_sources = ContextManager.extract_sources_from_tool_messages(state.get("messages", []))

    # Merge contiguous/overlapping chunks for clean citation boundaries
    merged_sources = ContextManager.merge_temporal_chunks(raw_sources)
    updates["sources"] = merged_sources

    formatted_context = ContextManager.format_context(merged_sources)
    prompt = answer_generation_prompt.format(query=query, context=formatted_context)

    llm = get_chat_model(model_name=runtime.context.llm_model, temperature=runtime.context.temperature)
    res = await llm.ainvoke(prompt)
    clean_answer = ContextManager.extract_text_content(res.content)

    return {
        **updates,
        "answer": clean_answer,
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }
