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
    extract_sources_from_tool_messages,
)
from src.services.rag.context import Context

from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict

async def invoke_generate_answer(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, str | int]:
    """Generate answer from context"""
    updates = {}

    context = get_latest_context(state.get("messages", []))
    source = extract_sources_from_tool_messages(context)
    query = get_latest_query(state.get("messages", []))

    norm_context = format_context(source)

    prompt = answer_generation_prompt.format(query=query, context=norm_context)

    llm = ChatGoogleGenerativeAI(
        model=runtime.context.llm_model,
        temperature=runtime.context.temperature
    )
    res = await llm.ainvoke(prompt)

    updates["answer"] = res.content
    updates["n_llm_calls"] = state.get("n_llm_calls", 0) + 1

    return updates