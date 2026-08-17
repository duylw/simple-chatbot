from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context
from src.services.rag.prompts import (
    query_rewrite_prompt,
)
from .utils import get_latest_query, extract_text_content

from langgraph.runtime import Runtime
from src.services.rag.llm_factory import get_chat_model
from langchain.messages import HumanMessage
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

async def invoke_query_rewrite(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    logger.info("NODE: query_rewrite")
    query = state.get("original_query") or get_latest_query(state.get("messages", []))
    
    # If this is a retry iteration with feedback from grade_answer, append the suggestion
    answer_grades = state.get("answer_grade", [])
    if answer_grades and answer_grades[-1].suggestion:
        query_input = f"{query}\n(Gợi ý mở rộng kiến thức từ lần tìm trước: {answer_grades[-1].suggestion})"
        logger.info(f"Query rewrite with feedback: {answer_grades[-1].suggestion}")
    else:
        query_input = query
    
    prompt = query_rewrite_prompt.format(
        query=query_input,
    )
    
    llm = get_chat_model(model_name=runtime.context.llm_model, temperature=runtime.context.temperature)
    res = await llm.ainvoke(prompt)
    clean_query = extract_text_content(res.content)
    return {
        "messages": [HumanMessage(content=clean_query)],
        "rewritten_query": [clean_query],
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }