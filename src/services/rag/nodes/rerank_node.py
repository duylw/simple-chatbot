from src.services.rag.nodes.utils import (
    extract_sources_from_tool_messages,
    get_latest_query
) 
from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context

from langgraph.runtime import Runtime
from langchain_core.documents import Document
from typing import Dict
import logging

logger = logging.getLogger(__name__)

async def invoke_rerank(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    logger.info("NODE: rerank")

    query = state.get("original_query") or get_latest_query(state.get("messages", []))
    docs = extract_sources_from_tool_messages(state.get("messages", []))
    
    if not docs: return {"source": []}

    pairs = [[query, d.page_content] for d in docs]
    # with torch.no_grad():
    #     inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    #     scores = reranker_model(**inputs, return_dict=True).logits.view(-1,).float()
    
    # ranked = sorted(zip(scores.tolist(), docs), key=lambda x: x[0], reverse=True)
    # return {"source": [d for s, d in ranked][:runtime.context.reranker_top_k]}

    pass