from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context

from langgraph.runtime import Runtime
from langchain_core.documents import Document
from typing import Dict
import logging

logger = logging.getLogger(__name__)
async def invoke_rerank(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, Document]:
    """Re-ranking retrieved documents for better precision"""
    logger.info("NODE: invoke_rerank")
    pass