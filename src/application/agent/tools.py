"""Hybrid Search Retriever Tools (RRF Algorithm)."""
from typing import Optional, List, Tuple
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever


def create_retriever_tool(
    vectordb_retriever: Optional[VectorStoreRetriever] = None,
    bm25_retriever: Optional[BM25Retriever] = None,
    top_k: int = 3,
    use_hybrid: bool = True,
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
):
    """Create a LangChain retriever tool wrapping Hybrid search service with RRF algorithm."""

    @tool(response_format="content_and_artifact")
    async def hybrid_search(query: str) -> Tuple[str, List[Document]]:
        """Hybrid Search using Reciprocal Rank Fusion (RRF) between Dense and Sparse indices."""
        semantic_res = []
        bm25_res = []

        if vectordb_retriever:
            semantic_res = await vectordb_retriever.ainvoke(query, k=top_k * 2)
        if bm25_retriever:
            bm25_res = await bm25_retriever.ainvoke(query, k=top_k * 2)

        rrf_scores = {}

        def add_results_to_rrf(results, weight: float):
            for rank, doc in enumerate(results):
                meta = doc.metadata or {}
                doc_id = f"{meta.get('video_name') or meta.get('video_id')}_{meta.get('timestamp')}_{doc.page_content[:60]}"

                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {"doc": doc, "score": 0.0}

                rrf_scores[doc_id]["score"] += weight * (1.0 / (rank + 1 + 60))

        add_results_to_rrf(semantic_res, weight=semantic_weight)
        add_results_to_rrf(bm25_res, weight=bm25_weight)

        reranked_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        final_results = [item["doc"] for item in reranked_docs[:top_k]]
        content_text = "\n\n".join([doc.page_content for doc in final_results])

        return content_text, final_results

    @tool(response_format="content_and_artifact")
    async def semantic_search(query: str) -> Tuple[str, List[Document]]:
        """Semantic search using vector database retriever."""
        results = []
        if vectordb_retriever:
            results = await vectordb_retriever.ainvoke(query, k=top_k)

        content_text = "\n\n".join([doc.page_content for doc in results])
        return content_text, results

    @tool(response_format="content_and_artifact")
    async def bm25_search(query: str) -> Tuple[str, List[Document]]:
        """BM25 keyword search using BM25 retriever."""
        results = []
        if bm25_retriever:
            results = await bm25_retriever.ainvoke(query, k=top_k)

        content_text = "\n\n".join([doc.page_content for doc in results])
        return content_text, results

    if use_hybrid:
        return [hybrid_search]

    return [semantic_search, bm25_search]
