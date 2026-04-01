from langchain_core.documents import Document
from langchain_core.tools import tool

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever


def create_retriever_tool(
    vectordb_retriever: VectorStoreRetriever | None = None,
    bm25_retriever: BM25Retriever | None = None,
    top_k: int = 3,
    use_hybrid: bool = True,
    semantic_weight: float = 1.0,  # Add parameter for semantic weight
    bm25_weight: float = 1.0,      # Add parameter for BM25 weight
):
    """Create a retriever tool that wraps Hybrid search service.

    :param vectordb_retriever: Existing vector database retriever
    :param bm25_retriever: Existing BM25 retriever
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :param semantic_weight: Weight applied to semantic search results in RRF
    :param bm25_weight: Weight applied to BM25 search results in RRF
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def hybrid_search(query:str) -> list[Document]:
        """Hybrid Search using Reciprocal Rank Fusion (RRF)."""

        semantic_res =  []
        bm25_res =  []

        if vectordb_retriever:
            semantic_res = await vectordb_retriever.ainvoke(query, k=top_k*2)
        if bm25_retriever:
            bm25_res = await bm25_retriever.ainvoke(query, k=top_k*2)

        rrf_scores = {}

        def add_results_to_rrf(results, weight: float):
            for rank, doc in enumerate(results):
                doc_id = doc.page_content

                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {"doc": doc, "score": 0.0}

                # Calculate RRF score with the specific weight
                rrf_scores[doc_id]["score"] += weight * (1.0 / (rank + 1 + 60))

        # Pass the weights to the helper function
        add_results_to_rrf(semantic_res, weight=semantic_weight)
        add_results_to_rrf(bm25_res, weight=bm25_weight)

        reranked_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        final_results = [item["doc"] for item in reranked_docs[:top_k]]

        return final_results
    
    @tool
    async def semantic_search(query: str) -> list[Document]:
        """Semantic search using vector database retriever."""
        results = []

        if vectordb_retriever:
            results = await vectordb_retriever.ainvoke(query, k=top_k)

        return results

    @tool
    async def bm25_search(query: str) -> list[Document]:
        """BM25 search using BM25 retriever."""
        results = []
        if bm25_retriever:
            results = await bm25_retriever.ainvoke(query, k=top_k)

        return results

    if use_hybrid:
        return hybrid_search

    return semantic_search, bm25_search