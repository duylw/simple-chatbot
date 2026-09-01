"""ChromaDB Vector Store Adapter implementing IVectorStoreAdapter."""
from typing import List, Optional, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.core.config import get_settings
from src.domain.interfaces.retrievers import IVectorStoreAdapter


class ChromaVectorStoreAdapter(IVectorStoreAdapter):
    """ChromaDB Adapter for semantic vector search."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        collection_name: str = "langchain",
        embedding_model: Optional[Any] = None,
    ):
        settings = get_settings()
        self.host = host or settings.CHROMA_HOST
        self.port = port or settings.CHROMA_PORT
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._vector_store = Chroma(
            host=self.host,
            port=self.port,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
        )

    @property
    def vector_store(self) -> Chroma:
        """Access underlying Chroma instance."""
        return self._vector_store

    async def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform semantic similarity search on Chroma collection."""
        retriever = self._vector_store.as_retriever(search_kwargs={"k": k})
        return await retriever.ainvoke(query)

    def as_retriever(self, k: int = 10) -> VectorStoreRetriever:
        """Return a LangChain VectorStoreRetriever."""
        return self._vector_store.as_retriever(search_kwargs={"k": k})


def make_vector_db_retriever(top_k: Optional[int] = None) -> VectorStoreRetriever:
    """Factory helper to build Chroma VectorStoreRetriever."""
    settings = get_settings()
    k = top_k or settings.retriever_top_k
    adapter = ChromaVectorStoreAdapter()
    return adapter.as_retriever(k=k)
