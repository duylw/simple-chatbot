"""Retriever Interfaces (Ports) for Domain Layer."""
from abc import ABC, abstractmethod
from typing import List, Any


class IVectorStoreAdapter(ABC):
    """Port for Vector Database operations (Chroma, Qdrant, etc.)."""

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 10) -> List[Any]:
        """Perform semantic similarity search on the vector index."""
        pass


class ISearchEngineAdapter(ABC):
    """Port for Keyword/Sparse Search operations (BM25, Elastic, etc.)."""

    @abstractmethod
    async def bm25_search(self, query: str, k: int = 10) -> List[Any]:
        """Perform BM25 keyword search on the indexed documents."""
        pass
