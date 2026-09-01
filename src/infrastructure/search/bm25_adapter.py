"""BM25 Search Adapter implementing ISearchEngineAdapter."""
import csv
import os
from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.domain.interfaces.retrievers import ISearchEngineAdapter
from src.core.config import get_settings

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


class BM25SearchAdapter(ISearchEngineAdapter):
    """BM25 In-Memory Index Adapter for keyword search."""

    def __init__(self, documents: Optional[List[Document]] = None, top_k: Optional[int] = None):
        settings = get_settings()
        self.top_k = top_k or settings.retriever_top_k
        docs = documents if documents is not None else self._load_documents_from_csv()

        if docs:
            self._retriever = BM25Retriever.from_documents(docs)
            self._retriever.k = self.top_k
        else:
            # Fallback placeholder if data is empty
            self._retriever = BM25Retriever.from_texts([""])
            self._retriever.k = self.top_k

    @property
    def retriever(self) -> BM25Retriever:
        """Access underlying BM25Retriever instance."""
        return self._retriever

    async def bm25_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform BM25 keyword search."""
        if k:
            self._retriever.k = k
        return await self._retriever.ainvoke(query)

    def _load_documents_from_csv(self) -> List[Document]:
        """Load transcript chunks from CSV files into LangChain Documents."""
        docs = []
        chunks_file = os.path.join(DATA_DIR, "chunks.csv")
        video_chunks_file = os.path.join(DATA_DIR, "video_chunks.csv")
        videos_file = os.path.join(DATA_DIR, "videos.csv")

        if not (os.path.exists(chunks_file) and os.path.exists(video_chunks_file) and os.path.exists(videos_file)):
            return docs

        video_names = {}
        with open(videos_file, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video_names[row["video_uuid"]] = row.get("video_name", "")

        chunk_video_map = {}
        with open(video_chunks_file, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                chunk_video_map[row["chunk_uuid"]] = row["video_uuid"]

        with open(chunks_file, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c_id = row["chunk_uuid"]
                v_id = chunk_video_map.get(c_id, "")
                v_name = video_names.get(v_id, "")
                metadata = {
                    "chunk_id": c_id,
                    "video_id": v_id,
                    "video_name": v_name,
                    "timestamp": float(row.get("timestamp", 0) or 0),
                    "duration": float(row.get("duration", 15) or 15),
                    "slide_number": int(row["slide_number"]) if row.get("slide_number") else None,
                }
                docs.append(Document(page_content=row.get("content", ""), metadata=metadata))

        return docs


def make_bm25_retriever(top_k: Optional[int] = None) -> BM25Retriever:
    """Factory helper to build BM25Retriever instance."""
    adapter = BM25SearchAdapter(top_k=top_k)
    return adapter.retriever
