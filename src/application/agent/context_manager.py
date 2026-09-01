"""ContextManager: Temporal Stitching, Deduplication, Formatting & Token Budget."""
import json
import ast
import logging
from typing import List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContextManager:
    """Centralized Context & Memory Manager for RAG and Video Transcript processing."""

    @staticmethod
    def format_timestamp(seconds: float | int | None) -> str:
        """Format seconds into human-readable MM:SS or HH:MM:SS string."""
        if seconds is None or seconds < 0:
            return "00:00"

        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @classmethod
    def merge_temporal_chunks(
        cls,
        documents: List[Document],
        max_gap_seconds: float = 15.0
    ) -> List[Document]:
        """Group and merge contiguous or overlapping video transcript chunks by video identifier.

        :param documents: Raw retrieved documents
        :param max_gap_seconds: Maximum time difference in seconds between adjacent chunks to stitch
        :returns: Merged documents with unified temporal boundaries and concatenated content
        """
        if not documents:
            return []

        # 1. Group documents by video identifier
        video_groups: dict[str, List[Document]] = {}
        for doc in documents:
            meta = doc.metadata or {}
            v_key = str(meta.get("video_id") or meta.get("video_name") or "default_video")
            video_groups.setdefault(v_key, []).append(doc)

        merged_results: List[Document] = []

        # 2. Process each video group in chronological order
        for v_key, docs in video_groups.items():
            sorted_docs = sorted(
                docs,
                key=lambda d: float((d.metadata or {}).get("timestamp", 0) or 0)
            )

            current_merged: List[Document] = []

            for doc in sorted_docs:
                meta = dict(doc.metadata or {})
                start_t = float(meta.get("timestamp", 0) or 0)
                dur = float(meta.get("duration", 15) or 15)
                end_t = start_t + dur
                content = doc.page_content.strip()

                if not current_merged:
                    meta["start_time"] = start_t
                    meta["end_time"] = end_t
                    meta["time_range"] = f"{cls.format_timestamp(start_t)} - {cls.format_timestamp(end_t)}"
                    current_merged.append(Document(page_content=content, metadata=meta))
                    continue

                prev_doc = current_merged[-1]
                prev_meta = prev_doc.metadata
                prev_start = float(prev_meta.get("start_time", prev_meta.get("timestamp", 0) or 0))
                prev_end = float(prev_meta.get("end_time", prev_start + float(prev_meta.get("duration", 15) or 15)))

                # If current chunk overlaps or is within max_gap_seconds of previous chunk
                if start_t <= prev_end + max_gap_seconds:
                    if content and content not in prev_doc.page_content:
                        prev_doc.page_content = f"{prev_doc.page_content} {content}".strip()

                    new_end = max(prev_end, end_t)
                    prev_meta["end_time"] = new_end
                    prev_meta["duration"] = new_end - prev_start
                    prev_meta["time_range"] = f"{cls.format_timestamp(prev_start)} - {cls.format_timestamp(new_end)}"
                else:
                    meta["start_time"] = start_t
                    meta["end_time"] = end_t
                    meta["time_range"] = f"{cls.format_timestamp(start_t)} - {cls.format_timestamp(end_t)}"
                    current_merged.append(Document(page_content=content, metadata=meta))

            merged_results.extend(current_merged)

        return merged_results

    @classmethod
    def format_context(cls, documents: List[Document]) -> str:
        """Build structured context string from documents with clear video and timestamp range metadata."""
        if not documents:
            return "Không tìm thấy nội dung bài giảng liên quan."

        merged_docs = cls.merge_temporal_chunks(documents)
        formatted_blocks: List[str] = []

        for doc in merged_docs:
            meta = doc.metadata or {}
            video_name = meta.get("video_name") or meta.get("source") or "Video Bài Giảng"

            time_range = meta.get("time_range")
            if not time_range:
                start_t = float(meta.get("timestamp", 0) or 0)
                dur = float(meta.get("duration", 0) or 0)
                time_range = f"{cls.format_timestamp(start_t)} - {cls.format_timestamp(start_t + dur)}"

            slide_info = f" | Slide: {meta.get('slide_number')}" if meta.get("slide_number") else ""

            formatted_blocks.append(
                f"[Nguồn: {video_name} | Mốc thời gian: {time_range}{slide_info}]\n"
                f"{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(formatted_blocks)

    @staticmethod
    def trim_history(messages: List[AnyMessage], n: int = 10) -> List[AnyMessage]:
        """Keep only the latest n messages to manage conversation token budget."""
        return messages[-n:]

    @staticmethod
    def filter_messages(messages: List[AnyMessage], n: Optional[int] = None) -> List[HumanMessage | AIMessage]:
        """Filter messages to retain only HumanMessage and AIMessage instances."""
        filtered = [msg for msg in reversed(messages) if isinstance(msg, (HumanMessage, AIMessage))]
        if n is not None:
            return filtered[:n]
        return filtered

    @staticmethod
    def extract_sources_from_tool_messages(messages: List[Any]) -> List[Document]:
        """Extract sources safely from tool messages without eval()."""
        sources: List[Document] = []

        for msg in messages:
            if isinstance(msg, ToolMessage):
                # 1. Direct Document artifacts
                if hasattr(msg, "artifact") and msg.artifact:
                    if isinstance(msg.artifact, list):
                        docs = []
                        for item in msg.artifact:
                            if isinstance(item, Document):
                                docs.append(item)
                            elif isinstance(item, dict):
                                docs.append(Document(
                                    page_content=item.get("page_content", ""),
                                    metadata=item.get("metadata", {})
                                ))
                        if docs:
                            sources = docs
                            continue

                # 2. Content is already list of Documents or dicts
                if isinstance(msg.content, list):
                    docs = []
                    for item in msg.content:
                        if isinstance(item, Document):
                            docs.append(item)
                        elif isinstance(item, dict):
                            docs.append(Document(
                                page_content=item.get("page_content", ""),
                                metadata=item.get("metadata", {})
                            ))
                    if docs:
                        sources = docs
                        continue

                # 3. Content is JSON string
                if isinstance(msg.content, str) and msg.content.strip().startswith("["):
                    try:
                        parsed = json.loads(msg.content)
                        if isinstance(parsed, list):
                            docs = []
                            for item in parsed:
                                if isinstance(item, dict):
                                    docs.append(Document(
                                        page_content=item.get("page_content", ""),
                                        metadata=item.get("metadata", {})
                                    ))
                            if docs:
                                sources = docs
                                continue
                    except Exception:
                        pass

                # 4. Safe AST parsing for string representations like [Document(...)]
                if isinstance(msg.content, str) and "Document(" in msg.content:
                    try:
                        tree = ast.parse(msg.content.strip(), mode="eval")
                        if isinstance(tree.body, ast.List):
                            docs = []
                            for elt in tree.body.elts:
                                if isinstance(elt, ast.Call) and getattr(elt.func, "id", "") == "Document":
                                    page_content = ""
                                    metadata = {}
                                    for kw in elt.keywords:
                                        if kw.arg == "page_content":
                                            page_content = ast.literal_eval(kw.value)
                                        elif kw.arg == "metadata":
                                            metadata = ast.literal_eval(kw.value)
                                    docs.append(Document(page_content=page_content, metadata=metadata))
                            if docs:
                                sources = docs
                                continue
                    except Exception:
                        pass

        return sources

    @staticmethod
    def extract_text_content(content: Any) -> str:
        """Safely extract plain text from LLM response content across types and thought signatures."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        texts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    texts.append(str(getattr(item, "text")))
                else:
                    texts.append(str(item))
            return " ".join(texts).strip()
        return str(content).strip()

    @staticmethod
    def get_latest_query(messages: List[Any]) -> str:
        """Get latest user query string from messages."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        raise ValueError("No user query found in messages")

    @staticmethod
    def get_latest_context(messages: List[Any]) -> List[Document]:
        """Get latest context from tool messages."""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                return msg.content if hasattr(msg, "content") else []
        return []
