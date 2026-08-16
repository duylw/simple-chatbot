import json
import logging
from typing import List, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def extract_sources_from_tool_messages(messages: List[Any]) -> List[Document]:
    """Extract sources safely from tool messages in conversation without eval().

    :param messages: List of messages from graph state
    :returns: List of Document objects
    """
    sources: List[Document] = []
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # 1. Check if tool message preserved raw Document objects in artifact
            if hasattr(msg, "artifact") and isinstance(msg.artifact, list):
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

            # 2. Check if content is already a list
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

            # 3. If content is JSON string
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

    return sources


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> List[Document]:
    """Get the latest context from tool messages.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content if hasattr(msg, "content") else []

    return []

def format_timestamp(seconds: float | int | None) -> str:
    """Format seconds into readable MM:SS or HH:MM:SS string."""
    if seconds is None or seconds < 0:
        return "00:00"
    
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def merge_temporal_chunks(documents: List[Document], max_gap_seconds: float = 15.0) -> List[Document]:
    """Group and merge adjacent or overlapping video transcript chunks by video_id/video_name.

    :param documents: Raw retrieved documents
    :param max_gap_seconds: Maximum time difference in seconds between chunks to merge
    :returns: Merged documents with continuous time boundaries and unified content
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
        # Sort chunks by start timestamp
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
                meta["time_range"] = f"{format_timestamp(start_t)} - {format_timestamp(end_t)}"
                current_merged.append(Document(page_content=content, metadata=meta))
                continue

            prev_doc = current_merged[-1]
            prev_meta = prev_doc.metadata
            prev_start = float(prev_meta.get("start_time", prev_meta.get("timestamp", 0) or 0))
            prev_end = float(prev_meta.get("end_time", prev_start + float(prev_meta.get("duration", 15) or 15)))

            # If current chunk overlaps or is within max_gap_seconds of previous chunk
            if start_t <= prev_end + max_gap_seconds:
                # Merge text without duplication
                if content and content not in prev_doc.page_content:
                    prev_doc.page_content = f"{prev_doc.page_content} {content}".strip()

                # Extend temporal boundary
                new_end = max(prev_end, end_t)
                prev_meta["end_time"] = new_end
                prev_meta["duration"] = new_end - prev_start
                prev_meta["time_range"] = f"{format_timestamp(prev_start)} - {format_timestamp(new_end)}"
            else:
                meta["start_time"] = start_t
                meta["end_time"] = end_t
                meta["time_range"] = f"{format_timestamp(start_t)} - {format_timestamp(end_t)}"
                current_merged.append(Document(page_content=content, metadata=meta))

        merged_results.extend(current_merged)

    return merged_results


def format_context(documents: List[Document]) -> str:
    """Build rich context string from documents with clear video and timestamp range metadata.

    :param documents: List of retrieved or merged documents
    :returns: Synthesized context formatted for LLM reasoning
    """
    if not documents:
        return "Không tìm thấy nội dung bài giảng liên quan."

    merged_docs = merge_temporal_chunks(documents)
    formatted_blocks: List[str] = []

    for doc in merged_docs:
        meta = doc.metadata or {}
        video_name = meta.get("video_name") or meta.get("source") or "Video Bài Giảng"
        
        # Determine time range
        time_range = meta.get("time_range")
        if not time_range:
            start_t = float(meta.get("timestamp", 0) or 0)
            dur = float(meta.get("duration", 0) or 0)
            time_range = f"{format_timestamp(start_t)} - {format_timestamp(start_t + dur)}"

        slide_info = f" | Slide: {meta.get('slide_number')}" if meta.get("slide_number") else ""

        formatted_blocks.append(
            f"[Nguồn: {video_name} | Mốc thời gian: {time_range}{slide_info}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted_blocks)

def filter_messages(messages: List, n: int = None) -> List[HumanMessage | AIMessage]:
    """Filter messages to keep only the latest n messages of Human or AI.

    :param messages: List of messages
    :param n: Number of latest messages to keep
    :returns: Filtered list of messages
    """
    if n is None:
        return [msg for msg in reversed(messages) if isinstance(msg, (HumanMessage, AIMessage))]
    
    return [msg for msg in reversed(messages) if isinstance(msg, (HumanMessage, AIMessage))][:n]

def trim_messages(messages: List, n: int = 10) -> List[AnyMessage]:
    """Trim messages to keep only the latest n messages.

    :param messages: List of messages
    :param n: Number of latest messages to keep
    :returns: Trimmed list of messages
    """
    return messages[-n:]