"""Observability & Langfuse Tracing Adapter."""
import uuid
import logging
from contextlib import contextmanager
from typing import Optional, Generator
from langfuse.langchain import CallbackHandler
from langfuse import get_client, propagate_attributes

logger = logging.getLogger(__name__)


def get_tracer_callback() -> CallbackHandler:
    """Initialize Langfuse callback handler."""
    return CallbackHandler()


@contextmanager
def trace_agent_execution(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_name: str = "rag-trace"
) -> Generator[None, None, None]:
    """Context manager wrapping Langfuse observation and trace session."""
    active_user_id = user_id or "anonymous"
    active_session_id = session_id or str(uuid.uuid4())

    try:
        langfuse = get_client()
        with langfuse.start_as_current_observation(as_type="span", name="langchain-call"):
            with propagate_attributes(
                session_id=active_session_id,
                user_id=active_user_id,
                trace_name=trace_name
            ):
                yield
    except Exception as exc:
        logger.debug(f"Tracing bypassed or disabled: {exc}")
        yield
