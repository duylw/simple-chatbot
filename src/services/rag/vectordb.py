from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from src.core.config import get_settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import logging
logger = logging.getLogger(__name__)

def make_vector_db() -> Chroma:
    """Factory function to create a Chroma vector database instance."""
    logger.info("Creating Chroma vector database instance...")
    try:
        settings = get_settings()
        logger.info(f"Embedding model for Chroma: {settings.EMBEDDING_MODEL}")
        
        embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)

        chroma_client = Chroma(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            embedding_function=embeddings
        )
        return chroma_client
    except Exception as e:
        logger.error(f"Error creating Chroma vector database: {e}")
        raise

def make_vector_db_retriever() -> VectorStoreRetriever:
    """Factory function to create a Chroma retriever instance."""
    logger.info("Creating Chroma retriever instance...")
    try:
        chroma_client = make_vector_db()
        # 'langchain' is the default collection name used by langchain_chroma
        retriever = chroma_client.as_retriever()
        return retriever
    except Exception as e:
        logger.error(f"Error creating Chroma retriever: {e}")
        raise