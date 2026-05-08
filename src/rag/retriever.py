"""
rag/retriever.py
----------------
Handles vectorstore path resolution and direct similarity search
for image chunk retrieval in DocStream.

This module is responsible for:
- Resolving the correct ChromaDB path based on subject/chapter selection
- Performing fallback image-specific similarity search when the main
  RAG chain does not return image chunks in its source documents

Usage:
    from rag.retriever import get_vector_db_path, search_image_chunks
"""

import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DEVICE,
    VECTOR_DB_DIR,
    CHAPTERS_VECTOR_DB_DIR,
)


def get_vector_db_path(chapter: str, subject: str) -> str:
    """
    Resolve the ChromaDB path for a given chapter and subject.

    For 'All Chapters', returns the whole-book vector DB path.
    For a specific chapter, returns the per-chapter vector DB path.

    Args:
        chapter (str): Selected chapter name or 'All Chapters'.
        subject (str): Subject name (e.g., 'Biology').

    Returns:
        str: Absolute path to the ChromaDB directory.
    """
    if chapter == "All Chapters":
        # Whole-book vector DB — named by subject
        db_name = f"class_12_{subject.lower()}_vector_db"
        return str(VECTOR_DB_DIR / db_name)

    # Per-chapter vector DB — named by chapter
    return str(CHAPTERS_VECTOR_DB_DIR / chapter)


def get_image_chunks_from_docs(source_docs: list) -> list:
    """
    Extract image chunks from a list of retrieved source documents.

    Filters documents where metadata source == 'image' and the
    image file exists on disk. Deduplicates by image path.

    Args:
        source_docs (list): List of LangChain Document objects from
                            the RAG chain's source_documents.

    Returns:
        list: List of (image_path, caption_label) tuples for valid images.
    """
    image_refs = []
    seen_paths = set()  # Track seen paths to avoid duplicate images

    for doc in source_docs:
        metadata = doc.metadata

        # Only process chunks tagged as image source
        if metadata.get("source") != "image":
            continue

        image_path = metadata.get("image_path", "")
        chapter = metadata.get("chapter", "Unknown chapter")
        page = metadata.get("page", "?")
        caption_label = f"Figure from {chapter}, page {page}"

        # Validate path exists and has not been shown already
        if image_path and os.path.exists(image_path) and image_path not in seen_paths:
            image_refs.append((image_path, caption_label))
            seen_paths.add(image_path)

    return image_refs


def search_image_chunks(query: str, vector_db_path: str, k: int = 5) -> list:
    """
    Perform a direct similarity search filtered to image chunks only.

    Used as a fallback when the main RAG chain does not return image
    chunks in its top-k results. Searches ChromaDB with a metadata
    filter for source == 'image'.

    Args:
        query (str): The user's question to search against.
        vector_db_path (str): Path to the ChromaDB directory.
        k (int): Number of image chunks to retrieve. Defaults to 5.

    Returns:
        list: List of (image_path, caption_label) tuples for valid images.
              Returns empty list if search fails or no images are found.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={"device": DEVICE}
        )
        vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings
        )

        # Filter search to image chunks only
        image_results = vectorstore.similarity_search(
            query,
            k=k,
            filter={"source": "image"}
        )

        return get_image_chunks_from_docs(image_results)

    except Exception as e:
        print(f"[retriever] Image fallback search error: {e}")
        return []
