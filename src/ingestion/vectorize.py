"""
ingestion/vectorize.py
----------------------
Handles PDF ingestion, text chunking, embedding, and ChromaDB storage
for DocStream's RAG pipeline.

This module provides functions to:
- Vectorize entire subject PDF collections (whole-book index)
- Vectorize individual chapter PDFs (per-chapter index)
- Load image captions from image_captions.json as LangChain Documents
- Add image caption documents to existing ChromaDB stores

Usage:
    from ingestion.vectorize import (
        vectorize_book,
        vectorize_chapters,
        vectorize_image_captions_into_book_db,
        vectorize_image_captions_into_chapter_dbs,
    )
"""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from config import (
    DEVICE,
    DATA_DIR,
    VECTOR_DB_DIR,
    CHAPTERS_VECTOR_DB_DIR,
    IMAGE_CAPTIONS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Initialize shared embedding model and text splitter
# RecursiveCharacterTextSplitter respects chunk size limits better than
# CharacterTextSplitter, preventing oversized chunks
_embeddings = HuggingFaceEmbeddings(model_kwargs={"device": DEVICE})
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _load_pdf(pdf_path: str) -> list:
    """
    Load a PDF file and return a list of LangChain Document objects.

    Uses PyMuPDF (fitz) for text extraction to avoid the deprecated
    UnstructuredFileLoader warning while maintaining compatibility.

    Args:
        pdf_path (str): Absolute path to the PDF file.

    Returns:
        list: List of LangChain Document objects, one per page.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    documents = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():  # Skip blank pages
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num + 1
                }
            ))

    doc.close()
    return documents


def vectorize_book(subject_path: str, vector_db_name: str) -> None:
    """
    Vectorize all PDFs in a subject directory and store as a whole-book index.

    Loads all PDF files from the subject directory, splits them into chunks,
    embeds them, and stores them in a ChromaDB persistent directory.

    Args:
        subject_path (str): Subject folder name inside data/class_12/.
                            Example: 'biology'
        vector_db_name (str): Name for the ChromaDB directory.
                              Example: 'class_12_biology_vector_db'
    """
    book_dir = DATA_DIR / subject_path
    vector_db_path = str(VECTOR_DB_DIR / vector_db_name)

    print(f"Vectorizing book: {book_dir}")

    all_chunks = []
    for pdf_file in sorted(book_dir.glob("*.pdf")):
        documents = _load_pdf(str(pdf_file))
        chunks = _text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {pdf_file.name}")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=_embeddings,
        persist_directory=vector_db_path
    )

    print(f"Saved {len(all_chunks)} total chunks to: {vector_db_name}")


def vectorize_chapters(subject_path: str) -> None:
    """
    Vectorize each PDF chapter individually into separate ChromaDB stores.

    Each chapter gets its own ChromaDB directory named after the chapter.
    This enables chapter-specific retrieval in the Streamlit UI.

    Args:
        subject_path (str): Subject folder name inside data/class_12/.
                            Example: 'biology'
    """
    book_dir = DATA_DIR / subject_path

    for pdf_file in sorted(book_dir.glob("*.pdf")):
        chapter_name = pdf_file.stem
        chapter_db_path = str(CHAPTERS_VECTOR_DB_DIR / chapter_name)

        print(f"Vectorizing chapter: {chapter_name}")

        documents = _load_pdf(str(pdf_file))
        chunks = _text_splitter.split_documents(documents)

        Chroma.from_documents(
            documents=chunks,
            embedding=_embeddings,
            persist_directory=chapter_db_path
        )

        print(f"  Saved {len(chunks)} chunks")


def load_image_captions_as_documents(subject_filter: str = None) -> list:
    """
    Load image captions from image_captions.json as LangChain Documents.

    Each caption becomes a Document with:
    - page_content: The GPT-4o generated caption text
    - metadata: source='image', chapter, subject, page, image_path, image_filename

    Args:
        subject_filter (str, optional): Filter captions by subject name.
                                        Example: 'biology'. If None, loads all.

    Returns:
        list: List of LangChain Document objects ready for embedding.
    """
    if not IMAGE_CAPTIONS_FILE.exists():
        print(f"No captions file found at {IMAGE_CAPTIONS_FILE}.")
        print("Run ingestion/extract_images.py first to generate captions.")
        return []

    with open(IMAGE_CAPTIONS_FILE, "r") as f:
        captions = json.load(f)

    documents = []
    for entry in captions:
        if subject_filter and entry["subject"].lower() != subject_filter.lower():
            continue

        doc = Document(
            page_content=entry["caption"],
            metadata={
                "source": "image",
                "image_filename": entry["image_filename"],
                "image_path": entry["image_path"],
                "subject": entry["subject"],
                "chapter": entry["chapter"],
                "page": entry["page"]
            }
        )
        documents.append(doc)

    subject_msg = f" for subject: {subject_filter}" if subject_filter else ""
    print(f"Loaded {len(documents)} image caption documents{subject_msg}")
    return documents


def vectorize_image_captions_into_book_db(subject: str, vector_db_name: str) -> None:
    """
    Add image caption documents into an existing whole-book ChromaDB store.

    Must be run AFTER vectorize_book() so the store already exists.

    Args:
        subject (str): Subject name to filter captions. Example: 'biology'
        vector_db_name (str): Name of the target ChromaDB directory.
    """
    vector_db_path = str(VECTOR_DB_DIR / vector_db_name)

    if not Path(vector_db_path).exists():
        print(f"Vector DB not found: {vector_db_path}")
        print("Run vectorize_book() first.")
        return

    caption_docs = load_image_captions_as_documents(subject_filter=subject)

    if not caption_docs:
        print("No image captions to add.")
        return

    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=_embeddings
    )
    vectorstore.add_documents(caption_docs)
    print(f"Added {len(caption_docs)} image caption chunks to: {vector_db_name}")


def vectorize_image_captions_into_chapter_dbs(subject: str) -> None:
    """
    Add image caption documents into their corresponding chapter ChromaDB stores.

    Each caption is matched to its chapter via the 'chapter' metadata field.

    Args:
        subject (str): Subject name to filter captions. Example: 'biology'
    """
    caption_docs = load_image_captions_as_documents(subject_filter=subject)

    if not caption_docs:
        print("No image captions to add.")
        return

    # Group caption documents by chapter name
    chapters: dict = {}
    for doc in caption_docs:
        chapter = doc.metadata["chapter"]
        if chapter not in chapters:
            chapters[chapter] = []
        chapters[chapter].append(doc)

    for chapter_name, docs in chapters.items():
        chapter_db_path = str(CHAPTERS_VECTOR_DB_DIR / chapter_name)

        if not Path(chapter_db_path).exists():
            print(f"Chapter DB not found for '{chapter_name}' — skipping")
            continue

        vectorstore = Chroma(
            persist_directory=chapter_db_path,
            embedding_function=_embeddings
        )
        vectorstore.add_documents(docs)
        print(f"Added {len(docs)} image captions to chapter: {chapter_name}")
