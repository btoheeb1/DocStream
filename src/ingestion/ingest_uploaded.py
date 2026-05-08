"""
ingestion/ingest_uploaded.py
-----------------------------
Handles ingestion of user-uploaded PDF files directly from the Streamlit UI.

Unlike vectorize.py which reads from fixed folder paths, this module:
- Accepts raw file bytes from st.file_uploader()
- Saves files to a session-specific uploads folder
- Runs the full ingestion pipeline (text + optional image captioning)
- Returns a ready-to-use ChromaDB vector store path

Each upload session gets a unique ChromaDB store so multiple users
do not overwrite each other's data.

Usage:
    from ingestion.ingest_uploaded import ingest_uploaded_pdfs
"""

import sys
import json
import uuid
import base64
import shutil
from pathlib import Path
from io import BytesIO
from typing import Callable, Optional

# Add src/ to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import fitz
import numpy as np
from PIL import Image
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from config import (
    DEVICE,
    OPENAI_API_KEY,
    UPLOADS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    ensure_directories,
)

# Initialize shared components
# RecursiveCharacterTextSplitter respects chunk size limits better,
# preventing the "chunk larger than specified" warnings
_embeddings = HuggingFaceEmbeddings(model_kwargs={"device": DEVICE})
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

# Non-educational phrases for filtering GPT-4o captions
NON_EDUCATIONAL_PHRASES = [
    "this is a qr code", "this is a logo", "it seems you uploaded a logo",
    "this appears to be a logo", "this image appears to be a logo",
    "this is a blank", "this image is blank", "this is a decorative",
    "i cannot identify", "no educational content", "this is not a diagram",
    "this appears to be a watermark", "this is a border",
    "this is a page number", "this is a header", "this is a footer",
]


def generate_session_id() -> str:
    """
    Generate a unique session ID for isolating user uploads.

    Returns:
        str: A short unique identifier string.
    """
    return uuid.uuid4().hex[:12]


def save_uploaded_files(uploaded_files: list, session_id: str) -> tuple:
    """
    Save Streamlit UploadedFile objects to disk in a session folder.

    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects.
        session_id (str): Unique session identifier for folder isolation.

    Returns:
        tuple: (session_dir, saved_paths) where session_dir is the Path
               to the session folder and saved_paths is a list of saved Paths.
    """
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = session_dir / uploaded_file.name
        with open(str(file_path), "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return session_dir, saved_paths


def _load_pdf_text(pdf_path: Path) -> list:
    """
    Load text from a PDF using PyMuPDF, one Document per page.

    Uses fitz directly instead of UnstructuredFileLoader to avoid
    the LangChain deprecation warning.

    Args:
        pdf_path (Path): Path to the PDF file.

    Returns:
        list: List of LangChain Document objects.
    """
    doc = fitz.open(str(pdf_path))
    documents = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_num + 1,
                    "filename": pdf_path.name
                }
            ))

    doc.close()
    return documents


def is_image_quality_sufficient(image_bytes: bytes) -> bool:
    """
    Check if an extracted image passes quality filters.

    Rejects images that are mostly black, blank, or a single flat color.

    Args:
        image_bytes (bytes): Raw image bytes.

    Returns:
        bool: True if the image passes all quality checks.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        total = img_array.size

        if np.sum(img_array < 20) / total > 0.90:
            return False
        if np.sum(img_array > 235) / total > 0.95:
            return False
        if np.std(img_array) < 8:
            return False

        return True
    except Exception:
        return False


def is_caption_educational(caption: str) -> bool:
    """
    Determine whether a GPT-4o caption describes educational content.

    Args:
        caption (str): Caption text from GPT-4o.

    Returns:
        bool: True if the caption is educational.
    """
    if caption.strip() == "NON_EDUCATIONAL":
        return False
    caption_lower = caption.lower()
    return not any(phrase in caption_lower for phrase in NON_EDUCATIONAL_PHRASES)


def extract_and_caption_images(
    pdf_path: Path,
    images_dir: Path,
    enable_captioning: bool = True,
    progress_callback: Optional[Callable] = None
) -> list:
    """
    Extract images from a PDF and optionally generate GPT-4o captions.

    Args:
        pdf_path (Path): Path to the PDF file.
        images_dir (Path): Directory to save extracted image files.
        enable_captioning (bool): If True, sends images to GPT-4o Vision.
        progress_callback (callable, optional): Called with status strings.

    Returns:
        list: List of LangChain Document objects with image captions.
              Empty if captioning is disabled or no educational images found.
    """
    if not enable_captioning or not OPENAI_API_KEY:
        return []

    client = OpenAI(api_key=OPENAI_API_KEY)
    doc = fitz.open(str(pdf_path))
    chapter_name = pdf_path.stem
    caption_docs = []
    images_dir.mkdir(parents=True, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)

            width = base_image["width"]
            height = base_image["height"]
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Size filter
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            # Quality filter
            if not is_image_quality_sufficient(image_bytes):
                continue

            # Save image to disk
            image_filename = f"{chapter_name}_p{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = images_dir / image_filename
            with open(str(image_path), "wb") as f:
                f.write(image_bytes)

            try:
                media_type = "image/jpeg" if image_ext.lower() in ["jpg", "jpeg"] else "image/png"
                base64_image = base64.b64encode(image_bytes).decode("utf-8")

                prompt = (
                    f"This image is from a PDF document, chapter: {chapter_name}. "
                    "If this is an educational diagram, figure, chart, or illustration, "
                    "describe it in detail for a student. "
                    "If it is a logo, QR code, decoration, or non-educational content, "
                    "respond with exactly: NON_EDUCATIONAL"
                )

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }],
                    max_tokens=500
                )

                caption = response.choices[0].message.content.strip()

                if not is_caption_educational(caption):
                    if image_path.exists():
                        image_path.unlink()
                    continue

                caption_docs.append(Document(
                    page_content=caption,
                    metadata={
                        "source": "image",
                        "image_filename": image_filename,
                        "image_path": str(image_path),
                        "chapter": chapter_name,
                        "page": page_num + 1
                    }
                ))

                if progress_callback:
                    progress_callback(f"Captioned image: {image_filename}")

            except Exception as e:
                print(f"[ingest_uploaded] Caption error for {image_filename}: {e}")
                continue

    doc.close()
    return caption_docs


def ingest_uploaded_pdfs(
    uploaded_files: list,
    session_id: str,
    enable_image_captioning: bool = True,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Full ingestion pipeline for user-uploaded PDF files.

    This is the main entry point called from the Streamlit UI.

    Pipeline:
    1. Save uploaded files to session folder
    2. Extract and chunk text from each PDF using PyMuPDF
    3. Optionally extract and caption images (GPT-4o Vision)
    4. Embed all documents and store in a session-specific ChromaDB
    5. Return the vector store path for use in the RAG chain

    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects.
        session_id (str): Unique session identifier.
        enable_image_captioning (bool): Whether to caption images with GPT-4o.
        progress_callback (callable, optional): Called with status strings.

    Returns:
        str: Path to the ChromaDB vector store for this session.

    Raises:
        ValueError: If no content could be extracted from the uploaded files.
    """
    ensure_directories()

    if progress_callback:
        progress_callback("Saving uploaded files...")

    # Step 1: Save files to disk
    session_dir, saved_paths = save_uploaded_files(uploaded_files, session_id)

    # Session-specific paths
    vector_db_path = UPLOADS_DIR / f"{session_id}_vectordb"
    images_dir = UPLOADS_DIR / f"{session_id}_images"

    all_documents = []

    for i, pdf_path in enumerate(saved_paths):
        file_name = pdf_path.name

        if progress_callback:
            progress_callback(f"Processing {file_name} ({i+1}/{len(saved_paths)})...")

        # Step 2: Load and chunk text
        try:
            documents = _load_pdf_text(pdf_path)
            text_chunks = _text_splitter.split_documents(documents)
            all_documents.extend(text_chunks)

            if progress_callback:
                progress_callback(
                    f"Chunked {len(text_chunks)} text segments from {file_name}"
                )

        except Exception as e:
            print(f"[ingest_uploaded] Text extraction error for {file_name}: {e}")
            continue

        # Step 3: Extract and caption images (optional)
        if enable_image_captioning:
            if progress_callback:
                progress_callback(f"Extracting images from {file_name}...")

            caption_docs = extract_and_caption_images(
                pdf_path=pdf_path,
                images_dir=images_dir,
                enable_captioning=True,
                progress_callback=progress_callback
            )
            all_documents.extend(caption_docs)

            if progress_callback and caption_docs:
                progress_callback(
                    f"Added {len(caption_docs)} image captions from {file_name}"
                )

    if not all_documents:
        raise ValueError(
            "No content could be extracted from the uploaded files. "
            "Please check that the PDFs contain readable text."
        )

    # Step 4: Embed and store in ChromaDB
    if progress_callback:
        progress_callback(
            f"Embedding {len(all_documents)} chunks into vector store..."
        )

    Chroma.from_documents(
        documents=all_documents,
        embedding=_embeddings,
        persist_directory=str(vector_db_path)
    )

    if progress_callback:
        progress_callback("Ingestion complete. Loading chat interface...")

    return str(vector_db_path)


def cleanup_session(session_id: str) -> None:
    """
    Remove all files and directories associated with a user session.

    Called when a user uploads new files to replace their previous session,
    preventing unbounded disk usage.

    Args:
        session_id (str): The session ID to clean up.
    """
    paths_to_remove = [
        UPLOADS_DIR / session_id,
        UPLOADS_DIR / f"{session_id}_vectordb",
        UPLOADS_DIR / f"{session_id}_images",
    ]

    for path in paths_to_remove:
        if path.exists():
            shutil.rmtree(str(path), ignore_errors=True)
