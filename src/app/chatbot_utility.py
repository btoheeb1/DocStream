"""
app/chatbot_utility.py
-----------------------
Helper utilities for the DocStream Streamlit application.

Provides functions for loading chapter lists and validating
the application configuration before startup.

Usage:
    from app.chatbot_utility import get_chapter_list, check_vector_db_exists
"""

import sys
from pathlib import Path

# Add src/ to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from config import CHAPTERS_VECTOR_DB_DIR, VECTOR_DB_DIR, DATA_DIR


def get_chapter_list(selected_subject: str) -> list:
    """
    Get a sorted list of available chapters for a given subject.

    Reads chapter names from the data directory for the subject
    and returns them sorted by chapter number.

    Args:
        selected_subject (str): Subject name as displayed in the UI
                                (e.g., 'Biology', 'Physics').

    Returns:
        list: Sorted list of chapter names (without .pdf extension).
              Returns empty list if no chapters are found.
    """
    subject_dir = DATA_DIR / selected_subject.lower()

    if not subject_dir.exists():
        return []

    # Get all PDF filenames, strip extension
    chapters = [f.stem for f in subject_dir.glob("*.pdf")]

    # Sort by chapter number (numeric prefix before the first dot)
    try:
        chapters.sort(key=lambda x: int(x.split(".")[0]))
    except (ValueError, IndexError):
        # Fall back to alphabetical sort if naming convention differs
        chapters.sort()

    return chapters


def check_vector_db_exists(chapter: str, subject: str) -> bool:
    """
    Check whether the ChromaDB vector store exists for a given chapter.

    Used in the Streamlit UI to show a friendly error message instead
    of crashing when the user selects a chapter that has not been
    vectorized yet.

    Args:
        chapter (str): Chapter name or 'All Chapters'.
        subject (str): Subject name (e.g., 'Biology').

    Returns:
        bool: True if the vector DB exists, False otherwise.
    """
    if chapter == "All Chapters":
        db_name = f"class_12_{subject.lower()}_vector_db"
        db_path = VECTOR_DB_DIR / db_name
    else:
        db_path = CHAPTERS_VECTOR_DB_DIR / chapter

    return db_path.exists()
