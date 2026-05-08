"""
config.py
---------
Central configuration module for DocStream.

All environment variables and project-wide path constants are defined here.
Every other module imports from this file instead of calling load_dotenv()
individually, ensuring a single source of truth for configuration.

Usage:
    from config import GROQ_API_KEY, OPENAI_API_KEY, DEVICE, PROJECT_ROOT
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── PROJECT ROOT ──────────────────────────────────────────────────────────────
# Dynamically resolves the project root regardless of where the script is run.
# This replaces all hardcoded "/Users/bahlow/Desktop/Study_Pal/" paths.
# PROJECT_ROOT is the top-level folder containing src/, data/, vector_db/, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── LOAD ENVIRONMENT VARIABLES ────────────────────────────────────────────────
# Searches for .env file starting from PROJECT_ROOT.
# override=True ensures fresh values are always loaded even if env vars
# were previously set in the shell session.
_env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

# ── API KEYS ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── DEVICE CONFIGURATION ──────────────────────────────────────────────────────
# Set to "cuda" in .env if a GPU is available for faster embedding generation.
DEVICE: str = os.getenv("DEVICE", "cpu")

# ── SUBJECTS ──────────────────────────────────────────────────────────────────
# Dynamically discovers available subjects from the data directory.
# Any folder inside data/class_12/ is treated as a subject.
# Falls back to a default list if the data directory does not exist yet.
_data_base = PROJECT_ROOT / "data" / "class_12"

if _data_base.exists():
    SUBJECTS = sorted([
        d.name for d in _data_base.iterdir()
        if d.is_dir()
    ])
else:
    # Default subjects — used before data directory is populated
    SUBJECTS = ["biology", "physics", "chemistry"]

# Human-readable subject names for the Streamlit UI dropdown
SUBJECTS_DISPLAY = [s.capitalize() for s in SUBJECTS]

# ── PROJECT PATHS ─────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "class_12"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
CHAPTERS_VECTOR_DB_DIR = PROJECT_ROOT / "chapters_vector_db"
EXTRACTED_IMAGES_DIR = PROJECT_ROOT / "extracted_images"
EVALUATION_RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
IMAGE_CAPTIONS_FILE = PROJECT_ROOT / "image_captions.json"

# Directory for user-uploaded PDFs via the Streamlit UI
# Each upload session gets its own subfolder named by session ID
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# ── RAG CONFIGURATION ─────────────────────────────────────────────────────────
# Chunking parameters for PDF text splitting
CHUNK_SIZE: int = 2000
CHUNK_OVERLAP: int = 500

# Number of chunks to retrieve per query
RETRIEVER_K: int = 5
RETRIEVER_FETCH_K: int = 20

# ── LLM CONFIGURATION ─────────────────────────────────────────────────────────
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.0

# ── IMAGE EXTRACTION CONFIGURATION ───────────────────────────────────────────
# Minimum image dimensions to filter out icons and decorative elements
MIN_IMAGE_WIDTH: int = 150
MIN_IMAGE_HEIGHT: int = 150


# ── VALIDATION ────────────────────────────────────────────────────────────────

def validate_config() -> list:
    """
    Validate that all required environment variables are set.

    Returns:
        list: A list of missing variable names. Empty list means all good.
    """
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    return missing


def ensure_directories() -> None:
    """
    Create all required project directories if they do not exist.
    Safe to call multiple times — will not overwrite existing directories.
    """
    dirs = [
        VECTOR_DB_DIR,
        CHAPTERS_VECTOR_DB_DIR,
        EXTRACTED_IMAGES_DIR,
        EVALUATION_RESULTS_DIR,
        UPLOADS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
