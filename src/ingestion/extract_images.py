"""
ingestion/extract_images.py
----------------------------
Extracts images from course PDFs and generates educational captions
using GPT-4o Vision for use in DocStream's multimodal RAG pipeline.

Pipeline:
1. Opens each PDF and extracts embedded images page by page
2. Filters out low-quality images (too small, blank, black, flat color)
3. Sends each qualifying image to GPT-4o Vision for captioning
4. Filters out non-educational captions (logos, QR codes, decorations)
5. Saves valid captions to image_captions.json

The script supports resuming — if interrupted, it skips already-processed
images and continues from where it left off.

Usage:
    python ingestion/extract_images.py

Prerequisites:
    - PDFs must be in data/class_12/<subject>/ folders
    - OPENAI_API_KEY must be set in .env
"""

import sys
import json
import base64
import os
from pathlib import Path
from io import BytesIO

# Add src/ to path so config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from openai import OpenAI

from config import (
    SUBJECTS,
    DATA_DIR,
    EXTRACTED_IMAGES_DIR,
    IMAGE_CAPTIONS_FILE,
    OPENAI_API_KEY,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    ensure_directories,
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ── NON-EDUCATIONAL PHRASE FILTER ─────────────────────────────────────────────
# If GPT-4o returns a caption containing any of these phrases,
# the image is considered non-educational and skipped.
NON_EDUCATIONAL_PHRASES = [
    "this is a qr code",
    "this is a logo",
    "it seems you uploaded a logo",
    "this appears to be a logo",
    "this image appears to be a logo",
    "this is a blank",
    "this image is blank",
    "this is a decorative",
    "i cannot identify",
    "no educational content",
    "this is not a diagram",
    "this appears to be a watermark",
    "this is a border",
    "this is a page number",
    "this is a header",
    "this is a footer",
]


def is_image_quality_sufficient(image_bytes: bytes) -> bool:
    """
    Check if an image passes quality filters before sending to GPT-4o.

    Rejects images that are:
    - Mostly black (>90% dark pixels) — blank or corrupted
    - Mostly white (>95% bright pixels) — blank pages or whitespace
    - Single flat color (std dev < 8) — solid fills, borders

    Args:
        image_bytes (bytes): Raw image bytes.

    Returns:
        bool: True if the image passes all quality checks.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        total_pixels = img_array.size

        # Reject mostly black images
        dark_pixels = np.sum(img_array < 20)
        if dark_pixels / total_pixels > 0.90:
            return False

        # Reject mostly white/blank images
        bright_pixels = np.sum(img_array > 235)
        if bright_pixels / total_pixels > 0.95:
            return False

        # Reject single flat color images (very low variance)
        if np.std(img_array) < 8:
            return False

        return True

    except Exception:
        # If image cannot be analyzed, skip it to be safe
        return False


def is_caption_educational(caption: str) -> bool:
    """
    Check if a GPT-4o caption describes educational content.

    Args:
        caption (str): The caption text returned by GPT-4o.

    Returns:
        bool: True if the caption describes educational content.
    """
    if caption.strip() == "NON_EDUCATIONAL":
        return False

    caption_lower = caption.lower()
    return not any(phrase in caption_lower for phrase in NON_EDUCATIONAL_PHRASES)


def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to a base64 string for the GPT-4o Vision API.

    Args:
        image_bytes (bytes): Raw image bytes.

    Returns:
        str: Base64-encoded image string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def generate_caption(
    image_bytes: bytes,
    image_ext: str,
    subject: str,
    chapter_name: str
) -> str:
    """
    Send an image to GPT-4o Vision and retrieve an educational caption.

    The prompt instructs GPT-4o to return 'NON_EDUCATIONAL' for logos,
    QR codes, decorations, and other non-educational content, making
    filtering straightforward.

    Args:
        image_bytes (bytes): Raw image bytes.
        image_ext (str): Image file extension (e.g., 'png', 'jpeg').
        subject (str): Subject name for context in the prompt.
        chapter_name (str): Chapter name for context in the prompt.

    Returns:
        str: Educational caption text, or 'NON_EDUCATIONAL' if not applicable.
    """
    base64_image = encode_image_to_base64(image_bytes)
    media_type = "image/jpeg" if image_ext.lower() in ["jpg", "jpeg"] else "image/png"

    prompt = (
        f"This image is extracted from a Class 12 {subject} textbook, "
        f"chapter: {chapter_name}. "
        "If this is a meaningful educational diagram, figure, chart, or illustration, "
        "describe it in detail — explain what it shows, what the labels mean, "
        "and how it relates to the topic, as if explaining to a student. "
        "If this is a logo, QR code, decorative border, blank image, watermark, "
        "or any non-educational content, respond with exactly: NON_EDUCATIONAL"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
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
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


def extract_images_from_pdf(
    pdf_path: Path,
    subject: str,
    chapter_name: str,
    output_dir: Path
) -> list:
    """
    Extract all quality images from a PDF file.

    Iterates through each page, extracts embedded images, applies
    size and quality filters, and saves passing images to disk.

    Args:
        pdf_path (Path): Path to the PDF file.
        subject (str): Subject name for metadata.
        chapter_name (str): Chapter name for metadata and filenames.
        output_dir (Path): Directory to save extracted image files.

    Returns:
        list: List of dicts containing image metadata and bytes.
    """
    doc = fitz.open(str(pdf_path))
    extracted = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            width = base_image["width"]
            height = base_image["height"]
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Filter 1: Minimum size check
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                print(f"  Skipping small image ({width}x{height}px)")
                continue

            # Filter 2: Quality check (black, blank, flat color)
            if not is_image_quality_sufficient(image_bytes):
                print(f"  Skipping low quality image")
                continue

            # Save the image to disk
            image_filename = (
                f"{chapter_name}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
            )
            image_path = output_dir / image_filename

            with open(str(image_path), "wb") as f:
                f.write(image_bytes)

            extracted.append({
                "image_path": str(image_path),
                "image_filename": image_filename,
                "image_bytes": image_bytes,
                "image_ext": image_ext,
                "subject": subject,
                "chapter": chapter_name,
                "page": page_num + 1
            })

    doc.close()
    return extracted


def process_all_subjects() -> None:
    """
    Run the full image extraction and captioning pipeline for all subjects.

    For each subject and chapter:
    1. Extracts quality images from PDFs
    2. Generates educational captions via GPT-4o Vision
    3. Saves valid captions to image_captions.json

    Supports resuming — already-processed images are skipped automatically.
    """
    ensure_directories()

    # Load existing captions to support resuming if interrupted
    if IMAGE_CAPTIONS_FILE.exists():
        with open(str(IMAGE_CAPTIONS_FILE), "r") as f:
            all_captions = json.load(f)
    else:
        all_captions = []

    already_processed = {entry["image_filename"] for entry in all_captions}

    for subject in SUBJECTS:
        subject_dir = DATA_DIR / subject

        if not subject_dir.exists():
            print(f"\nSkipping {subject} — directory not found: {subject_dir}")
            continue

        # Create subject-specific image output folder
        subject_images_dir = EXTRACTED_IMAGES_DIR / subject
        subject_images_dir.mkdir(parents=True, exist_ok=True)

        for pdf_file in sorted(subject_dir.glob("*.pdf")):
            chapter_name = pdf_file.stem
            print(f"\nProcessing: {subject} / {chapter_name}")

            # Step 1: Extract quality images
            extracted_images = extract_images_from_pdf(
                pdf_path=pdf_file,
                subject=subject,
                chapter_name=chapter_name,
                output_dir=subject_images_dir
            )
            print(f"  Extracted {len(extracted_images)} quality images")

            # Step 2: Generate captions
            for img_info in extracted_images:
                if img_info["image_filename"] in already_processed:
                    print(f"  Skipping (already processed): {img_info['image_filename']}")
                    continue

                try:
                    print(f"  Captioning: {img_info['image_filename']}")
                    caption = generate_caption(
                        image_bytes=img_info["image_bytes"],
                        image_ext=img_info["image_ext"],
                        subject=subject,
                        chapter_name=chapter_name
                    )

                    # Filter 3: Skip non-educational captions
                    if not is_caption_educational(caption):
                        print(f"  Skipping non-educational image")
                        # Remove saved image file since it won't be used
                        img_path = Path(img_info["image_path"])
                        if img_path.exists():
                            img_path.unlink()
                        already_processed.add(img_info["image_filename"])
                        continue

                    all_captions.append({
                        "image_filename": img_info["image_filename"],
                        "image_path": img_info["image_path"],
                        "subject": img_info["subject"],
                        "chapter": img_info["chapter"],
                        "page": img_info["page"],
                        "caption": caption,
                        "source": "image"
                    })

                    already_processed.add(img_info["image_filename"])

                    # Save after every caption to support resuming
                    with open(str(IMAGE_CAPTIONS_FILE), "w") as f:
                        json.dump(all_captions, f, indent=2)

                    print(f"  Caption saved successfully")

                except Exception as e:
                    print(f"  Error captioning {img_info['image_filename']}: {e}")
                    continue

    print(f"\n{'='*60}")
    print(f"Done! Total educational captions saved: {len(all_captions)}")
    print(f"Captions file: {IMAGE_CAPTIONS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all_subjects()
