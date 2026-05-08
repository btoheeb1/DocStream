"""
app/main.py
-----------
DocStream — Main Streamlit application entry point.

Architecture note on chat input placement:
Streamlit's st.chat_input() does not anchor to the bottom correctly
when placed inside st.tabs(). The solution is to render the chat input
ONCE at the top level — outside all tab containers — and use session
state to route the input to the correct tab's processing pipeline.

Usage:
    streamlit run src/app/main.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import streamlit as st

from config import SUBJECTS_DISPLAY, OPENAI_API_KEY, UPLOADS_DIR, validate_config
from app.chatbot_utility import get_chapter_list, check_vector_db_exists
from rag.chain import build_chain
from rag.retriever import get_vector_db_path, get_image_chunks_from_docs, search_image_chunks
from rag.youtube import (
    is_educational_query,
    should_show_images,
    generate_youtube_query,
    get_youtube_recommendations,
)
from ingestion.ingest_uploaded import (
    ingest_uploaded_pdfs,
    generate_session_id,
    cleanup_session,
)

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocStream",
    page_icon="S",
    layout="centered"
)


def initialize_session_state() -> None:
    """Initialize all Streamlit session state variables with safe defaults."""
    defaults = {
        "active_tab": "course",
        "course_chat_history": [],
        "course_video_history": [],
        "course_image_history": [],
        "course_chain": None,
        "course_llm": None,
        "course_selected_chapter": None,
        "course_selected_subject": None,
        "course_vector_db_path": None,
        "upload_chat_history": [],
        "upload_video_history": [],
        "upload_image_history": [],
        "upload_chain": None,
        "upload_llm": None,
        "upload_session_id": None,
        "upload_ready": False,
        "upload_file_names": [],
        "upload_vector_db_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_course_chain(chapter: str, subject: str) -> None:
    """
    Build and cache the RAG chain for the selected course chapter.

    Only rebuilds when the chapter selection changes.

    Args:
        chapter (str): Selected chapter name or 'All Chapters'.
        subject (str): Selected subject name (e.g., 'Biology').
    """
    if st.session_state.get("course_selected_chapter") != chapter:
        vector_db_path = get_vector_db_path(chapter, subject)
        chain, llm = build_chain(vector_db_path)
        st.session_state.course_chain = chain
        st.session_state.course_llm = llm
        st.session_state.course_selected_chapter = chapter
        st.session_state.course_selected_subject = subject
        st.session_state.course_vector_db_path = vector_db_path


def display_chat_history(
    history_key: str,
    image_key: str,
    video_key: str
) -> None:
    """
    Render all previous messages, images, and videos from session state.

    Args:
        history_key (str): Session state key for chat history.
        image_key (str): Session state key for image history.
        video_key (str): Session state key for video history.
    """
    for idx, message in enumerate(st.session_state[history_key]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if idx < len(st.session_state[image_key]):
                    image_refs = st.session_state[image_key][idx]
                    if image_refs:
                        for img_path, img_caption in image_refs:
                            if os.path.exists(img_path):
                                st.image(
                                    img_path,
                                    caption=img_caption,
                                    width="stretch"
                                )

                if idx < len(st.session_state[video_key]):
                    video_refs = st.session_state[video_key][idx]
                    if video_refs:
                        st.subheader("Video References")
                        for title, link in video_refs:
                            st.info(f"**{title}**\n\nLink: {link}")


def process_response(
    user_input: str,
    chain,
    llm,
    vector_db_path: str,
    history_key: str,
    image_key: str,
    video_key: str,
) -> None:
    """
    Process a user message through the full DocStream RAG pipeline.

    Args:
        user_input (str): The user's message.
        chain: The ConversationalRetrievalChain.
        llm: The ChatGroq LLM instance.
        vector_db_path (str): Path to the active ChromaDB store.
        history_key (str): Session state key for chat history.
        image_key (str): Session state key for image history.
        video_key (str): Session state key for video history.
    """
    st.session_state[history_key].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        image_refs = []
        video_refs = []

        with st.spinner("Thinking..."):
            response = chain.invoke({"question": user_input})

        answer = response.get("answer", response.get("result", ""))
        st.markdown(answer)

        educational = is_educational_query(llm, user_input)

        if educational:
            if should_show_images(user_input):
                source_docs = response.get("source_documents", [])
                image_refs = get_image_chunks_from_docs(source_docs)

                if not image_refs:
                    image_refs = search_image_chunks(user_input, vector_db_path)

                for img_path, img_caption in image_refs:
                    st.image(
                        img_path,
                        caption=img_caption,
                        width="stretch"
                    )

            search_query = generate_youtube_query(
                llm=llm,
                chat_history=st.session_state[history_key]
            )
            if search_query:
                titles, links = get_youtube_recommendations(search_query)
                if titles:
                    st.subheader("Video References")
                    for i in range(min(3, len(titles))):
                        st.info(f"**{titles[i]}**\n\nLink: {links[i]}")
                        video_refs.append((titles[i], links[i]))

    st.session_state[history_key].append({"role": "assistant", "content": answer})
    st.session_state[video_key].append(video_refs if video_refs else None)
    st.session_state[image_key].append(image_refs if image_refs else None)


def main() -> None:
    """
    Main entry point for DocStream.

    Key architectural decision: st.chat_input() is rendered ONCE here
    at the top level — completely outside st.tabs() — so Streamlit can
    correctly anchor it to the bottom of the viewport. The active tab
    is tracked in session state to route input to the right pipeline.
    """
    initialize_session_state()

    missing = validate_config()
    if missing:
        st.error(
            f"Missing environment variables: {', '.join(missing)}\n\n"
            "Copy `.env.example` to `.env` and fill in your API keys."
        )
        st.stop()

    st.title("DocStream")
    st.caption("Your AI-powered study assistant — ask questions from any document")

    # ── SINGLE CHAT INPUT — rendered at top level, outside all tabs ───────────
    # This is the correct pattern for Streamlit chat apps with tabs.
    # The input anchors to the bottom of the page correctly only when
    # it is not nested inside a tab, column, or container widget.
    user_input = st.chat_input("Ask a question...")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_course, tab_upload = st.tabs([
        "Course Materials",
        "Upload Your Own"
    ])

    # ── COURSE MATERIALS TAB ──────────────────────────────────────────────────
    with tab_course:
        # Track when user interacts with this tab
        st.session_state.active_tab = "course"

        st.markdown("### Course Materials")
        st.caption("Select a subject and chapter from pre-loaded course materials.")

        selected_subject = st.selectbox(
            label="Select a Subject",
            options=SUBJECTS_DISPLAY,
            index=None,
            placeholder="Choose a subject...",
            key="course_subject_select"
        )

        if not selected_subject:
            st.info("Select a subject to get started.")
        else:
            chapter_list = get_chapter_list(selected_subject) + ["All Chapters"]
            selected_chapter = st.selectbox(
                label=f"Select a Chapter — {selected_subject}",
                options=chapter_list,
                index=0,
                key="course_chapter_select"
            )

            if selected_chapter:
                if not check_vector_db_exists(selected_chapter, selected_subject):
                    st.error(
                        f"No vector database found for **{selected_chapter}**.\n\n"
                        "Please run:\n```\npython src/ingestion/vectorize_script.py\n```"
                    )
                else:
                    try:
                        load_course_chain(selected_chapter, selected_subject)
                    except Exception as e:
                        st.error(f"Failed to load the RAG chain: {e}")
                    else:
                        # Display chat history for this tab
                        display_chat_history(
                            "course_chat_history",
                            "course_image_history",
                            "course_video_history"
                        )

                        # Process input if this tab is active and input was submitted
                        if (
                            user_input
                            and st.session_state.course_chain is not None
                        ):
                            process_response(
                                user_input=user_input,
                                chain=st.session_state.course_chain,
                                llm=st.session_state.course_llm,
                                vector_db_path=st.session_state.course_vector_db_path,
                                history_key="course_chat_history",
                                image_key="course_image_history",
                                video_key="course_video_history",
                            )

    # ── UPLOAD TAB ────────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("### Upload Your Own Documents")
        st.caption("Upload any PDF — notes, textbooks, papers — and ask questions.")

        uploaded_files = st.file_uploader(
            label="Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        enable_captioning = st.toggle(
            label="Enable image captioning (GPT-4o Vision)",
            value=bool(OPENAI_API_KEY),
            disabled=not bool(OPENAI_API_KEY),
            help=(
                "Diagrams in your PDFs will be captioned by GPT-4o Vision."
                if OPENAI_API_KEY
                else "Set OPENAI_API_KEY in .env to enable."
            )
        )

        if uploaded_files:
            new_file_names = sorted([f.name for f in uploaded_files])
            files_changed = new_file_names != st.session_state.upload_file_names
            button_label = (
                "Re-process Documents"
                if st.session_state.upload_ready and files_changed
                else "Process Documents"
            )

            if st.button(button_label, type="primary", use_container_width=True):
                if st.session_state.upload_session_id:
                    cleanup_session(st.session_state.upload_session_id)

                st.session_state.upload_ready = False
                st.session_state.upload_chat_history = []
                st.session_state.upload_video_history = []
                st.session_state.upload_image_history = []
                st.session_state.upload_chain = None
                st.session_state.upload_llm = None

                session_id = generate_session_id()
                st.session_state.upload_session_id = session_id
                st.session_state.upload_file_names = new_file_names

                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_messages = []

                def update_progress(message: str) -> None:
                    """Update the progress bar and status text."""
                    status_messages.append(message)
                    progress_placeholder.info(f"Processing: {message}")
                    progress = min(
                        len(status_messages) / (len(uploaded_files) * 10), 0.95
                    )
                    progress_bar.progress(progress)

                try:
                    vector_db_path = ingest_uploaded_pdfs(
                        uploaded_files=uploaded_files,
                        session_id=session_id,
                        enable_image_captioning=enable_captioning,
                        progress_callback=update_progress
                    )
                    chain, llm = build_chain(vector_db_path)
                    st.session_state.upload_chain = chain
                    st.session_state.upload_llm = llm
                    st.session_state.upload_ready = True
                    st.session_state.upload_vector_db_path = vector_db_path

                    progress_bar.progress(1.0)
                    progress_placeholder.success(
                        f"Ready. Processed {len(uploaded_files)} file(s)."
                    )

                except Exception as e:
                    progress_placeholder.error(f"Ingestion failed: {e}")
                    progress_bar.empty()

        if st.session_state.upload_ready and st.session_state.upload_chain:
            if st.session_state.upload_file_names:
                st.caption(
                    f"Active documents: {', '.join(st.session_state.upload_file_names)}"
                )
            st.divider()

            # Display chat history for upload tab
            display_chat_history(
                "upload_chat_history",
                "upload_image_history",
                "upload_video_history"
            )

            # Process input if upload chain is active and input was submitted
            if (
                user_input
                and st.session_state.upload_chain is not None
                and st.session_state.upload_vector_db_path is not None
            ):
                process_response(
                    user_input=user_input,
                    chain=st.session_state.upload_chain,
                    llm=st.session_state.upload_llm,
                    vector_db_path=st.session_state.upload_vector_db_path,
                    history_key="upload_chat_history",
                    image_key="upload_image_history",
                    video_key="upload_video_history",
                )

        elif not uploaded_files:
            st.info("Upload one or more PDF files to get started.")


if __name__ == "__main__":
    main()
