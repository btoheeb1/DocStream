"""
rag/youtube.py
--------------
Handles YouTube video search and intent detection for DocStream.

This module provides:
- is_educational_query(): Two-stage intent classifier to determine
  whether a user message warrants video/image recommendations.
- should_show_images(): Keyword-based check for explicit image requests.
- generate_youtube_query(): LLM-based conversation summarizer for
  building focused YouTube search queries.
- get_youtube_recommendations(): Wrapper around youtubesearchpython
  to fetch top-k video titles and links.

Usage:
    from rag.youtube import (
        is_educational_query,
        should_show_images,
        generate_youtube_query,
        get_youtube_recommendations,
    )
"""

from youtubesearchpython import VideosSearch

# ── CONVERSATIONAL PHRASE LIST ─────────────────────────────────────────────────
# Messages matching these phrases are immediately classified as CONVERSATIONAL
# without making an LLM API call, saving latency and cost.
CONVERSATIONAL_PHRASES = {
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "thank you", "thanks", "thank u", "thx", "ty",
    "ok", "okay", "got it", "i see", "understood",
    "i have a few questions", "i have some questions", "i have a question",
    "can i ask", "may i ask", "sure", "yes", "no", "please", "great",
    "awesome", "nice", "cool", "sounds good", "alright", "bye", "goodbye",
    "see you", "that's all", "nothing else", "that's it", "nevermind",
    "never mind", "i'm good", "i am good", "no more questions"
}

# ── IMAGE KEYWORDS ─────────────────────────────────────────────────────────────
# Keywords that indicate the user explicitly wants visual content.
IMAGE_KEYWORDS = {
    "image", "images", "figure", "figures", "diagram", "diagrams",
    "picture", "pictures", "show me", "illustrate", "visual", "visually",
    "what does it look like", "draw", "chart", "graph", "photo"
}


def is_educational_query(llm, user_input: str) -> bool:
    """
    Classify a user message as educational or conversational using
    a two-stage approach.

    Stage 1: Fast phrase matching against CONVERSATIONAL_PHRASES.
             Returns False immediately if matched — no LLM call needed.
    Stage 2: LLM classification for longer or ambiguous messages.
             Uses a strict prompt that distinguishes genuine subject-matter
             questions from conversational setup phrases.

    Args:
        llm: The ChatGroq LLM instance.
        user_input (str): The user's message.

    Returns:
        bool: True if the message is an educational query, False otherwise.
    """
    # Stage 1: Quick phrase match — no LLM call needed
    cleaned = user_input.strip().lower().rstrip("?.!")
    if cleaned in CONVERSATIONAL_PHRASES:
        return False

    # Short messages with no question mark are likely conversational
    word_count = len(user_input.strip().split())
    if word_count <= 3 and "?" not in user_input:
        return False

    # Stage 2: LLM classification for ambiguous messages
    classification_prompt = (
        "You are classifying a student message in a study app.\n"
        "Classify the message as EDUCATIONAL or CONVERSATIONAL.\n\n"
        "EDUCATIONAL: A genuine subject-matter question that asks about a concept, "
        "process, definition, diagram, or topic from a textbook. "
        "Examples: 'What is photosynthesis?', 'Explain Bt cotton', "
        "'What are the steps of PCR?'\n\n"
        "CONVERSATIONAL: A greeting, thank you, acknowledgement, small talk, "
        "or setup phrase that does not ask a subject-matter question. "
        "Examples: 'Hi', 'Thank you', 'I have a few questions', 'Ok got it'\n\n"
        "Important: A message that says it HAS questions but does not ASK one "
        "is CONVERSATIONAL.\n\n"
        f"Message: \"{user_input}\"\n\n"
        "Respond with exactly one word — EDUCATIONAL or CONVERSATIONAL:"
    )

    response = llm.invoke(classification_prompt)
    result = response.content.strip().upper()
    return "EDUCATIONAL" in result


def should_show_images(user_input: str) -> bool:
    """
    Determine whether the user explicitly requested visual content.

    Checks for image-related keywords in the user's message.
    Images are only shown when explicitly requested to avoid
    cluttering responses with unrequested visuals.

    Args:
        user_input (str): The user's message.

    Returns:
        bool: True if the user asked for images or diagrams.
    """
    user_lower = user_input.lower()
    return any(keyword in user_lower for keyword in IMAGE_KEYWORDS)


def generate_youtube_query(llm, chat_history: list) -> str:
    """
    Generate a focused YouTube search query from conversation history.

    Uses the LLM to summarize the conversation into a concise search query
    rather than concatenating all user messages, which becomes noisy in
    longer sessions.

    Only considers substantive user messages (longer than 4 words) to
    avoid including greetings and acknowledgements in the search query.

    Args:
        llm: The ChatGroq LLM instance.
        chat_history (list): List of chat message dicts with 'role' and 'content'.

    Returns:
        str: A focused search query string, or empty string if no messages found.
    """
    # Filter to only substantive user messages
    user_messages = [
        item["content"]
        for item in chat_history
        if item["role"] == "user"
        and len(item["content"].strip().split()) > 4
    ]

    # Fall back to last user message if no substantive ones found
    if not user_messages:
        all_user = [i["content"] for i in chat_history if i["role"] == "user"]
        return all_user[-1] if all_user else ""

    # Single message — use directly without LLM call
    if len(user_messages) == 1:
        return user_messages[0]

    # Multiple messages — summarize into a focused query using last 5 only
    conversation_text = "\n".join([f"- {msg}" for msg in user_messages[-5:]])
    summary_prompt = (
        "Based on the following student questions from a study session, "
        "write a single short search query (5-10 words) that best captures "
        "the main topic being studied. Return only the search query, nothing else.\n\n"
        f"Student questions:\n{conversation_text}\n\n"
        "Search query:"
    )

    response = llm.invoke(summary_prompt)
    return response.content.strip()


def get_youtube_recommendations(query: str, limit: int = 3) -> tuple:
    """
    Search YouTube for videos matching the given query.

    Args:
        query (str): The search query string.
        limit (int): Number of videos to return. Defaults to 3.

    Returns:
        tuple: (video_titles, video_links) where each is a list of strings.
               Returns ([], []) if the search fails.
    """
    try:
        search = VideosSearch(query=query, limit=limit)
        result = search.result()
        titles = [video["title"] for video in result["result"]]
        links = [video["link"] for video in result["result"]]
        return titles, links
    except Exception as e:
        print(f"[youtube] Search error: {e}")
        return [], []
