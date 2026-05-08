"""
rag/chain.py
------------
Builds and returns the LangChain ConversationalRetrievalChain used by DocStream.

The chain combines:
- A ChromaDB vectorstore retriever (MMR search)
- Llama 3.3 70B via Groq API
- ConversationBufferMemory for multi-turn context
- A custom prompt that grounds answers in retrieved textbook content

Usage:
    from rag.chain import build_chain
    chain, llm = build_chain(vector_db_path)
"""

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from config import (
    DEVICE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
)

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
# Instructs the LLM to stay grounded in retrieved textbook context.
# Explicitly prevents the LLM from claiming it has no access to the textbook.
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are DocStream, a helpful and friendly AI study assistant.
You have access to the student's textbook content provided below as context.
Always answer based on the context provided from the textbook.
The context may include both text explanations and image descriptions from the textbook.
When the context contains image descriptions, use them to enrich your answer.
Only mention figures are displayed below if the student explicitly asked for images or diagrams.
Never say you don't have access to the textbook or images — you do, through the context provided.
For non-educational inputs like greetings or thank yous, respond briefly and friendly.

Chat History:
{chat_history}

Textbook Context:
{context}

Student Question: {question}

Answer:"""
)


def build_chain(vector_db_path: str) -> tuple:
    """
    Build and return a ConversationalRetrievalChain for the given vectorstore.

    Args:
        vector_db_path (str): Path to the ChromaDB persistent directory.

    Returns:
        tuple: (chain, llm) where chain is the ConversationalRetrievalChain
               and llm is the ChatGroq instance (used for intent detection
               and YouTube query generation).
    """
    # Load embeddings — same model used during ingestion for consistency
    embeddings = HuggingFaceEmbeddings(
        model_kwargs={"device": DEVICE}
    )

    # Load the persistent ChromaDB vectorstore
    vectorstore = Chroma(
        persist_directory=str(vector_db_path),
        embedding_function=embeddings
    )

    # Initialize the LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    # Initialize conversation memory
    # output_key='answer' tells memory which part of the chain output to store
    memory = ConversationBufferMemory(
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    # Build the retrieval chain with MMR search
    # fetch_k > k ensures MMR has enough candidates to select diverse chunks
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K
            }
        ),
        return_source_documents=True,
        get_chat_history=lambda h: h,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=False
    )

    return chain, llm
