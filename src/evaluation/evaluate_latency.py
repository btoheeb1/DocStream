"""
evaluation/evaluate_latency.py
-------------------------------
Measures end-to-end response latency for DocStream across different
query types and retrieval configurations.

Latency is measured from query submission to full response receipt,
covering embedding, retrieval, and LLM generation time.

Target: <= 10 seconds average on CPU

Usage:
    python evaluation/evaluate_latency.py

Output:
    - Per-query latency printed to terminal
    - Summary statistics (mean, min, max, std dev) by query type
    - Full results saved to evaluation_results/latency_results.json
"""

import sys
import json
import time
import statistics
from pathlib import Path

# Add src/ to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from config import (
    DEVICE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    EVALUATION_RESULTS_DIR,
    ensure_directories,
)
from rag.retriever import get_vector_db_path

# ── LATENCY TEST QUERIES ──────────────────────────────────────────────────────
# Mix of short, medium, and long queries plus All Chapters retrieval
# to benchmark performance across real usage patterns.
LATENCY_QUERIES = [
    # Short factual — single concept, minimal generation
    {
        "question": "What is Biotechnology?",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "short_factual"
    },
    {
        "question": "What is Bt cotton?",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "short_factual"
    },
    {
        "question": "What is gene therapy?",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "short_factual"
    },
    {
        "question": "What is ELISA?",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "short_factual"
    },
    {
        "question": "What are transgenic animals?",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "short_factual"
    },
    # Medium explanation — multi-sentence answers
    {
        "question": "Explain how RNA interference works to protect plants from pests.",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "medium_explanation"
    },
    {
        "question": "How is Agrobacterium tumefaciens used in genetic engineering of plants?",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "medium_explanation"
    },
    {
        "question": "Describe the applications of biotechnology in medicine.",
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "medium_explanation"
    },
    # Long multi-part — complex reasoning, longer generation
    {
        "question": (
            "Explain in detail the ethical issues surrounding genetically modified "
            "organisms and what steps have been taken to address them."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "long_multipart"
    },
    {
        "question": (
            "Compare and contrast the use of Bt toxin and RNA interference as methods "
            "for creating pest resistant plants, explaining the mechanism of each."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology",
        "query_type": "long_multipart"
    },
    # All Chapters — whole-book retrieval (larger index, slower search)
    {
        "question": "What is Biotechnology?",
        "chapter": "All Chapters",
        "subject": "biology",
        "query_type": "all_chapters"
    },
    {
        "question": "Explain the applications of biotechnology in agriculture.",
        "chapter": "All Chapters",
        "subject": "biology",
        "query_type": "all_chapters"
    },
]

# ── EVALUATION PROMPT ─────────────────────────────────────────────────────────
EVAL_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are DocStream, a helpful AI study assistant.
Answer the question based on the textbook context provided below.

Chat History:
{chat_history}

Textbook Context:
{context}

Question: {question}

Answer:"""
)


def build_eval_chain(chapter: str, subject: str):
    """
    Build a fresh RAG chain for latency measurement.

    A fresh chain is built for each query to simulate the real app
    behavior when a user first selects a chapter.

    Args:
        chapter (str): Chapter name or 'All Chapters'.
        subject (str): Subject name.

    Returns:
        ConversationalRetrievalChain: Configured evaluation chain.
    """
    vector_db_path = get_vector_db_path(chapter, subject)
    embeddings = HuggingFaceEmbeddings(model_kwargs={"device": DEVICE})
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K}
        ),
        return_source_documents=True,
        get_chat_history=lambda h: h,
        combine_docs_chain_kwargs={"prompt": EVAL_PROMPT},
        verbose=False
    )
    return chain


def measure_latency(chain, question: str) -> tuple:
    """
    Measure end-to-end latency for a single query.

    Args:
        chain: The ConversationalRetrievalChain to query.
        question (str): The test question.

    Returns:
        tuple: (answer, latency_seconds, num_contexts)
    """
    start_time = time.time()
    response = chain({"question": question})
    latency = time.time() - start_time

    return (
        response["answer"],
        latency,
        len(response.get("source_documents", []))
    )


def run_latency_evaluation() -> dict:
    """
    Run latency evaluation across all test queries.

    Measures per-query latency, groups results by query type,
    and computes summary statistics.

    Returns:
        dict: Evaluation results including summary statistics and per-query data.
    """
    ensure_directories()

    print("=" * 60)
    print("DocStream — Response Latency Evaluation")
    print("=" * 60)
    print(f"Total queries: {len(LATENCY_QUERIES)}")
    print(f"Device: {DEVICE}\n")

    all_latencies = []
    results_by_type: dict = {}
    per_query_results = []

    for i, item in enumerate(LATENCY_QUERIES):
        question = item["question"]
        chapter = item["chapter"]
        subject = item["subject"]
        query_type = item["query_type"]

        print(f"[{i+1}/{len(LATENCY_QUERIES)}] [{query_type}] {question[:60]}...")

        try:
            chain = build_eval_chain(chapter, subject)
            answer, latency, num_contexts = measure_latency(chain, question)

            all_latencies.append(latency)
            results_by_type.setdefault(query_type, []).append(latency)

            status = "PASS" if latency <= 10.0 else "FAIL"
            print(f"   Latency: {latency:.2f}s {status}")
            print(f"   Contexts: {num_contexts}")
            print(f"   Answer: {answer[:80]}...\n")

            per_query_results.append({
                "query_number": i + 1,
                "question": question,
                "chapter": chapter,
                "subject": subject,
                "query_type": query_type,
                "latency_seconds": round(latency, 4),
                "num_contexts": num_contexts,
                "within_target": latency <= 10.0,
                "answer_preview": answer[:150]
            })

            # Small delay to avoid Groq rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"   ERROR: {e}\n")
            continue

    if not all_latencies:
        print("No results collected — check errors above.")
        return {}

    # ── SUMMARY STATISTICS ────────────────────────────────────────────────────
    mean_lat = statistics.mean(all_latencies)
    min_lat = min(all_latencies)
    max_lat = max(all_latencies)
    std_lat = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
    within_target = sum(1 for lat in all_latencies if lat <= 10.0)
    target_met = mean_lat <= 10.0

    print("=" * 60)
    print("LATENCY SUMMARY")
    print("=" * 60)
    print(f"  Mean latency:    {mean_lat:.2f}s")
    print(f"  Min latency:     {min_lat:.2f}s")
    print(f"  Max latency:     {max_lat:.2f}s")
    print(f"  Std deviation:   {std_lat:.2f}s")
    print(f"  Within target:   {within_target}/{len(all_latencies)} queries <= 10s")

    print("\n── By Query Type ──")
    for qtype, latencies in results_by_type.items():
        avg = statistics.mean(latencies)
        status = "PASS" if avg <= 10.0 else "FAIL"
        print(f"  {qtype:<25} avg: {avg:.2f}s {status}")

    print(f"\n── Overall Target (<=10s): ", end="")
    print(f"PASS ({mean_lat:.2f}s)" if target_met else f"FAIL ({mean_lat:.2f}s)")

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────
    output = {
        "summary": {
            "total_queries": len(all_latencies),
            "mean_latency_seconds": round(mean_lat, 4),
            "min_latency_seconds": round(min_lat, 4),
            "max_latency_seconds": round(max_lat, 4),
            "std_deviation_seconds": round(std_lat, 4),
            "queries_within_target": within_target,
            "target_seconds": 10.0,
            "target_met": bool(target_met),
            "device": DEVICE
        },
        "by_query_type": {
            qtype: {
                "count": len(lats),
                "mean_latency_seconds": round(statistics.mean(lats), 4)
            }
            for qtype, lats in results_by_type.items()
        },
        "per_query": per_query_results
    }

    output_path = EVALUATION_RESULTS_DIR / "latency_results.json"
    with open(str(output_path), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: {output_path}")
    return output


if __name__ == "__main__":
    run_latency_evaluation()
