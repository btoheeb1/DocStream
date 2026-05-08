"""
evaluation/evaluate_ragas.py
-----------------------------
Evaluates DocStream's RAG pipeline using the RAGAS Faithfulness metric.

Faithfulness measures whether each claim in the LLM's answer is supported
by the retrieved context chunks, detecting hallucinations.

Score range: 0.0 to 1.0
Target: >= 0.80

Usage:
    python evaluation/evaluate_ragas.py

Output:
    - Per-question faithfulness scores printed to terminal
    - Full results saved to evaluation_results/ragas_results.json
"""

import sys
import json
import time
from pathlib import Path

# Add src/ to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
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

# ── EVALUATION DATASET ────────────────────────────────────────────────────────
# 10 hand-crafted questions with ground truth answers from
# Chapter 10: Biotechnology and its Application (Biology).
# Extend with additional chapters/subjects as needed.
EVAL_DATASET = [
    {
        "question": "What is Biotechnology?",
        "ground_truth": (
            "Biotechnology deals with industrial scale production of "
            "biopharmaceuticals and biologicals using genetically modified "
            "microbes, fungi, plants, and animals."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What is Bt cotton and how does it work?",
        "ground_truth": (
            "Bt cotton is a genetically modified cotton plant containing a gene "
            "from Bacillus thuringiensis that produces a toxin killing certain "
            "insects like the cotton bollworm."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What is RNA interference (RNAi)?",
        "ground_truth": (
            "RNA interference is a method of cellular defense involving silencing "
            "of a specific mRNA due to a complementary dsRNA molecule that binds "
            "to and prevents translation of the mRNA."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What is gene therapy?",
        "ground_truth": (
            "Gene therapy is a collection of methods that allows correction of a "
            "gene defect diagnosed in a child or embryo by inserting genes into "
            "a person's cells and tissues to treat a disease."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What is the role of Agrobacterium tumefaciens in plant biotechnology?",
        "ground_truth": (
            "Agrobacterium tumefaciens is a pathogen modified into a cloning vector "
            "to deliver genes of interest into plant cells by natural infection, "
            "resulting in transgenic plants."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What are transgenic animals?",
        "ground_truth": (
            "Transgenic animals have had their DNA manipulated to possess and express "
            "an extra foreign gene, used for studying gene function, disease models, "
            "and biological product production."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What is the significance of Rosie the cow in biotechnology?",
        "ground_truth": (
            "Rosie was the first transgenic cow produced in 1997 that produced human "
            "protein-enriched milk containing human alpha-lactalbumin, nutritionally "
            "more balanced for human babies than normal cow milk."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What is ELISA and what is it used for?",
        "ground_truth": (
            "ELISA stands for Enzyme Linked Immunosorbent Assay, a diagnostic technique "
            "used to detect antigens or antibodies in a sample, commonly used to diagnose "
            "HIV infection and other diseases."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What are the ethical issues related to genetically modified organisms?",
        "ground_truth": (
            "Ethical issues include concerns about altering organisms, loss of biodiversity, "
            "unknown long-term effects on ecosystems, and potential harm to non-target "
            "organisms. Bioethics involves careful consideration of moral issues around "
            "biological research."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
    {
        "question": "What is the Cry protein and which organism produces it?",
        "ground_truth": (
            "Cry proteins are insecticidal proteins produced by Bacillus thuringiensis, "
            "encoded by cry genes and toxic to specific insects, used in developing "
            "pest-resistant transgenic crops."
        ),
        "chapter": "10. Biotechnology and its Application",
        "subject": "biology"
    },
]

# ── EVALUATION PROMPT ─────────────────────────────────────────────────────────
# Strict prompt instructing the LLM to answer only from retrieved context,
# ensuring faithfulness scores reflect true retrieval grounding.
EVAL_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are DocStream, a helpful AI study assistant.
Answer the question based strictly on the textbook context provided below.
Do not use any outside knowledge. If the answer is not in the context, say so.

Chat History:
{chat_history}

Textbook Context:
{context}

Question: {question}

Answer:"""
)


def build_eval_chain(chapter: str, subject: str):
    """
    Build a RAG chain configured for evaluation.

    Uses verbose=False to keep evaluation output clean.

    Args:
        chapter (str): Chapter name.
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


def run_ragas_evaluation() -> dict:
    """
    Run RAGAS faithfulness evaluation over the test dataset.

    Collects LLM answers and retrieved contexts for each test question,
    then passes them to RAGAS for automated faithfulness scoring.

    Returns:
        dict: Evaluation results including per-question scores and summary.
    """
    ensure_directories()

    print("=" * 60)
    print("DocStream — RAGAS Faithfulness Evaluation")
    print("=" * 60)
    print(f"Total questions: {len(EVAL_DATASET)}\n")

    questions, answers, contexts, ground_truths = [], [], [], []
    per_question_results = []

    for i, item in enumerate(EVAL_DATASET):
        question = item["question"]
        chapter = item["chapter"]
        subject = item["subject"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(EVAL_DATASET)}] {question}")

        try:
            chain = build_eval_chain(chapter, subject)
            response = chain({"question": question})

            answer = response["answer"]
            source_docs = response.get("source_documents", [])
            retrieved_contexts = [doc.page_content for doc in source_docs]

            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved_contexts)
            ground_truths.append(ground_truth)

            per_question_results.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "chapter": chapter,
                "subject": subject,
                "num_contexts": len(retrieved_contexts)
            })

            print(f"   Answer preview: {answer[:100]}...")
            print(f"   Contexts retrieved: {len(retrieved_contexts)}\n")

            # Delay to avoid Groq rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"   ERROR: {e}\n")
            continue

    # ── RUN RAGAS SCORING ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Running RAGAS faithfulness scoring...")
    print("=" * 60)

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(ragas_dataset, metrics=[faithfulness])
    results_df = results.to_pandas()

    # ── PRINT RESULTS ─────────────────────────────────────────────────────────
    print("\n── Per-Question Faithfulness Scores ──")
    for i, row in results_df.iterrows():
        score = row.get("faithfulness", "N/A")
        q = questions[i][:60] if i < len(questions) else "Unknown"
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"  Q{i+1}: {q}")
        print(f"       Faithfulness: {score_str}")

    avg_faithfulness = results_df["faithfulness"].mean()
    target_met = avg_faithfulness >= 0.80

    print(f"\n── Overall Average Faithfulness: {avg_faithfulness:.4f} ──")
    print(f"── Target: >= 0.80 ──")
    print("TARGET MET" if target_met else "BELOW TARGET")

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────
    output = {
        "summary": {
            "total_questions": len(questions),
            "average_faithfulness": float(avg_faithfulness),
            "target": 0.80,
            "target_met": bool(target_met)
        },
        "per_question": []
    }

    for i, row in results_df.iterrows():
        if i < len(per_question_results):
            entry = per_question_results[i].copy()
            entry["faithfulness_score"] = float(row.get("faithfulness", 0))
            output["per_question"].append(entry)

    output_path = EVALUATION_RESULTS_DIR / "ragas_results.json"
    with open(str(output_path), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: {output_path}")
    return output


if __name__ == "__main__":
    run_ragas_evaluation()
