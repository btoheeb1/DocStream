# DocStream: A RAG-Powered Conversational Study Agent

**Final Project — CSC 7644: Applied LLM Development**

DocStream is a conversational AI study assistant that allows students to query course PDF materials in natural language. It returns document-grounded answers, displays relevant textbook diagrams inline, and automatically recommends YouTube videos to supplement learning — all within a single web interface. Users can either query pre-loaded course materials or upload any PDF of their own and begin chatting immediately.

---

## Key Features

- **Document-grounded Q&A** — answers are sourced directly from uploaded course PDFs using Retrieval-Augmented Generation (RAG), not generic web knowledge
- **PDF uploader** — any user can upload their own PDF documents through the browser and start querying them immediately, with no terminal interaction required
- **Multimodal image retrieval** — relevant diagrams and figures are extracted from PDFs, captioned using GPT-4o Vision, and displayed inline alongside text answers
- **Agentic YouTube recommendations** — contextually relevant YouTube videos are surfaced automatically after each educational response using an LLM-generated conversation summary as the search query
- **Smart intent detection** — two-stage classifier distinguishes genuine subject-matter questions from greetings and conversational messages, preventing irrelevant image or video output
- **Multi-turn memory** — conversation history is preserved across turns for context-aware follow-up questions
- **Chapter-level and whole-book retrieval** — users can query a specific chapter or search across the entire subject corpus

---

## Tech Stack and Architecture

### Models and APIs

* Component | Technology 
- LLM --> Llama 3.3 70B via Groq API 
- Embeddings --> HuggingFace sentence-transformers (all-mpnet-base-v2) 
- Image captioning --> GPT-4o Vision (OpenAI API) 
- YouTube search --> youtubesearchpython 

### Frameworks and Libraries

* Component | Technology 
- RAG orchestration --> LangChain (ConversationalRetrievalChain, MMR retrieval) 
- Vector store --> ChromaDB (local persistent storage) 
- Web UI --> Streamlit 
- PDF text extraction --> PyMuPDF (fitz) 
- PDF image extraction --> PyMuPDF (fitz) 
- Text splitting --> LangChain RecursiveCharacterTextSplitter 

### High-Level Architecture

```
User query (Streamlit UI)
        |
        v
Intent classifier (educational vs conversational)
        |
        v
MMR Retrieval (ChromaDB — text + image caption chunks)
        |
        v
Llama 3.3 70B (Groq API) + ConversationBufferMemory
        |
        |-- Text answer
        |-- Relevant diagrams (if user asked for images)
        |-- YouTube recommendations (if educational query)
        |
        v
Streamlit UI response
```

---

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- pip
- A Groq API account (free): https://console.groq.com
- An OpenAI API account (paid, required only for image captioning): https://platform.openai.com

### 1. Clone the repository

```bash
git clone https://github.com/btoheeb1/DocStream.git
cd DocStream
```

### 2. Create and activate a virtual environment

```bash
python -m venv docstream_env

# macOS / Linux
source docstream_env/bin/activate

# Windows
DocStream_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys. See `.env.example` for all required variables and descriptions.

### 5. Add course PDF documents (optional — for pre-loaded course materials tab)

Place your PDF files inside subject folders under `data/class_12/`:

```
data/
└── class_12/
    ├── biology/
    │   ├── 1. The Living World.pdf
    │   ├── 2. Biological Classification.pdf
    │   └── ...
    ├── physics/
    │   └── ...
    └── chemistry/
        └── ...
```

Any subject folder you create inside `data/class_12/` is automatically detected. You are not limited to Biology, Physics, and Chemistry.

### 6. Extract images and generate captions (optional — enables inline diagram display)

```bash
python src/ingestion/extract_images.py
```

This extracts images from your PDFs and sends them to GPT-4o Vision for educational captioning. It may take several minutes depending on document count. It supports automatic resuming if interrupted. OpenAI API credits are required (approximately $0.01 per image).

### 7. Run the ingestion pipeline (required for course materials tab)

```bash
python src/ingestion/vectorize_script.py
```

This chunks, embeds, and stores all PDF text and image captions in ChromaDB. Run once, then re-run whenever new documents are added.

---

## Running the Application

```bash
streamlit run src/app/main.py
```

The app opens automatically at `http://localhost:8501`.

### How to use

The application has two tabs:

**Course Materials tab**
1. Select a subject from the dropdown
2. Select a chapter, or choose "All Chapters" to search the entire subject
3. Type a question in the chat input at the bottom of the page
4. To see inline diagrams, include words like "diagram", "image", or "show me" in your question

**Upload Your Own tab**
1. Click "Upload PDF files" and select one or more PDF documents from your computer
2. Optionally toggle GPT-4o image captioning on or off
3. Click "Process Documents" and wait for the progress bar to complete
4. Start typing questions in the chat input — no terminal interaction required

---

## Running Evaluations

Activate your virtual environment first, then run from the project root:

```bash
# RAGAS Faithfulness evaluation (requires OPENAI_API_KEY)
python src/evaluation/evaluate_ragas.py

# Response latency benchmarking
python src/evaluation/evaluate_latency.py
```

Results are saved as JSON files to `evaluation_results/`.

---

## Repository Organization

```
DocStream/
|
|-- src/
|   |-- config.py                        # Central config: env vars, paths, constants
|   |
|   |-- app/
|   |   |-- main.py                      # Streamlit UI entry point (two-tab layout)
|   |   |-- chatbot_utility.py           # Chapter list loader and vector DB checker
|   |
|   |-- rag/
|   |   |-- chain.py                     # ConversationalRetrievalChain builder
|   |   |-- retriever.py                 # Vectorstore path resolution, image search
|   |   |-- youtube.py                   # Intent detection, YouTube query, video search
|   |
|   |-- ingestion/
|   |   |-- vectorize.py                 # PDF text chunking, embedding, ChromaDB storage
|   |   |-- vectorize_script.py          # CLI entry point for full ingestion pipeline
|   |   |-- extract_images.py            # PDF image extraction and GPT-4o captioning
|   |   |-- ingest_uploaded.py           # Ingestion pipeline for user-uploaded PDFs
|   |
|   |-- evaluation/
|       |-- evaluate_ragas.py            # RAGAS faithfulness metric evaluation
|       |-- evaluate_latency.py          # End-to-end response latency benchmarking
|
|-- data/
|   |-- class_12/                        # Place subject PDF folders here
|
|-- vector_db/                           # Auto-generated: whole-book ChromaDB stores
|-- chapters_vector_db/                  # Auto-generated: per-chapter ChromaDB stores
|-- extracted_images/                    # Auto-generated: extracted PDF image files
|-- uploads/                             # Auto-generated: user-uploaded session files
|-- evaluation_results/                  # Auto-generated: evaluation output JSON files
|
|-- .env.example                         # Template for required environment variables
|-- .gitignore                           # Files excluded from version control
|-- requirements.txt                     # Python dependencies
|-- README.md                            # This file
```

---

## Evaluation Results

| Metric | Score | Target | Status |
|---|---|---|---|
| RAGAS Faithfulness (avg) | 0.9818 | >= 0.80 | Met |
| Response Latency (mean) | 3.98s | <= 10s | Met |
| Queries within latency target | 11 / 12 | -- | -- |

---

## Attributions and Citations

- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*. https://arxiv.org/abs/2005.11401
- Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv*. https://arxiv.org/abs/2309.15217
- LangChain Documentation. (2024). https://docs.langchain.com
- ChromaDB Documentation. (2024). https://docs.trychroma.com
- Groq API Documentation. (2024). https://console.groq.com/docs
- OpenAI API Documentation. (2024). https://platform.openai.com/docs
- NCERT Textbooks — Class 12 Biology, Physics, Chemistry. National Council of Educational Research and Training, India. https://ncert.nic.in
