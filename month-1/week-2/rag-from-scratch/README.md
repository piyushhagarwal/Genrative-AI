# PDF Question Answering System
> A from-scratch RAG (Retrieval-Augmented Generation) pipeline — no LangChain, no LlamaIndex, just the mechanics.

---

## What This Is

A command-line tool that lets you ask natural language questions about any PDF. It loads the PDF, chunks the text, embeds the chunks using OpenAI, stores them in a local Chroma vector database, and retrieves the most relevant chunks to generate a grounded answer — with source references showing exactly which page the answer came from.

Built as a learning project to understand RAG at a mechanical level.

---

## Project Structure

```
pdf-qa/
├── main.py        ← CLI entry point and argument parsing
├── loader.py      ← PDF text extraction + scanned PDF detection
├── chunker.py     ← Three chunking strategies
├── embedder.py    ← OpenAI embeddings + cost tracking
├── store.py       ← Chroma vector store (insert + retrieve)
├── qa.py          ← Prompt building + answer generation
├── chroma_db/     ← Persisted vector store (auto-created)
├── sample_pdfs/   ← Put your PDFs here
└── .env           ← Your OpenAI API key
```

---

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# Clone and enter the project
git clone <your-repo-url>
cd pdf-qa

# Install dependencies
uv sync

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Usage

### Interactive mode (recommended)
Ask multiple questions without restarting:
```bash
uv run python main.py sample_pdfs/attention.pdf
```

### Single question mode
```bash
uv run python main.py sample_pdfs/attention.pdf --question "What is multi-head attention?"
```

### Specify a chunking strategy
```bash
uv run python main.py sample_pdfs/attention.pdf --strategy sentence
uv run python main.py sample_pdfs/attention.pdf --strategy fixed
uv run python main.py sample_pdfs/attention.pdf --strategy recursive   # default
```

### Force re-index with a new strategy
```bash
uv run python main.py sample_pdfs/attention.pdf --strategy fixed --force
```

### Tune retrieval
```bash
# Retrieve top 3 chunks instead of default 5
uv run python main.py sample_pdfs/attention.pdf --top-k 3
```

### Interactive mode commands
| Command | Action |
|---------|--------|
| Any text | Ask a question |
| `cost`  | Print cost breakdown for this session |
| `exit`  | Quit |

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `pdf_path` | required | Path to your PDF file |
| `--strategy` | `recursive` | Chunking strategy: `fixed`, `sentence`, `recursive` |
| `--chunk-size` | `500` | Max chunk size in characters |
| `--top-k` | `5` | Number of chunks to retrieve per question |
| `--force` | `False` | Delete existing index and re-embed from scratch |
| `--question` | None | Single question, non-interactive mode |

---

## How It Works

### Phase 1 — Ingestion (runs once per PDF)

```
PDF file → pypdf extracts text → chunker splits text → OpenAI embeds chunks → Chroma stores vectors
```

1. **Load** — `pypdf` extracts text page by page. Each page is stored with its page number as metadata. Scanned PDFs (no extractable text) are detected and rejected gracefully.

2. **Chunk** — Text is split into overlapping chunks. Three strategies are available:
   - `fixed` — splits every N characters with overlap. Fast, consistent sizes, but cuts mid-sentence.
   - `sentence` — respects sentence boundaries. Better semantic coherence, variable sizes.
   - `recursive` — tries paragraph breaks first, falls back to sentences, then words, then characters. Best semantic coherence. Default choice.

3. **Embed** — Each chunk is sent to OpenAI's `text-embedding-3-small` model and converted to a 1536-dimensional vector. Processed in batches of 100.

4. **Store** — Vectors, raw text, and metadata (page number, strategy, chunk ID) are stored in a local Chroma database. Each PDF gets its own named collection so multiple documents coexist without collision.

### Phase 2 — Query (runs on every question)

```
User question → embed question → cosine similarity search in Chroma → top-k chunks → LLM → answer + sources
```

1. **Embed query** — The question is embedded using the same model as the chunks. This is critical — same model means same vector space.

2. **Retrieve** — Chroma finds the top-k most similar chunks using cosine similarity. Returns text, page number, and similarity score.

3. **Generate** — Retrieved chunks are assembled into a prompt with source labels. `gpt-4o-mini` generates an answer using only the provided chunks — it is explicitly instructed not to use outside knowledge.

4. **Return** — The answer is printed alongside the source chunks, page numbers, and similarity scores so the user can verify.

---

## Chunking Strategies Explained

### Fixed-size
```
"The transformer model uses attention mechanisms. It was intro" | "troduced in 2017. The encoder pro..." 
```
Simple, predictable chunk sizes. The overlap parameter ensures boundary sentences aren't fully lost. Downside: cuts mid-word or mid-sentence.

### Sentence-boundary
```
"The transformer model uses attention mechanisms. It was introduced in 2017." | "The encoder processes input tokens."
```
Splits on sentence endings (`.`, `!`, `?`). Carries over the last N sentences as overlap. Better semantic units than fixed-size.

### Recursive (default)
```
Try "\n\n" (paragraphs) → too big? try "\n" (lines) → too big? try ". " (sentences) → try " " (words) → try "" (chars)
```
Preserves the largest natural boundary possible before falling back. Produces the most semantically coherent chunks. What LangChain's `RecursiveCharacterTextSplitter` does under the hood.

---

## Cost Tracking

Every API call is tracked. At any point during interactive mode, type `cost` to see:

```
========== COST SUMMARY ==========
Embedding tokens used    : 10,128
Generation input tokens  : 1,847
Generation output tokens : 312
Total cost this session  : $0.000634
==================================
```

**Models used and why:**
- Embeddings: `text-embedding-3-small` ($0.02/1M tokens) — good retrieval quality at low cost
- Generation: `gpt-4o-mini` ($0.15/$0.60 per 1M input/output tokens) — sufficient for summarizing retrieved chunks

In RAG, the bottleneck is retrieval quality, not generation power. Upgrading to GPT-4o won't fix bad retrieval. Better chunking and embedding choices will.

---

## Handling Scanned PDFs

If `pypdf` cannot extract text (scanned/image-based PDF), the system detects it and exits with a clear message rather than silently producing empty answers:

```
ValueError: This PDF appears to be scanned or image-based — no extractable text found.
You would need OCR (e.g. pytesseract) to process it.
```

The detection threshold is 80% — if 80% or more of pages have fewer than 20 characters, the PDF is flagged as scanned.

---

## Design Decisions

**Why no LangChain?**
LangChain abstracts away the parts that matter most for understanding. Writing the chunker, embedder, and retriever from scratch means you can explain exactly what's happening at every step — which matters when a system produces a wrong answer and you need to debug it.

**Why cosine similarity and not L2 distance?**
Cosine similarity measures the angle between vectors, not their magnitude. A long chunk and a short chunk about the same topic point in the same direction — cosine treats them as equally similar to a matching query. L2 distance would penalize the longer chunk just for being longer.

**Why `upsert` instead of `insert`?**
Re-running the ingestion pipeline on the same PDF should be idempotent — no duplicate chunks, no errors. `upsert` inserts if the ID is new, updates if it already exists.

**Why `temperature=0` for generation?**
Document QA is a retrieval task, not a creative one. We want the same answer every time for the same question and chunks. Temperature > 0 introduces randomness that can cause the model to paraphrase differently across runs, making it harder to debug retrieval problems.

---

## Dependencies

```toml
[dependencies]
pypdf = "*"
chromadb = "*"
openai = "*"
tiktoken = "*"
python-dotenv = "*"
```

---

## What to Build Next

- **OCR support** — run `pytesseract` when scanned PDF is detected
- **Multi-PDF querying** — query across multiple collections simultaneously  
- **Re-ranking** — after retrieval, use a cross-encoder to re-score chunks before sending to LLM
- **Hybrid search** — combine vector search with BM25 keyword search for better recall on technical terms
- **Evaluation** — build a small test set of question/answer pairs to measure retrieval accuracy across chunking strategies