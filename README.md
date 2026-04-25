# RAG Demo — Chunk Strategy Experiment

A minimal RAG (Retrieval-Augmented Generation) pipeline built with LangChain. The focus is not just running RAG, but **comparing how chunk size affects retrieval quality** through a controlled experiment.

## What it does

```
PDF → split into chunks → local embedding → vector search → LLM answer
```

Runs the same question twice with different chunk sizes and compares:

| | Experiment A | Experiment B |
|---|---|---|
| chunk_size | 200 | 800 |
| chunks produced | ~10 | ~3 |
| retrieval quality | fragmented, vague answer | coherent, specific answer |

**Finding:** Small chunks break semantic units — the retrieved fragments lack context and the LLM produces generic answers. Larger chunks preserve paragraph-level meaning and produce significantly better answers.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Document loader | `PyPDFLoader` | PDF parsing |
| Text splitter | `RecursiveCharacterTextSplitter` | Respects paragraph boundaries |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, no API needed |
| Vector store | `InMemoryVectorStore` | Zero-dependency, demo-appropriate |
| LLM | `Groq / llama-3.1-8b-instant` | Fast, free tier available |
| Chain | `LangChain LCEL` | Composable retriever + prompt + LLM |
| Observability | `LangSmith` | Trace every run — latency, tokens, retrieved docs |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)

# 3. Add a PDF
# Place any PDF in this directory named sample.pdf

# 4. Run
python rag_demo.py
```

First run downloads the embedding model (~90MB). Subsequent runs use the cached model.

## Known limitations

- `all-MiniLM-L6-v2` is trained primarily on English — for Chinese documents, `bge-large-zh` would give better retrieval accuracy
- `InMemoryVectorStore` is not persistent — for production use Chroma, Pinecone, or Weaviate
- No reranking step — a cross-encoder reranker would improve precision on ambiguous queries
- No evaluation metrics — production RAG should be measured with RAGAS (faithfulness, context precision, context recall)

## Observability

Every run is traced via [LangSmith](https://smith.langchain.com). Each trace shows the full pipeline breakdown:

| Node | Experiment A (small chunks) | Experiment B (large chunks) |
|---|---|---|
| VectorStoreRetriever | 0.02s | 0.02s |
| ChatGroq (LLM) | ~1.67s | ~0.90s |
| Total latency | 1.69s | 0.94s |
| Tokens used | 371 | 1,382 |

**Insight:** Large chunks use 3.7× more tokens but are 44% faster end-to-end and produce higher-quality answers. The LLM call dominates latency — retrieval is negligible at this scale.

To enable tracing, add to `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=rag-demo
```

## What this demonstrates

- RAG pipeline architecture: loader → splitter → embedder → retriever → generator
- Experimental thinking: controlled variable (chunk_size), observable outcome (answer quality)
- Provider flexibility: Groq for generation, local model for embeddings — decoupled by design
- Observability: LangSmith tracing to measure latency, token cost, and retrieval quality per run
