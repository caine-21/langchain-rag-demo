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

## What this demonstrates

- RAG pipeline architecture: loader → splitter → embedder → retriever → generator
- Experimental thinking: controlled variable (chunk_size), observable outcome (answer quality)
- Provider flexibility: Groq for generation, local model for embeddings — decoupled by design
