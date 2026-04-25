"""
Hybrid RAG Evaluation Pipeline — Day 5

Metrics:
  RAGAS (optional, may fail on Python 3.14):
    - Faithfulness
    - Answer Relevancy
  Custom (guaranteed):
    - Retrieval Coverage Score  (Groq LLM-as-judge, 1-5)
    - Answer Attribution Score  (cosine similarity, threshold sweep 0.55/0.60/0.65)

Usage:
  python evaluate_rag.py
"""

import os, sys, re, json, time
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from groq import Groq
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────

PDF_PATH   = "sample.pdf"
API_KEY    = os.environ.get("GROQ_API_KEY", "")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

CHUNK_CONFIGS = [
    {"label": "A (chunk=200)", "chunk_size": 200, "chunk_overlap": 20},
    {"label": "B (chunk=800)", "chunk_size": 800, "chunk_overlap": 100},
]

ATTRIBUTION_THRESHOLDS = [0.55, 0.60, 0.65]

QUESTIONS = [
    "这份文档的主要内容是什么？",
    "作者有哪些实习经历？",
    "作者的教育背景是什么？",
    "作者参与过哪些校园活动？",
    "作者有哪些技术技能？",
]

RAG_PROMPT = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题，如果上下文不包含答案请说明。\n\n"
    "上下文：\n{context}\n\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 1: Run RAG ─────────────────────────────────────────────────────────────

def build_rag(chunk_size: int, chunk_overlap: int, embeddings: HuggingFaceEmbeddings):
    pages = PyPDFLoader(PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(pages)
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=API_KEY)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain, retriever, len(chunks)

def run_rag_for_questions(chain, retriever, questions: list[str]) -> list[dict]:
    results = []
    for q in questions:
        answer = chain.invoke(q)
        docs = retriever.invoke(q)
        results.append({
            "question": q,
            "answer": answer,
            "contexts": [d.page_content for d in docs],
        })
        time.sleep(0.3)
    return results

# ── Step 2: Custom Metric — Retrieval Coverage ──────────────────────────────────

COVERAGE_PROMPT = """You are evaluating a RAG retrieval system.

Question: {question}

Retrieved chunks:
{contexts}

Score from 1 to 5: Do these retrieved chunks contain sufficient factual information to answer the question?
1 = Chunks contain no relevant information
3 = Chunks contain partial information
5 = Chunks contain complete information needed for a full answer

Respond ONLY with a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

def score_retrieval_coverage(question: str, contexts: list[str], client: Groq) -> dict:
    ctx_text = "\n---\n".join(f"[Chunk {i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = COVERAGE_PROMPT.format(question=question, contexts=ctx_text)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0,
    )
    parsed = json.loads(completion.choices[0].message.content or "{}")
    return {"score": parsed.get("score", 0), "reason": parsed.get("reason", "")}

# ── Step 3: Custom Metric — Answer Attribution ──────────────────────────────────

def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[。.!?！？])\s*|\n+', text)
    return [s.strip() for s in parts if len(s.strip()) > 10]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def score_attribution(answer: str, contexts: list[str],
                      embeddings: HuggingFaceEmbeddings,
                      thresholds: list[float]) -> dict:
    sentences = split_sentences(answer)
    if not sentences:
        return {t: 0.0 for t in thresholds}

    ctx_embeddings = [
        np.array(embeddings.embed_query(c)) for c in contexts
    ]

    results = {}
    for threshold in thresholds:
        attributed = 0
        for sent in sentences:
            sent_emb = np.array(embeddings.embed_query(sent))
            max_sim = max(cosine_similarity(sent_emb, ce) for ce in ctx_embeddings)
            if max_sim >= threshold:
                attributed += 1
        results[threshold] = round(attributed / len(sentences), 3)

    return results

# ── Step 4: RAGAS (optional) ────────────────────────────────────────────────────

def try_ragas(rag_results: list[dict]) -> dict | None:
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        from langchain_groq import ChatGroq as LCGroq

        data = {
            "question":  [r["question"] for r in rag_results],
            "answer":    [r["answer"]   for r in rag_results],
            "contexts":  [r["contexts"] for r in rag_results],
        }
        dataset = Dataset.from_dict(data)

        llm = LCGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=API_KEY)
        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
        )
        return {
            "faithfulness":     round(float(result["faithfulness"]), 3),
            "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        }
    except Exception as e:
        print(f"  [RAGAS skipped: {e}]")
        return None

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Hybrid RAG Evaluation Pipeline")
    print("="*60)

    print("\nLoading embedding model (cached after first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    groq_client = Groq(api_key=API_KEY)

    all_results = {}

    for cfg in CHUNK_CONFIGS:
        label = cfg["label"]
        print(f"\n{'─'*60}")
        print(f"Config: {label}")
        print(f"{'─'*60}")

        chain, retriever, n_chunks = build_rag(
            cfg["chunk_size"], cfg["chunk_overlap"], embeddings
        )
        print(f"Chunks produced: {n_chunks}")

        print("Running RAG for 5 questions...")
        rag_results = run_rag_for_questions(chain, retriever, QUESTIONS)

        coverage_scores = []
        attribution_scores = {t: [] for t in ATTRIBUTION_THRESHOLDS}

        for r in rag_results:
            # Coverage
            cov = score_retrieval_coverage(r["question"], r["contexts"], groq_client)
            coverage_scores.append(cov["score"])
            time.sleep(0.3)

            # Attribution
            attr = score_attribution(r["answer"], r["contexts"], embeddings, ATTRIBUTION_THRESHOLDS)
            for t in ATTRIBUTION_THRESHOLDS:
                attribution_scores[t].append(attr[t])

        print("Trying RAGAS...")
        ragas = try_ragas(rag_results)

        all_results[label] = {
            "coverage_mean":     round(sum(coverage_scores) / len(coverage_scores), 2),
            "attribution":       {str(t): round(sum(v)/len(v), 3) for t, v in attribution_scores.items()},
            "ragas":             ragas,
            "raw_coverage":      coverage_scores,
            "raw_attribution":   {str(t): attribution_scores[t] for t in ATTRIBUTION_THRESHOLDS},
        }

    # ── Print delta table ──────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  RESULTS — Delta Table (A vs B)")
    print("="*60)

    a = all_results[CHUNK_CONFIGS[0]["label"]]
    b = all_results[CHUNK_CONFIGS[1]["label"]]

    def delta(av, bv):
        return f"+{bv-av:.3f}" if bv >= av else f"{bv-av:.3f}"

    print(f"\n{'Metric':<35} {'chunk=200':>10} {'chunk=800':>10} {'Delta':>8}")
    print("-"*65)

    # Coverage
    ac, bc = a["coverage_mean"], b["coverage_mean"]
    print(f"{'Retrieval Coverage (1-5)':<35} {ac:>10.2f} {bc:>10.2f} {delta(ac,bc):>8}")

    # Attribution threshold sweep
    for t in ATTRIBUTION_THRESHOLDS:
        at = a["attribution"][str(t)]
        bt = b["attribution"][str(t)]
        print(f"  {'Answer Attribution (t='+str(t)+')':<33} {at:>10.3f} {bt:>10.3f} {delta(at,bt):>8}")

    # RAGAS
    def ragas_ok(r):
        return (r is not None
                and r.get("faithfulness") is not None
                and r.get("answer_relevancy") is not None)

    if ragas_ok(a["ragas"]) and ragas_ok(b["ragas"]):
        af, bf = a["ragas"]["faithfulness"], b["ragas"]["faithfulness"]
        ar_, br_ = a["ragas"]["answer_relevancy"], b["ragas"]["answer_relevancy"]
        print(f"{'Faithfulness (RAGAS)':<35} {af:>10.3f} {bf:>10.3f} {delta(af,bf):>8}")
        print(f"{'Answer Relevancy (RAGAS)':<35} {ar_:>10.3f} {br_:>10.3f} {delta(ar_,br_):>8}")
    else:
        print(f"{'Faithfulness (RAGAS)':<35} {'N/A':>10} {'N/A':>10} {'─':>8}")
        print(f"{'Answer Relevancy (RAGAS)':<35} {'N/A':>10} {'N/A':>10} {'─':>8}")

    print(f"\nNote: Attribution threshold sweep 0.55/0.60/0.65 shows sensitivity bounds.")
    print("Note: Max-pooling attribution measures evidence existence, not reasoning correctness.")

    # Save JSON
    out_path = "notes/rag-evaluation-metrics.json"
    os.makedirs("notes", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRaw results saved to {out_path}")

if __name__ == "__main__":
    main()
