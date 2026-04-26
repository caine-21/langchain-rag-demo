"""
Multi-Agent RAG — Day 6

Architecture:
  Query → Retriever → Generator → Critic Agent → Orchestrator → (Revise Agent) → Final Answer

Critic does perception only (detect issues). Orchestrator makes decisions (severity/verdict).
Max 1 revision loop. No recursive critique.

Usage:
  python multi_agent_rag.py
"""

import os, sys, json, time
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

# ── Config ──────────────────────────────────────────────────────────────────────

EXPERIMENT = "day7_policy_b_v1"  # policy B: conditional revise for medium severity

PDF_PATH = "sample.pdf"
API_KEY  = os.environ.get("GROQ_API_KEY", "")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

QUESTIONS = [
    "这份文档的主要内容是什么？",
    "作者有哪些实习经历？",
    "作者的教育背景是什么？",
    "作者参与过哪些校园活动？",
    "作者有哪些技术技能？",
]

# ── RAG Pipeline ────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题，如果上下文不包含答案请说明。\n\n"
    "上下文：\n{context}\n\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag(embeddings: HuggingFaceEmbeddings):
    pages = PyPDFLoader(PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    ).split_documents(pages)
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=API_KEY)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain, retriever

# ── Critic Agent ────────────────────────────────────────────────────────────────
# LLM role: perception only — detect issues, return findings. No verdict, no severity.

CRITIC_PROMPT = """You are a RAG answer critic. Identify factual issues in the answer relative to the retrieved context.

Question: {question}

Retrieved context:
{contexts}

Answer to evaluate:
{answer}

Detect:
1. hallucinated_claims: statements in the answer NOT supported by the context
2. missing_facts: key facts clearly present in the context but absent from the answer
3. contradictions: internally contradictory statements within the answer

Respond ONLY with a JSON object:
{{"hallucinated_claims": ["..."], "missing_facts": ["..."], "contradictions": ["..."]}}

Use empty lists if no issues found. Be specific — quote or paraphrase the problematic text."""

def critic_agent(question: str, answer: str, contexts: list[str], client: Groq) -> dict:
    ctx_text = "\n---\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(contexts))
    prompt = CRITIC_PROMPT.format(question=question, contexts=ctx_text, answer=answer)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0,
    )
    raw = json.loads(completion.choices[0].message.content or "{}")
    return {
        "hallucinated_claims": raw.get("hallucinated_claims", []),
        "missing_facts":       raw.get("missing_facts", []),
        "contradictions":      raw.get("contradictions", []),
    }

# ── Orchestrator (deterministic policy) ─────────────────────────────────────────
# Code role: interpret signals, compute severity, make verdict. Not the LLM.

def compute_severity(critique: dict) -> str:
    if len(critique["contradictions"]) > 0:
        return "high"
    if len(critique["hallucinated_claims"]) >= 2:
        return "high"
    if len(critique["hallucinated_claims"]) == 1 or len(critique["missing_facts"]) >= 2:
        return "medium"
    return "low"

def compute_verdict(severity: str, critique: dict, grounding_score: int) -> str:
    # policy B: medium is conditional — revise if hallucination present OR grounding < 3
    if severity == "high":
        return "revise"
    if severity == "medium":
        if len(critique["hallucinated_claims"]) > 0 or grounding_score < 3:
            return "revise"
    return "accept"

# ── Revise Agent ─────────────────────────────────────────────────────────────────

REVISE_PROMPT = """You are a RAG answer rewriter. Rewrite the answer using ONLY the provided context.

Question: {question}

Retrieved context:
{contexts}

Original answer:
{answer}

Issues to fix:
- Hallucinated claims (not in context): {hallucinated_claims}
- Missing facts (in context but absent): {missing_facts}
- Contradictions: {contradictions}

Instructions:
1. Remove or correct all hallucinated claims
2. Add the missing facts if they are clearly present in the context
3. Resolve any contradictions
4. Do NOT add information not present in the context
5. Respond with the revised answer only — no preamble or explanation."""

def revise_agent(question: str, answer: str, contexts: list[str], critique: dict, client: Groq) -> str:
    ctx_text = "\n---\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(contexts))
    prompt = REVISE_PROMPT.format(
        question=question,
        contexts=ctx_text,
        answer=answer,
        hallucinated_claims=critique["hallucinated_claims"] or "none",
        missing_facts=critique["missing_facts"] or "none",
        contradictions=critique["contradictions"] or "none",
    )
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0,
    )
    return completion.choices[0].message.content.strip()

# ── Answer Grounding Score ───────────────────────────────────────────────────────
# Measures how well the answer reflects the retrieved context (answer-aware).
# Used to compare original vs revised answer quality.

GROUNDING_PROMPT = """You are evaluating a RAG-generated answer.

Question: {question}

Retrieved context:
{contexts}

Answer:
{answer}

Score from 1 to 5: How well does this answer reflect the information in the retrieved context?
1 = Answer ignores or contradicts the context
3 = Answer partially reflects the context, with some gaps or unsupported claims
5 = Answer fully and accurately reflects what is in the context

Respond ONLY with a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

def score_grounding(question: str, answer: str, contexts: list[str], client: Groq) -> dict:
    ctx_text = "\n---\n".join(f"[Chunk {i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = GROUNDING_PROMPT.format(question=question, contexts=ctx_text, answer=answer)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0,
    )
    parsed = json.loads(completion.choices[0].message.content or "{}")
    return {"score": parsed.get("score", 0), "reason": parsed.get("reason", "")}

# ── Main Pipeline ────────────────────────────────────────────────────────────────

def run_multi_agent(question: str, chain, retriever, groq_client: Groq) -> dict:
    # Step 1: RAG — retrieve + generate
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    time.sleep(0.3)

    # Step 2: Critic — LLM detects issues only
    critique = critic_agent(question, answer, contexts, groq_client)
    time.sleep(0.3)

    # Step 3: Grounding score (before) — needed by Orchestrator for policy B
    grounding_before = score_grounding(question, answer, contexts, groq_client)
    time.sleep(0.3)

    # Step 4: Orchestrator — code computes severity and verdict (policy B)
    severity = compute_severity(critique)
    verdict  = compute_verdict(severity, critique, grounding_before["score"])

    # Step 5: Revise if verdict == "revise" (max 1 pass)
    if verdict == "revise":
        revised_answer = revise_agent(question, answer, contexts, critique, groq_client)
        time.sleep(0.3)
        grounding_after = score_grounding(question, revised_answer, contexts, groq_client)
        time.sleep(0.3)
    else:
        revised_answer = None
        grounding_after = grounding_before

    final_answer = revised_answer if revised_answer else answer

    return {
        "question":         question,
        "original_answer":  answer,
        "critique":         critique,
        "severity":         severity,
        "verdict":          verdict,
        "revised_answer":   revised_answer,
        "final_answer":     final_answer,
        "grounding_before": grounding_before,
        "grounding_after":  grounding_after,
    }

def main():
    print("\n" + "="*60)
    print(f"  Multi-Agent RAG — {EXPERIMENT}")
    print("  Critic + Orchestrator (policy B) + Revise Agent")
    print("="*60)

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"{PDF_PATH} not found — place your PDF here and rename it")

    print("\nLoading embedding model (cached after first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    groq_client = Groq(api_key=API_KEY)

    print("Building RAG pipeline (chunk_size=800)...")
    chain, retriever = build_rag(embeddings)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/5] {q}")
        print('─'*60)

        r = run_multi_agent(q, chain, retriever, groq_client)
        results.append(r)

        print(f"Original:  {r['original_answer'][:200]}")
        issues = r['critique']
        print(f"Critique:  hallucinations={len(issues['hallucinated_claims'])}  "
              f"missing={len(issues['missing_facts'])}  "
              f"contradictions={len(issues['contradictions'])}")
        print(f"Severity:  {r['severity']}  →  Verdict: {r['verdict'].upper()}")
        if r['revised_answer']:
            print(f"Revised:   {r['revised_answer'][:200]}")
        print(f"Grounding: before={r['grounding_before']['score']}/5  "
              f"after={r['grounding_after']['score']}/5"
              + (f"  ({r['grounding_after']['reason']})" if r['revised_answer'] else ""))

    # ── Summary ─────────────────────────────────────────────────────────────────

    revised_count   = sum(1 for r in results if r["verdict"] == "revise")
    avg_before      = sum(r["grounding_before"]["score"] for r in results) / len(results)
    avg_after       = sum(r["grounding_after"]["score"]  for r in results) / len(results)
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for r in results:
        severity_counts[r["severity"]] += 1

    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"Questions run:      {len(results)}")
    print(f"Accepted:           {len(results) - revised_count}/{len(results)}")
    print(f"Revised:            {revised_count}/{len(results)}")
    print(f"Severity breakdown: high={severity_counts['high']}  "
          f"medium={severity_counts['medium']}  low={severity_counts['low']}")
    print(f"Grounding (before): {avg_before:.2f}/5")
    print(f"Grounding (after):  {avg_after:.2f}/5")
    print(f"Delta:              {avg_after - avg_before:+.2f}")

    os.makedirs("notes", exist_ok=True)
    out_path = "notes/day7-results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": EXPERIMENT, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {out_path}")
    print(f"\n✓ {EXPERIMENT} complete.")

if __name__ == "__main__":
    main()
