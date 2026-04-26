# Day 6 — Multi-Agent Critique System: Design Document

**Status:** Design phase  
**Scope:** Add a Critic Agent + one-pass revision loop on top of existing RAG pipeline  
**Router:** Deferred — not in initial implementation  

---

## System Goal

Reduce hallucination and improve answer grounding by embedding a structured
critique loop into the RAG generation pipeline. The critic evaluates each
generated answer against retrieved context and either accepts it or triggers
one revision pass.

This directly targets the three failure modes identified in Day 5:

| Day 5 Finding | Day 6 Fix |
|---|---|
| Attribution bias (cosine similarity not chunk-invariant) | Critic checks grounding sentence-by-sentence |
| Retrieval coverage gaps | Critic detects missing key facts |
| Contradiction rewarded as creativity | Critic flags internal contradictions |

---

## Architecture

```
Query
  │
  ▼
Retriever  (existing RAG pipeline, chunk_size=800)
  │
  ▼
Generator  (existing LLM chain)
  │
  ▼
Critic Agent  ◄─── contexts (retrieved chunks)
  │
  ├── verdict: "accept"  ──────────────────────► Final Answer
  │
  └── verdict: "revise"
        │
        ▼
      Revise Agent  (rewrite with critic feedback)
        │
        ▼
      Final Answer  (no second critique — 1 loop max)
```

**Design constraint:** Maximum one revision loop. No recursive critique.
Prevents infinite loops and keeps latency predictable.

---

## Critic Agent Contract

The Critic receives: `question`, `answer`, `contexts`.  
It returns detected issues only — no verdict, no severity.

```json
{
  "hallucinated_claims": ["<claim in answer not supported by context>"],
  "missing_facts":       ["<key fact needed but absent from answer>"],
  "contradictions":      ["<internally contradictory statement>"]
}
```

**Design principle:** LLM only does perception (detect problems).  
LLM does not make control decisions (verdict, severity).  
All decision logic lives in the Orchestrator as deterministic code.

---

## Orchestrator Policy Layer

Severity and verdict are computed in code, not by the LLM.

```python
def compute_severity(critique: dict) -> str:
    if len(critique["contradictions"]) > 0:
        return "high"
    if len(critique["hallucinated_claims"]) >= 2:
        return "high"
    if len(critique["hallucinated_claims"]) == 1 \
            or len(critique["missing_facts"]) >= 2:
        return "medium"
    return "low"

def compute_verdict(severity: str) -> str:
    return "revise" if severity == "high" else "accept"
```

**Why this separation matters:**

| Layer | Responsibility |
|---|---|
| Critic (LLM) | Extract signals — hallucinations, gaps, contradictions |
| Orchestrator (code) | Interpret signals — compute severity |
| Policy (code) | Make decisions — revise or accept |

Benefits: deterministic, unit-testable, no LLM drift on control logic.  
`assert compute_severity({"contradictions": ["x"], ...}) == "high"` always passes.

---

## Revise Agent Contract

The Revise Agent receives: `question`, `original_answer`, `contexts`,
`critic_output` (the full JSON above).

Its instruction:
> "Rewrite the answer using only the provided context.
>  Fix the issues listed in hallucinated_claims, missing_facts,
>  and contradictions. Do not add information not in context."

Returns: `revised_answer` (plain text).

---

## Orchestrator Logic (pseudocode)

```python
def run(question, chunk_size=800):
    contexts, answer = rag_pipeline(question, chunk_size)
    critique = critic_agent(question, answer, contexts)

    if critique["verdict"] == "revise":
        final = revise_agent(question, answer, contexts, critique)
    else:
        final = answer

    return {
        "question":   question,
        "answer":     final,
        "revised":    critique["verdict"] == "revise",
        "critique":   critique,
    }
```

---

## Implementation Plan

| Step | File | What |
|---|---|---|
| 1 | `multi_agent_rag.py` | Orchestrator + Critic + Revise agents |
| 2 | Run on 5 questions | Collect accept/revise rate + before/after answers |
| 3 | `notes/day6-results.md` | Document revision examples and findings |

**Not in scope:** Router agent, multi-hop retrieval, streaming, async execution.

---

## Success Criteria

1. Critic correctly flags at least one hallucination or missing fact across 5 questions
2. Revised answers score higher on Retrieval Coverage than originals (reuse Day 5 metric)
3. System produces structured, inspectable output — not a black box

---

## Limitations (documented upfront)

- One revision pass only — second-order errors are not caught
- Critic uses same LLM as generator — risk of correlated failure
- No ground truth — "improvement" is measured by Coverage metric, not human eval
