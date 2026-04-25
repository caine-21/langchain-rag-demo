"""
RAG Demo — LangChain + Groq + local embeddings (sentence-transformers)

Loads a PDF, splits it into chunks, embeds locally, then runs two experiments
with different chunk sizes to compare retrieval quality.

Usage:
  1. Copy .env.example to .env and add your GROQ_API_KEY
  2. Place any PDF in this directory named sample.pdf
  3. python rag_demo.py
"""

import os, sys
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

# ── 配置 ──────────────────────────────────────────────────────
PDF_PATH = "sample.pdf"
QUESTION = "这份文档的主要内容是什么？请用中文回答。"

API_KEY = os.environ.get("GROQ_API_KEY", "")
if not API_KEY:
    raise ValueError("请在 .env 文件中设置 GROQ_API_KEY")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"找不到 {PDF_PATH}，请把 PDF 复制到此目录并改名为 sample.pdf")

CHUNK_A = dict(chunk_size=200, chunk_overlap=20)
CHUNK_B = dict(chunk_size=800, chunk_overlap=100)

# ── RAG 链 ────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题，如果上下文不包含答案请说明。\n\n"
    "上下文：\n{context}\n\n"
    "问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_experiment(label: str, chunk_size: int, chunk_overlap: int):
    print(f"\n{'='*55}")
    print(f"实验 {label}  chunk_size={chunk_size}  overlap={chunk_overlap}")
    print('='*55)

    # 1. 加载
    pages = PyPDFLoader(PDF_PATH).load()
    print(f"加载：{len(pages)} 页")

    # 2. 切块
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(pages)
    print(f"切块：{len(chunks)} 个 chunk")

    # 3. Embedding → 向量库（本地模型，无需联网）
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    retriever  = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LCEL chain
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=API_KEY)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    # 5. 运行
    answer = chain.invoke(QUESTION, config={"run_name": f"实验{label}"})
    source_docs = retriever.invoke(QUESTION)

    print(f"\n问题：{QUESTION}")
    print(f"\n答案：\n{answer}")
    print(f"\n--- 召回的 chunks（前3）---")
    for i, doc in enumerate(source_docs, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] 第{doc.metadata.get('page', 0)+1}页: {preview}...")

if __name__ == "__main__":
    run_experiment("A（小块）", **CHUNK_A)
    run_experiment("B（大块）", **CHUNK_B)
    print("\n\n✓ 两组实验完成。对比两次答案质量和 chunk 内容的差异。")
