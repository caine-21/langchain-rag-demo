# RAG Demo — 项目总结

## 我们做了什么

用 LangChain 构建了一个本地 RAG（检索增强生成）pipeline，并通过对比实验验证了 chunk 策略对检索质量的影响。

**完整链路：**

```
PDF 文档
  → PyPDFLoader 加载
  → RecursiveCharacterTextSplitter 切块
  → sentence-transformers (all-MiniLM-L6-v2) 本地 embedding
  → InMemoryVectorStore 向量存储
  → 相似度检索（top-k=3）
  → ChatGroq (llama-3.1-8b-instant) 生成答案
```

**对比实验：**

| | 实验 A | 实验 B |
|---|---|---|
| chunk_size | 200 | 800 |
| chunk_overlap | 20 | 100 |
| 切块数量 | 10 个 | 3 个 |
| 答案质量 | 模糊（"个人简历"） | 具体（列出教育/实习/项目经历） |
| 原因 | 碎片化，上下文不完整 | 语义完整，召回内容连贯 |

**结论：** chunk_size 过小导致语义被切断，召回的片段缺乏上下文，模型无法生成有效答案。chunk_size=800 时，单块包含完整段落，召回质量和答案质量都显著提升。

---

## 用到的技术栈

| 组件 | 技术 | 说明 |
|---|---|---|
| 文档加载 | `PyPDFLoader` | 读取 PDF，按页分割 |
| 文本切分 | `RecursiveCharacterTextSplitter` | 递归按段落/句子切块 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 本地运行，90MB，无需 API |
| 向量存储 | `InMemoryVectorStore` | 内存向量库，适合 demo 级别 |
| LLM | `Groq / llama-3.1-8b-instant` | 免费 API，响应快 |
| Chain 构建 | `LangChain LCEL` | RunnablePassthrough + Prompt + LLM |
| 环境管理 | `python-dotenv` | .env 管理 API key |

---

## 体现的能力

**1. RAG 链路理解（不只是调包）**
能解释每个组件的作用：为什么要切块、embedding 做什么、向量检索的原理、为什么要 rerank。

**2. 实验设计思维**
不只是"跑通"，而是设计了对照实验——控制变量（只改 chunk_size），观察结果差异，得出可解释的结论。这是 AI QA / AI PM 的核心工作方式。

**3. 工程排错能力**
过程中遇到并解决了：模块路径变更（langchain 1.x API 迁移）、Python 3.14 兼容性（faiss 替换为 InMemoryVectorStore）、Google API quota 耗尽（切换 provider）、Windows 编码问题（GBK → UTF-8）。

**4. 多 provider 经验**
尝试了 Google Gemini embedding、Anthropic、Groq，理解各 provider 的限制和适用场景。

---

## 如何描述这段经历

**30 秒版（面试开场）：**
> 我用 LangChain 搭了一个 RAG pipeline，把 PDF 文档做成可以对话的知识库。重点不是跑通，而是做了一个 chunk 策略的对比实验——同一份文档，chunk_size=200 切出 10 个碎片，答案很模糊；chunk_size=800 切出 3 个语义完整的块，答案质量明显提升。这让我对"召回率取决于 chunk 粒度"有了直接的实验感知。

**技术追问版：**
> Embedding 用的是本地 sentence-transformers，向量检索用 cosine similarity 取 top-3，生成用 Groq 的 llama-3.1-8b。Chain 用 LangChain LCEL 写的，retriever 和 prompt 通过 pipe 串联。

---

## 可能被问到的问题与回答

**Q1：RAG 和直接问 LLM 有什么区别？**
> 直接问 LLM 依赖模型训练时的知识，无法获取私有文档、实时信息或超出上下文窗口的内容。RAG 先检索再生成——把相关文档片段作为上下文喂给模型，回答基于真实文档而不是"记忆"，更可控、更可溯源。

**Q2：chunk_size 怎么选？**
> 没有固定答案，取决于文档结构和查询类型。短问题（事实查询）适合小 chunk，精确匹配；长问题（需要理解段落关系）适合大 chunk。实践中通常先测 512 左右，再根据召回质量调整。overlap 设为 chunk_size 的 10-15% 避免语义在边界处被切断。

**Q3：你这个 demo 和生产级 RAG 的差距在哪？**
> 三块：① 向量库用的是内存版，生产用 Pinecone / Chroma / Weaviate 支持持久化和规模化；② 没有 rerank 步骤，生产中会用 cross-encoder 对召回结果重排序提升精度；③ 没有处理多文档、文档更新、权限控制等工程问题。这个 demo 验证的是链路可行性和 chunk 策略的影响，不是生产方案。

**Q4：embedding 模型为什么用本地的，不用 API？**
> 实验过程中 Google embedding API 在当前网络环境不稳定，换成本地 all-MiniLM-L6-v2。实际上本地模型对中文支持较弱——这个模型主要训练于英文，如果文档是中文，生产环境应该用 `text2vec-large-chinese` 或 `bge-large-zh`。这也是这个 demo 的一个已知局限。

**Q5：你怎么评估 RAG 的效果？**
> 这个 demo 是定性观察（人工判断答案质量）。定量评估通常用 RAGAS 框架，衡量四个指标：faithfulness（答案是否基于召回内容）、answer relevancy（答案是否回答了问题）、context precision（召回的 chunk 是否相关）、context recall（相关内容是否被召回）。这块是我接下来要补的。
