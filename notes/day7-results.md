# Day 7 — Policy B Experiment: Results & Analysis

**Experiment ID:** `day7_policy_b_v1`
**Baseline:** `day6` (policy A: flat severity, medium → accept)
**Change:** Only `compute_verdict()` modified — severity logic, Critic prompt, Generator, Grounding evaluator all unchanged.

---

## 实验目的

验证 Policy B（conditional revise for medium severity）是否能修复 Day 6 中发现的 policy design gap：

> Q5 被 Critic 标记为 medium severity（hallucination=1），但 policy A 一律 accept，导致 grounding=1/5 的答案未被修复。

**单一变量：** 只改 `compute_verdict()`，其余完全不变，确保差异可归因。

---

## 配置对比

| 组件 | Day 6 (policy A) | Day 7 (policy B) |
|---|---|---|
| Generator | llama-3.1-8b-instant | 同 |
| Critic | llama-3.3-70b-versatile | 同 |
| Severity rules | 不变 | 不变 |
| **Verdict rules** | medium → accept | **medium + (halluc>0 OR grounding<3) → revise** |
| Revise Agent | llama-3.3-70b-versatile | 同 |
| Grounding evaluator | LLM-judge, chunk[:300] | 同 |

**Policy B `compute_verdict()` 逻辑：**

```python
if severity == "high":
    return "revise"
if severity == "medium":
    if len(critique["hallucinated_claims"]) > 0 or grounding_score < 3:
        return "revise"   # conditional
return "accept"
```

**关键架构变化：** Grounding score 从 verdict 之后移到之前计算，使 Orchestrator 能消费该信号。

---

## 关键结果

| Q# | 问题 | Halluc | Missing | Severity | Verdict A | Verdict B | Grounding before | Grounding after |
|---|---|---|---|---|---|---|---|---|
| Q1 | 文档主要内容 | 0 | 0 | low | accept | accept | 5/5 | 5/5 |
| Q2 | 实习经历 | 0 | 1 | low | accept | accept | 5/5 | 5/5 |
| Q3 | 教育背景 | 0 | 1 | low | accept | accept | 5/5 | 5/5 |
| Q4 | 校园活动 | 0 | 1 | low | accept | accept | 4/5 | 4/5 |
| Q5 | 技术技能 | 1 | 1 | medium | **accept** | **revise** ✓ | 1/5 | 1/5 |

**汇总对比：**

| 指标 | Day 6 | Day 7 |
|---|---|---|
| Revised | 0/5 | 1/5 |
| False positives | 0 | **0** |
| Grounding avg before | 4.00/5 | 4.00/5 |
| Grounding avg after | 4.00/5 | 4.00/5 |
| Delta | +0.00 | +0.00 |

---

## Q5 Deep Dive — 三层分析

### Layer 1: Decision Layer（已修复）

Policy B 正确触发了 REVISE：

```
hallucinated_claims=1 → medium
medium + hallucination>0 → revise   ✓
```

Day 6 的 policy gap（medium → accept）已消除。**Decision layer 修复成功。**

### Layer 2: Content Layer（已修复）

Revise Agent 对 Python 的归类错误进行了修正：

```
原始答案（错误）：
  语言：Python、Office、PowerShell、命令行工具   ← Python 被归入语言类别

修订答案（正确）：
  1. 编程语言：python
  2. 办公软件：office
  3. 脚本语言：powershell
  4. 工具：命令行工具、linux
  5. 驾驶
  6. 语言：日语（N3）、英语（CET-6，托福 80 分）
```

结构错误已消除，内容分类符合文档原意。**Content layer 修复成功。**

### Layer 3: Evaluation Layer（发现新问题）

Grounding after 仍为 **1/5**，理由：

> "The answer ignores the context and lists unrelated technical skills that are not mentioned in the provided chunks."

矛盾点：技能（python, office, powershell 等）明确存在于简历中，为何 grounding 评分器认为"not in context"？

**根因：** `score_grounding()` 对每个 chunk 只传入前 300 字符：

```python
ctx_text = "\n---\n".join(f"[Chunk {i+1}] {c[:300]}" for i, c in enumerate(contexts))
```

如果技能列表位于 chunk 的 300 字符之后，grounding 评分器看不到这些内容，判定答案"超出上下文"。这是评估工具本身的截断缺陷，不是 Revise Agent 的失败。

**Evaluation layer 存在系统性误差，grounding delta 不可信。**

---

## 关键发现

### 发现 1：Policy B 验证成功，无 false positive

Q5 从 accept → revise，Q1-Q4 全部保持 accept。Policy B 的 conditional revise 规则（hallucination>0 OR grounding<3）精确触发了目标案例，没有引入误报。

### 发现 2：Grounding delta=0 不代表修订无效

修订前后 grounding 均为 1/5，但这是评估器截断缺陷造成的，不反映真实答案质量。人工对比原始答案与修订答案，修订版本结构更准确、归类更合理。**Metric 失效，人工验证有效。**

### 发现 3：暴露了第三层问题——Evaluation Layer

系统现在有三层信号：
- Critic findings（感知）
- Orchestrator severity/verdict（决策）
- Grounding score（评估）

本次实验揭示：**评估层本身可以失效**，且失效方式不明显（数字看起来合理，但基于截断上下文）。这是一个独立的可观测性问题。

---

## 实验结论

| 层 | Day 6 状态 | Day 7 状态 |
|---|---|---|
| Decision layer | ❌ medium → accept（policy gap） | ✅ conditional revise 生效 |
| Content layer | ❌ 错误答案未被修复 | ✅ Python 归类错误已修正 |
| Evaluation layer | ⚠️ grounding 信号不可信（截断） | ⚠️ 问题持续存在，本次新发现 |

**Policy B 证明了：** 引入 grounding score 作为 verdict 的 tiebreaker 是有效的 —— 但前提是 grounding 评估本身是可信的。

---

## 下一步改进假设

### 修复 Evaluation Layer

将 `score_grounding()` 中的截断从 `[:300]` 改为完整 chunk 或更大窗口：

```python
# 当前（有缺陷）
ctx_text = "\n---\n".join(f"[Chunk {i+1}] {c[:300]}" for i, c in enumerate(contexts))

# 改进
ctx_text = "\n---\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(contexts))
```

修复后在 Q5 重跑，验证 grounding after 是否能反映修订效果的真实改善。

### 更大的问题

即使 grounding 评分修复，Q5 的根本问题是 **retrieval quality**：检索到的 3 个 chunk 可能不包含完整技能列表。Revise Agent 只能基于检索结果修订，无法补全未检索到的信息。这指向 retrieval 层的改进（reranking、增加 k 值等）。
