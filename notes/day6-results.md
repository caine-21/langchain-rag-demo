# Day 6 — Multi-Agent RAG: Results & Analysis

## 实验目的

验证以下 Orchestration 策略是否能有效捕获并修复 RAG 答案中的质量问题：

- **Critic Agent**（LLM）负责感知：检测幻觉声明、缺失事实、矛盾
- **Orchestrator**（确定性代码）负责决策：计算 severity、输出 verdict
- **Revise Agent**（LLM）负责修复：仅在 verdict=revise 时触发

核心设计原则：**LLM 做感知，代码做决策**。

---

## 配置

| 组件 | 模型 / 规则 |
|---|---|
| Generator | llama-3.1-8b-instant, chunk_size=800, k=3 |
| Critic | llama-3.3-70b-versatile, 返回 JSON findings only |
| Orchestrator | 确定性代码，severity 规则见下 |
| Revise Agent | llama-3.3-70b-versatile, max 1 pass |
| Grounding Score | llama-3.3-70b-versatile, LLM-judge 1–5 |

**Severity 规则（`compute_severity()`）：**

```
contradictions > 0           → high
hallucinated_claims >= 2     → high
hallucinated_claims == 1     → medium
missing_facts >= 2           → medium
otherwise                    → low
```

**Verdict 规则（`compute_verdict()`）：**

```
high   → revise
medium → accept   ← 本次实验的关键问题点
low    → accept
```

---

## 关键结果

| Q# | 问题 | Halluc | Missing | Contradict | Severity | Verdict | Grounding |
|---|---|---|---|---|---|---|---|
| Q1 | 文档主要内容 | 0 | 0 | 0 | low | accept | 5/5 |
| Q2 | 实习经历 | 0 | 1 | 0 | low | accept | 5/5 |
| Q3 | 教育背景 | 0 | 1 | 0 | low | accept | 5/5 |
| Q4 | 校园活动 | 0 | 1 | 0 | low | accept | 4/5 |
| Q5 | 技术技能 | 1 | 1 | 0 | **medium** | **accept** | **1/5** |

**汇总：**

- 5/5 接受，0/5 修订
- Grounding 均分：before=4.00/5，after=4.00/5，Delta=+0.00
- Severity 分布：high=0, medium=1, low=4

---

## Q5 Case Study — Policy Design Gap

**问题：** 作者有哪些技术技能？

### 答案问题

Generator 将 Python 错误分类为"语言"（与日语、英语并列），而原文档中 Python 属于"技能"类别。同时答案中加入了"AI编程辅助工具（Claude Code与VS Code Copilot）"这一编辑性评论，不在上下文中。

```
原始答案（节选）：
  1. 语言：Python、Office、PowerShell、命令行工具   ← Python 被错误归类为语言
  2. 语言：日语（N3）、英语（CET-6，托福 80 分）
  3. 技能：驾驶、Linux
  此外，作者还提到使用过AI编程辅助工具（Claude Code与VS Code Copilot）  ← 超出上下文
```

### Critic 检测结果

```json
{
  "hallucinated_claims": ["语言：Python"],
  "missing_facts": ["技能：python | office | powershell | 命令行工具 | 驾驶 | linux"],
  "contradictions": []
}
```

Critic 正确识别了 Python 的错误归类（hallucination=1），也标记了结构性缺失（missing=1）。

### Orchestrator 决策

```
hallucinated_claims=1 → severity=medium
medium → verdict=accept   ← 未触发修订
```

### 最终后果

- Grounding 评分：**1/5**（"Answer ignores the context and provides unrelated technical skills"）
- 答案存在已知错误，未被修复，直接输出为 final answer

---

## 关键发现

### 发现 1：medium severity 未触发修订，但 grounding=1/5

当前策略中 medium → accept，意味着"1个幻觉声明"不足以触发修订。然而 Q5 的 grounding 评分为 1/5，说明答案质量极差。**severity 为 medium，但实际影响为 high。**

### 发现 2：Grounding Score 与 Severity 信号解耦

系统同时计算了 severity（来自 Critic findings 计数）和 grounding score（来自独立 LLM judge）。两者理应相关，但 Q5 暴露了断层：Critic 的 finding 计数不足以反映答案的实际质量崩溃。

### 发现 3：Critic 自身的局限

Q5 中 Critic 标记 "missing_facts" 为完整技能列表，但这些技能实际上出现在答案中（只是归类错误）。Critic 无法区分"内容缺失"与"结构错误"，导致 findings 的语义不够精确。

---

## 影响分析

当前策略的风险：

**"label 没错，但 decision function 太粗"**

- Severity 只看 findings 数量（flat count），不考虑 findings 的语义权重
- medium severity 统一 → accept，忽略了某些 medium 案例的实际质量损失
- Grounding score 作为事后指标被计算，但未被 Orchestrator 使用

这是一个典型的 **AI Orchestration policy design gap**：感知层（Critic）和评分层（Grounding）提供了足够的信号，但决策层（Orchestrator）没有消费这些信号。

---

## 改进假设

### 方案 A：Risk-Weighted Severity

不再用 flat count，而是引入权重：

```python
def compute_severity(critique, grounding_score):
    if len(critique["contradictions"]) > 0:
        return "high"
    if len(critique["hallucinated_claims"]) >= 2:
        return "high"
    if len(critique["hallucinated_claims"]) == 1:
        # 引入 grounding score 作为 tiebreaker
        if grounding_score < 3:
            return "high"   # grounding 差 → 升级为 high
        return "medium"
    if len(critique["missing_facts"]) >= 2:
        return "medium"
    return "low"
```

### 方案 B：Conditional Revise for Medium

```python
def compute_verdict(severity, critique, grounding_score):
    if severity == "high":
        return "revise"
    if severity == "medium":
        has_hallucination = len(critique["hallucinated_claims"]) > 0
        low_grounding = grounding_score < 3
        if has_hallucination or low_grounding:
            return "revise"   # conditional revise
    return "accept"
```

### 对比

| 策略 | Q5 结果 |
|---|---|
| 当前（flat count, medium→accept） | accept，grounding=1/5，未修复 |
| 方案 A（grounding tiebreaker） | severity→high，触发 revise |
| 方案 B（conditional revise） | medium + hallucination=true → revise |

两种方案都能解决 Q5 问题。方案 B 更保守（只改 verdict 逻辑），方案 A 从根本上重新定义 severity 语义，更系统。

---

## 结论

Day 6 验证了多智能体架构的基本流程是通的（Critic → Orchestrator → Revise）。核心学习不是"系统有 bug"，而是揭示了一个**可复现的 policy design gap**：

> 当 Critic 的 findings 计数不足以反映实际答案质量时，flat severity 规则会系统性地低估风险，导致有质量问题的答案被接受。

下一步：实现方案 B（最小改动），在相同 5 个问题上重跑，验证 Q5 是否触发修订，并观察修订后 grounding 是否提升。
