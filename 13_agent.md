# Agent：ReAct / Tool Calling / Multi-Agent

## 1. 什么是 LLM Agent？

```
Agent = LLM + 工具调用 + 记忆 + 规划
```

LLM 本身只能生成文本，Agent 让 LLM 能 **观察环境、使用工具、迭代推理**。

---

## 2. ReAct (Reasoning + Acting)

**核心**：让 LLM 交替进行 **Thought（推理）→ Action（行动）→ Observation（观察）**。

```
Thought:  我需要查询今天的天气
Action:   search("北京今天天气")
Observation: 北京今天晴，最高温 25°C
Thought:  用户问的是明天，我需要再查
Action:   search("北京明天天气")
Observation: 北京明天多云，最高温 22°C
Thought:  现在可以回答了
Answer:   北京明天多云，最高温 22°C
```

### 手写代码

```python
import json
import re


def react_agent(llm, tools: dict, user_query: str, max_steps: int = 5) -> str:
    """
    llm:   callable, 输入 prompt 返回文本
    tools: {"tool_name": callable}
    """
    system_prompt = f"""你是一个 AI 助手，可以使用以下工具：
{json.dumps({name: func.__doc__ for name, func in tools.items()}, ensure_ascii=False)}

请按以下格式回答：
Thought: 你的推理过程
Action: tool_name(arg1, arg2)
如果不需要工具，直接回答：
Thought: ...
Answer: 最终答案
"""
    history = f"Question: {user_query}\n"

    for step in range(max_steps):
        response = llm(system_prompt + history)
        history += response + "\n"

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()

        # 解析 Action
        action_match = re.search(r'Action:\s*(\w+)\((.*?)\)', response)
        if action_match:
            tool_name = action_match.group(1)
            tool_args = action_match.group(2)
            if tool_name in tools:
                try:
                    result = tools[tool_name](tool_args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            history += f"Observation: {result}\n"

    return "达到最大步数，未能完成任务"
```

---

## 3. Tool Calling / Function Calling

**OpenAI 风格**：模型输出结构化 JSON 而非自由文本。

```json
{
  "name": "get_weather",
  "arguments": {"city": "北京", "date": "2026-04-06"}
}
```

### 手写代码

```python
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    }
]


def tool_calling_loop(llm, tools_impl: dict, messages: list, max_rounds: int = 5):
    """
    标准 Tool Calling 循环：
    1. LLM 生成回答（可能包含 tool_calls）
    2. 执行工具，结果追加到 messages
    3. 再次调用 LLM，直到不再调用工具
    """
    for _ in range(max_rounds):
        response = llm.chat(messages, tools=TOOL_SCHEMA)

        if response.tool_calls:
            messages.append({"role": "assistant", "tool_calls": response.tool_calls})
            for call in response.tool_calls:
                func_name = call.function.name
                func_args = json.loads(call.function.arguments)
                result = tools_impl[func_name](**func_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        else:
            return response.content

    return "达到最大轮数"
```

---

## 4. Planning：任务分解

### 4.1 Plan-and-Execute

```python
def plan_and_execute(llm, tools, user_query):
    # Step 1: 生成计划
    plan_prompt = f"请将以下任务分解为子步骤：\n{user_query}"
    plan = llm(plan_prompt)  # 返回步骤列表

    # Step 2: 逐步执行
    results = []
    for step in parse_steps(plan):
        result = react_agent(llm, tools, step)
        results.append(result)

    # Step 3: 汇总
    summary_prompt = f"任务：{user_query}\n执行结果：{results}\n请汇总最终答案。"
    return llm(summary_prompt)
```

### 4.2 Self-Refine

```python
def self_refine(llm, task, max_rounds=3):
    """生成 → 自我批评 → 修改，迭代改进"""
    output = llm(f"请完成以下任务：{task}")
    for _ in range(max_rounds):
        critique = llm(f"请批评以下回答的不足：\n{output}")
        if "没有明显问题" in critique:
            break
        output = llm(f"根据批评修改回答：\n原回答：{output}\n批评：{critique}")
    return output
```

---

## 5. Multi-Agent（结合你的 IoA 经验）

### 5.1 基本多智能体框架

```python
class Agent:
    def __init__(self, name: str, llm, system_prompt: str, tools: dict = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.memory: list[dict] = []

    def act(self, message: str) -> str:
        self.memory.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": self.system_prompt}] + self.memory
        response = self.llm(messages)
        self.memory.append({"role": "assistant", "content": response})
        return response


class MultiAgentOrchestrator:
    """多智能体协作：指定 agents 和通信拓扑"""
    def __init__(self, agents: list[Agent]):
        self.agents = {a.name: a for a in agents}

    def discussion(self, task: str, rounds: int = 3) -> str:
        """多轮讨论：每个 agent 看到其他人的发言后回复"""
        context = f"任务：{task}\n"
        for r in range(rounds):
            for name, agent in self.agents.items():
                response = agent.act(context)
                context += f"\n[{name}]: {response}\n"

        # 最后一个 agent 汇总
        summarizer = list(self.agents.values())[-1]
        return summarizer.act(f"请根据以上讨论给出最终答案。\n{context}")

    def route(self, task: str) -> str:
        """路由模式：根据任务类型分配给合适的 agent"""
        router_prompt = f"任务：{task}\n可用 agent：{list(self.agents.keys())}\n请选择最合适的 agent。"
        best_agent_name = self.agents[list(self.agents.keys())[0]].llm(router_prompt).strip()
        if best_agent_name in self.agents:
            return self.agents[best_agent_name].act(task)
        return self.agents[list(self.agents.keys())[0]].act(task)
```

---

## 6. Agent Memory

```python
class AgentMemory:
    """短期记忆 + 长期记忆"""
    def __init__(self, short_term_limit=20, embedding_model=None, vector_db=None):
        self.short_term: list[dict] = []
        self.short_term_limit = short_term_limit
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def add(self, message: dict):
        self.short_term.append(message)
        # 超出限制，旧消息存入长期记忆
        if len(self.short_term) > self.short_term_limit:
            old = self.short_term.pop(0)
            if self.vector_db:
                emb = self.embedding_model.encode(old["content"])
                self.vector_db.insert(emb, old)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """从长期记忆检索相关内容"""
        short = self.short_term
        long = []
        if self.vector_db:
            emb = self.embedding_model.encode(query)
            long = self.vector_db.search(emb, top_k)
        return long + short
```

---

## 7. 面试高频问

**Q: ReAct 的核心思想？**
> Thought-Action-Observation 交替循环。Thought 让 LLM 推理下一步该做什么，Action 调用工具，Observation 获取结果。比 Chain-of-Thought 多了与环境交互的能力。

**Q: Function Calling 和 ReAct 的区别？**
> Function Calling 是模型输出结构化 JSON（训练时内化了 tool schema），更可靠。ReAct 是 prompt engineering，用自然文本格式调用工具。

**Q: 多智能体协作有哪些模式？**
> ① Discussion（多轮讨论达成共识）② Debate（正反方辩论）③ Route（根据任务分配给专家 agent）④ Hierarchical（主管 agent 分配子任务给下属）。

**Q: IoA（你的论文）的核心贡献？**（结合简历）
> 提出异构智能体协作框架，支持不同类型的 agent（不同 LLM、不同工具）动态组队协作，在通用助手、具身 AI 和 RAG 基准上优于现有方案。

**Q: Agent 的 memory 怎么设计？**
> 短期记忆用 sliding window（最近 N 轮对话），长期记忆用向量数据库存储历史，检索时 embedding 相似度召回相关内容。

**Q: Agent 的常见失败模式？**
> ① 无限循环（反复调用同一工具）② 工具调用格式错误 ③ 幻觉（编造工具不存在的返回值）④ 任务分解过细/过粗。

---

## 8. 一句话速记

| 概念 | 一句话 |
|------|--------|
| ReAct | Thought → Action → Observation 循环，推理+行动交替 |
| Tool Calling | 模型输出结构化 JSON 调用工具，比 ReAct 更可靠 |
| Plan-and-Execute | 先分解任务成子步骤，再逐步执行 |
| Multi-Agent | 多个角色化 agent 协作/辩论/路由完成复杂任务 |
