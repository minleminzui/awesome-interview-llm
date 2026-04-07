# MoE (Mixture of Experts)

## 1. 核心思想

用 **多个 FFN（专家）** 替代单个 FFN，每个 token 只激活 **Top-K 个专家**。

```
标准 FFN:   每个 token 过 1 个 FFN        → 参数 = 计算量
MoE FFN:    每个 token 只过 K/N 个 FFN    → 参数 ↑↑, 计算量 ≈ 不变
```

> 例：Mixtral 8x7B = 8 个专家，每 token 选 2 个 → 总参数 46.7B，激活参数 ≈ 12.9B

---

## 2. 架构

```
Input x: (B, S, d)
        ↓
   ┌─ Gate(x) ─┐        ← Router / 门控网络
   │  (B,S,N)  │        ← N 个专家的权重
   │  Top-K    │
   ↓           ↓
Expert_i(x)  Expert_j(x)   ← 只激活 K 个专家
   ↓           ↓
 w_i · out_i + w_j · out_j  ← 加权求和
        ↓
   Output: (B, S, d)
```

---

## 3. 手写代码

### 3.1 Router (Gate)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)  # (d, N)
        self.top_k = top_k

    def forward(self, x):
        # x: (B, S, d)
        logits = self.gate(x)                              # (B, S, N)
        scores = F.softmax(logits, dim=-1)                 # (B, S, N)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)  # (B, S, K)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # 归一化
        return topk_scores, topk_indices
```

### 3.2 单个 Expert（就是 SwiGLU FFN）

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

### 3.3 MoE Layer（朴素实现）

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        # x: (B, S, d)
        B, S, d = x.shape
        scores, indices = self.router(x)          # (B, S, K), (B, S, K)
        output = torch.zeros_like(x)              # (B, S, d)

        # 朴素循环：每个专家处理分配给它的 token
        for k in range(self.top_k):
            for e_idx in range(len(self.experts)):
                mask = (indices[:, :, k] == e_idx)            # (B, S) 哪些 token 分给专家 e
                if mask.any():
                    expert_input = x[mask]                     # (n_tokens, d)
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += scores[:, :, k][mask].unsqueeze(-1) * expert_output

        return output
```

### 3.4 MoE Layer（高效 Token Dispatch 版本）

```python
class MoELayerEfficient(nn.Module):
    """用 scatter/gather 实现 token dispatch，避免逐专家循环"""
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        B, S, d = x.shape
        x_flat = x.view(-1, d)                              # (B*S, d)
        scores, indices = self.router(x)                     # (B, S, K)
        scores = scores.view(-1, self.top_k)                 # (B*S, K)
        indices = indices.view(-1, self.top_k)               # (B*S, K)

        output = torch.zeros_like(x_flat)                    # (B*S, d)

        for e_idx in range(self.num_experts):
            # 找出所有选了这个专家的 (token, k) 对
            token_mask = (indices == e_idx)                   # (B*S, K)
            if not token_mask.any():
                continue
            # 哪些 token 选了这个专家（至少在某个 top-k 槽位选了）
            selected = token_mask.any(dim=-1)                 # (B*S,)
            expert_input = x_flat[selected]                   # (n, d)
            expert_output = self.experts[e_idx](expert_input) # (n, d)
            # 对应的权重
            weight = (scores * token_mask.float()).sum(dim=-1)[selected]  # (n,)
            output[selected] += weight.unsqueeze(-1) * expert_output

        return output.view(B, S, d)
```

---

## 4. Load Balancing Loss（负载均衡损失）

**问题**：Router 容易"塌缩"——把大部分 token 都分给少数专家（赢者通吃）。

**解决**：加一个辅助 loss 鼓励专家负载均衡。

```python
def load_balancing_loss(router_logits, top_k_indices, num_experts):
    """
    Switch Transformer 的 auxiliary loss
    router_logits: (B*S, N) 路由的 softmax 前 logits
    top_k_indices: (B*S, K) 选中的专家 id
    """
    num_tokens = router_logits.shape[0]
    probs = F.softmax(router_logits, dim=-1)        # (B*S, N)

    # f_i: 分给专家 i 的 token 比例
    one_hot = F.one_hot(top_k_indices[:, 0], num_experts).float()  # (B*S, N) 取 top-1
    f = one_hot.mean(dim=0)                          # (N,) 每个专家的 token 占比

    # p_i: 路由概率的均值
    p = probs.mean(dim=0)                            # (N,) 每个专家的平均路由概率

    # 最小化 f·p 的不均衡程度
    # 理想情况 f_i = p_i = 1/N
    return num_experts * (f * p).sum()
```

> **直觉**：`f` 是实际分配比例，`p` 是路由概率。如果某专家 f 和 p 都高，说明它"垄断"了，loss 会惩罚这种情况。

---

## 5. 面试高频问

**Q: MoE 的核心优势？**
> 总参数量大但每个 token 只激活部分专家，实现"大模型容量，小模型计算量"。参数效率和计算效率解耦。

**Q: Top-K 怎么选？**
> 通常 K=1（Switch Transformer）或 K=2（Mixtral）。K=1 最高效但路由不稳定，K=2 更稳定且效果更好。

**Q: 负载均衡 loss 的作用？**
> 防止路由塌缩（所有 token 分给少数专家），让每个专家处理的 token 数量大致相等，提高并行效率。

**Q: MoE 的部署挑战？**
> ① 总参数大，显存高（需放下所有专家权重）② 专家并行需要 All-to-All 通信 ③ 负载不均导致 GPU 利用率低 ④ Top-K 路由增加内存随机访问。

**Q: Expert Parallelism 怎么做？**
> 不同专家放在不同 GPU 上。每步需要 All-to-All 通信：token dispatch（把 token 发到对应专家的 GPU）+ token combine（结果发回来）。通信量 ∝ token数 × hidden_dim。

**Q: MoE 和 Dense 模型怎么选？**
> 相同计算预算下 MoE 效果更好（更大模型容量）；但部署显存更高、对通信带宽要求高。适合训练预算充足但推理要求高效的场景。

---

## 6. 代表模型配置

| 模型 | 专家数 | Top-K | 总参数 | 激活参数 |
|------|--------|-------|--------|----------|
| Switch Transformer | 128 | 1 | 1.6T | 13B |
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B |
| DeepSeek-V2 | 160 | 6 | 236B | 21B |
| DeepSeek-V3 | 256 | 8 | 671B | 37B |
| Qwen-MoE | 60 | 4 | 14.3B | 2.7B |

---

## 7. 一句话速记

| 概念 | 一句话 |
|------|--------|
| MoE | 多个 FFN 专家，每 token 只选 Top-K 个，大容量小计算 |
| Router | 一层线性 + softmax + top-k，决定 token 去哪个专家 |
| Load Balance Loss | N × Σ(f_i · p_i)，防止路由塌缩 |
| Expert Parallelism | 专家分布在不同 GPU，靠 All-to-All 通信 |
