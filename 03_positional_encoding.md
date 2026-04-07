# 位置编码：Sinusoidal / RoPE / ALiBi

## 1. 为什么需要位置编码？

Self-Attention 是 **置换不变的**（permutation invariant），无法区分 token 顺序。位置编码注入序列位置信息。

---

## 2. Sinusoidal（原始 Transformer）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- 特点：固定、不可学习；理论上可外推但实际效果差
- 使用方式：**加到** embedding 上（additive）

---

## 3. RoPE (Rotary Position Embedding)

**核心思想**：将位置信息编码为 **旋转矩阵**，作用在 Q 和 K 上。

```
q̃_m = R(m) · q    k̃_n = R(n) · k
q̃_m · k̃_n = q^T R(n-m) k   ← 内积只依赖相对位置 (n-m)
```

**直觉**：把向量 **两两配对** 看成复数，乘以 e^{i·m·θ} 做旋转。

**关键优点**：
- 天然编码 **相对位置**，衰减远距离 attention
- 兼容线性 attention
- 支持外推（配合 NTK-aware scaling / YaRN 等）

**代表模型**：LLaMA 全系列、Mistral、Qwen、GPT-NeoX

### 长度外推方法

| 方法 | 做法 | 效果 |
|------|------|------|
| Position Interpolation | 将位置 id 线性缩放到训练长度内 | 需微调，简单有效 |
| NTK-aware Scaling | 调整 base frequency | 免微调或少量微调 |
| YaRN | NTK + attention scaling + 温度 | 效果最好 |
| Dynamic NTK | 推理时根据实际长度动态调 base | 简单免微调 |

---

## 4. ALiBi (Attention with Linear Biases)

**做法**：不加 PE，直接在 attention score 上加一个 **线性偏置**：

```
attention_score(i, j) = q_i · k_j - m · |i - j|
```

- m 是每个头固定的斜率（几何级数：2^{-8/h}, 2^{-16/h}, ...）
- 距离越远，惩罚越大 → 天然的 **相对位置 + 局部偏好**

**优点**：零额外参数，外推能力强
**代表模型**：BLOOM, MPT

---

## 5. 对比总结

| 方法 | 类型 | 作用位置 | 外推能力 | 代表模型 |
|------|------|----------|----------|----------|
| Sinusoidal | 绝对 | 加到 embedding | 差 | 原始 Transformer |
| Learned PE | 绝对 | 加到 embedding | 差 | BERT, GPT-2 |
| RoPE | 相对 | 乘到 Q, K | 中（需扩展） | LLaMA, Mistral |
| ALiBi | 相对 | 加到 attn score | 强 | BLOOM, MPT |

---

## 6. 手写代码

### 6.1 Sinusoidal Position Encoding

```python
import torch
import math

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()        # (max_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)
```

### 6.2 RoPE (Rotary Position Embedding)

```python
def precompute_freqs(d_head, max_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)            # (max_len, d_head/2)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    # x: (B, H, S, D)，D 必须是偶数
    x1, x2 = x[..., ::2], x[..., 1::2]       # 拆成偶数和奇数维
    # 旋转公式：[x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)

# 使用示例
cos, sin = precompute_freqs(d_head=64, max_len=2048)
# cos/sin: (S, d_head/2) → 广播到 (1, 1, S, d_head/2)
cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

> **记忆口诀**：两两配对、乘旋转矩阵、内积只看相对距离。

---

## 7. 面试高频问

**Q: RoPE 的核心原理？**
> 将 Q/K 向量两两配对，用旋转矩阵编码位置。旋转后的 Q·K 内积只依赖相对位置差，天然实现相对位置编码。

**Q: RoPE 如何做长度外推？**
> 常见方法：① Position Interpolation（线性压缩位置 id）② NTK-aware Scaling（调整频率基数）③ YaRN（结合 NTK + 注意力温度缩放）。核心思想都是让旋转频率适应更长序列。

**Q: ALiBi 和 RoPE 的区别？**
> RoPE 作用于 Q/K 向量（乘法），ALiBi 作用于 attention score（加法偏置）。ALiBi 外推更好但表达力可能不如 RoPE。
