# Transformer 架构总览

## 1. 标准 Transformer Block（Decoder-Only, LLaMA style）

```
Input
  ↓
RMSNorm → MHA/GQA (+RoPE) → + Residual
  ↓
RMSNorm → FFN (SwiGLU)    → + Residual
  ↓
Output
```

### 手写代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

---

## 2. FFN 变体

### 标准 FFN（原始 Transformer）

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))
```

### SwiGLU FFN（现代 LLM 标配）

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

> **SwiGLU = SiLU(gate) * up**，比 ReLU FFN 效果更好，现代 LLM 标配。

---

## 3. 完整 GPT 模型骨架

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers

    def forward(self, input_ids, mask=None):
        h = self.tok_emb(input_ids)
        # RoPE 在 attention 内部应用
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.norm(h)
        return self.lm_head(h)
```

---

## 4. Causal Mask（因果掩码）

```python
def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    # shape: (1, 1, S, S)，下三角为 1，上三角为 0
```

---

## 5. 现代 LLM 配置速查

| 模型 | 参数 | n_layers | n_heads | n_kv_heads | d_model | FFN | Norm | PE |
|------|------|----------|---------|------------|---------|-----|------|----|
| LLaMA-2 7B | 7B | 32 | 32 | 32 (MHA) | 4096 | SwiGLU 11008 | RMSNorm | RoPE |
| LLaMA-2 70B | 70B | 80 | 64 | 8 (GQA) | 8192 | SwiGLU 28672 | RMSNorm | RoPE |
| Mistral 7B | 7B | 32 | 32 | 8 (GQA) | 4096 | SwiGLU 14336 | RMSNorm | RoPE |
| Qwen-2 7B | 7B | 28 | 28 | 4 (GQA) | 3584 | SwiGLU 18944 | RMSNorm | RoPE |

---

## 6. 参数量计算

```
Embedding:       vocab_size × d_model
每层 Attention:  (n_heads + 2×n_kv_heads + n_heads) × d_head × d_model = 4d² (MHA)
每层 FFN:        3 × d_model × d_ff  (SwiGLU 有 3 个矩阵)
LM Head:         d_model × vocab_size (常与 embedding 共享)
```

> 例：LLaMA-7B 实际参数 ≈ 32 × (4×4096² + 3×4096×11008) + 2×32000×4096 ≈ 6.7B

---

## 7. 面试高频问

**Q: 手写一个 Transformer Decoder Block？**
> 见上方代码：RMSNorm → GQA → Residual → RMSNorm → SwiGLU → Residual。

**Q: SwiGLU 和 ReLU FFN 的区别？**
> SwiGLU 用 SiLU 激活的 gate 分支与 up 分支做逐元素乘法，多一个线性层但效果更好。

**Q: 为什么 LM Head 经常和 Embedding 共享权重（Tied Embedding）？**
> 减少参数量（vocab_size × d_model 很大），且语义上 embedding 和输出投影的作用相似。

**Q: 为什么用 Pre-RMSNorm + SwiGLU + GQA + RoPE 这套组合？**
> Pre-RMSNorm 训练稳定且快；SwiGLU 效果优于 ReLU；GQA 平衡精度和推理速度；RoPE 天然编码相对位置。这是目前效果和效率的帕累托最优组合。

---

## 8. 一句话速记

| 组件 | 现代标配 | 原始 Transformer |
|------|----------|-----------------|
| Norm | Pre-RMSNorm | Post-LayerNorm |
| Attention | GQA + RoPE | MHA + Sinusoidal |
| FFN | SwiGLU | ReLU + 2 层 |
| 掩码 | Causal (下三角) | 同 |
