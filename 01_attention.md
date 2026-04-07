# Attention 机制：MHA / MQA / GQA

## 1. Self-Attention 基础

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **Q/K/V** 分别由输入 X 经线性变换得到：`Q = XW_Q, K = XW_K, V = XW_V`
- **除以 √d_k** 的原因：防止点积值过大导致 softmax 梯度消失（方差缩放）

---

## 2. Multi-Head Attention (MHA)

**核心思想**：将 d_model 拆成 h 个头，每个头独立做 attention，最后拼接。

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
head_i = Attention(QW_Qi, KW_Ki, VW_Vi)
```

### 符号约定

| 符号 | 含义 | 典型值 (7B) |
|------|------|-------------|
| B | batch size | 1~64 |
| S | 序列长度 seq_len | 2048~128K |
| d | 隐藏维度 d_model | 4096 |
| h | Q 头数 n_heads | 32 |
| d_h | 每头维度 d_head = d/h | 128 |
| g | KV 头数 (GQA) | 8 (GQA) / 32 (MHA) / 1 (MQA) |

### 前向维度追踪（逐行标注）

```
输入    X:  (B, S, d)

── 线性投影 ──
Q = X @ W_Q:  (B, S, d) @ (d, d)       → (B, S, d)       FLOPs: 2·B·S·d²
K = X @ W_K:  (B, S, d) @ (d, d)       → (B, S, d)       FLOPs: 2·B·S·d²
V = X @ W_V:  (B, S, d) @ (d, d)       → (B, S, d)       FLOPs: 2·B·S·d²

── reshape + transpose 拆头 ──
Q:  (B, S, d) → (B, S, h, d_h) → (B, h, S, d_h)
K:  (B, S, d) → (B, S, h, d_h) → (B, h, S, d_h)
V:  (B, S, d) → (B, S, h, d_h) → (B, h, S, d_h)

── Attention 计算（每个头独立，共 h 个头）──
scores = Q @ K^T:  (B, h, S, d_h) @ (B, h, d_h, S) → (B, h, S, S)
                                                       FLOPs: 2·B·h·S²·d_h = 2·B·S²·d
scores = scores / √d_h
attn   = softmax(scores, dim=-1):                     (B, h, S, S)
out    = attn @ V:  (B, h, S, S) @ (B, h, S, d_h)  → (B, h, S, d_h)
                                                       FLOPs: 2·B·h·S²·d_h = 2·B·S²·d

── concat + 输出投影 ──
out:  (B, h, S, d_h) → transpose → (B, S, h, d_h) → view → (B, S, d)
out = out @ W_O:  (B, S, d) @ (d, d) → (B, S, d)     FLOPs: 2·B·S·d²
```

### FLOPs 汇总

| 步骤 | 计算 | FLOPs |
|------|------|-------|
| Q/K/V 投影 | 3 次矩阵乘 | 3 × 2BSd² = **6BSd²** |
| Q @ K^T | S×d_h 乘 d_h×S，h 个头 | **2BS²d** |
| attn @ V | S×S 乘 S×d_h，h 个头 | **2BS²d** |
| 输出投影 W_O | 1 次矩阵乘 | **2BSd²** |
| **总计** | | **8BSd² + 4BS²d** |

> **直觉**：投影部分 ∝ S·d²（线性于 S），attention 部分 ∝ S²·d（平方于 S）。
> 当 **S > 2d** 时 attention 成为瓶颈（如 d=4096, S>8192）。

### 参数量

| 权重 | shape | 参数数 |
|------|-------|--------|
| W_Q | (d, d) | d² |
| W_K | (d, d) | d² |
| W_V | (d, d) | d² |
| W_O | (d, d) | d² |
| **总计** | | **4d²** |

> 例：d=4096 → 4 × 4096² = **67M** 参数/层

### KV Cache 显存

```
每层每 token: 2 (K+V) × h × d_h × dtype_bytes = 2 × d × dtype_bytes
总共:        2 × n_layers × d × S × B × dtype_bytes
```
> 例：32 层, d=4096, S=4096, fp16 → 2×32×4096×4096×2 = **2 GB**

### 面试高频问

**Q: 多头注意力的作用？**
> 不同头关注不同子空间的特征（语法、语义、位置等），增强模型的表示能力。

**Q: 为什么要除以 √d_k？**
> 当 d_k 较大时，Q·K 的方差 ≈ d_k，值很大会让 softmax 输出趋近 one-hot，梯度接近 0。除以 √d_k 使方差归一。

**Q: MHA 的计算瓶颈在哪？**
> 当序列长度 S 较短时，瓶颈在 QKV 线性投影（compute-bound）；当 S 较长（S > 2d）时，瓶颈在 S² 的 attention 计算（memory-bound，因为要读写 N×N 矩阵）。

---

## 3. Multi-Query Attention (MQA)

**核心改动**：所有头 **共享一组 K 和 V**，只有 Q 保留多头。

| 对比项 | MHA | MQA |
|--------|-----|-----|
| K/V 头数 | h | **1** |
| KV Cache | 2 × h × d_head | 2 × **1** × d_head |
| 精度 | 最好 | 略降 |
| 推理速度 | 基线 | **显著加速** |

**代表模型**：PaLM, Falcon, StarCoder

---

## 4. Grouped-Query Attention (GQA)

**核心思想**：MHA 和 MQA 的折中——将 h 个 Q 头分成 **g 组**，每组共享一份 K/V。

```
g = 1  →  MQA
g = h  →  MHA
1 < g < h  →  GQA
```

### GQA 维度追踪

```
── 线性投影（区别在 K/V 的输出维度）──
Q = X @ W_Q:  (B, S, d) @ (d, h·d_h)   → (B, S, h·d_h)     FLOPs: 2·B·S·d·(h·d_h) = 2BSd²
K = X @ W_K:  (B, S, d) @ (d, g·d_h)   → (B, S, g·d_h)     FLOPs: 2·B·S·d·(g·d_h)
V = X @ W_V:  (B, S, d) @ (d, g·d_h)   → (B, S, g·d_h)     FLOPs: 2·B·S·d·(g·d_h)

── 拆头 ──
Q:  (B, S, h·d_h)  → (B, h, S, d_h)
K:  (B, S, g·d_h)  → (B, g, S, d_h)  → repeat_interleave(h/g) → (B, h, S, d_h)
V:  同 K

── Attention（和 MHA 完全相同）──
scores = Q @ K^T:  (B, h, S, d_h) @ (B, h, d_h, S) → (B, h, S, S)   FLOPs: 2BS²d
out    = attn @ V:                                                      FLOPs: 2BS²d

── 输出投影 ──
out = concat → (B, S, d) @ W_O → (B, S, d)                            FLOPs: 2BSd²
```

### MHA / GQA / MQA 全面对比

| 对比项 | MHA (g=h) | GQA (g 组) | MQA (g=1) |
|--------|-----------|-----------|-----------|
| KV 头数 | h | g | 1 |
| W_K / W_V shape | (d, d) | (d, g·d_h) | (d, d_h) |
| **QKV 投影参数** | 4d² | d²(2 + 2g/h) | d²(2 + 2/h) |
| **QKV 投影 FLOPs** | 8BSd² | 2BSd²(2 + 2g/h) | 2BSd²(2 + 2/h) |
| **Attention FLOPs** | 4BS²d | 4BS²d（不变） | 4BS²d（不变） |
| **KV Cache / 层** | 2·h·d_h·S | 2·g·d_h·S | 2·d_h·S |
| KV Cache 压缩比 | 1× | **h/g ×** | **h×** |
| 质量 | 最高 | 接近 MHA | 略降 |
| 推理速度 | 基线 | **快（decode 阶段）** | 最快 |

> **Attention 部分 FLOPs 不变**（逻辑上展开后都是 h 个头），省的是 **KV 投影参数 + KV Cache 读写带宽**。
> Decode 阶段是 memory-bound，KV Cache 读取量减少 → 推理实际加速显著。

**代表模型**：LLaMA-2 70B (g=8), Mistral 7B (g=8), Gemma

### 面试高频问

**Q: GQA 相比 MHA 的优势？**
> KV Cache 减少 h/g 倍，推理显存和延迟下降，同时质量损失很小。

**Q: GQA 的 Attention FLOPs 真的没有减少吗？**
> 数学上 attention 部分不变（都展开成 h 个头计算）。省的是 KV 投影的计算量和参数，以及推理时 KV Cache 的 **访存量**——后者才是 decode 阶段的真正瓶颈。

**Q: 如何从 MHA checkpoint 转换到 GQA？**
> 将同组的 K/V 权重取 **均值池化**（mean pooling），论文实验表明这比随机初始化 + 重训效果更好。

**Q: GQA 的 g 怎么选？**
> 常见做法：g = h/8 或 g = 8。太小接近 MQA 精度下降，太大接近 MHA 没有加速收益。

---

## 5. 手写代码

### 5.1 Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    # q: (B, h, S, d_h)   k: (B, h, S, d_h)   v: (B, h, S, d_h)
    d_k = q.size(-1)                                           # d_h
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)         # (B, h, S, S)     FLOPs: 2·B·h·S²·d_h
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)                           # (B, h, S, S)
    return attn @ v                                            # (B, h, S, d_h)   FLOPs: 2·B·h·S²·d_h
```

### 5.2 Multi-Head Attention

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads                                  # h
        self.d_head = d_model // n_heads                        # d_h = d / h
        self.wq = nn.Linear(d_model, d_model, bias=False)      # (d, d)    参数: d²
        self.wk = nn.Linear(d_model, d_model, bias=False)      # (d, d)    参数: d²
        self.wv = nn.Linear(d_model, d_model, bias=False)      # (d, d)    参数: d²
        self.wo = nn.Linear(d_model, d_model, bias=False)      # (d, d)    参数: d²

    def forward(self, x, mask=None):
        B, S, _ = x.shape                                      # x: (B, S, d)
        q = self.wq(x)                                         # (B, S, d)         FLOPs: 2BSd²
        q = q.view(B, S, self.n_heads, self.d_head)            # (B, S, h, d_h)
        q = q.transpose(1, 2)                                  # (B, h, S, d_h)
        k = self.wk(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # 同上
        v = self.wv(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # 同上
        out = scaled_dot_product_attention(q, k, v, mask)      # (B, h, S, d_h)    FLOPs: 4BS²d
        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # (B, S, d)
        return self.wo(out)                                    # (B, S, d)         FLOPs: 2BSd²
    # 总 FLOPs: 8BSd² + 4BS²d
```

### 5.3 Grouped-Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads                                  # h (Q 头数)
        self.n_kv_heads = n_kv_heads                            # g (KV 头数)
        self.n_groups = n_heads // n_kv_heads                   # h/g (每组 Q 头共享一份 KV)
        self.d_head = d_model // n_heads                        # d_h
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)      # (d, h·d_h)   参数: d²
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)   # (d, g·d_h)   参数: d·g·d_h
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)   # (d, g·d_h)   参数: d·g·d_h
        self.wo = nn.Linear(d_model, d_model, bias=False)                    # (d, d)       参数: d²

    def forward(self, x, mask=None):
        B, S, _ = x.shape                                      # x: (B, S, d)
        q = self.wq(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)      # (B, h, S, d_h)   FLOPs: 2BSd²
        k = self.wk(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)   # (B, g, S, d_h)   FLOPs: 2BS·d·g·d_h
        v = self.wv(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)   # (B, g, S, d_h)   FLOPs: 2BS·d·g·d_h
        k = k.repeat_interleave(self.n_groups, dim=1)          # (B, h, S, d_h)    ← 扩展，无 FLOPs
        v = v.repeat_interleave(self.n_groups, dim=1)          # (B, h, S, d_h)
        out = scaled_dot_product_attention(q, k, v, mask)      # (B, h, S, d_h)    FLOPs: 4BS²d (不变)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # (B, S, d)
        return self.wo(out)                                    # (B, S, d)         FLOPs: 2BSd²
    # 总 FLOPs: 2BSd²(2 + 2g/h) + 4BS²d
    # g=h → 8BSd² + 4BS²d (MHA)
    # g=1 → 2BSd²(2 + 2/h) + 4BS²d ≈ 4BSd² + 4BS²d (MQA, h 大时)
```

> **MQA** 就是 `n_kv_heads = 1` 的特例。

---

## 6. 一句话速记

| 方法 | 一句话 |
|------|--------|
| MHA | 每个头都有自己的 KV，质量最好，推理最慢 |
| MQA | 所有头共享一套 KV，推理最快，精度略降 |
| GQA | 分组共享 KV，精度和速度的最佳平衡 |
