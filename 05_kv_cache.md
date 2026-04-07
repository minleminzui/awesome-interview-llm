# KV Cache

## 1. 为什么需要 KV Cache？

自回归生成时，每生成一个新 token 都要对之前所有 token 做 attention。如果每次都重算 K/V，复杂度 O(S²)。

**KV Cache**：把历史 token 的 K/V 缓存下来，新 token 只需算自己的 Q/K/V，与缓存拼接即可。

```
无 Cache：生成第 t 个 token → 重算前 t 个 token 的 K,V → O(t²) 总计 O(S³)
有 Cache：生成第 t 个 token → 只算第 t 个的 K,V，与缓存拼接 → O(t) 总计 O(S²)
```

---

## 2. 显存占用

每层缓存 K 和 V，shape = `(B, n_kv_heads, S, d_head)`

```
KV Cache 总显存 = 2 × n_layers × n_kv_heads × d_head × seq_len × B × dtype_bytes
```

> 例：LLaMA-2 7B (32 层, 32 头, d_head=128, fp16, B=1, S=4096)
> = 2 × 32 × 32 × 128 × 4096 × 2 bytes = **2 GB**

GQA (n_kv_heads=8) → 只需 **0.5 GB**，省 4 倍。

---

## 3. 手写代码

### 3.1 带 KV Cache 的 Attention

```python
import torch
import torch.nn as nn
import math

class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        # x: (B, 1, D) decode 阶段只有 1 个新 token
        B, S_new, _ = x.shape
        q = self.wq(x).view(B, S_new, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, S_new, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, S_new, self.n_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # (B, H, S_old+1, D)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, S_new, -1)
        return self.wo(out), new_cache
```

### 3.2 生成循环

```python
def generate(model, prompt_ids, max_new_tokens=100):
    kv_caches = [None] * model.n_layers
    # Prefill: 一次性处理整个 prompt
    hidden, kv_caches = model.forward(prompt_ids, kv_caches)
    next_token = hidden[:, -1:].argmax(dim=-1)

    output = [next_token]
    for _ in range(max_new_tokens):
        # Decode: 每次只输入 1 个 token
        hidden, kv_caches = model.forward(next_token, kv_caches)
        next_token = hidden[:, -1:].argmax(dim=-1)
        output.append(next_token)
        if next_token.item() == eos_id:
            break
    return torch.cat(output, dim=1)
```

---

## 4. 面试高频问

**Q: KV Cache 的原理？**
> 自回归生成时缓存已计算的 K/V 矩阵，新 token 只需计算自己的 K/V 并拼接到缓存中，避免重复计算。

**Q: KV Cache 的显存怎么算？**
> `2 × n_layers × n_kv_heads × d_head × seq_len × batch × dtype_bytes`。GQA/MQA 通过减少 n_kv_heads 来压缩 Cache。

**Q: Prefill 和 Decode 的区别？**
> Prefill 是 compute-bound（一次处理所有 prompt token），Decode 是 memory-bound（每次只算一个 token，瓶颈在读 KV Cache）。

---

## 5. 一句话速记

| 概念 | 一句话 |
|------|--------|
| KV Cache | 缓存历史 K/V，新 token 只算增量，O(S³)→O(S²) |
| Prefill | 处理 prompt，compute-bound，可并行 |
| Decode | 逐 token 生成，memory-bound，瓶颈在读 KV Cache |
| GQA 收益 | KV 头数少 → Cache 小 → Decode 更快 |
