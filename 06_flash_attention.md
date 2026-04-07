# Flash Attention

## 1. 解决什么问题？

标准 Attention 需要 **O(N²)** 显存存储完整的 attention matrix（N = seq_len），长序列 OOM。

Flash Attention 通过 **tiling + 在线 softmax** 将显存降到 **O(N)**，同时因为减少 HBM 访问而更快。

---

## 2. 核心思想

### GPU 内存层次
```
SRAM (片上，20MB，19TB/s)  ←  快但小
HBM  (显存，80GB，2TB/s)   ←  大但慢（瓶颈）
```

### 标准 Attention 的问题
```
S = Q @ K^T    → 写 HBM（N×N）
P = softmax(S) → 读写 HBM（N×N）
O = P @ V      → 读 HBM（N×N）
```
多次读写 N×N 矩阵，**访存密集**。

### Flash Attention 做法
1. **Tiling（分块）**：将 Q/K/V 切成小块，每块可放进 SRAM
2. **在线 Softmax**：逐块计算 softmax，用 **running max + running sum** 在线归一化（Online Softmax / Milakov-Gimelfarb 算法）
3. **不存储完整 attention matrix**：只在 SRAM 中计算局部块，直接累加到输出
4. **重计算（反向传播时）**：不保存 P 矩阵，反向时重新计算 → 用计算换显存

---

## 3. 复杂度对比

| | 标准 Attention | Flash Attention |
|---|---|---|
| 显存 | O(N²) | **O(N)** |
| FLOPs | O(N²d) | O(N²d)（不变） |
| 实际速度 | 基线 | **2-4x 快**（减少 HBM 读写） |
| IO 复杂度 | O(N²) | O(N²d²/M)，M = SRAM 大小 |

---

## 4. Flash Attention 2 改进

| 改进 | 说明 |
|------|------|
| 减少非 matmul FLOPs | 将 rescaling 移到最后，减少中间 rescale 操作 |
| 更好的并行 | 沿序列长度维度并行（FA1 沿 batch×head） |
| Q/K/V 分块策略优化 | 外层循环 Q blocks，内层循环 KV blocks |
| 速度 | 相比 FA1 再提升 **~2x** |

---

## 5. Flash Attention 3 改进（Hopper 架构）

- 利用 H100 **WGMMA + TMA** 指令
- **异步**：overlap matmul 和 softmax（在 Tensor Core 和 CUDA Core 之间流水）
- FP8 支持
- 速度达到 H100 理论峰值的 **~75%**

---

## 6. 面试高频问

**Q: Flash Attention 为什么更快？计算量没减少为什么更快？**
> 计算量（FLOPs）不变，但大幅减少了 HBM 读写次数。GPU 瓶颈往往在访存而非计算，Flash Attention 通过 tiling 把计算搬到快速的 SRAM 上完成，属于 **IO-aware 优化**。

**Q: Flash Attention 如何处理 softmax？**
> 使用在线 softmax 算法：维护 running max 和 running sum，逐块更新。每处理一个新 KV 块时，先更新 max，用新旧 max 的差对之前的结果做修正（rescale），最后归一化。

**Q: Flash Attention 的反向传播怎么做？**
> 前向不保存 N×N 的 attention 矩阵 P，只保存 softmax 的统计量（logsumexp）。反向时重新从 Q/K/V 计算 P 的每个块，用计算换显存。

**Q: Flash Attention 和 xformers memory_efficient_attention 的区别？**
> 思路类似（都是 tiling + 在线 softmax），但 Flash Attention 的 CUDA kernel 实现更高效，成为标准。

---

## 7. 手写代码：在线 Softmax + Tiling 伪代码

### 7.1 标准 Attention（对照）

```python
def standard_attention(Q, K, V):
    S = Q @ K.T / math.sqrt(d)
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O
```

### 7.2 在线 Softmax（Flash Attention 的核心）

```python
def online_softmax_attention(Q, K, V, block_size=64):
    """逐块计算 attention，不存完整 NxN 矩阵"""
    S_len = Q.shape[0]
    d = Q.shape[1]
    O = torch.zeros(S_len, d)
    l = torch.zeros(S_len, 1)  # running sum of exp
    m = torch.full((S_len, 1), float('-inf'))  # running max

    for j in range(0, S_len, block_size):
        kj = K[j:j+block_size]
        vj = V[j:j+block_size]
        # 当前块的 scores
        sij = Q @ kj.T / math.sqrt(d)          # (S, block)

        # 更新 running max
        m_new = torch.max(m, sij.max(dim=-1, keepdim=True).values)

        # 修正旧的 l 和 O
        correction = torch.exp(m - m_new)
        l = l * correction
        O = O * correction

        # 累加当前块
        p = torch.exp(sij - m_new)
        l = l + p.sum(dim=-1, keepdim=True)
        O = O + p @ vj

        m = m_new

    return O / l  # 最终归一化
```

> **面试只需写出在线 softmax 的 3 步**：① 更新 max → ② 修正旧结果 → ③ 累加新块。

---

## 8. 一句话速记

| 概念 | 一句话 |
|------|--------|
| Flash Attention | Tiling + 在线 Softmax，不存完整 NxN 矩阵，IO-aware 优化 |
| 核心收益 | 显存 O(N²)→O(N)，速度 2-4x，精确无近似 |
| FA2 | 更好并行 + 减少非 matmul 操作，再快 2x |
| FA3 | Hopper 架构专用，异步流水 + FP8 |
