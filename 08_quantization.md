# 量化：INT8 / INT4 / GPTQ / AWQ

## 1. 为什么要量化？

```
FP16: 每参数 2 bytes → 7B 模型 ≈ 14 GB
INT4: 每参数 0.5 bytes → 7B 模型 ≈ 3.5 GB
```

量化 = 用低精度表示权重/激活，减少显存和计算量。

---

## 2. 量化基础

### 对称量化

```python
def symmetric_quantize(x, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.abs().max() / qmax
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale

def symmetric_dequantize(x_q, scale):
    return x_q.float() * scale
```

### 非对称量化

```python
def asymmetric_quantize(x, n_bits=8):
    qmin, qmax = 0, 2 ** n_bits - 1
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)
    x_q = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
    return x_q, scale, zero_point

def asymmetric_dequantize(x_q, scale, zero_point):
    return (x_q.float() - zero_point) * scale
```

### Per-channel vs Per-tensor

```python
def per_channel_quantize(weight, n_bits=8):
    """按输出通道（每行）独立计算 scale"""
    qmax = 2 ** (n_bits - 1) - 1
    scale = weight.abs().amax(dim=-1, keepdim=True) / qmax
    w_q = torch.round(weight / scale).clamp(-qmax, qmax).to(torch.int8)
    return w_q, scale
```

> **per-channel 精度远好于 per-tensor**，几乎是量化必选。

---

## 3. Weight-Only 量化

只量化权重，激活保持 FP16。适合推理场景。

### GPTQ

- 基于 **OBQ (Optimal Brain Quantization)**，逐列量化 + Hessian 补偿
- 用校准数据计算 Hessian 逆，量化误差传播到后续列进行补偿
- 核心：`w_q = round(w)`，残差 `δ = w - w_q` 按 Hessian 补偿到未量化列

### AWQ (Activation-Aware Weight Quantization)

- 观察：少数"显著"权重（对应激活值大的通道）对精度影响大
- 做法：按 **激活幅度** 找到重要通道，给这些通道的权重乘一个 scale（等效变换），保护重要权重
- 比 GPTQ 更快（不需要逐列迭代），效果相当

---

## 4. W8A8 量化（SmoothQuant）

同时量化权重和激活到 INT8。

**难点**：激活中存在 outlier（少数通道值很大）。

**SmoothQuant 做法**：将激活的 outlier "转移"到权重上。

```python
def smooth_quant_transform(weight, act_scales, alpha=0.5):
    """
    act_scales: 每个通道的激活绝对值最大值
    将激活除以 s，权重乘以 s，数学等价
    """
    s = act_scales.pow(alpha) / weight.abs().amax(dim=0).pow(1 - alpha)
    smooth_weight = weight * s.unsqueeze(0)
    # 推理时激活除以 s
    return smooth_weight, s
```

---

## 5. 面试高频问

**Q: 对称量化和非对称量化的区别？**
> 对称量化 zero_point=0，只有 scale 参数，适合分布对称的数据（如权重）。非对称量化有 zero_point，能更好处理偏斜分布（如 ReLU 后的激活）。

**Q: GPTQ 的核心思想？**
> 逐列量化权重，每列量化后产生的误差通过 Hessian 矩阵补偿到后续未量化的列，使整体输出误差最小。

**Q: AWQ 为什么不直接保留重要权重为高精度？**
> 混合精度实现复杂且硬件不友好。AWQ 通过等效缩放变换，在全 INT4 格式下保护重要通道。

**Q: 为什么 Weight-Only 量化比 W8A8 更常用？**
> LLM 推理瓶颈在访存（memory-bound），Weight-Only 减少权重读取量就能加速；激活量化额外增加开销但 decode 阶段收益不大。

---

## 6. 一句话速记

| 概念 | 一句话 |
|------|--------|
| 对称量化 | `x_q = round(x / scale)`，zero_point=0 |
| 非对称量化 | `x_q = round(x / scale + zp)`，能处理偏斜分布 |
| GPTQ | 逐列量化 + Hessian 补偿残差到后续列 |
| AWQ | 按激活幅度找重要通道，缩放保护 |
| SmoothQuant | 激活 outlier 转移到权重，实现 W8A8 |
