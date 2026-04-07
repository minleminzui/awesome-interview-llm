# 归一化：LayerNorm / RMSNorm

## 1. Layer Normalization (LayerNorm)

**公式**：
```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```
- μ, σ²：沿隐藏维度计算的均值和方差
- γ, β：可学习的缩放和偏移参数（shape = d_model）

**关键特点**：
- 对 **每个样本** 的隐藏维度独立归一化（不同于 BatchNorm 跨 batch）
- 训练和推理行为一致（不依赖 batch 统计量）
- 原始 Transformer 用 **Post-LN**：`x + LayerNorm(SubLayer(x))`

---

## 2. RMSNorm (Root Mean Square Normalization)

**公式**：
```
RMSNorm(x) = γ ⊙ x / RMS(x)
RMS(x) = √(1/d · Σ x_i²)
```

**与 LayerNorm 的区别**：
| | LayerNorm | RMSNorm |
|---|-----------|---------|
| 去均值（re-centering） | ✅ 减去 μ | ❌ 不做 |
| 缩放（re-scaling） | ✅ 除以 σ | ✅ 除以 RMS |
| 偏移 β | ✅ 有 | ❌ 无 |
| 计算量 | 需算 μ 和 σ² | 只算 RMS |
| 速度 | 基线 | **快 ~10-15%** |

**核心洞察**：论文发现 LayerNorm 的效果主要来自 **re-scaling 而非 re-centering**，去掉均值计算不影响性能。

**代表模型**：LLaMA 全系列、Mistral、Qwen、Gemma（现代 LLM 几乎全用 RMSNorm）

---

## 3. Pre-LN vs Post-LN

```
Post-LN:  x + SubLayer(LayerNorm(x))  ← 原始 Transformer（容易梯度爆炸）
Pre-LN:   x + LayerNorm(SubLayer(x))  ← 修正版（训练更稳定）
```

**现代 LLM 标配**：**Pre-RMSNorm**（先 Norm 再过 Attention/FFN）

| 方案 | 优点 | 缺点 |
|------|------|------|
| Post-LN | 收敛后效果可能更好 | 训练不稳定，需 warmup |
| Pre-LN | 训练稳定，不需小心调 lr | 理论收敛精度略低 |

---

## 4. 面试高频问

**Q: 为什么现代 LLM 用 RMSNorm 而非 LayerNorm？**
> RMSNorm 去掉了均值计算和偏移参数，计算更快，且实验表明不损失性能。

**Q: LayerNorm 和 BatchNorm 的区别？**
> BatchNorm 沿 batch 维度归一化，依赖 batch 统计量，推理需 running mean/var；LayerNorm 沿 hidden 维度归一化，每个样本独立，训推一致。NLP/LLM 中序列长度可变且 batch 小，LayerNorm 更合适。

**Q: Pre-LN 为什么训练更稳定？**
> Pre-LN 下残差路径上的梯度不经过 Norm 层，梯度可以直接流回底层，缓解梯度消失/爆炸。

**Q: DeepNorm 了解吗？**
> DeepNorm 是 Microsoft 提出的方案，在 Post-LN 基础上给残差加一个 α 缩放系数，结合精心设计的初始化，兼顾 Post-LN 的收敛效果和 Pre-LN 的训练稳定性。用于 1000 层级别的深层 Transformer。

---

## 5. 手写代码

### 5.1 LayerNorm

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 5.2 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

### 5.3 BatchNorm（对比用）

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

> **面试关键区分**：LayerNorm 沿最后一维（hidden）归一化，BatchNorm 沿第 0 维（batch）归一化。

---

## 6. 一句话速记

| 概念 | 一句话 |
|------|--------|
| LayerNorm | 减均值 + 除标准差 + 可学习缩放偏移 |
| RMSNorm | 只除 RMS，更快，效果不降 |
| Pre-LN | 先 Norm 再算，训练稳定，现代标配 |
| Post-LN | 先算再 Norm，原始方案，训练难调 |
