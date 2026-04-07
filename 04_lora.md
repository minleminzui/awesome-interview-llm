# LoRA / QLoRA：参数高效微调

## 1. LoRA 核心原理

**思想**：冻结预训练权重 W₀，用 **低秩分解** 学增量 ΔW。

```
W = W₀ + ΔW = W₀ + B·A
```
- A ∈ R^{r×d_in}（高斯初始化）
- B ∈ R^{d_out×r}（零初始化）→ 训练开始时 ΔW = 0
- r << min(d_in, d_out)，通常 r = 8, 16, 32, 64

**可训练参数**：`r × (d_in + d_out)` vs 全参 `d_in × d_out`
> 例：d_in = d_out = 4096, r = 16 → LoRA 参数 = 131K vs 全参 16.7M（压缩 ~128 倍）

**推理时**：W = W₀ + BA 可 **合并**，无额外延迟。

---

## 2. 关键超参

| 超参 | 说明 | 常用值 |
|------|------|--------|
| r (rank) | 低秩维度 | 8 ~ 64 |
| α (lora_alpha) | 缩放系数，实际缩放 = α/r | 16 ~ 32 |
| target_modules | 加 LoRA 的层 | q_proj, v_proj（最常见）；也可加 k, o, gate, up, down |
| dropout | LoRA 层 dropout | 0.05 ~ 0.1 |

**缩放公式**：`ΔW = (α/r) · B·A`
> α/r 控制 LoRA 更新的强度。α 和 r 等比例增大时效果近似不变。

---

## 3. QLoRA

**核心改进**：在 LoRA 基础上 + 三项技术

| 技术 | 作用 |
|------|------|
| **4-bit NormalFloat (NF4)** | 将冻结权重量化到 4bit（信息论最优量化） |
| **Double Quantization** | 对量化常数再量化，每参数再省 ~0.37bit |
| **Paged Optimizer** | 用 CPU 内存分页管理优化器状态，避免 OOM |

**效果**：65B 模型可在 **单张 48GB GPU** 上微调，性能接近全参 16-bit 微调。

---

## 4. LoRA 实践要点

### 加在哪些层？
```
效果排序（经验）：
q_proj + v_proj（经典最小配置）
< q + k + v + o（attention 全加）
< q + k + v + o + gate + up + down（attention + FFN 全加，效果最好）
```

### rank 怎么选？
- **任务简单 / 数据少**：r = 8 ~ 16 够用
- **任务复杂 / 数据多**：r = 32 ~ 64，甚至 128
- 过大的 r 可能过拟合，过小则欠拟合

### 常见坑
- **label mask 错位**：确保 loss 只算在 response token 上，不算 system/user prompt
- **学习率**：LoRA 一般用较大 lr（1e-4 ~ 3e-4），全参微调用小 lr（1e-5 ~ 2e-5）
- **合并后精度**：合并权重时注意 dtype，fp16 合并可能有精度损失
- **多 LoRA 服务**：推理时可热切换不同 LoRA adapter，基座权重共享

---

## 5. LoRA 变体速览

| 变体 | 改进点 |
|------|--------|
| LoRA+ | A 和 B 用不同学习率（B 更大） |
| DoRA | 将权重分解为 magnitude + direction，LoRA 只调 direction |
| AdaLoRA | 自适应分配不同层的 rank |
| rsLoRA | 缩放改为 α/√r，大 rank 更稳定 |

---

## 6. 面试高频问

**Q: LoRA 的原理？为什么有效？**
> 预训练模型微调时，权重更新矩阵 ΔW 是低秩的。LoRA 用两个小矩阵 B·A 近似 ΔW，大幅减少可训练参数。有效原因：微调本质是在预训练权重附近做小范围调整，低秩就够了。

**Q: LoRA 的 α 和 r 的关系？**
> 实际缩放因子 = α/r。α 控制 LoRA 更新的强度。增大 r 时通常等比增大 α 以保持缩放不变。

**Q: LoRA 为什么 A 用高斯初始化，B 用零初始化？**
> 确保训练开始时 ΔW = BA = 0，模型行为与预训练一致，从而稳定训练。

**Q: LoRA 和全参微调比，什么时候用哪个？**
> 数据充足 + 算力够 → 全参微调效果更好。数据少 / 算力有限 / 多任务切换 → LoRA 更实用。

**Q: 你在工作中做 LoRA 踩过什么坑？**（结合简历）
> ① label mask 错位导致 loss 算到了 prompt token 上，修复后效果提升明显
> ② null 回复样本占比过高（4k/11k），清洗到 500/8k 后模型不再过多输出空回复
> ③ 情节轻重数据分布不均，通过策略补样平衡分布

---

## 7. 手写代码

### 7.1 LoRA Linear 层

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad_(False)  # 冻结原权重
        self.lora_a = nn.Parameter(torch.randn(r, in_features) / math.sqrt(in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        self.linear.weight.data += (self.lora_b @ self.lora_a) * self.scaling
```

### 7.2 给模型注入 LoRA

```python
def inject_lora(model, target_modules=("q_proj", "v_proj"), r=8, alpha=16):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            lora = LoRALinear(module.in_features, module.out_features, r, alpha)
            lora.linear.weight = module.weight  # 共享原权重
            parent = dict(model.named_modules())['.'.join(name.split('.')[:-1])]
            setattr(parent, name.split('.')[-1], lora)
    # 只训练 LoRA 参数
    for n, p in model.named_parameters():
        p.requires_grad = 'lora_' in n
```

> **面试技巧**：写完 LoRALinear 就够了，注入逻辑口述即可。

---

## 8. 一句话速记

| 概念 | 一句话 |
|------|--------|
| LoRA | 冻结原权重，用低秩矩阵 BA 学增量，推理可合并无额外开销 |
| QLoRA | LoRA + 4bit量化冻结权重 + 双重量化 + 分页优化器 → 单卡练大模型 |
| rank | 越大越表达但可能过拟合，8~64 常用 |
| α/r | 控制更新强度的缩放因子 |
