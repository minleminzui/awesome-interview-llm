# 采样策略：Temperature / Top-K / Top-P / Repetition Penalty

## 1. 总览

LLM 输出 logits 后，如何选下一个 token？

```
logits → (温度缩放) → (Top-K 过滤) → (Top-P 过滤) → (重复惩罚) → softmax → 采样
```

---

## 2. 手写代码

### 2.1 Temperature

```python
def apply_temperature(logits, temperature=1.0):
    """
    temperature > 1: 更平坦（更随机）
    temperature < 1: 更尖锐（更确定）
    temperature → 0: 退化为 greedy
    """
    if temperature == 0:
        return logits  # 后续直接 argmax
    return logits / temperature
```

### 2.2 Top-K Sampling

```python
def top_k_filter(logits, k=50):
    """只保留概率最高的 K 个 token，其余设为 -inf"""
    if k == 0:
        return logits
    topk_values, _ = logits.topk(k)
    threshold = topk_values[..., -1:]              # 第 K 大的值
    logits = logits.masked_fill(logits < threshold, float('-inf'))
    return logits
```

### 2.3 Top-P (Nucleus) Sampling

```python
def top_p_filter(logits, p=0.9):
    """
    保留累积概率刚超过 p 的最小 token 集合
    自适应：高置信时选少量 token，不确定时选多个
    """
    sorted_logits, sorted_indices = logits.sort(descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumsum = probs.cumsum(dim=-1)

    # 找到累积概率超过 p 的位置，将之后的 token 移除
    mask = cumsum - probs > p                     # 第一个超过 p 的保留，之后的移除
    sorted_logits[mask] = float('-inf')

    # 恢复原始顺序
    original_logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    return original_logits
```

### 2.4 Repetition Penalty

```python
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    对已生成过的 token，降低其 logit
    penalty > 1: 抑制重复
    """
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def apply_repetition_penalty_batch(logits, generated_ids, penalty=1.2):
    """批量版本"""
    # generated_ids: (B, S)
    score = logits.gather(-1, generated_ids)       # (B, S)
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits.scatter_(-1, generated_ids, score)
    return logits
```

### 2.5 Frequency & Presence Penalty（OpenAI 风格）

```python
def apply_frequency_presence_penalty(logits, token_counts, freq_penalty=0.5, pres_penalty=0.5):
    """
    frequency_penalty: 按出现次数线性惩罚（出现越多惩罚越重）
    presence_penalty:  只要出现过就惩罚固定值（鼓励新 topic）
    logit -= freq_penalty * count + pres_penalty * (count > 0)
    """
    for token_id, count in token_counts.items():
        logits[token_id] -= freq_penalty * count + pres_penalty * (1 if count > 0 else 0)
    return logits
```

### 2.6 完整采样函数

```python
def sample_next_token(
    logits,               # (V,) 单个位置的 logits
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    generated_ids=None,
):
    """完整的采样 pipeline"""
    # 1. Repetition penalty
    if repetition_penalty != 1.0 and generated_ids is not None:
        logits = apply_repetition_penalty(logits.clone(), generated_ids, repetition_penalty)

    # 2. Temperature
    logits = apply_temperature(logits, temperature)

    # 3. Top-K
    logits = top_k_filter(logits, top_k)

    # 4. Top-P
    logits = top_p_filter(logits, top_p)

    # 5. 采样
    if temperature == 0:
        return logits.argmax()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()
```

---

## 3. Greedy vs Sampling vs Beam Search

| 方法 | 做法 | 特点 |
|------|------|------|
| Greedy | argmax | 最确定，但容易重复/无聊 |
| Sampling | 按概率采样 | 多样性好，可能不连贯 |
| Top-K | 只从 top-K 里采样 | 去掉长尾垃圾 token |
| Top-P | 只从累积概率 ≥ P 里采样 | 自适应 K，更灵活 |
| Beam Search | 保留 beam_width 个候选 | 质量高，多样性低 |
| Temperature | 缩放 logits | 控制随机程度 |

---

## 4. 面试高频问

**Q: Top-K 和 Top-P 的区别？**
> Top-K 固定取 K 个 token，不管概率分布如何。Top-P 按累积概率动态决定取多少 token——模型很确定时只取 1-2 个，不确定时取很多个。Top-P 更灵活。

**Q: Temperature 的原理？**
> softmax(logits / T)。T>1 让分布更平（更随机），T<1 更尖（更确定），T→0 退化为 argmax。本质是控制 softmax 的熵。

**Q: Repetition Penalty 和 Frequency Penalty 的区别？**
> Repetition Penalty 是乘/除一个固定系数（不管出现几次）。Frequency Penalty 按出现次数线性惩罚（出现越多扣越多）。Presence Penalty 只看"是否出现过"。

**Q: 推理服务中采样参数怎么设？**
> 创意写作：T=0.8~1.0, top_p=0.95。代码/数学：T=0~0.3（接近 greedy）。对话：T=0.6~0.8, top_p=0.9。

---

## 5. 一句话速记

| 概念 | 一句话 |
|------|--------|
| Temperature | `logits / T`，T 大更随机，T 小更确定 |
| Top-K | 只从概率最高的 K 个里采样 |
| Top-P | 只从累积概率达到 P 的最小集合里采样 |
| Repetition Penalty | 已出现的 token logit 除以惩罚系数 |
