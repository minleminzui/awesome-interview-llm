# 训练对齐：SFT / PPO / DPO / GRPO / GSPO

## 1. 三阶段训练流程

```
Pretrain (PT) → Supervised Fine-Tuning (SFT) → Alignment (RLHF / DPO / GRPO / ...)
```

---

## 2. SFT (Supervised Fine-Tuning)

用 `(instruction, response)` 对做有监督微调。

### 手写代码：SFT Loss（只算 response 部分）

```python
import torch
import torch.nn.functional as F

def sft_loss(logits, labels, loss_mask):
    """
    logits:    (B, S, V)  模型输出
    labels:    (B, S)     目标 token id，prompt 部分填 -100
    loss_mask: (B, S)     1=response, 0=prompt/pad
    """
    shift_logits = logits[:, :-1].contiguous()    # 预测下一个 token
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )
    loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum()
    return loss
```

> **重点**：prompt 部分的 label 设为 -100 或用 mask 屏蔽，只在 response token 上算 loss。

---

## 3. PPO (Proximal Policy Optimization)

### 原理

```
4 个模型：
  Actor  π_θ     —— 被优化的策略
  Critic V_φ     —— 估计状态价值
  Reward Model   —— 给回答打分（冻结）
  Ref Policy π_ref —— SFT 模型（冻结，用于 KL 惩罚）

流程：
  1. Actor 生成回答 y ~ π_θ(·|x)
  2. Reward Model 打分 r(x, y)
  3. Critic 估计 V(s) → 计算 advantage A = r - V
  4. PPO-clip 更新 Actor
  5. MSE 更新 Critic
```

### PPO 目标函数

```
ratio = π_θ(a|s) / π_old(a|s)
L_clip = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
L_total = -L_clip + c1·L_critic - c2·entropy + β·KL(π_θ || π_ref)
```

### 手写代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_per_token_logps(model, input_ids, response_mask):
    """计算每个 response token 的 log prob"""
    logits = model(input_ids).logits              # (B, S, V)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    token_logps = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_logps * response_mask[:, 1:]     # (B, S-1) mask 掉 prompt


def ppo_loss(
    old_logps,         # π_old 的 per-token log prob (B, S)
    new_logps,         # π_θ  的 per-token log prob (B, S)
    advantages,        # GAE advantage (B, S)
    clip_eps=0.2,
):
    """PPO-clip 目标（token-level）"""
    log_ratio = new_logps - old_logps              # log(π_θ / π_old)
    ratio = torch.exp(log_ratio)                   # π_θ / π_old

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()
    return loss


def critic_loss(values, returns):
    """Critic 更新：MSE(V, R)"""
    return F.mse_loss(values, returns)


def kl_penalty(new_logps, ref_logps):
    """KL(π_θ || π_ref) 的近似，防止策略偏离太远"""
    return (new_logps - ref_logps).mean()


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation
    δ_t = r_t + γ·V(t+1) - V(t)
    A_t = Σ (γλ)^l · δ_{t+l}
    """
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    def __init__(self, actor, critic, ref_model, reward_model,
                 lr=1e-5, clip_eps=0.2, kl_coef=0.1):
        self.actor = actor
        self.critic = critic
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)

    @torch.no_grad()
    def generate_and_score(self, prompts):
        """1. 生成回答  2. 打分  3. 算 advantage"""
        responses = self.actor.generate(prompts)
        rewards = self.reward_model(prompts, responses)
        values = self.critic(prompts, responses)
        old_logps = get_per_token_logps(self.actor, responses, mask=None)
        ref_logps = get_per_token_logps(self.ref_model, responses, mask=None)
        advantages, returns = compute_gae(rewards, values)
        return responses, old_logps, ref_logps, advantages, returns

    def step(self, responses, old_logps, ref_logps, advantages, returns):
        """PPO 更新一步"""
        new_logps = get_per_token_logps(self.actor, responses, mask=None)

        # Actor loss
        actor_loss = ppo_loss(old_logps, new_logps, advantages, self.clip_eps)
        kl = kl_penalty(new_logps, ref_logps)
        total_actor_loss = actor_loss + self.kl_coef * kl

        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()

        # Critic loss
        values = self.critic(responses)
        c_loss = critic_loss(values, returns)
        self.critic_optim.zero_grad()
        c_loss.backward()
        self.critic_optim.step()

        return {"actor_loss": actor_loss.item(), "kl": kl.item(), "critic_loss": c_loss.item()}
```

---

## 4. DPO (Direct Preference Optimization)

### 原理

跳过显式 Reward Model，直接用偏好数据优化策略。

**推导**：从 RLHF 目标的闭式解出发，reward 用 policy 的 log ratio 表示：

```
r(x,y) = β · log(π_θ(y|x) / π_ref(y|x)) + const
```

**DPO Loss**：

```
L = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

### 手写代码

```python
def get_sequence_logps(model, input_ids, labels, loss_mask):
    """计算每条样本在 response 部分的总 log prob"""
    logits = model(input_ids).logits                    # (B, S, V)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)   # (B, S-1, V)
    per_token = log_probs.gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, S-1)
    return (per_token * loss_mask[:, 1:]).sum(dim=-1)    # (B,) 每条样本的 log prob

def dpo_loss(policy_logps_w, policy_logps_l, ref_logps_w, ref_logps_l, beta=0.1):
    """
    policy_logps_w/l: π_θ  在 chosen/rejected 上的 log prob (B,)
    ref_logps_w/l:    π_ref 在 chosen/rejected 上的 log prob (B,)
    """
    log_ratio_w = policy_logps_w - ref_logps_w          # log(π_θ/π_ref) for chosen
    log_ratio_l = policy_logps_l - ref_logps_l          # log(π_θ/π_ref) for rejected
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))
    return loss.mean()


class DPOTrainer:
    def __init__(self, model, ref_model, beta=0.1, lr=1e-6):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask,
             chosen_labels, rejected_labels):
        # 当前策略 log prob
        pi_w = get_sequence_logps(self.model, chosen_ids, chosen_labels, chosen_mask)
        pi_l = get_sequence_logps(self.model, rejected_ids, rejected_labels, rejected_mask)
        # Ref 策略 log prob（不需要梯度）
        with torch.no_grad():
            ref_w = get_sequence_logps(self.ref_model, chosen_ids, chosen_labels, chosen_mask)
            ref_l = get_sequence_logps(self.ref_model, rejected_ids, rejected_labels, rejected_mask)

        loss = dpo_loss(pi_w, pi_l, ref_w, ref_l, self.beta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 监控指标
        with torch.no_grad():
            reward_w = self.beta * (pi_w - ref_w)
            reward_l = self.beta * (pi_l - ref_l)
            accuracy = (reward_w > reward_l).float().mean()
        return {"loss": loss.item(), "accuracy": accuracy.item()}
```

---

## 5. GRPO (Group Relative Policy Optimization)

**DeepSeek 提出**，去掉 Critic，用 **组内相对奖励** 作为 advantage。

```
1. 每个 prompt 采样 G 个回答
2. RM 打分 {r_1, ..., r_G}
3. 组内标准化 A_i = (r_i - mean) / std
4. PPO-clip 更新（用组内 advantage）
```

### 手写代码

```python
def grpo_loss(
    new_logps,      # π_θ  的 per-token log prob, (B*G, S)
    old_logps,      # π_old 的 per-token log prob, (B*G, S)
    ref_logps,      # π_ref 的 per-token log prob, (B*G, S)
    rewards,        # (B, G) 每个 prompt 的 G 个采样奖励
    response_mask,  # (B*G, S) response token mask
    clip_eps=0.2,
    beta=0.04,
):
    B, G = rewards.shape

    # 1. 组内标准化 advantage
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True) + 1e-8
    advantages = ((rewards - mean) / std).view(-1)  # (B*G,)

    # 2. PPO-clip（sequence-level）
    seq_new = (new_logps * response_mask).sum(dim=-1)  # (B*G,)
    seq_old = (old_logps * response_mask).sum(dim=-1)
    log_ratio = seq_new - seq_old
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3. KL 惩罚（per-token，防止偏离 ref）
    kl = (new_logps - ref_logps) * response_mask
    kl_loss = beta * kl.sum() / response_mask.sum()

    return policy_loss + kl_loss


class GRPOTrainer:
    def __init__(self, model, ref_model, reward_model, G=8, beta=0.04, lr=1e-6):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.G = G
        self.beta = beta
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    @torch.no_grad()
    def sample_and_score(self, prompts):
        """每个 prompt 采样 G 个回答并打分"""
        all_responses, all_rewards = [], []
        for prompt in prompts:
            responses = [self.model.generate(prompt) for _ in range(self.G)]
            rewards = [self.reward_model(prompt, r) for r in responses]
            all_responses.extend(responses)
            all_rewards.append(rewards)
        return all_responses, torch.tensor(all_rewards)  # rewards: (B, G)

    def step(self, prompts):
        responses, rewards = self.sample_and_score(prompts)
        # ... 计算 old_logps, new_logps, ref_logps, mask ...
        loss = grpo_loss(new_logps, old_logps, ref_logps, rewards, mask, beta=self.beta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "mean_reward": rewards.mean().item()}
```

---

## 6. GSPO (Group Sorted Policy Optimization)

**核心改进**：在 GRPO 基础上，不依赖 Reward Model，用 **排序构造偏好对**。

```
思路：
  1. 每个 prompt 采样 G 个回答
  2. 用规则/验证器打分（如数学题检查答案对错）—— 不需要 RM
  3. 按分数排序，构造 chosen/rejected 对
  4. 用类 DPO 的 pairwise loss 优化
```

### 与 GRPO 的区别

| | GRPO | GSPO |
|---|------|------|
| 需要 RM | ✅ | ❌（规则打分即可） |
| Advantage | 组内标准化连续值 | 排序构造离散偏好对 |
| Loss | PPO-clip | Pairwise ranking loss |
| 适用 | 通用 | 有验证器的场景（数学/代码） |

### 手写代码

```python
def gspo_loss(
    policy_logps,     # (B, G) π_θ 对 G 个回答的 log prob
    ref_logps,        # (B, G) π_ref 的 log prob
    scores,           # (B, G) 每个回答的分数（如 0/1 对错）
    beta=0.1,
):
    """
    按分数排序，高分为 chosen，低分为 rejected
    对每个 (chosen, rejected) 对算 DPO-style loss
    """
    B, G = scores.shape
    total_loss = 0.0
    n_pairs = 0

    for b in range(B):
        # 按分数排序的索引
        sorted_idx = torch.argsort(scores[b], descending=True)
        # 构造偏好对：排名高 vs 排名低
        for i in range(G):
            for j in range(i + 1, G):
                if scores[b, sorted_idx[i]] <= scores[b, sorted_idx[j]]:
                    continue  # 分数相同跳过
                w = sorted_idx[i]  # chosen (高分)
                l = sorted_idx[j]  # rejected (低分)
                log_ratio_w = policy_logps[b, w] - ref_logps[b, w]
                log_ratio_l = policy_logps[b, l] - ref_logps[b, l]
                total_loss += -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))
                n_pairs += 1

    return total_loss / max(n_pairs, 1)


# 向量化版本（高效）
def gspo_loss_vectorized(policy_logps, ref_logps, scores, beta=0.1):
    """
    对每个 prompt，取最高分和最低分回答构成偏好对
    """
    B, G = scores.shape
    best_idx = scores.argmax(dim=-1)    # (B,) 最高分
    worst_idx = scores.argmin(dim=-1)   # (B,) 最低分

    # 取对应 log prob
    pi_w = policy_logps[torch.arange(B), best_idx]
    pi_l = policy_logps[torch.arange(B), worst_idx]
    ref_w = ref_logps[torch.arange(B), best_idx]
    ref_l = ref_logps[torch.arange(B), worst_idx]

    log_ratio_w = pi_w - ref_w
    log_ratio_l = pi_l - ref_l
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))

    # 过滤掉最高分=最低分的样本（没有偏好）
    valid = (best_idx != worst_idx)
    return (loss * valid).sum() / valid.sum().clamp(min=1)
```

---

## 7. 对比总结

| 方法 | 需要 RM | 需要 Critic | 数据 | 在线采样 | 复杂度 |
|------|---------|-------------|------|----------|--------|
| SFT | ❌ | ❌ | (x, y) 对 | ❌ | 最简单 |
| PPO | ✅ | ✅ | 在线采样 | ✅ | 最复杂（4 模型） |
| DPO | ❌ | ❌ | 偏好对 (y_w, y_l) | ❌ | 简单（2 模型） |
| GRPO | ✅ | ❌ | 在线采样 G 个 | ✅ | 中等（3 模型） |
| GSPO | ❌ | ❌ | 采样 + 规则验证 | ✅ | 中等（2 模型） |

---

## 8. 面试高频问

**Q: PPO 的 clip 机制是什么？为什么需要？**
> clip(ratio, 1-ε, 1+ε) 限制策略更新幅度。如果新旧策略差异太大，ratio 被截断，防止策略崩溃（catastrophic update）。

**Q: PPO 需要几个模型？分别是什么？**
> 4 个：Actor（被优化）、Critic（估 V）、Reward Model（打分，冻结）、Ref Policy（KL 锚点，冻结）。

**Q: DPO 相比 PPO 的优势？**
> 不需要 RM 和 Critic，只需 2 个模型 + 离线偏好数据，训练简单稳定。但有 distribution shift 问题。

**Q: DPO 的缺点？**
> 离线方法，偏好数据由旧策略生成导致 distribution shift；对数据质量敏感。

**Q: GRPO 相比 PPO 的改进？**
> 去掉 Critic，用同 prompt 多采样的组内相对排名估计 advantage，减少一个模型。

**Q: GSPO 和 GRPO 的区别？**
> GSPO 不需要 RM，用规则/验证器打分后排序构造偏好对，用 DPO-style pairwise loss。适合数学/代码等有明确对错的场景。

**Q: SFT 的 loss mask 为什么重要？**
> 如果 prompt 部分也算 loss，模型会学"复述 prompt"而非"回答问题"。

**Q: GAE 的作用？**
> Generalized Advantage Estimation 平衡 advantage 估计的偏差和方差。λ→0 低方差高偏差，λ→1 高方差低偏差。

---

## 9. 一句话速记

| 概念 | 一句话 |
|------|--------|
| SFT | 有监督微调，只在 response 上算 cross-entropy |
| PPO | ratio·advantage + clip 限幅 + KL 惩罚，4 模型最复杂 |
| DPO | 偏好对直接优化 log ratio 差值，跳过 RM |
| GRPO | 同 prompt 多采样，组内标准化做 advantage，省掉 Critic |
| GSPO | 采样 + 规则排序构造偏好对，省掉 RM |
