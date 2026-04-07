"""
KL 散度 (Kullback-Leibler Divergence) 手写实现

定义：
  KL(P || Q) = Σ P(x) · log(P(x) / Q(x))

含义：用分布 Q 近似分布 P 时的信息损失（不对称）

性质：
  1. KL(P || Q) ≥ 0（Gibbs 不等式）
  2. KL(P || Q) = 0  ⟺  P = Q
  3. 不对称：KL(P || Q) ≠ KL(Q || P)
  4. 不是距离（不满足三角不等式）

LLM 中的应用：
  - RLHF/PPO: KL(π_θ || π_ref) 防止策略偏离太远
  - 知识蒸馏: KL(teacher || student) 让学生模拟老师
  - VAE: KL(q(z|x) || p(z)) 正则化项
"""

import torch
import torch.nn.functional as F
import numpy as np


# ========== 1. 离散 KL 散度（最基础）==========
def kl_divergence(p, q):
    """
    p, q: 概率分布（已归一化，同 shape）
    KL(P || Q) = Σ p_i * log(p_i / q_i)
    """
    assert (p >= 0).all() and (q > 0).all()
    return (p * torch.log(p / q)).sum()


# ========== 2. 用 log prob 算（数值更稳定）==========
def kl_from_logits(logits_p, logits_q):
    """
    从 logits 算 KL，避免手动 softmax 的数值问题
    等价于 F.kl_div，但手写展开
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = torch.exp(log_p)
    return (p * (log_p - log_q)).sum(dim=-1)   # (B,) 每个样本的 KL


# ========== 3. PyTorch 官方写法 ==========
def kl_pytorch(logits_p, logits_q):
    """
    注意 F.kl_div 的输入是 log_q（不是 logits），且 target 是 p（概率）
    """
    log_q = F.log_softmax(logits_q, dim=-1)
    p = F.softmax(logits_p, dim=-1)
    return F.kl_div(log_q, p, reduction='batchmean')  # KL(P || Q)


# ========== 4. LLM 中 token-level KL（PPO/GRPO 用）==========
def token_level_kl(new_logps, ref_logps, response_mask):
    """
    new_logps: π_θ 的 per-token log prob (B, S)
    ref_logps: π_ref 的 per-token log prob (B, S)
    KL ≈ Σ (new_logp - ref_logp) 的近似
    精确版: Σ exp(new_logp) * (new_logp - ref_logp)
    """
    kl = (torch.exp(new_logps) * (new_logps - ref_logps)) * response_mask
    return kl.sum() / response_mask.sum()


# ========== 5. 知识蒸馏 loss ==========
def distillation_loss(student_logits, teacher_logits, temperature=4.0, alpha=0.5,
                      hard_labels=None):
    """
    L = α · KL(teacher_soft || student_soft) · T²  +  (1-α) · CE(student, hard_labels)
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    # KL(teacher || student)，T² 补偿温度缩放对梯度的影响
    kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * temperature ** 2

    if hard_labels is not None:
        ce = F.cross_entropy(student_logits, hard_labels)
        return alpha * kl + (1 - alpha) * ce
    return kl


# ========== 6. JS 散度（对称版 KL）==========
def js_divergence(p, q):
    """
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    M = (P + Q) / 2
    JS 是对称的，取值 [0, log2]
    """
    m = (p + q) / 2
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


if __name__ == "__main__":
    torch.manual_seed(42)

    # 离散 KL
    p = torch.tensor([0.4, 0.3, 0.2, 0.1])
    q = torch.tensor([0.25, 0.25, 0.25, 0.25])
    print(f"KL(P || Q)     = {kl_divergence(p, q):.6f}")
    print(f"KL(Q || P)     = {kl_divergence(q, p):.6f}  ← 不对称！")
    print(f"JS(P, Q)       = {js_divergence(p, q):.6f}  ← 对称")

    # logits 版本
    logits_p = torch.randn(4, 10)
    logits_q = torch.randn(4, 10)
    print(f"\nKL from logits = {kl_from_logits(logits_p, logits_q)}")
    print(f"KL pytorch     = {kl_pytorch(logits_p, logits_q):.6f}")

    # 知识蒸馏
    student = torch.randn(4, 10)
    teacher = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    print(f"\nDistill loss   = {distillation_loss(student, teacher, hard_labels=labels):.6f}")
