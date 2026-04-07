"""
InfoNCE Loss（对比学习核心损失）

原理：
  给定 anchor，拉近 positive，推远 negatives
  本质是一个 (N+1)-way 分类问题：正样本是第 0 类，其余 N 个负样本是干扰项

公式：
  L = -log( exp(sim(q, k+) / τ) / Σ_i exp(sim(q, k_i) / τ) )

  等价于 CrossEntropy(similarity_matrix / τ, labels=对角线)

应用：SimCLR, MoCo, CLIP, 句向量对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 版本 1：最简写法（面试首选）==========
def infonce_loss(q, k, temperature=0.07):
    """
    q: (B, D) anchor embeddings (已 L2 归一化)
    k: (B, D) positive embeddings (已 L2 归一化)
    同一 batch 内其他样本互为负样本
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    # 相似度矩阵: (B, B)，对角线是正样本对
    logits = q @ k.T / temperature              # (B, B)
    labels = torch.arange(q.size(0), device=q.device)  # [0, 1, 2, ..., B-1]
    return F.cross_entropy(logits, labels)


# ========== 版本 2：对称 InfoNCE（SimCLR / CLIP 风格）==========
def symmetric_infonce_loss(z1, z2, temperature=0.07):
    """
    z1, z2: (B, D) 同一样本的两个 view
    对称计算：z1→z2 和 z2→z1 的 loss 取平均
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature             # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)     # z1 query, z2 key
    loss_21 = F.cross_entropy(logits.T, labels)   # z2 query, z1 key
    return (loss_12 + loss_21) / 2


# ========== 版本 3：显式正负样本写法（理解原理用）==========
def infonce_explicit(q, k_pos, k_negs, temperature=0.07):
    """
    q:      (B, D)     anchor
    k_pos:  (B, D)     正样本
    k_negs: (B, N, D)  N 个负样本
    """
    q = F.normalize(q, dim=-1)
    k_pos = F.normalize(k_pos, dim=-1)
    k_negs = F.normalize(k_negs, dim=-1)

    # 正样本相似度: (B, 1)
    pos_sim = (q * k_pos).sum(dim=-1, keepdim=True) / temperature

    # 负样本相似度: (B, N)
    neg_sim = torch.bmm(k_negs, q.unsqueeze(-1)).squeeze(-1) / temperature

    # 拼接: (B, 1+N)，label=0（正样本在第 0 列）
    logits = torch.cat([pos_sim, neg_sim], dim=-1)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


# ========== 版本 4：CLIP 双塔 ==========
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable_temp=True):
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        else:
            self.log_temp = torch.log(torch.tensor(1.0 / temperature))

    def forward(self, image_emb, text_emb):
        """image_emb, text_emb: (B, D)"""
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        temperature = torch.exp(self.log_temp)         # 可学习温度
        logits = image_emb @ text_emb.T * temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


if __name__ == "__main__":
    B, D = 32, 128
    q = torch.randn(B, D)
    k = torch.randn(B, D)

    print(f"InfoNCE loss:           {infonce_loss(q, k):.4f}")
    print(f"Symmetric InfoNCE loss: {symmetric_infonce_loss(q, k):.4f}")

    k_negs = torch.randn(B, 16, D)
    print(f"Explicit InfoNCE loss:  {infonce_explicit(q, k, k_negs):.4f}")

    clip = CLIPLoss()
    print(f"CLIP loss:              {clip(q, k):.4f}")
