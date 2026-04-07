"""
Beam Search 解码

原理：
  Greedy: 每步只保留概率最高的 1 个 token → 容易局部最优
  Beam Search: 每步保留概率最高的 beam_width 个候选序列
              → 在更大搜索空间中找全局更优解

流程（beam_width = k）：
  1. 初始 k 个候选，每个只有 <bos>
  2. 每步：每个候选扩展 vocab_size 个分支 → k × V 个候选
  3. 按累积 log prob 排序，保留 top-k
  4. 遇到 <eos> 的候选存入结果池
  5. 所有 beam 结束或达到 max_len → 从结果池选最优
"""

import torch
import torch.nn.functional as F


def beam_search(model, input_ids, beam_width=3, max_len=50, eos_id=2):
    """
    model:     语言模型，输入 (B, S) 返回 logits (B, S, V)
    input_ids: (1, prompt_len) prompt token ids
    """
    device = input_ids.device
    # 复制 beam_width 份 prompt
    alive_seqs = input_ids.repeat(beam_width, 1)  # (k, prompt_len)
    alive_scores = torch.zeros(beam_width, device=device)  # 累积 log prob
    finished = []  # 已结束的候选 (score, seq)

    for step in range(max_len):
        logits = model(alive_seqs).logits[:, -1]          # (k, V) 取最后一个位置
        log_probs = F.log_softmax(logits, dim=-1)         # (k, V)

        # 每个 beam 扩展所有 vocab → k*V 个候选
        scores = alive_scores.unsqueeze(1) + log_probs    # (k, V)
        scores_flat = scores.view(-1)                     # (k*V,)

        # 取 top-k
        topk_scores, topk_ids = scores_flat.topk(beam_width)  # (k,)
        beam_ids = topk_ids // log_probs.size(-1)              # 来自哪个 beam
        token_ids = topk_ids % log_probs.size(-1)              # 选了哪个 token

        # 拼接新 token
        alive_seqs = torch.cat([
            alive_seqs[beam_ids],
            token_ids.unsqueeze(1)
        ], dim=1)  # (k, seq_len+1)
        alive_scores = topk_scores

        # 检查 EOS
        is_eos = (token_ids == eos_id)
        for i in range(beam_width):
            if is_eos[i]:
                # 长度归一化：score / len^α，α=0.6 常用
                norm_score = alive_scores[i] / (alive_seqs.size(1) ** 0.6)
                finished.append((norm_score.item(), alive_seqs[i].clone()))

        # 移除已结束的 beam，补充继续的
        if is_eos.any():
            keep = ~is_eos
            if keep.sum() == 0:
                break
            alive_seqs = alive_seqs[keep]
            alive_scores = alive_scores[keep]
            # 补齐到 beam_width（复制最优的）
            while alive_seqs.size(0) < beam_width:
                alive_seqs = torch.cat([alive_seqs, alive_seqs[:1]])
                alive_scores = torch.cat([alive_scores, alive_scores[:1]])

    # 把未结束的也加入候选
    for i in range(alive_seqs.size(0)):
        norm_score = alive_scores[i] / (alive_seqs.size(1) ** 0.6)
        finished.append((norm_score.item(), alive_seqs[i]))

    # 返回得分最高的序列
    finished.sort(key=lambda x: x[0], reverse=True)
    return finished[0][1]


# ---------- 最简版本（面试白板核心逻辑）----------
def beam_search_simple(log_prob_fn, beam_width=3, max_len=20, eos_id=2):
    """
    log_prob_fn(seqs) → (k, V) 的 log prob
    最精简版本，只保留核心循环
    """
    beams = [(0.0, [0])]  # (score, token_list)，0 = <bos>

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_id:
                candidates.append((score, seq))
                continue
            log_probs = log_prob_fn(seq)  # (V,)
            topk = log_probs.topk(beam_width)
            for i in range(beam_width):
                candidates.append((
                    score + topk.values[i].item(),
                    seq + [topk.indices[i].item()]
                ))
        # 保留 top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        # 全部结束则提前退出
        if all(seq[-1] == eos_id for _, seq in beams):
            break

    return max(beams, key=lambda x: x[0])[1]
