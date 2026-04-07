"""
AUC (Area Under ROC Curve) 手写实现

ROC 曲线：
  X 轴 = FPR = FP / (FP + TN)    假阳性率
  Y 轴 = TPR = TP / (TP + FN)    真阳性率（召回率）
  遍历所有阈值，每个阈值得到一个 (FPR, TPR) 点

AUC 的概率意义：
  随机取一个正样本和一个负样本，模型给正样本的打分 > 负样本的概率

三种实现：
  1. 排序 + 梯形面积法（标准做法）
  2. 正负样本对统计法（概率意义的直接计算）
  3. 排序 + 秩次公式（O(NlogN) 最高效）
"""


# ========== 方法 1：排序 + 遍历阈值画 ROC + 梯形面积 ==========
def auc_roc_curve(y_true: list, y_score: list) -> float:
    """
    1. 按预测分数从高到低排序
    2. 依次把每个分数当阈值，算 (FPR, TPR)
    3. 梯形法求面积
    """
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    P = sum(y_true)           # 正样本总数
    N = len(y_true) - P       # 负样本总数

    tp, fp = 0, 0
    points = [(0.0, 0.0)]     # ROC 起点

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / P          # 真阳性率
        fpr = fp / N          # 假阳性率
        points.append((fpr, tpr))

    # 梯形面积法
    auc = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        auc += (x1 - x0) * (y0 + y1) / 2  # 梯形面积

    return auc


# ========== 方法 2：正负样本对计数（概率意义）==========
def auc_pairwise(y_true: list, y_score: list) -> float:
    """
    AUC = P(score_pos > score_neg)
    遍历所有正负样本对，统计正样本分数 > 负样本分数的比例
    时间复杂度 O(P*N)，数据量大时慢
    """
    pos_scores = [s for s, y in zip(y_score, y_true) if y == 1]
    neg_scores = [s for s, y in zip(y_score, y_true) if y == 0]

    count = 0
    total = len(pos_scores) * len(neg_scores)
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                count += 1
            elif ps == ns:
                count += 0.5    # 相等算半个

    return count / total


# ========== 方法 3：秩次公式（面试最优解，O(NlogN)）==========
def auc_rank(y_true: list, y_score: list) -> float:
    """
    公式：AUC = (Σ rank_i - P*(P+1)/2) / (P * N)
    其中 rank_i 是正样本在所有样本中按分数升序排列的秩次（1-based）

    推导：正样本的秩次 = 比它分数低的样本数 + 1
         Σ(比正样本分数低的样本数) = Σ rank_i - P*(P+1)/2
         除以总对数 P*N 就是 AUC
    """
    n = len(y_true)
    P = sum(y_true)
    N = n - P

    # 按分数升序排，分配秩次
    paired = sorted(zip(y_score, y_true))
    rank_sum = 0
    for rank, (score, label) in enumerate(paired, 1):  # rank 从 1 开始
        if label == 1:
            rank_sum += rank

    return (rank_sum - P * (P + 1) / 2) / (P * N)


if __name__ == "__main__":
    y_true = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    y_score = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.35, 0.3, 0.2, 0.1]

    print(f"AUC (ROC curve):  {auc_roc_curve(y_true, y_score):.4f}")
    print(f"AUC (pairwise):   {auc_pairwise(y_true, y_score):.4f}")
    print(f"AUC (rank):       {auc_rank(y_true, y_score):.4f}")

    from sklearn.metrics import roc_auc_score
    print(f"AUC (sklearn):    {roc_auc_score(y_true, y_score):.4f}")
