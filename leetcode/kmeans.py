"""
K-Means 聚类 手写实现

算法流程：
  1. 随机选 k 个点作为初始中心
  2. E 步（Assign）：每个点分配到最近的中心
  3. M 步（Update）：每个簇重新算中心 = 簇内均值
  4. 重复 2-3 直到中心不再变化或达到最大迭代次数

时间复杂度：O(n * k * d * T)   n=样本数, k=簇数, d=维度, T=迭代次数
"""

import numpy as np


def kmeans(X: np.ndarray, k: int, max_iters: int = 100) -> tuple:
    """
    X: (N, D) 数据矩阵
    k: 簇数
    返回: (labels, centers)
    """
    N, D = X.shape

    # 1. 随机选 k 个样本作为初始中心（不重复）
    idx = np.random.choice(N, k, replace=False)
    centers = X[idx].copy()                          # (k, D)

    for _ in range(max_iters):
        # 2. E 步：计算每个点到每个中心的距离，分配标签
        #    dist[i][j] = ||X[i] - centers[j]||²
        #    展开: ||x-c||² = ||x||² - 2x·c + ||c||²
        xx = np.sum(X ** 2, axis=1, keepdims=True)   # (N, 1)
        cc = np.sum(centers ** 2, axis=1)             # (k,)
        dist = xx - 2 * X @ centers.T + cc            # (N, k)  广播
        labels = np.argmin(dist, axis=1)              # (N,)

        # 3. M 步：更新中心 = 簇内均值
        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)
            else:
                new_centers[j] = X[np.random.randint(N)]  # 空簇随机重置

        # 收敛判断
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels, centers


# ========== 面试极简版（去掉优化，逻辑最清晰）==========
def kmeans_simple(X, k, max_iters=100):
    N = X.shape[0]
    centers = X[np.random.choice(N, k, replace=False)].copy()

    for _ in range(max_iters):
        # E: 每个点找最近中心
        labels = np.array([
            np.argmin([np.sum((x - c) ** 2) for c in centers])
            for x in X
        ])
        # M: 更新中心
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels, centers


# ========== K-Means++ 初始化（面试加分项）==========
def kmeans_pp_init(X: np.ndarray, k: int) -> np.ndarray:
    """
    K-Means++ 初始化：让初始中心尽量分散
    1. 随机选第一个中心
    2. 后续每个中心以 D(x)² 为概率选（离已有中心越远，被选中概率越大）
    """
    N = X.shape[0]
    centers = [X[np.random.randint(N)]]

    for _ in range(1, k):
        # 每个点到最近中心的距离²
        dist = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in X])
        probs = dist / dist.sum()
        next_idx = np.random.choice(N, p=probs)
        centers.append(X[next_idx])

    return np.array(centers)


if __name__ == "__main__":
    np.random.seed(42)
    # 生成 3 簇数据
    X = np.vstack([
        np.random.randn(50, 2) + [0, 0],
        np.random.randn(50, 2) + [5, 5],
        np.random.randn(50, 2) + [10, 0],
    ])
    labels, centers = kmeans(X, k=3)
    print(f"Centers:\n{centers}")
    print(f"Cluster sizes: {[np.sum(labels == i) for i in range(3)]}")
