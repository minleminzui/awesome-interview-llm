"""
蒙特卡洛法估算圆周率 π

原理：
  正方形边长 2r，内切圆半径 r
  面积比 = 圆 / 正方形 = πr² / (2r)² = π/4
  随机撒点，落在圆内的比例 ≈ π/4  →  π ≈ 4 × (圆内点数 / 总点数)

判定：点 (x, y) 在单位圆内  ⟺  x² + y² ≤ 1
"""

import random


def estimate_pi(n: int = 1_000_000) -> float:
    inside = 0
    for _ in range(n):
        x, y = random.random(), random.random()  # [0, 1) 均匀采样
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


# ---------- numpy 向量化版本（快 ~50x）----------
def estimate_pi_numpy(n: int = 10_000_000) -> float:
    import numpy as np
    pts = np.random.rand(n, 2)
    inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0).sum()
    return 4.0 * inside / n


if __name__ == "__main__":
    for n in [10_000, 100_000, 1_000_000, 10_000_000]:
        pi_est = estimate_pi(n) if n <= 1_000_000 else estimate_pi_numpy(n)
        print(f"n={n:>10,}  π≈{pi_est:.6f}  误差={abs(pi_est - 3.141592653589793):.6f}")
