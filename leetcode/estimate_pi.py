"""
求圆周率 π 的多种方法（面试高频）

方法一览：
  1. 蒙特卡洛法        ← 随机撒点，O(N)，精度低
  2. Leibniz 级数      ← π/4 = 1 - 1/3 + 1/5 - ...，收敛极慢
  3. Wallis 乘积       ← π/2 = (2/1)(2/3)(4/3)(4/5)...
  4. Buffon 投针       ← 另一种蒙特卡洛
  5. Machin 公式       ← π/4 = 4·arctan(1/5) - arctan(1/239)，收敛快
  6. Bailey-Borwein-Plouffe (BBP) ← 可直接算第 n 位十六进制数
"""

import random
import math


# ========== 1. 蒙特卡洛法（最经典）==========
def pi_monte_carlo(n: int = 1_000_000) -> float:
    """
    在 [0,1)×[0,1) 正方形内随机撒点
    落入四分之一圆内的概率 = π/4
    """
    inside = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


# ========== 2. Leibniz 级数 ==========
def pi_leibniz(n_terms: int = 1_000_000) -> float:
    """
    π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
    = Σ (-1)^k / (2k+1)    k=0,1,2,...

    收敛极慢：100万项才 5 位精度
    """
    s = 0.0
    for k in range(n_terms):
        s += (-1) ** k / (2 * k + 1)
    return 4 * s


# ========== 3. Wallis 乘积 ==========
def pi_wallis(n_terms: int = 100_000) -> float:
    """
    π/2 = Π (4k²) / (4k²-1)    k=1,2,3,...
        = (2·2)/(1·3) × (4·4)/(3·5) × (6·6)/(5·7) × ...
    """
    product = 1.0
    for k in range(1, n_terms + 1):
        product *= (4.0 * k * k) / (4.0 * k * k - 1)
    return 2 * product


# ========== 4. Buffon 投针 ==========
def pi_buffon(n: int = 1_000_000, L: float = 1.0, D: float = 2.0) -> float:
    """
    平行线间距 D，针长 L (L ≤ D)
    随机投针 n 次
    相交概率 P = 2L / (πD)  →  π = 2L·n / (D·交叉次数)
    """
    cross = 0
    for _ in range(n):
        center = random.uniform(0, D / 2)     # 针中心到最近线的距离
        angle = random.uniform(0, math.pi / 2) # 针与线的夹角
        if center <= (L / 2) * math.sin(angle):
            cross += 1
    if cross == 0:
        return float('inf')
    return (2 * L * n) / (D * cross)


# ========== 5. Machin 公式（收敛很快）==========
def pi_machin(n_terms: int = 50) -> float:
    """
    π/4 = 4·arctan(1/5) - arctan(1/239)

    arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...  (泰勒展开)

    50 项就能达到机器精度
    """
    def arctan_taylor(x, n):
        s = 0.0
        for k in range(n):
            s += (-1) ** k * x ** (2 * k + 1) / (2 * k + 1)
        return s

    return 4 * (4 * arctan_taylor(1/5, n_terms) - arctan_taylor(1/239, n_terms))


# ========== 6. BBP 公式 ==========
def pi_bbp(n_terms: int = 20) -> float:
    """
    π = Σ (1/16^k) × (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))

    20 项即可达到 float64 精度极限
    """
    s = 0.0
    for k in range(n_terms):
        s += (1.0 / 16 ** k) * (
            4.0 / (8 * k + 1)
            - 2.0 / (8 * k + 4)
            - 1.0 / (8 * k + 5)
            - 1.0 / (8 * k + 6)
        )
    return s


if __name__ == "__main__":
    PI = math.pi

    methods = [
        ("蒙特卡洛 (N=1M)",    pi_monte_carlo, dict(n=1_000_000)),
        ("Leibniz (100万项)",   pi_leibniz,     dict(n_terms=1_000_000)),
        ("Wallis (10万项)",     pi_wallis,      dict(n_terms=100_000)),
        ("Buffon 投针 (N=1M)",  pi_buffon,      dict(n=1_000_000)),
        ("Machin (50项)",       pi_machin,      dict(n_terms=50)),
        ("BBP (20项)",          pi_bbp,         dict(n_terms=20)),
    ]

    print(f"{'方法':<22s}  {'估算值':>14s}  {'误差':>12s}")
    print("-" * 55)
    for name, fn, kwargs in methods:
        est = fn(**kwargs)
        print(f"{name:<22s}  {est:>14.10f}  {abs(est - PI):>12.2e}")
