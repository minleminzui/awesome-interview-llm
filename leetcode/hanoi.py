"""
三柱汉诺塔 —— 输出最少步骤

规则：
  1. 每次只能移动一个盘子
  2. 大盘不能放在小盘上面
  3. 将 n 个盘从 A 移到 C，借助 B

递归思路（3 步）：
  1. 把上面 n-1 个盘从 A → B（借助 C）
  2. 把最大盘从 A → C
  3. 把 n-1 个盘从 B → C（借助 A）

最少步数 = 2^n - 1
"""


def hanoi(n: int, src: str = "A", dst: str = "C", aux: str = "B") -> list[str]:
    moves = []

    def solve(n, src, dst, aux):
        if n == 0:
            return
        solve(n - 1, src, aux, dst)       # 上面 n-1 个: src → aux
        moves.append(f"{src} → {dst}")    # 最大盘: src → dst
        solve(n - 1, aux, dst, src)       # n-1 个: aux → dst

    solve(n, src, dst, aux)
    return moves


if __name__ == "__main__":
    for n in range(1, 5):
        moves = hanoi(n)
        print(f"\nn={n}  最少步数={len(moves)}  (2^{n}-1={2**n - 1})")
        for i, m in enumerate(moves, 1):
            print(f"  第{i}步: {m}")
