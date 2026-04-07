"""
Dropout 手写实现

原理：
  训练时：以概率 p 随机将神经元输出置 0，剩余值除以 (1-p) 做缩放（Inverted Dropout）
  推理时：不做任何操作，直接返回

为什么除以 (1-p)？
  训练时期望值 E[output] = (1-p) * x，除以 (1-p) 使训练和推理的期望一致
  这样推理时不需要额外缩放（Inverted Dropout 的好处）

作用：
  - 防止过拟合（随机丢弃 → 隐式集成多个子网络）
  - 减少神经元共适应（co-adaptation）
"""

import torch
import torch.nn as nn


# ========== NumPy 版本 ==========
def dropout_numpy(x, p=0.5, training=True):
    """
    x: 输入数组
    p: 丢弃概率（不是保留概率）
    """
    import numpy as np
    if not training or p == 0:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(np.float32)  # 1=保留, 0=丢弃
    return x * mask / (1 - p)


# ========== PyTorch 版本 ==========
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).float()  # 伯努利采样
        return x * mask / (1 - self.p)                 # inverted scaling


# ========== 函数版（面试最简）==========
def dropout_fn(x, p=0.5, training=True):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.ones(2, 10)

    print("Training mode:")
    for i in range(3):
        out = dropout_fn(x, p=0.5, training=True)
        print(f"  run {i+1}: {out[0].tolist()}")

    print(f"\nInference mode:")
    out = dropout_fn(x, p=0.5, training=False)
    print(f"  {out[0].tolist()}")

    print(f"\n验证期望一致：")
    results = torch.stack([dropout_fn(x, p=0.3, training=True) for _ in range(10000)])
    print(f"  输入均值: {x.mean():.4f}")
    print(f"  Dropout 后均值 (10000次): {results.mean():.4f}  ← 应接近 1.0")
