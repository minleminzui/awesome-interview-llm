"""
线性回归 + 手动反向传播（从零实现，不用 autograd）

模型：  y_hat = X @ w + b
损失：  L = (1/2N) Σ (y_hat - y)²     (MSE, 带 1/2 方便求导)

前向：
  z = X @ w + b                         # (N, 1)
  L = (1/2N) * ||z - y||²

反向（链式法则）：
  dL/dz = (1/N) * (z - y)               # (N, 1)
  dL/dw = X^T @ dL/dz                   # (D, 1)   ← 关键：X 转置乘上游梯度
  dL/db = sum(dL/dz)                    # (1,)     ← 广播的反向是 sum

更新：
  w = w - lr * dL/dw
  b = b - lr * dL/db
"""

import numpy as np


class LinearRegressionManualBP:
    def __init__(self, n_features: int):
        self.w = np.random.randn(n_features, 1) * 0.01
        self.b = np.zeros((1,))

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X: (N, D)  →  y_hat: (N, 1)
        return X @ self.w + self.b

    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        # MSE with 1/2 for clean gradient
        N = y.shape[0]
        return 0.5 * np.mean((y_hat - y) ** 2)

    def backward(self, X: np.ndarray, y_hat: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        dz = (y_hat - y) / N          # dL/dz:  (N, 1)
        dw = X.T @ dz                 # dL/dw:  (D, 1) = (D, N) @ (N, 1)
        db = np.sum(dz, axis=0)       # dL/db:  (1,)   广播反向 = sum
        return dw, db

    def step(self, dw: np.ndarray, db: np.ndarray, lr: float = 0.01):
        self.w -= lr * dw
        self.b -= lr * db

    def fit(self, X, y, epochs=100, lr=0.01):
        for epoch in range(epochs):
            y_hat = self.forward(X)
            l = self.loss(y_hat, y)
            dw, db = self.backward(X, y_hat, y)
            self.step(dw, db, lr)
            if (epoch + 1) % 20 == 0:
                print(f"epoch {epoch+1:>4d}  loss={l:.6f}")


# ========== PyTorch 对照版（验证手动 BP 正确性）==========
def linear_regression_pytorch():
    import torch

    torch.manual_seed(42)
    N, D = 100, 3
    X = torch.randn(N, D)
    w_true = torch.tensor([[2.0], [-1.0], [0.5]])
    y = X @ w_true + 0.3 + torch.randn(N, 1) * 0.1

    w = torch.randn(D, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.1

    for epoch in range(100):
        # --- 前向 ---
        y_hat = X @ w + b                            # (N, 1)
        loss = 0.5 * torch.mean((y_hat - y) ** 2)

        # --- 反向（autograd）---
        loss.backward()

        # --- 手动验证梯度 ---
        dz = (y_hat.detach() - y) / N                # (N, 1)
        dw_manual = X.T @ dz                         # (D, 1)
        db_manual = dz.sum(dim=0)                    # (1,)
        assert torch.allclose(w.grad, dw_manual, atol=1e-5)
        assert torch.allclose(b.grad, db_manual, atol=1e-5)

        # --- 更新 ---
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1:>4d}  loss={loss.item():.6f}  w={w.detach().T}")

    print(f"\nw_true = {w_true.T}")
    print(f"w_fit  = {w.detach().T}")


if __name__ == "__main__":
    print("=" * 50)
    print("NumPy 手动 BP")
    print("=" * 50)
    np.random.seed(42)
    N, D = 100, 3
    X = np.random.randn(N, D)
    w_true = np.array([[2.0], [-1.0], [0.5]])
    y = X @ w_true + 0.3 + np.random.randn(N, 1) * 0.1

    model = LinearRegressionManualBP(D)
    model.fit(X, y, epochs=100, lr=0.1)
    print(f"w_true = {w_true.flatten()}")
    print(f"w_fit  = {model.w.flatten()}")
    print(f"b_fit  = {model.b}")

    print("\n" + "=" * 50)
    print("PyTorch 对照（验证梯度一致）")
    print("=" * 50)
    linear_regression_pytorch()
