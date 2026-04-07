# 并行策略：DP / TP / PP / EP

## 1. 总览

| 并行方式 | 切什么 | 通信 | 适用 |
|----------|--------|------|------|
| **DP** (Data) | 数据 | AllReduce 梯度 | 最简单，模型放得下单卡 |
| **TP** (Tensor) | 权重矩阵列/行 | AllReduce 中间激活 | 模型太大放不下单卡 |
| **PP** (Pipeline) | 层 | 前后向传激活/梯度 | 超深模型 |
| **EP** (Expert) | MoE 专家 | All-to-All token dispatch | MoE 模型专用 |

> 实际大模型训练常组合使用：DP × TP × PP（3D 并行）

---

## 2. Data Parallelism (DP)

```
每张 GPU 持有完整模型副本
数据切成 N 份，各 GPU 独立前向+反向
AllReduce 同步梯度 → 更新
```

### 手写代码

```python
import torch
import torch.distributed as dist


def all_reduce_gradients(model):
    """AllReduce：所有 GPU 的梯度取平均"""
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


# DP 训练循环
def dp_train_step(model, dataloader, optimizer, rank, world_size):
    for batch in dataloader:
        # 每个 rank 拿 1/N 的数据
        local_batch = batch[rank::world_size]
        loss = model(local_batch).loss
        loss.backward()
        all_reduce_gradients(model)    # 同步梯度
        optimizer.step()
        optimizer.zero_grad()
```

### ZeRO (DeepSpeed)

| 阶段 | 切分 | 显存节省 |
|------|------|----------|
| ZeRO-1 | 优化器状态 | ~4x |
| ZeRO-2 | + 梯度 | ~8x |
| ZeRO-3 | + 模型参数 | ~N 倍（N = GPU 数） |

```python
# ZeRO-3 核心思想伪代码
def zero3_forward(layer, input, rank, world_size):
    # 只存 1/N 的参数，前向时 AllGather 拼完整权重
    full_weight = all_gather(layer.weight_shard)   # 通信
    output = F.linear(input, full_weight)
    del full_weight  # 用完释放，省显存
    return output
```

---

## 3. Tensor Parallelism (TP)

**核心**：把单层权重矩阵切分到多张 GPU 上。

### 3.1 列并行 (Column Parallel)

```
权重 W: (d, 4d) 按列切成 [W1, W2]，每张 GPU 存一半

GPU 0:  y1 = x @ W1    # (B, S, 2d)
GPU 1:  y2 = x @ W2    # (B, S, 2d)

ReLU 等非线性可以各 GPU 独立做（列切分不影响）
```

### 3.2 行并行 (Row Parallel)

```
权重 W: (4d, d) 按行切成 [W1; W2]

GPU 0:  y1 = x1 @ W1   # x1 是上一步列并行的输出
GPU 1:  y2 = x2 @ W2

y = AllReduce(y1 + y2)  # 需要通信
```

### 3.3 MHA 的 Tensor Parallelism

```
Q/K/V 头天然可切：每张 GPU 负责 h/N 个头
  GPU 0: heads [0, ..., h/N-1]
  GPU 1: heads [h/N, ..., 2h/N-1]
  ...
最后 W_O 做行并行 + AllReduce

每层 Transformer Block 需要 2 次 AllReduce：
  1. Attention 输出的 AllReduce
  2. FFN 输出的 AllReduce
```

### 手写代码

```python
class ColumnParallelLinear(nn.Module):
    """列并行：输出维度切分"""
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0
        self.local_out = out_features // world_size
        self.linear = nn.Linear(in_features, self.local_out, bias=False)
        self.rank = rank

    def forward(self, x):
        return self.linear(x)   # (B, S, local_out)，无需通信


class RowParallelLinear(nn.Module):
    """行并行：输入维度切分，输出需 AllReduce"""
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert in_features % world_size == 0
        self.local_in = in_features // world_size
        self.linear = nn.Linear(self.local_in, out_features, bias=False)

    def forward(self, x):
        local_out = self.linear(x)         # (B, S, out)
        dist.all_reduce(local_out)         # AllReduce 求和
        return local_out


# FFN 的 TP：第一层列并行，第二层行并行
class TPFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, world_size, rank):
        super().__init__()
        self.w1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)    # 列切
        self.w2 = RowParallelLinear(d_ff, d_model, world_size, rank)       # 行切

    def forward(self, x):
        x = F.gelu(self.w1(x))   # 各 GPU 独立激活
        x = self.w2(x)           # AllReduce
        return x
```

---

## 4. Pipeline Parallelism (PP)

**核心**：按层切分，不同 GPU 负责不同层。

```
GPU 0: layers 0-7    GPU 1: layers 8-15
GPU 2: layers 16-23  GPU 3: layers 24-31

问题：朴素实现只有 1 张 GPU 在工作 → bubble 很大
解决：micro-batch 流水线（GPipe / 1F1B）
```

### GPipe vs 1F1B

```
GPipe:  所有 micro-batch 前向完 → 所有反向
        bubble = (P-1)/M（P=stages, M=micro-batches）

1F1B:   前向和反向交替
        bubble 更小，显存更低（不需要同时存所有 micro-batch 的激活）
```

```python
# GPipe 伪代码
def gpipe_forward(stages, micro_batches):
    activations = [[] for _ in stages]
    # 前向：逐 micro-batch 过所有 stage
    for mb in micro_batches:
        x = mb
        for s, stage in enumerate(stages):
            x = stage(x)
            activations[s].append(x)
    # 反向：逆序
    for mb_idx in reversed(range(len(micro_batches))):
        for s in reversed(range(len(stages))):
            activations[s][mb_idx].backward()
```

---

## 5. Expert Parallelism (EP)

MoE 模型专用，不同专家放在不同 GPU 上。

```
Token Dispatch:
  1. Router 决定每个 token 去哪个专家
  2. All-to-All：把 token 发到对应专家所在的 GPU
  3. 各 GPU 计算本地专家
  4. All-to-All：把结果发回来

通信量 = tokens × hidden_dim × 2（一来一回）
```

```python
def expert_parallel_forward(tokens, router, experts, rank, world_size):
    """
    experts: 本 GPU 上的专家（总共 N 个专家，每 GPU 放 N/world_size 个）
    """
    # 1. 路由
    scores, indices = router(tokens)                    # indices: (B*S, K) 专家 id

    # 2. All-to-All dispatch：按专家 id 发到对应 GPU
    send_counts = compute_send_counts(indices, world_size)
    recv_tokens = all_to_all(tokens, send_counts)

    # 3. 本地计算
    local_output = torch.zeros_like(recv_tokens)
    for local_expert_id, expert in enumerate(experts):
        global_id = rank * len(experts) + local_expert_id
        mask = (recv_expert_ids == global_id)
        if mask.any():
            local_output[mask] = expert(recv_tokens[mask])

    # 4. All-to-All combine：把结果发回来
    output = all_to_all(local_output, recv_counts)
    return output
```

---

## 6. 面试高频问

**Q: TP 和 PP 的区别？什么时候用哪个？**
> TP 切权重矩阵（层内并行），通信频繁但延迟低，适合节点内高带宽（NVLink）。PP 切层（层间并行），通信少但有 pipeline bubble，适合跨节点。通常节点内 TP，跨节点 PP。

**Q: TP 每层需要几次通信？**
> 2 次 AllReduce：Attention 输出 1 次 + FFN 输出 1 次。

**Q: ZeRO-3 和 TP 的区别？**
> ZeRO-3 把参数切分到多 GPU，用时 AllGather 拼回来。TP 是把计算本身分布式做。ZeRO-3 更灵活但通信更多，TP 通信模式固定更高效。

**Q: Pipeline Parallelism 的 bubble 怎么减小？**
> 增加 micro-batch 数量（M 越大 bubble 比例越小）；用 1F1B 调度（前向反向交替，减少峰值显存）。

**Q: MoE 的 EP 通信瓶颈？**
> All-to-All 通信量 ∝ token数 × hidden_dim，且模式不规则（取决于路由）。优化方法：Capacity Factor 限制每专家 token 数、Token Dropping、Hierarchical All-to-All。

---

## 7. 一句话速记

| 并行 | 一句话 |
|------|--------|
| DP | 复制模型，切数据，AllReduce 梯度 |
| ZeRO | DP 基础上切优化器/梯度/参数，按需 AllGather |
| TP | 切权重矩阵（列切+行切），层内并行，每层 2 次 AllReduce |
| PP | 切层，micro-batch 流水线，有 bubble |
| EP | 切 MoE 专家到不同 GPU，All-to-All dispatch/combine |
