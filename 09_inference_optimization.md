# 推理优化：Speculative Decoding / Continuous Batching / PagedAttention

## 1. Speculative Decoding（投机解码）

**问题**：LLM 自回归生成是 memory-bound，大模型逐 token 生成很慢。

**思路**：用小模型"猜"，大模型"验"。

```
1. Draft Model（小/快）连续生成 K 个候选 token
2. Target Model（大/准）一次前向验证所有 K 个 token
3. 用 rejection sampling 决定接受哪些 token
4. 接受的 token 越多，加速越大
```

### 手写代码

```python
def speculative_decode(draft_model, target_model, input_ids, K=5):
    """K = 每轮投机生成的 token 数"""
    # 1. Draft model 连续生成 K 个 token
    draft_ids = input_ids.clone()
    draft_probs_list = []
    for _ in range(K):
        logits = draft_model(draft_ids).logits[:, -1]
        p = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(p, 1)
        draft_probs_list.append(p)
        draft_ids = torch.cat([draft_ids, next_id], dim=-1)

    # 2. Target model 一次前向验证全部
    target_logits = target_model(draft_ids).logits
    n_accepted = 0

    for i in range(K):
        pos = input_ids.shape[1] + i - 1
        q = torch.softmax(target_logits[:, pos], dim=-1)  # target 分布
        p = draft_probs_list[i]                             # draft 分布
        token = draft_ids[:, input_ids.shape[1] + i]

        # 3. Rejection sampling: 以 min(1, q/p) 概率接受
        accept_prob = (q[:, token] / p[:, token]).clamp(max=1.0)
        if torch.rand(1) < accept_prob:
            n_accepted += 1
        else:
            break

    # 返回接受的 token + target 模型在拒绝位置的采样
    accepted = draft_ids[:, :input_ids.shape[1] + n_accepted]
    bonus_logits = target_logits[:, input_ids.shape[1] + n_accepted - 1]
    bonus_token = torch.multinomial(torch.softmax(bonus_logits, dim=-1), 1)
    return torch.cat([accepted, bonus_token], dim=-1)
```

> **关键性质**：rejection sampling 保证输出分布与直接用 target model 生成 **完全一致**（无损）。

---

## 2. Continuous Batching（连续批处理）

**问题**：静态 batching 中，最长序列决定整个 batch 的结束时间，短序列 GPU 空转。

**做法**：
```
静态 Batching：等所有请求结束才开始下一批
连续 Batching：某个请求生成完毕后立即插入新请求，GPU 持续满载
```

### 手写代码（调度器核心逻辑）

```python
class ContinuousBatchScheduler:
    def __init__(self, max_batch_size):
        self.max_batch = max_batch_size
        self.running = []   # 正在生成的请求
        self.waiting = []   # 等待队列

    def step(self, model):
        # 移除已完成的请求
        finished = [r for r in self.running if r.is_done()]
        self.running = [r for r in self.running if not r.is_done()]

        # 填充空位
        while len(self.running) < self.max_batch and self.waiting:
            new_req = self.waiting.pop(0)
            new_req.prefill(model)  # prefill 新请求
            self.running.append(new_req)

        if not self.running:
            return finished

        # 所有 running 请求一起做一步 decode
        batch_input = self.collate([r.last_token for r in self.running])
        batch_output = model.decode_step(batch_input)
        for req, token in zip(self.running, batch_output):
            req.append_token(token)

        return finished
```

---

## 3. PagedAttention (vLLM)

**问题**：KV Cache 预分配连续显存 → 碎片化严重，实际利用率低。

**做法**：借鉴 OS 虚拟内存分页。

```
- 将 KV Cache 分成固定大小的 Page（如 16 tokens/page）
- 用 Page Table 映射逻辑位置 → 物理显存块
- 不需要连续显存，按需分配/回收
```

**收益**：
- 显存利用率从 ~20-40% 提升到 **>95%**
- 支持 beam search、parallel sampling 时 KV Cache 共享（copy-on-write）

### 核心数据结构

```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size, n_heads, d_head):
        self.block_size = block_size  # 每个 block 存几个 token
        # 物理 KV 块池
        self.k_pool = torch.zeros(num_blocks, block_size, n_heads, d_head)
        self.v_pool = torch.zeros(num_blocks, block_size, n_heads, d_head)
        self.free_blocks = list(range(num_blocks))

    def allocate_block(self):
        return self.free_blocks.pop()

    def free_block(self, block_id):
        self.free_blocks.append(block_id)

class SequenceKVCache:
    def __init__(self, paged_cache):
        self.paged_cache = paged_cache
        self.page_table = []  # 逻辑页 → 物理块 id
        self.cur_pos = 0

    def append(self, k, v):
        block_offset = self.cur_pos % self.paged_cache.block_size
        if block_offset == 0:  # 需要新 page
            new_block = self.paged_cache.allocate_block()
            self.page_table.append(new_block)
        block_id = self.page_table[-1]
        self.paged_cache.k_pool[block_id, block_offset] = k
        self.paged_cache.v_pool[block_id, block_offset] = v
        self.cur_pos += 1
```

---

## 4. 面试高频问

**Q: Speculative Decoding 为什么能加速且无损？**
> 大模型验证 K 个 token 和生成 1 个 token 的成本接近（都是 1 次前向），但一次可接受多个 token。rejection sampling 保证输出分布不变。

**Q: Continuous Batching 的核心优势？**
> 短序列完成后立即释放资源、插入新请求，避免 GPU 空转，吞吐量提升 2-3x。

**Q: PagedAttention 解决什么问题？**
> KV Cache 不需要连续显存，按 page 动态分配回收，解决碎片化问题，显存利用率从 20-40% 提升到 95%+。

**Q: vLLM 的核心技术栈？**
> PagedAttention + Continuous Batching + Prefix Caching（相同 system prompt 共享 KV Cache）。

---

## 5. 一句话速记

| 概念 | 一句话 |
|------|--------|
| Speculative Decoding | 小模型猜、大模型验，rejection sampling 保证无损 |
| Continuous Batching | 请求完成即替换，GPU 不空转 |
| PagedAttention | KV Cache 分页管理，解决显存碎片化 |
