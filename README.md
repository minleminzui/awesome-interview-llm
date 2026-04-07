# awesome-interview-llm

LLM / 多模态 / Agent 高频面试速查手册。每篇包含原理、手写代码、面试问答。

## 目录

### Transformer 核心

| 文件 | 内容 |
|------|------|
| [01_attention.md](01_attention.md) | MHA / MQA / GQA（含维度 & FLOPs 分析） |
| [02_normalization.md](02_normalization.md) | LayerNorm / RMSNorm / BatchNorm |
| [03_positional_encoding.md](03_positional_encoding.md) | RoPE / ALiBi / Sinusoidal |
| [10_transformer_arch.md](10_transformer_arch.md) | Transformer 架构总览 & SwiGLU & GPT 骨架 |

### 训练 & 微调

| 文件 | 内容 |
|------|------|
| [04_lora.md](04_lora.md) | LoRA / QLoRA 参数高效微调 |
| [07_training_alignment.md](07_training_alignment.md) | SFT / PPO / DPO / GRPO / GSPO |

### 推理优化

| 文件 | 内容 |
|------|------|
| [05_kv_cache.md](05_kv_cache.md) | KV Cache / Prefill vs Decode |
| [06_flash_attention.md](06_flash_attention.md) | Flash Attention 在线 Softmax |
| [08_quantization.md](08_quantization.md) | 量化：GPTQ / AWQ / SmoothQuant |
| [09_inference_optimization.md](09_inference_optimization.md) | Speculative Decoding / Continuous Batching / PagedAttention |
| [16_sampling.md](16_sampling.md) | Temperature / Top-K / Top-P / Repetition Penalty |

### 架构扩展

| 文件 | 内容 |
|------|------|
| [11_moe.md](11_moe.md) | MoE：Router / Expert / Load Balance |
| [15_parallelism.md](15_parallelism.md) | DP / TP / PP / EP / ZeRO |

### 多模态 & Agent & RAG

| 文件 | 内容 |
|------|------|
| [12_vit_vlm.md](12_vit_vlm.md) | ViT / CLIP / LLaVA / 动态分辨率 |
| [13_agent.md](13_agent.md) | ReAct / Tool Calling / Multi-Agent |
| [14_rag.md](14_rag.md) | Embedding / Chunking / Reranking |

### 手写代码题

| 文件 | 内容 |
|------|------|
| [leetcode/beam_search.py](leetcode/beam_search.py) | Beam Search 解码 |
| [leetcode/infonce.py](leetcode/infonce.py) | InfoNCE / CLIP Loss |
| [leetcode/dropout.py](leetcode/dropout.py) | Dropout (Inverted) |
| [leetcode/linear_regression_bp.py](leetcode/linear_regression_bp.py) | 线性回归手动 BP |
| [leetcode/kmeans.py](leetcode/kmeans.py) | K-Means 聚类 |
| [leetcode/auc.py](leetcode/auc.py) | AUC 三种实现 |
| [leetcode/estimate_pi.py](leetcode/estimate_pi.py) | 蒙特卡洛 + 5 种求 π |
| [leetcode/kl_divergence.py](leetcode/kl_divergence.py) | KL 散度 / JS 散度 / 知识蒸馏 |
| [leetcode/hanoi.py](leetcode/hanoi.py) | 汉诺塔 |
