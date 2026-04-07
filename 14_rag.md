# RAG：检索增强生成

## 1. RAG 流程

```
Query → Embedding → 向量检索 → Top-K 文档 → Rerank → Prompt 拼接 → LLM → 回答
```

---

## 2. Embedding + 向量检索

### 手写余弦相似度检索

```python
import torch
import torch.nn.functional as F


def cosine_retrieval(query_emb, doc_embs, top_k=5):
    """
    query_emb: (d,) 查询向量
    doc_embs:  (N, d) 文档向量库
    """
    query_emb = F.normalize(query_emb.unsqueeze(0), dim=-1)   # (1, d)
    doc_embs = F.normalize(doc_embs, dim=-1)                   # (N, d)
    scores = (query_emb @ doc_embs.T).squeeze(0)               # (N,)
    topk_scores, topk_indices = scores.topk(top_k)
    return topk_indices, topk_scores
```

### 最大内积搜索（MIPS）

```python
def mips_retrieval(query_emb, doc_embs, top_k=5):
    """不归一化，直接内积"""
    scores = doc_embs @ query_emb   # (N,)
    topk_scores, topk_indices = scores.topk(top_k)
    return topk_indices, topk_scores
```

---

## 3. 文本切分 (Chunking)

```python
def chunk_by_tokens(text: str, tokenizer, chunk_size: int = 512, overlap: int = 64):
    """按 token 数切分，保留重叠"""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def chunk_by_separator(text: str, separators: list[str] = ["\n\n", "\n", "。"],
                       max_chunk_size: int = 500):
    """按语义分隔符递归切分"""
    if len(text) <= max_chunk_size:
        return [text]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                if len(current) + len(part) + len(sep) <= max_chunk_size:
                    current += (sep if current else "") + part
                else:
                    if current: chunks.append(current)
                    current = part
            if current: chunks.append(current)
            return chunks

    # 最后兜底：硬切
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
```

---

## 4. Reranking（二次排序）

```python
class CrossEncoderReranker:
    """
    初筛用 Bi-Encoder（快，embedding 预计算）
    精排用 Cross-Encoder（慢但准，query 和 doc 一起编码）
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[tuple]:
        scores = []
        for doc in documents:
            inputs = self.tokenizer(query, doc, return_tensors="pt",
                                    truncation=True, max_length=512)
            with torch.no_grad():
                score = self.model(**inputs).logits.squeeze().item()
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(documents[i], s) for i, s in ranked[:top_k]]
```

---

## 5. 完整 RAG Pipeline

```python
class RAGPipeline:
    def __init__(self, embedding_model, vector_db, llm, reranker=None):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.llm = llm
        self.reranker = reranker

    def ingest(self, documents: list[str], chunk_size=512):
        """离线索引：切分 → 编码 → 入库"""
        for doc in documents:
            chunks = chunk_by_tokens(doc, self.embedding_model.tokenizer, chunk_size)
            for chunk in chunks:
                emb = self.embedding_model.encode(chunk)
                self.vector_db.insert(emb, {"text": chunk})

    def query(self, question: str, top_k=5, rerank_top_k=3) -> str:
        # 1. 向量检索
        q_emb = self.embedding_model.encode(question)
        candidates = self.vector_db.search(q_emb, top_k)

        # 2. Rerank（可选）
        if self.reranker:
            docs = [c["text"] for c in candidates]
            reranked = self.reranker.rerank(question, docs, rerank_top_k)
            context = "\n\n".join([doc for doc, _ in reranked])
        else:
            context = "\n\n".join([c["text"] for c in candidates[:rerank_top_k]])

        # 3. LLM 生成
        prompt = f"根据以下参考资料回答问题。\n\n参考资料：\n{context}\n\n问题：{question}\n回答："
        return self.llm(prompt)
```

---

## 6. 面试高频问（结合你的 legal-agent 经验）

**Q: Bi-Encoder 和 Cross-Encoder 的区别？**
> Bi-Encoder 分别编码 query 和 doc，doc 可预计算，检索快（ANN）但精度略低。Cross-Encoder 把 query+doc 拼接一起编码，更准但 O(N) 不可预计算。实际用 Bi-Encoder 粗筛 + Cross-Encoder 精排。

**Q: Chunking 的策略？**
> ① 固定长度（按 token 数，简单通用）② 语义分隔符（按段落/句号切，保留完整语义）③ 递归切分（先大粒度再小粒度）。overlap 20~50 token 保证跨 chunk 信息不丢。

**Q: 检索质量差怎么优化？**
> ① Query 改写（让 LLM 扩写/分解 query）② HyDE（先让 LLM 生成假设性回答，用回答做检索）③ 多路召回（关键词 BM25 + 向量语义）④ 微调 embedding 模型 ⑤ Reranking。

**Q: 长文本切片和溯源高亮怎么做？**（结合简历）
> 切片时记录每个 chunk 的 start/end offset。检索到 chunk 后，在原文中定位对应区间高亮。Milvus 中用 metadata 存储 doc_id + offset + group_id。

**Q: 你在 legal-agent 中的 Milvus 优化？**（结合简历）
> ① 用 group_id 支持多知识库隔离 ② 长文本切片保留 overlap ③ 修复线程安全问题（Milvus client 不是线程安全的，改用连接池）④ 批量插入优化。

---

## 7. 一句话速记

| 概念 | 一句话 |
|------|--------|
| RAG | 检索相关文档拼到 prompt 里，让 LLM 基于证据回答 |
| Bi-Encoder | query/doc 分别编码，快但粗 |
| Cross-Encoder | query+doc 拼接编码，慢但准 |
| Chunking | 按 token 数或语义切分，overlap 防信息丢失 |
| HyDE | 先生成假设性回答再检索，提升召回 |
