# 多模态：ViT / CLIP / VLM 架构

## 1. Vision Transformer (ViT)

**核心思想**：把图像切成 patch → 线性投影成 token → 送入标准 Transformer。

```
图像 (3, 224, 224)
  ↓ 切成 14×14 个 patch，每个 patch 16×16×3
  ↓ 线性投影 → (196, d)
  ↓ + [CLS] token + Position Embedding → (197, d)
  ↓ Transformer Encoder × L
  ↓ [CLS] 输出做分类
```

### 手写代码

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2          # 196
        # 用 Conv2d 实现 patch 切分 + 线性投影（等价于 reshape + Linear）
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)  # (3, 16, 16) → d

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)           # (B, d, 14, 14)
        x = x.flatten(2)          # (B, d, 196)
        x = x.transpose(1, 2)    # (B, 196, d)
        return x


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 d_model=768, n_layers=12, n_heads=12, n_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        B = x.shape[0]
        x = self.patch_embed(x)                                 # (B, 196, d)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                          # (B, 197, d)
        x = x + self.pos_embed                                  # + 位置编码
        x = self.encoder(x)                                     # (B, 197, d)
        x = self.norm(x[:, 0])                                  # 取 [CLS] token
        return self.head(x)                                     # (B, n_classes)
```

---

## 2. CLIP (Contrastive Language-Image Pretraining)

**思路**：图像编码器 + 文本编码器，对比学习对齐到同一空间。

```
Image → ViT → image_emb (B, d)
Text  → Transformer → text_emb (B, d)
loss = symmetric_infonce(image_emb, text_emb)
```

### 手写代码

```python
class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_embed=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = nn.Linear(image_encoder.d_model, d_embed, bias=False)
        self.text_proj = nn.Linear(text_encoder.d_model, d_embed, bias=False)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, input_ids):
        img_emb = self.image_proj(self.image_encoder(images))   # (B, d_embed)
        txt_emb = self.text_proj(self.text_encoder(input_ids))  # (B, d_embed)
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        temp = torch.exp(self.log_temp)
        logits = img_emb @ txt_emb.T * temp                    # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
```

---

## 3. VLM 架构：如何把视觉接入 LLM

### 3.1 三种主流方案

| 方案 | 做法 | 代表模型 |
|------|------|----------|
| **线性投影** | vision tokens 经 MLP 投影到 LLM 空间 | LLaVA, MiniCPM-V |
| **Cross Attention** | LLM 层间插入交叉注意力，attend to 视觉 | Flamingo, Qwen-VL |
| **Q-Former** | 用可学习 query 从视觉中提取固定数量 token | BLIP-2, InstructBLIP |

### 3.2 LLaVA 架构（最简单，面试首选）

```
Image → CLIP ViT → visual tokens (N, d_v)
                       ↓
                  MLP Projector → visual tokens (N, d_llm)
                       ↓
     [visual tokens, text tokens] → LLM → response
```

```python
class LLaVAModel(nn.Module):
    def __init__(self, vision_encoder, llm, d_vision, d_llm):
        super().__init__()
        self.vision_encoder = vision_encoder   # 冻结的 CLIP ViT
        self.projector = nn.Sequential(        # 2 层 MLP
            nn.Linear(d_vision, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )
        self.llm = llm

    def forward(self, images, input_ids, labels=None):
        # 1. 提取视觉 token
        with torch.no_grad():
            vis_tokens = self.vision_encoder(images)     # (B, N_img, d_vision)
        vis_tokens = self.projector(vis_tokens)          # (B, N_img, d_llm)

        # 2. 拼接文本 embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, S_text, d_llm)
        # 在 <image> 占位符位置插入 visual tokens
        combined = torch.cat([vis_tokens, text_embeds], dim=1)    # (B, N_img+S_text, d_llm)

        # 3. 送入 LLM
        outputs = self.llm(inputs_embeds=combined, labels=labels)
        return outputs
```

### 3.3 Cross Attention 方案（Flamingo 风格）

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # 初始化为 0，逐渐打开

    def forward(self, text_hidden, visual_tokens):
        # text_hidden: (B, S, d)  LLM 的中间表示
        # visual_tokens: (B, N, d) 视觉 token
        attn_out, _ = self.cross_attn(
            query=text_hidden,
            key=visual_tokens,
            value=visual_tokens
        )
        return text_hidden + torch.tanh(self.gate) * attn_out
```

> **Gated**：用 tanh(gate) 控制视觉信息的注入强度，gate 初始为 0 → 训练开始时不影响 LLM。

---

## 4. 动态分辨率（面试加分，你做过 MiniCPM-V）

**问题**：固定分辨率会丢信息或浪费计算。

**做法**：
```
1. 保持原始宽高比，resize 到最近的 patch 网格
2. 切成多个 tile（如 448×448），每个 tile 独立过 ViT
3. 加一个全局缩略图 tile 保留全局信息
4. 所有 tile 的 token 拼接后投影给 LLM
```

```python
def dynamic_preprocess(image, min_tiles=1, max_tiles=6, tile_size=448):
    """将图像切成自适应数量的 tile"""
    w, h = image.size
    aspect = w / h
    # 找最优 tile 排列 (rows, cols)
    best_layout = (1, 1)
    min_waste = float('inf')
    for total in range(min_tiles, max_tiles + 1):
        for rows in range(1, total + 1):
            cols = total // rows
            if rows * cols > max_tiles: continue
            canvas_aspect = (cols * tile_size) / (rows * tile_size)
            waste = abs(canvas_aspect - aspect)
            if waste < min_waste:
                min_waste = waste
                best_layout = (rows, cols)

    rows, cols = best_layout
    resized = image.resize((cols * tile_size, rows * tile_size))
    # 切成 rows * cols 个 tile
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tile = resized.crop((c*tile_size, r*tile_size,
                                 (c+1)*tile_size, (r+1)*tile_size))
            tiles.append(tile)
    # 加全局缩略图
    thumbnail = image.resize((tile_size, tile_size))
    tiles.append(thumbnail)
    return tiles  # 每个 tile 独立过 ViT
```

---

## 5. 面试高频问

**Q: ViT 的 patch embedding 本质是什么？**
> 用 kernel_size=stride=patch_size 的 Conv2d 实现，等价于把每个 patch 展平后接一个 Linear。一步完成切分+投影。

**Q: CLIP 的训练目标是什么？**
> 对称的 InfoNCE loss：图文对是正样本，batch 内其他组合是负样本。可学习温度系数。

**Q: LLaVA 和 Flamingo 的架构区别？**
> LLaVA 用 MLP 投影 visual tokens，与 text tokens **拼接**后送入 LLM（简单高效）。Flamingo 在 LLM 层间插入 **Cross Attention** 让文本 attend to 视觉（更灵活但更复杂）。

**Q: 为什么 Flamingo 用 Gated Cross Attention？**
> gate 初始为 0，训练开始时 LLM 行为不变（不破坏预训练权重），随训练逐渐引入视觉信息。

**Q: 动态分辨率的好处？**
> 保持原始宽高比，避免 resize 变形；高分辨率图切更多 tile 保留细节；低分辨率图少切 tile 节省计算。

**Q: VLM 的训练流程？**（结合你的 MiniCPM-V / Pawbie 经验）
> 典型三阶段：① Pretrain 阶段冻结 LLM 和 ViT，只训 projector ② 解冻 LLM 做指令微调 ③ 可选：解冻 ViT 全参训练。数据上需要图文对（pretrain）和指令数据（SFT）。

---

## 6. 一句话速记

| 概念 | 一句话 |
|------|--------|
| ViT | 图像切 patch → Conv2d 投影 → Transformer Encoder |
| CLIP | 图文双塔 + 对比学习对齐，可学习温度 |
| LLaVA | ViT tokens → MLP 投影 → 拼到 text tokens 前面 → LLM |
| Flamingo | LLM 层间插 Gated Cross Attention attend to 视觉 |
| 动态分辨率 | 保持宽高比，切多 tile + 全局缩略图 |
