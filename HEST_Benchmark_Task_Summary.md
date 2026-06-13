# HEST Benchmark 复现任务总结与学习计划

> 导师任务记录 | 预计周期：1–2 周 | 更新日期：2026-06

---

## 一、核心科学问题

**能否从 H&E 组织形态图像中预测对应位置的基因表达谱（Spatial Transcriptomics）？**

在 Visium / Xenium 平台中，苏木精-伊红（H&E）染色图像与空间转录组检测结果天然配准。由于转录组学定义蛋白质表达、进而决定组织形态，形态图像中理应蕴含可预测基因表达的线索，而高质量的 AI 特征提取器有望将这些信号揭示出来。

---

## 二、涉及工具与资源

### 必读文献

| 资源 | 链接 | 重点 |
|---|---|---|
| AI in Pathology 综述（Nature Reviews Bioengineering） | https://www.nature.com/articles/s44222-023-00096-8 | 专注 **Patch Feature Extractor** 一节 |
| HEST-1k 论文（arXiv, NeurIPS'24 Spotlight） | https://arxiv.org/abs/2406.16192 | HEST-Benchmark 协议、Table 1 结果 |
| SEAL 论文（arXiv 2602.14177） | https://arxiv.org/pdf/2602.14177 | 附加任务参考，CONCH-SEAL / UNIv2-SEAL 架构 |

### 核心代码仓库

| 工具 | 链接 | 用途 |
|---|---|---|
| HEST | https://github.com/mahmoodlab/HEST | 配对 H&E + ST 数据集，含处理流程 |
| Trident | https://github.com/mahmoodlab/trident/ | 大规模 WSI 图块特征提取工具包 |
| SEAL | https://github.com/mahmoodlab/SEAL | CONCH-SEAL / UNIv2-SEAL 模型权重与代码 |

---

## 三、技术背景

### Patch Feature Extractor 是什么？

将 H&E 切片中裁出的图块（通常 256×256 像素 @ 20× 倍率）编码为固定维度向量，供下游任务使用。

**三类训练范式：**

1. **ImageNet 预训练**（ResNet 系列）— 基线，迁移能力有限
2. **病理图像自监督预训练**
   - **CONCH**：对比学习 + 视觉-语言对齐，在大规模病理图文对上训练
   - **Virchow2**：基于 DINOv2，在 300 万张 WSI 上预训练，参数量达 ViT-H 级别
3. **ST 分子监督微调（SEAL）**：在上述基础模型之上，额外注入空间转录组局部分子信号，通过参数高效微调（PEFT）增强形态-分子耦合表达能力

### HEST-Benchmark 评测逻辑

```
For each ST sample:
    patches   = crop H&E at each ST spot location (256px @ 20x)
    features  = encoder(patches)              # [N_spots, D_embed]

    # 按官方 train/test split
    model     = RidgeRegression().fit(features_train, expr_train)
    pred      = model.predict(features_test)

    # 对每个 highly variable gene 分别计算，然后取平均
    score     = mean( pearson_corr(pred[:, g], expr_test[:, g])
                      for g in HVGs )
```

评测指标为 **Pearson 相关系数**（越高越好），在各数据集的 highly variable genes（HVGs）上取均值后报告。

---

## 四、主任务

### 目标

复现 HEST arXiv 论文 **Table 1** 中的基准结果（局部子集）。

### 范围限定（全数据集约 2TB，按子集操作）

| 维度 | 选择 |
|---|---|
| 癌症类型 | 推荐：**BRCA**（乳腺癌）、**CRC**（结直肠癌）、**LUAD**（肺腺癌）|
| Patch Encoder | **CONCH**、**Virchow2** |
| 评测协议 | 严格按 HEST-Benchmark 官方配置（`bench_config/`）|

---

## 五、附加任务（Bonus）

将 SEAL 论文中的 **CONCH-SEAL** 或 **UNIv2-SEAL** 集成进 Trident pipeline，重复上述评测并与主任务结果对比。

**核心工程工作：**

```python
# 继承 Trident 的 BaseEncoder，封装 SEAL 模型
from trident.encoders import BaseEncoder
import torch

class CONCHSEALEncoder(BaseEncoder):
    def __init__(self, weights_path: str):
        super().__init__()
        # 加载 SEAL 微调后的 CONCH 权重
        self.model = load_seal_model(weights_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x)   # [B, D]
```

**参考：** https://github.com/mahmoodlab/SEAL

---

## 六、两周学习与执行计划

### 第一周 — 理论夯基 + 环境搭建

| 天 | 任务 | 输出 |
|---|---|---|
| Day 1–2 | 精读 Pathology AI 综述，重点 Patch Extractor 节；了解 CONCH / Virchow2 / DINOv2 架构 | 笔记：各模型预训练策略对比 |
| Day 3–4 | 精读 HEST 论文，理解 spot-patch 对齐、Benchmark 协议、Table 1 指标含义 | 笔记：评测流程图 |
| Day 5–6 | 安装 HEST-Library，下载 BRCA / CRC / LUAD 子集；跑通官方 Tutorial | 本地数据目录结构确认 |
| Day 7 | 安装 Trident，跑通单张切片图块分割 + CONCH 特征提取 Demo | 输出 `.h5` 特征文件 |

### 第二周 — 实验复现 + 结果分析

| 天 | 任务 | 输出 |
|---|---|---|
| Day 8–9 | 批量提取 3 个癌型的 CONCH / Virchow2 特征 | 特征文件集 |
| Day 10–11 | 按 HEST-Benchmark 协议训练岭回归，计算 Pearson 相关系数 | 结果 CSV |
| Day 12 | 与 Table 1 原始结果对比，分析差异来源（权重版本、patch size、基因数） | 对比报告 |
| Day 13–14 *(Bonus)* | 封装 CONCH-SEAL / UNIv2-SEAL encoder，重跑 Benchmark，对比三组结果 | Bonus 结果表 |

---

## 七、常见坑与注意事项

| 问题 | 说明 |
|---|---|
| **数据访问权限** | HEST 部分数据集需在 HuggingFace 申请访问权限，提前申请 |
| **模型权重 Token** | CONCH / Virchow2 / SEAL 权重均需 HuggingFace `hf_token`，在 `~/.cache/huggingface/` 配置 |
| **Patch 分辨率对齐** | 确保提取时使用 256px @ 20× 倍率，与 HEST-Benchmark 配置一致 |
| **基因数量** | 默认评测 top-50 HVGs，改动此参数会显著影响结果 |
| **GPU 显存** | Virchow2（ViT-H）推理建议 ≥ 40GB VRAM；可用 `--batch_size 16` 降低显存占用 |

---

## 八、关键参考结果（Table 1 期望量级）

> 以下为论文中报告的大致水平，供实验对照参考：

| Encoder | BRCA Pearson | CRC Pearson |
|---|---|---|
| ResNet-50 (ImageNet) | ~0.10–0.15 | ~0.10–0.14 |
| CONCH | ~0.20–0.28 | ~0.18–0.25 |
| Virchow2 | ~0.22–0.30 | ~0.20–0.27 |
| SEAL models *(Bonus)* | **预期高于上述** | — |

> 具体数值以原论文 Table 1 为准，此处仅为量级参考。

---

## 九、沟通原则

> 导师原话：*"I am not expecting perfect results, but rather want to see how you think through this and implement it. Feel free to let me know throughout if some parts do not make sense or you need some more understanding of certain topic. I think communication is the key here."*

遇到任何问题——数据下载、模型接口、结果异常——第一时间与导师沟通，不必追求完美，思路清晰、记录完整即是好结果。
