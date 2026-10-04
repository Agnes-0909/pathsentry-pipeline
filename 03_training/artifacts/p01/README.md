# P01 Git 可追踪实验产物

这些文件从被 Git 忽略的 `03_training/runs/` 复制而来，供 [`MODEL_OPTIMIZATION.md`](../../MODEL_OPTIMIZATION.md) 在 GitHub 等仓库页面中直接展示和查看。

| 目录 | 内容 |
|---|---|
| `figures/` | P01 指标对比图和训练曲线 |
| `metrics/` | E0、A1、A2、A3、B1、C1 的指标、训练历史和推理基准 |
| `visuals/` | 各实验固定样本的代表性预测图 |
| `materials/sam3_raw/` | 原始 SAM 素材摘要和 contact sheet |
| `materials/sam3_1_curated/` | 审核版素材清单、摘要和 contact sheet |
| `materials/previews_curated/` | Copy-Paste 原图/增强图预览 |
| `checkpoints.md` | 被忽略的 `.pt` 权重的来源、大小和 SHA256 |

完整训练权重和全部逐帧输出仍保留在本机 `03_training/runs/`。权重单个约 115–129 MB，未复制到 Git；`checkpoints.md` 可用于核对本地文件身份。这里复制的是报告依赖的轻量证据，不改变原始实验目录。
