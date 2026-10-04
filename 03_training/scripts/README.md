# YOLO11s 多任务训练

模型使用 YOLO11s 的检测主干、颈部和原生 P3/P4/P5 检测头，类别为 `person` 与 `vehicle`。可行驶区域分割头融合 P2/P3/P4/P5 特征，输出一个二值 logit。检测损失沿用本机 Ultralytics 实现；分割损失为 BCE 与 Dice 的平均值。两个任务共享同一次前向传播。

使用本机 `yolo` 环境运行，已在 PyTorch 2.9.1、Ultralytics 8.3.233 下验证，无需安装额外 Python 包。训练默认从 `/home/huace/下载/yolo11s.pt` 加载本地预训练权重，类别数变为 2 后检测分类层会重新初始化。

```bash
/home/huace/miniconda3/envs/yolo/bin/python scripts/train.py \
  --data dataset/yolo_multitask/dataset.yaml \
  --weights /home/huace/下载/yolo11s.pt \
  --output runs/yolo11s_multitask
```

默认输入尺寸为 `960×832`（宽×高），batch 为 2、训练 100 epoch。可通过 `--width`、`--height`、`--batch`、`--epochs`、`--lr`、`--seg-weight`、`--device` 调整。默认使用 CUDA 混合精度；数值不稳定时可用 `--no-amp` 切换 FP32，训练记录会标明 `amp` 状态。输入宽高必须是 32 的倍数。图像按比例 letterbox；填充区域在分割损失与评测中忽略。图像、框和 mask 同步水平翻转；mask 始终使用最近邻缩放。

每个 epoch 验证一次，`runs/yolo11s_multitask/` 中会产生 `best.pt`、`last.pt` 和 `history.jsonl`。多任务模型按 `(检测 mAP50-95 + 分割 IoU) / 2` 选取最佳权重；`--seg-weight 0` 的纯检测模型只按检测 mAP50-95 选取。中断后使用 `--resume runs/yolo11s_multitask/last.pt` 继续训练，保持相同输入尺寸和训练参数。

验证集与测试集独立评测：

```bash
/home/huace/miniconda3/envs/yolo/bin/python scripts/evaluate.py \
  --checkpoint runs/yolo11s_multitask/best.pt --split val \
  --output runs/yolo11s_multitask/val_metrics.json

/home/huace/miniconda3/envs/yolo/bin/python scripts/evaluate.py \
  --checkpoint runs/yolo11s_multitask/best.pt --split test \
  --output runs/yolo11s_multitask/test_metrics.json
```

评测输出检测 Precision、Recall、mAP50、mAP50-95 和各类别指标，以及可行驶区域 IoU、Dice。检测 AP 使用低置信度阈值 `0.001` 与 NMS IoU `0.7`。正式评测不要设置 `--limit`；该参数只用于冒烟测试。

## 五组实验

`run_experiments.py` 顺序执行 E0 基线（960×832，分割权重 5）、E1 低权重（1）、E2 高权重（10）、E3 低分辨率（640×544，权重 5）和 E4 纯检测对照（权重 0）。默认先各训练 20 epoch，然后从 E0–E3 中选验证集综合分数最高的两组，从预训练权重重新训练 100 epoch。最后只对入选的最佳多任务模型评测测试集。E4 的分割指标输出为 `null`，不参与多任务候选排序。

```bash
/home/huace/miniconda3/envs/yolo/bin/python scripts/run_experiments.py \
  --output-root runs/yolo11s_experiments
```

可以先用 `--dry-run` 查看命令。`--stage pilot` 只跑五组筛选，随后用 `--stage final` 继续最终训练；中断后加 `--resume-incomplete` 续跑，已完成且配置一致的实验会跳过。`--full-det-baseline` 会额外把 E4 也重新训练 100 epoch，供同训练预算下比较检测指标，但不会参与最终多任务模型选取。

实验目录分别保存 `pilot/e0` 到 `pilot/e4`、`final/<实验编号>`，根目录保存 `pilot_summary.json` 和 `summary.json`。最终入选模型目录另有 `val_metrics.json` 和 `test_metrics.json`。筛选阶段的 epoch 与最终阶段不同，最终阶段始终从 YOLO11s 预训练权重重新开始，不从 20-epoch checkpoint 续训。

## P01 小目标检测与 Copy-Paste 实验

优化思路、基线与实验台账见 [`MODEL_OPTIMIZATION.md`](../MODEL_OPTIMIZATION.md)。P01-A1 在 YOLO11s 检测头增加 P2/4；A2 只把输入提高到 1280×1088；A3 组合两项。`--p2-detect` 会将新增检测层与原 YOLO11s 可兼容的预训练权重映射加载。评测脚本从 checkpoint 自动重建对应检测结构。高分辨率 batch 2 若显存不足，可用 `--batch 1 --accumulate 2`，并在实验记录中注明差异。

```bash
YOLO_PY=/home/huace/miniconda3/envs/yolo/bin/python
$YOLO_PY scripts/train.py --p2-detect --width 960 --height 832 --epochs 100 --patience 20 --output runs/p01/a1
$YOLO_PY scripts/train.py --width 1280 --height 1088 --epochs 100 --patience 20 --output runs/p01/a2
$YOLO_PY scripts/train.py --p2-detect --width 1280 --height 1088 --epochs 100 --patience 20 --output runs/p01/a3
```

使用本机 `vllm` 环境和 `prelabel/sam3_1/sam3.1_multiplex.pt` 只从训练集检测框提取目标实例抠图；SAM 3.1 的本机实现要求 CUDA。脚本默认按时间间隔抽样，过滤过宽的车辆框和与原检测框不一致的抠图，并输出素材清单、统计与 contact sheet。素材预览通过后，Copy-Paste 仅在训练集加载时启用，验证和测试仍使用原图。

```bash
VLLM_PY=/home/huace/miniconda3/envs/vllm/bin/python
$VLLM_PY scripts/prepare_copy_paste.py --output runs/p01/materials/sam3_1
$YOLO_PY scripts/curate_copy_paste.py --raw runs/p01/materials/sam3_1 \
  --output runs/p01/materials/sam3_1_curated
$YOLO_PY scripts/preview_copy_paste.py --bank runs/p01/materials/sam3_1_curated
$YOLO_PY scripts/train.py --copy-paste-bank runs/p01/materials/sam3_1_curated --copy-paste-prob 0.3 \
  --width 960 --height 832 --epochs 100 --patience 20 --output runs/p01/b1
```

本轮 A 组按验证集整体 mAP50-95 选择 A2 作为 C1 的结构。C1 从相同 YOLO11s 预训练权重重新训练，命令为：

```bash
$YOLO_PY scripts/train.py --copy-paste-bank runs/p01/materials/sam3_1_curated \
  --copy-paste-prob 0.3 --width 1280 --height 1088 \
  --epochs 100 --patience 20 --output runs/p01/c1
```

五组实验已完成；完整验证/测试结果与素材审核规则见 [`MODEL_OPTIMIZATION.md`](../MODEL_OPTIMIZATION.md)。`runs/p01/<实验编号>/` 下保存最佳权重、训练历史、验证指标、推理基准和固定样本预测图。按验证集整体 mAP50-95，B1 是本轮最佳候选，其测试集评测保存在 `runs/p01/b1/test_metrics.json`。
