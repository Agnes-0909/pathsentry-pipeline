# 05 · 模型量化与 RDK X5 转换

本阶段把 03_training 的多任务模型转换为地平线 RDK-X5 可运行的 `bayes-e` `.bin`，并比较未剪枝 B1、结构优化 H2 和 raw-head H2 的量化误差与转换性能。模型共享一个主干，同时保留检测头和分割头。

## 阶段结论

当前推荐的 RDK-X5 部署接口是 H2 raw-head：模型图只输出三个尺度的分类 logits、DFL 回归 logits 和分割 logits；DFL 解码、sigmoid、坐标变换和 NMS 在端侧后处理完成。这样检测图不再包含 DFL 解码，避免了旧版检测输出被拆出第二个子图。

raw-head H2 已通过 X5 `bayes-e` checker 和 `.bin` 编译。20 张真实验证图片的端侧复测结果为：检测输出 cosine `0.999631`，分割输出 cosine `0.998864`，分割 mask IoU `99.914%`。raw-head 主 BPU 子图的工具链静态估算为 `30.0969 ms`，实际应用仍需加上 DFL、NMS、分割阈值化和输入输出调度时间。

通用双输出版本也已保留，用于和训练模型做数值对照；它在模型图内部完成 DFL 解码，因此会产生两个 BPU 子图。

## 目录结构

```text
05_quantization/
├── README.md
├── CONVERSION_REPORT.md                 # 转换、子图和实板精度报告
├── calibration_images/                  # 固定 200 张校准图的软链接，实际数据来自验证集
├── configs/
│   ├── b1_bayese_dual.yaml              # B1 通用双输出配置
│   ├── h2_bayese_dual.yaml              # H2 通用双输出配置
│   └── h2_raw_head_bayese.yaml          # H2 raw-head 配置
├── models/
│   ├── README.md
│   ├── manifest.json                    # 来源、大小和 SHA256
│   ├── b1_original/                     # 未剪枝 B1 checkpoint 和 ONNX
│   ├── s1_pruned/                       # S1 剪枝 checkpoint 对照
│   └── h2_best/                         # H2 checkpoint、通用 ONNX 和 raw-head ONNX
├── compiled_b1/
│   └── b1_dual_output_opset11_bayese_960x832.bin
├── compiled_h2/
│   └── h2_dual_output_opset11_bayese_960x832.bin
├── compiled_h2_raw/
│   └── h2_raw_head_opset11_bayese_960x832.bin
└── scripts/
    ├── README.md
    ├── export_opset11.py                # 通用双输出导出
    ├── export_raw_head_opset11.py       # raw-head 导出
    ├── dfl_postprocess.py               # DFL 后处理
    ├── validate_raw_head.py             # ONNX raw-head 数值验证
    ├── rdk_mapper.py                    # Docker 内 PTQ 和 bin 编译
    ├── board_infer_dump.cpp             # X5 实板输出导出
    └── compare_board_outputs.py         # 双输出模型的实板误差统计
```

工具链产生的 `mapper_workspace/`、`hb_perf_result/`、checker 中间文件、量化 ONNX、日志和 Python 缓存已清理。这些文件可由下面的命令重新生成，不作为阶段交付物提交。

## 模型和输出

| 模型 | 输入 | 检测输出 | 分割输出 | 用途 |
| --- | --- | --- | --- | --- |
| B1 dual | `[1,3,832,960]` | `[1,6,16380]` | `[1,1,208,240]` | 未剪枝基线对照 |
| H2 dual | `[1,3,832,960]` | `[1,6,16380]` | `[1,1,208,240]` | 通用 ONNX 和量化对照 |
| H2 raw-head | `[1,3,832,960]` | `cls/bbox: 104×120、52×60、26×30` | `[1,1,208,240]` | 推荐 X5 部署接口 |

raw-head 的 bbox 每个位置有 `4×16` 个 DFL logits。后处理使用 stride `(8,16,32)` 和半像素 anchor，输出格式与 Ultralytics 解码结果一致。

## 转换约束

- 目标架构：`bayes-e`
- 输入尺寸：固定 batch 1、`832×960`
- ONNX opset：11。X5 当前工具链不接受原始 opset 17
- 训练输入：RGB、NCHW、float32
- 运行时输入：NV12；如果应用直接提供 RGB，需要同步修改配置
- 校准：目录保留验证集固定抽取的 200 张图片；当前已验证编译使用其中 50 张，校准文件保持 0–255 原始量程，由 `data_scale=1/255` 缩放一次
- 编译：`O3`、`latency`、8 jobs

## 复现

从仓库根目录执行。镜像为 `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`。

### 导出 raw-head ONNX

```bash
YOLO_PY=/Users/xurongtang/miniconda3/envs/yolo/bin/python

$YOLO_PY 05_quantization/scripts/export_raw_head_opset11.py \
  --checkpoint 05_quantization/models/h2_best/best_fused.pt \
  --output 05_quantization/models/h2_best/h2_raw_head_opset11.onnx
```

### checker 和编译

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace/CV_pipeline \
  -w /workspace/CV_pipeline \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 \
  bash -lc 'hb_mapper checker --model-type onnx --march bayes-e \
    --model 05_quantization/models/h2_best/h2_raw_head_opset11.onnx'

docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace/CV_pipeline \
  -w /workspace/CV_pipeline \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 \
  bash -lc 'python3 05_quantization/scripts/rdk_mapper.py \
    --onnx 05_quantization/models/h2_best/h2_raw_head_opset11.onnx \
    --cal-images 05_quantization/calibration_images \
    --output-dir 05_quantization/compiled_h2_raw \
    --cal-sample-num 50'
```

正式转换建议把 `--cal-sample-num` 提高到 200，并使用覆盖部署场景的分层校准集。

### 验证 DFL

```bash
$YOLO_PY 05_quantization/scripts/validate_raw_head.py \
  --decoded 05_quantization/models/h2_best/h2_dual_output_opset11.onnx \
  --raw 05_quantization/models/h2_best/h2_raw_head_opset11.onnx \
  --images 02_prelabel/labeled_data/labelme_check/yolo_multitask/images/val \
  --limit 20
```

预期结果为平均 cosine 约 `0.999999994`，最大绝对误差约 `1.83e-4`。

## 结果索引

- [转换报告](CONVERSION_REPORT.md)：工具链版本、子图原因、PTQ 相似度、实板损失和 raw-head 验证
- [模型清单](models/README.md)：checkpoint、ONNX 和 `.bin` 来源
- [脚本说明](scripts/README.md)：导出、编译和实板验证脚本
- [官方 RDK Ultralytics 导出 patch](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5_legacy/samples/vision/ultralytics_yolo/x86/export_monkey_patch.py)
- [RDK 工具链 PTQ 流程](https://github.com/D-Robotics/rdk_doc/blob/main/docs/07_Advanced_development/04_toolchain_development/intermediate/ptq_process.md)
