# 预标注流程实现

本目录实现 [阶段 02](../README.md) 的双模型预标注流程。`custom_prelabel.py` 先运行 SAM 3.1 的可行驶区域分割，再运行 Qwen2.5-VL 的障碍物定位，最后按图像编号合并逐帧 JSON。两个推理阶段可单独重跑。

## 代码结构

| 路径 | 作用 |
|---|---|
| [`custom_prelabel.py`](custom_prelabel.py) | 参数、左右目任务编排、分阶段复用与合并 |
| [`sam3_1_prelabel/`](sam3_1_prelabel/) | SAM 3.1 调用、掩码清理和多边形转换 |
| [`qwen2_5_7B_prelabel/`](qwen2_5_7B_prelabel/) | Qwen2.5-VL 推理、检测结果解析 |
| [`tests/`](tests/) | 主流程的单元测试 |

## 标注约定

| 类型 | 标签 | 形状 | 来源 |
|---|---|---|---|
| 可行驶区域 | `drivable_area` | `polygon` | SAM 3.1 |
| 人员 | `person` | `rectangle` | Qwen2.5-VL |
| 车辆及骑行目标 | `vehicle` | `rectangle` | Qwen2.5-VL |

检测提示词还会产生其他障碍物候选，见 [`prompt.py`](qwen2_5_7B_prelabel/src/qwen25_prelabel/prompt.py)。最终采用哪些类别，应以人工检查后的标签集为准。检测框使用图像的绝对像素坐标，区域边界保存为多边形顶点。

## 运行

两个 Python 子工程的依赖分别列在各自的 `pyproject.toml` 中。准备好与模型兼容的 CUDA/PyTorch、SAM 3.1 和 vLLM 环境后，从仓库根目录运行：

```bash
python -m pip install -e 02_prelabel/ps_proj/sam3_1_prelabel
python -m pip install -e 02_prelabel/ps_proj/qwen2_5_7B_prelabel
python 02_prelabel/ps_proj/custom_prelabel.py --help
```

主脚本支持 `--sides left|right|both`、`--left-input`、`--right-input`、`--left-output`、`--right-output`。图像和模型的默认路径来自原运行机器，因此换机器时应显式指定。`--skip-sam` 或 `--skip-qwen` 可复用已完成的阶段；`--overwrite` 会重新写入对应结果，使用前先确认目标路径。

分割阶段默认使用道路、路面相关提示词，并对掩码做闭运算与小区域清理。可通过 `--sam-prompts`、`--sam-close-kernel-size`、`--sam-min-component-area-ratio` 调整。检测阶段的主要资源参数包括 `--qwen-gpu-memory-utilization`、`--qwen-max-model-len` 和 `--qwen-max-image-pixels`；完整定义见 [`build_parser`](custom_prelabel.py)。

## 格式与校验

合并结果的顶层字段为 `version`、`flags`、`shapes`、`imagePath`、`imageData`、`imageHeight` 和 `imageWidth`。`imageData` 为 `null`，读取时需要能够访问 `imagePath` 指向的原图。

当前生成器将 `source`、`confidence` 和可选 `reason` 写在形状的 `flags` 中。LabelMe 6.x 要求 `flags` 为字符串到布尔值的映射，因此直接打开原始合并结果可能报错；用于人工检查的 JSON 需先将这些字段移到形状的 `description`，并将 `flags` 设为 `{}`。文档首页的示例采用了这种兼容格式。

单元测试从本目录运行：

```bash
PYTHONPATH=sam3_1_prelabel/src:qwen2_5_7B_prelabel/src:. \
  python -m unittest tests/test_custom_prelabel.py sam3_1_prelabel/tests/test_geometry.py
```
