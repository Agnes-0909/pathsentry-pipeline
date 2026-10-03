# Qwen2.5-VL 目标检测预标注

本子工程负责道路附近障碍物的候选检测，是[主预标注流程](../README.md)的检测阶段。提示词定义在 [`prompt.py`](src/qwen25_prelabel/prompt.py)，解析器将模型返回的类别、像素坐标框和置信度整理成结构化记录。

## 输入与输出

- 输入：单张道路图像或图像目录；`--sample-size` 控制抽样数量，非正数表示处理全部图像。
- 候选类别：`person`、`vehicle` 和 `other_obstacle`。这些是模型输出类别，不代表人工检查后的最终类别集合。
- 输出：逐图检测记录、抽样清单，并可选生成 YOLO 标签和框叠加预览。独立命令的输出不等同于主流程合并后的多边形标注。

## 使用

依赖见 [`pyproject.toml`](pyproject.toml)。推理需要兼容的 vLLM、Transformers 和 CUDA 环境。安装后查看参数：

```bash
python -m pip install -e 02_prelabel/ps_proj/qwen2_5_7B_prelabel
qwen25-prelabel --help
```

显式传入当前机器的图像、模型和结果路径：

```bash
qwen25-prelabel /path/to/images --model-dir /path/to/model --output-dir /path/to/results --sample-size 200
```

只检查抽样清单时可加 `--dry-run`；`--seed` 固定抽样结果。`--min-confidence`、`--min-box-area` 控制框过滤，`--gpu-memory-utilization` 和 `--max-model-len` 控制推理资源。完整参数见 [`cli.py`](src/qwen25_prelabel/cli.py)。

测试：

```bash
python -m unittest discover -s 02_prelabel/ps_proj/qwen2_5_7B_prelabel/tests -p 'test_*.py'
```
