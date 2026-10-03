# SAM 3.1 区域分割预标注

本子工程是[主预标注流程](../README.md)的分割阶段：通过文本提示定位可行驶区域，得到掩码，并转换成多边形标注。也可以使用 `sam3-prelabel` 命令独立查看单张或一批图像的分割结果。

## 输入与输出

- 输入：单张图像或图像目录，以及至少一个文本提示词。
- 输出：每张图像的原图、掩码、叠加预览和区域清单。主流程会进一步合并掩码、清理小区域并转换为 `drivable_area` 多边形。
- 独立命令输出的是区域分割检查结果，不是含检测框的最终标注。

## 使用

依赖见 [`pyproject.toml`](pyproject.toml)。准备好与 SAM 3.1 兼容的 PyTorch 环境后，从仓库根目录安装并查看参数：

```bash
python -m pip install -e 02_prelabel/ps_proj/sam3_1_prelabel
sam3-prelabel --help
```

显式传入图像、模型和结果路径：

```bash
sam3-prelabel /path/to/images \
  --prompts "drivable road surface, roadway, pavement" \
  --checkpoint-path /path/to/checkpoint \
  --output-dir /path/to/results
```

可用 `--prompt-file` 从文本文件读取提示词；`--recursive` 扫描子目录；`--confidence-threshold` 和 `--max-regions` 控制候选区域。主流程的掩码后处理参数见 [`custom_prelabel.py`](../custom_prelabel.py)。

测试：

```bash
python -m unittest discover -s 02_prelabel/ps_proj/sam3_1_prelabel/tests -p 'test_*.py'
```
