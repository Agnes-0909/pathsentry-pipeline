# 量化输入模型

本目录保存量化前的 checkpoint、固定输入 ONNX 和 raw-head ONNX。文件来源、大小和 SHA256 见 [manifest.json](manifest.json)。工具链生成的中间 ONNX 位于被忽略的 mapper workspace 中，已清理。

| 路径 | 作用 |
| --- | --- |
| `b1_original/best.pt` | 未剪枝 P01-B1 原始 checkpoint，作为基线 |
| `b1_original/b1_dual_output_opset11.onnx` | B1 通用双输出 X5 输入，检测输出已含 DFL 解码 |
| `s1_pruned/finetuned_best.pt` | S1 结构化通道剪枝 checkpoint，作为剪枝对照 |
| `h2_best/finetuned_best.pt` | H2 未融合训练 checkpoint |
| `h2_best/best_fused.pt` | H2 Conv/BN 融合 checkpoint，推荐导出起点 |
| `h2_best/reference_dual_output.onnx` | H2 opset 17 通用参考模型 |
| `h2_best/h2_dual_output_opset11.onnx` | H2 通用双输出 X5 输入 |
| `h2_best/h2_raw_head_opset11.onnx` | H2 raw-head X5 输入，DFL 放在后处理 |

最终部署文件是 `compiled_b1/`、`compiled_h2/` 和 `compiled_h2_raw/` 下的 `.bin`，应以 checker、实板加载和后处理验证结果为准。
