# 05-quantization 脚本说明

所有命令从仓库根目录执行。

| 脚本 | 作用 |
| --- | --- |
| `export_opset11.py` | 导出固定输入、双输出、opset 11 ONNX |
| `export_raw_head_opset11.py` | 参考官方 monkey patch，导出三尺度 cls/bbox raw-head 和分割输出 |
| `dfl_postprocess.py` | 按 Ultralytics 的 anchor、stride 和 softmax expectation 解码 DFL |
| `validate_raw_head.py` | 对齐 raw-head DFL 解码与原始解码 ONNX |
| `rdk_mapper.py` | 生成校准数据、执行 `hb_mapper makertbin` 并复制 `.bin` |
| `board_infer_dump.cpp` | RDK-X5 上加载模型，把每个输出保存为 float32 文件 |
| `compare_board_outputs.py` | 统计双输出模型的检测/分割 cosine、MAE、RMSE 和 mask IoU |

`rdk_mapper.py` 使用 `bayes-e`、`O3`、`latency` 和 RGB/NCHW 训练输入，运行时默认 NV12。脚本会把校准图片 resize 到 `960×832`，保存为 RGB、NCHW、float32；配置中的 `data_scale=1/255` 负责唯一一次归一化。

raw-head 的 DFL 后处理示例：

```bash
python3 05_quantization/scripts/validate_raw_head.py \
  --decoded 05_quantization/models/h2_best/h2_dual_output_opset11.onnx \
  --raw 05_quantization/models/h2_best/h2_raw_head_opset11.onnx \
  --images 02_prelabel/labeled_data/labelme_check/yolo_multitask/images/val \
  --limit 20
```
