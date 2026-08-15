# 05 · 模型量化

## 1. 总体策略

量化分两层，目标形态以**地平线工具链量化为主**（最终上板的是工具链产出的 HBM INT8 模型），通用 PyTorch 量化作为方法实践与对照：

| 层 | 方法 | 用途 |
|---|---|---|
| 主线 | OpenExplorer PTQ（校准 INT8） | 真正部署用的量化产物 |
| 进阶 | QAT（PyTorch 侧，或工具链 QAT） | PTQ 掉点超标的兜底 |
| 对照 | pytorch/aimet PTQ | 方法对比实验 |

## 2. 主线：OpenExplorer 后训练量化（PTQ）

流程：
1. 剪枝模型 → ONNX（opset 17，静态 shape，预处理/后处理拆出图外）；
2. 准备**校准集**：从本地训练集分层抽 200~500 张（必须覆盖逆光/黄昏/草地/室内全场景矩阵，含长尾类出现帧）——校准集分布是 PTQ 精度的第一决定因素；
3. `hb_mapper makertbin`：配置 yaml（输入 layout、mean/std、INT16/INT16 混合精度策略），产出量化模型 + 精度日志；
4. 分析逐层余弦相似度/KLD 报告，对掉点严重的层标记为 INT16 高精度（工具链支持混合精度）。

精度验证：
- 用 `hb_eval`/仿真器跑量化模型在金标准 test 集推理，比对 FP32：mAP 掉点 ≤1 个点、mIoU 掉点 ≤2 个点；
- 单独核对**置信度分布漂移**（INT8 后 conf 阈值需重标定：画 PR 曲线选新阈值，不沿用 FP32 阈值）。

## 3. 进阶：QAT（触发条件式）

仅当 PTQ 掉点超标（mAP >-1 不达标）才启用：
- PyTorch 侧对剪枝模型插伪量化节点（torch.ao / aimet），本地域数据 20~30 epoch，模拟 INT8 前向；
- 训练完成后再走工具链转换（或用工具链自身 QAT 流程）；
- 注意 QAT 与剪枝的顺序固定为 **剪枝 → QAT**，不交叉迭代，控制实验成本。

## 4. 评价

| 验证项 | 达标线 |
|---|---|
| test 集 mAP 相对剪枝模型 | ≥-1 个点 |
| test 集 mIoU | ≥-2 个点 |
| 长尾类单类 AP | wire/pit 相对掉点 ≤3 个点（INT8 对细小目标更敏感，单独监控） |
| 上板冒烟 | X5 实测输出与仿真输出一致（bbox 偏差 <1px 量级） |

## 5. 常见坑与对策
- 检测头 DF Loss 回归分支量化误差大 → 该分支 INT16；
- 输入动态范围异常（逆光过曝帧）导致激活截断 → 校准集必须含极端曝光帧，必要时做直方图截断 percentile 调参（工具链 `run_qsh`/`max_percentile`）；
- 分割 logits 逐通道量化漂移 → mask 上采样/argmax 移到 CPU 后处理（C++），图内只保留到 logits。

## 6. 交付物
- [ ] 校准集构建脚本 `quantization/build_calibset.py`
- [ ] OpenExplorer 配置 `quantization/oe_yaml/`（检测+分割模型）
- [ ] 量化产物 `models/quantized/*.bin` + 逐层精度日志
- [ ] 报告 `reports/quant_eval.md`（PTQ/QAT 对照、阈值重标定记录）
