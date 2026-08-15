# 02 · VLM 预标注（双分支）

两个分支**逻辑独立、二选一落地**：
- **分支 A（大 VLM）**：免训练、上手快、单帧成本高 → 适合冷启动标注 1~3k 帧"种子数据"；
- **分支 B（微调小 VLM）**：需先用种子数据微调、之后单帧成本低 → 适合 5k~50k 帧大规模标注。

推荐路径：**A 产种子 → B 蒸馏放大**，但两者都完整实践并对比评估。

## 分支 A：大 VLM 开放词汇预标注

### 1. 模型选型
| 候选 | 用法 |
|---|---|
| Qwen2.5-VL-72B-Instruct（本地 vLLM 部署或 API） | 主力：grounding 检测（输出 bbox）+ 语义描述 |
| GLM-4.5V / GPT-4o（API） | 对照组 |
| **SAM2** | 分割 mask 生成：VLM 给文本/bbox 提示，SAM2 出精细 mask |

检测用 VLM grounding 直接出框；分割不指望 VLM 直接画 mask（边界质量差），统一走 **VLM 定位 + SAM2 精分割** 的两段式。

### 2. Prompt 工程（核心资产）
- 系统级：注入类别体系全文（含每类 2~3 句视觉描述、易混淆类辨析、可行驶性定义）；
- 输出约束为严格 JSON Schema：
```json
{"objects": [{"class": "cone", "bbox": [x1,y1,x2,y2], "conf": 0.9,
              "seg_prompt": "orange traffic cone on the road"}],
 "drivable_regions": [{"class": "drivable_road", "text_prompt": "flat paved path the robot can pass"}]}
```
- 少样本：附 2~3 个人工标好的示例帧输出；
- 失败处理：JSON 解析失败/越界框自动重试 1 次，仍失败则丢弃该帧标记 `need_manual`。

### 3. 后处理
- bbox 规范化（clamp、去重 NMS、最小尺寸过滤）；
- 分割：bbox/text prompt → SAM2 → mask → 与检测框一致性校验（mask 与框 IoU<0.3 则丢弃）；
- 类别映射回 12+4 类体系，输出 COCO JSON + PNG mask，与人工标注格式完全一致。

### 4. 评价
- 金标准 500 帧上：预标注 vs 人工 = 用"以预标注为预测、人工为真值"算 precision/recall/mAP@0.5、分割 mIoU；
- 达标线：检测 P/R ≥0.85，mIoU ≥0.70 → 即可作为训练初标，人工只做修正而非从零标注；
- 记录单帧成本（GPU 时/调用费）与吞吐，形成成本-质量报告。

## 分支 B：微调小 VLM 做预标注

### 1. 模型与数据
- 底座：**Qwen2.5-VL-3B-Instruct**（LoRA 微调，单卡 24G 可训）；
- 训练数据（两条来源，可混用）：
  1. 分支 A 的种子标注 + 人工修正版（质量最高）；
  2. 开源 grounding 数据映射到自有类别体系（RELLIS-3D mask→bbox 等，做类别蒸馏）；
- 规模：2~5k 帧、多轮对话格式（image + 指令 → 上述 JSON）。

### 2. 微调方案
- LoRA：rank 32，目标 q/k/v/o + MLP，lr 1e-4，3 epoch，bs（按显存梯度累积）；
- 数据增广：颜色抖动、模糊，不做几何翻转破坏坐标语义（或同步变换坐标）；
- 监控：JSON 格式合法率、训练/验证 grounding 精度。

### 3. 推理与放大
- vLLM 部署 3B 模型，批量吞吐标注剩余 5k~50k 帧；
- 不确定帧过滤：conf <阈值 或 自评 logit 低 → 送人工抽检队列。

### 4. 评价
- 与分支 A 同一金标准对比 P/R/mAP/mIoU + 成本/千帧；
- 消融：只用开源蒸馏 vs 混合种子数据，验证"大模型种子"的增益；
- 决策输出：成本-质量散点图，选定生产分支（预期 B 在 ≥5k 帧时单帧成本低于 A 一个量级）。

## 人工抽检与闭环（两分支共用）
- 抽检率 10%~20%，按场景分层抽样；修正结果回流为训练数据；
- 预标注置信度低的帧优先送人工，形成主动学习闭环；
- 所有预标注帧带 `source: vlm_large / vlm_sft` 元数据，训练时可做来源消融。

## 交付物
- [ ] Prompt 模板库 `prelabel/prompts/`
- [ ] 分支 A 流水线 `prelabel/large_vlm/`（含 SAM2 两段式）
- [ ] 分支 B 训练+推理 `prelabel/sft_vlm/`（LoRA 配置、数据转换）
- [ ] 质量评测报告 `reports/prelabel_eval.md`（含分支对比结论）
