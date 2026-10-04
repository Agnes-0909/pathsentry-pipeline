# 04-pruning 脚本说明

所有命令从仓库根目录执行。脚本默认读取 P01-B1：03_training/runs/p01/b1/best.pt，输出写入 04_pruning/。完整模型 checkpoint 使用 Python pickle 保存，只加载本工程生成的可信文件，并由脚本显式设置 weights_only=False。脚本启动时会注册 checkpoint_compat.py，以兼容目录改名前保存的 prune.architectures 权重。

| 脚本 | 作用 | 默认输出 |
| --- | --- | --- |
| run_experiments.py | 独立运行 S1/S2/S3 结构化通道剪枝、BN 校准、微调和测速 | 04_pruning/experiments/ |
| run_architecture_experiments.py | 从 B1 初始化 H1/H2 分割头，冻结检测分支并微调分割头 | 04_pruning/experiments/ |
| sensitivity_scan.py | 对 H2 检测主干/颈部逐层删除通道并记录验证变化 | 04_pruning/sensitivity/ |
| joint_prune_probe.py | 按敏感度结果组合低敏感层做物理剪枝探针 | 由 --output 指定 |
| finetune_joint_probe.py | 恢复训练联合探针并比较质量 | 04_pruning/sensitivity_full/joint_finetuned/ |
| evaluate_checkpoint.py | 评测完整模型 checkpoint 的 val/test 指标 | 标准输出或 --output |
| finalize_best.py | 对 H2 做 Conv/BN 融合、验证、测速、可视化和 ONNX 导出 | 04_pruning/final/ |

推荐执行顺序：先运行 run_experiments.py 和 run_architecture_experiments.py，再运行敏感度探针，最后使用 finalize_best.py 生成部署文件。脚本支持 --help 查看 batch、epoch、workers、device 和输出目录参数。
