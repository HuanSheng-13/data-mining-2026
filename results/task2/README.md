# 任务2最终实验结果

先阅读[实验报告](report.md)，环境和数据放置方式见[任务2 README](../../task2_image_anomaly/README.md)。

| 文件 | 内容 |
| --- | --- |
| report.md | 问题定义、特征与模型、实验设置、指标、误判分析和复现范围 |
| compare/metrics_table.csv | 两个类别的 AE、记忆库和探索性融合指标，保留4位小数 |
| run.json | 本次运行参数、环境版本、预训练权重摘要和基础提交号 |
| verification.json | 8项测试通过记录、完整流程设置和主方法独立复跑一致性记录 |
| autoencoder/summary.json | 10轮 AE 重建基线的两类指标汇总 |
| autoencoder/各类别/metrics.json | AE基线的精确指标、固定阈值和最终训练重建误差 |
| patchcore/summary.json | 冻结 ResNet-18 特征与正常记忆库主方案的两类指标汇总 |
| patchcore/各类别/metrics.json | 主方案的精确指标、固定阈值、骨干网络和记忆库规模 |
| fusion_summary.json | 融合权重、校准集缩放范围和融合指标；仅作探索分析 |
| compare/roc_compare_各类别.png | AE、主方案和探索性融合的ROC对比图 |
| 各方法/各类别/roc_curve.png、score_dist.png | 单方法ROC曲线及正常/异常分数分布 |
| patchcore/各类别/heatmap_*.png | 每类3张正常图像、3张异常图像的局部距离热力图 |

主方案只使用训练正常样本：每类180张用于构建记忆库，20张用于阈值校准。训练异常样本不参与拟合；测试标签只用于正式方法的最终评估。每类记忆库包含1000个384维patch特征。

| 类别 | AUROC | AP | F1 | 精确率 | 召回率 | 准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hazelnut | 1.0000 | 1.0000 | 0.9677 | 0.9375 | 1.0000 | 0.9818 |
| zipper | 0.94375 | 0.8917 | 0.8667 | 0.8667 | 0.8667 | 0.9149 |

分类阈值是各类别独立正常校准分数的99%分位数，不在测试集上搜索最优阈值。融合权重在测试集上扫描，不能将融合指标作为独立泛化效果。

从仓库根目录复现本次完整设置：

```powershell
.\.venv\Scripts\python.exe task2_image_anomaly/main.py --data-dir data/image_anomaly --output-dir results/task2/reproduced --experiment all --epochs 10 --device cpu --seed 42 --save-heatmaps
```

首次运行会下载约45MB官方ResNet-18权重，也可用 `--weights-path` 指定本地官方文件。新产物直接位于 `results/task2/reproduced/`。主方案无需AE权重；只运行主方案时可改用 `--experiment patchcore`。AE默认训练60轮，本次对照明确采用10轮，比较已保存结果时应保留上述参数。

本次实验生成时使用基础提交 `225dba2` 之后的本地修复，相关代码与本报告一起保存；`run.json` 中的基础提交号不能单独代表完整运行源码。原始数据、`.pt`/`.pth`权重、`scores.npz`、`calibration_scores.npz`和日志不纳入版本管理；需逐样本检查时可在本地运行后读取。已有验证是在当前Windows CPU环境中完成的完整运行和主方法复跑，尚未进行任务1那样的干净克隆全流程复现。
