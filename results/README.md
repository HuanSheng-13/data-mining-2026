# 实验结果

各任务的最终结果分别存放于task1、task2、task3和task4目录。

## 任务1

入口为[任务1结果说明](task1/README.md)和[实验报告](task1/report.md)。

提交内容包括四个方案的指标汇总、16次实验的逐次指标、运行环境与数据摘要、各方案种子42的配置和逐图像分组，以及报告使用的图表。完整运行记录和特征缓存在本地保留。

按照[任务1复现说明](../task1_clustering/README.md)运行后，新产物位于 `task1/reproduced/all/`，可与提交的最终指标比较。

## 任务2

入口为[任务2结果说明](task2/README.md)和[实验报告](task2/report.md)。

本目录保存冻结 ResNet-18 特征与正常 patch 记忆库的主方案结果、10轮 AE 重建基线、探索性分数融合，以及指标表、ROC曲线、分数分布和热力图示例。主方案的 hazelnut、zipper 图像级 AUROC 分别为1.0000、0.94375；分类阈值由独立正常校准集确定。融合权重使用测试标签选择，仅作补充分析。

按照[任务2复现说明](../task2_image_anomaly/README.md)运行，建议用 `--output-dir results/task2/reproduced` 保存新产物，与本目录的 `compare/metrics_table.csv` 和各方法的 `metrics.json` 比较。原始数据、权重、逐样本分数和完整日志保留在本地。
