# 实验结果

本目录仅说明四项任务的结果文件与主要结果，正式实验报告分别位于 [task1.md](../report/task1.md)、[task2.md](../report/task2.md)、[task3.md](../report/task3.md)、[task4.md](../report/task4.md)。下表路径均相对于 `results/`；原始数据、模型权重、逐样本分数及完整日志保留在本地。

<a id="task1"></a>

## 任务1：图像无监督聚类

### 结果文件

| 文件 | 内容 |
| --- | --- |
| `task1/summary.csv`、`task1/metrics.csv` | 四个方案汇总及16次聚类的逐次指标 |
| `task1/run.json`、`task1/verification.json` | 实验设置、环境与验证记录 |
| `task1/audit.json`、`task1/manifest.csv` | 数据核对、指纹及600张图像的摘要 |
| `task1/experiments/` | 代表运行的配置、指标与逐图像分组 |
| `task1/figures/` | 方法对比、二维投影、列联热力图和代表图像 |

### 结果概述

四个方案的ARI、NMI、ACC均为1.0。主方案PCA输出29维；三个K-Means方案在五个种子下的外部指标一致。

<a id="task2"></a>

## 任务2：图像异常检测

### 结果文件

| 文件 | 内容 |
| --- | --- |
| `task2/compare/metrics_table.csv` | 两个类别的AE、记忆库及探索性融合指标 |
| `task2/autoencoder/summary.json`、`task2/patchcore/summary.json` | 两个独立方法的指标汇总 |
| `task2/autoencoder/各类别/metrics.json`、`task2/patchcore/各类别/metrics.json` | 各类别的精确指标与固定阈值 |
| `task2/fusion_summary.json` | 探索性融合权重、缩放范围和指标 |
| `task2/run.json`、`task2/verification.json` | 实验设置、环境与验证记录 |
| `task2/compare/*.png`、`task2/各方法/各类别/*.png` | ROC对比、分数分布与局部距离热力图 |
| `task2/supplement/summary.json`、`metrics.csv` | 固定正常划分的AE多种子统计与独立重训检查 |
| `task2/supplement/evidence/` | 补充实验逐次指标、逐图预测及环境/数据清单 |

### 结果概述

主方案hazelnut、zipper的AUROC分别为1.0000、0.94375，检出15/15、13/15张异常图像，分别误报1张和2张正常图像。AE基线较弱；融合没有提升主方案的AUROC，且使用测试标签选权重，仅作补充分析。

<a id="task3"></a>

## 任务3：时间序列预测

### 结果文件

| 文件 | 内容 |
| --- | --- |
| `task3/metrics.csv` | 三个方法的测试MAE、RMSE、R²及验证集选定标记 |
| `task3/selection.json`、`task3/validation_search.csv` | 模型选择规则、候选参数与验证指标 |
| `task3/data_quality.json`、`task3/environment.json` | 数据检查、时间划分、环境与文件摘要 |
| `task3/forecast_48h.png`、`task3/forecast_48h.pdf` | 测试段前288个合格目标的真实值和预测曲线 |
| `task3/test_rmse.png`、`task3/test_rmse.pdf` | 三个方法的测试RMSE对比 |

### 结果概述

验证集选定HistGB，测试MAE为1.7590 ppm、RMSE为3.2104 ppm、R²为0.9799。RMSE较持续值基线降低5.34%，在本次时间划分中优于基线。

<a id="task4"></a>

## 任务4：无监督疾病判断

### 结果文件

| 文件 | 内容 |
| --- | --- |
| `task4/metrics.csv` | 三个固定方法的指标、阈值和混淆计数 |
| `task4/protocol.json` | 主方法、预设alpha、模型参数及拟合/校准行号 |
| `task4/threshold_sensitivity.csv` | LOF在预设校准水平下的指标 |
| `task4/data_quality.json`、`task4/environment.json` | 数据检查、重复向量、环境与文件摘要 |
| `task4/roc_pr.png`、`task4/roc_pr.pdf` | 三个方法的ROC和PR曲线 |
| `task4/confusion_matrix.png`、`task4/confusion_matrix.pdf` | LOF主方法的混淆矩阵 |

### 结果概述

LOF的ROC-AUC为0.9628、AP为0.7361，检出80/94个患病样本，误报117/1839个正常样本。召回率85.11%、精确率40.61%，排序能力较好但误报偏多；固定对照中的孤立森林表现更好。
