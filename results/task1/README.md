# 任务1最终实验结果

先阅读[实验报告](report.md)，运行方式见[任务1 README](../../task1_clustering/README.md)。

| 文件 | 内容 |
| --- | --- |
| summary.csv | 四个方案的指标均值、标准差与特征维度 |
| metrics.csv | 16次聚类的逐次指标 |
| run.json | 实验环境、种子、代码摘要、特征变换及耗时 |
| audit.json、manifest.csv | 数据统计、数据指纹与600张图像的内容摘要 |
| verification.json | 自动化测试及独立环境复现记录 |
| experiments/各方案/config.json | 种子42的特征、变换、聚类参数 |
| experiments/各方案/metrics.json | 代表运行的指标及簇大小 |
| experiments/各方案/assignments.csv | 600张图像的簇编号、真实类别和匹配类别 |
| figures/ | 报告使用的对比图、二维投影、列联热力图和代表图像 |

四个方案的ARI、NMI、ACC均为1.0。三个K-Means方案各运行五个随机种子，Ward层次聚类运行一次；各方案的代表输出采用种子42。

复现实验产生于reproduced目录。可用summary.csv、metrics.csv核对整体结果，用assignments.csv核对逐图像分组。源码摘要记录本次实验使用的Python模块内容，实验基础提交号记录运行时的仓库版本。
