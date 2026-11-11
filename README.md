# 数据挖掘导论 2026 课程大作业

课程项目包含图像无监督聚类、图像异常检测、时间序列预测和无监督疾病判断四项任务。

代码仓库：[HuanSheng-13/data-mining-2026](https://github.com/HuanSheng-13/data-mining-2026)

| 任务 | 状态 | 说明与结果 |
| --- | --- | --- |
| 任务1：图像无监督聚类 | 已完成 | [复现说明](task1_clustering/README.md) · [实验报告](results/task1/report.md) · [指标汇总](results/task1/summary.csv) |
| 任务2：图像异常检测 | 已完成 | [复现说明](task2_image_anomaly/README.md) · [指标与图表](results/task2/) |
| 任务3：时间序列预测 | 待实现 | [任务说明](task3_forecasting/README.md) |
| 任务4：无监督疾病判断 | 待实现 | [任务说明](task4_thyroid/README.md) |

## 任务1快速复现

环境：Windows 64位、Python 3.11、CPU。在PowerShell中执行：

```powershell
git clone https://github.com/HuanSheng-13/data-mining-2026.git
cd data-mining-2026
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r task1_clustering/requirements-lock-cpu.txt
```

从[课程数据地址](https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665)下载Cluster.zip，将包内Cluster目录放入 `data/cluster/`。实际结构为：

```text
data/cluster/Cluster/
├── cluster_labels.json
└── dataset/                    # 600张PNG图像
```

运行完整实验：

```powershell
.\.venv\Scripts\python.exe task1_clustering/main.py --data-dir data/cluster --output-dir results/task1/reproduced --seed 42 --experiment all --device cpu
```

首次运行自动下载ResNet-18预训练权重。运行结果生成于 `results/task1/reproduced/all/`，可与仓库保存的[summary.csv](results/task1/summary.csv)和[metrics.csv](results/task1/metrics.csv)比较。任务1四个方案的ARI、NMI、ACC均为1.0，完整实验包含16次聚类。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest task1_clustering/tests -q
```

## 项目结构

```text
data-mining-2026/
├── README.md
├── requirements.txt
├── data/README.md              # 各任务的数据下载与放置说明
├── task1_clustering/           # 任务1代码、锁定依赖与测试
├── task2_image_anomaly/        # 任务2代码、复现说明与独立入口
├── task3_forecasting/          # 任务3说明
├── task4_thyroid/              # 任务4说明
├── results/task1/              # 任务1最终报告、指标、配置和图表
├── results/task2/              # 任务2最终指标与图表
└── report/README.md            # 课程报告材料索引
```

仓库保存代码、依赖、复现说明和精简的最终结果。数据从课程链接获取，模型权重由程序下载，虚拟环境和特征缓存保存在本地。各任务分别使用自己的数据与结果目录，独立运行。

## 课程提交内容

本仓库提供供检查与复现的代码和实验结果。各任务的实验报告及小组实际分工在完成后统一整理为课程要求的PDF。任务1报告见[results/task1/report.md](results/task1/report.md)；其余任务完成后补充对应结果与报告。
