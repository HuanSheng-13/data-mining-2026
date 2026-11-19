# 数据挖掘导论 2026 课程大作业

完成图像无监督聚类、图像异常检测、时间序列预测和无监督疾病判断四项任务。

## 项目结构

```text
data-mining-2026/
├── README.md                       # 数据、环境与四项任务的运行说明
├── requirements.txt                # 四项任务及测试的统一CPU依赖
├── data/                           # 本地数据，组织方式见下文
│   └── README.md
├── task1_clustering/                # 图像聚类代码与测试
├── task2_image_anomaly/             # 图像异常检测代码与测试
├── task3_forecasting/               # 预测代码、推理入口、工具与pytest测试
├── task4_thyroid/                   # 检测代码、推理入口、工具与pytest测试
├── results/
│   ├── README.md                   # 四项任务的结果文件说明
│   └── task1/ … task4/             # 指标、配置、数据检查和图表
└── report/
    └── task1.md … task4.md         # 四项任务的实验报告
```

## 数据存放方式

从[课程数据地址](https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665)下载以下四个文件。路径以仓库根目录为准；获取仓库的命令见下一节。

| 下载文件 | 存放操作 | 程序使用的数据目录 |
| --- | --- | --- |
| `Cluster.zip` | 将压缩包内容直接解压到 `data/cluster/` | `data/cluster` |
| `Image_Anomaly_Detection.zip` | 将压缩包内容直接解压到 `data/image_anomaly/` | `data/image_anomaly` |
| `weather.csv` | 不需要解压，直接复制到 `data/weather/weather.csv` | `data/weather` |
| `thyroid.zip` | 将压缩包内容直接解压到 `data/thyroid/` | `data/thyroid` |

保留ZIP包内自带的文件夹，不需要解压后手动移动文件。例如，把 `Cluster.zip` 解压到 `data/cluster/` 后，实际图像位于 `data/cluster/Cluster/dataset/`。解压软件应直接以表中目录为目标，不要额外套一层压缩包同名文件夹。

解压后的完整结构为：

```text
data/
├── README.md
├── cluster/
│   └── Cluster/
│       ├── cluster_labels.json
│       └── dataset/                # 600张图像
├── image_anomaly/
│   └── Image_Anomaly_Detection/
│       ├── image_anomaly_labels.json
│       ├── hazelnut/
│       │   ├── train/
│       │   │   ├── good/
│       │   │   └── bad/
│       │   └── test/
│       └── zipper/
│           ├── train/
│           │   ├── good/
│           │   └── bad/
│           └── test/
├── weather/
│   └── weather.csv
└── thyroid/
    └── thyroid/
        ├── train-set.csv
        └── test-set.csv
```

四项任务的命令已对应上述结构。`FinanceQA.zip` 是选做拓展数据，`course-report-template.zip` 是报告模板，运行四项必做任务不需要这两个文件。可复制的PowerShell解压命令见 [data/README.md](data/README.md)。

## 环境配置

以下配置使用Windows 64位、Python 3.11和CPU。若尚未安装Git或Python 3.11，可先在支持WinGet的PowerShell中执行对应安装命令；已安装的项目可以跳过：

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.11 -e --source winget
```

若系统没有 `winget`，可分别通过 [Git for Windows](https://git-scm.com/install/win) 和 [Python官方下载页](https://www.python.org/downloads/windows/)安装Git及64位Python 3.11，安装Python时保留Python Launcher（`py`）。安装完成后重新打开PowerShell，依次执行以下命令获取仓库、创建环境并安装全部依赖：

```powershell
git --version
py -3.11 --version
git clone https://github.com/HuanSheng-13/data-mining-2026.git
cd data-mining-2026
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

根目录 `requirements.txt` 是唯一依赖清单，包含四项任务、测试及其传递依赖的锁定版本；PyTorch和torchvision使用CPU版本，安装一次即可运行全部任务。最后一条命令正常输出 `No broken requirements found.`。如果已经打开本仓库，跳过 `git clone` 和 `cd` 两行，从创建虚拟环境开始执行。后续命令均在仓库根目录运行，并直接使用虚拟环境内的解释器，无需额外执行激活命令。

依赖安装需要联网。任务1、2的默认方案首次运行还会下载约45MB的官方ResNet-18权重；已有权重时可用 `--weights-path` 指定本地文件。任务3、4在依赖与数据就位后可离线运行。四项任务可以分别运行，不依赖其他任务先生成结果。

<a id="task1"></a>

## 任务1：图像无监督聚类

### 运行

```powershell
.\.venv\Scripts\python.exe task1_clustering/main.py --data-dir data/cluster --output-dir results/task1/reproduced --seed 42 --experiment all --device cpu
```

完整实验运行四个方案，共16次聚类，输出位于 `results/task1/reproduced/all/`。

### 方法

主方案使用冻结的ResNet-18提取图像特征，经L2归一化和PCA保留95%累计解释方差，再用K-Means分成6簇。对照包括HSV＋LBP传统特征、不降维的深度特征及Ward层次聚类。

### 结果

四个方案的ARI、NMI和ACC均为1.0；主方案PCA保留29维特征。数据中的类别差异清晰，传统颜色与纹理特征也能完成分组。

[实验报告](report/task1.md) · [结果文件说明](results/README.md#task1)

<a id="task2"></a>

## 任务2：图像异常检测

### 运行

```powershell
.\.venv\Scripts\python.exe task2_image_anomaly/main.py --data-dir data/image_anomaly --output-dir results/task2/reproduced --experiment all --epochs 10 --device cpu --seed 42 --save-heatmaps
```

输出位于 `results/task2/reproduced/`，包含AE基线、记忆库主方案和融合分析。本次保存的AE对照为10轮训练，因此命令显式指定 `--epochs 10`；若只运行主方案，将 `--experiment all` 改为 `--experiment patchcore`。

### 方法

冻结ResNet-18，提取多尺度局部特征并构建正常patch记忆库，取最近邻距离最大的3个patch的均值作为异常分数。每类180张正常图像用于拟合、20张用于校准，阈值由正常校准分数的99%分位数确定；训练异常图像不参与拟合。

### 结果

hazelnut、zipper的主方案AUROC分别为1.0000、0.94375，异常检出率分别为100%、86.67%。10轮AE重建基线较弱，融合未进一步改善主方案；融合权重使用测试标签选择，只作为探索分析。

[实验报告](report/task2.md) · [结果文件说明](results/README.md#task2)

AE的跨环境小幅差异与主方案应分开核对。新增[复现与差异定位说明](task2_image_anomaly/REPRODUCIBILITY.md)，提供固定正常划分的多种子检查命令，并自动保存逐图预测、混淆计数、训练损失、数据指纹和环境版本；原Windows参考结果保留，新增补充实验单列于 `results/task2/supplement/`。

<a id="task3"></a>

## 任务3：时间序列预测

### 运行

```powershell
.\.venv\Scripts\python.exe task3_forecasting/main.py --data-dir data/weather --output-dir results/task3/reproduced --seed 2026
```

输出位于 `results/task3/reproduced/`。如需使用保存的HistGB模型预测新数据：

```powershell
.\.venv\Scripts\python.exe task3_forecasting/predict.py --model results/task3/reproduced/HistGB.joblib --input your_recent_weather.csv --output results/task3/reproduced/inference.csv
```

新CSV使用原数据的列名，包含 `date` 和全部21个指标，时间戳唯一且递增，至少包含12个连续十分钟观测。

### 方法

以过去12个时间点的全部21个指标构造252维输入，预测下一时刻CO₂浓度。按时间划分训练、验证和测试，预处理仅使用拟合数据；比较持续值基线、Ridge与HistGB，并按验证集RMSE选模型。两个学习模型预测浓度变化量后加回上一时刻浓度。

### 结果

选定HistGB，测试MAE为1.7590 ppm、RMSE为3.2104 ppm、R²为0.9799，RMSE比持续值基线降低5.34%。

[实验报告](report/task3.md) · [结果文件说明](results/README.md#task3)

<a id="task4"></a>

## 任务4：无监督疾病判断

### 运行

```powershell
.\.venv\Scripts\python.exe task4_thyroid/main.py --data-dir data/thyroid --output-dir results/task4/reproduced --seed 2026 --alpha 0.05
```

输出位于 `results/task4/reproduced/`。如需使用保存的LOF模型判断新样本：

```powershell
.\.venv\Scripts\python.exe task4_thyroid/predict.py --model results/task4/reproduced/LOF.joblib --input your_thyroid_features.csv --output results/task4/reproduced/inference.csv
```

新CSV保留 `feature_1` 至 `feature_6` 六列即可。

### 方法

仅用正常样本学习分布，以LOF新颖性检测为主方法，固定比较孤立森林和One-Class SVM。相同特征向量按组划分为拟合与校准子集；标准化器和模型只在拟合子集学习，阈值仅使用正常校准分数。

### 结果

LOF测试ROC-AUC为0.9628，检出94个患病样本中的80个，误报117个正常样本，召回率85.11%、精确率40.61%。孤立森林在固定对照中指标更高，但主方法仍按预先设定为LOF。

[实验报告](report/task4.md) · [结果文件说明](results/README.md#task4)

## 验证检查

四项任务的测试位于各自的 `tests/` 目录。准备好数据并完成上述四项实验后，可执行：

```powershell
.\.venv\Scripts\python.exe -m pytest task1_clustering/tests task2_image_anomaly/tests task3_forecasting/tests task4_thyroid/tests -q
```
