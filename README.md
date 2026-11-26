# 数据挖掘导论 2026 课程大作业

完成图像无监督聚类、图像异常检测、时间序列预测和无监督疾病判断四项必做任务，以及选做拓展二：企业财务问答Agent。

拓展二已完成全部37题train和首次正式19题test，官方支持率分别为94.59%和94.74%。方法与运行见[Agent开发说明](extension2_finance_agent/README.md)，分析见[初版报告](report/extension2.md)，预测、工具日志与评分索引见[结果说明](results/README.md#extension2)。

## 项目结构

```text
data-mining-2026/
├── README.md                       # 数据、环境与各任务的运行说明
├── requirements.txt                # 四项任务及测试的统一CPU依赖
├── data/                           # 本地数据，组织方式见下文
│   └── README.md
├── task1_clustering/                # 图像聚类代码与测试
├── task2_image_anomaly/             # 图像异常检测代码与测试
├── task3_forecasting/               # 预测代码、推理入口、工具与pytest测试
├── task4_thyroid/                   # 检测代码、推理入口、工具与pytest测试
├── extension2_finance_agent/        # 财务问答Agent、配置、Skill与测试
├── results/
│   ├── README.md                   # 各任务的结果文件说明
│   ├── task1/ … task4/             # 指标、配置、数据检查和图表
│   └── extension2/                 # train/test预测、工具日志、评分与冻结证据
└── report/
    ├── task1.md … task4.md         # 四项必做任务的实验报告
    └── extension2.md               # 财务问答Agent初版报告
```

## 数据存放方式

从[课程数据地址](https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665)获取数据，按下表存放。前四项为必做任务，FinanceQA为拓展二数据。路径以仓库根目录为准；获取仓库的命令见下一节。

| 下载文件 | 存放操作 | 程序使用的数据目录 |
| --- | --- | --- |
| `Cluster.zip` | 将压缩包内容直接解压到 `data/cluster/` | `data/cluster` |
| `Image_Anomaly_Detection.zip` | 将压缩包内容直接解压到 `data/image_anomaly/` | `data/image_anomaly` |
| `weather.csv` | 不需要解压，直接复制到 `data/weather/weather.csv` | `data/weather` |
| `thyroid.zip` | 将压缩包内容直接解压到 `data/thyroid/` | `data/thyroid` |
| `FinanceQA.zip` | 将包内 `FinanceQA/` 放到 `data/financeqa/`，保留内部结构 | `data/financeqa/FinanceQA` |

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
├── thyroid/
│   └── thyroid/
│       ├── train-set.csv
│       └── test-set.csv
└── financeqa/
    └── FinanceQA/
        ├── agent_workspace/        # 数据库、年报、问题与固定协议
        ├── evaluation_only/        # 独立检查器和参考材料，不接入作答Agent
        ├── README.md
        └── TASK.md
```

运行四项必做任务不需要FinanceQA数据或API密钥。`course-report-template.zip` 是报告模板。可复制的PowerShell解压命令见 [data/README.md](data/README.md)。原始数据不提交Git，派生索引与新运行输出另存。

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

根目录 `requirements.txt` 包含四项任务、测试及其传递依赖的锁定版本；PyTorch和torchvision使用CPU版本。
拓展二运行仅使用Python标准库，测试复用其中的pytest，无需另装Agent框架。
最后一条命令正常输出 `No broken requirements found.`。

依赖安装需要联网。任务1、2的默认方案首次运行还会下载约45MB的官方ResNet-18权重；已有权重时可用 `--weights-path` 指定本地文件。任务3、4在依赖与数据就位后可离线运行。

拓展二作答和独立评分需要联网访问模型服务。复制 `extension2_finance_agent/.env.example` 为同目录 `.env`（已有文件不要覆盖），填写 `FINANCEQA_API_KEY` 与 `FINANCEQA_JUDGE_API_KEY`；可使用同一密钥，前者用于作答，后者用于独立检查器。默认服务为DeepSeek，模型为 `deepseek-flash`，检查器地址和模型也在模板中配置。本地 `.env` 不提交Git，密钥不写入日志。

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

<a id="extension2"></a>

## 拓展二：企业财务问答 Agent

### 运行

数据与本地 `.env` 配置完成后，准备索引并运行完整train：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent prepare
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent run --split train --output extension2_finance_agent/outputs/train_reproduce
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/train_reproduce/predictions.json --output extension2_finance_agent/outputs/train_reproduce/score.json
```

所有输出使用新目录，不覆盖已保存结果。确定方法并提交代码、保持工作区干净后，创建新的冻结快照，再自动完成全部19题test并独立评分：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent freeze --output extension2_finance_agent/outputs/frozen_reproduce.json
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent run --split test --frozen extension2_finance_agent/outputs/frozen_reproduce.json --output extension2_finance_agent/outputs/test_reproduce
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent evaluate --split test --predictions extension2_finance_agent/outputs/test_reproduce/predictions.json --output extension2_finance_agent/outputs/test_reproduce/score.json
```

首次正式test的答案、冻结文件和评分已保存在 `results/extension2/test/`，上述命令用于另建复现。每题状态独立，全部作答后才评分；不得用test参考答案或反馈调整方法。文档调整也会改变快照指纹，不能直接复用历史冻结文件。

### 方法

配置为 `extension2_finance_agent/configs/final.json`。DeepSeek自主调用只读SQL、指标发现、年报检索、相邻片段读取和标量计算工具；财务Skill指导期间、单位、口径和来源核查。每题最多16轮、24次工具调用，程序约束工具协议、引用可追溯性与最终回答，记录真实工具调用、错误、用量和耗时。

### 结果

train获支持35/37（94.59%），首次正式test获支持18/19（94.74%）；56题均完成，37项自动化测试通过。自动支持率不代表逐句人工确认的完整正确率，三条矛盾结果及其来源分析原样保留。

[初版实验报告](report/extension2.md) · [结果文件说明](results/README.md#extension2) · [完整开发与运行说明](extension2_finance_agent/README.md)

## 验证检查

四项任务的测试位于各自的 `tests/` 目录。准备好数据并完成上述四项实验后，可执行：

```powershell
.\.venv\Scripts\python.exe -m pytest task1_clustering/tests task2_image_anomaly/tests task3_forecasting/tests task4_thyroid/tests -q
```

拓展二的自动化测试和已交付证据核查不调用付费模型接口：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest extension2_finance_agent/tests -q
.\.venv\Scripts\python.exe -X utf8 -m results.extension2.build_delivery --verify-only
```
