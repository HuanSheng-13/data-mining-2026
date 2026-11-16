# 任务 2：图像异常检测

本任务对课程数据中的 `hazelnut`（榛子）和 `zipper`（拉链）图像进行无监督异常检测：训练阶段只读取正常样本，测试阶段输出每张图像的异常分数和评估指标。

已保存的代码、指标与图像对应同一套设置：ImageNet 预训练 ResNet-18 的多尺度 patch 特征、最近邻记忆库以及 Top-3 patch 距离均值。原始数据、训练权重和测试分数文件不进 Git，避免仓库体积膨胀和泄露课程数据。

## 方法与边界

| 方法 | 训练数据 | 图像级异常分数 | 作用 |
| --- | --- | --- | --- |
| Autoencoder（AE） | 拟合集中的 `train/good` | 像素重建误差的均值 | 对比基线 |
| 预训练 ResNet-18 + 记忆库 | 拟合集中的 `train/good` | patch 到正常记忆库最近邻距离的 Top-3 均值 | 主方法 |
| 分数融合 | 读取前两者的测试分数 | 归一化后的加权和 | 补充分析，不作为泛化性能结论 |

主方法冻结 ResNet-18，提取 layer2/layer3，分别进行 3×3 局部平均池化，再上采样对齐并拼接成 384 维 patch 特征，L2 归一化后用随机候选池与贪心远点采样（Coreset）压缩记忆库。测试时逐 patch 查最近邻距离；局部异常常集中在少量区域，因此取最大的 3 个距离的平均值作为图像级分数。

默认 `--backbone resnet18` 首次从 PyTorch 官方下载约45MB权重，保存到输出目录的 `patchcore/checkpoints/`，也可通过 `--weights-path` 指定本地官方 `resnet18-f37072fd.pth`，无需依赖任务1。`--backbone ae` 保留为不需外部权重的对照；`wideresnet` 是另一个预训练对照。

## 环境与数据

在仓库根目录创建 Python 3.11 或更高版本环境，并安装根目录依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

从课程链接下载并解压 `Image_Anomaly_Detection.zip`，使目录为：

```text
data/image_anomaly/
├── image_anomaly_labels.json
├── hazelnut/
│   ├── train/good/
│   ├── train/bad/            # 不参与训练
│   └── test/
└── zipper/
    ├── train/good/
    ├── train/bad/            # 不参与训练
    └── test/
```

实际数据中每个类别有 200 张训练正常样本。程序以固定随机种子将它们切成 90% 拟合集和 10% 正常校准集；训练异常样本保留在原目录中但不读取。正常校准分数的 99% 分位数是固定判定阈值，测试标签只用于计算 AUROC、AP 及固定阈值下的指标，不参与训练、缩放或阈值学习。

## 运行

完整复现（AE 基线、记忆库、融合分析和对比图）：

```powershell
.\.venv\Scripts\python.exe task2_image_anomaly/main.py --data-dir data/image_anomaly --experiment all --epochs 10 --device cpu --save-heatmaps
```

Mac/Linux 可把解释器换成 `python`，并可将 `--device cpu` 改成 `--device mps` 或 `--device cuda`。只运行某一项：

```powershell
# AE 重建基线
python task2_image_anomaly/main.py --data-dir data/image_anomaly --experiment autoencoder

# 默认预训练 ResNet-18 + 记忆库，不训练 AE
python task2_image_anomaly/main.py --data-dir data/image_anomaly --experiment patchcore
```

常用参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--data-dir` | `data/image_anomaly` | 解压后的数据根目录 |
| `--output-dir` | `results/task2` | 本次运行的输出位置 |
| `--epochs` | `60` | AE 训练轮数 |
| `--batch-size` | `16` | 批大小 |
| `--seed` | `42` | 模型初始化、训练打乱、正常样本划分和 Coreset 采样的随机种子 |
| `--calibration-fraction` | `0.1` | 从训练正常样本中划出的阈值校准比例 |
| `--threshold-quantile` | `0.99` | 正常校准分数的阈值分位数 |
| `--backbone` | `resnet18` | 预训练主方案；`ae`、`wideresnet` 为对照 |
| `--weights-path` | 自动下载 | 本地官方 ResNet-18 权重 |
| `--save-heatmaps` | 关闭 | 保存每类正常/异常样本的局部异常热力图 |

## 缓存与运行约定

`--seed` 同时控制模型初始化、数据打乱、正常样本划分和记忆库采样。AE 权重记录实际拟合集的文件名及内容摘要、类别、预处理、种子、学习率、轮数和批大小。复用缓存时全部核对；旧版权重或配置不匹配时自动重训。子脚本显式指定 `--ae_weights` 且不匹配时会报错，避免悄悄使用其他训练设置。

数据可按上图直接放置，也可保留 `data/image_anomaly/Image_Anomaly_Detection/` 外层目录，加载器自动定位。所有命令从仓库根目录执行，CPU即可运行。

## 已保存结果

仓库中的 `results/task2/` 保存本次提交使用的指标、ROC 曲线、分数分布图、融合对比图和少量热力图示例；权重 `.pt`/`.pth`、逐样本分数 `.npz`、正常校准分数和原始数据均被忽略。`run.json` 记录具体运行参数和环境；本次 AE 对照训练10轮，默认60轮仍可显式用于更长训练。阈值相关指标均使用固定的正常校准阈值，不能与旧的 Youden 阈值结果混用。

| 类别 | AE 重建 AUROC | 记忆库 AUROC | 融合 AUROC |
| --- | ---: | ---: | ---: |
| hazelnut | 0.635 | 1.000 | 1.000 |
| zipper | 0.544 | 0.944 | 0.944 |

主方案在固定正常校准阈值下，hazelnut 的召回率为100%、F1为0.9677；zipper 的召回率为86.67%、F1为0.8667。AE重建基线在10轮设置下明显较弱，保留原始指标作为对照，不将其当作有效主方案。Windows CPU上本次完整运行约7分钟；单独运行主方法不需要AE训练。

融合权重是在测试集上扫描得到，只用于说明两种分数的互补性；正式效果比较应以 AE 和记忆库这两个独立方法为准。完整数值见 [`results/task2/compare/metrics_table.csv`](../results/task2/compare/metrics_table.csv)。

## 目录说明

```text
task2_image_anomaly/
├── main.py                    # 统一入口
├── src/                       # 数据读取、AE、记忆库与评价工具
└── scripts/                   # 各方法和可视化的独立入口
results/task2/                 # 已保存的精简指标与图表
```

提交前至少运行 `python task2_image_anomaly/main.py --help`，并核对 `data/image_anomaly` 的标签 JSON 与目录结构。数据、权重、校准分数和虚拟环境不得加入版本控制。
