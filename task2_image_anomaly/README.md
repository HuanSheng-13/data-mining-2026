# 任务 2：图像异常检测

本任务对课程数据中的 `hazelnut`（榛子）和 `zipper`（拉链）图像进行无监督异常检测：训练阶段只读取正常样本，测试阶段输出每张图像的异常分数和二分类评估指标。

已保存的代码、指标与图像对应同一套设置：自编码器表征的多尺度 patch 特征、最近邻记忆库以及 Top-3 patch 距离均值。原始数据、训练权重和测试分数文件不进 Git，避免仓库体积膨胀和泄露课程数据。

## 方法与边界

| 方法 | 训练数据 | 图像级异常分数 | 作用 |
| --- | --- | --- | --- |
| Autoencoder（AE） | 仅 `train/good` | 像素重建误差的均值 | 对比基线 |
| AE 表征 + 记忆库 | 仅 `train/good` | patch 到正常记忆库最近邻距离的 Top-3 均值 | 主方法 |
| 分数融合 | 读取前两者的测试分数 | 归一化后的加权和 | 补充分析，不作为泛化性能结论 |

主方法将 AE 编码器的两层特征上采样并拼接成 192 维 patch 特征，L2 归一化后用随机候选池与贪心远点采样（Coreset）压缩记忆库。测试时逐 patch 查最近邻距离；局部异常常集中在少量区域，因此取最大的 3 个距离的平均值作为图像级分数。

`--backbone wideresnet` 保留为可选对照：它会首次从 PyTorch 下载 ImageNet 预训练 WideResNet-50 权重。课程结果使用默认的 `ae`，因此在网络受限环境也能复现。

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

实际数据中每个类别的训练正常样本为 200 张，训练异常样本保留在原目录中但不读取；测试标签仅用于最终评估，不参与训练或阈值学习。

## 运行

完整复现（AE 基线、记忆库、融合分析和对比图）：

```powershell
.\.venv\Scripts\python.exe task2_image_anomaly/main.py --data-dir data/image_anomaly --experiment all --device cpu --save-heatmaps
```

Mac/Linux 可把解释器换成 `python`，并可将 `--device cpu` 改成 `--device mps` 或 `--device cuda`。只运行某一项：

```powershell
# AE 重建基线
python task2_image_anomaly/main.py --data-dir data/image_anomaly --experiment autoencoder

# 默认 AE 表征 + 记忆库；若无已训练 AE 权重，会先在正常样本上训练
python task2_image_anomaly/main.py --data-dir data/image_anomaly --experiment patchcore
```

常用参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--data-dir` | `data/image_anomaly` | 解压后的数据根目录 |
| `--output-dir` | `results/task2` | 本次运行的输出位置 |
| `--epochs` | `60` | AE 训练轮数 |
| `--batch-size` | `16` | 批大小 |
| `--backbone` | `ae` | `ae` 不需外部权重；`wideresnet` 需联网下载 |
| `--save-heatmaps` | 关闭 | 保存每类正常/异常样本的局部异常热力图 |

## 已保存结果

仓库中的 `results/task2/` 保存本次提交使用的指标、ROC 曲线、分数分布图、融合对比图和少量热力图示例；权重 `.pt`、逐样本分数 `.npz`、原始数据均被忽略。

| 类别 | AE 重建 AUROC | 记忆库 AUROC | 融合 AUROC |
| --- | ---: | ---: | ---: |
| hazelnut | 0.895 | 0.827 | 0.912 |
| zipper | 0.413 | 0.675 | 0.675 |

融合权重是在测试集上扫描得到，只用于说明两种分数的互补性；正式效果比较应以 AE 和记忆库这两个独立方法为准。完整数值见 [`results/task2/compare/metrics_table.csv`](../results/task2/compare/metrics_table.csv)。

## 目录说明

```text
task2_image_anomaly/
├── main.py                    # 统一入口
├── src/                       # 数据读取、AE、记忆库与评价工具
└── scripts/                   # 各方法和可视化的独立入口
results/task2/                 # 已保存的精简指标与图表
```

提交前至少运行 `python task2_image_anomaly/main.py --help`，并核对 `data/image_anomaly` 的标签 JSON 与目录结构。数据、权重、虚拟环境不得加入版本控制。
