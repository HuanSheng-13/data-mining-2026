# 任务1：图像无监督聚类

使用ResNet-18图像特征、PCA与K-Means，将课程提供的600张图像分为6簇，并完成传统特征、降维与聚类算法的对比实验。

[实验报告](../results/task1/report.md) · [指标汇总](../results/task1/summary.csv) · [逐次实验指标](../results/task1/metrics.csv)

## 环境配置

环境：Windows 64位、Python 3.11、CPU。以下命令在PowerShell中执行：

```powershell
git clone https://github.com/HuanSheng-13/data-mining-2026.git
cd data-mining-2026
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r task1_clustering/requirements-lock-cpu.txt
```

## 数据准备

从[课程数据地址](https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665)下载 `Cluster.zip`，解压至：

```text
data/cluster/Cluster/
├── cluster_labels.json
└── dataset/
    ├── 001.png
    ├── 002.png
    └── ...（共600张PNG）
```

六类为bottle、cable、leather、pill、tile、transistor，各100张。`cluster_labels.json`保存图片文件名与类别的对应关系。程序读取全部图像进行聚类，类别标签用于数据核验和最终评估。

## 运行与复现

在仓库根目录运行完整实验：

```powershell
.\.venv\Scripts\python.exe task1_clustering/main.py --data-dir data/cluster --output-dir results/task1/reproduced --seed 42 --experiment all --device cpu
```

首次运行自动下载约45MB的[ResNet-18官方权重](https://download.pytorch.org/models/resnet18-f37072fd.pth)。完整实验运行16次聚类，生成的指标和报告位于 `results/task1/reproduced/all/`；仓库保存的实验结果位于 `results/task1/`。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest task1_clustering/tests -q
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--data-dir` | `data/cluster` | 数据根目录 |
| `--output-dir` | `results/task1` | 运行输出目录；上述复现命令指定为reproduced |
| `--seed` | `42` | 起始随机种子 |
| `--experiment` | `main` | main运行主方案一次，all运行四个方案 |
| `--device` | `auto` | auto、cpu或cuda |
| `--batch-size` | `32` | 特征提取批大小 |
| `--weights-path` | 自动下载 | 本地官方权重文件路径 |

所有相对路径以执行命令的当前目录为基准。

## 方法与对照实验

图像转换为RGB，短边双线性缩放至256，中心裁剪为224×224。ResNet-18使用 `IMAGENET1K_V1` 权重及其配套归一化，将分类层替换为恒等映射，提取512维全局平均池化特征。编码器参数冻结。

| 方案 | 特征处理 | 聚类算法 |
| --- | --- | --- |
| A：传统特征 | HSV每通道16个直方图区间，拼接uniform LBP（P=8、R=1）的10维直方图，共58维；StandardScaler后L2归一化 | K-Means |
| B：深度特征 | 512维ResNet-18特征，L2归一化 | K-Means |
| C：主方案 | ResNet-18特征L2归一化，PCA保留95%累计解释方差，再次L2归一化 | K-Means |
| D：算法对照 | 与C相同 | Ward凝聚层次聚类 |

传统特征的各直方图先做L1归一化再拼接，LBP使用8位灰度图。PCA采用完整SVD和原始主成分尺度。K-Means固定6簇，采用k-means++初始化、20次重启、最多300轮Lloyd迭代，收敛阈值为 `1e-4`。Ward使用欧氏距离，聚合为6簇。

A、B、C分别使用随机种子42—46运行五次；D运行一次。评价指标包括ARI、NMI、最优一对一匹配后的ACC，以及轮廓系数和Davies–Bouldin指数。

## 实验结果

四个方案的ARI、NMI、ACC均为1.0；三个K-Means方案在五个种子下的外部指标标准差均为0。主方案PCA输出29维，累计解释方差95.0285%。

传统颜色与纹理特征已足以区分这批图像；PCA主要减少特征维度与后续聚类计算。完整结果、图像观察和方法分析见[实验报告](../results/task1/report.md)。

## 提交文件

```text
results/task1/
├── README.md
├── report.md
├── summary.csv / metrics.csv
├── run.json / audit.json / manifest.csv
├── verification.json
├── experiments/
│   └── 各方案/                  # 种子42的配置、指标、600张图像的分组结果
└── figures/                     # 报告使用的四张图
```

`run.json`记录实验环境、超参数、源码摘要与耗时。`experiments/`保存四个方案的代表运行，`metrics.csv`保存全部16次运行的指标。原始数据、预训练权重、特征缓存和复现运行产物存放在本地。
