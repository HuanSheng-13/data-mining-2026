# 数据来源、下载与放置

## 来源及下载

使用课程《作业说明_2026.pdf》提供的数据集，不将原始数据上传到 Git 仓库。

课程数据下载地址：<https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665>

请在浏览器打开上述地址，按需要下载 `Cluster.zip`、`Image_Anomaly_Detection.zip`、`weather.csv` 和 `thyroid.zip`。如果访问权限或链接发生变化，以老师提供的最新链接为准。

## 本地放置方式

在仓库根目录按以下方式放置：

```text
data/
├── README.md
├── cluster/          # Cluster.zip 解压内容
├── image_anomaly/    # Image_Anomaly_Detection.zip 解压内容
├── weather/
│   └── weather.csv
└── thyroid/          # thyroid.zip 解压内容
```

- 将各压缩包解压内容放入对应目录，保留包内文件名及目录结构。例如 Cluster 中的 `dataset/` 若为包内目录，则保留为 `data/cluster/dataset/`。
- 不把压缩包本身当作程序输入，不在 Git 中保存下载压缩包。
- 当前尚未检查实际压缩包内容。实现数据加载器时，请补充下面的实际目录树，并在对应任务 README 中明确程序读取的位置；不要根据文件扩展名猜测数据格式。
- `--data-dir` 指向表中对应的任务数据根目录；加载器在此目录内定位实际文件。
- 下载和解压后应核对样本数、训练/测试划分及标签含义。不要为了方便而混合老师已划分的训练集和测试集。

| 任务 | 输入数据 | 传入的目录 |
| --- | --- | --- |
| 任务1：图像聚类 | Cluster.zip | `data/cluster` |
| 任务2：图像异常检测 | Image_Anomaly_Detection.zip | `data/image_anomaly` |
| 任务3：时间序列预测 | weather.csv | `data/weather` |
| 任务4：疾病判断 | thyroid.zip | `data/thyroid` |

## 待补充：实际解压结构和核对记录

首次读取数据后在此补充：实际目录树、文件格式、必要的列名/标签约定、样本数，以及下载或数据版本记录。若代码对目录层级有具体要求，请给出可照做的例子。

## 版本管理

根目录 `.gitignore` 忽略本目录内除本 README 外的所有内容，所以数据目录只在本地存在。克隆仓库后按上述结构新建目录并放置数据即可。不要用 `git add -f` 强制上传数据。
