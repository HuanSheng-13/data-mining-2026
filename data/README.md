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
- 任务1、任务2压缩包均已核验，结构见下文；其他任务仍需在实现数据加载器时核实实际内容，不要根据文件扩展名猜测数据格式。
- `--data-dir` 指向表中对应的任务数据根目录；加载器在此目录内定位实际文件。
- 下载和解压后应核对样本数、训练/测试划分及标签含义。不要为了方便而混合老师已划分的训练集和测试集。

| 任务 | 输入数据 | 传入的目录 |
| --- | --- | --- |
| 任务1：图像聚类 | Cluster.zip | `data/cluster` |
| 任务2：图像异常检测 | Image_Anomaly_Detection.zip | `data/image_anomaly` |
| 任务3：时间序列预测 | weather.csv | `data/weather` |
| 任务4：疾病判断 | thyroid.zip | `data/thyroid` |

## 任务1实际解压结构和核对记录

```text
data/cluster/Cluster/
├── cluster_labels.json
└── dataset/
    ├── 001.png
    └── ...（共600张PNG）
```

cluster_labels.json将图片文件名映射到类别字符串。六类为bottle、cable、leather、pill、tile、transistor，各100张；600张均可解码为RGB，未发现字节内容重复。尺寸1024×1024共300张，800×800、840×840、900×900各100张。来源为上述课程下载链接；每次运行的audit.json记录数据及标签摘要，可校验具体版本。

任务1继续传入 `--data-dir data/cluster` 即可，程序会定位外层Cluster中的dataset。标签只用于数据核验及最终评估。其他任务的实际结构和核对记录待实现时补充。

## 任务2实际解压结构和核对记录

```text
data/image_anomaly/
├── image_anomaly_labels.json
├── hazelnut/
│   ├── train/good/             # 训练实际读取的正常样本
│   ├── train/bad/              # 保留但不参与无监督训练
│   └── test/
└── zipper/
    ├── train/good/
    ├── train/bad/
    └── test/
```

任务2程序以 `image_anomaly_labels.json` 为测试集标签来源，仅在最终评估阶段读取标签。训练阶段严格只读取两个类别的 `train/good`。具体命令、结果与参数见 [`task2_image_anomaly/README.md`](../task2_image_anomaly/README.md)。

## 版本管理

根目录 `.gitignore` 忽略本目录内除本 README 外的所有内容，所以数据目录只在本地存在。克隆仓库后按上述结构新建目录并放置数据即可。不要用 `git add -f` 强制上传数据。
