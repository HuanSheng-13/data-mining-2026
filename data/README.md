# 数据下载与文件组织

四项必做任务使用[课程数据地址](https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665)中的 `Cluster.zip`、`Image_Anomaly_Detection.zip`、`weather.csv` 和 `thyroid.zip`。

## 拓展二 FinanceQA

当前 FinanceQA 数据包根目录为 `data/financeqa/FinanceQA/`，其下包含 `agent_workspace/`、`evaluation_only/` 和老师提供的说明文件。保留包内结构和原始文件；若本机解压层级不同，修改 Agent 配置的 `data_root` 即可。

`agent_workspace/data/` 中的数据库与年报用于作答，`train/` 用于开发，`test/` 在方法冻结后统一运行。`evaluation_only/` 及评分输出仅用于独立评测，不接入作答 Agent。代码在仓库根目录的 `extension2_finance_agent/` 开发，索引和输出另存，不放入原始数据目录。

完整数据约定、准备方法和 B、C 的开发规范见[拓展二开发说明](../extension2_finance_agent/README.md)。

## 解压方式

以下路径均相对于仓库根目录。先新建四个目标目录，再将压缩包内容直接解压到对应目录，保留包内原始文件名和层级：

| 下载文件 | 存放操作 | 程序使用的数据目录 |
| --- | --- | --- |
| `Cluster.zip` | 将压缩包内容直接解压到 `data/cluster/` | `data/cluster` |
| `Image_Anomaly_Detection.zip` | 将压缩包内容直接解压到 `data/image_anomaly/` | `data/image_anomaly` |
| `weather.csv` | 直接复制到 `data/weather/weather.csv` | `data/weather` |
| `thyroid.zip` | 将压缩包内容直接解压到 `data/thyroid/` | `data/thyroid` |

ZIP内已自带 `Cluster/`、`Image_Anomaly_Detection/` 或 `thyroid/` 文件夹，解压后保留它们即可，无需把其中的文件再移到上一层。解压软件的目标目录请选择表中的路径，不要再额外创建压缩包同名目录。`weather.csv` 是独立CSV文件，不是压缩包。

如将四个下载文件保存到Windows的“下载”目录，可从仓库根目录在PowerShell执行以下命令。若下载到其他位置，仅修改第一行；解压目标应为新建的空目录：

## 解压后的结构

下面与本地数据直接解压后的实际结构一致，四个任务均支持直接读取：

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

## 数据核对

| 任务 | 数据规模与标签 |
| --- | --- |
| 任务1 | 6类、每类100张，共600张图像；`cluster_labels.json` 用于核验和评估 |
| 任务2 | hazelnut、zipper各有200张训练正常图像和50张训练异常图像；测试标签在 `image_anomaly_labels.json` 中 |
| 任务3 | `weather.csv` 有26200行、日期列与21个数值指标 |
| 任务4 | 训练CSV为1839行、6个特征；测试CSV为1933行，并包含 `label` 列 |
