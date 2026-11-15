# 任务3：时间序列预测

使用最近 12 个连续十分钟观测的全部 21 个指标，预测下一时刻的 CO₂ 浓度（ppm）。数据源于课程发布的 `weather.csv`。放置位置为 `data/weather/weather.csv`，其余下载说明见 [数据说明](../data/README.md)。

从仓库根目录在 Windows PowerShell 执行。任务使用 Python 3.11 和 CPU，依赖版本与任务1锁定环境一致。如果已按任务1的 CPU 锁文件安装依赖，可直接运行最后一条命令。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-task3-task4.txt
.\.venv\Scripts\python.exe task3_forecasting/main.py --data-dir data/weather --output-dir results/task3/reproduced --seed 2026
```

默认数据目录、输出目录和种子与上述命令一致。数据就位且依赖安装完成后，完整实验可离线运行。每次运行会在指定输出目录写入指标、图表、逐点预测和模型。如需保留多次实验，请为每次运行指定不同的输出目录。

程序处理重复时间戳、缺失标记与不连续采样，按时间切分训练、验证、测试，构造 252 维窗口。对比上一时刻浓度基线、岭回归及直方图梯度提升树。两个回归模型拟合浓度变化量，再加回上一时刻浓度。验证集 RMSE 决定正式模型，测试集报告 MAE、RMSE 与 R²。

正式结果和完整方法见 [实验报告](../results/task3/report.md) 与 [验证集选择记录](../results/task3/selection.json)。如需使用已训练模型预测新的连续气象记录，可执行 `python predict.py --task 3 --model results/task3/reproduced/HistGB.joblib --input your_recent_weather.csv --output results/task3/reproduced/inference.csv`。输入使用原表列名，并按唯一时间戳递增排列。
