# 任务4：无监督疾病判断

使用六个脱敏数值特征学习正常样本分布，并在测试集上判定异常，标签 0 为正常，1 为患病。将课程 `thyroid.zip` 的两个 CSV 放入 `data/thyroid/`。完整解压后若保留包内 `thyroid/` 同名子目录，程序也可定位文件。参见 [数据说明](../data/README.md)。

从仓库根目录在 Windows PowerShell 执行。任务使用 Python 3.11 和 CPU。如果已安装任务1的 CPU 锁定环境或任务3的依赖，可直接运行最后一条命令。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-task3-task4.txt
.\.venv\Scripts\python.exe task4_thyroid/main.py --data-dir data/thyroid --output-dir results/task4/reproduced --seed 2026 --alpha 0.05
```

默认参数与上述命令一致。按重复特征向量分组，将正常训练样本划分为拟合与校准子集。标准化和模型只在拟合子集学习，异常阈值仅使用正常校准分数。正式方法为新颖性检测模式下的 LOF，同时固定比较孤立森林和 One-Class SVM。测试标签只用于最终评价。

运行后可查看异常分数、逐条预测、测试指标、阈值敏感性、ROC/PR 曲线和混淆矩阵。正式结果见 [实验报告](../results/task4/report.md) 与 [测试指标](../results/task4/metrics.csv)。依赖与数据准备完成后可离线运行。

对新增 CSV 打分可执行 `python predict.py --task 4 --model results/task4/reproduced/LOF.joblib --input your_thyroid_features.csv --output results/task4/reproduced/inference.csv`。输入保留六列特征名即可。按默认路径跑完两项实验后，可从仓库根目录执行 `python -m unittest test_tasks3_4 -v` 检查窗口边界、阈值与预测的一致性。
