# 数据挖掘导论 2026 课程大作业

本项目用于完成四项必做任务：图像无监督聚类、图像异常检测、时间序列预测和无监督疾病判断。各任务分别使用指定数据集，独立实现、运行和评估，最终汇总实验报告和代码。

本仓库目前是项目骨架，尚未实现算法或运行入口。下文的命令为后续实现时应遵循的接口约定，当前不能直接运行。任务要求依据《作业说明_2026.pdf》整理；如有课程后续通知，以老师最新要求为准。

## 目录结构

```text
data-mining-2026/
├── README.md
├── requirements.txt             # 汇总经实际验证的 Python 依赖及版本
├── .gitignore
├── data/                        # 原始数据仅存本地，下载和放置见内部 README
│   ├── README.md
│   ├── cluster/
│   ├── image_anomaly/
│   ├── weather/
│   └── thyroid/
├── task1_clustering/README.md
├── task2_image_anomaly/README.md
├── task3_forecasting/README.md
├── task4_thyroid/README.md
├── results/README.md             # 保留最终指标和报告使用的图表
└── report/README.md              # 报告材料和分工说明
```

## 环境配置

建议每人在本地建立虚拟环境；Python 版本及依赖版本待实验验证后统一补充。不要将虚拟环境提交到仓库。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前 requirements.txt 只有占位说明，不包含算法所需依赖。引入新依赖时应同步更新该文件；若某任务需要额外依赖或独立环境，应在其 README 中说明并提供对应依赖文件。

## 四个任务独立运行的约定

1. 每个任务在自己的目录中提供 `main.py`，能够独立完成本任务所需的数据处理、训练（如需要）和评估；不得要求先运行另一个任务。
2. 所有示例命令均从仓库根目录执行。数据和结果路径通过参数传入，不得写死个人电脑路径。使用 Python 脚本定位资源时，应明确相对路径的基准。
3. 统一支持 `--data-dir`、`--output-dir`、`--seed` 参数；其他参数及默认值写入各任务 README。
4. 结果写入各自独立的目录，不修改原始数据，不覆盖其他任务的产物。
5. 如需预训练权重、网络下载或 GPU，应说明获取方法、运行条件及限制。可复用工具代码，但不能依赖另一个任务已生成的中间文件。

待实现的运行接口：

```powershell
python task1_clustering/main.py --data-dir data/cluster --output-dir results/task1 --seed 42
python task2_image_anomaly/main.py --data-dir data/image_anomaly --output-dir results/task2 --seed 42
python task3_forecasting/main.py --data-dir data/weather --output-dir results/task3 --seed 42
python task4_thyroid/main.py --data-dir data/thyroid --output-dir results/task4 --seed 42
```

每个任务的具体要求、实现记录和待补充运行说明，请查看相应目录中的 README。

## 协作及代码管理方案

- 共用一个远程仓库，`main` 分支保存经检查、可交付的内容。
- 在修改前同步主分支，再按任务或功能创建分支，例如 `task1/features`、`task3/baseline`、`docs/readme`；不要求固定人员与分支对应。
- 每次提交围绕一个明确改动，提交信息说明做了什么，例如 `task3: add chronological split`。
- 完成一段工作后推送分支并发起 Pull Request / Merge Request，说明改动、运行方式、实际验证情况和未完成事项，经过另一位成员检查后合并。
- 公共文件（总 README、依赖文件、报告）修改前先沟通；合并冲突时确认双方修改意图，不直接用整份文件覆盖。
- 不随意强制推送共享分支，不改写其他成员已共享的提交历史。
- 不复制多个 `final`、`final2` 文件保存版本；使用 Git 记录历史。未完成代码和已验证结果应在说明中明确区分。

基本命令示例（已克隆并配置远程仓库后使用）：

```powershell
git switch main
git pull --ff-only
git switch -c task1/features
# 修改并检查后，只添加与本次改动有关的文件
git add task1_clustering/README.md
git commit -m "docs: update task1 instructions"
git push -u origin task1/features
```

## 数据、结果和实验注意事项

- 原始数据、下载压缩包、模型权重、特征缓存和本地环境不提交；数据下载及放置方法见 [data/README.md](data/README.md)。先检查 `.gitignore`，提交前用 `git status` 确认文件范围。
- 提交复现实验所需的代码、配置、依赖说明，以及精简的最终指标和报告图表。大体积结果另行存储，并记录获取方式。
- 训练和测试应分离：预处理参数只从训练数据估计，不用测试标签或测试结果反复挑选模型、参数或阈值；如需调参，设置验证方案。
- 时间序列按时间划分数据，避免未来信息泄漏。记录窗口与目标的对齐方式，以及划分边界的处理方式。
- 固定随机种子，记录算法、超参数、数据划分、指标、运行环境和必要的硬件信息。结果需与报告描述对应。
- 不提交密码、访问令牌或个人本地配置。首次使用数据前确认放置结构与各任务代码一致，尽量保留老师原始数据的内部结构。

## 提交与复现检查

课程要求提交：实验报告 PDF、代码仓库链接（README 写明环境配置和运行方法）、分工说明 PDF。提交时确保老师能访问仓库；压缩包命名和提交地址以作业说明为准。

作业说明中的截止时间：2026 年 12 月 27 日 23:59 前。

提交前在一个新的克隆目录中按文档安装环境、放置数据，分别运行四项任务，确认不依赖个人电脑的隐藏文件或其他任务产物；检查报告中的结果可对应到保存的实验配置与输出。

拓展作业是自愿加分项，后续如开展，应单独增加目录及说明，保持四项必做任务可以独立运行。
