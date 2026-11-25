# A 的完整 train 实验

`full_train.py` 在同一干净提交上运行四配置、全部 37 道 train 题。`combined.json` 将 B 的数据准备和工具工厂与 C 的策略资源组合；其余模型、温度、输出长度、轮数、工具预算与数据根目录保持一致。不会运行正式 test。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest extension2_finance_agent/tests -q
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent.experiments.full_train --output extension2_finance_agent/outputs/full_train_v1 --jobs 4
```

运行前提交全部方法文件。入口校验四配置的控制变量，分别构建基础和 B 索引，记录准备成本和输入指纹。最多四组并发，每组内部逐题运行，每题重新初始化状态；全部 148 条预测保存后才独立评分。模型服务负载和缓存可能影响用量与耗时解释，不能将单轮时间差称为可靠加速。

输出根目录包含 `design.json`、四组原始预测/日志/配置、独立评分、`comparison.json`、逐题配对、`metrics.csv` 和运行后 trace audit。空答案和失败题仍保留在 37 题分母中；服务失败必须修复评分后再比较，不通过重新作答替换失败结果。

只重试 `API_FAILED` 请求时，使用独立恢复入口。它复用老师原评测函数、提示词和相同检查器，只对失败题重试，保留原评分，并在派生报告中记录恢复轨迹和预测指纹。成功判定不重评：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent.experiments.recover_judge --output extension2_finance_agent/outputs/full_train_v1 --source-id judge_v1 --target-id judge_recovered_v1
```

只修复检查器服务并对同一批已保存的预测重评时，用新的评分目录，保留第一次结果：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent.experiments.full_train --output extension2_finance_agent/outputs/full_train_v1 --evaluate-only --evaluation-id judge_v2
```

仅重新生成统计表可用 `--summarize-only --evaluation-id judge_v1`，不会调用模型。支持率是宽松指标，需要结合矛盾、空答案、工具错误、伪工具正文和人工财务口径核对。根据 train 选择下一阶段方案，正式 test 留到方法冻结后。

本轮完整实验与交接见[四配置train结果](../../results/extension2/full_train_v1/README.md)。
