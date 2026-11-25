# A 基础 demo 真实模型验收记录

2026 年 11 月 25 日，使用 `deepseek-flash` 运行 4 道 train 题，并使用老师原始 `evaluate.py` 独立评分。模型和工具均真实执行，没有用脚本响应代替模型。本次功能自验通过，后续仍按小组规则由 B 审核 A 的 PR。

## 版本与范围

- 作答代码提交：`a02da9e`，分支 `codex/financeqa-demo`；开始作答时工作区干净。
- 配置：`extension2_finance_agent/configs/baseline.json`，关闭 thinking，12 轮上限、16 次工具调用上限。
- 作答服务：`https://api.deepseek.com/chat/completions`；密钥仅从本地 `.env` 读取，不写入结果。
- 题目范围：Snowflake 3 题、Moderna 1 题，覆盖战略、产品技术与收入趋势。
- 本次未运行完整 37 题 train、正式 19 题 test，也未使用 test 参考答案或反馈。

## 逐题结果

| 题号 | 内容 | 模型轮数 | 工具调用 | 作答耗时 | 检查器判定 |
| --- | --- | ---: | ---: | ---: | --- |
| FQA-1640147-01 | Snowflake 增长战略 | 4 | 6 | 9.39 秒 | 支持 |
| FQA-1640147-05 | Snowflake 收入增长趋势 | 6 | 11 | 9.86 秒 | 支持 |
| FQA-1640147-06 | Snowflake 产品与技术举措 | 4 | 6 | 8.84 秒 | 支持 |
| FQA-1682852-05 | Moderna 2022 至 2024 年收入趋势 | 5 | 7 | 8.36 秒 | 支持 |

整批作答约 37.06 秒；逐题耗时之和约 36.46 秒，另有前置检查、写盘与对象构建等开销。实际值以 [acceptance.json](acceptance.json) 和 [summary.json](summary.json) 为准。检查器耗时和 token 用量没有由老师原脚本单独记录，不包含在以上作答统计中。

## 工具与依据核对

实际工具调用共 30 次：`get_schema` 4 次、`discover_metrics` 2 次、`query_sql` 3 次、`search_reports` 12 次、`calculate` 9 次。真实调用参数、结果、来源和耗时见 [tool_calls.jsonl](tool_calls.jsonl)。

Snowflake 收入题的第一次 SQL 因将空维度字段当作 NULL 而未查到数据，模型随后调整条件重新查询，获得实际记录后计算增速，展示了基于工具返回继续决策的过程。这里没有手工补写查询或修改答案。

验收核对了以下事项：

- 4 题均有独立的开始、工具调用和结束日志，全部状态为 `completed`。
- 年报检索返回的公司与当前题目公司一致；答案中完整写出的报告片段 ID 能对应到实际检索结果。
- Snowflake 年度收入和增速、Moderna 年度收入和下降比例与查询及计算结果对应。该核对不构成对每句定性描述的穷尽审查。
- 数据库与 9 份原始报告的指纹未变；作答和评分输出未包含两项已配置密钥的实际值。
- 36 项自动化回归测试通过；逐题状态重置、协议和工具限制另有相应自动化测试。

服务返回的作答累计用量为输入 161,357 tokens、输出 4,831 tokens，总计 166,188 tokens；输入中缓存命中 104,832、未命中 56,525。该统计累计多轮上下文，真实费用以账户计费为准。本次没有为了提高评分而反复作答。

## 评分口径

[原始格式检查](official_format_check.json) 与 [原始评分](official_score.json) 均直接保留老师程序输出。任意选题 demo 没有使用 `--limit`，因为老师的该参数只支持 train 的前 N 题。

老师原脚本固定读取全部 37 道 train 题。本次有 4 条非空预测，其余 33 题标记为缺失并计入信息不足，因此原始总支持率为 `4 / 37 = 10.8108%`。评分流程完整，无 API 失败；`evaluation_complete=true` 表示检查器流程完成，不表示全部 37 题均已作答。

从同一原始评分中仅取本次 4 条预测对应的判定：支持 4、矛盾 0、信息不足 0，作答子集支持率为 `4 / 4 = 100%`。这是选定 4 题的 demo 观察结果，不能用于推断完整 train 或 test 的表现，支持性判定也不等于严格完整答对率。

## 文件与复现

| 文件 | 内容 |
| --- | --- |
| [predictions.json](predictions.json) | 原样保存的 4 条真实模型预测 |
| [summary.json](summary.json) | 原样保存的逐题答案、状态、用量和耗时 |
| [tool_calls.jsonl](tool_calls.jsonl) | 从原始逐题日志中原样筛出的实际工具事件 |
| [official_format_check.json](official_format_check.json) | 老师原始格式检查输出，无模型调用 |
| [official_score.json](official_score.json) | 老师原始独立评分输出，保留 37 题分母 |
| [acceptance.json](acceptance.json) | 验收摘要、统计口径、原始输出和日志指纹 |

完整原始会话和方法快照位于本机 `extension2_finance_agent/outputs/demo_live_v1/`，其指纹记入 `acceptance.json`。以下命令展示本次运行方式，重复运行需更换输出路径：

```powershell
.\.venv\Scripts\python.exe -m extension2_finance_agent run --split train --question-id FQA-1640147-01 --question-id FQA-1640147-05 --question-id FQA-1640147-06 --question-id FQA-1682852-05 --output extension2_finance_agent/outputs/demo_live_v1
.\.venv\Scripts\python.exe -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/demo_live_v1/predictions.json --check-only --output extension2_finance_agent/outputs/evaluation/demo_live_v1_format.json
.\.venv\Scripts\python.exe -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/demo_live_v1/predictions.json --output extension2_finance_agent/outputs/evaluation/demo_live_v1_score.json
```

B 可基于真实工具记录分析检索冗余和指标发现；C 可基于逐题决策、空结果恢复和口径核对设计策略优化。报告正文仍由 C 统一撰写，本文件是 A 提供的功能验证材料。
