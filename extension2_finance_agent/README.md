# 企业财务问答 Agent

项目只维护 `configs/final.json` 这一套方案：DeepSeek 工具调用 Agent、只读财务数据库、按公司与报告期检索的年报记忆库、参数化财务 Skill，以及程序执行的协议恢复和最终答案校验。方法由 train 开发结果确定；正式 test 必须先固定方法，整批作答保存后独立评分。

## 数据与目录

```text
extension2_finance_agent/
├── configs/final.json       # 唯一运行配置
├── core/                    # API、状态循环、协议恢复、配置和本地.env
├── tools/data.py            # 查询、检索、上下文读取、计算
├── prepare/reports.py       # 仅正常数据准备与输入指纹
├── strategies/finance.py    # 财务口径与停止指导
├── prompts/agent.txt
├── skills/                  # 指标、期间、计算、来源、错误恢复规则
├── experiments/             # 运行、冻结、独立评分、失败评分重试与核查
├── tests/                   # 仅针对当前方案的测试
├── artifacts/final/         # 可重建索引，不提交
└── outputs/                 # 新运行与冻结文件，不覆盖已有结果
results/extension2/          # 当前最终方案的train/test证据与交付说明
data/financeqa/FinanceQA/    # 老师的原始数据包，保持不变
```

数据包的 `agent_workspace/data/` 包含 `finance.db`、`schema.md`、`business_rules.md`、`companies.json` 和9份年报HTML。输入来自 `train/questions.json` 或 `test/questions.json`，必须含 `question_id`、字符串 `company_cik` 和英文 `question`。输出为 JSON 数组，每条含字符串 `question_id` 和 `answer`。

作答工具仅访问公开数据库、年报及派生产物，不提供任意文件、shell或URL访问。原始数据、协议与老师评测程序不改；派生索引另存。train参考答案和反馈可用于开发，test答案和反馈仅在方法固定及整批作答后独立使用，不能用于设计或修改Agent。评分输出不进入检索。

## 环境与本地配置

Python 3.11及以上；运行使用标准库，测试使用根目录已有pytest。图表与正式报告不是运行依赖。沿用仓库 `.venv`：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest extension2_finance_agent/tests -q
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent prepare
```

在本目录 `.env` 填写 `FINANCEQA_API_KEY`；端点和模型默认为 `https://api.deepseek.com/chat/completions`、`deepseek-flash`。检查器使用 `FINANCEQA_JUDGE_API_KEY`、`FINANCEQA_JUDGE_URL`、`FINANCEQA_JUDGE_MODEL`，可复用同一密钥。实际 `.env` 不进入Git，仅提交空密钥模板 `.env.example`。

配置优先级：非空进程环境变量、本目录 `.env`、Windows用户环境变量、JSON默认值。固定读取本目录文件，不搜索其他目录；支持单行KEY=VALUE、引号和注释，不展开变量或执行代码。其他机器可修改唯一配置中的 `data_root` 与输出位置。

## 方法与工具

模型自行选择调用顺序，读取结果后决定继续或结束。`get_schema` 返回核心表及报告目录；`discover_metrics` 发现实际标签；`query_sql` 只读、有时间/步数/结果限制；`search_reports` 返回公司及报告期对应的片段；`read_report_chunks` 取已知片段邻域解决表格切断；`calculate` 只接受标量算术，不执行代码。

输入使用公开业务规则与财务Skill指导：核对指标、单位、实际期间、重复披露和报表口径；从真实操作数计算；区分财务分部与业务领域、实际结果与风险/假设/计划。规则是模型指导，不是语义正确性的保证。

每题新建消息与工具实例。程序默认最多16轮、24次工具调用，最后两轮用于总结。若模型将DSML调用误放在正文，程序仅恢复注册工具的参数、真实执行并记录恢复事件；预算耗尽后不再执行。最终总结请求不再提供工具定义，残留工具标记会被拒绝并要求重答；完整写出的报告片段ID也会与已检索来源核对。无法形成有效最终答案则记录失败和空回答，不能冒充完成。

## 作答与结果

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent run --split train --output extension2_finance_agent/outputs/train_new
```

train开发可用 `--question-id` 或 `--limit`；完整验证使用全部37题。每次输出目录必须新建。保存 `predictions.json`、`summary.json`、`run.json` 与逐题 `logs/*.jsonl`，包括真实模型响应、工具参数/结果/错误、协议恢复、最终拒绝、用量、耗时、源码和输入指纹。公开日志不保存API授权头和隐藏思维链。

`completed` 代表通过程序的运行检查，不等于每句话财务正确。token用量来自服务响应，超时期间未知计费不当作零；不能直接用tokens推算账单。

## 冻结与正式 test

先根据train验证确定方法并提交代码，保持工作区干净。冻结文件记录配置、方法资源、输入和索引指纹。正式test使用同一提交，不允许选题或限制数量：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent freeze --output extension2_finance_agent/outputs/frozen.json
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent run --split test --frozen extension2_finance_agent/outputs/frozen.json --output extension2_finance_agent/outputs/test_first
```

每题从同一初始状态开始；全部19题预测保存后，才使用独立检查器。首次正式结果保留，不根据test评分改方法或覆盖回答。跨机器应在本机重新校验输入并创建冻结快照；首次提交结果的代码版本以其 `run.json` 为准。

## 独立评分与核查

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/train_new/predictions.json --check-only --output extension2_finance_agent/outputs/train_format.json
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/train_new/predictions.json --output extension2_finance_agent/outputs/train_score.json
```

test评分替换为 `--split test` 与对应路径，要求完整test运行已结束。老师原脚本独立比较原问题、参考答案和已保存回答，宽松支持率不等于严格完整正确率。评分服务失败时，仅对失败请求重试，沿用原提示词与检查器，成功判定不重新打分：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent.experiments.recover_judge --predictions extension2_finance_agent/outputs/train_new/predictions.json --source extension2_finance_agent/outputs/train_score.json --output extension2_finance_agent/outputs/train_score_recovered.json
.\.venv\Scripts\python.exe -X utf8 -m extension2_finance_agent.experiments.audit --run extension2_finance_agent/outputs/train_new --score extension2_finance_agent/outputs/train_score_recovered.json --output extension2_finance_agent/outputs/train_audit.json
```

没有评分失败时直接使用原评分进行核查。恢复报告记录原评分与预测指纹，不修改原始文件。评分、参考答案与恢复过程均不进入作答Agent。

## 交付

最终运行证据见 [results/extension2](../results/extension2/README.md)。仅保留本方案的实现、测试、文档和结果，早期方案保存在Git历史。课程正式报告由C依据最终证据统一撰写，工程交接见[最终协作说明](../report/extension2_work_plan.md)。
