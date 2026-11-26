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
results/extension2/          # 当前最终方案的train/test原始证据
report/extension2.md         # 方法、结果与分析的初版报告
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

### 年报准备（prepare）

`prepare/reports.py` 读取公开年报目录，提取HTML正文与表格单元格分隔，去掉隐藏XBRL内容，按1,800字符窗口、200字符重叠切块。片段保留公司、期末、文件、提取文本位置和稳定ID，临近片段可由工具补读。本方案不重建完整复杂表格，也不使用外网资料。

准备产物写入 `artifacts/final/`，记录数据库、报告和片段文件指纹。运行前验证输入与产物是否失配，变化后重新准备并另建输出目录。原始数据不改；全部六家公司资料可无标签准备，参考答案与评分不得进入产物。

### 数据工具（tools）

`tools/data.py` 是唯一工具注册表。模型自行选择调用顺序，读取结果后决定继续或结束。

| 工具 | 用途与限制 |
| --- | --- |
| `get_schema` | 返回核心表、字段和当前公司的报告目录 |
| `discover_metrics` | 按当前公司发现实际指标标签，关键词按字面子串匹配 |
| `query_sql` | 只读SQL查询，限制SQL长度、结果量、值大小与执行步数 |
| `search_reports` | 按当前公司和可选报告期检索年报，保留片段位置与来源 |
| `read_report_chunks` | 读取同公司片段及邻域，补全被切断的表格或段落，不接受任意文件路径 |
| `calculate` | 标量算术计算，只接受加减乘除与括号，不执行代码 |

所有工具返回 `ok`、`data`、`sources`、`error`，结果必须可严格JSON序列化。结构与指标发现不冒充数值证据；空查询无来源；报告来源保留文件、期末、片段ID和提取文本位置。

数据库通过只读连接与授权器限制写入、附加库和扩展函数。模型仍需核对指标、单位、期间与重复披露；源数据NULL不能替换成零。报告检索和邻域读取始终绑定当前公司。真实调用、失败和协议恢复均由公共循环记录。

### 财务Skill与策略（skills、strategies）

`strategies/finance.py` 加载唯一提示词及三份财务Skill，绑定已校验的公司CIK和当前预算，题目原文保留在user消息。每次创建全新消息，不携带上一题状态。模型根据实际返回的证据决策，不按题号选择答案、SQL或固定步骤。

| 初始资源 | 内容 |
| --- | --- |
| `skills/financial_basis.txt` | 指标定位、单位与期间口径 |
| `skills/calculation_and_claims.txt` | 可比计算、事实与判断、来源引用 |
| `skills/recovery_and_stop.txt` | 失败恢复、预算与停止条件 |

三份Skill属于本财务Agent的初始资源，不是Codex系统技能。每题加载同一版本，参数仅为CIK和预算，不含逐题答案或test反馈。

输入使用公开业务规则与财务Skill指导：核对指标、单位、实际期间、重复披露和报表口径；从真实操作数计算；区分财务分部与业务领域、实际结果与风险/假设/计划。规则是模型指导，不是语义正确性的保证。

程序校验协议和引用片段可追溯性，模型负责财务口径与语义判断；工具被调用或有引用不自动证明结论正确。正式test前固定资源，评分后仅可整理结果，不用反馈改变方法。

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

`experiments/` 统一负责运行与独立评测，只维护当前单一方案，不包含方法切换或候选比较。

| 模块 | 职责 |
| --- | --- |
| `runner.py` | 完整预测、状态、真实日志、方法快照和冻结 |
| `evaluate.py` | 在独立子进程调用老师原脚本，记录检查器设置及预测指纹 |
| `recover_judge.py` | 仅重试API失败的评分请求 |
| `audit.py` | 核查已保存记录，不生成或改写答案与判定 |

所有输出新建目录，原结果不覆盖。正式test每题独立，整批保存后才评分，参考答案与评分材料不提供给Agent。

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

方法、结果与局限见[初版实验报告](../report/extension2.md)，文件索引见[结果说明](../results/README.md#extension2)，原始证据位于 `results/extension2/`。仅保留本方案的实现、测试、文档和结果，早期方案保存在Git历史。报告初稿供C统一核对、完善和导出正式课程报告。

只核查已交付材料、不调用模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m results.extension2.build_delivery --verify-only
```

已交付的56题答案和日志保留原始字节，`checksums.json` 可用于核查。重跑应使用新输出路径，例如 `outputs/frozen_reproduce.json`、`outputs/test_reproduce`。本次文档整理更改了方法快照覆盖的README文件，旧冻结快照只证明首次运行，不能直接用于当前提交；复现时应在干净提交上重新冻结，不覆盖旧证据。
