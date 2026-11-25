# 企业财务问答 Agent 基础开发说明

这里是拓展二的实际开发目录。A 提供基础 Agent、可替换工具接口、运行和评测入口；B 在基础版上优化数据与检索；C 优化策略和 Skill，并独自撰写报告。详细任务和审核关系见[三人分工文档](../report/extension2_work_plan.md)。

B、C 合并后的 A 集成包含 `configs/combined.json` 和[完整四配置train入口](experiments/README.md)。37题全量实验、错误与用量、候选方案及正式test前待办见[本轮交接结果](../results/extension2/full_train_v1/README.md)。

## 1 当前能力与数据位置

基础版使用 Python 标准库实现 Chat Completions 工具调用循环，默认对接 `https://api.deepseek.com/chat/completions` 的 `deepseek-flash`，关闭 thinking。每轮由模型选择真实工具，根据工具返回继续决策或作答。模型、调用预算和入口均在 [baseline.json](configs/baseline.json) 配置，没有新增运行依赖。协议参考 [DeepSeek 工具调用说明](https://api-docs.deepseek.com/guides/tool_calls/)。

工具包含数据库结构发现、指标名发现、只读 SQL、按公司和报告期过滤的年报关键词检索，以及受限算术计算。工具不能读取任意文件、执行 shell 或访问互联网。基础检索以固定字符窗口切分，保留表格单元格分隔；章节结构和跨窗口表格关系的优化交给 B。

数据包实际位于 `data/financeqa/FinanceQA/`，多一层 `FinanceQA` 是当前仓库的实际结构，保持即可：

```text
data/financeqa/FinanceQA/
├── agent_workspace/
│   ├── contracts/                 # 老师固定的输入输出协议
│   ├── data/
│   │   ├── finance.db             # SQLite，只读
│   │   ├── schema.md              # 表与字段说明
│   │   ├── business_rules.md      # 指标、单位、期间和去重规则
│   │   ├── companies.json         # CIK、公司名和划分
│   │   └── filings/               # 年报目录说明与 9 份 HTML
│   ├── train/                    # 37 题，可用于开发
│   └── test/                     # 19 题，方法冻结后运行
└── evaluation_only/              # 独立评测，不接入作答 Agent
```

不同机器通过配置的 `data_root` 指向包根目录。相对配置路径均以仓库根目录解析；运行命令统一在仓库根目录执行。原始数据库、年报、问题、参考答案、协议和评测程序均不修改，索引等派生产物另存。

## 2 数据规范

单题输入严格沿用老师协议：`question_id`、字符串 `company_cik`、英文 `question`。预测是普通 JSON 数组，每条包含字符串 `question_id` 和 `answer`。工具日志独立保存，不能手工改写已生成的预测。

数据库中 CIK 是字符串。财务值必须同时核对单位、指标定义、期间和重复披露；`fiscal_year`、`filed_date` 和 `end_date` 含义不同，不能将季度流量当成年值或把重复披露求和。NULL 不代表零。主实验事实依据限于包内数据库和年报。

train 参考答案和评分反馈可用于人工开发、优化，但基础作答工具不读取任何参考答案。允许对全部公司数据库和年报做无标签预处理。正式 test 每题从相同初始状态开始，整批作答完成后才评测；test 答案和反馈不用于调整方法。评测请求导出和评分输出也不得进入 Agent 的检索范围。

## 3 环境与准备

沿用根目录 Python 3.11 虚拟环境及依赖。Agent 运行只需要标准库，测试使用已有 pytest：

```powershell
.\.venv\Scripts\python.exe -m extension2_finance_agent --help
.\.venv\Scripts\python.exe -m extension2_finance_agent prepare
.\.venv\Scripts\python.exe -m extension2_finance_agent smoke-tools --output extension2_finance_agent/outputs/tools_check.json
.\.venv\Scripts\python.exe -m pytest extension2_finance_agent/tests -q
```

`prepare` 构建确定性的文本块和输入指纹，输出到 `artifacts/baseline/`。运行前自动验证数据库、原始年报和文本块是否发生变化；变化后应重新准备并使用新实验目录。`smoke-tools` 直接运行真实数据工具，其中计算器使用明确标注的示例数值；其输出不是模型答案或问答分数。

推荐将 [.env.example](.env.example) 复制为本目录的 `.env`，在本机填入密钥。程序自动读取固定位置 `extension2_finance_agent/.env`，无需设置 Windows 永久环境变量，也不需要手动加载文件。已有 `.env` 时保留其内容，不用模板覆盖：

```dotenv
FINANCEQA_API_KEY=在本机填入你的密钥
FINANCEQA_CHAT_URL=https://api.deepseek.com/chat/completions
FINANCEQA_MODEL=deepseek-flash
```

查找顺序为非空的当前进程环境变量、本目录 `.env`、Windows 用户环境变量；端点与模型最后回退到 JSON 配置。当前终端已有变量会优先于文件。只读取这个固定位置，不从工作目录或父目录搜索其他 `.env`，不把文件内容写入全局环境、配置快照或日志。

支持单行 `KEY=VALUE`、可选 `export`、单引号或双引号，以及空白分隔的行尾注释；带 `#` 的值可加引号。不展开 `${变量}`、转义序列或多行内容。文件格式错误只报告行号，不回显原始行。真实 `.env` 被 Git 忽略，仅提交无密钥的 `.env.example`。

`FINANCEQA_CHAT_URL`、`FINANCEQA_MODEL` 可覆盖端点和模型；其他参数通过复制配置文件修改。默认关闭 DeepSeek thinking；其他服务需兼容 `messages`、`tools`、`tool_choice`、非流式返回及 `choices[0].message.tool_calls`。不同服务的兼容性需用实际请求验证。

## 4 单题与 demo 运行

以下 train 题覆盖文字检索、数值趋势和计算。每次使用全新的输出目录：

```powershell
.\.venv\Scripts\python.exe -m extension2_finance_agent run --split train --question-id FQA-1640147-01 --output extension2_finance_agent/outputs/single_v1
.\.venv\Scripts\python.exe -m extension2_finance_agent run --split train --question-id FQA-1640147-01 --question-id FQA-1640147-05 --question-id FQA-1640147-06 --question-id FQA-1682852-05 --output extension2_finance_agent/outputs/demo_v1
```

输出包括：

| 文件 | 内容 |
| --- | --- |
| `predictions.json` | 老师要求的预测数组，每完成一题即保存 |
| `summary.json` | 逐题状态、耗时、模型调用、工具调用和模型返回的 token 用量 |
| `logs/000.jsonl` 等 | 逐题输入、公开模型响应、真实工具参数和结果、错误与耗时 |
| `run.json` | Git 提交及未提交状态、完整无密钥配置、输入和代码指纹、准备产物指纹及运行状态 |

每题新建会话、工具注册表和策略对象。模型调用失败、没有取得证据、截断和预算耗尽会单独记录，失败回答保存为空字符串，进程最终返回非零状态。有效的 SQL 或报告证据是正常完成的前提。相同输出目录不会被覆盖。

`model_calls` 统计逻辑请求轮数，HTTP 重试次数在对应模型事件的 `attempts` 中。用量仅累计服务实际返回的数据，连接失败后的未知计费不能当作零成本。日志只保存公开回答与工具调用，不保存隐藏思维链或授权头。

## 5 独立评测与完整 train

评测入口单独启动老师原始脚本，不进入 Agent 的工具列表。先检查格式：

```powershell
.\.venv\Scripts\python.exe -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/demo_v1/predictions.json --check-only --output extension2_finance_agent/outputs/evaluation/demo_v1_format.json
```

任意选题 demo 的格式检查仍以全部 train 为预期集合，其他未作答题会标为缺失；这不等于 demo 格式错误，也不产生支持率。官方 `--limit` 仅选择 train 的前 N 题，不能与任意题号子集混用。

完整 train 基线在 demo 审核后运行，命令去掉选题参数：

```powershell
.\.venv\Scripts\python.exe -m extension2_finance_agent run --split train --output extension2_finance_agent/outputs/train_baseline_v1
.\.venv\Scripts\python.exe -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/train_baseline_v1/predictions.json --check-only --output extension2_finance_agent/outputs/evaluation/train_baseline_v1_format.json
```

真实评分在同一个本地 `.env` 中配置以下三项。需要复用作答密钥时，在本机将同一个实际值填入检查器密钥项；不能写 `${FINANCEQA_API_KEY}`，程序不会展开变量。评测与作答流程保持分离：

```dotenv
FINANCEQA_JUDGE_API_KEY=在本机填入检查器密钥
FINANCEQA_JUDGE_URL=https://api.deepseek.com/chat/completions
FINANCEQA_JUDGE_MODEL=deepseek-flash
```

配置后可直接执行；显式 `--judge-url`、`--model` 参数优先于文件和环境设置：

```powershell
.\.venv\Scripts\python.exe -m extension2_finance_agent evaluate --split train --predictions extension2_finance_agent/outputs/train_baseline_v1/predictions.json --output extension2_finance_agent/outputs/evaluation/train_baseline_v1_score.json
```

支持率是宽松的支持性指标，不能称为完整答对率；检查 `evaluation_complete` 和服务失败记录。多组对比必须保持相同检查器配置。评分失败时可重评同一批已保存答案，不能据 test 反馈修改答案。

## 6 B 和 C 在哪里开发

| 人员 | 主要目录与文件 | 开发方式 |
| --- | --- | --- |
| A | `core/`、`__main__.py`、`experiments/`、公共依赖和本 README | 维护公共运行、集成和评测 |
| B | `tools/`、`prepare/`、新建的数据优化配置 | 新增优化模块，通过工厂替换工具和准备流程，保留 baseline |
| C | `strategies/`、`prompts/`、`skills/`、新建的策略优化配置 | 新增策略或 Skill，通过工厂和资源配置接入，保留 baseline |
| C | 仓库 `report/extension2.md` 与对应 PDF | 唯一报告撰写、编辑和导出负责人 |

具体接口见 [工具开发规范](tools/README.md)、[准备产物规范](prepare/README.md)、[策略开发规范](strategies/README.md)。C 的 Skill 目录用于本项目 Agent 的任务指导，不会自动安装为 Codex skill。

B 复制 `configs/baseline.json` 为 `data_optimized.json`，修改工具与准备工厂；C 复制为 `skill_optimized.json`，修改策略工厂、提示词和 `skill_files`。两人分别保持另一维度为基础实现，A 集成时再创建组合配置。命令通过 `--config extension2_finance_agent/configs/新配置.json` 选择，不通过覆盖 baseline 实现切换。

新增工具返回字段只能向后兼容；修改核心接口、根依赖和公共运行参数先在 PR 说明，由 A 协调。每个优化 PR 附运行命令、train 范围、配置和真实结果，不提交数据原文件、密钥或虚构日志。

## 7 分支与审核规范

A 在 `codex/financeqa-demo` 完成基础开发与 4 题真实 demo 验收，已向 `main` 提交[基础 PR 5](https://github.com/HuanSheng-13/data-mining-2026/pull/5)，等待审核合并。PR 不指定固定审核人，由作者以外的成员按小组安排审核；基础 PR 合并并确定提交号后，B、C 再从共同基线创建各自分支。报告由 C 独自撰写并提交 PR，A、B 只提供结构化材料与事实核对意见。当前阶段状态与待办见[分工文档](../report/extension2_work_plan.md)。

本目录内 `artifacts/` 和 `outputs/` 被 Git 忽略；`data/` 也被仓库忽略。源码、配置、测试和文档在本目录开发，精选验证材料放 `results/extension2/`。完整原始日志需本地保存并随最终提交材料交付。每个 PR 写明已完成与未完成的验证，不把离线脚本响应当成真实模型表现。

## 8 正式 test 前冻结

基础 demo 阶段不运行正式 test。选定最终方案并提交全部方法文件后，使用同一配置冻结和运行：

```powershell
.\.venv\Scripts\python.exe -m extension2_finance_agent freeze --output extension2_finance_agent/outputs/frozen_method.json
.\.venv\Scripts\python.exe -m extension2_finance_agent run --split test --frozen extension2_finance_agent/outputs/frozen_method.json --output extension2_finance_agent/outputs/test_v1
```

test 必须使用干净的 Git 工作区、同一提交和方法指纹，不允许选题或 `--limit`。冻结文件和运行输出应放在被忽略的输出目录，以免改变工作区状态。跨机器使用者需在本机校验数据并重新生成冻结快照，其中路径为本机绝对路径。全部 19 题作答后，再调用独立评测入口。首次正式预测与评测保留。

## 9 已知范围

当前基础检索没有语义向量、章节建模和复杂表格重建；SQL 的指标选择与期间判断主要依赖模型及公开业务规则。SQL 返回最多 100 行、单元值长度受限，过宽查询需模型缩小范围。基础配置限制 12 轮、16 次工具调用，最后一轮要求模型总结；配置可调整，但正式对比须保持一致。

本地测试包含合成数据、脚本化模型响应和本地 HTTP 协议检查，均不是实际模型准确率。真实数据工具验证和真实模型运行分别记录，具体完成状态见[验证记录](../results/extension2/README.md)。
