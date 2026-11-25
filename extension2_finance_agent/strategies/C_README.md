# C：财务问答策略与 Skill 验收

负责人：王悠然（24371186，软件学院）。共同基线为 PR 5 合并提交 `8021a1628c48990030d3ae2dfe2a8381102ca0da`，开发分支为 `codex/financeqa-agent-skill`。

## 实现

`optimized.py` 遵守 A 的 `create_strategy(config).messages(task)` 接口，只构造初始指导，不修改公共调用循环或工具。模型仍根据真实工具返回选择下一步行动。系统消息绑定当前 CIK 和原调用预算；题目原文放在 user 消息中，不通过题号选择答案、SQL 或操作顺序。

| 资源 | 作用 |
| --- | --- |
| `prompts/skill_optimized.txt` | 按问题选择数值、文字或混合证据，先取证再回答 |
| `skills/financial_basis.txt` | 指标发现、期间与单位检查、重复披露及修订选择 |
| `skills/calculation_and_claims.txt` | 可比操作数、计算、趋势、百分比/百分点与因果边界 |
| `skills/recovery_and_stop.txt` | 根据报错或空结果调整查询，相关性核查、预算内停止 |
| `configs/skill_optimized.json` | 独立启用 C 策略，保留全部基础工具和预算 |

Skill 是带条件的模型指导，不是强制执行的程序校验器。CIK 绑定、模板解析和初始消息隔离由代码校验；期间、单位、去重及证据是否充分仍由模型根据工具结果判断，需要从真实日志与答案人工复核。未新增题间可变 memory；每题恢复同一版本资源和新的消息列表。

## 运行

在仓库根目录运行，先按主 README 准备 Python 与 FinanceQA 数据。macOS/Linux 使用 `python3` 或虚拟环境内的 Python；Windows 可替换为 `.\.venv\Scripts\python.exe`。

```sh
python3 -m extension2_finance_agent prepare
python3 -m extension2_finance_agent run --config extension2_finance_agent/configs/skill_optimized.json --split train --question-id FQA-1682852-05 --output extension2_finance_agent/outputs/c_single_v1
python3 -m pytest extension2_finance_agent/tests -q
python3 -m extension2_finance_agent.strategies.compare_train --output extension2_finance_agent/outputs/c_train_compare_new
```

比较入口先验证两配置只有策略工厂、提示词、Skill 三项不同；固定 6 道 train 题，依次运行基础和优化策略，保存两组预测后独立调用老师原评测程序。它只接受 train，不运行正式 test。输出目录必须新建，原预测不手工修改。

本地 `.env` 配置作答和检查器，密钥不进入 Git。两者可以用同一个用户密钥，但用途仍相互独立。当前比较均为 `deepseek-flash`，温度 0、最大输出 2500 tokens、12 轮模型调用、16 次工具调用，基础数据库与字符片段检索不变。`compare_train.py` 会记录实际配置和指纹，环境覆盖也进入快照。

## 运行后核查

```sh
python3 -m extension2_finance_agent.strategies.audit_trace --run extension2_finance_agent/outputs/c_train_compare_new/skill_optimized --output extension2_finance_agent/outputs/c_train_compare_new/skill_trace_audit.json
```

核查器只读保存的日志，统计真实工具报错、原样重复失败、成功计算及最终正文中的已知工具标记，不改变回答或官方评分，也不将“有来源”直接当作正确。核查器本身的合成测试不算真实模型实验。

## 结果与交接

真实结果、逐题对照与案例见 `results/extension2/c_strategy_v1/`。无 API Key 的成员仍可运行离线测试和真实数据工具检查，不能将此称作模型效果验证。

本轮为 train 开发子集，不能替代 A 的 37 题四组统一对比或 19 题首次正式 test。单次支持性评分不是严格完整正确率；模型即使温度 0 也可能波动。额外提示词会增加输入成本，不保证指标提升。真实评分若显示无提升，按实际结果保留，不选择性丢弃失败题。

A 集成时保留 C 的三个资源配置项，用自己的组合配置叠加 B 的工具和准备工厂；不要覆盖 baseline。C 的 PR 由作者以外成员审核，本分支不自行合并。按当前安排，报告等待 B 完成及 A 提供完整对比、首次 test 后再统一撰写；本次 PR 只交付 C 的策略与验证材料。
