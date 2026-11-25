# Agent 的任务指导文件

C 在此编写供财务问答 Agent 使用的 Skill 或通用初始 memory，通过配置 `skill_files` 加载。基础版列表为空，不预先实现 C 的优化。文件应可版本追溯、跨问题复用，每题从同一初始内容开始，不包含 test 答案或评分反馈。

策略接口、验证和报告责任见[策略开发规范](../strategies/README.md)。

C 的三份指导分别为 `financial_basis.txt`、`calculation_and_claims.txt`、`recovery_and_stop.txt`。优化配置通过 `strategies.optimized:create_strategy` 绑定 `$company_cik`、`$max_turns` 和 `$max_tool_calls`，基线列表仍为空。规则均为跨问题条件指导，不保存题号答案或跨题变化的 memory。实际效果和局限见 [验收记录](../../results/extension2/c_strategy_v1/README.md)。

最终规则的第二轮实际验收见 [c_strategy_v2](../../results/extension2/c_strategy_v2/README.md)。两轮基础支持分别为 4/6、6/6，C 均 6/6；不可把首轮差异当作稳定准确率提升。最终 48 项测试通过；报告按用户要求暂停，本次只验收 C 策略。
