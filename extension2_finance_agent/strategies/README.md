# C 的策略与 Skill 开发规范

保留 `baseline.py` 和 `prompts/baseline.txt`。C 可以先复制基础配置，在 `skill_files` 中列出新增指导文件，无需修改公共调用循环：

```json
{
  "strategy_factory": "extension2_finance_agent.strategies.baseline:create_strategy",
  "prompt_file": "extension2_finance_agent/prompts/baseline.txt",
  "skill_files": ["extension2_finance_agent/skills/period_checks.txt"]
}
```

也可新增策略模块，通过 `strategy_factory` 配置加载。统一接口为 `create_strategy(config)` 返回对象，该对象的 `messages(task)` 返回新的 system/user 消息列表。不得修改传入任务、在题目之间保留执行状态，或读取独立评测资料。通用初始 memory 可作为版本化指导文件加载，每题恢复同一内容。

当前公共循环根据模型真实返回的工具调用执行行动，策略负责初始指导；如需题内动态策略回调，应先与 A 对齐接口，不直接覆盖公共循环。指导可包括指标定位、期间与单位检查、去重、计算条件、证据核查和失败恢复，不能按 test 题号硬编码答案或固定每题全部步骤。

策略优化配置保持 `tool_factory`、准备工厂及数据配置为基础版，确保独立于 B 的优化运行。新增测试放在 `tests/test_strategy_optimized.py`，验证指导载入、状态重置和协议兼容，再使用 train 分析实际表现。

C 是报告唯一作者，正式报告放仓库 `report/extension2.md`，PDF 由 C 统一导出。A、B 只提供结构化方法和实验材料。C 的策略 PR 与报告 PR 均由 A 审核。
