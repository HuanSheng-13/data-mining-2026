# Agent 的任务指导文件

C 在此编写供财务问答 Agent 使用的 Skill 或通用初始 memory，通过配置 `skill_files` 加载。基础版列表为空，不预先实现 C 的优化。文件应可版本追溯、跨问题复用，每题从同一初始内容开始，不包含 test 答案或评分反馈。

策略接口、验证和报告责任见[策略开发规范](../strategies/README.md)。
