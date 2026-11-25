# B 的工具开发规范

在本目录新增如 `optimized.py`，保留 `baseline.py`。公共 Agent 只依赖以下接口：

```python
def create_tools(config, company_cik):
    # 每题返回新对象，绑定当前公司的检索上下文。
    return Registry(...)

class Registry:
    definitions = [...]  # Chat Completions function tools JSON schema

    def call(self, name: str, arguments: dict) -> dict:
        return {"ok": True, "data": ..., "sources": [...], "error": None}
```

`definitions` 必须与实际函数参数一致。失败返回相同四字段，`ok=False`、`data=None`、`sources=[]`、`error` 为可供模型修复调用的信息。结果可严格 JSON 序列化，不返回 NaN、二进制或不可序列化对象。

`sources` 非空表示取得了可用于回答的证据。结构发现、指标名列表、纯计算和空查询不应伪装为财务事实来源。SQL 结果记录查询与参数，并尽可能返回事实 ID、指标、单位、期间和 accession；报告结果保留原文、文件、报告期、片段 ID 和位置。基线位置是规范化文本的字符偏移，不是 HTML 行号。

基础工具名为 `get_schema`、`discover_metrics`、`query_sql`、`search_reports`、`calculate`，参数见 `baseline.DEFINITIONS`。兼容已有名称、参数和结果字段；新增字段可选。基础 SQL 工具只读，并限制结果行数、语句与值大小、执行步数和时间。查询由模型决定，工具不按题号准备答案。

工具只访问包内 `agent_workspace/data/` 及已登记的派生产物；不提供任意文件、shell、URL 或评测资料入口。公司资料可跨期检索，但必须明确报告身份。各题不共享可变执行状态。

将优化配置的 `tool_factory` 设为 `extension2_finance_agent.tools.optimized:create_tools`。数据优化单独运行时，策略工厂和提示词保持基础版。新增验证放在 `tests/test_tools_optimized.py`，使用临时合成数据检查接口及失败情况，再用 train 案例验证真实效果。

B 的 PR 由 A 审核，附索引构建命令、对比配置、实际验证结果与来源案例。根依赖和公共接口的变更由 A 协调。
