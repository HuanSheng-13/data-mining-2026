# B 的准备产物规范

基础构建命令为 `python -m extension2_finance_agent prepare`。`reports.py` 只读取公开年报目录、HTML 和数据库，生成 `chunks.json` 与 `manifest.json`。文本块有稳定片段 ID、公司 CIK、报告期、源文件、内容和提取后字符范围；manifest 包含数据库、年报和文本块指纹。

优化构建可在本目录新增模块，并在数据优化配置中设置：

```json
{
  "prepare_factory": "extension2_finance_agent.prepare.optimized:prepare",
  "validate_factory": "extension2_finance_agent.prepare.optimized:validate_artifacts",
  "artifacts_dir": "extension2_finance_agent/artifacts/data_optimized"
}
```

`prepare(config)` 构建并返回可序列化 manifest，至少包含 `reports` 列表和整数 `chunks`，供 CLI 展示。`validate_artifacts(config)` 检查原始输入、准备参数和产物指纹，失配时抛出 `ValueError`，成功返回同样稳定的 manifest。验证不应重新构建数据或修改产物。运行快照还会记录 `artifacts_dir` 内所有文件指纹。

派生产物只放 `config.artifacts` 或明确定义的本地缓存，原始文件不改。源数据更新后重建，避免静默沿用旧索引；构建参数、版本和耗时需留档。公开的六家公司资料均可无标签预处理，train/test 参考答案、评分和导出的评测请求不得混入索引。

构建优化产物后，用相同数据配置运行工具或 Agent。不要将索引加入 Git；提交构建代码、参数和说明即可。
