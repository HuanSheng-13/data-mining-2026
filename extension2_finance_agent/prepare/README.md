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

在仓库根目录用 PowerShell 执行以下命令。也可用 `py -3.11` 替换虚拟环境中的解释器。

```powershell
$config = 'extension2_finance_agent/configs/data_optimized.json'
.\.venv\Scripts\python.exe -m extension2_finance_agent prepare --config $config
.\.venv\Scripts\python.exe -m extension2_finance_agent smoke-tools --config $config --output extension2_finance_agent/outputs/b-smoke-reproduced.json
.\.venv\Scripts\python.exe -m extension2_finance_agent prepare
.\.venv\Scripts\python.exe -m extension2_finance_agent.prepare.verify_tools --output extension2_finance_agent/outputs/b-comparison-reproduced.json
```

`data_optimized.json` 沿用基础模型配置、策略和提示词，替换工具与准备工厂。工具验证和对比无需 API 密钥。使用基础 Agent 运行 train 时，按项目 README 配置模型后，为 `run` 添加 `--config $config`，并提供新的输出目录。对比脚本只读取 train 题目及公共数据，使用手动给定的查询。

构建需要 Python 自带的 SQLite 启用 FTS5。索引由年报段落和表格行组成。使用 Porter 词干和带字段权重的 [SQLite FTS5 BM25](https://www.sqlite.org/fts5.html#the_bm25_function) 排序，正文、章节、表头的权重分别为 1.0、0.25、0.45。检索按公司限定，可用 `report_end` 精确限定期末。每个片段包含至多 1,300 字符。表格行保留原顺序、空格位、跨列和跨行信息，以及至多三行推断出的表头，超长结构可回查原 HTML。

来源的 `source` 相对于 `agent_workspace`。用 Python 以 UTF-8、`errors="replace"` 读取同一原文件后，`html_start:html_end` 可定位对应原始段或行，包含的标签可能跨边界。它们是解码后的字符偏移，使用零起始和右端开区间。表格另带起始位置和行号。 `char_start:char_end` 则位于优化版拼接的规范化文本中，与基础版的偏移属于各自的提取文本。

指标目录按公司、XBRL 概念、分类和单位聚合实际披露。别名用于找候选标签。查询沿用基础版只读限制，结果保留空值和重复披露。同一次查询返回已投影的期间、单位、fact ID 和 accession 等字段供追踪；计算或别名字段的来源需结合原 SQL 判断。

技术记录、成本及具体证据位置见[任务 B 交接](../../results/extension2/b_data_tools_v1/README.md)。
