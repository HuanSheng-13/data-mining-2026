"""Real-data tool checks. These are NOT model-generated answers or scores."""
import time
from extension2_finance_agent.core.config import load_factory, write_json


def smoke_tools(config, output):
    if output.exists():
        raise ValueError("Use a new smoke output file")
    started = time.perf_counter()
    manifest = load_factory(config.validate_factory)(config)
    tools = load_factory(config.tool_factory)(config, "1640147")
    cases = [("get_schema", {}), ("discover_metrics", {"keyword": "Revenue"}),
             ("query_sql", {"sql": "SELECT id, fact_name, fact_value, unit, end_date, frame, accession_number FROM financial_facts WHERE cik=? AND fact_name LIKE ? ORDER BY end_date DESC LIMIT 5", "params": ["1640147", "%Revenue%"]}),
             ("search_reports", {"query": "artificial intelligence product technology", "report_end": "2025-01-31", "top_k": 3}),
             ("calculate", {"expression": "(120 / 100 - 1) * 100"}),
             ("query_sql", {"sql": "DELETE FROM financial_facts", "params": []})]
    results = [{"tool": name, "arguments": args, "result": tools.call(name, args)} for name, args in cases]
    assert all(row["result"]["ok"] for row in results[:-1])
    assert results[2]["result"]["data"]["rows"]
    assert results[3]["result"]["data"]["hits"]
    assert not results[-1]["result"]["ok"]
    report = {"kind": "real_data_tool_smoke_not_model_answers", "reports": len(manifest["reports"]),
              "chunks": manifest["chunks"], "elapsed_seconds": time.perf_counter() - started,
              "calculator_operands": "Synthetic 120/100 used only to validate arithmetic", "checks": results}
    write_json(output, report)
    return {"passed": len(results), "reports": len(manifest["reports"]), "chunks": manifest["chunks"]}
