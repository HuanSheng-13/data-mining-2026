"""Behavior tests with synthetic data and explicitly scripted model responses."""
from dataclasses import replace
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from extension2_finance_agent.core.agent import answer_question
from extension2_finance_agent.core.client import ChatClient, ModelError
from extension2_finance_agent.core.config import Config, sha256, write_json
from extension2_finance_agent.experiments.runner import load_tasks, run, freeze
from extension2_finance_agent.prepare.reports import ReportParser, prepare, validate_artifacts
from extension2_finance_agent.strategies.baseline import create_strategy
from extension2_finance_agent.tools.baseline import arithmetic, create_tools


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "FinanceQA"
    work = root / "agent_workspace"
    data = work / "data"
    (data / "filings/123").mkdir(parents=True)
    (data / "filings/456").mkdir()
    (data / "filings/README.md").write_text(
        "| Company | CIK | Report end | Local file |\n"
        "| Example | 123 | 2024-12-31 | [report](123/example.htm) |\n"
        "| Other | 456 | 2024-12-31 | [report](456/other.htm) |\n", encoding="utf-8")
    (data / "filings/123/example.htm").write_text("<html><head><style>hide</style></head><body><h1>Products</h1><p>Artificial intelligence product launches.</p><table><tr><td>Revenue</td><td>120</td></tr></table></body></html>", encoding="utf-8")
    (data / "filings/456/other.htm").write_text("<p>Unrelated other company confidential marker</p>", encoding="utf-8")
    (data / "business_rules.md").write_text("Keep units and periods comparable.", encoding="utf-8")
    (data / "schema.md").write_text("Financial facts have IDs and units.", encoding="utf-8")
    write_json(data / "companies.json", [{"cik": "123"}, {"cik": "456"}])
    with sqlite3.connect(data / "finance.db") as db:
        db.executescript("CREATE TABLE financial_facts (id INTEGER, cik TEXT, fact_name TEXT, fact_category TEXT, fact_value REAL, unit TEXT, end_date TEXT, frame TEXT, accession_number TEXT);"
                         "INSERT INTO financial_facts VALUES (1,'123','Revenue','us-gaap',120,'USD','2024-12-31','CY2024','source-1');")
    for split in ("train", "test"):
        write_json(work / split / "questions.json", [
            {"question_id": "first", "company_cik": "123", "question": "Revenue?"},
            {"question_id": "second", "company_cik": "123", "question": "Products?"}])
    cfg = Config(data_root=str(root), artifacts_dir=str(tmp_path / "artifacts"), retries=0)
    prepare(cfg)
    return cfg


def test_report_parser_preserves_cells_and_removes_hidden():
    parser = ReportParser()
    parser.feed("<head><style>bad</style></head><ix:hidden>secret</ix:hidden><p>Hello</p><table><tr><td>Revenue</td><td>120</td></tr></table>")
    assert "secret" not in parser.text() and "bad" not in parser.text()
    assert "Revenue | 120" in parser.text()


def test_preparation_is_deterministic_and_detects_stale_data(config):
    before = sha256(config.artifacts / "chunks.json")
    prepare(config)
    assert before == sha256(config.artifacts / "chunks.json")
    validate_artifacts(config)
    with (config.workspace / "data/filings/123/example.htm").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="Reports changed"):
        validate_artifacts(config)


def test_search_is_company_and_period_scoped(config):
    tools = create_tools(config, "123")
    hit = tools.call("search_reports", {"query": "artificial intelligence", "report_end": "2024-12-31"})
    assert hit["ok"] and hit["sources"]
    assert hit["data"]["hits"][0]["company_cik"] == "123"
    assert tools.call("search_reports", {"query": "confidential marker"})["data"]["hits"] == []
    assert tools.call("search_reports", {"query": "intelligence", "report_end": "2023-12-31"})["data"]["hits"] == []
    assert not tools.call("search_reports", {"query": "x", "top_k": True})["ok"]
    assert not tools.call("read_file", {"path": "../evaluation_only/test_reference_answers.json"})["ok"]


@pytest.mark.parametrize("sql", ["DELETE FROM financial_facts", "ATTACH DATABASE ':memory:' AS other", "PRAGMA table_info(financial_facts)",
                                  "WITH x AS (SELECT 1) DELETE FROM financial_facts", "SELECT load_extension('bad')", "SELECT 1; DELETE FROM financial_facts"])
def test_sql_blocks_writes_and_escape(config, sql):
    tools = create_tools(config, "123")
    before = sha256(tools.database)
    assert not tools.call("query_sql", {"sql": sql, "params": []})["ok"]
    assert before == sha256(tools.database)


def test_query_returns_sources_limits_rows_and_handles_errors(config):
    tools = create_tools(config, "123")
    result = tools.call("query_sql", {"sql": "SELECT id, fact_value, unit FROM financial_facts WHERE cik=?", "params": ["123"]})
    assert result["data"]["rows"] == [[1, 120, "USD"]] and result["sources"]
    assert not tools.call("query_sql", {"sql": "SELECT * FROM missing", "params": []})["ok"]
    result = tools.call("query_sql", {"sql": "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<150) SELECT x FROM n", "params": []})
    assert len(result["data"]["rows"]) == 100 and result["data"]["truncated"]
    assert not tools.call("query_sql", {"sql": "SELECT randomblob(1000000)", "params": []})["ok"]
    assert tools.call("discover_metrics", {"keyword": "Revenue"})["data"]["metrics"][0]["unit"] == "USD"
    assert tools.call("discover_metrics", {"keyword": "%"})["data"]["metrics"] == []


@pytest.mark.parametrize("expression", ["__import__('os').getcwd()", "1/0", "2**100", "True", "1e999", "[1,2]", "1" * 501])
def test_calculator_rejects_code_and_invalid_arithmetic(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        arithmetic(expression)


def test_calculator():
    assert arithmetic("(120/100-1)*100") == pytest.approx(20)
    assert arithmetic("-2 + 3 * (4 / 2)") == 4


def response(content=None, calls=None, finish_reason="stop"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"message": message, "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "finish_reason": finish_reason}


def call(name, arguments, identifier="tool1"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.histories = []

    def complete(self, messages, definitions, allow_tools):
        self.histories.append(json.loads(json.dumps(messages)))
        return next(self.responses)


def test_agent_uses_results_and_recovers_from_bad_arguments(config):
    tools = create_tools(config, "123")
    client = ScriptedClient([response(calls=[call("query_sql", "bad JSON")]),
                             response(calls=[call("query_sql", json.dumps({"sql": "SELECT * FROM financial_facts", "params": []}))]),
                             response("Revenue was USD 120 [financial_facts id 1].")])
    events = []
    task = load_tasks(config, "train")[0]
    answer = answer_question(task, config, client, tools, create_strategy(config), events.append)
    assert answer.status == "completed" and answer.tool_calls == 2
    assert answer.usage["prompt_tokens"] == 30
    assert any(e["event"] == "tool_result" and not e["result"]["ok"] for e in events)
    assert '"rows": [[1' in client.histories[2][-1]["content"]


def test_agent_requires_evidence_and_respects_budget(config):
    cfg = replace(config, max_turns=2, max_tool_calls=1)
    task = load_tasks(config, "train")[0]
    no_evidence = ScriptedClient([response("Unsupported assertion"), response("Still unsupported")])
    answer = answer_question(task, cfg, no_evidence, create_tools(cfg, "123"), create_strategy(cfg), lambda e: None)
    assert answer.answer == "" and answer.status == "no_evidence"
    client = ScriptedClient([response(calls=[call("calculate", '{"expression":"1+1"}'), call("calculate", '{"expression":"2+2"}', "tool2")]), response("No sources")])
    events = []
    answer = answer_question(task, cfg, client, create_tools(cfg, "123"), create_strategy(cfg), events.append)
    assert answer.tool_calls == 1
    assert [e for e in events if e["event"] == "tool_result"][1]["result"]["error"] == "Tool budget exhausted"


def test_agent_records_model_failure_and_truncation(config):
    task = load_tasks(config, "train")[0]
    answer = answer_question(task, config, ScriptedClient([]), create_tools(config, "123"), create_strategy(config), lambda e: None)
    assert answer.status == "model_error" and answer.answer == ""
    answer = answer_question(task, config, ScriptedClient([response("partial", finish_reason="length")]), create_tools(config, "123"), create_strategy(config), lambda e: None)
    assert answer.status == "truncated_response" and answer.answer == ""


def test_actual_http_protocol_and_batch_isolation(config, tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if request["messages"][-1]["role"] == "user":
                answer = response(calls=[call("query_sql", '{"sql":"SELECT * FROM financial_facts","params":[]}')])
            else:
                assert json.loads(request["messages"][-1]["content"])["data"]["rows"][0][4] == 120
                answer = response("Synthetic fixture revenue USD 120, source fact 1.")
            envelope = {"choices": [{"message": answer["message"], "finish_reason": "stop"}], "usage": answer["usage"]}
            raw = json.dumps(envelope).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_FINANCEQA_KEY", "test-only-secret")
    cfg = replace(config, chat_url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions", api_key_env="TEST_FINANCEQA_KEY")
    try:
        output = tmp_path / "run"
        result = run(cfg, "train", output)
        assert result["status"] == "completed"
        predictions = json.loads((output / "predictions.json").read_text())
        assert [p["question_id"] for p in predictions] == ["first", "second"]
        assert len(requests) == 4 and len(requests[0]["messages"]) == len(requests[2]["messages"]) == 2
        assert "test-only-secret" not in "".join(p.read_text() for p in output.rglob("*") if p.is_file())
        with pytest.raises(ValueError, match="new output"):
            run(cfg, "train", output)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_formal_test_requires_freeze_and_no_subset(config, tmp_path):
    with pytest.raises(ValueError, match="requires --frozen"):
        run(config, "test", tmp_path / "run", limit=1)


def test_api_secrets_and_configuration(monkeypatch):
    monkeypatch.setenv("TEST_FINANCEQA_KEY", "local-test")
    with pytest.raises(ValueError, match="credentials"):
        ChatClient(Config(api_key_env="TEST_FINANCEQA_KEY", chat_url="https://user:password@example.com/chat"))
    with pytest.raises(ValueError, match="HTTPS"):
        ChatClient(Config(api_key_env="TEST_FINANCEQA_KEY", chat_url="http://example.com/chat"))


def test_plugin_factories_and_skills_are_independent(config, tmp_path):
    from extension2_finance_agent.core.config import load_factory
    skill = tmp_path / "period_checks.txt"
    skill.write_text("Check the reporting period before arithmetic.", encoding="utf-8")
    cfg = replace(config, skill_files=[str(skill)])
    strategy = load_factory(cfg.strategy_factory)(cfg)
    task = load_tasks(cfg, "train")[0]
    first = strategy.messages(task)
    first.append({"role": "assistant", "content": "previous answer"})
    second = strategy.messages(task)
    assert len(second) == 2 and "Check the reporting period" in second[0]["content"]
    assert load_factory(cfg.validate_factory)(cfg)["chunks"] > 0
    assert load_factory(cfg.tool_factory)(cfg, "123").call("get_schema", {})["ok"]
    with pytest.raises(ValueError, match="inside extension2"):
        load_factory("os:system")


def test_freeze_rejects_dirty_tree_and_changed_resources(config, tmp_path, monkeypatch):
    import extension2_finance_agent.experiments.runner as runner
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "test-commit", "branch": "test", "dirty": True})
    with pytest.raises(ValueError, match="Commit the method"):
        freeze(config, tmp_path / "frozen.json")
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "test-commit", "branch": "test", "dirty": False})
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Initial method.", encoding="utf-8")
    cfg = replace(config, prompt_file=str(prompt))
    frozen = tmp_path / "frozen.json"
    freeze(cfg, frozen)
    prompt.write_text("Changed method.", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen"):
        run(cfg, "test", tmp_path / "formal_run", frozen=frozen)


def test_config_validation_and_unknown_questions(config, tmp_path):
    path = tmp_path / "invalid.json"
    write_json(path, {"max_turns": 0})
    with pytest.raises(ValueError, match="positive integer"):
        Config.load(path)
    with pytest.raises(ValueError, match="Unknown question"):
        load_tasks(config, "train", question_ids=["missing"])
    with pytest.raises(ValueError, match="limit must be positive"):
        load_tasks(config, "train", limit=0)
