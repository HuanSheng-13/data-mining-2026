from dataclasses import replace
import json
import sqlite3
import pytest
from extension2_finance_agent.core.config import Config, write_json, sha256
from extension2_finance_agent.prepare.reports import prepare

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
