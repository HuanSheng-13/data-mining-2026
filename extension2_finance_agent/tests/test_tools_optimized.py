"""Regression checks for B's independent preparation and compatible tools."""

from contextlib import closing
from dataclasses import replace
import json
import sqlite3

import pytest

from extension2_finance_agent.core.config import Config, sha256
from extension2_finance_agent.prepare.optimized import (
    INDEX,
    prepare,
    validate_artifacts,
)
from extension2_finance_agent.tools.optimized import create_tools


HTML = (
    "<html><head><title>private marker</title></head><body>"
    '<div id="business">ITEM 1. BUSINESS</div><div>Regional deployments expanded.</div>'
    "<ix:header><div>invisible data</div></ix:header>"
    '<table><tr><th colspan="2">Year ended December 31</th></tr>'
    "<tr><th></th><th>2024</th><th>2023</th></tr>"
    "<tr><td>Railway operating ratio</td><td>65.8%</td><td>67.4%</td></tr></table></body></html>"
)


@pytest.fixture
def fixture_config(tmp_path):
    config = Config(
        data_root=str(tmp_path / "raw"), artifacts_dir=str(tmp_path / "derived")
    )
    data = config.workspace / "data"
    filings = data / "filings"
    filings.mkdir(parents=True)
    catalog = []
    for cik, year in [("123", "2024"), ("123", "2023"), ("124", "2024")]:
        name = f"{cik}-{year}.html"
        (filings / name).write_text(HTML, encoding="utf-8")
        catalog.append(f"| Company {cik} | {cik} | {year}-12-31 | [{name}]({name}) |")
    (filings / "README.md").write_text("\n".join(catalog), encoding="utf-8")
    for name in ("schema.md", "business_rules.md"):
        (data / name).write_text("Public input", encoding="utf-8")
    (data / "companies.json").write_text(
        json.dumps([{"cik": "123"}, {"cik": "124"}]), encoding="utf-8"
    )
    with closing(sqlite3.connect(data / "finance.db")) as db:
        db.executescript(
            """
            CREATE TABLE financial_facts (
                id INTEGER PRIMARY KEY, cik TEXT, fact_name TEXT, fact_category TEXT, unit TEXT,
                fact_value REAL, end_date TEXT, accession_number TEXT
            );
            INSERT INTO financial_facts VALUES
              (1,'123','Revenues','us-gaap','USD',42,'2024-12-31','acc-1'),
              (2,'123','Revenues','us-gaap','USD',NULL,'2024-12-31','acc-2'),
              (3,'124','SalesRevenueNet','us-gaap','USD',12,'2024-12-31','other');
        """
        )
    assert not (config.workspace / "train").exists()
    return config


def test_report_search_keeps_table_context_and_resolves_source_span(fixture_config):
    prepare(fixture_config)
    registry = create_tools(fixture_config, "123")
    result = registry.call(
        "search_reports",
        {"query": "railway operating ratio", "report_end": "2024-12-31"},
    )
    assert result["ok"]
    hits = result["data"]["hits"]
    assert {(hit["company_cik"], hit["report_end"]) for hit in hits} == {
        ("123", "2024-12-31")
    }
    first = hits[0]
    assert first["kind"] == "table_row"
    assert first["section"] == "ITEM 1. BUSINESS"
    assert first["anchor"] == "business"
    assert first["headers"][0]["cells"][0]["colspan"] == 2
    assert [cell["text"] for cell in first["cells"]] == [
        "Railway operating ratio",
        "65.8%",
        "67.4%",
    ]
    raw = HTML[first["html_start"] : first["html_end"]]
    assert "65.8%" in raw and "67.4%" in raw
    assert result["sources"][0]["chunk_id"] == first["chunk_id"]
    hidden = registry.call("search_reports", {"query": "invisible"})
    assert hidden["ok"] and hidden["sources"] == []
    assert registry.call("search_reports", {"query": 'unknown OR "*"'})["ok"]
    assert not registry.call(
        "search_reports", {"query": "railway", "report_end": "2024-02-30"}
    )["ok"]


def test_metric_discovery_and_provenance_preserve_source_rows(fixture_config):
    prepare(fixture_config)
    registry = create_tools(fixture_config, "123")
    result = registry.call("discover_metrics", {"keyword": "sales revenue"})
    (first,) = result["data"]["metrics"]
    assert result["sources"] == []
    assert first["fact_name"] == "Revenues" and first["null_values"] == 1
    assert first["disclosures"] == 2 and first["distinct_end_dates"] == 1
    sql = (
        "SELECT id AS fact_id, fact_name, fact_value, unit, end_date, accession_number "
        "FROM financial_facts WHERE cik=? ORDER BY id"
    )
    query = registry.call("query_sql", {"sql": sql, "params": ["123"]})
    assert query["data"]["rows"][1] == [
        2,
        "Revenues",
        None,
        "USD",
        "2024-12-31",
        "acc-2",
    ]
    assert query["data"]["field_guide"]["end_date"]
    provenance = query["sources"][0]["projected_provenance"]
    assert [
        row[provenance["columns"].index("fact_id")] for row in provenance["rows"]
    ] == [1, 2]
    assert registry.call("query_sql", {"sql": sql, "params": ["999"]})["sources"] == []
    denied = registry.call(
        "query_sql", {"sql": "DELETE FROM financial_facts", "params": []}
    )
    assert not denied["ok"]


def test_build_is_repeatable_invalidates_changes_and_does_not_modify_raw(
    fixture_config,
):
    config = fixture_config
    database = config.workspace / "data/finance.db"
    original = sha256(database)
    first = prepare(config)
    assert prepare(config) == validate_artifacts(config) == first
    assert sha256(database) == original
    with pytest.raises(ValueError, match="outside"):
        prepare(replace(config, artifacts_dir=str(config.data / "unsafe")))
    with (config.artifacts / INDEX).open("ab") as output:
        output.write(b"modified")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_artifacts(config)
    prepare(config)
    (config.workspace / "data/business_rules.md").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source inputs"):
        validate_artifacts(config)
