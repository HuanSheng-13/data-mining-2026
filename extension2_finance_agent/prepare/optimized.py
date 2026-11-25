"""Build the data-tools index using only the bundled public source documents."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from extension2_finance_agent.core.config import read_json, sha256, write_json
from extension2_finance_agent.prepare.reports import report_catalog
from extension2_finance_agent.prepare.structured_html import (
    MAX_CHARS,
    evidence_chunks,
    row_text,
)


INDEX = "evidence.sqlite"
PARAMETERS = {
    "version": "finance-evidence-v1",
    "max_excerpt_chars": MAX_CHARS,
    "tokenizer": "porter unicode61",
    "bm25_weights": [1.0, 0.25, 0.45],
    "max_query_terms": 24,
    "period_filter": "exact report_end when supplied",
}

METRIC_CATALOG_SQL = """
    SELECT cik, fact_name, fact_category, unit, COUNT(*) AS disclosures,
           COUNT(*) - COUNT(fact_value) AS null_values,
           COUNT(DISTINCT end_date) AS distinct_end_dates,
           MIN(end_date) AS first_end_date, MAX(end_date) AS last_end_date
    FROM financial_facts
    GROUP BY cik, fact_name, fact_category, unit
    ORDER BY cik, fact_name, fact_category, unit
"""


def read_only(path):
    return closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True))


def input_manifest(config):
    root = config.workspace / "data"
    paths = [
        "finance.db",
        "schema.md",
        "business_rules.md",
        "companies.json",
        "filings/README.md",
    ]
    return {
        "parser_version": PARAMETERS["version"],
        "parameters": PARAMETERS,
        "reports": report_catalog(config),
        "inputs": {name: sha256(root / name) for name in paths},
    }


def prepare(config):
    if config.artifacts.is_relative_to(config.data):
        raise ValueError("artifacts_dir must be outside the original data package")
    manifest = input_manifest(config)
    config.artifacts.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="build-", dir=config.artifacts) as directory:
        path = Path(directory) / INDEX
        with closing(sqlite3.connect(path)) as index:
            index.executescript(
                """
                CREATE TABLE evidence (id INTEGER PRIMARY KEY, cik TEXT, report_end TEXT,
                                       chunk_id TEXT UNIQUE, payload TEXT);
                CREATE INDEX evidence_company_end ON evidence (cik, report_end);
                CREATE VIRTUAL TABLE passages USING fts5(body, section, header, tokenize='porter unicode61');
                CREATE TABLE metrics (cik TEXT, fact_name TEXT, fact_category TEXT, unit TEXT,
                    disclosures INTEGER, null_values INTEGER, distinct_end_dates INTEGER,
                    first_end_date TEXT, last_end_date TEXT);
                CREATE INDEX metrics_company ON metrics(cik);
            """
            )
            count = 0
            for report in manifest["reports"]:
                html = (config.workspace / report["source"]).read_text(
                    encoding="utf-8", errors="replace"
                )
                for block in evidence_chunks(html):
                    count += 1
                    chunk_id = f'{report["company_cik"]}/{Path(report["source"]).stem}:{block["char_start"]}'
                    payload = {**report, **block, "chunk_id": chunk_id}
                    index.execute(
                        "INSERT INTO evidence VALUES (?, ?, ?, ?, ?)",
                        (
                            count,
                            report["company_cik"],
                            report["report_end"],
                            chunk_id,
                            json.dumps(payload, ensure_ascii=False, allow_nan=False),
                        ),
                    )
                    header = " ".join(row_text(row) for row in block.get("headers", []))
                    index.execute(
                        "INSERT INTO passages(rowid, body, section, header) VALUES (?, ?, ?, ?)",
                        (count, block["text"], block["section"], header),
                    )
            with read_only(config.workspace / "data/finance.db") as source:
                metrics = source.execute(METRIC_CATALOG_SQL).fetchall()
            index.executemany(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", metrics
            )
            index.commit()
        manifest.update(
            chunks=count, metric_groups=len(metrics), index_sha256=sha256(path)
        )
        path.replace(config.artifacts / INDEX)
        write_json(config.artifacts / "manifest.json", manifest)
    return manifest


def validate_artifacts(config):
    manifest = read_json(config.artifacts / "manifest.json")
    expected = input_manifest(config)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Data tools source inputs or index settings changed; run prepare again"
        )
    if manifest.get("index_sha256") != sha256(config.artifacts / INDEX):
        raise ValueError("Data tools index fingerprint mismatch; run prepare again")
    return manifest
