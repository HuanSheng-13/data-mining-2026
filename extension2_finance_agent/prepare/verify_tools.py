"""Repeatable B handoff: compare real train tools without a model or official scoring."""

import argparse
from html import unescape
import json
import re
import sys
from time import perf_counter

from extension2_finance_agent.core.config import (
    Config,
    load_factory,
    local_path,
    read_json,
    write_json,
)
from extension2_finance_agent.prepare.optimized import read_only


CONFIG_DIR = "extension2_finance_agent/configs/"
CASES = (
    (
        "FQA-1640147-02",
        "2025-01-31",
        "cloud geographic regions",
        ("47 regional deployments",),
    ),
    (
        "FQA-1640147-06",
        "2025-01-31",
        "product technology initiatives Cortex Snowpark",
        ("snowpark", "cortex ai"),
    ),
    (
        "FQA-702165-05",
        "2024-12-31",
        "railway operating ratio adjusted operating ratio",
        ("railway operating ratio", "65.8", "67.4"),
    ),
)


def checked_call(registry, tool, arguments):
    started = perf_counter()
    result = registry.call(tool, arguments)
    elapsed = perf_counter() - started
    if not result["ok"]:
        raise ValueError(f'{tool}: {result["error"]}')
    return {"arguments": arguments, "elapsed_seconds": elapsed, "result": result}


def first_evidence(hits, markers):
    return next(
        (
            i
            for i, hit in enumerate(hits, 1)
            if all(marker in hit["text"].casefold() for marker in markers)
        ),
        None,
    )


def verify_raw_span(config, hit, markers):
    path = config.workspace / hit["source"]
    raw = path.read_text(encoding="utf-8", errors="replace")
    fragment = raw[hit["html_start"] : hit["html_end"]]
    text = unescape(re.sub(r"<[^>]*>", "", fragment))
    normalized = " ".join(text.split()).casefold()
    if not all(marker in normalized for marker in markers):
        raise ValueError(
            f'Could not locate evidence markers in source span {hit["chunk_id"]}'
        )


def audit(base, optimized):
    if base.workspace != optimized.workspace:
        raise ValueError("Both configurations must use the same FinanceQA workspace")
    configs = {"baseline": base, "data_optimized": optimized}
    manifests = {}
    for name, config in configs.items():
        manifests[name] = load_factory(config.validate_factory)(config)
    train = {
        row["question_id"]: row
        for row in read_json(base.workspace / "train/questions.json")
    }
    retrieval = []
    for question_id, period, query, markers in CASES:
        task = train[question_id]
        checks = {}
        for name, config in configs.items():
            tools = load_factory(config.tool_factory)(config, task["company_cik"])
            run = checked_call(
                tools,
                "search_reports",
                {"query": query, "report_end": period, "top_k": 5},
            )
            hits = run["result"]["data"]["hits"]
            if any(
                hit["company_cik"] != task["company_cik"] or hit["report_end"] != period
                for hit in hits
            ):
                raise ValueError("Search escaped requested company or report period")
            rank = first_evidence(hits, markers)
            run["first_evidence_rank"] = rank
            if name == "data_optimized" and rank is not None:
                verify_raw_span(config, hits[rank - 1], markers)
                run["source_span_verified"] = True
            checks[name] = run
        retrieval.append({"task": task, "markers": list(markers), "checks": checks})

    task = train["FQA-1640147-05"]
    sql = (
        "SELECT id AS fact_id, cik, fact_name, fact_category, fact_value, unit, end_date, "
        "fiscal_year, fiscal_period, frame, filed_date, form_type, accession_number "
        "FROM financial_facts WHERE cik=? AND fact_name=? AND end_date=? "
        "ORDER BY filed_date DESC, id LIMIT 5"
    )
    params = [
        task["company_cik"],
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "2025-01-31",
    ]
    numeric = {}
    for name, config in configs.items():
        tools = load_factory(config.tool_factory)(config, task["company_cik"])
        discovery = checked_call(
            tools, "discover_metrics", {"keyword": "sales revenue"}
        )
        discovery["candidate_count"] = len(discovery["result"]["data"]["metrics"])
        discovery["result"]["data"]["metrics"] = discovery["result"]["data"]["metrics"][
            :5
        ]
        numeric[name] = {
            "discovery": discovery,
            "sql": checked_call(tools, "query_sql", {"sql": sql, "params": params}),
        }
    with read_only(base.workspace / "data/finance.db") as database:
        original = [list(row) for row in database.execute(sql, params)]
    if not original or any(
        case["sql"]["result"]["data"]["rows"] != original for case in numeric.values()
    ):
        raise ValueError("Returned rows do not match source SQLite results")
    return {
        "kind": "manual_train_tool_comparison_no_model_or_checker",
        "baseline_commit": "8021a1628c48990030d3ae2dfe2a8381102ca0da",
        "configuration": {name: config.snapshot() for name, config in configs.items()},
        "manifests": manifests,
        "retrieval": retrieval,
        "numeric": {"task": task, "source_rows_verified": True, "checks": numeric},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=CONFIG_DIR + "baseline.json")
    parser.add_argument("--optimized", default=CONFIG_DIR + "data_optimized.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, optimized = Config.load(args.baseline), Config.load(args.optimized)
    output = local_path(args.output)
    if output.exists() or any(
        output.is_relative_to(config.data) for config in (base, optimized)
    ):
        parser.error("Choose a new output file outside the original data package")
    try:
        result = audit(base, optimized)
        # Record tool and preparation configurations only. Service settings are irrelevant to this audit.
        for name, config in result["configuration"].items():
            result["configuration"][name] = {
                key: config[key]
                for key in (
                    "data_root",
                    "artifacts_dir",
                    "tool_factory",
                    "prepare_factory",
                    "validate_factory",
                    "strategy_factory",
                    "prompt_file",
                    "skill_files",
                )
            }
        write_json(output, result)
    except (ValueError, OSError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "retrieval_cases": len(result["retrieval"]),
                "numeric_cases": 1,
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
