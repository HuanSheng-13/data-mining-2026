"""Opt-in B tools. Keep baseline calls and response fields compatible."""

from collections import Counter
from copy import deepcopy
from datetime import date
import json
import re
import sqlite3

from extension2_finance_agent.core.config import read_json
from extension2_finance_agent.prepare.optimized import INDEX, PARAMETERS, read_only
from extension2_finance_agent.tools.baseline import BaselineTools, DEFINITIONS


STOPWORDS = set(
    """
a about an and are as at be by can company could did do does for from had has have how in is it its
major of on or should that the their this to was were what when where which who why will with would
""".split()
)
WORDS = re.compile(r"[a-z0-9]+")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
ALIASES = {
    "sales": (
        "revenues",
        "sales revenue net",
        "revenue from contract with customer excluding assessed tax",
    ),
    "revenue": (
        "revenues",
        "sales revenue net",
        "revenue from contract with customer excluding assessed tax",
    ),
    "sales revenue": (
        "revenues",
        "sales revenue net",
        "revenue from contract with customer excluding assessed tax",
    ),
    "net profit": ("net income loss", "profit loss"),
    "operating profit": ("operating income loss",),
    "operating cash flow": ("net cash provided by used in operating activities",),
    "capital expenditures": ("payments to acquire property plant and equipment",),
    "eps": ("earnings per share basic", "earnings per share diluted"),
}

FIELD_GUIDE = {
    "cik": "Company identifier stored as text. Join filings using both CIK and accession number.",
    "fact_name": "Raw XBRL concept. Related concepts may have different definitions.",
    "fact_category": "Taxonomy namespace. Compare concepts only under compatible definitions.",
    "fact_value": "Value in the returned unit. NULL is missing and never implies zero.",
    "unit": "Original units. USD values are unscaled dollars.",
    "end_date": "Fact period end. It can precede the filing's fiscal year.",
    "fiscal_year": "Fiscal-year label of the disclosure. Verify the fact's actual end_date.",
    "fiscal_period": "Source fiscal-period label. Check period length and frame for flows.",
    "frame": "Supplementary period indicator. A missing frame does not establish annual duration.",
    "filed_date": "Submission date, distinct from the fact period end.",
    "form_type": "Source filing form. One period may be disclosed in several filings.",
    "accession_number": "SEC submission identifier. A matching filings row may be absent.",
    "dimension_segment": "Segment scope if supplied. Absence alone does not prove comparability.",
    "dimension_geography": "Geographic scope if supplied. Absence alone does not prove comparability.",
}

RULES = [
    "Check end_date, fiscal_period, frame and form_type before comparing flows across periods.",
    "Check concept, unit, period length and dimensions before identifying repeats.",
    "Repeated disclosures can reflect revisions. Retain fact IDs and accession numbers and do not sum repeats.",
    "Do not conflate net income, parent-attributable income, operating income and pretax income.",
]
LOCATION = "Offsets are in decoded HTML characters and normalized optimized text, respectively (0-based, end-exclusive)."


def terms(value):
    return [
        word
        for word in WORDS.findall(value.lower())
        if len(word) > 1 and word not in STOPWORDS
    ]


def metric_words(value):
    return " ".join(WORDS.findall(CAMEL_BOUNDARY.sub(" ", value).lower()))


class OptimizedTools(BaselineTools):
    definitions = deepcopy(DEFINITIONS)
    definitions[1]["function"]["description"] = (
        "Discover actual XBRL tags for this company using a keyword, plain-language concept or exact tag. "
        "Returns taxonomy, units and disclosure-period coverage. Candidates are not interchangeable."
    )
    definitions[3]["function"]["description"] = (
        "Search annual-report evidence for this company with optional exact report_end YYYY-MM-DD. "
        "Returns bounded excerpts with section, original file, source offsets and any table context."
    )

    def __init__(self, config, company_cik):
        self.config = config
        self.company_cik = company_cik
        self.database = config.workspace / "data/finance.db"
        self.index = config.artifacts / INDEX
        manifest = read_json(config.artifacts / "manifest.json")
        if manifest.get("parameters") != PARAMETERS or not self.index.is_file():
            raise ValueError("Run prepare with the data_optimized configuration first")
        self.reports = manifest["reports"]
        if company_cik not in {
            item["cik"] for item in read_json(config.workspace / "data/companies.json")
        }:
            raise ValueError("Unknown company CIK")

    def get_schema(self):
        data, sources = super().get_schema()
        data["field_guide"] = FIELD_GUIDE
        data["interpretation_notes"] = RULES
        return data, sources

    def discover_metrics(self, keyword):
        if not isinstance(keyword, str) or len(keyword) > 200:
            raise ValueError("Keyword must be a string of at most 200 characters")
        keyword = keyword.strip()
        normalized = metric_words(keyword)
        if keyword and not normalized:
            return {"metrics": [], "truncated": False}, []
        query_terms = set(terms(normalized))
        ranked = []
        with read_only(self.index) as index:
            index.row_factory = sqlite3.Row
            for row in index.execute(
                "SELECT * FROM metrics WHERE cik=?", (self.company_cik,)
            ):
                name = row["fact_name"]
                words = metric_words(name)
                compact = words.replace(" ", "")
                overlap = query_terms & set(terms(words))
                if not keyword:
                    match, priority = "browse", 0
                elif normalized == words or keyword.casefold() == name.casefold():
                    match, priority = "exact", 5
                elif words in ALIASES.get(normalized, ()):
                    match, priority = "alias_candidate", 4
                elif normalized.replace(" ", "") in compact:
                    match, priority = "substring", 3
                elif query_terms and overlap == query_terms:
                    match, priority = "all_terms", 2
                elif overlap:
                    match, priority = "partial_terms", 1
                else:
                    continue
                entry = dict(row)
                entry["match"] = match
                ranked.append(
                    (
                        -priority,
                        -len(overlap),
                        name,
                        row["fact_category"],
                        row["unit"],
                        entry,
                    )
                )
        ranked.sort(key=lambda item: item[:-1])
        return {
            "metrics": [item[-1] for item in ranked[:100]],
            "truncated": len(ranked) > 100,
            "coverage_basis": "Raw disclosures, including repeats. Date range is end_date, not verified fiscal coverage.",
            "notes": RULES,
        }, []

    def query_sql(self, sql, params):
        data, sources = super().query_sql(sql, params)
        columns = [name.lower() for name in data["columns"]]
        counts = Counter(columns)
        unique = {name: i for i, name in enumerate(columns) if counts[name] == 1}
        data["field_guide"] = {
            field: FIELD_GUIDE[field] for field in unique if field in FIELD_GUIDE
        }
        if sources:
            fields = [
                key
                for key in (
                    "cik",
                    "fact_name",
                    "fact_category",
                    "unit",
                    "end_date",
                    "fiscal_year",
                    "fiscal_period",
                    "frame",
                    "filed_date",
                    "accession_number",
                    "form_type",
                    "dimension_segment",
                    "dimension_geography",
                )
                if key in unique
            ]
            id_key = (
                "fact_id"
                if "fact_id" in unique
                else ("id" if "fact_name" in unique else "")
            )
            if id_key and id_key in unique:
                fields.append(id_key)
            if fields:
                sources[0]["projected_provenance"] = {
                    "columns": fields,
                    "rows": [
                        [row[unique[key]] for key in fields]
                        for row in data["rows"][:20]
                    ],
                    "truncated": len(data["rows"]) > 20 or data["truncated"],
                    "basis": "Values projected by the supplied SELECT, not inferred source lineage.",
                }
            if {"fact_value", "fact_name"} & set(columns):
                data["interpretation_notes"] = RULES
        return data, sources

    def search_reports(self, query, report_end="", top_k=5):
        if not isinstance(query, str) or not 0 < len(query) <= 1000:
            raise ValueError("Query must be 1 to 1000 characters")
        if not isinstance(report_end, str) or (
            report_end
            and (
                not re.fullmatch(r"\d{4}-\d{2}-\d{2}", report_end)
                or not date.fromisoformat(report_end)
            )
        ):
            raise ValueError("report_end must be empty or YYYY-MM-DD")
        if type(top_k) is not int or not 1 <= top_k <= 8:
            raise ValueError("top_k must be between 1 and 8")
        tokens = list(dict.fromkeys(terms(query)))[: PARAMETERS["max_query_terms"]]
        hits = []
        if tokens:
            match = " OR ".join(f'"{word}"' for word in tokens)
            filters = " AND e.report_end = ?" if report_end else ""
            params = [*PARAMETERS["bm25_weights"], match, self.company_cik]
            if report_end:
                params.append(report_end)
            params.append(top_k * 8)
            with read_only(self.index) as index:
                candidates = index.execute(
                    "SELECT e.payload, bm25(passages, ?, ?, ?) AS rank FROM passages "
                    "JOIN evidence e ON e.id=passages.rowid WHERE passages MATCH ? AND e.cik=?"
                    + filters
                    + " ORDER BY rank, e.report_end DESC, e.id LIMIT ?",
                    params,
                ).fetchall()
            seen = set()
            for payload, rank in candidates:
                item = json.loads(payload)
                key = (item["report_end"], item["text"].casefold())
                if key in seen:
                    continue
                seen.add(key)
                item["score"] = -rank
                if len(json.dumps(item.get("cells", []))) > 3200:
                    item.pop("cells", None)
                    item["cells_omitted"] = True
                if len(json.dumps(item.get("headers", []))) > 2400:
                    item.pop("headers", None)
                    item["headers_omitted"] = True
                hits.append(item)
                if len(hits) == top_k:
                    break
        location_fields = (
            "chunk_id",
            "source",
            "company_cik",
            "report_end",
            "section",
            "anchor",
            "char_start",
            "char_end",
            "html_start",
            "html_end",
            "table_start",
            "row_number",
        )
        sources = [
            {key: item[key] for key in location_fields if key in item} for item in hits
        ]
        return {"hits": hits, "location_basis": LOCATION}, sources


def create_tools(config, company_cik):
    return OptimizedTools(config, company_cik)
