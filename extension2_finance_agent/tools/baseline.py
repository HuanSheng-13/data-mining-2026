"""Read-only data tools. No arbitrary file, shell, URL or answer access."""
import ast
import collections
from contextlib import closing
import math
import operator
import re
import sqlite3
import time

from extension2_finance_agent.core.config import read_json


def definition(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}}


DEFINITIONS = [
    definition("get_schema", "Discover database tables, columns and available annual reports for this company.", {}),
    definition("discover_metrics", "Find fact names for the current company; keyword is a literal substring, not SQL.",
               {"keyword": {"type": "string"}}, ["keyword"]),
    definition("query_sql", "Run a read-only SELECT/WITH query; use supplied CIK, inspect periods/units and retain fact IDs for citations. At most 100 rows.",
               {"sql": {"type": "string"}, "params": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]}}}, ["sql", "params"]),
    definition("search_reports", "Keyword search of bundled annual reports for the current company. Optional exact report_end YYYY-MM-DD. Returns cited excerpts with positions in extracted text.",
               {"query": {"type": "string"}, "report_end": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 8}}, ["query"]),
    definition("calculate", "Evaluate numeric arithmetic (+ - * / and parentheses). Example YoY: (120 / 100 - 1) * 100. No code execution.",
               {"expression": {"type": "string"}}, ["expression"]),
]


def arithmetic(expression):
    if not isinstance(expression, str) or not 0 < len(expression) <= 500:
        raise ValueError("Expression must be 1 to 500 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Expression too complex")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError("Only numeric + - * / and parentheses are allowed")
        if not math.isfinite(value):
            raise ValueError("Non-finite arithmetic result")
        return value
    return visit(tree.body)


class BaselineTools:
    definitions = DEFINITIONS

    def __init__(self, config, company_cik):
        self.config = config
        self.company_cik = company_cik
        self.database = config.workspace / "data/finance.db"
        self.chunks = read_json(config.artifacts / "chunks.json")
        self.reports = read_json(config.artifacts / "manifest.json")["reports"]
        if company_cik not in {company["cik"] for company in read_json(config.workspace / "data/companies.json")}:
            raise ValueError("Unknown company CIK")

    def connect(self):
        connection = sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 65536)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 12000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 100)
        return closing(connection)

    def call(self, name, arguments):
        functions = {"get_schema": self.get_schema, "discover_metrics": self.discover_metrics,
                     "query_sql": self.query_sql, "search_reports": self.search_reports,
                     "calculate": self.calculate}
        if name not in functions:
            return {"ok": False, "data": None, "sources": [], "error": "Unknown tool"}
        try:
            data, sources = functions[name](**arguments)
            return {"ok": True, "data": data, "sources": sources, "error": None}
        except (sqlite3.Error, ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return {"ok": False, "data": None, "sources": [], "error": f"{type(exc).__name__}: {exc}"}

    def get_schema(self):
        with self.connect() as connection:
            rows = connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        return {"tables": [{"name": row[0], "sql": row[1]} for row in rows], "company_cik": self.company_cik,
                "reports": [r for r in self.reports if r["company_cik"] == self.company_cik]}, []

    def discover_metrics(self, keyword):
        if not isinstance(keyword, str) or len(keyword) > 200:
            raise ValueError("Keyword must be a string of at most 200 characters")
        # Escape LIKE wildcards, so this is truly substring search.
        keyword = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.connect() as connection:
            cursor = connection.execute("SELECT fact_name, fact_category, unit, COUNT(*) AS disclosures "
                                        "FROM financial_facts WHERE cik=? AND fact_name LIKE ? ESCAPE '\\' "
                                        "GROUP BY fact_name, fact_category, unit ORDER BY fact_name LIMIT 101",
                                        (self.company_cik, "%" + keyword + "%"))
            rows = [dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()]
        return {"metrics": rows[:100], "truncated": len(rows) > 100}, []

    def query_sql(self, sql, params):
        if not isinstance(sql, str) or len(sql) > 12000 or not isinstance(params, list):
            raise ValueError("Expected SQL string (<=12000 chars) and params array")
        if not re.match(r"\s*(SELECT|WITH)\b", sql, flags=re.I):
            raise ValueError("Only SELECT or WITH queries are allowed")
        if any(type(p) not in (str, int, float, type(None)) or (type(p) is float and not math.isfinite(p)) for p in params):
            raise ValueError("Parameters must be finite numbers, strings or null")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

        def authorize(action, arg1, arg2, database, source):
            if action not in allowed:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ("load_extension", "readfile", "writefile"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        deadline = time.monotonic() + 3
        steps = 0

        def progress():
            nonlocal steps
            steps += 1
            return int(steps > 2000 or time.monotonic() > deadline)

        with self.connect() as connection:
            connection.set_authorizer(authorize)
            connection.set_progress_handler(progress, 1000)
            cursor = connection.execute(sql, params)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(101)
        if any(isinstance(v, bytes) or (isinstance(v, float) and not math.isfinite(v)) for row in rows for v in row):
            raise ValueError("Binary and non-finite values are not supported")
        data = {"columns": columns, "rows": [list(row) for row in rows[:100]], "truncated": len(rows) > 100}
        # Empty results do not count as retrieved evidence.
        sources = [{"database": "data/finance.db", "sql": sql, "params": params}] if rows else []
        return data, sources

    def search_reports(self, query, report_end="", top_k=5):
        if not isinstance(query, str) or not 0 < len(query) <= 1000:
            raise ValueError("Query must be 1 to 1000 characters")
        if not isinstance(report_end, str) or (report_end and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", report_end)):
            raise ValueError("report_end must be empty or YYYY-MM-DD")
        if type(top_k) is not int or not 1 <= top_k <= 8:
            raise ValueError("top_k must be between 1 and 8")
        stop = {"the", "and", "for", "was", "what", "how", "company", "its", "are", "did", "with", "from", "this"}
        terms = set(re.findall(r"[a-z0-9]{2,}", query.lower())) - stop
        ranked = []
        for chunk in self.chunks:
            if chunk["company_cik"] != self.company_cik or (report_end and chunk["report_end"] != report_end):
                continue
            tokens = collections.Counter(re.findall(r"[a-z0-9]{2,}", chunk["text"].lower()))
            score = sum(1 + min(tokens[t], 5) / 10 for t in terms if tokens[t])
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["char_start"]))
        hits = [{**chunk, "score": score} for score, chunk in ranked[:top_k]]
        sources = [{key: h[key] for key in ("chunk_id", "source", "report_end", "char_start", "char_end")} for h in hits]
        return {"hits": hits, "location_basis": "character offsets in normalized extracted text, not raw HTML"}, sources

    def calculate(self, expression):
        return {"expression": expression, "value": arithmetic(expression)}, []


def create_tools(config, company_cik):
    return BaselineTools(config, company_cik)
