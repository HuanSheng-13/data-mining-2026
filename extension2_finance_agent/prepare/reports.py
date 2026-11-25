"""Deterministic plain-text chunks, not a semantic retrieval optimization."""
from html.parser import HTMLParser
import re
from pathlib import Path
from extension2_finance_agent.core.config import read_json, sha256, write_json

PARSER_VERSION = "baseline-html-v1"


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "ix:hidden"):
            self.hidden += 1
        if not self.hidden:
            if tag in ("p", "div", "tr", "br", "h1", "h2", "h3", "li"):
                self.parts.append("\n")
            elif tag in ("td", "th"):
                self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "ix:hidden"):
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in ("p", "div", "tr", "h1", "h2", "h3", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        return "\n".join(line for part in "".join(self.parts).splitlines()
                         if (line := re.sub(r"\s+", " ", part).strip()))


def report_catalog(config):
    root = config.workspace / "data/filings"
    catalog = []
    for line in (root / "README.md").read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not cells[1].isdigit():
            continue
        match = re.search(r"\]\(([^)]+)\)", cells[3])
        if not match:
            raise ValueError("Invalid report catalog entry")
        path = (root / match.group(1)).resolve()
        if not path.is_relative_to(root.resolve()) or path.suffix.lower() not in (".htm", ".html"):
            raise ValueError("Catalog must point to local HTML under filings")
        catalog.append({"company": cells[0], "company_cik": cells[1], "report_end": cells[2],
                        "source": "data/filings/" + path.relative_to(root.resolve()).as_posix(),
                        "sha256": sha256(path)})
    if not catalog:
        raise ValueError("No reports in catalog")
    return catalog


def prepare(config):
    config.artifacts.mkdir(parents=True, exist_ok=True)
    catalog = report_catalog(config)
    chunks = []
    for report in catalog:
        parser = ReportParser()
        parser.feed((config.workspace / report["source"]).read_text(encoding="utf-8", errors="replace"))
        text = parser.text()
        for start in range(0, len(text), 1600):
            excerpt = text[start:start + 1800]
            chunks.append({**report, "chunk_id": f'{report["company_cik"]}/{Path(report["source"]).stem}:{start}',
                           "char_start": start, "char_end": start + len(excerpt), "text": excerpt})
    manifest = {"parser_version": PARSER_VERSION, "database_sha256": sha256(config.workspace / "data/finance.db"),
                "reports": catalog, "chunks": len(chunks), "chunk_chars": 1800, "overlap_chars": 200}
    write_json(config.artifacts / "chunks.json", chunks)
    manifest["chunks_sha256"] = sha256(config.artifacts / "chunks.json")
    write_json(config.artifacts / "manifest.json", manifest)
    return manifest


def validate_artifacts(config):
    manifest = read_json(config.artifacts / "manifest.json")
    if manifest["parser_version"] != PARSER_VERSION:
        raise ValueError("Parser version changed; rebuild artifacts")
    if manifest["reports"] != report_catalog(config):
        raise ValueError("Reports changed; rebuild artifacts")
    if manifest["database_sha256"] != sha256(config.workspace / "data/finance.db"):
        raise ValueError("Database changed; rebuild artifacts")
    if manifest["chunks_sha256"] != sha256(config.artifacts / "chunks.json"):
        raise ValueError("Chunks changed; rebuild artifacts")
    return manifest
