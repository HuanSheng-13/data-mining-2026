"""Extract searchable paragraphs and table rows with offsets in the local HTML."""

from html.parser import HTMLParser
import re


BLOCKS = {"div", "p", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6"}
HIDDEN = {"head", "script", "style", "ix:hidden", "ix:header"}
ITEM = re.compile(r"^(?:part\s+[ivx]+\b|item\s+\d{1,2}[a-z]?\s*[.\-–:]?)", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
MAX_CHARS = 1300


def clean(parts):
    return " ".join("".join(parts).split())


class StructuredReport(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.line_starts = [0, *(m.end() for m in re.finditer("\n", html))]
        self.elements = []
        self.pending = []
        self.start = 0
        self.section = ""
        self.section_anchor = ""
        self.pending_anchor = ""
        self.heading = False
        self.hidden = 0
        self.tables = []
        self.feed(html)
        self.close()
        self.flush(len(html))

    def html_position(self):
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def flush(self, end):
        value = clean(self.pending)
        self.pending.clear()
        if not value:
            return
        if (self.heading or ITEM.match(value)) and len(value) <= 160:
            self.section = value
            self.section_anchor = self.pending_anchor
        self.elements.append(
            {
                "text": value,
                "section": self.section,
                "anchor": self.section_anchor,
                "html_start": self.start,
                "html_end": end,
                "kind": "paragraph",
            }
        )
        self.pending_anchor = ""

    def handle_starttag(self, tag, attrs):
        if tag in HIDDEN:
            self.hidden += 1
        if self.hidden:
            return
        end = self.html_position()
        if tag == "table":
            self.flush(end)
            self.tables.append(
                {
                    "rows": [],
                    "cells": None,
                    "parts": None,
                    "row_start": None,
                    "table_start": end,
                    "section": self.section,
                    "anchor": self.section_anchor,
                }
            )
            return
        if self.tables:
            table = self.tables[-1]
            if tag == "tr":
                table.update(cells=[], parts=None, row_start=end)
            elif tag in {"td", "th"} and table["cells"] is not None:
                table.update(parts=[], cell_attrs=dict(attrs), cell_tag=tag)
            elif tag in BLOCKS and table["parts"] is not None:
                table["parts"].append(" ")
            return
        if tag in BLOCKS:
            self.flush(end)
            self.heading = tag.startswith("h")
        attributes = dict(attrs)
        anchor = attributes.get("id") or (attributes.get("name") if tag == "a" else "")
        if anchor:
            self.pending_anchor = anchor

    def handle_endtag(self, tag):
        if tag in HIDDEN:
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        end = self.html_position()
        if self.tables:
            table = self.tables[-1]
            if tag in {"td", "th"} and table["parts"] is not None:
                cell = {
                    "text": clean(table["parts"]),
                    "header": table["cell_tag"] == "th",
                }
                for field in ("colspan", "rowspan"):
                    value = table["cell_attrs"].get(field, "1") or "1"
                    cell[field] = (
                        int(value)
                        if value.isascii() and value.isdecimal() and len(value) < 5
                        else 1
                    )
                table["cells"].append(cell)
                table["parts"] = None
            elif tag == "tr" and table["cells"]:
                if any(cell["text"] for cell in table["cells"]):
                    table["rows"].append(
                        {
                            "cells": table["cells"],
                            "html_start": table["row_start"],
                            "html_end": end,
                        }
                    )
                table.update(cells=None, parts=None)
            elif tag == "table":
                self.tables.pop()
                if table["rows"]:
                    self.elements.append({"kind": "table", **table})
            return
        if tag in BLOCKS:
            self.flush(end)
            self.heading = False

    def handle_data(self, value):
        if self.hidden:
            return
        if self.tables:
            parts = self.tables[-1]["parts"]
            if parts is not None:
                parts.append(value)
        else:
            if not self.pending:
                if not value.strip():
                    return
                self.start = self.html_position()
            self.pending.append(value)


def row_text(row):
    return " | ".join(cell["text"] for cell in row["cells"] if cell["text"])


def split_text(value):
    while len(value) > MAX_CHARS:
        cut = value.rfind(" ", MAX_CHARS // 2, MAX_CHARS + 1)
        cut = cut if cut != -1 else MAX_CHARS
        yield value[:cut]
        value = value[cut:].lstrip()
    if value:
        yield value


def evidence_chunks(html):
    """Yield bounded blocks; each table row carries nearby original column headers."""
    parser = StructuredReport(html)
    offset = 0
    for block in sorted(
        parser.elements,
        key=lambda item: item.get("table_start", item.get("html_start")),
    ):
        if block["kind"] == "paragraph":
            rows = [(block["text"], block["html_start"], block["html_end"], {})]
        else:
            headers = []
            rows = []
            # Preserve original cell order and colspan/rowspan, including empty cells.
            for index, row in enumerate(block["rows"], 1):
                value = row_text(row)
                cells = [cell["text"] for cell in row["cells"] if cell["text"]]
                dates = len(YEAR.findall(value)) >= 2 and all(
                    YEAR.search(cell) or not re.search(r"\d", cell) for cell in cells
                )
                if len(value) < 300 and (
                    any(c["header"] for c in row["cells"]) or dates or index == 1
                ):
                    headers.append(row)
                    headers = headers[-3:]
                rows.append(
                    (
                        value,
                        row["html_start"],
                        row["html_end"],
                        {
                            "table_start": block["table_start"],
                            "row_number": index,
                            "cells": row["cells"],
                            "headers": [h for h in headers if h is not row],
                        },
                    )
                )
        for value, start, end, structure in rows:
            for excerpt in split_text(value):
                yield {
                    "text": excerpt,
                    "section": block["section"],
                    "anchor": block["anchor"],
                    "kind": "table_row" if structure else "paragraph",
                    **structure,
                    "html_start": start,
                    "html_end": end,
                    "char_start": offset,
                    "char_end": offset + len(excerpt),
                }
                offset += len(excerpt) + 1
