"""Recover model-emitted DSML requests; never accept markup as a final answer."""
from html import unescape
import json
import re

MARKUP = re.compile(r"<\s*/?\s*(?:｜｜DSML｜｜|tool_call\b|function_calls\b)", re.I)


def has_tool_markup(text):
    return isinstance(text, str) and bool(MARKUP.search(text))


def recover_dsml(text, allowed_names, turn):
    """Only requests emitted by the assistant, bound to registered tool names."""
    normalized = re.sub(r"<(/?)｜｜DSML｜｜\s+", r"<\1", text)
    blocks = re.findall(r'<invoke\s+name="([A-Za-z_][A-Za-z0-9_]*)"\s*>(.*?)</invoke\s*>', normalized, re.S)
    if not blocks:
        raise ValueError("No recoverable DSML calls")
    calls = []
    for index, (name, body) in enumerate(blocks):
        if name not in allowed_names:
            raise ValueError("Unregistered DSML tool")
        params = re.findall(r'<parameter\s+name="([A-Za-z_][A-Za-z0-9_]*)"\s+string="(true|false)"\s*>(.*?)</parameter\s*>', body, re.S)
        remainder = re.sub(r'<parameter\s+name="[A-Za-z_][A-Za-z0-9_]*"\s+string="(?:true|false)"\s*>.*?</parameter\s*>', '', body, flags=re.S).strip()
        if remainder:
            raise ValueError("Malformed DSML parameters")
        arguments = {}
        for key, is_string, value in params:
            if key in arguments:
                raise ValueError("Duplicate DSML parameter")
            value = unescape(value)
            arguments[key] = value if is_string == "true" else json.loads(value)
        calls.append({"id": f"recovered_{turn}_{index}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    return calls
