"""Read project-local .env without mutating the process environment."""
import os
from pathlib import Path
import re

DOTENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def read_dotenv(path=None):
    """A literal, single-line KEY=VALUE subset; no expansion or execution."""
    path = Path(path) if path is not None else DOTENV_FILE
    if not path.exists():
        return {}
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        key, separator, value = text.partition("=")
        key, value = key.strip(), value.strip()
        error = f"Invalid .env syntax on line {number}; expected single-line KEY=VALUE"
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(error)
        if value.startswith(("'", '"')):
            quote = value[0]
            end = value.find(quote, 1)
            if end == -1 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#")):
                raise ValueError(error)
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            if value.startswith("#"):
                value = ""
        values[key] = value
    return values


def user_environment_value(name):
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
                return value if isinstance(value, str) and value else None
        except OSError:
            pass
    return None


def environment_value(name):
    """Nonempty process env > fixed project .env > Windows user env."""
    return os.environ.get(name) or read_dotenv().get(name) or user_environment_value(name)
