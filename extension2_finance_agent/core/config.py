"""Configuration and reproducible paths; never load reference answers."""
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from extension2_finance_agent.core.env import environment_value

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "extension2_finance_agent"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_factory(name):
    module, attribute = name.split(":", 1)
    if not module.startswith("extension2_finance_agent."):
        raise ValueError("Factories must be inside extension2_finance_agent")
    return getattr(importlib.import_module(module), attribute)


@dataclass(frozen=True)
class Config:
    data_root: str = "data/financeqa/FinanceQA"
    artifacts_dir: str = "extension2_finance_agent/artifacts/final"
    chat_url: str = "https://api.deepseek.com/chat/completions"
    model: str = "deepseek-flash"
    api_key_env: str = "FINANCEQA_API_KEY"
    temperature: float = 0
    max_tokens: int = 3000
    max_turns: int = 16
    max_tool_calls: int = 24
    timeout_seconds: float = 90
    retries: int = 2
    tool_factory: str = "extension2_finance_agent.tools.data:create_tools"
    prepare_factory: str = "extension2_finance_agent.prepare.reports:prepare"
    validate_factory: str = "extension2_finance_agent.prepare.reports:validate_artifacts"
    strategy_factory: str = "extension2_finance_agent.strategies.finance:create_strategy"
    prompt_file: str = "extension2_finance_agent/prompts/agent.txt"
    skill_files: tuple = ("extension2_finance_agent/skills/financial_basis.txt", "extension2_finance_agent/skills/calculation_and_claims.txt", "extension2_finance_agent/skills/recovery_and_stop.txt")

    @property
    def data(self):
        return local_path(self.data_root)

    @property
    def workspace(self):
        return self.data / "agent_workspace"

    @property
    def artifacts(self):
        return local_path(self.artifacts_dir)

    def snapshot(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        values = read_json(local_path(path))
        for field, env in (("chat_url", "FINANCEQA_CHAT_URL"), ("model", "FINANCEQA_MODEL")):
            value = environment_value(env)
            if value:
                values[field] = value
        config = cls(**values)
        if config.artifacts.is_relative_to(config.data):
            raise ValueError("artifacts_dir must be outside the original data package")
        for field in ("max_tokens", "max_turns", "max_tool_calls"):
            if type(getattr(config, field)) is not int or getattr(config, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if config.max_turns < 3:
            raise ValueError("max_turns must reserve a tool phase and two finalization turns")
        if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if type(config.retries) is not int or not 0 <= config.retries <= 5:
            raise ValueError("retries must be an integer between 0 and 5")
        if not math.isfinite(config.temperature) or not 0 <= config.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return config
