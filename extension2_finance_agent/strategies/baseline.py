"""The stable strategy interface: create_strategy(config).messages(task)."""
import json
from extension2_finance_agent.core.config import local_path


class BaselineStrategy:
    def __init__(self, config):
        prompt = local_path(config.prompt_file).read_text(encoding="utf-8")
        rules = (config.workspace / "data/business_rules.md").read_text(encoding="utf-8")
        skills = [local_path(path).read_text(encoding="utf-8") for path in config.skill_files]
        self.system = "\n\n".join([prompt, "Public data rules:\n" + rules, *skills])

    def messages(self, task):
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]


def create_strategy(config):
    return BaselineStrategy(config)
