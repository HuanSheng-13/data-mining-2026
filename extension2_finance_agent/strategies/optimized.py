"""C 的参数化指导：仅生成初始消息，行动仍由模型按工具结果决定。"""
import json
from string import Template

from extension2_finance_agent.core.config import local_path


class FinancialSkillStrategy:
    def __init__(self, config):
        self.config = config
        self.prompt = local_path(config.prompt_file).read_text(encoding="utf-8")
        self.rules = (config.workspace / "data/business_rules.md").read_text(encoding="utf-8")
        self.skills = tuple(Template(local_path(p).read_text(encoding="utf-8"))
                            for p in config.skill_files)

    def messages(self, task):
        # 只有已校验的 CIK 和配置预算进入系统指导，题目原文仍放 user 消息。
        # 不读取题号映射、参考答案或上题历史；每次返回全新的列表与字典。
        cik = task["company_cik"]
        if not isinstance(cik, str) or not cik.isascii() or not cik.isdigit():
            raise ValueError("company_cik must be an ASCII digit string")
        bindings = {"company_cik": cik, "max_turns": str(self.config.max_turns),
                    "max_tool_calls": str(self.config.max_tool_calls)}
        sections = [self.prompt, "Public data rules:\n" + self.rules]
        sections.extend(skill.substitute(bindings) for skill in self.skills)
        return [{"role": "system", "content": "\n\n".join(sections)},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]


def create_strategy(config):
    return FinancialSkillStrategy(config)
