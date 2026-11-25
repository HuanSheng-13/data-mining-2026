"""Independent process invoking the unchanged teacher evaluator."""
import os
import subprocess
import sys
from extension2_finance_agent.core.client import environment_secret


def evaluate(config, split, predictions, output, check_only=False, judge_url=None, model=None, limit=None):
    if not predictions.is_file():
        raise ValueError("Predictions file not found")
    if output.exists():
        raise ValueError("Use a new evaluation output path")
    if split == "test" and limit is not None:
        raise ValueError("Test evaluation cannot use --limit")
    command = [sys.executable, str(config.data / "evaluation_only/evaluate.py"), "--split", split,
               "--predictions", str(predictions), "--output", str(output)]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    env = os.environ.copy()
    if check_only:
        command.append("--check-only")
    else:
        judge_url = judge_url or environment_secret("FINANCEQA_JUDGE_URL")
        model = model or environment_secret("FINANCEQA_JUDGE_MODEL")
        if not judge_url or not model:
            raise ValueError("Set judge URL/model explicitly; use the same checker configuration across experiments")
        key = environment_secret("FINANCEQA_JUDGE_API_KEY")
        if not key:
            raise ValueError("Missing FINANCEQA_JUDGE_API_KEY")
        env["LLM_API_KEY"] = key
        command.extend(["--judge-url", judge_url, "--model", model])
    return subprocess.run(command, cwd=config.workspace, env=env, check=False).returncode
