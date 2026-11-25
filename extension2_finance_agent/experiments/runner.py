"""Run artifacts and method snapshots, isolated from evaluator materials."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
import subprocess
import time
from pathlib import Path

from extension2_finance_agent.core.agent import answer_question
from extension2_finance_agent.core.client import ChatClient
from extension2_finance_agent.core.config import PACKAGE, ROOT, load_factory, read_json, sha256, write_json


def git_state():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain"))}


def method_snapshot(config):
    files = [p for p in PACKAGE.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".txt", ".md")
             and not set(p.relative_to(PACKAGE).parts) & {"artifacts", "outputs", "__pycache__", ".pytest_cache"}]
    files.extend([config.workspace / "data/business_rules.md", config.workspace / "data/schema.md",
                  config.workspace / "data/companies.json", config.workspace / "data/filings/README.md"])
    from extension2_finance_agent.core.config import local_path
    files.extend(local_path(p) for p in [config.prompt_file, *config.skill_files])
    return {"config": config.snapshot(), "files": {str(p.resolve()): sha256(p) for p in sorted(set(files))},
            "prepared_files": {p.relative_to(config.artifacts).as_posix(): sha256(p)
                               for p in sorted(config.artifacts.rglob("*")) if p.is_file()},
            "artifacts": load_factory(config.validate_factory)(config)}


def freeze(config, output):
    if output.exists():
        raise ValueError("Freeze output already exists")
    state = git_state()
    if state["dirty"]:
        raise ValueError("Commit the method before freezing a formal test run")
    snapshot = method_snapshot(config)
    # Compare configs as serialized JSON, normalizing tuples to arrays.
    write_json(output, {"git": state, "method": snapshot})
    return {"commit": state["commit"], "frozen": str(output)}


def load_tasks(config, split, question_ids=None, limit=None):
    tasks = read_json(config.workspace / split / "questions.json")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Questions must be a nonempty array")
    ids = set()
    for task in tasks:
        if set(task) != {"question_id", "company_cik", "question"} or any(not isinstance(v, str) or not v for v in task.values()):
            raise ValueError("Invalid task input contract")
        if not task["company_cik"].isdigit() or task["question_id"] in ids:
            raise ValueError("Invalid CIK or duplicate question ID")
        ids.add(task["question_id"])
    if question_ids:
        if not set(question_ids) <= ids:
            raise ValueError("Unknown question IDs")
        tasks = [task for task in tasks if task["question_id"] in set(question_ids)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    return tasks


def run(config, split, output, question_ids=None, limit=None, frozen=None):
    if output.exists():
        raise ValueError("Use a new output directory; previous runs are never overwritten")
    started = time.perf_counter()
    snapshot = method_snapshot(config)
    state = git_state()
    if split == "test":
        if question_ids or limit is not None or frozen is None:
            raise ValueError("Formal test requires --frozen and all questions; no subset options")
        saved = read_json(frozen)
        normalized = json.loads(json.dumps(snapshot))
        if state["dirty"] or saved["git"]["commit"] != state["commit"] or saved["method"] != normalized:
            raise ValueError("Method differs from frozen configuration/commit/artifacts")
    tasks = load_tasks(config, split, question_ids, limit)
    client = ChatClient(config)  # Fail before creating a run if credentials are absent.
    create_tools = load_factory(config.tool_factory)
    create_strategy = load_factory(config.strategy_factory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    manifest = {"split": split, "started_utc": datetime.now(timezone.utc).isoformat(), "git": state,
                "python": platform.python_version(), "method": snapshot, "question_ids": [t["question_id"] for t in tasks],
                "status": "running", "questions_sha256": sha256(config.workspace / split / "questions.json"),
                "preflight_seconds": time.perf_counter() - started}
    write_json(output / "run.json", manifest)
    predictions, summaries = [], []
    for task in tasks:
        # Do not use question_id in filenames: the external contract does not promise path-safe IDs.
        log_path = output / "logs" / f"{len(summaries):03d}.jsonl"
        with log_path.open("x", encoding="utf-8") as stream:
            def emit(event):
                stream.write(json.dumps({"question_id": task["question_id"], **event}, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            emit({"event": "task_start", "task": task})
            answer = answer_question(task, config, client, create_tools(config, task["company_cik"]), create_strategy(config), emit)
            emit({"event": "task_end", **asdict(answer)})
        predictions.append({"question_id": task["question_id"], "answer": answer.answer})
        summaries.append({"question_id": task["question_id"], "log": log_path.relative_to(output).as_posix(), **asdict(answer)})
        write_json(output / "predictions.json", predictions)
        write_json(output / "summary.json", summaries)
        print(json.dumps({"question_id": task["question_id"], "status": answer.status,
                          "tool_calls": answer.tool_calls, "seconds": round(answer.elapsed_seconds, 2)}), flush=True)
    manifest.update(status="completed" if all(s["status"] == "completed" for s in summaries) else "completed_with_errors",
                    total_seconds=time.perf_counter() - started)
    write_json(output / "run.json", manifest)
    return manifest
