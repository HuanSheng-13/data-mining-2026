"""Independent teacher evaluator, immutable prediction provenance and safe retries."""
from copy import deepcopy
import importlib.util
import os
import subprocess
import sys

from extension2_finance_agent.core.client import environment_secret
from extension2_finance_agent.core.config import read_json, write_json, sha256


def evaluate(config, split, predictions, output, check_only=False, judge_url=None, model=None, limit=None):
    if not predictions.is_file() or output.exists():
        raise ValueError("Predictions must exist and evaluation output must be new")
    if split == "test" and limit is not None:
        raise ValueError("Test evaluation cannot use --limit")
    if split == "test":
        run = read_json(predictions.parent / "run.json")
        tasks = read_json(config.workspace / "test/questions.json")
        rows = read_json(predictions)
        if run["split"] != "test" or run["status"] not in ("completed", "completed_with_errors") or len(rows) != len(tasks) or {p['question_id'] for p in rows} != {q['question_id'] for q in tasks}:
            raise ValueError("Save the complete formal test run before evaluation")
    command = [sys.executable, str(config.data / "evaluation_only/evaluate.py"), "--split", split,
               "--predictions", str(predictions), "--output", str(output)]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    env = os.environ.copy()
    metadata = {"split": split, "predictions_sha256": sha256(predictions), "mode": "check_only" if check_only else "judge_api"}
    if check_only:
        command.append("--check-only")
    else:
        judge_url = judge_url or environment_secret("FINANCEQA_JUDGE_URL")
        model = model or environment_secret("FINANCEQA_JUDGE_MODEL")
        key = environment_secret("FINANCEQA_JUDGE_API_KEY")
        if not judge_url or not model or not key:
            raise ValueError("Configure checker URL, model and FINANCEQA_JUDGE_API_KEY")
        metadata.update(judge_url=judge_url, judge_model=model)
        env["LLM_API_KEY"] = key
        command.extend(["--judge-url", judge_url, "--model", model])
    sidecar = output.with_suffix(".config.json")
    if sidecar.exists():
        raise ValueError("Evaluation configuration already exists")
    write_json(sidecar, metadata)
    return subprocess.run(command, cwd=config.workspace, env=env, check=False).returncode


def recover_rows(rows, make_judgment, max_attempts=3):
    repaired, attempts = deepcopy(rows), []
    for index, row in enumerate(rows):
        if row.get("context_quality") != "API_FAILED":
            continue
        for attempt in range(1, max_attempts + 1):
            try:
                replacement = make_judgment(row)
                if replacement["context_quality"] not in ("CORRECT_INFO", "INCORRECT_INFO", "INSUFFICIENT_INFO"):
                    raise ValueError("Invalid judgment")
                repaired[index] = replacement
                attempts.append({"question_id": row["question_id"], "attempt": attempt, "status": "recovered"})
                break
            except Exception as exc:
                attempts.append({"question_id": row["question_id"], "attempt": attempt,
                                 "status": "API_FAILED", "error_type": type(exc).__name__})
    return repaired, attempts


def recover(config, predictions, source, output):
    """Retry transport/parse failures only; never regenerate or select better answers."""
    if output.exists():
        raise ValueError("Recovery output must be new")
    original = read_json(source)
    metadata = read_json(source.with_suffix(".config.json"))
    if metadata["predictions_sha256"] != sha256(predictions):
        raise ValueError("Predictions changed after original evaluation")
    split = metadata["split"]
    if split not in ("train", "test") or original["split"] != split:
        raise ValueError("Evaluation split mismatch")
    model, url = metadata["judge_model"], metadata["judge_url"]
    if model != environment_secret("FINANCEQA_JUDGE_MODEL") or url != environment_secret("FINANCEQA_JUDGE_URL"):
        raise ValueError("Keep the original checker configuration")
    key = environment_secret("FINANCEQA_JUDGE_API_KEY")
    if not key:
        raise ValueError("Missing checker credentials")
    spec = importlib.util.spec_from_file_location("teacher_evaluator", config.data / "evaluation_only/evaluate.py")
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    tasks = {q["question_id"]: q for q in read_json(config.workspace / split / "questions.json")}
    reference_path = config.workspace / "train/reference_answers.json" if split == "train" else config.data / "evaluation_only/test_reference_answers.json"
    references = {r["question_id"]: r for r in read_json(reference_path)}
    answers = {p["question_id"]: p["answer"] for p in read_json(predictions)}
    if original["expected_questions"] != len(tasks):
        raise ValueError("Only complete-split evaluations can be resumed")
    system = (config.data / "evaluation_only/evaluator_system.txt").read_text(encoding="utf-8")
    template = (config.data / "evaluation_only/evaluator_user.txt").read_text(encoding="utf-8")
    os.environ["LLM_API_KEY"] = key

    def judge(row):
        qid = row["question_id"]
        if tasks[qid]["question"] != references[qid]["question"]:
            raise ValueError("Reference question mismatch")
        text = template.format(context=answers[qid], question=tasks[qid]["question"], ground_truth=references[qid]["answer"])
        parsed, raw = teacher.call_judge(url, model, [{"role": "system", "content": system}, {"role": "user", "content": text}])
        return {"question_id": qid, "company_cik": row["company_cik"], "status": "SCORED",
                **teacher.normalize_judgment(parsed), "raw_judge_response": raw}

    rows, attempts = recover_rows(original["results"], judge)
    result = {**original, "results": rows, **teacher.summarize(rows, len(tasks))}
    groups = {cik: teacher.summarize([r for r in rows if r["company_cik"] == cik], sum(r["company_cik"] == cik for r in rows))
              for cik in dict.fromkeys(q["company_cik"] for q in tasks.values())}
    result.update(per_company=groups, average_support_percentage=sum(g["support_percentage"] for g in groups.values()) / len(groups),
                  model_calls=original["model_calls"] + len(attempts),
                  recovery={"source_sha256": sha256(source), "predictions_sha256": sha256(predictions),
                            "teacher_script_sha256": sha256(config.data / "evaluation_only/evaluate.py"), "attempts": attempts})
    write_json(output.with_suffix(".config.json"), metadata)
    write_json(output, result)
    return result
