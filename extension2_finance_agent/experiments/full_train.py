"""A's four-configuration train experiment. Never runs or reads test tasks."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from extension2_finance_agent.core.config import Config, ROOT, load_factory, local_path, read_json, write_json
from extension2_finance_agent.core.env import environment_value
from extension2_finance_agent.experiments.evaluate import evaluate
from extension2_finance_agent.experiments.runner import git_state, load_tasks
from extension2_finance_agent.strategies.audit_trace import audit_run

NAMES = ("baseline", "data_optimized", "skill_optimized", "combined")
CONFIGS = {name: f"extension2_finance_agent/configs/{name}.json" for name in NAMES}
DATA_FIELDS = {"artifacts_dir", "tool_factory", "prepare_factory", "validate_factory"}
STRATEGY_FIELDS = {"strategy_factory", "prompt_file", "skill_files"}


def validate_design(configs):
    if set(configs) != set(NAMES):
        raise ValueError("Exactly four configurations are required")
    snapshots = {name: configs[name].snapshot() for name in NAMES}
    base = snapshots["baseline"]
    for name, allowed in (("data_optimized", DATA_FIELDS), ("skill_optimized", STRATEGY_FIELDS),
                          ("combined", DATA_FIELDS | STRATEGY_FIELDS)):
        if any(value != base[key] for key, value in snapshots[name].items() if key not in allowed):
            raise ValueError(f"{name} changes controlled model/data/budget settings")
    for field in base:
        expected = (snapshots["data_optimized"][field] if field in DATA_FIELDS else
                    snapshots["skill_optimized"][field] if field in STRATEGY_FIELDS else base[field])
        if snapshots["combined"][field] != expected:
            raise ValueError(f"Combined configuration does not compose B and C: {field}")
    ids = [task["question_id"] for task in load_tasks(configs["baseline"], "train")]
    if len(ids) != 37:
        raise ValueError("This experiment expects the supplied 37-question train set")
    for config in configs.values():
        if [t["question_id"] for t in load_tasks(config, "train")] != ids:
            raise ValueError("All configurations must use identical train questions")
    return ids


def check_coverage(output, ids):
    for name in NAMES:
        folder = output / name
        predictions = read_json(folder / "predictions.json")
        summaries = read_json(folder / "summary.json")
        run = read_json(folder / "run.json")
        if run["status"] not in ("completed", "completed_with_errors"):
            raise ValueError(f"{name} run is interrupted; preserve it and investigate")
        if len(predictions) != len(ids) or {p["question_id"] for p in predictions} != set(ids):
            raise ValueError(f"{name} is missing or duplicating train questions")
        if len(summaries) != len(ids) or {s["question_id"] for s in summaries} != set(ids):
            raise ValueError(f"{name} summaries do not cover the full train set")
        if {p["question_id"]: p["answer"] for p in predictions} != {s["question_id"]: s["answer"] for s in summaries}:
            raise ValueError(f"{name} predictions differ from saved summaries")


def summarize(output, evaluation_id):
    design = read_json(output / "design.json")
    ids = design["question_ids"]
    check_coverage(output, ids)
    rows, paired = [], {qid: {"question_id": qid} for qid in ids}
    for name in NAMES:
        summaries = read_json(output / name / "summary.json")
        score = read_json(output / "evaluations" / evaluation_id / (name + "_score.json"))
        if not score["evaluation_complete"] or score["api_failed"]:
            raise ValueError("Checker failures must be resolved on saved predictions before comparing methods")
        if score["expected_questions"] != len(ids) or {r["question_id"] for r in score["results"]} != set(ids):
            raise ValueError("Official evaluation denominator differs from the full train set")
        outcomes = {r["question_id"]: r for r in score["results"]}
        usage = Counter()
        for summary in summaries:
            usage.update({k: v for k, v in summary["usage"].items() if type(v) in (int, float)})
            paired[summary["question_id"]][name] = {
                "status": summary["status"], "quality": outcomes[summary["question_id"]]["context_quality"],
                "tool_calls": summary["tool_calls"], "model_calls": summary["model_calls"],
                "elapsed_seconds": summary["elapsed_seconds"], "usage": summary["usage"]}
        trace = audit_run(output / name)
        write_json(output / (name + "_trace_audit.json"), trace)
        row = {"configuration": name, "questions": len(ids),
               "completed": sum(s["status"] == "completed" for s in summaries),
               "statuses": dict(Counter(s["status"] for s in summaries)),
               **{key: score[key] for key in ("correct_info", "incorrect_info", "insufficient_info",
                                             "missing_predictions", "empty_answers", "support_percentage", "per_company")},
               "tool_calls": sum(s["tool_calls"] for s in summaries),
               "model_calls": sum(s["model_calls"] for s in summaries),
               "question_seconds": sum(s["elapsed_seconds"] for s in summaries),
               "usage": dict(usage), "trace_audit": trace["totals"]}
        rows.append(row)
    result = {"scope": "Complete 37-question train development comparison; not test performance",
              "design": design, "evaluation_id": evaluation_id, "configurations": rows,
              "paired": list(paired.values()),
              "limitations": ["One run per configuration; shared API caching/load affects timing.",
                              "Support grading is permissive, not strict complete correctness.",
                              "Token counts are not billed monetary cost.",
                              "Manual source and financial-basis checks remain necessary."]}
    write_json(output / "comparison.json", result)
    fields = ["configuration", "questions", "completed", "correct_info", "incorrect_info", "insufficient_info",
              "support_percentage", "tool_calls", "model_calls", "question_seconds", "total_tokens"]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fields if k != "total_tokens"},
                             "total_tokens": row["usage"].get("total_tokens")})
    return result


def generate_one(name, output):
    with (output / (name + "_execution.txt")).open("x", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", "extension2_finance_agent", "run",
                                 "--config", CONFIGS[name], "--split", "train", "--output", str(output / name)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"generation {name}: exit {result.returncode}", flush=True)
    return result.returncode


def score_one(name, configs, output, evaluation_id):
    folder = output / "evaluations" / evaluation_id
    for check, suffix in ((True, "_format.json"), (False, "_score.json")):
        code = evaluate(configs[name], "train", output / name / "predictions.json", folder / (name + suffix), check_only=check)
        if code:
            raise RuntimeError(f"{name} evaluator process exited with {code}")
    print(f"evaluation {name}: saved", flush=True)


def execute(output, jobs=4, evaluation_id="judge_v1", evaluate_only=False):
    configs = {name: Config.load(path) for name, path in CONFIGS.items()}
    ids = validate_design(configs)
    if not re_safe_label(evaluation_id):
        raise ValueError("evaluation-id must be a simple directory name")
    for key in (configs["baseline"].api_key_env, "FINANCEQA_JUDGE_API_KEY", "FINANCEQA_JUDGE_MODEL", "FINANCEQA_JUDGE_URL"):
        if not environment_value(key):
            raise ValueError(f"Missing local configuration: {key}")
    if any(output.is_relative_to(c.data) for c in configs.values()):
        raise ValueError("Outputs must be outside original data")
    if not evaluate_only:
        if output.exists():
            raise ValueError("Use a new output directory")
        state = git_state()
        if state["dirty"]:
            raise ValueError("Commit the method before the complete comparison")
        preparations = {}
        for config in configs.values():
            key = config.artifacts_dir
            if key not in preparations:
                tick = time.perf_counter()
                manifest = load_factory(config.prepare_factory)(config)
                preparations[key] = {"seconds": time.perf_counter() - tick, "manifest": manifest}
        output.mkdir(parents=True)
        write_json(output / "design.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "git": state,
                   "question_ids": ids, "configurations": {name: c.snapshot() for name, c in configs.items()},
                   "preparations": preparations, "generation_jobs": jobs, "evaluation_jobs": jobs,
                   "judge_model": environment_value("FINANCEQA_JUDGE_MODEL"),
                   "judge_url": environment_value("FINANCEQA_JUDGE_URL"), "formal_test": False,
                   "selection_policy": "Inspect full-train support, contradictions, trace failures, manual basis checks and token usage; do not use test feedback."})
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(lambda name: generate_one(name, output), NAMES))
    else:
        design = read_json(output / "design.json")
        if design["judge_model"] != environment_value("FINANCEQA_JUDGE_MODEL") or design["judge_url"] != environment_value("FINANCEQA_JUDGE_URL"):
            raise ValueError("Re-evaluation must retain the checker configuration")
        if design["question_ids"] != ids:
            raise ValueError("Re-evaluation question set changed")
    # Barrier: all four full prediction arrays must be saved before any scoring.
    check_coverage(output, ids)
    evaluation_folder = output / "evaluations" / evaluation_id
    evaluation_folder.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda name: score_one(name, configs, output, evaluation_id), NAMES))
    return summarize(output, evaluation_id)


def re_safe_label(value):
    return bool(value) and all(c.isascii() and (c.isalnum() or c in "_-") for c in value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--evaluation-id", default="judge_v1")
    parser.add_argument("--evaluate-only", action="store_true", help="Score saved predictions with new output paths; never regenerate answers")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not re_safe_label(args.evaluation_id):
        parser.error("evaluation-id must be a simple directory name")
    output = local_path(args.output)
    result = summarize(output, args.evaluation_id) if args.summarize_only else execute(output, args.jobs, args.evaluation_id, args.evaluate_only)
    print({r["configuration"]: r["support_percentage"] for r in result["configurations"]})


if __name__ == "__main__":
    main()
