"""Resume only API_FAILED train judgments using the unchanged teacher functions."""
import argparse
from copy import deepcopy
import importlib.util
import os

from extension2_finance_agent.core.config import Config, local_path, read_json, sha256, write_json
from extension2_finance_agent.core.env import environment_value
from extension2_finance_agent.experiments.full_train import NAMES, check_coverage, summarize, re_safe_label


def recover_rows(rows, make_judgment, max_attempts=3):
    repaired, attempts = deepcopy(rows), []
    for i, row in enumerate(rows):
        if row.get("context_quality") != "API_FAILED":
            continue
        for attempt in range(1, max_attempts + 1):
            try:
                replacement = make_judgment(row)
                if replacement["context_quality"] not in ("CORRECT_INFO", "INCORRECT_INFO", "INSUFFICIENT_INFO"):
                    raise ValueError("Invalid judgment")
                repaired[i] = replacement
                attempts.append({"question_id": row["question_id"], "attempt": attempt, "status": "recovered"})
                break
            except Exception as exc:
                # Do not expose request headers or raw exception contents.
                attempts.append({"question_id": row["question_id"], "attempt": attempt,
                                 "status": "API_FAILED", "error_type": type(exc).__name__})
    return repaired, attempts


def recover(output, source_id, target_id):
    if not re_safe_label(source_id) or not re_safe_label(target_id):
        raise ValueError("Use simple evaluation directory names")
    design = read_json(output / "design.json")
    check_coverage(output, design["question_ids"])
    config = Config.load("extension2_finance_agent/configs/baseline.json")
    model, url = environment_value("FINANCEQA_JUDGE_MODEL"), environment_value("FINANCEQA_JUDGE_URL")
    if model != design["judge_model"] or url != design["judge_url"]:
        raise ValueError("Keep the original checker model and URL")
    secret = environment_value("FINANCEQA_JUDGE_API_KEY")
    if not secret:
        raise ValueError("Missing checker credentials")
    teacher_path = config.data / "evaluation_only/evaluate.py"
    spec = importlib.util.spec_from_file_location("teacher_train_evaluator", teacher_path)
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    questions = {q["question_id"]: q for q in read_json(config.workspace / "train/questions.json")}
    references = {q["question_id"]: q for q in read_json(config.workspace / "train/reference_answers.json")}
    system = (config.data / "evaluation_only/evaluator_system.txt").read_text(encoding="utf-8")
    template = (config.data / "evaluation_only/evaluator_user.txt").read_text(encoding="utf-8")
    target = output / "evaluations" / target_id
    target.mkdir(parents=True, exist_ok=False)
    os.environ["LLM_API_KEY"] = secret  # Only in this independent evaluator process.
    for name in NAMES:
        source = output / "evaluations" / source_id / (name + "_score.json")
        report = read_json(source)
        if report["judge_model"] != model or report["expected_questions"] != len(questions):
            raise ValueError("Original report checker or denominator mismatch")
        prediction_path = output / name / "predictions.json"
        predictions = {p["question_id"]: p for p in read_json(prediction_path)}

        def judge(row):
            qid = row["question_id"]
            question, reference = questions[qid], references[qid]
            if question["question"] != reference["question"]:
                raise ValueError("Reference question differs")
            text = template.format(context=predictions[qid]["answer"], question=question["question"], ground_truth=reference["answer"])
            parsed, raw = teacher.call_judge(url, model, [{"role": "system", "content": system}, {"role": "user", "content": text}])
            return {"question_id": qid, "company_cik": row["company_cik"], "status": "SCORED",
                    **teacher.normalize_judgment(parsed), "raw_judge_response": raw}

        rows, attempts = recover_rows(report["results"], judge)
        report.update(results=rows, **teacher.summarize(rows, len(questions)))
        groups = {cik: teacher.summarize([r for r in rows if r["company_cik"] == cik],
                                       sum(r["company_cik"] == cik for r in rows))
                  for cik in dict.fromkeys(q["company_cik"] for q in questions.values())}
        report["per_company"] = groups
        report["average_support_percentage"] = sum(g["support_percentage"] for g in groups.values()) / len(groups)
        report["recovery"] = {"kind": "derived report resuming failed requests with original teacher functions",
                              "source": str(source.relative_to(output)), "source_sha256": sha256(source),
                              "teacher_script_sha256": sha256(teacher_path), "predictions_sha256": sha256(prediction_path),
                              "initial_model_calls": report["model_calls"], "retry_model_calls": len(attempts), "attempts": attempts}
        report["model_calls"] += len(attempts)
        write_json(target / (name + "_score.json"), report)
        print(f"{name}: retried {len(attempts)} requests, remaining API failures {report['api_failed']}", flush=True)
    return summarize(output, target_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", default="judge_v1")
    parser.add_argument("--target-id", default="judge_recovered_v1")
    args = parser.parse_args()
    result = recover(local_path(args.output), args.source_id, args.target_id)
    print({r["configuration"]: r["support_percentage"] for r in result["configurations"]})


if __name__ == "__main__":
    main()
