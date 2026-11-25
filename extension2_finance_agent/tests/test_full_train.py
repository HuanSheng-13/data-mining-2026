"""Controlled 2x2 integration and complete-denominator experiment checks."""
from dataclasses import replace
import pytest
from extension2_finance_agent.core.config import Config, read_json, write_json, ROOT
from extension2_finance_agent.experiments import full_train


def configs():
    return {name: Config(**read_json(ROOT / path)) for name, path in full_train.CONFIGS.items()}


def test_combined_is_exact_composition_and_budget_cannot_drift(monkeypatch):
    monkeypatch.setattr(full_train, "load_tasks", lambda *args: [{"question_id": str(i)} for i in range(37)])
    values = configs()
    assert len(full_train.validate_design(values)) == 37
    wrong = {**values, "combined": replace(values["combined"], max_tool_calls=32)}
    with pytest.raises(ValueError, match="controlled"):
        full_train.validate_design(wrong)
    wrong = {**values, "combined": replace(values["combined"], tool_factory=values["baseline"].tool_factory)}
    with pytest.raises(ValueError, match="compose"):
        full_train.validate_design(wrong)


def test_full_train_summary_keeps_failures_and_rejects_checker_failure(tmp_path):
    ids = [f"q{i}" for i in range(37)]
    write_json(tmp_path / "design.json", {"question_ids": ids})
    for name in full_train.NAMES:
        summaries, predictions, scores = [], [], []
        for i, qid in enumerate(ids):
            answer = "" if i == 0 else "Fixture answer"
            summaries.append({"question_id": qid, "answer": answer, "status": "model_error" if i == 0 else "completed",
                              "tool_calls": 1, "model_calls": 2, "elapsed_seconds": .5,
                              "usage": {"total_tokens": 12}, "log": f"logs/{i:03d}.jsonl"})
            predictions.append({"question_id": qid, "answer": answer})
            scores.append({"question_id": qid, "context_quality": "INSUFFICIENT_INFO" if i == 0 else "CORRECT_INFO"})
            log = tmp_path / name / f"logs/{i:03d}.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('{"event":"task_start"}\n', encoding="utf-8")
        write_json(tmp_path / name / "run.json", {"status": "completed_with_errors"})
        write_json(tmp_path / name / "summary.json", summaries)
        write_json(tmp_path / name / "predictions.json", predictions)
        write_json(tmp_path / "evaluations/judge_v1" / (name + "_score.json"), {
            "results": scores, "expected_questions": 37, "evaluation_complete": True, "api_failed": 0,
            "correct_info": 36, "incorrect_info": 0, "insufficient_info": 1, "missing_predictions": 0,
            "empty_answers": 1, "support_percentage": 36 / 37 * 100, "per_company": {}})
    result = full_train.summarize(tmp_path, "judge_v1")
    assert all(row["questions"] == 37 and row["completed"] == 36 for row in result["configurations"])
    assert all(row["support_percentage"] == pytest.approx(36 / 37 * 100) for row in result["configurations"])
    assert (tmp_path / "metrics.csv").is_file()
    score_file = tmp_path / "evaluations/judge_v1/baseline_score.json"
    score = read_json(score_file)
    write_json(score_file, {**score, "api_failed": 1, "evaluation_complete": False})
    with pytest.raises(ValueError, match="Checker failures"):
        full_train.summarize(tmp_path, "judge_v1")


def test_coverage_rejects_partial_run_and_label_path_escape(tmp_path):
    for name in full_train.NAMES:
        write_json(tmp_path / name / "run.json", {"status": "running"})
        write_json(tmp_path / name / "predictions.json", [])
        write_json(tmp_path / name / "summary.json", [])
    with pytest.raises(ValueError, match="interrupted"):
        full_train.check_coverage(tmp_path, ["q"])
    assert full_train.re_safe_label("judge_v2")
    assert not full_train.re_safe_label("../score")
