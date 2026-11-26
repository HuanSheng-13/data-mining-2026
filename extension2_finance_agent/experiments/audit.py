"""Summarize a saved single-method run; never changes answers or judgments."""
import argparse
from collections import Counter
import json
from pathlib import Path
from extension2_finance_agent.core.config import read_json, write_json, sha256
from extension2_finance_agent.core.protocol import has_tool_markup


def audit_run(folder, score=None):
    folder = Path(folder)
    run = read_json(folder / "run.json")
    summaries = read_json(folder / "summary.json")
    predictions = read_json(folder / "predictions.json")
    assert len(summaries) == len(predictions) == len(run["question_ids"])
    assert {p["question_id"]: p["answer"] for p in predictions} == {s["question_id"]: s["answer"] for s in summaries}
    usage, tools, events_count = Counter(), Counter(), Counter()
    rows, hashes = [], {}
    for summary in summaries:
        log = (folder / summary["log"]).resolve()
        if not log.is_relative_to(folder.resolve()):
            raise ValueError("Log path escapes run")
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert events[0]["event"] == "task_start" and events[-1]["event"] == "task_end"
        assert events[-1]["answer"] == summary["answer"]
        assert all(e["question_id"] == summary["question_id"] for e in events)
        calls = [e for e in events if e["event"] == "tool_result"]
        denied = sum(e["result"].get("error") == "Tool budget exhausted" for e in calls)
        assert len(calls) - denied == summary["tool_calls"]
        tools.update(e["name"] for e in calls)
        events_count.update(e["event"] for e in events)
        usage.update({k: v for k, v in summary["usage"].items() if type(v) in (int, float)})
        rows.append({"question_id": summary["question_id"], "status": summary["status"],
                     "final_tool_markup": has_tool_markup(summary["answer"]),
                     "tool_errors": sum(not e["result"]["ok"] for e in calls),
                     "protocol_recoveries": sum(e["event"] == "protocol_recovery" for e in events),
                     "final_rejections": sum(e["event"] == "final_rejected" for e in events)})
        hashes[summary["log"]] = sha256(log)
    result = {"split": run["split"], "questions": len(summaries), "generation_commit": run["git"]["commit"],
              "statuses": dict(Counter(s["status"] for s in summaries)), "tool_usage": dict(tools),
              "tool_calls": sum(s["tool_calls"] for s in summaries), "model_calls": sum(s["model_calls"] for s in summaries),
              "question_seconds": sum(s["elapsed_seconds"] for s in summaries), "usage": dict(usage),
              "events": dict(events_count), "final_markup_answers": sum(r["final_tool_markup"] for r in rows),
              "questions_audit": rows, "predictions_sha256": sha256(folder / "predictions.json"), "log_sha256": hashes,
              "meaning": "Operational checks and permissive support; not exhaustive financial correctness"}
    if score:
        evaluation = read_json(score)
        if evaluation["expected_questions"] != len(summaries):
            raise ValueError("Score denominator differs from run")
        result["evaluation"] = {k: evaluation[k] for k in ("evaluation_complete", "correct_info", "incorrect_info", "insufficient_info", "api_failed", "support_percentage", "per_company")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new audit output")
    result = audit_run(args.run, args.score)
    write_json(args.output, result)
    print({k: result[k] for k in ("split", "questions", "statuses", "final_markup_answers")})


if __name__ == "__main__":
    main()
