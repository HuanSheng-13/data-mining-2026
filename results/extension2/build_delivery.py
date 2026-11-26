"""Publish and verify the single final method's immutable run evidence."""
import argparse
import json
from pathlib import Path
import shutil

from extension2_finance_agent.core.config import ROOT, read_json, write_json, sha256
from extension2_finance_agent.core.env import environment_value
from extension2_finance_agent.experiments.audit import audit_run


def validate_records(folder):
    """Check coverage and recompute per-question accounting from saved events."""
    run = read_json(folder / "run.json")
    predictions = read_json(folder / "predictions.json")
    summaries = read_json(folder / "summary.json")
    ids = run["question_ids"]
    assert len(ids) == len(set(ids))
    assert [p["question_id"] for p in predictions] == ids
    assert [s["question_id"] for s in summaries] == ids
    for summary in summaries:
        path = (folder / summary["log"]).resolve()
        assert path.is_relative_to(folder.resolve())
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert events[-1]["event"] == "task_end"
        assert all(events[-1][k] == v for k, v in summary.items() if k not in ("question_id", "log"))
        models = [e for e in events if e["event"] in ("model_response", "model_error")]
        assert len(models) == summary["model_calls"]
        usage = {}
        for event in models:
            for key, value in event.get("usage", {}).items():
                if type(value) in (int, float):
                    usage[key] = usage.get(key, 0) + value
        assert usage == summary["usage"]


def publish(split, source, score, frozen=None):
    root = Path(__file__).resolve().parent
    target = root / split
    if split not in ("train", "test") or target.exists():
        raise ValueError("Choose a new train/test evidence directory")
    source, score = source.resolve(), score.resolve()
    if not score.is_relative_to(source):
        raise ValueError("Score must belong to the saved run directory")
    run = read_json(source / "run.json")
    expected = 37 if split == "train" else 19
    if run["split"] != split or len(run["question_ids"]) != expected or run["status"] not in ("completed", "completed_with_errors"):
        raise ValueError("Incomplete saved run")
    evaluation = read_json(score)
    if not evaluation["evaluation_complete"] or evaluation["api_failed"] or evaluation["expected_questions"] != expected:
        raise ValueError("Resolve checker failures before publishing")
    validate_records(source)
    assert read_json(score.with_suffix(".config.json"))["predictions_sha256"] == sha256(source / "predictions.json")
    extra = {}
    if split == "test":
        if frozen is None:
            raise ValueError("Formal test delivery requires the original frozen method")
        snapshot = read_json(frozen)
        if run["git"]["commit"] != snapshot["git"]["commit"] or run["git"]["dirty"] or run["method"] != snapshot["method"]:
            raise ValueError("Test method differs from its frozen version")
        train = read_json(root / "train/run.json")
        if train["method"] != run["method"]:
            raise ValueError("Train and test used different method resources")
        extra.update(frozen_method_sha256=sha256(frozen), train_test_method_identical=True)
    shutil.copytree(source, target)
    if frozen:
        shutil.copyfile(frozen, target / "frozen.json")
    audit = audit_run(target, target / score.relative_to(source))
    if audit["final_markup_answers"]:
        raise ValueError("Final tool markup remains in delivery")
    write_json(target / "audit.json", audit)
    keys = {environment_value(k) for k in ("FINANCEQA_API_KEY", "FINANCEQA_JUDGE_API_KEY")} - {None, ""}
    for path in target.rglob("*"):
        if path.is_file():
            assert not any(key in path.read_text(encoding="utf-8") for key in keys), "Secret in outputs"
    write_json(target / "delivery.json", {
        "split": split, "questions": expected, "score_file": score.relative_to(source).as_posix(),
        "generation_commit": run["git"]["commit"], "generation_worktree_clean": not run["git"]["dirty"],
        "final_markup_answers": 0, "source_directory": str(source.relative_to(ROOT)),
        "configured_secret_scan_passed": True, **extra})
    files = {p.relative_to(target).as_posix(): sha256(p) for p in sorted(target.rglob("*")) if p.is_file() and p.name != "checksums.json"}
    write_json(target / "checksums.json", {"basis": "Exact original bytes; Git -text preserves evidence files", "files": files})
    return audit


def verify(root):
    result = {}
    for split in ("train", "test"):
        folder = root / split
        checks = read_json(folder / "checksums.json")["files"]
        assert set(checks) == {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json"}
        for relative, digest in checks.items():
            assert sha256(folder / relative) == digest, (split, relative)
        validate_records(folder)
        delivery = read_json(folder / "delivery.json")
        score = folder / delivery["score_file"]
        assert read_json(score.with_suffix(".config.json"))["predictions_sha256"] == sha256(folder / "predictions.json")
        result[split] = audit_run(folder, folder / delivery["score_file"])
    test, frozen = read_json(root / "test/run.json"), read_json(root / "test/frozen.json")
    assert test["git"] == frozen["git"] and not test["git"]["dirty"]
    assert read_json(root / "train/run.json")["method"] == test["method"] == frozen["method"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(Path(__file__).resolve().parent)
        print({name: {"questions": r["questions"], "support_percentage": r["evaluation"]["support_percentage"]} for name, r in result.items()})
    else:
        if not args.split or not args.source or not args.score:
            parser.error("Publishing requires split, source and score")
        result = publish(args.split, args.source, args.score, args.frozen)
        print({"split": args.split, "questions": result["questions"], "evaluation": result["evaluation"]})


if __name__ == "__main__":
    main()
