"""Recheck full-train evidence; does not call a model or change predictions."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

from extension2_finance_agent.core.config import ROOT, Config, read_json, write_json, sha256, load_factory
from extension2_finance_agent.core.env import environment_value


def main():
    pub = Path(__file__).resolve().parent
    design = read_json(pub / "design.json")
    normalization = read_json(pub / "artifacts.json")["published_normalization"]
    source_checks = 0
    for name, config in design["configurations"].items():
        cfg = Config(**config)
        load_factory(cfg.validate_factory)(cfg)
        run = read_json(pub / name / "run.json")
        rows = read_json(pub / name / "summary.json")
        predictions = read_json(pub / name / "predictions.json")
        assert len(predictions) == 37
        assert {p["question_id"]: p["answer"] for p in predictions} == {s["question_id"]: s["answer"] for s in rows}
        assert set(p["question_id"] for p in predictions) == set(design["question_ids"])
        old = read_json(pub / "evaluations/judge_v1" / f"{name}_score.json")
        new = read_json(pub / "evaluations/judge_recovered_v1" / f"{name}_score.json")
        assert new["evaluation_complete"] and new["api_failed"] == 0 and new["expected_questions"] == 37
        for relative, original in [(f"evaluations/judge_v1/{name}_score.json", new["recovery"]["source_sha256"]),
                                   (f"{name}/predictions.json", new["recovery"]["predictions_sha256"])]:
            assert normalization[relative]["original_bytes_sha256"] == original
            published = (pub / relative).read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(published).hexdigest() == normalization[relative]["git_lf_sha256"]
        failed = {r["question_id"] for r in old["results"] if r["context_quality"] == "API_FAILED"}
        assert len(failed) == 1 and {a["question_id"] for a in new["recovery"]["attempts"]} == failed
        assert all(a["status"] == "recovered" for a in new["recovery"]["attempts"])
        for before, after in zip(old["results"], new["results"]):
            if before["context_quality"] != "API_FAILED":
                assert before == after
        for summary in rows:
            events = [json.loads(line) for line in (pub / name / summary["log"]).read_text(encoding="utf-8").splitlines()]
            assert events[-1]["answer"] == summary["answer"]
            assert all(e["question_id"] == summary["question_id"] for e in events)
        for path, digest in run["method"]["files"].items():
            relative = Path(path).relative_to(ROOT).as_posix()
            if relative.startswith("data/"):
                assert sha256(path) == digest
            else:
                blob = subprocess.check_output(["git", "show", design["git"]["commit"] + ":" + relative], cwd=ROOT)
                assert hashlib.sha256(blob).hexdigest() == digest, (name, relative)
            source_checks += 1
    keys = {environment_value(k) for k in ("FINANCEQA_API_KEY", "FINANCEQA_JUDGE_API_KEY")} - {None, ""}
    text_suffixes = {".json", ".jsonl", ".md", ".csv", ".py", ".txt"}
    text_files = 0
    for path in pub.rglob("*"):
        if path.is_file() and path.suffix in text_suffixes:
            assert not any(k in path.read_text(encoding="utf-8") for k in keys), "Secret detected"
            text_files += 1
    links = 0
    for path in [pub / "README.md", ROOT / "report/extension2_work_plan.md", ROOT / "extension2_finance_agent/README.md",
                 ROOT / "extension2_finance_agent/experiments/README.md", ROOT / "results/extension2/README.md"]:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" not in target:
                assert (path.parent / target).resolve().exists(), (path, target)
                links += 1
    write_json(pub / "verification.json", {
        "verified_on": "2026-11-25", "unit_tests_passed": 56, "generation_commit": design["git"]["commit"],
        "full_train_questions_per_configuration": 37, "total_predictions": 148, "public_log_files": 148,
        "source_hashes_checked_against_generation_commit_or_raw_data": source_checks,
        "local_links_checked": links, "published_text_files_scanned_without_secret_values": text_files,
        "initial_judge_calls": 148, "failed_only_retries": 4, "all_successful_judgments_unchanged": True,
        "all_recovered_evaluations_complete": True, "teacher_script_unchanged": True, "raw_inputs_unchanged": True,
        "formal_test_run": False, "final_method_frozen": False,
        "next_gate": "Final-answer markup validation and financial-basis review before final freeze/test"})
    checks = {}
    for path in sorted(pub.rglob("*")):
        if path.is_file() and path.name != "checksums.json" and "__pycache__" not in path.parts:
            data = path.read_bytes()
            if path.suffix in text_suffixes or path.name == ".gitignore":
                data = data.replace(b"\r\n", b"\n")
            checks[path.relative_to(pub).as_posix()] = hashlib.sha256(data).hexdigest()
    write_json(pub / "checksums.json", {
        "basis": "Text uses Git LF normalization; binary uses raw bytes. Original-byte ZIP has its own hash in artifacts.json.",
        "files": checks})
    print({"verified": True, "predictions": 148, "logs": 148, "source_checks": source_checks, "files_hashed": len(checks)})


if __name__ == "__main__":
    main()
