from extension2_finance_agent.experiments.recover_judge import recover_rows


def test_retry_only_api_failures_preserves_scored_and_missing_rows():
    rows = [{"question_id": "good", "context_quality": "CORRECT_INFO"},
            {"question_id": "missing", "context_quality": "INSUFFICIENT_INFO"},
            {"question_id": "failed", "context_quality": "API_FAILED"}]
    called = []

    def judge(row):
        called.append(row["question_id"])
        if len(called) == 1:
            raise TimeoutError("private request must not be logged")
        return {"question_id": row["question_id"], "context_quality": "INCORRECT_INFO"}

    result, attempts = recover_rows(rows, judge)
    assert called == ["failed", "failed"]
    assert result[:2] == rows[:2] and result[2]["context_quality"] == "INCORRECT_INFO"
    assert rows[2]["context_quality"] == "API_FAILED"
    assert attempts[0]["error_type"] == "TimeoutError" and "private" not in str(attempts)


def test_failed_retry_stays_in_denominator():
    rows = [{"question_id": "failed", "context_quality": "API_FAILED"}]

    def judge(row):
        raise TimeoutError()

    result, attempts = recover_rows(rows, judge, max_attempts=2)
    assert result == rows and len(attempts) == 2
