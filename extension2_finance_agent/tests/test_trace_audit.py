"""核查器不应把 runner 完成或伪工具正文当成真实执行与正确性。"""
import json
import pytest

from extension2_finance_agent.core.config import write_json
from extension2_finance_agent.strategies.audit_trace import audit_run


def test_completed_markup_is_flagged_without_changing_predictions(tmp_path):
    (tmp_path / 'logs').mkdir()
    events = [{'event': 'tool_result', 'name': 'calculate', 'arguments': '1;2',
               'result': {'ok': False, 'sources': [], 'error': 'invalid'}}] * 2
    events.append({'event': 'tool_result', 'name': 'calculate', 'arguments': '1+2',
                   'result': {'ok': True, 'sources': [], 'data': {'value': 3}}})
    log = tmp_path / 'logs/000.jsonl'
    log.write_text('\n'.join(json.dumps(e) for e in events))
    write_json(tmp_path / 'summary.json', [{'question_id': 'synthetic', 'log': 'logs/000.jsonl',
               'answer': '<｜｜DSML｜｜ calls>not an executed tool</>', 'status': 'completed'}])
    original = log.read_bytes()
    result = audit_run(tmp_path)
    row = result['questions'][0]
    assert row['final_contains_tool_markup'] and row['tool_results_logged'] == 3
    assert row['successful_calculations'] == 1 and row['evidence_results'] == 0
    assert row['tool_errors'] == 2 and row['repeated_identical_failures'] == 1
    assert log.read_bytes() == original


def test_external_log_path_is_rejected(tmp_path):
    write_json(tmp_path / 'summary.json', [{'log': '../outside.jsonl'}])
    with pytest.raises(ValueError, match='escapes'):
        audit_run(tmp_path)
