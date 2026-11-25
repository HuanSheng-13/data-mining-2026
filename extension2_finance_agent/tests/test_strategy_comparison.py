"""比较入口应拒绝不公平配置，并维持官方分母与子集分母的区别。"""
from dataclasses import replace

import pytest

from extension2_finance_agent.core.config import Config, write_json
from extension2_finance_agent.strategies.compare_train import validate_pair, summarize


def test_comparison_rejects_budget_change(monkeypatch):
    import extension2_finance_agent.strategies.compare_train as comparison
    monkeypatch.setattr(comparison, 'environment_secret', lambda _: 'test-only')
    with pytest.raises(ValueError, match='only vary strategy'):
        validate_pair({'baseline': Config(), 'skill_optimized': replace(Config(), max_tool_calls=32)})


def test_subset_summary_preserves_official_denominator(tmp_path):
    for name in ('baseline', 'skill_optimized'):
        write_json(tmp_path / name / 'summary.json', [{
            'question_id': 'synthetic', 'status': 'completed', 'tool_calls': 1,
            'model_calls': 2, 'elapsed_seconds': 0.5, 'usage': {'total_tokens': 12}}])
        write_json(tmp_path / (name + '_score.json'), {
            'results': [{'question_id': 'synthetic', 'context_quality': 'CORRECT_INFO'}],
            'expected_questions': 37, 'support_percentage': 100/37,
            'evaluation_complete': True, 'api_failed': 0})
    result = summarize(tmp_path, ['synthetic'])
    row = result['configurations'][0]
    assert row['subset_support_percentage'] == 100
    assert row['official_support_percentage'] == pytest.approx(100/37)
    assert row['official_expected_questions'] == 37
    with pytest.raises(ValueError, match='missing selected'):
        summarize(tmp_path, ['synthetic', 'missing'])
