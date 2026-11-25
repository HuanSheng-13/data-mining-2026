"""C 的 train 开发对比入口；使用 A 的原 runner 和官方独立评测，不运行 test。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

from extension2_finance_agent.core.client import environment_secret
from extension2_finance_agent.core.config import Config, ROOT, load_factory, read_json, write_json
from extension2_finance_agent.experiments.evaluate import evaluate
from extension2_finance_agent.experiments.runner import run, load_tasks

# 只按公开 train 问题的类型覆盖选择；不含答案或固定 SQL。
DEVELOPMENT_IDS = ('FQA-1640147-01', 'FQA-1682852-05', 'FQA-1682852-09',
                   'FQA-1373715-07', 'FQA-702165-05', 'FQA-702165-07')
CONFIGS = {'baseline': 'extension2_finance_agent/configs/baseline.json',
           'skill_optimized': 'extension2_finance_agent/configs/skill_optimized.json'}


def validate_pair(configs):
    left, right = (configs[name].snapshot() for name in CONFIGS)
    allowed = {'strategy_factory', 'prompt_file', 'skill_files'}
    if {k for k in left if left[k] != right[k]} - allowed:
        raise ValueError('Comparison may only vary strategy resources')
    for key in ('FINANCEQA_API_KEY', 'FINANCEQA_JUDGE_API_KEY',
                'FINANCEQA_JUDGE_URL', 'FINANCEQA_JUDGE_MODEL'):
        if not environment_secret(key):
            raise ValueError(f'Missing {key}; configure the local .env first')


def summarize(output, ids):
    """以作答子集为分析分母，同时保留官方 37 题分母，避免误读 demo 分数。"""
    rows = []
    for name in CONFIGS:
        folder = output / name
        stats = read_json(folder / 'summary.json')
        score = read_json(output / (name + '_score.json'))
        selected = {r['question_id']: r for r in score['results'] if r['question_id'] in ids}
        if set(selected) != set(ids):
            raise ValueError('Evaluator is missing selected question records')
        counts = Counter(r['context_quality'] for r in selected.values())
        usage = Counter()
        for s in stats:
            usage.update({k: v for k, v in s['usage'].items() if type(v) in (int, float)})
        rows.append({'configuration': name, 'questions': len(ids),
                     'completed': sum(s['status'] == 'completed' for s in stats),
                     'statuses': dict(Counter(s['status'] for s in stats)),
                     'subset_quality_counts': dict(counts),
                     'subset_support_percentage': 100 * counts['CORRECT_INFO'] / len(ids),
                     'official_expected_questions': score['expected_questions'],
                     'official_support_percentage': score['support_percentage'],
                     'evaluation_complete': score['evaluation_complete'],
                     'api_failed': score['api_failed'],
                     'model_calls': sum(s['model_calls'] for s in stats),
                     'tool_calls': sum(s['tool_calls'] for s in stats),
                     'question_seconds': sum(s['elapsed_seconds'] for s in stats),
                     'usage': dict(usage)})
    paired = []
    for qid in ids:
        pair = {'question_id': qid}
        for name in CONFIGS:
            s = next(s for s in read_json(output / name / 'summary.json') if s['question_id'] == qid)
            result = next(s for s in read_json(output / (name + '_score.json'))['results'] if s['question_id'] == qid)
            pair[name] = {'status': s['status'], 'context_quality': result['context_quality'],
                          'tool_calls': s['tool_calls'], 'model_calls': s['model_calls'],
                          'seconds': s['elapsed_seconds'], 'usage': s['usage']}
        paired.append(pair)
    result = {'scope': 'train development; not a full train/test benchmark',
              'question_ids': list(ids), 'configurations': rows, 'paired': paired,
              'limitations': ['One run per configuration; model and judge are stochastic.',
                             'Official support is permissive, not strict complete correctness.',
                             'No monetary cost inferred from tokens; consult provider billing.']}
    write_json(output / 'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        print(summarize(output, DEVELOPMENT_IDS))
        return
    configs = {name: Config.load(path) for name, path in CONFIGS.items()}
    validate_pair(configs)
    if output.exists() or any(output.is_relative_to(c.data) for c in configs.values()):
        raise ValueError('Use a new output directory outside the original data package')
    for c in configs.values():
        load_tasks(c, 'train', question_ids=DEVELOPMENT_IDS)
    started = time.perf_counter()
    prepared = load_factory(configs['baseline'].prepare_factory)(configs['baseline'])
    preparation_seconds = time.perf_counter() - started
    output.mkdir(parents=True)
    write_json(output / 'design.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
               'common_baseline': '8021a1628c48990030d3ae2dfe2a8381102ca0da',
               'question_ids': list(DEVELOPMENT_IDS), 'preparation_seconds': preparation_seconds,
               'reports': len(prepared['reports']), 'chunks': prepared['chunks'],
               'judge_model': environment_secret('FINANCEQA_JUDGE_MODEL'),
               'judge_url': environment_secret('FINANCEQA_JUDGE_URL'),
               'order': list(CONFIGS), 'test_run': False})
    # 先保存两组全部预测，再独立评分。失败输出保留，不手工修答案。
    for name, c in configs.items():
        run(c, 'train', output / name, question_ids=DEVELOPMENT_IDS)
    for name, c in configs.items():
        for check_only, suffix in ((True, '_format.json'), (False, '_score.json')):
            code = evaluate(c, 'train', output / name / 'predictions.json', output / (name + suffix),
                            check_only=check_only)
            if code:
                raise RuntimeError(f'{name} evaluation failed with exit {code}; preserve outputs')
    summarize(output, DEVELOPMENT_IDS)


if __name__ == '__main__':
    main()
