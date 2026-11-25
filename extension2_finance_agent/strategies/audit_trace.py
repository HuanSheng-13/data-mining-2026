"""C 的运行后核查：只分析已保存日志，不参与作答或改变官方评分。"""
import argparse
from collections import Counter
import json
from pathlib import Path

from extension2_finance_agent.core.config import read_json, write_json


def audit_run(folder):
    folder = Path(folder)
    summaries = read_json(folder / 'summary.json')
    rows = []
    for summary in summaries:
        log = (folder / summary['log']).resolve()
        if not log.is_relative_to(folder.resolve()):
            raise ValueError('Log path escapes the run directory')
        events = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
        tool_events = [e for e in events if e['event'] == 'tool_result']
        errors = [e for e in tool_events if not e['result']['ok']]
        # “看起来像工具调用”的正文不是已经执行过的调用；不将 completed 当作答对。
        answer = summary['answer']
        markers = ('<｜｜DSML｜｜', '<tool_call>', '<function_calls>')
        suspicious = any(marker in answer for marker in markers)
        failed_signatures = Counter((e['name'], e['arguments']) for e in errors)
        rows.append({'question_id': summary['question_id'], 'runner_status': summary['status'],
                     'tool_results_logged': len(tool_events),
                     'tool_usage': dict(Counter(e['name'] for e in tool_events)),
                     'tool_errors': len(errors),
                     'repeated_identical_failures': sum(n-1 for n in failed_signatures.values() if n > 1),
                     'successful_calculations': sum(e['name'] == 'calculate' and e['result']['ok'] for e in tool_events),
                     'evidence_results': sum(bool(e['result'].get('sources')) and e['result']['ok'] for e in tool_events),
                     'final_contains_tool_markup': suspicious,
                     'nonempty_answer': bool(answer.strip()),
                     'log': summary['log']})
    return {'kind': 'post-run observable trace audit, not an answer correctness score',
            'questions': rows,
            'totals': {name: sum(r[name] for r in rows) for name in
                       ('tool_errors', 'repeated_identical_failures', 'successful_calculations',
                        'final_contains_tool_markup')},
            'limitations': ['A source or calculator call does not prove the final claim is correct.',
                           'Markup detection only flags known forms; it cannot detect every invalid answer.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new audit output; original predictions are never changed')
    write_json(args.output, audit_run(args.run))


if __name__ == '__main__':
    main()
