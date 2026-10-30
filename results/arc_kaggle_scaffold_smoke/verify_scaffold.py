"""Run the ARC notebook locally and save a transcript plus explicit scaffold checks.

This is a local CPU harness, not a Kaggle execution and not an ARC benchmark.
It executes every code cell in order, then applies checks that the notebook
itself does not perform (distinct second attempts, per-task error recovery,
strict-mode behaviour) and records the results.
"""
import contextlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = Path(os.environ.get("CGR_VERIFICATION_OUTPUT_DIR", str(Path(__file__).resolve().parent)))
out.mkdir(parents=True, exist_ok=True)
notebook = root / 'notebooks' / 'ARC_PRIZE_2026_CGR_KAGGLE.ipynb'
os.environ['CGR_ARC_MODE'] = 'demo'
os.environ['CGR_ARC_DEMO_DIR'] = str(out)
os.environ.setdefault('CGR_LIVE_QWEN', '0')
os.environ.setdefault('CGR_LIVE_ADAPTER', '0')
# Adapter acknowledgement must be supplied explicitly by the caller.
os.chdir(root)
record = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
          'mode': 'demo', 'live_qwen_requested': os.environ['CGR_LIVE_QWEN'],
          'live_adapter_requested': os.environ['CGR_LIVE_ADAPTER'],
          'licence': 'MIT-0', 'status': 'running', 'checks': {}}
scope = {'__name__': '__main__'}
with (out / 'notebook_run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    try:
        nb = json.loads(notebook.read_text(encoding='utf-8'))
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                print('\n--- Notebook cell', i, '---', flush=True)
                exec(compile(''.join(cell['source']), f'notebook-cell-{i}', 'exec'), scope)
        create = scope['create_submission']
        validate = scope['validate_grid']
        identity_submission = scope['identity_submission']

        # 1. Two test inputs, one accepted transform, second attempt must be distinct.
        task = {'train': [{'input': [[1, 2, 3]], 'output': [[3, 2, 1]]}],
                'test': [{'input': [[1, 2, 3]]}, {'input': [[4, 5, 6]]}]}
        predictions, logs = create({'multiple_inputs': task})
        assert len(predictions['multiple_inputs']) == 2
        assert predictions['multiple_inputs'][0]['attempt_1'] == [[3, 2, 1]]
        assert predictions['multiple_inputs'][1]['attempt_1'] == [[6, 5, 4]]
        assert all(set(x) == {'attempt_1', 'attempt_2'} for x in predictions['multiple_inputs'])
        assert all(x['attempt_1'] != x['attempt_2'] for x in predictions['multiple_inputs']), \
            'second attempt duplicated the first'
        record['checks']['multiple_test_inputs'] = {'passed': True, 'predictions': predictions,
                                                    'distinct_attempts': logs['multiple_inputs']['distinct_attempts']}

        # 2. Invalid grids are rejected.
        invalid = [[[1, 2], [3]], [[10]], [[True]], []]
        rejects = []
        for grid in invalid:
            try:
                validate(grid)
            except ValueError as exc:
                rejects.append({'grid': grid, 'error': str(exc)})
            else:
                raise AssertionError(f'Invalid grid accepted: {grid}')
        record['checks']['invalid_grid_rejections'] = {'passed': True, 'cases': rejects}

        # 3. Fallback fires when no transform fits; 1x1 admits only one distinct grid.
        fallback_task = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
        fallback_predictions, fallback_logs = create({'fallback': fallback_task})
        assert fallback_logs['fallback']['fallback'] is True
        assert fallback_predictions['fallback'] == [{'attempt_1': [[3]], 'attempt_2': [[3]]}]
        assert fallback_logs['fallback']['distinct_attempts'] == [False]
        record['checks']['fallback'] = {'passed': True, 'predictions': fallback_predictions,
                                         'selected': fallback_logs['fallback']['selected'],
                                         'note': '1x1 grid: all seven transforms agree, so a distinct '
                                                 'second attempt does not exist for this task'}

        # 4. Partial-fit ranking: the runner-up, not identity, supplies attempt_2.
        partial = {'train': [{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}],
                   'test': [{'input': [[5, 6], [7, 8]]}]}
        partial_predictions, partial_logs = create({'partial': partial})
        assert partial_logs['partial']['accepted'] == ['rot180']
        assert partial_predictions['partial'][0]['attempt_2'] != partial_predictions['partial'][0]['attempt_1']
        record['checks']['partial_fit_second_attempt'] = {
            'passed': True, 'predictions': partial_predictions,
            'selected': partial_logs['partial']['selected']}

        # 5. Non-strict mode recovers from a malformed task instead of losing the submission.
        broken = {'broken': {'train': [{'input': [[1]]}], 'test': [{'input': [[2]]}]}}
        recovered, recovered_logs = create(broken, strict=False)
        assert recovered['broken'] == identity_submission(broken['broken'])
        assert 'error' in recovered_logs['broken']
        try:
            create(broken, strict=True)
        except KeyError:
            strict_raised = True
        else:
            strict_raised = False
        assert strict_raised, 'strict mode should surface the task error'
        record['checks']['task_error_recovery'] = {
            'passed': True, 'predictions': recovered,
            'error': recovered_logs['broken']['error'], 'strict_raised': strict_raised}

        # 6. The placeholder written before solving is a complete, valid mapping.
        placeholder = {task_id: identity_submission(t) for task_id, t in
                       {'multiple_inputs': task, 'fallback': fallback_task}.items()}
        assert set(placeholder) == {'multiple_inputs', 'fallback'}
        for entries in placeholder.values():
            for entry in entries:
                assert set(entry) == {'attempt_1', 'attempt_2'}
                validate(entry['attempt_1']); validate(entry['attempt_2'])
        record['checks']['placeholder_submission'] = {'passed': True, 'tasks': len(placeholder)}

        # 7. The synthetic demo emits two distinct attempts.
        toy_predictions, toy_logs = create(scope['toy'])
        assert toy_predictions['toy_flip'][0]['attempt_1'] != toy_predictions['toy_flip'][0]['attempt_2']
        assert toy_logs['toy_flip']['distinct_attempts'] == [True]
        record['checks']['synthetic_distinct_attempts'] = {
            'passed': True, 'predictions': toy_predictions,
            'selected': toy_logs['toy_flip']['selected']}

        if os.environ['CGR_LIVE_ADAPTER'] == '1':
            record['factorial'] = {'base_answer': scope['base_answer'],
                                   'base_check': scope['verify'](scope['base_answer']),
                                   'coder_answer': scope['adapter_answer'],
                                   'coder_check': scope['verify'](scope['adapter_answer'])}
        if os.environ['CGR_LIVE_QWEN'] == '1':
            record['cache'] = scope['cache_report']
        record['status'] = 'passed'
        print('\nExplicit scaffold checks passed.')
    except Exception:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (out / 'verification_results.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
if record['status'] != 'passed':
    print('Verification failed; inspect notebook_run.log', file=sys.stderr)
    sys.exit(1)
print('Saved:', out / 'notebook_run.log', 'and', out / 'verification_results.json')
