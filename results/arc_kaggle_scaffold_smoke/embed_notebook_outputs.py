"""Execute the demo path of the ARC notebook and embed the resulting cell outputs.

This records *demo-mode* outputs only. The optional model diagnostics are skipped
here because no checkpoint is attached; their evidence lives in
`results/arc_kaggle_scaffold_smoke/`.
"""
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
nb_path = root / 'notebooks' / 'ARC_PRIZE_2026_CGR_KAGGLE.ipynb'

os.environ['CGR_ARC_MODE'] = 'demo'
os.environ['CGR_ARC_DEMO_DIR'] = str(here)
os.environ['CGR_LIVE_QWEN'] = '0'
os.environ['CGR_LIVE_ADAPTER'] = '0'
os.chdir(root)

nb = json.loads(nb_path.read_text(encoding='utf-8'))
scope = {'__name__': '__main__'}
count = 0
failed = False

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(''.join(cell['source']), f'cell-{count}', 'exec'), scope)
        text = buf.getvalue()
        cell['execution_count'] = count
        cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout',
                             'text': text.splitlines(keepends=True)}] if text else [])
    except Exception as exc:
        cell['execution_count'] = count
        cell['outputs'] = [{'output_type': 'error', 'ename': type(exc).__name__,
                            'evalue': str(exc),
                            'traceback': traceback.format_exc().splitlines()}]
        failed = True
        break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('embedded outputs for', count, 'code cells into', nb_path)
if failed:
    print('execution failed; see the error output cell', file=sys.stderr)
    sys.exit(1)
