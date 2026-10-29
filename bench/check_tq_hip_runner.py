"""Verify retained results, resume and required/optional I/O failures."""
import argparse
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
options = parser.parse_args()
root = options.output
root.mkdir(parents=True, exist_ok=True)
source = root / 'source'
source.mkdir(exist_ok=True)
generator = torch.Generator().manual_seed(20261024)
w = (torch.randn(8, 256, generator=generator) * .02).bfloat16()
save_file({f'model.language_model.layers.{i}.mlp.up_proj.weight': w.clone() for i in range(2)},
          source / 'model.safetensors')
spec = importlib.util.spec_from_file_location('tq_checkpoint', Path(__file__).with_name('tq_hip_checkpoint.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
write_json = runner.write_json
original_print = builtins.print


def run(name, *, optional_faults=False, compute_fault=False):
    output = root / name
    args = ['runner', '--source', str(source), '--output', str(output), '--gpus', '0',
            '--bits', '1', '8', '--search', 'fast', '--column-scaling']

    def write(path, value):
        if optional_faults and path.name == 'progress.json':
            raise PermissionError('injected optional progress error')
        return write_json(path, value)

    original_fit = runner.ColumnFitter.fit

    def fit(fitter, bits, *a, **kw):
        if compute_fault and bits == 8:
            raise RuntimeError('injected required compute error')
        return original_fit(fitter, bits, *a, **kw)

    def stdout(*a, **kw):
        if optional_faults and kw.get('file', sys.stdout) is sys.stdout:
            raise BrokenPipeError('injected display error')
        return original_print(*a, **kw)

    exit_code = 0
    with patch.object(sys, 'argv', args), patch.object(runner, 'write_json', write), \
            patch.object(runner.ColumnFitter, 'fit', fit), patch.object(builtins, 'print', stdout):
        try:
            runner.main()
        except SystemExit as error:
            exit_code = error.code
    report = json.loads((output/'report.json').read_text())
    return dict(exit_code=exit_code, completed=report['completed'], total=report['total'],
                fitted_parameters=report['fitted_parameters'], verified=report['verified'],
                failures=len(report['failures']))


results = dict(optional_faults=run('optional', optional_faults=True),
               resume=run('optional'), compute_fault=run('compute', compute_fault=True))
for metadata in (root/'optional').glob('Q*/*.json'):
    entry=json.loads(metadata.read_text())
    entry['quantizer_revision']='previous-search-policy'
    write_json(metadata,entry)
results['changed_quantizer']=run('optional')
assert results['optional_faults']['verified'] and results['optional_faults']['completed'] == 4
assert results['optional_faults']['fitted_parameters'] == 8192
assert results['resume']['verified'] and results['resume']['fitted_parameters'] == 0
assert results['changed_quantizer']['verified'] and results['changed_quantizer']['fitted_parameters'] == 8192
assert results['compute_fault']['exit_code'] == 1 and not results['compute_fault']['verified']
assert results['compute_fault']['completed'] == 2 and results['compute_fault']['failures'] == 2
(root/'checks.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results), flush=True)
