"""Fit disjoint projection shards in independent single-GPU processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error('GPU IDs must be unique')
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name('tq_hip_checkpoint.py')
    started = time.perf_counter()
    children = []
    for rank, gpu in enumerate(args.gpus):
        environment = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
        command = [sys.executable, str(runner), '--gpus', '0', '--output', str(args.output),
                   '--shard', str(rank), str(len(args.gpus)), *remaining]
        children.append(subprocess.Popen(command, env=environment))
    codes = [child.wait() for child in children]
    entries, reports = [], []
    failures = []
    for rank, (gpu, code) in enumerate(zip(args.gpus, codes)):
        file = args.output / (f'worker{rank}-report.json' if len(args.gpus) > 1 else 'report.json')
        if file.exists():
            report = json.loads(file.read_text())
            for entry in report.pop('entries'):
                entries.append(dict(entry, gpu=gpu))
            reports.append(dict(report, gpu=gpu))
            failures.extend(report['failures'])
        else:
            failures.append(dict(gpu=gpu, error='Worker did not produce a final report'))
        if code:
            failures.append(dict(gpu=gpu, error=f'Worker exited with status {code}'))
    seconds = time.perf_counter() - started
    fitted = sum(e['parameters'] for e in entries if not e['resumed'])
    result = dict(completed=len(entries), total=sum(r['total'] for r in reports),
                  seconds=seconds, fitted_parameters=fitted,
                  reused_parameters=sum(e['parameters'] for e in entries if e['resumed']),
                  mweights_per_second=fitted / seconds / 1e6, failures=failures,
                  gpus=args.gpus, workers=reports, entries=entries,
                  verified=not failures and len(reports) == len(args.gpus) and
                  all(r['verified'] for r in reports))
    temporary = args.output / 'report.json.partial'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temporary, args.output / 'report.json')
    try:
        print(json.dumps({k: v for k, v in result.items() if k not in ('entries', 'workers')}), flush=True)
    except (OSError, ValueError):
        pass
    if not result['verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
