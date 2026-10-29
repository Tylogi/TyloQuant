"""Byte-exact native checks using captured 5090 production inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mfq.quantize.cuda.tq_fitter import encode

parser = argparse.ArgumentParser()
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
fixture = args.fixture
args.output.parent.mkdir(parents=True, exist_ok=True)
records = []
for path in sorted(fixture.glob('native-b*-call*.npz')):
    bits = int(path.name.split('-')[1][1:])
    data = np.load(path)
    y, states, bounds = encode(torch.from_numpy(data['x']).cuda(),
                               torch.from_numpy(data['metric']).cuda(), bits,
                               fast_columns=True, column_scaling=True)
    record = dict(fixture=path.name, bits=bits, rings=len(y))
    for key, value in [('y', y), ('states', states), ('bounds', bounds)]:
        record[key + '_differences'] = int(np.count_nonzero(value.cpu().numpy() != data[key]))
    records.append(record)
    print(json.dumps(record), flush=True)
result = dict(cases=records, exact=all(not r[k] for r in records for k in
              ['y_differences', 'states_differences', 'bounds_differences']))
assert len(records) == 16
args.output.write_text(json.dumps(result, indent=2) + '\n')
assert result['exact'], result
print('NATIVE_ALL_EIGHT_EXACT', flush=True)
