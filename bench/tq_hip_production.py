"""Compare the HIP production path with retained 5090 Qwen3.5-9B results."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from mfq.formats.tq import TQTensor, fixed_codebook
from mfq.quantize.cuda.tq_fitter import _library, fit_columns, pack, prepare_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-reference-match', action='store_true',
                        help='Require a reference produced by the same search policy')
    args = parser.parse_args()
    fixture = args.fixture
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    data = np.load(fixture / 'input.npz')
    weight = torch.from_numpy(data['weight_bf16']).view(torch.bfloat16).cuda().float()
    importance = torch.from_numpy(data['importance']).cuda().float()
    importance /= importance.mean()
    _library(16, True)
    columns = prepare_columns(weight)
    expected_columns = TQTensor.load(fixture / 'TQ1.tq').scales
    print('COLUMN_SCALE_DIFFERENCES', int(np.count_nonzero(columns.cpu().numpy() != expected_columns)), flush=True)
    book = torch.tensor(fixed_codebook(), device='cuda')
    rows, width = weight.shape
    records = []
    for bits in args.bits:
        torch.cuda.synchronize()
        start = time.perf_counter()
        states, anchors = fit_columns(weight, bits, importance, columns, state_bits=16, fast=True)
        decoded = (book[states.to(torch.int32) & 65535].float() *
                   (anchors.float()[:, None] * columns.float()[None, :])).half().float()
        squared = (weight - decoded).square()
        weighted = (squared * importance).sum(-1).cpu().numpy()
        unweighted = squared.sum(-1).cpu().numpy()
        packed = pack(states, bits).cpu().numpy()
        descriptors = np.arange(rows, dtype=np.uint64) * (width * bits // 32)
        descriptors |= np.uint64(bits - 1) << np.uint64(32)
        descriptors |= anchors.cpu().numpy().view(np.uint16).astype(np.uint64) << np.uint64(40)
        tensor = TQTensor(rows, width, 512, True, descriptors, columns.cpu().numpy(), packed)
        payload = tensor.to_bytes()
        path = output / f'TQ{bits}.tq'
        path.write_bytes(payload)
        stream = io.BytesIO()
        np.savez(stream, weighted=weighted, unweighted=unweighted)
        (output / f'TQ{bits}-error.npz').write_bytes(stream.getvalue())
        seconds = time.perf_counter() - start
        reference = TQTensor.load(fixture / f'TQ{bits}.tq')
        previous = json.loads((fixture / f'TQ{bits}.json').read_text())
        record = dict(bits=bits, shape=list(weight.shape), seconds=seconds,
                      mweights_per_second=weight.numel() / seconds / 1e6,
                      cuda_seconds=previous['seconds'],
                      exact=payload == reference.to_bytes(),
                      differing_packed_bytes=int(np.count_nonzero(tensor.packed != reference.packed)),
                      differing_anchors=int(np.count_nonzero(tensor.anchors != reference.anchors)),
                      weighted_sse=float(weighted.sum()), cuda_weighted_sse=previous['weighted_sse'],
                      sha256=hashlib.sha256(payload).hexdigest())
        record['relative_weighted_sse_difference'] = abs(record['weighted_sse'] - record['cuda_weighted_sse']) / max(record['cuda_weighted_sse'], 1e-30)
        record['reference_match_required'] = args.require_reference_match
        if args.require_reference_match:
            assert record['relative_weighted_sse_difference'] < 1e-6, record
        records.append(record)
        (output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        del states, anchors, decoded, squared, tensor, payload
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
