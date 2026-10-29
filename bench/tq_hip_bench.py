"""Measure TQ fitting and calibration on retained Qwen projection weights."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from mfq.kernels.hip.tq import Projection, library as projection_library
from mfq.quantize.cuda.tq_encoder import _library as encoder_library
from mfq.quantize.cuda.tq_fitter import _library as fitter_library
from mfq.quantize.tq import quantize


def load_weight(source_root, model, projection):
    key = f"model.language_model.layers.0.mlp.{projection}_proj.weight"
    for path in sorted((source_root / model).glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as checkpoint:
            if key in checkpoint.keys():
                return checkpoint.get_tensor(key).cuda(), key
    raise KeyError(key)


def elapsed_gpu(function, repeats=5):
    function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--cases', nargs='+', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("Qwen3.5-0.8B", "up", bits) for bits in (1, 2, 3, 4)]
    cases += [("Qwen3.5-4B", projection, 4) for projection in ("up", "down")]
    encoder_library()
    fitter_library()
    projection_library()
    records = []
    for index, (model, projection, bits) in enumerate(cases):
        if args.cases is not None and index not in args.cases:
            continue
        name = f'{model}-{projection}-q{bits}'
        weight, key = load_weight(args.source_root, model, projection)
        importance = torch.linspace(.25, 2, weight.shape[1], device=weight.device)
        importance[::11] = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.profile and index == 4:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA]) as prof:
                tensor = quantize(weight, bits, imatrix=importance)
                torch.cuda.synchronize()
            (args.output / 'profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=30))
            prof.export_chrome_trace(str(args.output / 'profile.json'))
        else:
            tensor = quantize(weight, bits, imatrix=importance)
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        payload = tensor.to_bytes()
        (args.output / (name + '.tq')).write_bytes(payload)
        reference = args.reference / (name + '.tq') if args.reference else None
        record = dict(name=name, source_tensor=key, shape=list(weight.shape), bits=bits,
                      seconds=seconds, parameters=weight.numel(),
                      mweights_per_second=weight.numel() / seconds / 1e6,
                      sha256=hashlib.sha256(payload).hexdigest(),
                      reference_exact=(reference.read_bytes() == payload) if reference else None,
                      profiled=bool(args.profile and index == 4))
        if reference is not None:
            assert record['reference_exact'], f'Quantized payload changed: {name}'
        decoded = torch.from_numpy(tensor.dequantize(weight.shape[1])).cuda().float()
        loss = ((decoded - weight.float()).square() * importance).sum()
        norm = (weight.float().square() * importance).sum()
        record['weighted_relative_rmse'] = float((loss / norm).sqrt())
        if bits == 4:
            op = Projection(tensor, input_columns=weight.shape[1])
            record['calibration'] = []
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for tokens in (128, 512):
                    torch.manual_seed(42)
                    x = torch.randn(tokens, weight.shape[1], device='cuda', dtype=dtype)
                    dy = torch.randn(tokens, weight.shape[0], device='cuda', dtype=dtype)
                    out = torch.empty(tokens, weight.shape[0], device='cuda', dtype=dtype)
                    dx = torch.empty_like(x)
                    forward_ms = elapsed_gpu(lambda: op.forward(x, out))
                    backward_ms = elapsed_gpu(lambda: op.backward(dy, dx))
                    reference_y = x.float() @ decoded.T
                    reference_dx = dy.float() @ decoded
                    relative_y = float((out.float() - reference_y).square().mean().sqrt()
                                       / reference_y.square().mean().sqrt())
                    relative_dx = float((dx.float() - reference_dx).square().mean().sqrt()
                                        / reference_dx.square().mean().sqrt())
                    limit = {torch.float16: .001, torch.bfloat16: .004,
                             torch.float32: 4e-6}[dtype]
                    assert max(relative_y, relative_dx) < limit, (name, dtype, relative_y, relative_dx)
                    record['calibration'].append(dict(dtype=str(dtype), tokens=tokens,
                        forward_ms=forward_ms, backward_ms=backward_ms,
                        forward_relative_error=relative_y, backward_relative_error=relative_dx))
        records.append(record)
        (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        del weight, decoded
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
