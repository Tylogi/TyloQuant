"""Fit text projections on independent GPUs and retain restartable TQ payloads."""

import argparse
import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open

from mfq.formats.tq import TQTensor
from mfq.quantize.cuda.tq_encoder import _library as encoder_library
from mfq.quantize.cuda.tq_fitter import _library as fitter_library
from mfq.quantize.cuda.tq_fitter import ColumnFitter, pack, prepare_columns
from mfq.quantize.tq import quantize


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--bits', nargs='+', type=int, default=[4], choices=range(1, 9))
    parser.add_argument('--groupsize', type=int, default=512)
    parser.add_argument('--search', choices=['adaptive', 'fast', 'full'], default='adaptive')
    parser.add_argument('--column-scaling', action='store_true')
    parser.add_argument('--shard', nargs=2, type=int, default=[0, 1], metavar=('INDEX', 'COUNT'))
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error('GPU IDs must be unique')
    if len(set(args.bits)) != len(args.bits):
        parser.error('Bit widths must be unique')
    if args.column_scaling and args.search == 'full':
        parser.error('Column scaling requires adaptive or fast fitting')
    rank, shard_count = args.shard
    if shard_count < 1 or not 0 <= rank < shard_count:
        parser.error('Shard index must be in [0, count)')
    prefix = f'worker{rank}-' if shard_count > 1 else ''
    progress_path = args.output / (prefix + 'progress.json')
    report_path = args.output / (prefix + 'report.json')
    args.output.mkdir(parents=True, exist_ok=True)
    quantizer_paths = ['mfq/quantize/tq.py', 'mfq/quantize/cuda/tq_columns.py',
                       'mfq/quantize/cuda/tq_scale_repair.py', 'mfq/quantize/cuda/tq_fitter.py',
                       'mfq/quantize/hip/tq_fitter_columns.hip']
    package_root = Path(__file__).resolve().parents[1]
    quantizer_revision = hashlib.sha256(b''.join((package_root / p).read_bytes() for p in quantizer_paths)).hexdigest()
    started = time.perf_counter()
    torch.set_num_threads(1)
    jobs = []
    for path in sorted(args.source.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as checkpoint:
            for name in checkpoint.keys():
                shape = checkpoint.get_slice(name).get_shape()
                if (name.startswith('model.language_model.layers.') and
                        name.endswith('.weight') and len(shape) == 2):
                    jobs.append((path, name, shape))
    if not jobs:
        raise ValueError('No text projection weights found')
    jobs.sort(key=lambda item: item[2][0] * item[2][1], reverse=True)
    jobs = jobs[rank::shard_count]
    work = queue.Queue()
    for item in jobs:
        work.put(item)
    if args.column_scaling:
        fitter_library(16, args.search == 'fast', columns=True)
        fitter_library(16, True)  # All-width packing library.
    else:
        encoder_library()
        fitter_library()
    setup_seconds = time.perf_counter() - started
    lock = threading.Lock()
    completed, failures = [], []
    active = 0
    progress_file_enabled = True
    progress_stdout_enabled = True

    def publish(value, *, save=False):
        nonlocal progress_file_enabled, progress_stdout_enabled
        if save and progress_file_enabled:
            try:
                write_json(progress_path, value)
            except OSError:
                progress_file_enabled = False
        if progress_stdout_enabled:
            try:
                print(json.dumps({k: v for k, v in value.items() if k != 'entries'}), flush=True)
            except (OSError, ValueError):
                progress_stdout_enabled = False

    def snapshot():
        seconds = time.perf_counter() - started
        parameters = sum(entry['parameters'] for entry in completed)
        fitted = sum(entry['parameters'] for entry in completed if not entry['resumed'])
        return dict(completed=len(completed), total=len(jobs) * len(args.bits), pending=work.qsize(),
                    active_workers=active, parameters=parameters, seconds=seconds,
                    fitted_parameters=fitted, reused_parameters=parameters - fitted,
                    mweights_per_second=fitted / seconds / 1e6,
                    delivery_mweights_per_second=parameters / seconds / 1e6,
                    failures=list(failures), gpus=args.gpus, bits=args.bits,
                    search=args.search, column_scaling=args.column_scaling, quantizer_revision=quantizer_revision,
                    groupsize=args.groupsize, setup_seconds=setup_seconds,
                    shard=args.shard, source=str(args.source), entries=list(completed))

    def worker(gpu):
        nonlocal active, progress_file_enabled
        torch.cuda.set_device(gpu)
        while True:
            fatal_device_error = False
            try:
                path, name, shape = work.get_nowait()
            except queue.Empty:
                return
            with lock:
                active += 1
            try:
                before = time.perf_counter()
                with safe_open(path, framework='pt', device='cpu') as checkpoint:
                    source = checkpoint.get_tensor(name)
                source_hash = hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest()
                device_weight, columns, importance, column_fitter = None, None, None, None
                host_columns = None
                checkpoints = {}
                for bits in args.bits:
                    parent = args.output if len(args.bits) == 1 else args.output / f'Q{bits}'
                    target = parent / (name + '.tq')
                    metadata = target.with_suffix('.json')
                    cached = json.loads(metadata.read_text()) if metadata.exists() else None
                    reusable = (cached and target.exists() and cached.get('quantizer_revision') == quantizer_revision
                                and cached['source_sha256'] == source_hash
                                and cached['bits'] == bits and cached['groupsize'] == args.groupsize
                                and cached.get('search', 'adaptive') == args.search
                                and cached.get('column_scaling', False) == args.column_scaling)
                    payload = target.read_bytes() if reusable else None
                    reusable = bool(reusable and cached['sha256'] == hashlib.sha256(payload).hexdigest())
                    checkpoints[bits] = (parent, target, metadata, reusable, payload)
                for bits in args.bits:
                    before_candidate = time.perf_counter()
                    parent, target, metadata, reusable, payload = checkpoints[bits]
                    parent.mkdir(exist_ok=True)
                    if reusable:
                        tensor = TQTensor.load(target)
                        payload = tensor.to_bytes()
                    else:
                        if device_weight is None:
                            device_weight = source.to(f'cuda:{gpu}')
                            if args.column_scaling:
                                logical = device_weight.shape[1]
                                width = (logical + 255) // 256 * 256
                                device_weight = torch.nn.functional.pad(device_weight.float(), (0, width-logical))
                                importance = torch.nn.functional.pad(torch.ones(logical, device=device_weight.device), (0, width-logical))
                                columns = prepare_columns(device_weight)
                                column_fitter = ColumnFitter(device_weight, importance, columns, fast=args.search == 'fast')
                        if args.column_scaling:
                            rows, width = device_weight.shape
                            states, anchors = column_fitter.fit(bits)
                            packed = pack(states, bits).cpu().numpy()
                            host_anchors = anchors.cpu().numpy()
                            if host_columns is None:
                                host_columns = columns.cpu().numpy()
                            del states, anchors
                            descriptors = np.arange(rows, dtype=np.uint64) * (width * bits // 32)
                            descriptors |= np.uint64(bits - 1) << np.uint64(32)
                            descriptors |= host_anchors.view(np.uint16).astype(np.uint64) << np.uint64(40)
                            tensor = TQTensor(rows, width, args.groupsize, True, descriptors, host_columns, packed)
                            del packed
                        else:
                            tensor = quantize(device_weight, bits, groupsize=args.groupsize, search=args.search)
                        payload = tensor.to_bytes()
                    if args.reference:
                        reference_parent = args.reference if len(args.bits) == 1 else args.reference / f'Q{bits}'
                        if payload != (reference_parent / target.name).read_bytes():
                            raise AssertionError('Payload differs from reference: ' + name)
                    if not reusable:
                        temporary = target.with_suffix('.tq.partial')
                        temporary.write_bytes(payload)
                        os.replace(temporary, target)
                    entry = dict(source=name, shape=shape, gpu=gpu, bits=bits,
                                 search=args.search, column_scaling=args.column_scaling, quantizer_revision=quantizer_revision,
                                 groupsize=args.groupsize, parameters=source.numel(),
                                 seconds=time.perf_counter() - before_candidate, resumed=bool(reusable),
                                 source_sha256=source_hash, sha256=hashlib.sha256(payload).hexdigest(),
                                 reference_exact=True if args.reference else None)
                    write_json(metadata, entry)
                    with lock:
                        completed.append(entry)
                del device_weight, columns, importance, column_fitter
            except Exception as error:
                fatal_device_error = isinstance(error, getattr(torch, 'AcceleratorError', ()))
                with lock:
                    failures.append(dict(source=name, gpu=gpu, error=repr(error)))
                    publish(dict(event='failure', **failures[-1]))
            finally:
                with lock:
                    active -= 1
                    state = snapshot()
                    if progress_file_enabled:
                        try:
                            write_json(progress_path, state)
                        except OSError:
                            progress_file_enabled = False
                    if len(completed) % 20 == 0 or failures:
                        publish(state)
                work.task_done()
            if fatal_device_error:
                return

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    result = snapshot()
    result['verified'] = not failures and len(completed) == len(jobs) * len(args.bits)
    write_json(report_path, result)
    publish(result)
    if not result['verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
