"""Non-MTP TQ packed projection benchmark with random per-row precision."""

import argparse
import json
import struct
import time

import mlx.core as mx
import numpy as np

from mfq.formats.tq import TQTensor
from mfq.kernels.metal.tq import Projection


def random_tensor(rows, width, *, seed, bits=(1, 8), gs=512, columns=False, byte_budget=None):
    rng = np.random.default_rng(seed)
    width = (width + 255) // 256 * 256
    qs = rng.integers(bits[0], bits[1] + 1, rows, dtype=np.uint64)
    if byte_budget is not None:
        overhead = 64 + rows * 8 + (width * 2 if columns else (rows * ((width + gs - 1) // gs) + 1) // 2)
        target = round((byte_budget - overhead) * 8 / width)
        if not rows * bits[0] <= target <= rows * bits[1]:
            raise ValueError("reference byte budget is outside the requested TQ range")
        delta = target - int(qs.sum())
        while delta:
            eligible = np.flatnonzero(qs < bits[1] if delta > 0 else qs > bits[0])
            chosen = rng.permutation(eligible)[:min(abs(delta), len(eligible))]
            if delta > 0:
                qs[chosen] += 1
                delta -= len(chosen)
            else:
                qs[chosen] -= 1
                delta += len(chosen)
    desc = np.r_[np.uint64(0), np.cumsum(width * qs[:-1] // 32)]
    desc |= (qs - 1) << np.uint64(32)
    desc |= np.uint64(np.float16(.02).view(np.uint16)) << np.uint64(40)
    n = (rows * ((width + gs - 1) // gs) + 1) // 2
    scales = np.ones(width, np.float16) if columns else rng.integers(0, 256, n, dtype=np.uint8)
    data = rng.integers(0, 256, int((width * qs // 8).sum()), dtype=np.uint8)
    return TQTensor(rows, width, gs, columns, desc, scales, data)


def expert_bytes(tensor, logical, experts):
    costs = np.zeros(9, np.int64)
    for q in range(1, 9):
        words = set()
        for col in range(0, logical, 8):
            first = (248 - (col % 256)) * q // 32
            for j in range(3):
                words.add((col // 256) * (8 * q) + (first + j) % (8 * q))
        costs[q] = len(words) * 4 + 8
    per_row = costs[tensor.bits]
    per_expert = per_row.reshape(experts, -1).sum(-1)
    out = tensor.rows // experts
    if tensor.columns:
        per_expert += logical * 2
    elif tensor.gs >= 256:
        per_expert += out * ((logical + tensor.gs - 1) // tensor.gs) * 4
    else:
        starts = np.arange(tensor.rows, dtype=np.int64) * tensor.groups
        first = starts // 2
        last = (starts + (logical + tensor.gs - 1) // tensor.gs - 1) // 2
        shared = np.r_[False, first[1:] == last[:-1]]
        shared[::out] = False
        per_expert += (last - first + 1 - shared).reshape(experts, out).sum(-1)
    return per_expert


def reference_expert_budgets(store, name, experts):
    record = store.records[name]
    view = store.blob_view(record)
    try:
        if record.dtype != "MFE" or bytes(view[:4]) != b"MFE1":
            raise ValueError("reference must contain native MFE expert tensors")
        _, count, _, _, pools = struct.unpack_from("<4sIIII", view)
        if experts > count:
            raise ValueError("fixture experts exceed the reference model")
        budgets = np.zeros(experts, np.float64)
        offset = 20
        for _ in range(pools):
            n, dtype_bytes, payload_bytes, runtime_bytes = struct.unpack_from("<IIQQ", view, offset)
            offset += 24
            ids = np.frombuffer(view, "<i4", n, offset)
            budgets[ids[ids < experts]] = (payload_bytes + runtime_bytes) / n
            del ids
            offset += n * 4 + dtype_bytes + runtime_bytes + payload_bytes
        return budgets
    finally:
        view.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", choices=("qwen", "deepseek"), default="qwen")
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--full-experts", action="store_true")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--lanes", type=int, nargs="+", default=[32, 16])
    p.add_argument("--write-fixture")
    p.add_argument("--reference-qwen")
    p.add_argument("--reference-deepseek")
    a = p.parse_args()
    if a.write_fixture:
        from mfq.formats.header import FileHeader
        from mfq.formats.io import open_mmap, save
        tensors, shapes, sizes, expert_sizes = {}, {}, {}, {}
        budgets = {}
        for label, h, inter, topk in (("qwen", 2560, 640, 10), ("deepseek", 5120, 2304, 6)):
            experts = (512 if label == "qwen" else 384) if a.full_experts else a.experts
            path = getattr(a, "reference_" + label)
            reference = open_mmap(path) if path else None
            for proj, out, width in (("gate", inter, h), ("up", inter, h), ("down", h, inter)):
                name = f"{label}.{proj}"
                budget = None
                if reference:
                    per_expert = reference_expert_budgets(reference,
                        f"model.block.0.mlp.experts.{proj}.weight", experts)
                    budget = float(per_expert.sum())
                t = random_tensor(experts * out, width, seed=20261022 + (proj == "up"), byte_budget=budget)
                tensors[name] = t
                shapes[name] = [experts, out, width]
                expert_sizes[name] = expert_bytes(t, width, experts).tolist()
                sizes[name] = np.mean(expert_sizes[name]) * topk
                if budget is not None:
                    budgets[name] = dict(reference_bytes=budget, tq_bytes=t.nbytes,
                        logical_bpw=t.nbytes * 8 / (experts * out * width),
                        runtime_bytes=t.nbytes + (t.rows * t.groups * 4 - t.scales.nbytes
                            if not t.columns and t.gs >= 256 else 0))
            if reference:
                reference.close()
        for name, columns, gs in (("check0", False, 16), ("check1", True, 512), ("check2", False, 512)):
            t = random_tensor(27, 513, seed=145, gs=gs, columns=columns)
            if name == "check2":
                from dataclasses import replace
                anchors = np.random.default_rng(72).uniform(.001, .02, t.rows).astype(np.float16)
                anchors[:2] = [0., 2**-20]
                desc = t.descriptors & ~(np.uint64(65535) << np.uint64(40))
                desc |= anchors.view(np.uint16).astype(np.uint64) << np.uint64(40)
                t = replace(t, descriptors=desc)
            tensors[name] = t
            tensors[name + ".reference"] = t.dequantize(513)
            shapes[name] = [27, 513]
        for projection, shape in {
            "experts.gate.weight": (8, 128, 256), "experts.up.weight": (8, 128, 256),
            "experts.down.weight": (8, 256, 128), "router.weight": (8, 256),
            "shared_expert.gate.weight": (128, 256), "shared_expert.up.weight": (128, 256),
            "shared_expert.down.weight": (256, 128),
        }.items():
            name = "unit.mlp." + projection
            tensors[name] = random_tensor(int(np.prod(shape[:-1])), shape[-1], seed=123)
            shapes[name] = list(shape)
        tensors["unit.mlp.router.bias"] = np.zeros(8, np.float32)
        save(a.write_fixture, FileHeader(version=2, model_arch="tq_test",
            extra={"tq_shapes": shapes, "benchmark_bytes": sizes,
                   "benchmark_expert_bytes": expert_sizes, "reference_budgets": budgets,
                   "synthetic": True}), tensors)
        print(json.dumps(budgets, indent=2))
        return
    h, inter, topk = (2560, 640, 10) if a.architecture == "qwen" else (5120, 2304, 6)
    for projection, out, width in (("gate", inter, h), ("up", inter, h), ("down", h, inter)):
        t = random_tensor(a.experts * out, width, seed=20261022)
        op = Projection(t, input_columns=width, experts=a.experts)
        rng = np.random.default_rng(42)
        schedule = np.stack([rng.choice(a.experts, topk, replace=False) for _ in range(a.steps)])
        ids = [mx.array(row[None, :].astype(np.int32)) for row in schedule]
        x = mx.array(rng.normal(size=(1 if projection != "down" else topk, width)).astype(np.float16))
        mx.eval(x, *ids)
        sizes = expert_bytes(t, width, a.experts)
        read = np.mean([sizes[row].sum() for row in schedule])
        for lanes in a.lanes:
            for i in range(min(8, a.steps)):
                mx.eval(op._run(x, ids[i], lanes=lanes))
            times = []
            for repeat in range(a.repeat):
                start = time.perf_counter()
                for route in ids:
                    mx.eval(op._run(x, route, lanes=lanes))
                ms = (time.perf_counter() - start) * 1000 / a.steps
                times.append(ms)
                print(json.dumps(dict(architecture=a.architecture, projection=projection,
                    out=out, k=width, experts=a.experts, topk=topk, lanes=lanes,
                    repeat=repeat, ms=ms, bytes=read, gb_s=read / ms / 1e6,
                    boundary="Python graph+dispatch+GPU completion", mtp=False)), flush=True)
            print(json.dumps(dict(summary=True, projection=projection, lanes=lanes,
                                  median_ms=float(np.median(times)))), flush=True)


if __name__ == "__main__":
    main()
