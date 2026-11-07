# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Experimental MLX row Trellis fitting and compressed M1--6 projections."""

from functools import lru_cache
import math
import operator

import mlx.core as mx
import numpy as np

from mfq.quantize.row_trellis import (
    CONSTANTS, GAINS, RowTrellisTensor, allocate_bits, assemble_rows,
    fixed_nbytes, validate_shape,
)
from . import _row_trellis_kernels as kernels


@lru_cache(None)
def _search_kernel():
    return mx.fast.metal_kernel(
        name="mfq_row_trellis_search",
        input_names=["input_tiles", "config", "codebook", "objectives"],
        output_names=["quantized", "indices", "temp_costs", "temp_edges"],
        header=kernels.HEADER, source=kernels.SEARCH,
    )


@lru_cache(None)
def _anchor_kernel():
    return mx.fast.metal_kernel(
        name="mfq_row_trellis_anchor", input_names=["input"], output_names=["output"],
        source=kernels.ANCHOR,
    )


def round_anchor(value):
    return _anchor_kernel()(
        inputs=[value], template=[("COUNT", value.size)], grid=(value.size, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[value.shape], output_dtypes=[mx.float16],
    )[0]


@lru_cache(None)
def encoder_codebook(state_bits):
    kernel = mx.fast.metal_kernel(
        name="mfq_row_trellis_codebook", input_names=["states"], output_names=["values"],
        header=kernels.HEADER,
        source="uint i=thread_position_in_grid.x; if(i<COUNT) values[i]=exl3_decode_state(states[i],2);",
    )
    count = 1 << state_bits
    result = kernel(
        inputs=[mx.arange(count, dtype=mx.uint32)], template=[("COUNT", count)],
        grid=(count, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(count,)], output_dtypes=[mx.float16],
    )[0]
    mx.eval(result)
    return result

@lru_cache(None)
def pack_kernel():
    return mx.fast.metal_kernel(
        name="mfq_row_trellis_pack", input_names=["normalized", "states", "config"],
        output_names=["packed"], header=kernels.HEADER, source=kernels.PACK,
    )


def search(tiles,q,step,state_bits=16,*,objectives=None,objective_candidates=1):
    q = operator.index(q)
    if not 1 <= q <= 8 or state_bits not in (10, 16):
        raise ValueError("q must be in [1, 8] and state_bits must be 10 or 16")
    if tiles.ndim != 2 or tiles.shape[0] == 0 or tiles.shape[1] not in (128, 256):
        raise ValueError("tiles must have shape (count, 128 or 256)")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    objective_candidates = operator.index(objective_candidates)
    if objective_candidates < 1:
        raise ValueError("objective_candidates must be positive")
    tiles = tiles.astype(mx.float32)
    base=min(q,3)
    edges=1<<(state_bits-base)
    length=tiles.shape[-1]
    threads=min(1024 if state_bits==16 else 512,max(32,edges))
    local=edges<=2048
    packed_bits=base if base<3 else 4
    per_word=32//packed_bits
    words=edges//per_word
    groups=min(tiles.size//length,256,(256<<20)//(length*words*4+(0 if local else 8*edges)))
    if objectives is None:
        objectives=mx.ones((length,),mx.float32)
    if objectives.size == 0 or objectives.size % length or objectives.ndim not in (1, 2):
        raise ValueError("objectives must contain complete tile-sized groups")
    objectives = objectives.astype(mx.float32)
    result=_search_kernel()(inputs=[mx.contiguous(tiles),mx.array([step,1./step if q>3 else 0.],mx.float32),
        encoder_codebook(state_bits),mx.contiguous(objectives)],
        template=[('BITS',base),('REFINE_BITS',q-base),('CODEBOOK',2),('TILE_LEN',length),
            ('STATE_SHIFT',state_bits-base),('MAX_SYMBOLS',1<<base),('EDGES',edges),('THREADS',threads),
            ('NUM_TILES',tiles.size//length),('ACTIVE_GROUPS',groups),('LOCAL_COSTS',local),
            ('LOCAL_COST_COUNT',2*edges if local else 1),('OBJECTIVE_CANDIDATES',objective_candidates),
            ('OBJECTIVE_GROUPS',objectives.size//length),('PACK_BITS',packed_bits),
            ('PER_WORD',per_word),('TRACE_WORDS',words)],grid=(groups*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[tiles.shape,tiles.shape,(1,) if local else (groups,2,edges),(groups,length,words)],
        output_dtypes=[mx.float32,mx.int16,mx.float32,mx.uint32])
    mx.eval(result[0],result[1])
    return result[:2]


def refine_fixed(normalized,indices,q,step,state_bits):
    base=encoder_codebook(state_bits).astype(mx.float32)[indices.view(mx.uint16).astype(mx.uint32)]
    if q<=3:
        return base
    maximum=float((1<<(q-3))-1)
    r=mx.clip(mx.round((normalized-base)/step+maximum*.5),0.,maximum)
    return base+(r-maximum*.5)*step


def _encode_joint(weight,q,constants,*,state_bits=16,group_size=256,iterations=2,column_factors=None):
    rows,columns=weight.shape
    assert columns%group_size==0
    groups=columns//group_size
    w=weight.reshape(rows,groups,group_size)
    factors=mx.ones((group_size,),mx.float32) if column_factors is None else column_factors.astype(mx.float32)
    field=factors.reshape(-1,group_size)
    if field.shape[0]==1:
        field=mx.broadcast_to(field,(groups,group_size))
    gain=constants[str(q)]['gain']
    step=constants[str(q)]['step']
    lut=mx.array(GAINS).astype(mx.float32)
    whitened=w/field[None,:,:]
    anchor=round_anchor(mx.sqrt(mx.mean(whitened*whitened,axis=(1,2)))/gain)
    best_error=mx.full((rows,),float('inf'))
    best_anchor=anchor
    best_gain=mx.zeros((rows,groups),mx.uint8)
    best_indices=mx.zeros((rows,groups,group_size),mx.int16)
    best_z=mx.zeros((rows,groups,group_size),mx.float32)
    trace=[]
    for iteration in range(iterations):
        candidate_codes=mx.broadcast_to(mx.arange(16,dtype=mx.uint32),(rows,groups,16))
        scales=anchor.astype(mx.float32)[:,None,None]*lut[candidate_codes]
        normalized=w[:,:,None,:]/mx.maximum(scales[:,:,:,None]*field[None,:,None,:],mx.array(1e-20))
        z,indices=search(normalized.reshape(-1,group_size),q,step,state_bits,objectives=factors,objective_candidates=16)
        z=z.reshape(rows,groups,16,group_size)
        indices=indices.reshape(rows,groups,16,group_size)
        costs=mx.sum((z*scales[:,:,:,None]*field[None,:,None,:]-w[:,:,None,:])**2,axis=3)
        best_candidate=mx.argmin(costs,axis=2).astype(mx.uint32)
        selected=mx.take_along_axis(candidate_codes,best_candidate[:,:,None],axis=2).squeeze(2).astype(mx.uint8)
        take=mx.broadcast_to(best_candidate[:,:,None,None],(rows,groups,1,group_size))
        selected_z=mx.take_along_axis(z,take,axis=2).squeeze(2)
        selected_indices=mx.take_along_axis(indices,take,axis=2).squeeze(2)
        selected_basis=selected_z*lut[selected.astype(mx.uint32)][:,:,None]*field[None,:,:]
        row_error=mx.sum((selected_basis*anchor.astype(mx.float32)[:,None,None]-w)**2,axis=(1,2))
        improve=row_error<best_error
        best_error=mx.minimum(best_error,row_error)
        best_anchor=mx.where(improve,anchor,best_anchor)
        best_gain=mx.where(improve[:,None],selected,best_gain)
        best_indices=mx.where(improve[:,None,None],selected_indices,best_indices)
        best_z=mx.where(improve[:,None,None],selected_z,best_z)
        mx.eval(best_error,best_anchor,best_gain,best_indices,best_z)
        trace.append(float(mx.sum(best_error).item()))
        numerator=mx.sum(w*selected_basis,axis=(1,2))
        denominator=mx.sum(selected_basis**2,axis=(1,2))
        anchor=round_anchor(mx.maximum(numerator,0.)/mx.maximum(denominator,mx.array(1e-30)))
        del z,indices,normalized,costs,take,selected_indices,selected_z
    scales=best_anchor.astype(mx.float32)[:,None]*lut[best_gain.astype(mx.uint32)]
    normalized=(w/mx.maximum(scales[:,:,None]*field[None,:,:],mx.array(1e-20))).reshape(-1,group_size)
    words=rows*columns*q//32
    packed=pack_kernel()(inputs=[normalized,best_indices.reshape(-1,group_size),mx.array([step,1./step],mx.float32)],
        template=[('Q',q),('BASE_BITS',min(q,3)),('REFINE_BITS',max(0,q-3)),('TILE_LEN',group_size),('WORDS',words)],
        grid=(words,1,1),threadgroup=(256,1,1),output_shapes=[(rows,columns*q//32)],output_dtypes=[mx.uint32])[0]
    reconstructed=best_z*scales[:,:,None]*field[None,:,:]
    mx.eval(packed,reconstructed,best_anchor,best_gain)
    return packed,best_anchor,best_gain,reconstructed,trace


def _encode_coarse_fine(weight,q,constants,*,state_bits=16,group_size=256,iterations=1,column_factors=None):
    _,anchor,selected,coarse,trace=_encode_joint(weight,q,constants,state_bits=10,
        group_size=group_size,iterations=iterations,column_factors=column_factors)
    rows,columns=weight.shape
    groups=columns//group_size
    step=constants[str(q)]['step']
    w=weight.reshape(rows,groups,group_size)
    factors=mx.ones((group_size,),mx.float32) if column_factors is None else column_factors.astype(mx.float32)
    field=mx.broadcast_to(factors,(columns,)).reshape(groups,group_size) if factors.size==1 else factors.reshape(-1,group_size)
    if field.shape[0]==1:
        field=mx.broadcast_to(field,(groups,group_size))
    lut=mx.array(GAINS).astype(mx.float32)
    scale=anchor.astype(mx.float32)[:,None]*lut[selected.astype(mx.uint32)]
    normalized=(w/mx.maximum(scale[:,:,None]*field[None,:,:],mx.array(1e-20))).reshape(-1,group_size)
    z,indices=search(normalized,q,step,state_bits,objectives=factors)
    z=z.reshape(rows,groups,group_size)
    indices=indices.reshape(rows,groups,group_size)
    basis=z*lut[selected.astype(mx.uint32)][:,:,None]*field[None,:,:]
    optimum=round_anchor(mx.maximum(mx.sum(w*basis,axis=(1,2)),0.)/mx.maximum(mx.sum(basis*basis,axis=(1,2)),mx.array(1e-30)))
    old_error=mx.sum((z*scale[:,:,None]*field[None,:,:]-w)**2,axis=(1,2))
    new_error=mx.sum((basis*optimum.astype(mx.float32)[:,None,None]-w)**2,axis=(1,2))
    anchor=mx.where(new_error<old_error,optimum,anchor)
    scale=anchor.astype(mx.float32)[:,None]*lut[selected.astype(mx.uint32)]
    normalized=(w/mx.maximum(scale[:,:,None]*field[None,:,:],mx.array(1e-20))).reshape(-1,group_size)
    z=refine_fixed(normalized,indices.reshape(-1,group_size),q,step,state_bits).reshape(rows,groups,group_size)
    reconstructed=z*scale[:,:,None]*field[None,:,:]
    packed=pack_kernel()(inputs=[normalized,indices.reshape(-1,group_size),mx.array([step,1./step],mx.float32)],
        template=[('Q',q),('BASE_BITS',min(q,3)),('REFINE_BITS',max(0,q-3)),('TILE_LEN',group_size),
            ('WORDS',rows*columns*q//32)],grid=(rows*columns*q//32,1,1),threadgroup=(256,1,1),
        output_shapes=[(rows,columns*q//32)],output_dtypes=[mx.uint32])[0]
    mx.eval(packed,reconstructed)
    trace.append(float(mx.sum((reconstructed-w)**2).item()))
    return packed,anchor,selected,reconstructed,trace


def quantize(weight, precisions=None, *, target_bpw=None, group_size=256,
             state_bits=16, iterations=1, column_scaling=True):
    """Fit explicit per-row q or an exact byte-budget SSE allocation.

    The input and returned container are CPU arrays. Fitting uses Metal; this
    prototype is deliberately not installed in the serving dispatch or CLI.
    """
    source = np.asarray(weight, dtype=np.float32)
    rows, columns = validate_shape(source.shape, group_size, state_bits)
    if not np.isfinite(source).all():
        raise ValueError("weight must contain finite values")
    iterations = operator.index(iterations)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if (precisions is None) == (target_bpw is None):
        raise ValueError("specify either precisions or target_bpw, not both")
    if target_bpw is not None:
        if not math.isfinite(target_bpw) or target_bpw <= 0:
            raise ValueError("target_bpw must be finite and positive")
        target_bytes = math.floor(target_bpw * source.size / 8)
        max_q_sum = (target_bytes - fixed_nbytes(source.shape, group_size, column_scaling)) // (columns // 8)
        if max_q_sum < rows:
            raise ValueError("byte budget cannot fit q1 and container metadata")
        candidates = range(1, 9)
    else:
        qs = np.asarray(precisions)
        if qs.ndim == 0:
            qs = np.full(rows, qs.item())
        if qs.shape != (rows,) or qs.dtype.kind not in "ui" or np.any((qs < 1) | (qs > 8)):
            raise ValueError("one integer precision in [1, 8] is required per row")
        qs = qs.astype(np.uint8)
        candidates = np.unique(qs)

    column_codes = None
    column_factors = None
    if column_scaling:
        rms = np.sqrt(np.mean(source.astype(np.float64)**2, axis=0))
        relative = rms / max(float(np.mean(rms)), 1e-30)
        column_codes = np.argmin(
            np.abs(np.log2(np.maximum(relative[:, None], 1e-30))
                   - np.log2(GAINS.astype(np.float64))), axis=1,
        ).astype(np.uint8)
        column_factors = mx.array(GAINS[column_codes])
    weights = mx.array(np.ascontiguousarray(source))
    fitted, expected = {}, {}
    errors = np.empty((rows, 8), np.float64)
    for q in candidates:
        q = int(q)
        packed, anchor, gains, reconstructed, _ = _encode_coarse_fine(
            weights, q, CONSTANTS, state_bits=state_bits, group_size=group_size,
            iterations=iterations, column_factors=column_factors,
        )
        packed, anchor, gains, reconstructed = map(np.array, (packed, anchor, gains, reconstructed))
        reconstructed = reconstructed.reshape(source.shape)
        fitted[q] = packed, anchor, gains
        expected[q] = reconstructed
        errors[:, q - 1] = np.sum(
            (reconstructed.astype(np.float64) - source.astype(np.float64))**2, axis=1,
        )
    if target_bpw is not None:
        qs = allocate_bits(errors, max_q_sum)
    buckets = {row: tuple(v[row] for v in fitted[int(qs[row])]) for row in range(rows)}
    if column_scaling:
        wanted = np.stack([expected[int(qs[row])][row] for row in range(rows)])
        basis = wanted.astype(np.float64) / GAINS[column_codes].astype(np.float64)
        desired = np.sum(source.astype(np.float64) * basis, axis=0) / np.maximum(
            np.sum(basis * basis, axis=0), 1e-30,
        )
        column_codes = np.argmin(
            np.abs(desired[:, None] - GAINS.astype(np.float64)), axis=1,
        ).astype(np.uint8)
    return assemble_rows(
        buckets, qs, source.shape, group_size=group_size, state_bits=state_bits,
        column_codes=column_codes,
    )


@lru_cache(None)
def _decode_kernel():
    return mx.fast.metal_kernel(
        name="mfq_row_trellis_decode",
        input_names=["payload", "rows", "blocks", "lut", "steps", "columns_scale"],
        output_names=["output"], header=kernels.HEADER, source=kernels.DECODE,
    )


@lru_cache(None)
def _projection_kernel():
    return mx.fast.metal_kernel(
        name="mfq_row_trellis_project",
        input_names=["payload", "descriptors", "blocks", "lut", "steps", "columns_scale", "input"],
        output_names=["output"], header=kernels.HEADER, source=kernels.PROJECT,
    )


def _inputs(tensor):
    if not isinstance(tensor, RowTrellisTensor):
        raise TypeError("expected a RowTrellisTensor")
    return [
        mx.array(tensor.payload), mx.array(tensor.descriptors), mx.array(tensor.blocks),
        mx.array(GAINS), mx.array([CONSTANTS[str(q)]["step"] for q in range(1, 9)], mx.float32),
        mx.array(GAINS[tensor.column_codes]) if tensor.column_codes.size else mx.ones((1,), mx.float16),
    ]


def decode(tensor):
    rows, columns = tensor.shape
    result = _decode_kernel()(
        inputs=_inputs(tensor),
        template=[("COUNT", rows * columns), ("COLUMNS", columns),
                  ("GROUP_SIZE", tensor.group_size), ("STATE_BITS", tensor.state_bits),
                  ("SCALE_COLUMNS", bool(tensor.column_codes.size))],
        grid=(rows * columns, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[tensor.shape], output_dtypes=[mx.float32],
    )[0]
    mx.eval(result)
    return result


def project(tensor, x):
    """Multiply M1--6 vectors by compressed weights without dense materialization."""
    if len(x.shape) != 2 or x.shape[1] != tensor.columns or not 1 <= x.shape[0] <= 6:
        raise ValueError("input must have shape (M, columns) with M in [1, 6]")
    rows, columns = tensor.shape
    m = x.shape[0]
    result = _projection_kernel()(
        inputs=[*_inputs(tensor), mx.contiguous(mx.array(x, dtype=mx.float32))],
        template=[("ROWS", rows), ("COLUMNS", columns), ("GROUP_SIZE", tensor.group_size),
                  ("STATE_BITS", tensor.state_bits), ("M", m), ("THREADS", 128),
                  ("SCALE_COLUMNS", bool(tensor.column_codes.size))],
        grid=(rows * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(m, rows)], output_dtypes=[mx.float32],
    )[0]
    mx.eval(result)
    return result
