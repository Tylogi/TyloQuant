"""Embedding lookup kernels."""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def embedding(weight: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather rows from ``weight [vocab,D]`` for int64 token ids."""

    return ext().embedding_lookup_cuda(
        weight.contiguous(),
        token_ids.contiguous().to(device=weight.device, dtype=torch.int64),
    )


def nint_embedding(g: dict, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather and dequantize selected NINT embedding rows."""

    if g.get("q") is not None and g.get("d_eff") is not None and g.get("m_eff") is not None:
        return ext().nint_embedding_lookup_cuda(
            g["q"],
            g["d_eff"].contiguous(),
            g["m_eff"].contiguous(),
            token_ids.contiguous().to(device=g["q"].device, dtype=torch.int64),
            int(g["neuron_len"]),
            int(g["gs"]),
        )
    if g.get("sub_scale") is not None:
        if int(g.get("bits", 4)) != 4:
            return ext().nint_embedding_lookup_packed_compact_bits_cuda(
                g["q_packed"],
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                token_ids.contiguous().to(device=g["q_packed"].device, dtype=torch.int64),
                int(g["neuron_len"]),
                int(g["gs"]),
                int(g["bits"]),
            )
        return ext().nint_embedding_lookup_packed_compact_cuda(
            g["q_packed"],
            g["sub_scale"],
            g["sub_min"],
            g["neuron_scale"],
            g["neuron_min"],
            token_ids.contiguous().to(device=g["q_packed"].device, dtype=torch.int64),
            int(g["neuron_len"]),
            int(g["gs"]),
        )
    return ext().nint_embedding_lookup_packed_eff_cuda(
        g["q_packed"],
        g["eff_pair_h"].contiguous(),
        token_ids.contiguous().to(device=g["q_packed"].device, dtype=torch.int64),
        int(g["neuron_len"]),
        int(g["gs"]),
    )
