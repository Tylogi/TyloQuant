"""Regression tests for large packed-NINT Metal bit addresses.

The largest Qwen3.6 embedding/lm-head has more than 2**32 packed bits, but
still has a byte offset that fits in Metal's 32-bit ``uint``.  These tests
exercise the address algebra directly, so they do not need to allocate the
corresponding 0.6--1.3 GB packed buffer.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_NINT = _ROOT / "mfq/kernels/metal/nint.py"
_CPP_NINT = _ROOT / "cpp_runtime/metal/mlx_nint.cpp"
_CPP_GROUPED = _ROOT / "cpp_runtime/metal/mlx_grouped_linear.cpp"
_PYTHON_MOE = _ROOT / "mfq/kernels/metal/moe.py"
_CPP_MOE = _ROOT / "cpp_runtime/metal/mlx_moe.cpp"
_PYTHON_VQ = _ROOT / "mfq/kernels/metal/vq.py"
_CPP_VQ = _ROOT / "cpp_runtime/metal/mlx_vq.cpp"
_PYTHON_TPQ = _ROOT / "mfq/kernels/metal/tpq.py"
_CPP_CCCP = _ROOT / "cpp_runtime/metal/mlx_cccp.cpp"

_PACKED_METAL_SOURCES = (
    _PYTHON_NINT,
    _CPP_NINT,
    _CPP_GROUPED,
    _PYTHON_MOE,
    _CPP_MOE,
    _PYTHON_VQ,
    _CPP_VQ,
    _PYTHON_TPQ,
    _CPP_CCCP,
)

_ADDRESS_PATTERN = re.compile(
    r"uint residual_bits = \(value_index & 7u\) \* bits;\s*"
    r"uint byte_index =\s*"
    r"\(value_index >> 3\) \* bits \+ \(residual_bits >> 3\);\s*"
    r"uint shift = residual_bits & 7u;"
)

_UNSAFE_ADDRESS_PATTERN = re.compile(
    r"uint\s+(?:bit_index|bit_offset)\s*=\s*"
    r"(?:value_index|index)\s*\*\s*bits"
    r"|\((?:value_index|quantized_index|index)\s*\*\s*[1-8]u\)\s*>>\s*3"
)

_BASED_ADDRESS_PATTERN = re.compile(
    r"uint residual_bits = \((?P<index>value_index|index) & 7u\) \* bits;\s*"
    r"uint byte_offset =\s*"
    r"(?:byte_base\s*\+\s*)?"
    r"\((?P=index) >> 3\) \* bits\s*\+\s*\(residual_bits >> 3\);\s*"
    r"uint shift = residual_bits & 7u;"
)


def _function_body(source: str, name: str) -> str:
    """Return one non-nested Metal helper body from an embedded source string."""

    signature = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", source)
    assert signature is not None, f"missing Metal helper {name}"
    body_start = signature.end()
    body_end = source.find("\n}", body_start)
    assert body_end >= 0, f"unterminated Metal helper {name}"
    return source[body_start:body_end]


def _host_address(value_index: int, bits: int) -> tuple[int, int]:
    residual_bits = (value_index & 7) * bits
    byte_index = (value_index >> 3) * bits + (residual_bits >> 3)
    return byte_index, residual_bits & 7


def _wrapped_old_address(value_index: int, bits: int) -> tuple[int, int]:
    bit_index = (value_index * bits) & 0xFFFF_FFFF
    return bit_index >> 3, bit_index & 7


@pytest.mark.parametrize(
    ("path", "helper"),
    [
        (_PYTHON_NINT, "mfq_nint_read_bits"),
        (_CPP_NINT, "mfq_nint_read_bits"),
        (_CPP_GROUPED, "mfq_grouped_nint_read_bits"),
    ],
)
def test_all_nint_metal_helpers_use_overflow_safe_address(
    path: Path,
    helper: str,
):
    """Lock all three production helpers to the overflow-safe decomposition."""

    body = _function_body(path.read_text(), helper)
    assert _ADDRESS_PATTERN.search(body)
    assert not re.search(
        r"uint\s+(?:bit_index|bit_offset)\s*=\s*value_index\s*\*\s*bits",
        body,
    )


@pytest.mark.parametrize(
    "path",
    [_PYTHON_NINT, _CPP_NINT, _PYTHON_MOE, _CPP_MOE],
)
def test_specialized_nint3_nint6_addresses_do_not_multiply_before_shift(
    path: Path,
):
    """The fast GEMV/GEMM specializations must not reintroduce the same bug."""

    source = path.read_text()
    for bits in (3, 6):
        assert f"(quantized_index * {bits}u) >> 3" not in source
        expected = re.compile(
            rf"\(quantized_index >> 3\) \* {bits}u\s*"
            rf"\+ \(\(\(quantized_index & 7u\) \* {bits}u\) >> 3\)"
        )
        assert expected.search(source)


@pytest.mark.parametrize("path", _PACKED_METAL_SOURCES)
def test_no_metal_packed_index_multiplies_bits_before_reducing(path: Path):
    """Audit all current NINT/VQ/CCCP embedded Metal address helpers."""

    source = path.read_text()
    match = _UNSAFE_ADDRESS_PATTERN.search(source)
    assert match is None, (
        f"{path.relative_to(_ROOT)} still has overflow-prone Metal address "
        f"algebra: {match.group(0) if match else ''}"
    )


@pytest.mark.parametrize(
    ("path", "minimum_safe_addresses"),
    [
        (_PYTHON_NINT, 1),
        (_CPP_NINT, 1),
        (_CPP_GROUPED, 2),
        (_PYTHON_MOE, 1),
        (_CPP_MOE, 2),
        (_PYTHON_VQ, 2),
        (_CPP_VQ, 2),
    ],
)
def test_generic_packed_helpers_keep_quotient_remainder_formula(
    path: Path,
    minimum_safe_addresses: int,
):
    """Ensure generic NINT/VQ helpers retain the exact safe decomposition."""

    assert len(_ADDRESS_PATTERN.findall(path.read_text())) >= minimum_safe_addresses


@pytest.mark.parametrize(
    ("path", "safe_addresses"),
    [
        (_PYTHON_TPQ, 3),
        (_CPP_CCCP, 1),
        (_CPP_MOE, 1),
    ],
)
def test_cccp_packed_helpers_keep_quotient_remainder_formula(
    path: Path,
    safe_addresses: int,
):
    """Lock both standalone and MoE CCCP packed-index helpers."""

    assert len(_BASED_ADDRESS_PATTERN.findall(path.read_text())) >= safe_addresses


@pytest.mark.parametrize("bits", range(2, 9))
def test_large_nint_address_host_algebra_is_exact(bits: int):
    """Cover a value whose bit offset exceeds 2**32 for every relevant width."""

    value_index = (1 << 32) // bits + 19
    assert value_index <= 0xFFFF_FFFF
    assert value_index * bits > 0xFFFF_FFFF

    actual = _host_address(value_index, bits)
    expected = divmod(value_index * bits, 8)
    assert actual == expected
    assert _wrapped_old_address(value_index, bits) != expected
    assert actual[0] <= 0xFFFF_FFFF


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_qwen36_vocab_projection_crosses_old_uint_bit_offset(bits: int):
    """Use the final scalar of Qwen3.6's 248320 x 5120 lm-head matrix."""

    value_index = 248_320 * 5_120 - 1
    assert value_index * bits > 0xFFFF_FFFF
    assert _host_address(value_index, bits) == divmod(value_index * bits, 8)
    assert _wrapped_old_address(value_index, bits) != _host_address(
        value_index,
        bits,
    )


def test_large_nint_address_exact_metal_probe():
    """Run the production address statements on Metal without a large buffer."""

    mx = pytest.importorskip("mlx.core")
    try:
        mx.device_info()
    except RuntimeError:
        pytest.skip("Metal device unavailable")

    source_text = _PYTHON_NINT.read_text()
    helper = _function_body(source_text, "mfq_nint_read_bits")
    match = _ADDRESS_PATTERN.search(helper)
    assert match is not None

    # This is the exact address block extracted from the production helper.
    # The probe intentionally does not dereference q_packed.
    address_source = match.group(0)
    probe = mx.fast.metal_kernel(
        name="mfq_test_nint_large_bit_address",
        input_names=["value_indices", "bit_widths"],
        output_names=["byte_offsets", "shifts"],
        source=f"""
            uint linear = thread_position_in_grid.x;
            if (linear >= uint(COUNT)) {{
                return;
            }}
            uint value_index = uint(value_indices[linear]);
            uint bits = uint(bit_widths[linear]);
            {address_source}
            byte_offsets[linear] = byte_index;
            shifts[linear] = shift;
        """,
    )

    qwen_last = 248_320 * 5_120 - 1
    widths = np.asarray([4, 5, 6, 7, 8], dtype=np.uint32)
    indices = np.asarray(
        [
            qwen_last,
            qwen_last,
            qwen_last,
            (1 << 32) // 7 + 19,
            qwen_last,
        ],
        dtype=np.uint32,
    )
    byte_offsets, shifts = probe(
        inputs=[mx.array(indices), mx.array(widths)],
        template=[("COUNT", len(indices))],
        grid=(len(indices), 1, 1),
        threadgroup=(len(indices), 1, 1),
        output_shapes=[indices.shape, indices.shape],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    mx.eval(byte_offsets, shifts)

    expected = np.asarray(
        [_host_address(int(index), int(bits)) for index, bits in zip(indices, widths)],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(np.asarray(byte_offsets), expected[:, 0])
    np.testing.assert_array_equal(np.asarray(shifts), expected[:, 1])
