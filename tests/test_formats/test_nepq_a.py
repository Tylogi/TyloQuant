import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.nepq import (
    NEPQ0_A,
    NEPQ0_S,
    NEPQ1_A,
    NEPQ1_S,
    dequantize_nepq,
    pack_nepq,
    unpack_nepq,
)
from tests.test_formats.test_nepq import _tensor


def _a_tensor(spec):
    base_spec = NEPQ0_S if spec is NEPQ0_A else NEPQ1_S
    tensor = unpack_nepq(pack_nepq(_tensor(base_spec)))
    base = dequantize_nepq(tensor)
    tensor.spec = spec
    blocks = tensor.residual_blocks_per_row
    dictionary = np.zeros((1024, 8), dtype=np.float16)
    dictionary[1] = np.linspace(-0.25, 0.25, 8, dtype=np.float16)
    tensor.residual_codebook = dictionary
    tensor.residual_first = np.full(
        tensor.shape[:2] + (blocks,),
        1 << spec.residual_position_bits,
        dtype=np.uint16,
    )
    expected = base.copy()
    expected[..., :8] += dictionary[1].astype(np.float32)
    if spec is NEPQ1_A:
        mask = np.zeros(tensor.shape[:2] + (blocks,), dtype=np.uint8)
        mask.reshape(-1)[::2] = 1
        second_record = np.uint16(1 | (1 << spec.residual_position_bits))
        tensor.residual_second_mask = mask
        tensor.residual_second_records = np.full(
            np.count_nonzero(mask), second_record, dtype=np.uint16
        )
        selected_rows = np.flatnonzero(mask.reshape(-1))
        flat_expected = expected.reshape(-1, tensor.neuron_len)
        flat_expected[selected_rows, 8:16] += dictionary[1].astype(np.float32)
        tensor.residual_padding_nbytes = 7
    return tensor, expected


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
def test_nepq_a_roundtrip_and_sparse_residual_decode(spec):
    tensor, expected = _a_tensor(spec)
    payload = pack_nepq(tensor)
    restored = unpack_nepq(payload)
    assert restored.spec is spec
    assert len(payload) == 36 + restored.payload_nbytes
    np.testing.assert_array_equal(restored.residual_first, tensor.residual_first)
    np.testing.assert_array_equal(restored.residual_codebook, tensor.residual_codebook)
    if spec is NEPQ1_A:
        np.testing.assert_array_equal(restored.residual_second_mask, tensor.residual_second_mask)
        np.testing.assert_array_equal(
            restored.residual_second_records,
            tensor.residual_second_records,
        )
        assert restored.residual_padding_nbytes == 7
    np.testing.assert_allclose(dequantize_nepq(restored), expected, rtol=0, atol=0)


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
def test_nepq_a_file_dtype_roundtrip(spec, tmp_path):
    tensor, expected = _a_tensor(spec)
    path = tmp_path / f"{spec.label}.mfq"
    io.save(
        path,
        FileHeader(model_arch="nepq-a-test", num_tensors=1),
        {"experts.weight": tensor},
    )
    _, store = io.load_mmap(path)
    try:
        assert store.records["experts.weight"].dtype == spec.label
        np.testing.assert_allclose(
            dequantize_nepq(store["experts.weight"]), expected, rtol=0, atol=0
        )
    finally:
        store.close()


def test_nepq1_a_rejects_second_mask_count_mismatch():
    tensor, _ = _a_tensor(NEPQ1_A)
    tensor.residual_second_records = tensor.residual_second_records[:-1]
    with pytest.raises(ValueError, match="does not match its mask"):
        pack_nepq(tensor)


def test_nepq_a_requires_hadamard_rotation():
    tensor, _ = _a_tensor(NEPQ0_A)
    tensor.rotation_block = 0
    tensor.rotation_seed = 0
    with pytest.raises(ValueError, match="requires a Hadamard"):
        pack_nepq(tensor)


def test_nepq_a_rejects_values_that_overflow_serialized_fp16():
    tensor, _ = _a_tensor(NEPQ0_A)
    tensor.neuron_scale = tensor.neuron_scale.astype(np.float32)
    tensor.neuron_scale.reshape(-1)[0] = 70_000.0
    with pytest.raises(ValueError, match="anchors must be finite"):
        pack_nepq(tensor)

    tensor, _ = _a_tensor(NEPQ0_A)
    tensor.residual_codebook = tensor.residual_codebook.astype(np.float32)
    tensor.residual_codebook[1, 0] = 70_000.0
    with pytest.raises(ValueError, match="dictionary must be finite"):
        pack_nepq(tensor)
