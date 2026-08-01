import numpy as np

from mfq.formats.nint8_one import quantize_nint8_one


def test_quantize_nint8_one_matches_q8_1_rounding_and_sum_rule():
    values = np.zeros((1, 32), dtype=np.float32)
    values[0, :5] = [127.0, 63.5, -63.5, 0.5, -0.5]

    block = quantize_nint8_one(values)

    np.testing.assert_array_equal(
        block.q[0, 0, :5],
        np.array([127, 64, -64, 1, -1], dtype=np.int8),
    )
    assert block.d.dtype == np.float16
    assert block.d[0, 0] == np.float16(1.0)
    assert block.s.dtype == np.float16
    assert block.s[0, 0] == np.float16(127.0)
    np.testing.assert_array_equal(
        block.reconstructed[0, :5],
        np.array([127.0, 64.0, -64.0, 1.0, -1.0], dtype=np.float16),
    )


def test_quantize_nint8_one_zero_group_is_all_zero():
    block = quantize_nint8_one(np.zeros((2, 32), dtype=np.float32))

    np.testing.assert_array_equal(block.q, np.zeros((2, 1, 32), dtype=np.int8))
    np.testing.assert_array_equal(block.d, np.zeros((2, 1), dtype=np.float16))
    np.testing.assert_array_equal(block.s, np.zeros((2, 1), dtype=np.float16))
    np.testing.assert_array_equal(
        block.reconstructed, np.zeros((2, 32), dtype=np.float16)
    )


def test_quantize_nint8_one_pads_tail_without_exposing_padding():
    values = np.array([[1.0, -1.0, 0.25]], dtype=np.float32)

    block = quantize_nint8_one(values)

    assert block.q.shape == (1, 1, 32)
    np.testing.assert_array_equal(block.q[0, 0, 3:], np.zeros(29, dtype=np.int8))
    assert block.reconstructed.shape == values.shape
    np.testing.assert_allclose(
        block.reconstructed.astype(np.float32),
        np.array([[1.0, -1.0, 32.0 / 127.0]], dtype=np.float32),
        atol=5e-4,
        rtol=0.0,
    )


def test_quantize_nint8_one_uses_float_scale_for_codes_and_fp16_scale_for_decode():
    values = np.zeros((1, 32), dtype=np.float32)
    values[0, :3] = [1.0, 0.5, -0.5]

    block = quantize_nint8_one(values)

    np.testing.assert_array_equal(
        block.q[0, 0, :3], np.array([127, 64, -64], dtype=np.int8)
    )
    stored_d = np.float16(np.float32(1.0 / 127.0))
    assert block.d[0, 0] == stored_d
    np.testing.assert_array_equal(
        block.reconstructed[0, :3],
        (
            np.array([127, 64, -64], dtype=np.float32)
            * np.float32(stored_d)
        ).astype(np.float16),
    )


def test_quantize_nint8_one_rejects_nonfinite_or_empty_input():
    for values in (
        np.empty((0, 32), dtype=np.float32),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[np.inf]], dtype=np.float32),
    ):
        try:
            quantize_nint8_one(values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {values!r}")
