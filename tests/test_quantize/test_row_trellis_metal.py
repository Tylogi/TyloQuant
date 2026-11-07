import hashlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal device required", allow_module_level=True)

from mfq.quantize.metal.row_trellis import decode, project, quantize, search
from mfq.quantize.row_trellis import RowTrellisTensor, decode_cpu
from tests.test_quantize.test_row_trellis import example_tensor

# FP32 two-pass reference Viterbi results, before packed traceback/prefetch.
GOLDEN = {
    "128-10-1": "3e80ed75d884b0df9903068dc94ca2e3fc5803157d96c6a0375afb6465a59619",
    "128-10-2": "609296140de12fb954b242290910711a8501e8f575f9d2b0ab5e3295b6e79ce1",
    "128-10-3": "c62f17b72f595e55657ede428f22b5048acd087f3fa00236426fbdb2db48a108",
    "128-10-4": "5010931287c2a180dc1fec016ba6eda47dae559285eb1bc97e5e25e2e13fe2fc",
    "128-10-5": "504123460ae36436d199c81d71d49c6422bde3ed99d4d89018a10396711d4237",
    "128-10-6": "2d7d4a8bdd3f0cb37f65fe7c6602c657c07e4887b24b47514e2c585f5e993c4a",
    "128-10-7": "f03ce37292e68f23cd38e740a7b811e5acccfffb02c1ebe56a754dc538fc232a",
    "128-10-8": "eb336ae714a7226da69752e9dc5d005410e9244d5c9fc79847a6303a1417796f",
    "128-16-1": "212d56abf5da450009de68b3b40b8976ec8909ecd1abc159bf5cfb1cfc6a419f",
    "128-16-2": "bb56719e7cbe3b4e9e2fb970f4a161216e04c1bd081675438c40cba507e58461",
    "128-16-3": "3d8f9dbf6c4e2b9f11ccfcd0d0e777befc6ee764df93ae58deac3db803319ccf",
    "128-16-4": "d40f5e35d112b163421ea14032ac811f168cf06c8fac3a496992bebd07f90201",
    "128-16-5": "f7da0e3b59b12bbaed701659a7033b702921bc3c41025fa39ecd2856d8f7d8ef",
    "128-16-6": "406c3d3762d5dc565c79758d1d6f8c68b9f945691a7a09281ed6e6db754e6702",
    "128-16-7": "c7e942169d3e48c9e9701e907fd31e5c4b81037b6a886648e434a59ff61f7538",
    "128-16-8": "da6e08b40db5a32dced497870dd3bf1865a76e8fe93c99ffd332450785e8a0b3",
    "256-10-1": "6fdcf1de5b6c53da1ebe94b48a83b9b963da5b6d09cda87dcca2e1a4cb5536d6",
    "256-10-2": "fa94e6f8a12086c2bd62febe5482f51d3d93890e0ea88620d04f5324598285d2",
    "256-10-3": "f422836961cf024e3b8c48b292055345a0b391ba9b1794f762702e44a08c288a",
    "256-10-4": "ca7323032917114d04cd05ac0356cb5de45d42a330ea05cfcf8f16e46c86c811",
    "256-10-5": "894a1e35562edeb3903968b68a0d52c6893a96e21716a90f26be08b6f06a3c9d",
    "256-10-6": "0f3f4f7691664f2539bcf508b9ba94cf97cd760598a4e9bb99be6dcf02416f3b",
    "256-10-7": "2ed1bbbe7ffbee9237b912f72c965e47a9f753d788cd6035485a218b7366193c",
    "256-10-8": "416244ee78c16e9c91716639da7fb047544409942e55b9c9d22fb61a56509b82",
    "256-16-1": "d258081bcbf08acef1e01311e36ad16ede7b00be117747302364d5f930a72651",
    "256-16-2": "891c07233595c37b2b3939a83669e1bb792fb4930bd31788521056dd766a8411",
    "256-16-3": "bc7ef7379c8fc0a767d2d97c89c27dd610474de7b46368789e9e6b6e937e2334",
    "256-16-4": "e3a90cc1a9d13413112abcf0be712f9b7968311d504599bea1b7bd86bd17ff4d",
    "256-16-5": "86f561ae0655207dd20999b4151805cc78abd8509e44b53629f929900a880465",
    "256-16-6": "76276171286f8ee5e515f3517bf14cf0ebf9320f0ae3f7b4c19be6fa7447050d",
    "256-16-7": "a948b9869642107631d5d53061638ecbdb2688b9a8e59259e0bfbac44f6bd9ad",
    "256-16-8": "14be393c13aaf0b8feb007fd0c4ef508b509b325f38146fbd844ee60e640e1da"
}


@pytest.mark.parametrize("length", [128, 256])
@pytest.mark.parametrize("state_bits", [10, 16])
@pytest.mark.parametrize("q", range(1, 9))
def test_search_preserves_reference_output(length, state_bits, q):
    rng = np.random.default_rng(128519)
    source = rng.normal(size=(12, length)).astype(np.float32)
    source[0] = 0
    source[1] *= .001
    source[2] *= 100
    objectives = mx.array(rng.uniform(.25, 2., (3, length)).astype(np.float32))
    values, states = search(
        mx.array(source), q, .3, state_bits, objectives=objectives, objective_candidates=4,
    )
    values, states = np.array(values), np.array(states)
    digest = hashlib.sha256(states.astype("<i2").tobytes() + values.astype("<f4").tobytes()).hexdigest()
    assert digest == GOLDEN[f"{length}-{state_bits}-{q}"]
    unsigned = states.view(np.uint16).astype(np.uint32)
    base = min(q, 3)
    np.testing.assert_array_equal(
        unsigned >> base, np.roll(unsigned & ((1 << (state_bits - base)) - 1), 1, axis=1),
    )


@pytest.mark.parametrize("state_bits", [10, 16])
@pytest.mark.parametrize("group_size,columns", [(128, 640), (256, 768)])
@pytest.mark.parametrize("columns_scale", [False, True])
def test_compressed_projection_matches_numpy(state_bits, group_size, columns, columns_scale):
    tensor = example_tensor(columns=columns, group_size=group_size, state_bits=state_bits, columns_scale=columns_scale)
    tensor = RowTrellisTensor.from_bytes(tensor.to_bytes())
    dense = np.array(decode(tensor))
    np.testing.assert_allclose(dense, decode_cpu(tensor), rtol=2e-6, atol=1e-7)
    rng = np.random.default_rng(42)
    for m in range(1, 7):
        inputs = rng.normal(size=(m, columns)).astype(np.float32)
        actual = np.array(project(tensor, inputs))
        expected = inputs.astype(np.float64) @ dense.astype(np.float64).T
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-6)


@pytest.mark.parametrize("column_scaling", [False, True])
def test_quantize_mixed_rows_and_zero_weights(column_scaling):
    rng = np.random.default_rng(14)
    weight = rng.normal(0, .02, (8, 256)).astype(np.float32)
    weight[0] = 0
    qs = np.arange(1, 9, dtype=np.uint8)
    tensor = quantize(weight, qs, group_size=128, column_scaling=column_scaling)
    np.testing.assert_array_equal(tensor.precisions, qs)
    decoded = np.array(decode(tensor))
    np.testing.assert_allclose(decoded, decode_cpu(tensor), rtol=2e-6, atol=1e-8)
    np.testing.assert_array_equal(decoded[0], np.zeros(256))
    assert np.sum((decoded - weight)**2) < np.sum(weight**2)


def test_quantize_target_bpw_counts_all_metadata():
    rng = np.random.default_rng(50)
    weight = rng.normal(0, .02, (5, 256)).astype(np.float32)
    tensor = quantize(weight, target_bpw=3.5)
    assert len(tensor.to_bytes()) <= int(weight.size * 3.5 / 8)
    np.testing.assert_allclose(np.array(decode(tensor)), decode_cpu(tensor), rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("kwargs", [
    {}, {"precisions": 0}, {"precisions": 9}, {"precisions": 3.0},
    {"precisions": [1, 2]}, {"target_bpw": .01}, {"target_bpw": float("nan")},
    {"precisions": 3, "target_bpw": 3}, {"precisions": 3, "iterations": 0},
    {"precisions": 3, "state_bits": 14}, {"precisions": 3, "group_size": 64},
])
def test_quantize_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        quantize(np.zeros((3, 256), np.float32), **kwargs)
