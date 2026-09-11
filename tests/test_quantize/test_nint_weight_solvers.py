import numpy as np
import torch

from mfq.formats.nint import NintSpec, NintTensor
from mfq.quantize.gptq import GptqConfig, GptqSolver
from mfq.quantize.gsq import GsqConfig, GsqSolver
from mfq.quantize.nint_quant import dequantize
from mfq.quantize.nint_solver import NintGridCodec
from mfq.quantize.nint_v2 import NintV2Allocation, solve_nint_v2_weights
from mfq.quantize.weight_solver import WeightSolverProblem


def _problem() -> WeightSolverProblem:
    torch.manual_seed(0)
    weight = torch.randn(4, 16)
    latent = torch.randn(128, 4)
    inputs = latent @ torch.randn(4, 16) + 0.05 * torch.randn(128, 16)
    return WeightSolverProblem(
        "model.layers.0.mlp.up.weight", weight, calibration_inputs=inputs
    )


def _codec() -> NintGridCodec:
    return NintGridCodec(
        NintSpec(2, 8, 5),
        row_q_bits=np.asarray([2, 3, 2, 4], dtype=np.uint8),
        row_sub_bits=np.asarray([5, 4, 6, 7], dtype=np.uint8),
    )


def test_gptq_seals_directly_into_mixed_qk_nintv2():
    problem = _problem()
    codec = _codec()
    result = GptqSolver(GptqConfig(block_size=8)).solve(problem, codec)

    assert isinstance(result.encoded, NintTensor)
    np.testing.assert_array_equal(result.encoded.row_q_bits, [2, 3, 2, 4])
    np.testing.assert_array_equal(result.encoded.row_sub_bits, [5, 4, 6, 7])
    assert result.objective.total < result.baseline.total
    np.testing.assert_allclose(
        dequantize(result.encoded), result.reconstruction.cpu().numpy(), rtol=0, atol=0
    )
    for row, bits in enumerate(result.encoded.row_q_bits):
        assert int(result.encoded.q[row].max()) <= (1 << int(bits)) - 1


def test_gsq_learns_nint_neuron_anchors_without_changing_qk_template():
    problem = _problem()
    codec = _codec()
    gptq = GptqSolver(GptqConfig(block_size=8)).solve(problem, codec)
    result = GsqSolver(
        GsqConfig(
            steps=20,
            row_chunk_size=2,
            hard_eval_interval=4,
            use_gumbel_noise=False,
        ),
        gptq_config=GptqConfig(block_size=8),
    ).solve(problem, codec, initial=gptq.grid)

    assert isinstance(result.encoded, NintTensor)
    assert result.objective.total <= result.baseline.total
    np.testing.assert_array_equal(result.encoded.row_q_bits, [2, 3, 2, 4])
    np.testing.assert_array_equal(result.encoded.row_sub_bits, [5, 4, 6, 7])
    np.testing.assert_array_equal(
        result.encoded.sub_scale, gptq.encoded.sub_scale
    )
    np.testing.assert_array_equal(result.encoded.sub_min, gptq.encoded.sub_min)
    np.testing.assert_allclose(
        dequantize(result.encoded), result.reconstruction.cpu().numpy(), rtol=0, atol=0
    )


def test_resolved_nintv2_template_binds_to_generic_weight_solver():
    problem = _problem()
    allocation = NintV2Allocation(
        row_q_bits=np.asarray([2, 3, 2, 4], dtype=np.uint8),
        row_sub_bits=np.asarray([5, 4, 6, 7], dtype=np.uint8),
        target_variable_bits=0,
        actual_variable_bits=0,
        selected_loss=0.0,
        uniform_loss=0.0,
        solver="test",
    )
    result = solve_nint_v2_weights(
        problem.weight,
        NintSpec(2, 8, 5),
        allocation,
        GptqSolver(GptqConfig(block_size=8)),
        calibration_inputs=problem.calibration_inputs,
        tensor_key=problem.tensor_key,
    )

    assert isinstance(result.encoded, NintTensor)
    np.testing.assert_array_equal(result.encoded.row_q_bits, allocation.row_q_bits)
    np.testing.assert_array_equal(result.encoded.row_sub_bits, allocation.row_sub_bits)
