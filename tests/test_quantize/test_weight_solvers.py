import torch

from mfq.quantize.gptq import GptqConfig, GptqSolver
from mfq.quantize.gsq import GsqConfig, GsqSolver
from mfq.quantize.weight_solver import (
    QuadraticReconstructionObjective,
    UniformAffineCodec,
    WeightSolverProblem,
)


def _correlated_problem(seed: int = 0) -> WeightSolverProblem:
    torch.manual_seed(seed)
    weight = torch.randn(4, 16)
    latent = torch.randn(128, 4)
    inputs = latent @ torch.randn(4, 16) + 0.05 * torch.randn(128, 16)
    return WeightSolverProblem(
        "model.layers.0.mlp.gate.weight",
        weight,
        calibration_inputs=inputs,
    )


def test_activation_and_hessian_objective_views_are_equivalent():
    problem = _correlated_problem()
    candidate = problem.weight + torch.randn_like(problem.weight) * 0.05
    objective = QuadraticReconstructionObjective()
    hessian = problem.calibration_inputs.mT @ problem.calibration_inputs
    hessian = hessian / problem.calibration_inputs.shape[0]
    hessian_problem = WeightSolverProblem(
        problem.tensor_key, problem.weight, hessian=hessian
    )

    torch.testing.assert_close(
        objective.row_losses(problem, candidate),
        objective.row_losses(hessian_problem, candidate),
        rtol=2e-5,
        atol=2e-6,
    )


def test_gptq_propagates_hessian_error_and_beats_rtn_on_correlated_inputs():
    problem = _correlated_problem()
    codec = UniformAffineCodec(2, 8, fit_iterations=0)
    initial = codec.initialize(problem)
    result = GptqSolver(GptqConfig(block_size=8)).solve(
        problem, codec, initial=initial
    )

    assert result.metrics["accepted"] == 1.0
    assert result.objective.total < result.baseline.total * 0.5
    assert result.grid.codes.dtype == torch.int16
    assert torch.all(result.grid.codes >= 0)
    assert torch.all(result.grid.codes <= 3)
    torch.testing.assert_close(result.encoded.dequantize(), result.reconstruction)


def test_gsq_jointly_improves_hard_codes_and_scales_without_layout_change():
    torch.manual_seed(2)
    weight = torch.randn(5, 12)
    inputs = torch.randn(96, 12)
    problem = WeightSolverProblem("model.layers.0.attn.q.weight", weight, inputs)
    codec = UniformAffineCodec(3, 4)
    gptq = GptqSolver(GptqConfig(block_size=4)).solve(problem, codec)
    result = GsqSolver(
        GsqConfig(
            steps=30,
            row_chunk_size=3,
            hard_eval_interval=5,
            use_gumbel_noise=False,
        ),
        gptq_config=GptqConfig(block_size=4),
    ).solve(problem, codec, initial=gptq.grid)

    assert result.metrics["accepted"] == 1.0
    assert result.objective.total < result.baseline.total
    assert result.grid.codes.shape == gptq.grid.codes.shape
    assert result.grid.q_bits.tolist() == gptq.grid.q_bits.tolist()
    assert result.grid.group_size == gptq.grid.group_size
    assert not torch.equal(result.grid.scale_parameters, gptq.grid.scale_parameters)


def test_gsq_seeded_gumbel_path_is_reproducible_and_hard():
    torch.manual_seed(7)
    problem = WeightSolverProblem(
        "model.layers.0.mlp.down.weight",
        torch.randn(3, 8),
        calibration_inputs=torch.randn(32, 8),
    )
    codec = UniformAffineCodec(2, 4)
    config = GsqConfig(
        steps=8,
        row_chunk_size=2,
        hard_eval_interval=2,
        seed=91,
        use_gumbel_noise=True,
        use_gptq_init=False,
    )
    first = GsqSolver(config).solve(problem, codec)
    second = GsqSolver(config).solve(problem, codec)

    torch.testing.assert_close(first.grid.codes, second.grid.codes, rtol=0, atol=0)
    torch.testing.assert_close(
        first.grid.scale_parameters, second.grid.scale_parameters, rtol=0, atol=0
    )
    assert first.objective.total == second.objective.total
    assert torch.all(first.grid.codes >= 0)
    assert torch.all(first.grid.codes <= 3)


def test_mixed_width_grid_handles_tail_padding_and_singular_hessian():
    weight = torch.linspace(0.25, 2.5, 30).reshape(3, 10)
    problem = WeightSolverProblem(
        "model.layers.0.attn.k.weight",
        weight,
        calibration_inputs=torch.zeros(16, 10),
    )
    codec = UniformAffineCodec(torch.tensor([2, 3, 4]), 4)
    initial = codec.initialize(problem)
    result = GptqSolver(GptqConfig(block_size=4)).solve(
        problem, codec, initial=initial
    )

    assert initial.padded_count == 12
    assert result.reconstruction.shape == weight.shape
    assert torch.isfinite(result.reconstruction).all()
    for row, bits in enumerate([2, 3, 4]):
        assert int(result.grid.codes[row].min()) >= 0
        assert int(result.grid.codes[row].max()) <= (1 << bits) - 1
