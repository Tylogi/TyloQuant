"""Sparse clipping repair, shared-column fitting and payload compatibility."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mfq.quantize.cuda import tq_scale_repair as repair


def test_proposals_preserve_zero_and_well_covered_rows():
    w = torch.tensor([[0., 0., 0., 0.], [.1, -.2, .3, -.4], [0., 0., 0., 20.]])
    h = torch.tensor([0., 1., 1., 10.])
    anchors = torch.ones(3, dtype=torch.float16)
    loss = torch.tensor([0., .01, 2800.])
    actual = repair.propose_scales(w, h, torch.ones(4).half(), anchors, loss)
    assert torch.equal(actual[:2], anchors[:2])
    assert actual[2] > anchors[2]
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm unavailable")
def test_accelerated_proposals_match_torch_and_current_stream(monkeypatch):
    torch.manual_seed(721)
    w = torch.randn(17, 513, device="cuda") * .02
    w[0] = 0
    w[1:, 7] *= 40
    h = torch.linspace(.1, 20., 513, device="cuda"); h[::7] = 0
    a = w.square().mean(-1).sqrt().clamp_min(2**-24).half()
    c = torch.ones(513, device="cuda", dtype=torch.float16)
    loss = (w.square() * h).sum(-1) * .1
    reference = repair._torch_proposals(w, h, c, a, loss)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = repair.propose_scales(w, h, c, a, loss)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=1e-7)
    storage = torch.empty(17, 1026, device="cuda")
    storage[:, ::2] = w
    torch.testing.assert_close(repair.propose_scales(storage[:, ::2], h, c, a, loss), reference,
                               rtol=0, atol=0)
    monkeypatch.setattr(repair, "triton", None)
    torch.testing.assert_close(repair.propose_scales(w, h, c, a, loss), reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm unavailable")
@pytest.mark.parametrize("search", ["fast", "adaptive"])
def test_mixed_row_fitting_repair_never_increases_weighted_error(search):
    from mfq.formats.tq import TQTensor
    from mfq.quantize.tq import quantize

    torch.manual_seed(931)
    w = torch.randn(17, 513, device="cuda") * .02
    w[0] = 0
    w[1::2, 31] *= 45
    h = torch.linspace(.1, 3., 513, device="cuda"); h[::13] = 0
    bits = np.arange(17) % 8 + 1
    before = quantize(w, bits, imatrix=h, search=search, column_scaling=True, repair_scales=False)
    after = quantize(w, bits, imatrix=h, search=search, column_scaling=True)
    np.testing.assert_array_equal(after.bits, bits)
    np.testing.assert_array_equal(after.scales, before.scales)
    assert len(after.to_bytes()) == len(before.to_bytes())
    assert after.to_bytes() == TQTensor.from_bytes(after.to_bytes()).to_bytes()
    scores = []
    for tensor in [before, after]:
        decoded = torch.tensor(tensor.dequantize(513), device="cuda").float()
        scores.append(((w - decoded).square() * h).sum(-1))
        assert torch.count_nonzero(decoded[0]) == 0
    assert bool((scores[1] <= scores[0] * (1 + 1e-6) + 1e-9).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm unavailable")
def test_repair_recovers_an_outlier_outside_the_scale_screen():
    from mfq.formats.tq import fixed_codebook
    from mfq.quantize.cuda.tq_columns import fit_columns

    torch.manual_seed(149)
    w = torch.randn(32, 4096, device="cuda") * .02
    w[17, 93] = .4  # Row 17 is absent from the 16-row scale screen.
    h = torch.ones(4096, device="cuda")
    columns = torch.ones(4096, device="cuda", dtype=torch.float16)
    book = torch.tensor(fixed_codebook(), device="cuda").float()
    scores = []
    for enabled in [False, True]:
        state, anchor = fit_columns(w, 8, h, columns, fast=True,
                                    repair_scales=enabled, allow_clipping=True)
        decoded = (book[state.int() & 65535] * (anchor.float()[:, None] * columns.float())).half().float()
        scores.append((w - decoded).square().sum(-1))
    assert bool((scores[1] <= scores[0] * (1 + 1e-6) + 1e-9).all())
    assert scores[1][17] < scores[0][17] * .5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm unavailable")
@pytest.mark.parametrize("options", [dict(search="full"), dict(scale_factors=(1.,))])
def test_column_fitting_rejects_incompatible_search(options):
    from mfq.quantize.tq import quantize
    with pytest.raises(ValueError, match="Column fitting requires"):
        quantize(torch.ones(2, 256, device="cuda"), 4, column_scaling=True, **options)
