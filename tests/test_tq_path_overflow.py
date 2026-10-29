"""Regression for imatrix-amplified FP16 path overflow and row-scale collapse."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from mfq.quantize.cuda import tq_columns as columns


def test_path_metric_changes_only_ring_objective_scale():
    x = torch.zeros(3, 256)
    x[1, 7] = 50.
    x[2, 11] = -100.
    metric = torch.ones_like(x)
    metric[1, 7] = 40.
    metric[2, 11] = 80.
    actual = columns._path_metric(x, metric)
    torch.testing.assert_close(actual[0], metric[0], rtol=0, atol=0)
    ratios = actual / metric
    assert torch.equal(ratios, ratios[:, :1].expand_as(ratios))
    torch.testing.assert_close(torch.log2(ratios), torch.log2(ratios).round(), rtol=0, atol=0)
    bound = float(abs(columns.fixed_codebook()).max())
    assert bool((((x.abs()+bound)*actual.abs()).amax(-1) <= 32.).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_fused_path_metric_matches_cpu_and_reuses_outputs():
    torch.manual_seed(808)
    x = torch.randn(27, 256) * 30
    metric = torch.rand_like(x) * 50
    expected = columns._path_metric(x, metric)
    gx, gm = x.cuda(), metric.cuda()
    before = gm.clone()
    for _ in range(2):
        torch.testing.assert_close(columns._path_metric(gx, gm).cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(gm, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('bits', [1, 2, 3, 4, 5, 6, 7, 8])
def test_high_importance_outlier_does_not_collapse_positive_row(bits):
    from mfq.formats.tq import fixed_codebook, unpack_states
    from mfq.quantize.cuda.tq_fitter import pack
    torch.manual_seed(124)
    w = torch.randn(2, 4096, device='cuda') * .01
    w[:, 3994] = .5
    w[0] = 0
    h = torch.ones(4096, device='cuda')
    h[3994] = 3000.
    c = torch.ones(4096, device='cuda', dtype=torch.float16)
    state, anchor = columns.fit_columns(w, bits, h, c, fast=True)
    assert anchor[0] == 0
    assert anchor[1] > .001
    book = torch.tensor(fixed_codebook(), device='cuda')
    decoded = (book[state.int() & 65535].float() * anchor.float()[:, None]).half().float()
    assert ((w[1]-decoded[1]).square()*h).sum() < (w[1].square()*h).sum()*.5
    packed = pack(state, bits).cpu().numpy().view(np.uint8)
    actual = unpack_states(packed.reshape(-1, 32*bits), bits).reshape(2, 4096)
    np.testing.assert_array_equal(actual, (state.int() & 65535).cpu().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('fast', [True, False])
@pytest.mark.parametrize('bits', [1, 2, 3, 4, 5, 6, 7, 8])
def test_default_column_fitting_preserves_complete_source_range(bits, fast):
    torch.manual_seed(391)
    w = torch.randn(4, 4096, device='cuda') * .01
    w[0] = 0
    w[1, 27] = .5
    w[2, 147] = -.7
    h = torch.ones(4096, device='cuda')
    h[27] = 3000.
    h[147] = 0.  # Even unweighted source coordinates must remain covered.
    c = torch.linspace(.3, 2., 4096, device='cuda').half()
    state, anchor = columns.fit_columns(w, bits, h, c, fast=fast)
    book = torch.tensor(columns.fixed_codebook(), device='cuda').float()
    scale = anchor.float()[:, None] * c.float()[None, :]
    assert bool((w >= (scale * book.min()).half().float()).all())
    assert bool((w <= (scale * book.max()).half().float()).all())
    assert anchor[0] == 0
    assert torch.isfinite(anchor).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_plain_quantize_defaults_to_zero_clipping_for_mixed_row_bits():
    from mfq.formats.tq import TQTensor
    from mfq.quantize.tq import quantize
    torch.manual_seed(912)
    w = torch.randn(8, 513, device='cuda') * .02
    w[0] = 0
    w[1, 27] = .5
    w[2, 147] = -.7
    h = torch.ones(513, device='cuda')
    h[27] = 3000.
    tensor = quantize(w, np.arange(1, 9), imatrix=h)
    assert tensor.columns
    np.testing.assert_array_equal(tensor.bits, np.arange(1, 9))
    scale = tensor.anchors.astype(np.float32)[:, None] * tensor.scales.astype(np.float32)[None, :]
    book = columns.fixed_codebook().astype(np.float32)
    source = w.cpu().numpy()
    assert (source >= (scale[:, :513] * book.min()).astype(np.float16).astype(np.float32)).all()
    assert (source <= (scale[:, :513] * book.max()).astype(np.float16).astype(np.float32)).all()
    assert tensor.to_bytes() == TQTensor.from_bytes(tensor.to_bytes()).to_bytes()
