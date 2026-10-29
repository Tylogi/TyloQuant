"""Production column fitting, mixed precisions and padded logical widths."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip('ROCm unavailable', allow_module_level=True)

from mfq.formats.tq import unpack_states
from mfq.kernels.hip.tq import Projection
from mfq.quantize.cuda.tq_fitter import fit_columns, pack, prepare_columns
from mfq.quantize.tq import quantize


@pytest.mark.parametrize('bits', range(1, 9))
def test_fast_column_fit_and_serialization(bits):
    torch.manual_seed(73)
    weight = torch.randn(9, 512, device='cuda') * .02
    weight[0] = 0
    importance = torch.linspace(.1, 2, 512, device='cuda')
    importance[::7] = 0
    columns = prepare_columns(weight)
    state, anchors = fit_columns(weight, bits, importance, columns, fast=True)
    packed = pack(state, bits).cpu().numpy()
    decoded = unpack_states(packed.reshape(-1, 32 * bits), bits).reshape(weight.shape)
    np.testing.assert_array_equal(decoded, state.cpu().numpy().view(np.uint16))
    assert anchors[0] == 0
    assert torch.isfinite(anchors).all()
    tensor = quantize(weight, bits, imatrix=importance, search='fast', column_scaling=True)
    np.testing.assert_array_equal(tensor.packed, packed)
    np.testing.assert_array_equal(tensor.anchors, anchors.cpu().numpy())


def test_mixed_fast_columns_padded_projection_and_backward():
    torch.manual_seed(19)
    weight = torch.randn(16, 273, device='cuda') * .02
    weight[0] = 0
    bits = np.array([1, 8, 3, 6, 2, 7, 4, 5] * 2)
    importance = torch.ones(273, device='cuda')
    importance[::11] = 0
    tensor = quantize(weight, bits, imatrix=importance, search='fast', column_scaling=True)
    assert tensor.columns and tensor.width == 512
    np.testing.assert_array_equal(tensor.bits, bits)
    assert tensor.anchors[0] == 0
    decoded = torch.tensor(tensor.dequantize(273), device='cuda').float()
    op = Projection(tensor, input_columns=273)
    x = torch.randn(33, 273, device='cuda', requires_grad=True)
    gradient = torch.randn(33, 16, device='cuda')
    output = op(x)
    torch.testing.assert_close(output, x @ decoded.T, rtol=2e-5, atol=2e-5)
    output.backward(gradient)
    torch.testing.assert_close(x.grad, gradient @ decoded, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('bits', [2, 8])
def test_adaptive_columns_all_bitwidths(bits):
    torch.manual_seed(11)
    weight = torch.randn(3, 256, device='cuda') * .01
    tensor = quantize(weight, bits, column_scaling=True)
    assert tensor.columns
    assert np.isfinite(tensor.dequantize()).all()



def test_reused_column_fitter_preserves_independent_bitwidth_results():
    from mfq.quantize.cuda.tq_fitter import ColumnFitter
    torch.manual_seed(82)
    weight = torch.randn(9, 512, device='cuda') * .02
    weight[0] = 0
    importance = torch.linspace(0, 3, 512, device='cuda')
    importance[::9] = 0
    columns = prepare_columns(weight)
    fitter = ColumnFitter(weight, importance, columns, fast=True)
    for bits in range(1, 9):
        state, anchor = fitter.fit(bits)
        expected_state, expected_anchor = fit_columns(weight, bits, importance, columns, fast=True)
        assert torch.equal(state, expected_state)
        assert torch.equal(anchor, expected_anchor)
