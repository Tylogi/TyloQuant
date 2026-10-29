"""Verify new search policy reaches the HIP production kernel and stays reusable."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
torch=pytest.importorskip('torch')
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip('ROCm unavailable',allow_module_level=True)

from mfq.quantize.cuda import tq_columns as columns
from mfq.quantize.cuda import tq_encoder, tq_fitter
from mfq.quantize.tq import quantize


def test_default_entry_uses_protected_hip_fast_kernel(monkeypatch):
    def reference_forbidden(*args,**kwargs):
        raise AssertionError('Generic HIP encoder must not handle default column fitting')
    monkeypatch.setattr(tq_encoder,'encode',reference_forbidden)
    original=tq_fitter.encode
    calls=[]
    def recorded(x,metric,bits,**options):
        assert options['fast_columns'] and options['column_scaling']
        assert ((x.abs()+columns._BOOK_BOUND)*metric.abs()).amax() <= 32.
        calls.append(bits)
        return original(x,metric,bits,**options)
    monkeypatch.setattr(tq_fitter,'encode',recorded)
    torch.manual_seed(738)
    w=torch.randn(8,513,device='cuda')*.02
    w[0]=0;w[1,27]=.5
    h=torch.ones(513,device='cuda');h[27]=3000
    tensor=quantize(w,np.arange(1,9),imatrix=h)
    assert set(calls)==set(range(1,9))
    assert tensor.columns and np.isfinite(tensor.dequantize()).all()
    assert tensor.anchors[0]==0


@pytest.mark.parametrize('fast',[False,True])
@pytest.mark.parametrize('allow_clipping',[False,True])
def test_reused_fitter_matches_upstream_search_with_same_encoder(fast,allow_clipping):
    path=Path(__file__).with_name('fixtures')/'tq_columns_search_6d08119.py'
    spec=importlib.util.spec_from_file_location('mfq.quantize.cuda._upstream_columns',path)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    upstream.encode=columns.encode
    torch.manual_seed(915)
    w=torch.randn(9,512,device='cuda')*.02
    w[0]=0;w[7,23]=.8
    h=torch.linspace(0,4,512,device='cuda');h[::7]=0
    c=columns.prepare_columns(w)
    torch.testing.assert_close(c,upstream.prepare_columns(w),rtol=0,atol=0)
    fitter=columns.ColumnFitter(w,h,c,fast=fast,allow_clipping=allow_clipping)
    for bits in range(1,9):
        actual=fitter.fit(bits)
        expected=upstream.fit_columns(w,bits,h,c,fast=fast,allow_clipping=allow_clipping)
        assert torch.equal(actual[0],expected[0]),(fast,allow_clipping,bits,'states')
        assert torch.equal(actual[1],expected[1]),(fast,allow_clipping,bits,'anchors')


def test_cached_inputs_on_a_nondefault_stream():
    torch.manual_seed(198)
    w=torch.randn(5,512,device='cuda')*.02
    h=torch.ones(512,device='cuda')
    c=columns.prepare_columns(w)
    reference=columns.fit_columns(w,3,h,c,fast=True)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fitter=columns.ColumnFitter(w,h,c,fast=True)
        actual=fitter.fit(3)
        fitter.fit(1)
        repeated=fitter.fit(3)
    torch.cuda.current_stream().wait_stream(stream)
    for expected in (actual,repeated):
        assert torch.equal(reference[0],expected[0])
        assert torch.equal(reference[1],expected[1])
