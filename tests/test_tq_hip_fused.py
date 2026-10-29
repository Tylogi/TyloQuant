"""Exact TQ grouped projections and fused SwiGLU."""
import ctypes as ct
import pytest

torch=pytest.importorskip('torch')
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip('ROCm unavailable',allow_module_level=True)

from mfq.kernels.cuda.tq import ProjectionView
from mfq.kernels.hip.tq import Projection
from tests.test_tq_hip import fixture


class Job(ct.Structure):
    _fields_=[('weight',ProjectionView),('output',ct.c_void_p),('block_end',ct.c_int)]
class Plan(ct.Structure):
    _fields_=[('jobs',Job*4),('count',ct.c_int)]


def functions(lib):
    lib.mfq_tq_group.argtypes=[Plan,ct.c_void_p,ct.c_int,ct.c_void_p]
    lib.mfq_tq_group.restype=ct.c_int
    lib.mfq_tq_swiglu.argtypes=[ProjectionView,ProjectionView,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
    lib.mfq_tq_swiglu.restype=ct.c_int
    lib.mfq_tq_residual.argtypes=[ProjectionView,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
    lib.mfq_tq_residual.restype=ct.c_int
    return lib


@pytest.mark.parametrize('dtype,code',[(torch.float16,0),(torch.bfloat16,1),(torch.float32,2)])
@pytest.mark.parametrize('count',[2,3,4])
@pytest.mark.parametrize('columns,gs',[(False,16),(False,512),(True,512)])
@pytest.mark.parametrize('width',[273,512])
def test_projection_group_graph(dtype,code,count,columns,gs,width):
    ops=[Projection(fixture(rows,width,columns,gs),input_columns=width,
        bias=torch.linspace(-.03,.03,rows,device='cuda')) for rows in [19,33,91,129][:count]]
    lib=functions(ops[0].lib)
    x=torch.randn(1,width,device='cuda',dtype=dtype)
    outputs=[torch.empty(1,op.rows,device='cuda',dtype=dtype) for op in ops]
    plan=Plan();plan.count=count
    for i,(op,y) in enumerate(zip(ops,outputs)):plan.jobs[i]=Job(op.view,y.data_ptr(),0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert lib.mfq_tq_group(plan,x.data_ptr(),code,torch.cuda.current_stream().cuda_stream)==0
    for y in outputs:y.fill_(float('nan'))
    graph.replay()
    for op,y in zip(ops,outputs):
        expected=op(x)
        torch.testing.assert_close(y,expected,rtol={0:.002,1:.015,2:2e-5}[code],atol={0:.0002,1:.0015,2:2e-6}[code])


@pytest.mark.parametrize('dtype,code',[(torch.float16,0),(torch.bfloat16,1),(torch.float32,2)])
@pytest.mark.parametrize('columns,gs',[(False,16),(False,512),(True,512)])
@pytest.mark.parametrize('width',[273,512])
def test_swiglu_group_graph(dtype,code,columns,gs,width):
    rows=91
    gate=Projection(fixture(rows,width,columns,gs),input_columns=width,
        bias=torch.linspace(-.02,.02,rows,device='cuda'))
    up=Projection(fixture(rows,width,columns,gs),input_columns=width,
        bias=torch.linspace(.03,-.03,rows,device='cuda'))
    lib=functions(gate.lib)
    x=torch.randn(1,width,device='cuda',dtype=dtype)
    y=torch.empty(1,rows,device='cuda',dtype=dtype)
    expected=torch.nn.functional.silu(gate(x))*up(x)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert lib.mfq_tq_swiglu(gate.view,up.view,x.data_ptr(),y.data_ptr(),code,
            torch.cuda.current_stream().cuda_stream)==0
    y.fill_(float('nan'));graph.replay()
    error=(y.double()-expected.double()).square().mean().sqrt()/expected.double().square().mean().sqrt()
    assert error<{0:.001,1:.004,2:3e-6}[code]


@pytest.mark.parametrize('dtype,code',[(torch.float16,0),(torch.bfloat16,1),(torch.float32,2)])
@pytest.mark.parametrize('columns,gs',[(False,16),(False,512),(True,512)])
@pytest.mark.parametrize('width',[273,512])
def test_residual_preserves_projection_rounding(dtype,code,columns,gs,width):
    rows=91
    op=Projection(fixture(rows,width,columns,gs),input_columns=width,
        bias=torch.linspace(-.02,.02,rows,device='cuda'))
    lib=functions(op.lib)
    x=torch.randn(1,width,device='cuda',dtype=dtype)
    skip=torch.randn(1,rows,device='cuda',dtype=dtype)*.1
    expected=op(x)+skip
    output=torch.empty_like(skip)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert lib.mfq_tq_residual(op.view,x.data_ptr(),skip.data_ptr(),output.data_ptr(),code,
            torch.cuda.current_stream().cuda_stream)==0
    output.fill_(float('nan'));graph.replay()
    assert torch.equal(output,expected)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('columns,gs',[(False,16),(False,512),(True,512)])
def test_single_token_basis_reconstructs_exact_weights(dtype,columns,gs):
    width,rows=273,19
    tensor=fixture(rows,width,columns,gs)
    op=Projection(tensor,input_columns=width)
    expected=torch.from_numpy(tensor.dequantize(width).T.copy()).to(device='cuda',dtype=dtype)
    basis=torch.eye(width,device='cuda',dtype=dtype)
    result=torch.empty(width,rows,device='cuda',dtype=dtype)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for column in range(width):op.forward(basis[column:column+1],result[column:column+1])
    result.fill_(float('nan'));graph.replay()
    assert torch.equal(result,expected)
