"""Grouped Q8 calls preserve the CUDA Q8 algorithm and output rounding."""
import ctypes as ct
import pytest
torch=pytest.importorskip('torch')
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip('ROCm unavailable',allow_module_level=True)
from mfq.kernels.cuda.tq import ProjectionView
from mfq.kernels.hip.tq import Projection
from tests.test_tq_hip import fixture
from tests.test_tq_hip_fused import Job,Plan
from tests.test_tq_hip_q8 import q8


@pytest.mark.parametrize('dtype,code',[(torch.float16,0),(torch.float32,2)])
@pytest.mark.parametrize('columns,gs',[(False,16),(False,512),(True,512)])
@pytest.mark.parametrize('count',[2,4])
def test_q8_group_matches_individual_projections(q8,dtype,code,columns,gs,count):
    width=273
    ops=[Projection(fixture(r,width,columns,gs),input_columns=width) for r in [19,33,91,129][:count]]
    x=torch.randn(1,width,device='cuda',dtype=dtype)
    outputs=[torch.empty(1,op.rows,device='cuda',dtype=dtype) for op in ops]
    scratch=torch.empty(count*32*24,device='cuda',dtype=torch.uint8)
    expected=[];stream=torch.cuda.current_stream().cuda_stream
    for op,y in zip(ops,outputs):
        assert q8(op.view,x.data_ptr(),y.data_ptr(),1,code,scratch.data_ptr(),32*24,stream)==0
        expected.append(y.clone())
    plan=Plan();plan.count=count
    for i,(op,y) in enumerate(zip(ops,outputs)):plan.jobs[i]=Job(op.view,y.data_ptr(),0)
    fun=q8.lib.mfq_tq_q8_group
    fun.argtypes=[Plan,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_void_p];fun.restype=ct.c_int
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert fun(plan,x.data_ptr(),code,scratch.data_ptr(),scratch.numel(),torch.cuda.current_stream().cuda_stream)==0
    for y in outputs:y.fill_(float('nan'))
    graph.replay()
    assert all(torch.equal(y,z) for y,z in zip(outputs,expected))
    assert fun(plan,x.data_ptr(),code,scratch.data_ptr(),scratch.numel()-1,stream)!=0


@pytest.mark.parametrize('dtype,code',[(torch.float16,0),(torch.float32,2)])
@pytest.mark.parametrize('columns,gs',[(False,16),(False,512),(True,512)])
def test_q8_swiglu_matches_individual_calls(q8,dtype,code,columns,gs):
    width,rows=273,91
    gate=Projection(fixture(rows,width,columns,gs),input_columns=width,
        bias=torch.linspace(-.02,.02,rows,device='cuda'))
    up=Projection(fixture(rows,width,columns,gs),input_columns=width,
        bias=torch.linspace(.03,-.03,rows,device='cuda'))
    x=torch.randn(1,width,device='cuda',dtype=dtype)
    a=torch.empty(1,rows,device='cuda',dtype=dtype);b=torch.empty_like(a);out=torch.empty_like(a)
    scratch=torch.empty(2*32*24,device='cuda',dtype=torch.uint8);stream=torch.cuda.current_stream().cuda_stream
    for op,y in [(gate,a),(up,b)]:
        assert q8(op.view,x.data_ptr(),y.data_ptr(),1,code,scratch.data_ptr(),32*24,stream)==0
    expected=torch.nn.functional.silu(a)*b
    fun=q8.lib.mfq_tq_q8_swiglu
    fun.argtypes=[ProjectionView,ProjectionView,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_void_p]
    fun.restype=ct.c_int
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert fun(gate.view,up.view,x.data_ptr(),out.data_ptr(),code,scratch.data_ptr(),scratch.numel(),
            torch.cuda.current_stream().cuda_stream)==0
    out.fill_(float('nan'));graph.replay()
    rms=(out.double()-expected.double()).square().mean().sqrt()/expected.double().square().mean().sqrt()
    assert rms<(.001 if code==0 else 3e-6)
