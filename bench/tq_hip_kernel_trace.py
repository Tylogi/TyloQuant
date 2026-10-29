"""Trace actual native graph dispatches and summarize complete decode tokens."""
import argparse,csv,collections,json,os
from pathlib import Path
from model_bench import RuntimeProcess

p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--context',type=int,default=16384)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--profiler',default='rocprofv3')
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
command=['timeout','--signal=TERM','--kill-after=5s','65s',args.profiler,'--kernel-trace',
    '--log-level','error','--output-format','csv','--output-directory',str(args.output),'--',
    str(args.runtime),'--model',str(args.model),
    '--transport','stdio','--ctx-size',str(args.context)]
runtime=RuntimeProcess(command,dict(os.environ),Path.cwd())
try:
    result=runtime.generate('profile',args.length,256,False,60)
    (args.output/'status.json').write_text(json.dumps(result,indent=2))
    assert result['status']=='ok',result
finally:runtime.close()
files=list(args.output.rglob('*kernel_trace.csv'));assert len(files)==1,files
rows=list(csv.DictReader(files[0].open()))
ends=[i for i,r in enumerate(rows) if r['Kernel_Name']=='decode_graph_commit_kernel(long const*, long*, long*, long*, long*, long*)']
assert len(ends)>=101,len(ends)
start,end=ends[-101]+1,ends[-1]+1;times=collections.defaultdict(lambda:[0,0.]);grids=collections.defaultdict(collections.Counter)
for r in rows[start:end]:
    v=times[r['Kernel_Name']];v[0]+=1;v[1]+=(int(r['End_Timestamp'])-int(r['Start_Timestamp']))/100000
    grids[r['Kernel_Name']][r['Grid_Size_X']]+=1
summary=dict(prompt_tokens=args.length,measured_tokens=100,dispatches=end-start,
    wall_ms_per_token=(int(rows[end-1]['End_Timestamp'])-int(rows[start]['Start_Timestamp']))/100e6,
    kernel_us_per_token=sum(v[1] for v in times.values()),
    kernels=[dict(name=n,calls=v[0],us_per_token=v[1],grid=dict(grids[n])) for n,v in sorted(times.items(),key=lambda x:-x[1][1])])
(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
