"""Retain each validated native TQ model timing as it completes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from model_bench import RuntimeProcess


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',nargs='+',type=int,default=[1,2048,8192])
    p.add_argument('--decode',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--context',type=int,default=16384)
    p.add_argument('--timeout',type=float,default=180,help='Per-request timeout in seconds')
    p.add_argument('--prefill-chunk-size',type=int,help='Override the runtime prefill chunk size')
    args=p.parse_args()
    if not args.lengths or any(n<1 or n+args.decode>args.context for n in args.lengths):
        p.error('Invalid prompt/context sizes')
    if not math.isfinite(args.timeout) or args.timeout<=0:
        p.error('Timeout must be finite and positive')
    if args.prefill_chunk_size is not None and args.prefill_chunk_size<1:
        p.error('Prefill chunk size must be positive')
    identity=dict(model=str(args.model),model_sha256=sha256(args.model),
                  runtime=str(args.runtime),runtime_sha256=sha256(args.runtime),
                  lengths=args.lengths,decode=args.decode,repeats=args.repeats,context=args.context,
                  gpu=os.environ.get('HIP_VISIBLE_DEVICES'),mtp=False,
                  require_complete_generation=True,
                  tq_q8=os.environ.get('MFQ_HIP_TQ_Q8','0'))
    if args.prefill_chunk_size is not None:
        identity['prefill_chunk_size']=args.prefill_chunk_size
    result=dict(identity=identity,runs=[],verified=False)
    if args.output.exists():
        result=json.loads(args.output.read_text())
        if result['identity']!=identity:raise ValueError('Benchmark checkpoint identity differs')
    result['request_timeout_seconds']=args.timeout
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=args.output.with_suffix('.json.partial')
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        os.replace(temporary,args.output)
    def display(value):
        try:print(json.dumps(value),flush=True)
        except (OSError,ValueError):pass
    started=time.perf_counter()
    command=[str(args.runtime),'--model',str(args.model),'--transport','stdio',
             '--ctx-size',str(args.context)]
    if args.prefill_chunk_size is not None:
        command+=['--prefill-chunk-size',str(args.prefill_chunk_size)]
    runtime=RuntimeProcess(command,dict(os.environ),Path.cwd())
    try:
        for length in args.lengths:
            for repeat in range(args.repeats+1):
                if any(r['length']==length and r['repeat']==repeat for r in result['runs']):continue
                display(dict(phase='request_start',length=length,repeat=repeat,
                             completed=len(result['runs']),total=len(args.lengths)*(args.repeats+1),
                             elapsed_seconds=time.perf_counter()-started))
                response=runtime.generate(f'tq-{length}-{repeat}',length,args.decode,False,args.timeout)
                if response['status']!='ok':
                    result['failure']=response;save()
                    raise RuntimeError(str(response))
                run=dict(length=length,repeat=repeat,warmup=repeat==0,**response['metrics'])
                if run['prompt_tokens']!=length or run['completion_tokens']!=args.decode:
                    result['failure']=dict(reason='Requested token counts were not completed',run=run)
                    result['verified']=False;save()
                    raise RuntimeError(str(result['failure']))
                result['runs'].append(run);save()
                display(dict(completed=len(result['runs']),total=len(args.lengths)*(args.repeats+1),
                             elapsed_seconds=time.perf_counter()-started,
                             **{key:run[key] for key in ['length','repeat','warmup','prompt_tokens',
                                 'completion_tokens','decode_tps','prefill_tps','ttft_ms']}))
    finally:
        runtime.close()
    result['summary']=[]
    for length in args.lengths:
        runs=[r for r in result['runs'] if r['length']==length and not r['warmup']]
        if len(runs)!=args.repeats:raise RuntimeError('Missing benchmark samples')
        result['summary'].append(dict(prompt_tokens=length,samples=len(runs),
            **{key:statistics.median(r[key] for r in runs) for key in ['decode_tps','prefill_tps','ttft_ms']}))
    result['verified']=True;save();display(dict(verified=True,summary=result['summary']))


if __name__=='__main__':main()
