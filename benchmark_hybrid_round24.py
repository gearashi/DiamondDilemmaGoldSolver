"""Serial fixed-work sampling comparisons, with independently checked boards.

Example: python -B benchmark_hybrid_round24.py --data data/tiles.json --output audit/sampling.json --steps1024
Use a dedicated hardware window. Proposal counts are not uniform operation costs:
experimental chains currently rescore full boards, unlike the legacy delta path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from search_problem import SearchProblem
import gpu_sampling as sampling

FLAGS=('ejection_chains','move_tabu','replica_exchange')

def run(data,*,steps=1024,replicas=128,seeds=(1,2,3,4),backend='cuda',gold=False,emit=lambda row:None):
    if type(steps)is not int or steps<1 or type(replicas)is not int or replicas<8:raise ValueError('Positive steps and at least8 lanes required.')
    problem=SearchProblem(data);inputs=sampling._inputs(problem,gold_aware=gold);rows=[]
    for seed in seeds:
        for flags in range(8):
            options={key:bool(flags&(1<<i)) for i,key in enumerate(FLAGS)}
            start=time.monotonic();engine=sampling._make_sampler(problem,*inputs,replicas,seed,backend,experiments=options)
            setup=time.monotonic()-start;start=time.monotonic();done=launches=0;seen=set();repeat=observed=0;first_gold=None;first_240=None
            try:
                while done<steps:
                    work=min(16,steps-done);engine.launch(work);done+=work;launches+=1
                    if flags&4 and launches%8==0:engine.exchange((launches//8-1)&1)
                    boards,scores=engine.top(min(replicas,16))
                    for board,value in zip(boards,scores):
                        codes=list(map(int,board));check=sampling.gold_score(problem,codes) if gold else sampling.score_board(problem,codes)
                        if int(value)!=check['gold_objective' if gold else 'objective']:raise AssertionError('Native objective disagrees with independent check.')
                        key=tuple(codes);repeat+=key in seen;observed+=1;seen.add(key)
                        if check['matched_edges']==len(problem.edges):
                            if first_240 is None:first_240=done*replicas
                            if problem.validate(codes)['valid'] and first_gold is None:first_gold=done*replicas
                best,_=engine.top(1);codes=list(map(int,best[0]));metrics=sampling.score_board(problem,codes);validation=problem.validate(codes)
                row={'flags':options,'seed':seed,'backend':backend,'device':engine.device,'gold_scoring':gold,'replicas':replicas,
                     'proposals':done*replicas,'launches':launches,'compute_seconds':time.monotonic()-start,'setup_seconds':setup,
                     'best':metrics,'validation':validation,'observed_archive_repeats':repeat,'observed_archive_boards':observed,
                     'first_edge_perfect_proposals_upper_bound':first_240,'first_gold_proposals_upper_bound':first_gold,
                     'counters':engine.experiment_stats()}
                rows.append(row);emit(row)
            finally:engine.close()
    return {'scope':'Fixed proposals on original input; archive-repeat samples do not count every proposed board. No search coverage claim.',
            'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('gpu_sampling.py','sampling_round24.cu','sampling_round24.wgsl','sampling_kernels.wgsl')},'records':rows}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1024);parser.add_argument('--replicas',type=int,default=128);parser.add_argument('--seeds',default='1,2,3,4');parser.add_argument('--backend',choices=('cuda','webgpu'),default='cuda');parser.add_argument('--gold',action='store_true');args=parser.parse_args()
    raw=args.data.read_bytes();report=run(json.loads(raw.decode('utf-8-sig')),steps=args.steps,replicas=args.replicas,seeds=tuple(map(int,args.seeds.split(','))),backend=args.backend,gold=args.gold,emit=lambda row:print(json.dumps(row),flush=True))
    report['input_sha256']=hashlib.sha256(raw).hexdigest();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
