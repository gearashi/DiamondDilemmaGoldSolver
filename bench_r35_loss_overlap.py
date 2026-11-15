"""Compare accepted sets on actual ranked arrivals and record overlapping physical loss prices."""
from collections import Counter
from pathlib import Path
import json
from bench_r35_loss_proof import original, read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r35_overlap_loss import OverlapPattern
from r35_loss_witness import physical_price,cover_missing
from io_utils import atomic_json

if __name__=='__main__':
    p,ref,pack,req,out=original('../data/tiles.json')
    _,_,groups=reference_groups(p,read_json('docs/R29_INCUMBENT.json')['codes'])
    rows=[]
    for errors,stop in ((4,6903),(5,65536)):
        for label,charged in [('packed',pack),('all',groups)]:
            search=OutsideSyndromes(p,ref,groups,req)
            work=Counter()
            test=OverlapPattern(p,pack,charged,req,out,17,False,work)
            arrivals=[]
            try:
                def collect(indices,deadline):
                    if test(indices,deadline):
                        arrivals.append(sorted(indices))
                    return False
                result=search.indexed(17,errors,300,start=0,stop=stop,accept_errors=collect,accept_status='PHASE_RELAXATION_FEASIBLE')
            finally:
                test.close()
            rows.append(dict(scope=label,errors=errors,stop=stop,arrivals=arrivals,
                             native=result['work'],finite=dict(work),completed=result['next_rank']==stop))
            atomic_json(Path('docs/R35_LOSS_OVERLAP_STREAMS.json'),rows)
            print(label,errors,len(arrivals),result['next_rank'],work,flush=True)
