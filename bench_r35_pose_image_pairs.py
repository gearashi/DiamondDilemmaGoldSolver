"""Compare scoped two-image quota pruning with the normal geographic receiver and compiled AC."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from random import Random
from unittest.mock import patch

from bench_r35_pose_frontiers import Compiled
from io_utils import atomic_json
from r30_gold import search
from r35_pose_departures import reference_images
from r35_pose_image_pairs import pair_bounds
from bench_r35_pose_pair_weights import pair_owner_bounds
from constraint_advanced import _complete_matching_supports
from search_problem import SearchProblem


def fixtures(p,images):
    tables=[dict(image) for image in images]
    full=(1<<(3*p.n))-1
    for name,a,b,forced,supplement in (
        ('open',3,8,0,0),('below',2,5,129,0),('threshold',4,7,130,0),
        ('held-out',3,8,131,0),('mixed',3,8,128,4),('complete',0,1,160,0)):
        sites=list(range(p.n));Random(893582).shuffle(sites)
        ds=[full]*p.n
        for c in sites[:forced]: ds[c]=tables[a][c]|tables[b][c]
        for c in sites[forced:forced+supplement]:
            paired=tables[a][c]|tables[b][c]
            ds[c]=paired|next(1<<r for r in range(3) if not paired&(1<<r))
        yield name,ds
    for name,root,nfree,tiles,a,b in (
        ('spatial-29',0,26,[27,65,111],3,8),
        ('spatial-29-held-out',55,24,[27,65,111,12,78],1,7),
        ('spatial-30',91,27,[27,65,111],2,9),
        ('spatial-32',43,29,[27,65,111],0,5)):
        sites=[root];seen={root}
        for c in sites:
            for d,_ in p.board.neighbors[c]:
                if d not in seen:sites.append(d);seen.add(d)
        free=set(sites[:nfree]);pool=sum(7<<(3*t) for t in tiles)
        yield name,[full if c in free else tables[a][c]|tables[b][c]|pool for c in range(p.n)]

    receipt=json.loads((Path(__file__).parent/'docs/R35_POSE_IMAGE_PAIRS.json').read_bytes())
    for item in receipt['seam_covered_donors']:
        a,b=item['images'];pool=sum(7<<(3*t) for t in item['shared_tiles'])
        yield f'covered-{a}-{b}',[tables[a][c]|tables[b][c]|pool for c in range(p.n)]


def runs(tiles,output,seconds,native,selected):
    p=SearchProblem(json.loads(tiles.read_bytes()),physical_gold_support=True)
    codes=json.loads((Path(__file__).parent/'docs/R29_INCUMBENT.json').read_bytes())['codes']
    images=reference_images(tuple(codes))
    def skip(domains,images,cap):
        return domains,{'over':False,'removed':0}
    cpu=Compiled(p,native)
    rows=[]
    try:
        for name,ds in fixtures(p,images):
            if selected and name not in selected:continue
            plain,pair=pair_bounds(ds,images,65)
            final,owner=pair_owner_bounds(plain,images,65) if not pair['over'] else (plain,None)
            ac=cpu.narrow(ds)
            next_ac=cpu.narrow(final)
            row={'case':name,'domain_poses':sum(d.bit_count() for d in ds),'pair':pair,'owners':owner,
                 'compiled_ac_poses_before':None if ac is None else sum(d.bit_count() for d in ac),
                 'compiled_ac_poses_after':None if next_ac is None else sum(d.bit_count() for d in next_ac)}
            snapshot=ds
            trace=[]
            starts=((1<<(3*p.n))-1)//7
            while True:
                reduced=cpu.narrow(snapshot)
                if reduced is None:
                    trace.append([0,0]);break
                physical=[(mask | mask>>1 | mask>>2)&starts for mask in reduced]
                owners,_=_complete_matching_supports(physical)
                if owners is None:
                    trace.append([sum(m.bit_count() for m in reduced),0]);break
                limited=tuple(m & (free|free<<1|free<<2) for m,free in zip(reduced,owners))
                trace.append([sum(m.bit_count() for m in reduced),sum(m.bit_count() for m in limited)])
                if tuple(snapshot)==limited:break
                snapshot=limited
            row['full_geographic_owner_trace']=trace
            for policy in ('without-pairs','integrated'):
                with patch('r35_pose_image_pairs.pair_bounds',side_effect=skip) if policy=='without-pairs' else nullcontext():
                    out=search(p,geography=True,reference_bound=True,domains=ds,seconds=seconds)
                report=out.get('certified_pose_departures',{})
                row[policy]=dict(status=out['status'],cp_calls=out['cp_calls'],branches=out['branches'],
                                 reason=report.get('reason'),steps=report.get('support_steps',[]))
            rows.append(row);atomic_json(output,rows)
            print(name,[(row[k]['status'],row[k]['cp_calls'],row[k]['branches']) for k in ('without-pairs','integrated')],flush=True)
    finally:
        cpu.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--seconds',type=float,default=6);cli.add_argument('--native',type=Path,default=Path('../.native'))
    cli.add_argument('--cases',nargs='+')
    args=cli.parse_args();runs(args.tiles,args.output,args.seconds,args.native,args.cases)
