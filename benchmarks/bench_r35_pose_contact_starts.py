"""Compare minimum incidental loss as a soft priority on the complete fresh-board search trajectory."""
import argparse,json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from r28_exchange import read_json
from r31_fresh import Assemblies
from r32_fresh import fresh
from r32_fresh_phases import fit
from r32_gold_bonds import inventory
from r35_pose_contacts import Boundaries
from search_problem import SearchProblem


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True);cli.add_argument('--count',type=int,default=16)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));required=inventory(p).required
    raw=Boundaries(p,False);baseline=Assemblies.seed;out=[]
    for seed in range(8540,8540+args.count):
        fitted=fit(p,required,{0:0},seed=seed,effort=1)
        if 'phases' not in fitted:continue
        case=dict(seed=seed)
        for soft in (False,True):
            info=Counter();sample={}
            def gate(a,hint,rng,*aa,**kw):
                if a.bonds<=required:return baseline(a,hint,rng,*aa,**kw)
                original=a.groups
                info['largest_group']=max(map(lambda item:len(item[0]),original))
                if soft:
                    fitted_rows=[]
                    for members,rows in original:
                        vals=[raw.describe(row)[3] for row in rows] if len(members)>2 else [0]*len(rows)
                        least=min(vals);better=[pose for pose,loss in zip(rows,vals) if loss==least]
                        fitted_rows.append((members,better))
                        info['lower_priority']+=len(rows)-len(better)
                    a.groups=fitted_rows
                try:
                    try:return baseline(a,hint,rng,*aa,**kw)
                    except ValueError:
                        if not soft:raise
                        a.groups=original;info['fallback']+=1
                        return baseline(a,hint,rng,*aa,**kw)
                finally:a.groups=original
            def snapshot(s):
                if s['cursor'] in (16,64):
                    sample[s['cursor']]={k:s['best'][k] for k in ('score','signed','components','gold')}
            with patch('r32_fresh.fit',return_value=fitted),patch.object(Assemblies,'seed',gate):
                x=fresh(p,seed,rounds=64,cube_bits=0,merges=32,incremental_merges=True,save=snapshot)
            case['soft' if soft else 'plain']=dict(diagnostics=info,horizons=sample,work={k:v for k,v in x['work'].items()
                if k in ('rounds','repair_calls','pose_nodes','signed_cache_misses','assignment_cache_misses','speculative_join_probes')})
        out.append(case);args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
        print(seed,case['plain']['horizons'].get(64),case['soft']['horizons'].get(64),flush=True)
if __name__=='__main__':main()
