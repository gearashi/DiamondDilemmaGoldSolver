"""Compare shared geographic proofs, naked physical assumptions and actual native prefixes."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from r27_structure_filters import domain_bits
from r28_exchange import read_json
from r30_gold import search
from r31_geographic import GeographicConflicts
from r32_gold_loss import Certificates
from r35_geographic_replay import replay_for
from search_problem import SearchProblem


def archive():
    source=read_json('docs/R35_ALTERNATING_SEWINGS.json')
    cores=[row['geography_only']['geographic_certificate']['pairs'] for row in source['original_long_exchange']]
    cores.append(source['fixed_phase_parent']['exchange']['geographic_certificate']['pairs'])
    cores.append(read_json('docs/R35_TURN_WORDS.json')['original_residuals']['distinct_geographic_cores'][0]['cycle'])
    return [frozenset(map(tuple,c)) for c in cores]


def screen(p,path,cores):
    with np.load(path,allow_pickle=False) as ar:
        order=ar['order']
        binding=sha256(p.masks.astype(np.uint16).tobytes()+p.neighbors.astype(np.int16).tobytes()+
                       p.sides.astype(np.int8).tobytes()+order.tobytes()).hexdigest()
        if binding!=str(ar['fingerprint']):raise ValueError('Classic checkpoint drawing, mesh or order differs.')
        board=ar['boards'];depth=ar['depths'];status=ar['states']
    result=[]
    for core in cores:
        hist=Counter();actionable=0
        for start in range(0,board.shape[1],2048):
            partial=board[:,start:start+2048]; n=partial.shape[1]
            occupied,lane=np.nonzero(partial>=0); values=partial[occupied,lane]
            where=np.full((p.n,n),-1,np.int16);rot=np.zeros((p.n,n),np.int8)
            where[values//3,lane]=occupied;rot[values//3,lane]=values%3
            hit=np.zeros(n,np.int16);absent_unassigned=np.zeros(n,bool)
            for a,b in core:
                t,s=divmod(a,3);u,j=divmod(b,3)
                c=where[t];d=where[u];exposed=(s+rot[t])%3
                present=((c>=0)&(d>=0)&(p.neighbors[c,exposed]==d)&(p.sides[c,exposed]==(j+rot[u])%3))
                hit+=present
                absent_unassigned|=(c<0)|(d<0)
            hist.update(map(int,hit))
            actionable+=int(np.sum((hit==len(core)-1)&absent_unassigned))
        result.append(dict(size=len(core),seen=dict(sorted(hist.items())),all_but_one=hist[len(core)-1],
                           complete=hist[len(core)],unplaced_owner_at_unit=actionable))
    return dict(binding=binding,lanes=board.shape[1],max_depth=int(max(depth)),active=int(np.sum(status==0)),cores=result)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json');cli.add_argument('--checkpoint',type=Path)
    cli.add_argument('--output',type=Path);cli.add_argument('--dfs-checkpoints',nargs='*',type=Path,default=[])
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));ds=domain_bits(p);cert=Certificates(p)
    cores=archive();checked=replay_for(p);rows=[]
    for c in cores:
        record=dict(size=len(c),previous_physical_clauses=len(cert.conflicts(c)))
        record['replayed']=bool(checked.check(ds,[c]))
        assert record['replayed']
        for trial in range(4):assert checked.check(ds,[c]) and checked.check(ds,[c])
        if len(c)<=27:
            from unittest.mock import patch
            import r30_gold
            original=r30_gold.GoldMatching
            def prior(problem,domains,*args,**kwargs):
                check=original(problem,domains,*args,**kwargs)
                if c<=check.mate.keys() and GeographicConflicts(problem,domains).impossible(c):
                    check.m.add_bool_or([~check.mate[e] for e in c])
                return check
            for mode in ('baseline','prior_with_same_hint','reused'):
                with patch.object(r30_gold,'GoldMatching',prior if mode=='prior_with_same_hint' else original):
                    out=search(p,bonds=c,geography=True,geographic_cores=[c] if mode=='reused' else (),max_rounds=1,seconds=3)
                record[mode]={k:out[k] for k in ('status','cp_calls','branches')}
        rows.append(record)
    report=dict(geographic_cores=[sorted(c) for c in cores],original=rows,replay_work=dict(checked.work))
    if args.checkpoint:report['native_screen']=screen(p,args.checkpoint,cores)
    if args.dfs_checkpoints:
        from r32_gold_watch import BondWatcher
        report['domain_stacks']=[]
        for path in args.dfs_checkpoints:
            data=read_json(path)['payload'];data=data.get('dfs',data)
            if data['fingerprint']!=p.fingerprint:raise ValueError('Installed DFS stack drawing differs.')
            watcher=BondWatcher(p,(),geographic=cores);totals=Counter()
            for frame in data['stack']:
                domains=[int(v,16) for v in frame['domains']]
                changed=watcher.prune(domains);totals['frames']+=1;totals['changed']+=changed
                totals['unit_or_conflict_frames']+=bool(watcher.actions)
            report['domain_stacks'].append(dict(stream=path.parent.name,removed=watcher.removed,
                                                 checked_pairs=watcher.pair_checks,**totals))
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))

if __name__=='__main__':main()
