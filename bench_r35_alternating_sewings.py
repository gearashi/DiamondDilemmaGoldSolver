"""Compare rollback/memoized alternating seam exchanges with independent complete loop traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random

from r27_structure_filters import domain_bits
from r28_exchange import read_json
from r29_gold_phase import EndpointCuts
from r30_gold import search
from r32_gold_bonds import endpoints,loop_groups
from r35_alternating_sewings import Family,common_phases,complete
from search_problem import SearchProblem


def partner(p,original,lengths,seed=35,include_blanks=True):
    rng=Random(seed)
    inputs=EndpointCuts(p)
    phases=original['phases']
    by_class=defaultdict(list)
    for a,b in original['pairs']:
        first=(inputs.sides[a][0],inputs.sides[a][1]^phases[a//3])
        second=(inputs.sides[b][0],inputs.sides[b][1]^phases[b//3])
        if first>second:a,b,first,second=b,a,second,first
        by_class[first].append((a,b))
    for pairs in by_class.values():rng.shuffle(pairs)
    donor={tuple(sorted(e)) for e in original['pairs']}
    def take(size,blank):
        pools=sorted((lst for (mask,_),lst in by_class.items() if bool(mask)!=blank and len(lst)>=size),
                     key=len,reverse=True)
        for pool in pools:
            for _ in range(40):
                choice=rng.sample(pool,size)
                rotated=choice[1:]+choice[:1]
                trial={tuple(sorted((a,d))) for (a,b),(c,d) in zip(choice,rotated)}
                if any(a//3==b//3 for a,b in trial):continue
                for e in choice:pool.remove(e)
                donor.difference_update(tuple(sorted(e)) for e in choice)
                donor.update(trial)
                return True
        return False
    for size in lengths:
        if not take(size,False):break
    if include_blanks:
        for _ in range(3):take(3,True)
    result=dict(phases=list(phases),pairs=sorted(donor))
    complete(p,result['pairs'],phases,inputs)
    return result


def oracle(p,family,count):
    arcs=endpoints(p)
    slots=[tuple(k for k in range(1,12) if (d,k) in arcs) for d in range(3*p.n)]
    return sum(not loop_groups(arcs,slots,family.pairs(mask),p.total_segments) for mask in range(count))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    cli.add_argument('--save-parents',type=Path)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles));saved=read_json('docs/R35_CONNECTED_TOURS.json')
    root=saved['contract']
    phases,state=common_phases(p,saved['raw'],root)
    out=dict(distinct_unrestricted_phase_status=state)
    cases=[]
    for width in (4,10,18):
        donor=partner(p,root,([3,4,5]*6)[:width])
        fam=Family(p,root,donor,root['phases'])
        measure=dict(width=width)
        for memo in (False,True):
            result=fam.solve(max_nodes=5000000,memo=memo,geography=True,count_after_exclusion=True)
            assert result['complete'] and result['gold'] is None
            measure['memo' if memo else 'rollback']=result
        assert measure['memo']['connected_full']==measure['rollback']['connected_full']
        measure['geography_only']=fam.solve(geography=True)
        assert measure['geography_only']['geographic_status']=='INFEASIBLE'
        if width<=10:
            count=1<<len(fam.groups)
            checked=oracle(p,fam,count)
            assert checked==measure['memo']['connected_nonblank']
            measure['independent']=dict(tour_traces=count,connected=checked)
        if width==18 and args.save_parents:
            args.save_parents.write_text(json.dumps([root,donor],separators=(',',':'))+'\n',encoding='utf-8')
        cases.append(measure)
    out['original_long_exchange']=cases
    # A genuinely different full CP parent under the first parent's fixed phase.
    selected=next(tuple(pair) for pair in root['pairs'] if p.masks[3*(pair[0]//3),pair[0]%3]
                  and tuple(pair) not in set(map(tuple,saved['raw']['pairs'])))
    follow=search(p,domains=domain_bits(p),pins=dict(enumerate(root['phases'])),forbidden=[selected],
                  tour_mode='contract',project_blanks=True,seconds=30,max_rounds=4)
    out['fixed_phase_parent']={k:v for k,v in follow.items() if k in ('status','cp_calls','branches')}
    if follow['status']=='CONNECTED_SEWING':
        out['fixed_phase_parent']['phases']=follow['phases']
        out['fixed_phase_parent']['pairs']=follow['pairs']
        fam=Family(p,root,follow,root['phases'])
        out['fixed_phase_parent']['exchange']=fam.solve(max_nodes=1000000,memo=True,geography=True,count_after_exclusion=True)
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':
    main()
