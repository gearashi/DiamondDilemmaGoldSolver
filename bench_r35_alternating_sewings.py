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


def near_family(problem,reference,required,source):
    """Exhaust two recorded donors, including every labelled blank-side choice."""
    from bench_r35_near_reference_tour import raw_cycles
    from r30_gold import embedding
    from r31_geographic import ribbon_genus
    import hashlib
    phase=int(source['phase'],0)
    phases=[(phase>>tile)&1 for tile in range(problem.n)]
    parents=[dict(pairs=row['pairs'],phases=phases) for row in source['parents']]
    checked,status=common_phases(problem,*parents)
    if status!='COMPATIBLE' or checked!=phases:raise ValueError('Expected two common-phase donors.')
    seams=[set(map(tuple,row['pairs'])) for row in parents]
    fixed=set.intersection(*seams)
    neighbors=defaultdict(set)
    for a,b in set.union(*seams)-fixed:
        neighbors[a].add(b);neighbors[b].add(a)
    def match(open_sides,selected):
        if not open_sides:
            yield fixed|selected
            return
        a=min(open_sides)
        for b in sorted(neighbors[a]&open_sides):
            yield from match(open_sides-{a,b},selected|{tuple(sorted((a,b)))})
    family=Family(problem,*parents,phases)
    counted=family.solve(geography=True,memo=True,count_after_exclusion=True)
    direct=family.solve(geography=True)
    domains=domain_bits(problem)
    rows=[]
    for mesh in match(set(neighbors),set()):
        assert len(mesh)==240 and required<=mesh
        cycles=raw_cycles(problem,mesh)
        mate=[-1]*(3*problem.n)
        for a,b in mesh:mate[a]=b;mate[b]=a
        codes=embedding(problem,domains,mate)
        if codes is not None and not problem.validate(codes)['valid']:
            raise ValueError('Geographic witness failed independent Gold validation.')
        rows.append(dict(connected=len(cycles)==1,genus=ribbon_genus(problem,mesh),
                         retained=len(reference&mesh),cycles=cycles,gold=codes,
                         mate_sha256=hashlib.sha256(json.dumps(sorted(mesh),separators=(',',':')).encode()).hexdigest()))
    assert counted['complete'] and direct['complete']
    assert len(rows)==len({r['mate_sha256'] for r in rows})==counted['possible_full_sewings']
    assert sum(r['connected'] for r in rows)==counted['connected_full']
    assert bool(direct['gold'])==any(r['gold'] for r in rows)
    return dict(phase=source['phase'],fingerprint=problem.fingerprint,parents=source['parents'],
                shared_seams=len(fixed),shared_reference=len(fixed&reference),
                identical_to_second_retention=(fixed&reference)==(seams[1]&reference),
                nonblank_group_sizes=[len(g[0]) for g in family.groups],
                blank_group_sizes=[len(g[0]) for g in family.blanks],
                permitted_root_poses=sum(sum(bool(mask&(1<<rot)) for rot in range(3)) for mask in domains),
                projection=direct,connected_count=counted,all_choices=rows)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    cli.add_argument('--save-parents',type=Path)
    cli.add_argument('--near-parents',type=Path,help='Check each complete sewing from two recorded common-phase parents.')
    args=cli.parse_args()
    if args.near_parents:
        from bench_r35_loss_boundary344 import problem_scope
        problem,reference,_,required=problem_scope(args.tiles)
        result=near_family(problem,reference,required,read_json(args.near_parents))
        if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(result['connected_count']['possible_full_sewings'],result['connected_count']['connected_full'],result['projection']['geographic_status'])
        return
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
