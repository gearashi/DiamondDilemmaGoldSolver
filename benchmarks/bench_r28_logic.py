"""Reproducible scoped comparisons for the round 28 logic candidates."""
import argparse
import json
import random
import time
from pathlib import Path
from r28_logic_tail import binary_search,cut_bound
from r28_logic_sat import seam_hitting_set,cegar,determining_support,project_models
from r28_logic_reuse import direct_cnf,ClauseCache,TrailReplay,dominance,live_session
from r28_logic_region import regional_interpolant
from r27_reason_tail import bounded_tail
from r27_reason_maxsat import maximize_seams
from search_fixtures import planted_gold
from search_problem import SearchProblem
from pysat.solvers import Glucose4


def native(clauses,assumptions=()):
    with Glucose4(bootstrap_with=clauses) as solver:
        answer=solver.solve(assumptions=list(assumptions))
        return dict(answer=answer,**solver.accum_stats())


def measured(data,board,budget):
    p=SearchProblem(data)
    if not p.validate(board)['unique_tiles']:
        raise ValueError('Invalid benchmark incumbent.')
    out={}
    planted,surface,reference=planted_gold(28)
    fixture=SearchProblem(planted,surface)
    def rows(p,codes,width,seed):
        rng=random.Random(seed)
        active=rng.sample(range(p.n),width)
        return [sorted({v,3*(v//3)+(v+1)%3}) if c in active else [v] for c,v in enumerate(codes)]
    # Preserve the actual tile permutation while allowing a seeded rotation choice.
    for name,problem,codes in [('incumbent',p,board),('planted',fixture,reference)]:
        for count in (8,32,160):
            r=rows(problem,codes,count,28)
            masks=[sum(1<<v for v in row) for row in r]
            old=bounded_tail(problem,masks,'gold',lambda:None,seconds=2) if count<=8 else None
            scc,b,work=binary_search(problem,r,'gold',seconds=budget)
            if scc=='witness' and not problem.validate(b)['valid']:
                raise RuntimeError('Bad SCC Gold witness.')
            bound=cut_bound(problem,r)
            out[f'{name}_{count}']=dict(scc=scc,scc_work=work,old_tail=None if old is None else [old[0],old[2]],
                                      cut_upper=bound['upper'],cut_score=bound['score'],frustrated=bound['frustrated'])
    r=rows(p,board,160,28)
    out['incumbent_cegar']=cegar(p,r,seconds=budget)
    core=seam_hitting_set(p,r,seconds=budget,threshold=208)
    replay=seam_hitting_set(p,r,previous=core['cores'],seconds=budget,threshold=208)
    baseline=maximize_seams(p,domains=r,seed_board=board,threshold=208,target='edges',seconds=budget)
    def core_stats(result):
        nontrivial=[set(c) for c in result['cores'] if len(c)>1]
        return dict(status=result['status'],calls=result['calls'],upper=result['upper'],cores=len(result['cores']),
                    multi=len(nontrivial),overlap_pairs=sum(bool(a&b) for i,a in enumerate(nontrivial)
                                                               for b in nontrivial[i+1:]))
    out['incumbent_cores']=dict(cold=core_stats(core),rechecked=core_stats(replay),
                                baseline={k:baseline.get(k) for k in ('status','calls','edge_upper_bound')})
    region=regional_interpolant(p,r,range(16))
    out['incumbent_interpolant']={k:region.get(k) for k in ('status','shared','gold_cuts','verified')}
    out['incumbent_interpolant']['widths']=[len(c) for c in region.get('clauses',[])]
    # Use finite feasible physical regions to evaluate determining support,
    # non-bijective replacements and clause factoring on nontrivial witnesses.
    small,small_surface,gold=planted_gold(11,size=1,extra_edges=0)
    small=SearchProblem(small,small_surface)
    tiny=rows(small,gold,small.n,28)
    ids,clauses,_=direct_cnf(small,tiny)
    variables=[v for row in ids for v in row.values()]
    support=determining_support(clauses,variables)
    models=project_models(clauses,support,variables)
    controls=project_models(clauses,variables,variables)
    if set(models)!=set(controls):
        raise RuntimeError('Different physical projections.')
    out['support']=dict(all=len(variables),necessary=len(support),models=len(models))
    more=rows(small,gold,6,28)
    dom=dominance(small,more,[0,1,2])
    out['dominance']={k:dom[k] for k in ('removed','checks','status')}
    out['dominance']['replacements_with_representative']=sum(rep is not None for _,rep in dom['replacements'])
    ids,plain,_=direct_cnf(small,small.domains)
    ids,factored,factors=direct_cnf(small,small.domains,factored=True)
    out['factoring']=dict(plain=len(plain),factored=len(factored),factors=factors,
                          plain_decisions=native(plain)['decisions'],factored_decisions=native(factored)['decisions'])
    for encoding in ('bitcode','support'):
        _,formula,_=direct_cnf(small,small.domains,encoding=encoding)
        out['factoring'][encoding]=dict(clauses=len(formula),decisions=native(formula)['decisions'])
    real_ids,real_cnf,_=direct_cnf(fixture,rows(fixture,reference,32,28))
    cache=ClauseCache()
    warm,cold=TrailReplay(real_cnf),TrailReplay(real_cnf)
    for cell in (0,1,2,3,4,5,6,7):
        assumption=[real_ids[cell][reference[cell]]]
        cache.solve(real_cnf,assumption)
        if warm.propagate(assumption)!=cold.propagate(assumption,reuse=False):
            raise RuntimeError('Replay closure diverged.')
    out['cache_trail']=dict(cache_hits=cache.hits,component_solves=cache.calls,
                           clause_scans=cold.scans,warm_scans=warm.scans,replayed=warm.reused,
                           replay_checks=warm.checks)
    out['native_real']=live_session(factored,slice_conflicts=4,slices=48,seconds=budget)
    # Pigeonhole adversary forces actual interrupted native calls on the tested build.
    n=7
    var=lambda a,b:1+a*n+b
    php=[[var(i,j) for j in range(n)] for i in range(n+1)]
    php.extend([-var(a,k),-var(b,k)] for a in range(n+1) for b in range(a) for k in range(n))
    out['native_stress']=live_session(php,slice_conflicts=128,slices=512,seconds=budget)
    from r28_logic_worker import session_probe
    out['process_reconnect']=session_probe(php,milliseconds=12,max_slices=128)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('board',type=Path)
    parser.add_argument('--seconds',type=float,default=20)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    print(json.dumps(measured(read(args.data),read(args.board)['codes'],args.seconds),indent=2))


if __name__=='__main__':
    main()
