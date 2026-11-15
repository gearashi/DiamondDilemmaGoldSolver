"""Measure scarce-anchor elimination after signed mate and Euler tests on fresh phase panels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bench_r35_geographic_neighbors import four_flip_neighbors
from io_utils import atomic_json
from r31_gold_connectivity import PhaseInterface
from r32_gold_bonds import inventory
from r35_phase_anchors import allowed_mates,narrow
from r35_signed_mate_support import matching,raw_edges,reciprocal_classes
from search_problem import SearchProblem


def record(problem,phase,required,forbidden,interface):
    pins={t:(phase>>t)&1 for t in range(problem.n)}
    sides,families=reciprocal_classes(problem.tiles)
    allowed=allowed_mates(problem,pins,required,forbidden)
    degrees=Counter(side for edge in allowed for side in edge)
    def rank(tile):
        a,b,c=sorted(degrees[3*tile+s] for s in range(3))
        return a*b*c,b,c,tile
    scarce=sorted(range(problem.n),key=rank)[:5]
    failed=[]
    for f in families:
        if len(f)>16:continue
        edges=raw_edges(sides,f,allowed)
        if not matching(f,edges,pins):failed.append(len(f))
    out={}
    for label,anchors in [('tile_zero',[0]),('scarce',scarce)]:
        result=narrow(problem,None,pins,required=required,forbidden=forbidden,anchors=anchors,unique=False,project=False,repeat=False)
        out[label]={k:v for k,v in result.items() if k!='domains'}
    weight=sum(row[(phase>>i)&1][-1] for i,row in enumerate(interface.rows))
    return dict(phase=hex(phase),euler_weight=weight,euler=weight in interface.euler,small_mate_failures=failed,anchors=scarce,results=out)


def source_reuse(problem, rule, required, *, public):
    from bench_r35_mixed_sewings import union_sat
    from bench_r35_phase_rules import matches
    from r30_gold import search
    from r35_anchor_rules import check_rules
    from r35_signed_conditions import SignedConditions

    pins = dict(rule['pins'])
    accepted = allowed_mates(problem,pins)
    independent = union_sat(problem,accepted,anchor=rule['anchor'])
    if independent['unresolved_roots']:
        raise AssertionError('The reduced pin conflict failed independent SAT replay.')
    source = int(rule['source_phase'],16)
    count, nearby = four_flip_neighbors(problem,required,source,100000)
    if count != len(nearby):
        raise ValueError('The sampled phase neighborhood is incomplete.')
    euler = SignedConditions(problem,euler=True)
    mates = SignedConditions(problem,mates=True)
    hits = [phase for phase in nearby if matches(phase,rule['pins'])]
    checked = {phase:check_rules(problem,{t:phase>>t&1 for t in range(problem.n)},float('inf'))[1]
               for phase in hits}
    if any(not stats['terminal'] for stats in checked.values()):
        raise ArithmeticError('The public rule cache failed to reproduce a covered neighbor.')
    one_hits = sum(matches(source^(1<<i),rule['pins']) for i in range(problem.n))
    report = dict(source=hex(source),pins=len(pins),native=independent,
                  four_flip_neighbors=count,four_flip_hits=len(hits),
                  covered_passing_euler=sum(euler.accepts(x) for x in hits),
                  covered_passing_mates=sum(mates.accepts(x) for x in hits),
                  covered_passing_both=sum(euler.accepts(x) and mates.accepts(x) for x in hits),
                  one_flip_neighbors=problem.n,one_flip_hits=one_hits,
                  verified_neighbor_queries=len(checked),
                  neighbor_rule_replays=sum(s['replayed'] for s in checked.values()))
    if public:
        full_pins = {t:source>>t&1 for t in range(problem.n)}
        outcomes = {}
        for mode in (None,'roots','rules'):
            result = search(problem,geography=True,matching=True,learned_geometry=True,pins=full_pins,
                            seconds=8,phase_anchors=mode)
            outcomes[mode or 'baseline'] = {k:result[k] for k in ('status','cp_calls','branches','phase_anchors',
                                                                  'phase_rules') if k in result}
        report['public']=outcomes
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--cases',type=Path,help='Resume or save the individual held-out measurements.')
    cli.add_argument('--compare',action='store_true',help='Also run the baseline and two public anchor policies.')
    args=cli.parse_args()
    read=lambda path:json.loads(path.read_bytes())
    root=Path(__file__).resolve().parent/'docs'
    p=SearchProblem(read(args.tiles));inv=inventory(p);interface=PhaseInterface(p)
    phases=set()
    for donor in read(root/'R35_GEOGRAPHIC_LOSS_CORES.json')['reports']:
        phase=int(donor['phase'],16)
        _,nearby=four_flip_neighbors(p,inv.required,phase,100000)
        phases.update(nearby);phases.add(phase)
    exclude={int(row['phase'],16) for row in read(root/'R35_PHASE_ANCHORS.json')['cases']}
    phases-=exclude
    report=read(args.cases) if args.cases and args.cases.exists() else dict(fingerprint=p.fingerprint,cases=[])
    completed={int(row['phase'],16) for row in report['cases']}
    if report['fingerprint']!=p.fingerprint or not completed<=phases or len(completed)!=len(report['cases']):
        raise ValueError('The saved cases do not belong to this puzzle and phase panel.')
    for phase in sorted(phases-completed):
        report['cases'].append(record(p,phase,inv.required,inv.forbidden,interface))
        if args.cases:atomic_json(args.cases,report)
    rows=report['cases']
    summary=dict(fingerprint=p.fingerprint,held_out=len(rows),modes={})
    for label in ('tile_zero','scarce'):
        rejected=[r for r in rows if r['results'][label]['status']=='INFEASIBLE']
        summary['modes'][label]=dict(rejected=len(rejected),passing_euler=sum(r['euler'] for r in rejected),
            passing_small_mate_checks=sum(not r['small_mate_failures'] for r in rejected),
            passing_both=sum(r['euler'] and not r['small_mate_failures'] for r in rejected))
    summary['near_source']=[source_reuse(p,r,inv.required,public=args.compare)
                            for r in read(root/'R35_PHASE_ANCHOR_CORES.json')['rules']]
    atomic_json(args.output,summary)


if __name__=='__main__':
    main()
