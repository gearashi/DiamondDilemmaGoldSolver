"""Finite work-count comparisons for the R24 learning candidates.

Run serially from frozen sources. This is not an original Gold solve benchmark.
The legacy fixpoint includes default seam/Hall/cluster/neighborhood/parity plus
existing path consistency and scalar conservation. Native SAT query comparisons
use exactly the same relaxed model and supported-value target on both sides.
"""
import argparse,hashlib,json,platform,random,time
from pathlib import Path
from constraint_deep import DeepPruner
from constraint_algebra import AffinePruner
from constraint_pb import PBLearner,row,derive_rows,filter_rows
from constraint_interface import InterfaceProjector
from constraint_relations import capacity_supports
from constraint_sat_assist import SatAssist
from test_constraint_sat_domains import fixture,ordinary_fixpoint,admitted
from test_constraint_dfs import brute


def legacy_fixpoint(problem,domains):
    domains=list(domains);deep=DeepPruner(problem);passes=0
    while all(domains):
        before=tuple(domains);domains=ordinary_fixpoint(problem,domains)
        if all(domains):deep.paths(domains,max_values=12)
        if all(domains):deep.conservation(domains)
        passes+=1
        if tuple(domains)==before:break
    return domains,passes


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=60);parser.add_argument('--seed',type=int,default=240071)
    args=parser.parse_args();rng=random.Random(args.seed);problem,first=fixture();solutions=brute(problem,'edges')
    report={'scope':'finite four-cell exact-support comparisons; no original puzzle solve claim',
            'seed':args.seed,'python':platform.python_version(),'cases':args.cases,'baseline':'default propagation + existing path consistency and scalar conservation to fixpoint',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('.').glob('constraint_*.py'))},
            'filters':{},'sat':{},'discriminators':{}}
    variants={'joint_conservation':(lambda:DeepPruner(problem),'joint_conservation'),
              'affine_reasoning':(lambda:AffinePruner(problem),'prune'),
              'pb_learning':(lambda:PBLearner(problem),'prune'),
              'obstruction_rules':(lambda:PBLearner(problem,derive=False,discover=True),'prune'),
              'interface_formulas':(lambda:InterfaceProjector(problem),'prune')}
    contexts=[]
    for case in range(args.cases):
        domains=first.copy() if case==0 else [sum(1<<v for v in row if rng.random()<.75) for row in problem.domains]
        before=admitted(solutions,domains);fixed,passes=legacy_fixpoint(problem,domains)
        assert admitted(solutions,fixed)==before
        contexts.append((fixed,before))
    for name,(factory,method) in variants.items():
        helper=factory();extra=failures=eligible=0
        for fixed,expected in contexts:
            if not all(fixed):continue
            current=fixed.copy();getattr(helper,method)(current)
            assert admitted(solutions,current)==expected
            extra+=sum((a&~b).bit_count() for a,b in zip(fixed,current));failures+=not all(current);eligible+=1
        report['filters'][name]={'nonempty_contexts':eligible,'additional_deletions_after_legacy_fixpoint':extra,
                                 'additional_failures':failures,'stats':helper.stats}
    try:
        import pysat
        native=True
    except ImportError:native=False
    if native:
        for name,options in [('baseline',{}),('pool',{'witness_pool':True}),('cover',{'support_cover':True}),('cover_pool',{'support_cover':True,'witness_pool':True})]:
            initial=[sum(1<<v for v in row) for row in problem.domains];helper=SatAssist(problem,initial,**options)
            try:
                assert helper.prepare(seconds=5)
                answers=[]
                for fixed,expected in contexts:
                    if not all(fixed):continue
                    result=helper.filter_domains(fixed,seconds=2,max_candidates=128)
                    assert admitted(solutions,result)==expected;answers.append(result)
                report['sat'][name]={'stats':helper.stats.copy(),'answer_digest':hashlib.sha256(repr(answers).encode()).hexdigest()}
            finally:helper.close()
        assert len({item['answer_digest'] for item in report['sat'].values()})==1
    else:report['sat']['status']='unavailable; skipped native query comparisons'
    choices=[{0,1}]*3+[{2,3}]*3;capacities={0:1,1:1,2:2,3:2}
    report['discriminators']['joint_capacity']={'individual_bounds_pass':all(sum(row=={k} for row in choices)<=v<=sum(k in row for row in choices) for k,v in capacities.items()),
                                               'joint_proved_infeasible':capacity_supports(choices,capacities) is None}
    base=[row([(0,1,-1),(1,1,-1)],-1),row([(0,1,-1),(2,1,-1)],-1),row([(1,1,-1),(2,1,-1)],-1),row([(0,1,1),(1,1,1),(2,1,1)],1)]
    original=[3]*3;filter_rows(original,base);learned,steps,work=derive_rows(base);current=[3]*3;filter_rows(current,learned)
    report['discriminators']['pb_combination']={'primitive_bounds_leave_all_values':original==[3]*3,'derived_proves_infeasible':0 in current,'cuts':len(steps),'pair_checks':work}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(args.output),'filters':{k:v['additional_deletions_after_legacy_fixpoint'] for k,v in report['filters'].items()},
                      'sat_calls':{k:v.get('stats',{}).get('sat_assist_domain_calls') for k,v in report['sat'].items() if isinstance(v,dict)}}))

if __name__=='__main__':main()
