"""Reproducible structural search comparisons. Every number is a work count, without time extrapolation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from r27_structure_filters import Work, _perfect_supports, _regions, bits, domain_bits, filter_domains, oriented_segments, physical_graph
from r27_structure_fixtures import independent, planted
from r27_structure_region import regional_count
from r27_structure_sat import sat_boards
from r27_structure_search import _edge_filter, enumerate_boards
from search_problem import SearchProblem


def clean(r):
    return {k:v for k,v in r.items() if k not in ('witnesses','witness','cursor')}


def circuit(problem,ds):
    from constraint_circuit import build_circuit_model
    from ortools.sat.python import cp_model
    built=build_circuit_model(problem,[tuple(bits(v)) for v in ds],contract_fixed=True)
    s=cp_model.CpSolver()
    s.parameters.num_search_workers=1
    s.parameters.max_deterministic_time=.8
    results=set()
    branches=conflicts=0
    while True:
        status=s.solve(built.model)
        branches+=s.num_branches;conflicts+=s.num_conflicts
        if status==cp_model.INFEASIBLE:break
        if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):
            return dict(complete=False,solutions=len(results),branches=branches,conflicts=conflicts)
        result=tuple(s.value(c) for c in built.codes)
        if not problem.validate(list(result))['valid']:raise AssertionError('Invalid circuit board.')
        if result in results:raise AssertionError('Repeated projected circuit board.')
        results.add(result)
        built.model.add_forbidden_assignments(built.codes,[result])
    return dict(complete=True,solutions=len(results),branches=branches,conflicts=conflicts)


def finite():
    from test_constraint_sat_domains import fixture,ordinary_fixpoint
    from constraint_dfs import solve
    from r27_structure_cover import cover_page
    results={}
    for label,p,ds in [('legacy-four',*fixture()),('sphere-three-free',*planted(1,3)),
                       ('sphere-seven-free',*planted(1,7,37)),('repeated-types',*planted(1,5,17,uniform_ports=True)),('sealed-split',*planted(1,0,23,True))]:
        controls=ordinary_fixpoint(p,ds)
        res={}
        expected=None if label in ('sphere-seven-free','repeated-types') else len(independent(p,ds))
        res['independent']=expected
        res['ordinary_remaining']=sum(d.bit_count() for d in controls)
        for mode in ('cell','seam','seam-table','coarse','cover','glue','embedded','tour','directed-tour'):
            try:r=clean(enumerate_boards(p,mode,domains=ds,max_work=500000))
            except ValueError as exc:r={'unsupported':str(exc)}
            if expected is not None and r.get('complete') and r['solutions']!=expected:raise AssertionError((label,mode,r))
            res[mode]=r
        if expected is None:
            if not res['cell']['complete']:raise AssertionError('Missing complete finite control.')
            expected=res['cell']['solutions']
        for mode,mod in [('lazy',None),('reach',None),('eliminate',None),('mod-small',3),('mod-full',p.total_segments)]:
            r=clean(sat_boards(p,ds,'modular' if mod else mode,modulus=mod,max_work=1300000))
            if r['complete'] and r['solutions']!=expected:raise AssertionError((label,mode))
            res[mode]=r
        res['circuit']=circuit(p,ds)
        if res['circuit']['complete'] and res['circuit']['solutions']!=expected:raise AssertionError('circuit disagreement')
        # Released single-witness search uses the same restricted original domains.
        p.domains=[list(bits(x)) for x in ds]
        from time import monotonic
        baseline=solve(p,target='gold',seconds=8,options={'native':'python'})
        res['released_dfs']={'complete':baseline['complete'],'nodes':baseline.get('nodes'),
                             'status':baseline.get('status')}
        res['filter']={}
        for rule in ('partners','regions','euler','separator','passages'):
            after,stats=filter_domains(p,tuple(controls),[rule],max_work=500000)
            res['filter'][rule]={'removed':sum(m.bit_count()-n.bit_count() for m,n in zip(controls,after)),
                                'infeasible':not all(after),'work':stats}
        if label!='legacy-four':
            res['regional']={}
            for cut,cells in [('face',[0]),('strip',[0,1]),('hemisphere',range(5))]:
                for mode,compact,det in [('pairs',False,False),('word',True,False),('trees',True,True)]:
                    row=clean(regional_count(p,cells,ds,compact=compact,determinant=det,max_work=800000))
                    if row['complete'] and row['solutions']!=expected:raise AssertionError((label,cut,mode))
                    res['regional'][cut+'-'+mode]=row
        results[label]=res
    return results


def full(private=None):
    p=SearchProblem(json.loads(Path('data/tiles.json').read_text(encoding='utf-8-sig')))
    ds=domain_bits(p)
    result={'original':{'segments':p.total_segments,'crossing_tiles':sum(_regions(oriented_segments(p,i*3)) is None for i in range(p.n))}}
    for rule in ('partners','regions','euler','separator','passages'):
        after,stats=filter_domains(p,ds,[rule],max_work=2600000 if rule=='separator' else 800000)
        result['original'][rule]={'work':stats,'removed':sum((a^b).bit_count() for a,b in zip(ds,after))}
    w=Work(1200000)
    graph=physical_graph(p,ds,w)
    allowed=_perfect_supports(graph,w)
    result['original']['physical']={'forced':sum(len(v)==1 for v in allowed)//2,
                      'unsupported_pairs':sum(len(a)-len(b) for a,b in zip(graph,allowed))//2,'work':dict(w.counters)}
    for s in ('seam','seam-table','coarse','glue','embedded','tour','directed-tour'):
        row=clean(enumerate_boards(p,s,domains=ds,max_work=90000))
        result['original'][s]=row
    for mode,mod in [('eliminate',None),('mod5',5),('mod73',73),('mod365',365)]:
        # Distinguish a broad-root encoding limit from actual satisfiability.
        result['original'][mode]=clean(sat_boards(p,ds,'modular' if mod else mode,modulus=mod,max_work=80000))
    if private:
        codes=json.loads(Path(private).read_text(encoding='utf-8-sig'))['codes']
        report=p.validate(codes)
        fixed=tuple(1<<v for v in codes)
        result['incumbent']={'matched':report['matched_edges'],'segments':p.total_segments}
        for mode,mod in [('lazy',None),('eliminate',None),('mod5',5),('mod73',73),('mod365',365)]:
            result['incumbent'][mode]=clean(sat_boards(p,fixed,'modular' if mod else mode,modulus=mod,max_work=1600000))
    p,ds=planted(4,8,37)
    fullcase={'source':'synthetic noncrossing on the 160-cell board','free_cells':8}
    for mode in ('cell','seam','seam-table','coarse','cover','glue','embedded','tour','directed-tour'):
        fullcase[mode]=clean(enumerate_boards(p,mode,domains=ds,max_work=500000))
    for name,cells in [('face',range(16)),('strip',range(32)),('hemisphere',range(80))]:
        for mode,compact,det in [('pairs',False,False),('word',True,False),('trees',True,True)]:
            fullcase[name+'-'+mode]=clean(regional_count(p,cells,ds,compact=compact,determinant=det,max_work=700000))
    for rule in ('partners','regions','euler','separator','passages'):
        after,stats=filter_domains(p,ds,[rule],max_work=500000)
        fullcase[rule]={'work':stats,'removed':sum((a^b).bit_count() for a,b in zip(ds,after))}
    result['planted160']=fullcase
    from r27_structure_cycle_fixture import planted_365
    p,ds=planted_365()
    result['positive365']={}
    for mod in (5,73,365):
        row=clean(sat_boards(p,ds,'modular',modulus=mod,max_work=1500000))
        if not row['complete'] or row['solutions']!=1:raise AssertionError('Lost planted 365-strand cycle.')
        result['positive365'][str(mod)]=row
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',choices=['finite','full'],default='finite')
    parser.add_argument('--incumbent',help='Optional original private reference board; only aggregate counters are emitted.')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    data=finite() if args.section=='finite' else full(args.incumbent)
    contents=json.dumps(data,indent=2,sort_keys=True)
    if args.output:args.output.write_text(contents+'\n',encoding='utf-8')
    else:print(contents)

