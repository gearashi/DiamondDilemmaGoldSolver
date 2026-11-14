"""Round 28 scoped exact logic tools for the Diamond Dilemma solver."""
import argparse
import json
from pathlib import Path

from r27_reason_maxsat import domain_rows
from r28_logic_tail import binary_search,cut_bound
from r28_logic_reuse import dominance,direct_cnf,ClauseCache,TrailReplay,live_session
from r28_logic_sat import seam_hitting_set,cegar,determining_support,project_models
from r28_logic_region import regional_interpolant
from search_problem import SearchProblem


def run(problem,domains,method,*,seconds=20,target='gold',threshold=None,cores=(),cells=(0,),
        max_assignments=4096,encoding='support'):
    if method=='scc':
        status,codes,work = binary_search(problem,domains,target,seconds=seconds)
        return dict(status=status,codes=codes,work=work)
    if method=='cut':
        return cut_bound(problem,domains)
    if method=='cores':
        return seam_hitting_set(problem,domains,previous=cores,seconds=seconds,threshold=threshold)
    if method=='cegar':
        return cegar(problem,domains,target,seconds=seconds)
    if method=='interpolate':
        return regional_interpolant(problem,domains,cells,target=target)
    if method=='dominance':
        return dominance(problem,domains,cells,target=target,max_assignments=max_assignments)
    ids,clauses,factors = direct_cnf(problem,domains,encoding=encoding)
    variables = [v for row in ids for v in row.values()]
    if method=='support':
        support = determining_support(clauses,variables)
        return dict(physical=len(variables),support=support,
                    board_projections=project_models(clauses,support,variables,limit=1))
    if method=='process-session':
        from r28_logic_worker import session_probe
        return session_probe(clauses,max_slices=max(1,int(seconds*40)))
    if method=='session':
        return live_session(clauses,seconds=seconds)
    if method=='replay':
        replay = TrailReplay(clauses)
        original = replay.propagate(())
        if original is None:
            return dict(status='unit_conflict',scans=replay.scans)
        result = replay.propagate(())
        return dict(status='unit_fixpoint',propagated=len(result),reused=replay.reused,
                    scans=replay.scans,replay_checks=replay.checks)
    if method!='factor':
        raise ValueError('Unknown logic method.')
    cache = ClauseCache()
    model = cache.solve(clauses)
    result = dict(factors=factors,clauses=len(clauses),component_solves=cache.calls,
                  status='infeasible' if model is None else 'edge_witness')
    if model is not None:
        # Simplification can omit unconstrained physical Boolean variables.
        codes = [next(v for v,lit in row.items() if lit in model) for row in ids]
        report = problem.validate(codes)
        if not report['unique_tiles'] or report['matched_edges'] != len(problem.edges):
            raise RuntimeError('CNF component witness violated physical validation.')
        result.update(codes=codes,gold_valid=report['valid'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=('scc','cut','cores','cegar','interpolate','support','dominance',
                                           'factor','replay','session','process-session'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--domains',type=Path,required=True,help='Explicit legal orientation-code rows, one per cell')
    parser.add_argument('--target',choices=('edges','gold'),default='gold',help='For SCC, CEGAR, interpolation and dominance')
    parser.add_argument('--encoding',choices=('pair','biclique','support','bitcode'),default='support')
    parser.add_argument('--seconds',type=float,default=20,help='For SCC, core, CEGAR and session methods')
    parser.add_argument('--threshold',type=int,help='Stop the core optimizer once this edge score is impossible')
    parser.add_argument('--cores',type=Path,help='Previous cores result JSON; each imported core is rechecked')
    parser.add_argument('--cells',type=int,nargs='+',default=[0])
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))
    result = run(SearchProblem(read(args.data)),read(args.domains),args.method,seconds=args.seconds,
                 target=args.target,threshold=args.threshold,cores=read(args.cores)['cores'] if args.cores else (),
                 cells=args.cells,encoding=args.encoding)
    print(json.dumps(result))
    return result


if __name__=='__main__':
    main()
