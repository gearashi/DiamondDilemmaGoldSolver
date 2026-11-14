"""Gold physical conflict, geographic seed, and conditional topology experiments."""
import argparse
import json
from pathlib import Path

from r28_exchange import read_json
from r28_geo_sewing import ring_census, side_options
from r29_gold_geography import checked_domains
from r31_geographic import GeographicConflicts, departure_cores, physical_bonds, salvage
from r31_gold_connectivity import PhaseInterface, connected_colors
from r31_gold_pricing import price_rings
from r31_gold_switch import sewn_switches
from search_problem import SearchProblem


def run(problem, method, request, seconds=5, work=10000):
    domains = checked_domains(problem,None if 'domains' not in request else tuple(
        sum(1<<v for v in set(row)) for row in request['domains']))
    bonds = request.get('bonds',())
    if method=='geography':
        checker=GeographicConflicts(problem,domains)
        core=checker.learn(bonds)
        return dict(status='UNKNOWN' if core is None else 'INFEASIBLE',
                    core=None if core is None else sorted(core),queries=checker.queries)
    if method=='departure':
        return departure_cores(problem,physical_bonds(problem,request['codes']),
                               trials=request.get('trials',96),seconds=seconds)
    if method=='salvage':
        return salvage(problem,domains,request.get('required',()),bonds,seconds)
    if method in ('parents','flow','lazy'):
        return connected_colors(problem,domains,method,seconds)
    if method=='rings':
        options=side_options(problem,domains)
        out=price_rings(problem,options,max_work=work)
        if request.get('recheck'):
            out['integer_check']=ring_census(problem,options,max_work=work,seconds=seconds)
        return out
    if method=='phases':
        interface=PhaseInterface(problem)
        queries=request.get('queries',[request])
        return [interface.project({int(t):v for t,v in row.get('pins',{}).items()},
                                  row.get('boundary',()),work) for row in queries]
    if method=='switches':
        return sewn_switches(problem,domains,bonds,request['phases'],request['trades'],work)
    raise ValueError("Unknown Gold experiment.")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('method',choices=('geography','departure','salvage','parents','flow','lazy','rings','phases','switches'))
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--input',type=Path)
    cli.add_argument('--output',type=Path)
    cli.add_argument('--seconds',type=float,default=5)
    cli.add_argument('--work',type=int,default=10000)
    args=cli.parse_args()
    if args.seconds<=0 or args.work<1:cli.error('Budgets must be positive.')
    out=run(SearchProblem(read_json(args.tiles)),args.method,read_json(args.input) if args.input else {},
            args.seconds,args.work)
    if args.output:
        from io_utils import atomic_json
        atomic_json(args.output,out)
    print(json.dumps(out))


if __name__=='__main__':
    main()
