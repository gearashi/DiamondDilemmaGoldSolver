"""Matched public DFS/GPU calls on original scoped frontiers, including resume."""
from pathlib import Path
import argparse,json
from constraint_search import run_solver
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem
from time import perf_counter


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,default=Path('../data/tiles.json'))
    ap.add_argument('--gpu',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    data=json.loads(args.tiles.read_bytes())
    record=active_catalog(SearchProblem(data))
    seed=record['codes']
    cases = {'mixed_two':(78,150),'mixed_three':(96,130,144),'mixed_five':(99,100,105,106,107),
             'binary':(5,102),'ternary':(103,129,144),'guard6':record['post_master'][1],
             'guard8':record['post_master'][0],'guard5':record['post_master'][3]}
    out={}
    for name,cells in cases.items():
        rows=[]
        for switch in (False,True):
            p=SearchProblem(data,physical_gold_support=True)
            p.reference_pose_cuts=switch
            for c in cells:p.domains[c]=[seed[c]]
            opts={}
            if args.gpu:opts={'gpu_cooperative':True,'gpu_event_propagation':True}
            started=perf_counter()
            ans=run_solver(p,'gpu-domains' if args.gpu else 'dfs',seconds=12,target='gold',replicas=1,options=opts)
            stats=ans['stats']
            rows.append(dict(status=ans['status'],nodes=ans['nodes'],seconds=round(perf_counter()-started,3),
                             counts={k:stats.get(k,0) for k in ('reference_pose_closed','reference_pose_removed','backtracks','propagations','arc_revisions','hall_checks','domain_removals','support_checks','gpu_domain_state_bytes','gpu_domain_completed')}))
        out[name]=dict(held=cells,paired=rows)
        print(name,[(r['status'],r['nodes'],r['counts']['reference_pose_closed'],r['seconds']) for r in rows],flush=True)
        args.output.write_text(json.dumps(out,indent=2),encoding='utf8')

if __name__=='__main__':main()
