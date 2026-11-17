"""Check repeatable physical gains from the remote pool policy across fresh-board trajectories."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
from r31_fresh import Engine
from r28_exchange import read_json
from search_problem import SearchProblem
from io_utils import atomic_json


def run(p,out):
    result=[]
    with TemporaryDirectory(dir=out.parent) as folder:
        for seed in (4,5,6,7,11,19):
            row={}
            for imports in (0,2,3):
                e=Engine(p,joint=True,forest=True,contacts=-1,pool=8,remote_imports=imports)
                db=str(Path(folder)/'routes.sqlite')
                args=dict(backend='cpu',dispatches=0,replicas=128,ranked=True,trajectory_db=db)
                prefix=e.run(seed,rounds=3,**args)
                trial=e.run(seed,rounds=12,**args)
                warm=e.run(seed,rounds=12,**args)
                assert trial['codes']==warm['codes'] and warm['new_repair_steps']==0
                assert trial['reused_repair_steps']==3
                assert p.validate(trial['codes'])['matched_edges']==trial['score']
                row[str(imports)]=dict(prefix=prefix['score'],score=trial['score'],native=trial.get('work',trial.get('stats',{})),
                                       warm_rounds=warm.get('new_repair_steps'),physical=trial['codes'])
            result.append(dict(seed=seed,modes=row))
            atomic_json(out,result)
            print(seed,[(k,v['score']) for k,v in row.items()],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='../data/tiles.json')
    cli.add_argument('--output',type=Path,required=True)
    a=cli.parse_args()
    run(SearchProblem(read_json(a.tiles)),a.output)
