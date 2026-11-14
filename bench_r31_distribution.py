"""Compare geographic budgets on matched scopes and run native prefix-replay controls."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile

from r27_incumbent import BoardModel
from r27_reason_maxsat import maximize_seams
from r28_exchange import read_json
from r31_defect_bands import solve
from r31_sampling import ranked_start, sample_archive
from search_problem import SearchProblem


def compare(problem, rows, seconds, calls, target):
    kwargs = dict(target=target, domains=rows, seconds=seconds, max_calls=calls)
    old = maximize_seams(problem,domains=rows,threshold=target,target='edges',seconds=seconds,conflicts=2000*calls)
    out = {'released':{k:old.get(k) for k in ('status','calls','native','edge_upper_bound')}}
    for tag, flag in [('global',False),('bands',True)]:
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp)/'result.sqlite'
            result = solve(problem,db,split=flag,**kwargs)
            out[tag]={k:v for k,v in result.items() if k not in ('witness','groups')}
            out[tag]['score']=result['witness']['score'] if result['witness'] else None
            if result['status'] != 'open':
                out[tag]['warm_calls']=solve(problem,db,split=flag,**kwargs)['calls']
    return out


def gpu_control(problem):
    with tempfile.TemporaryDirectory() as tmp:
        hot, cold = [Path(tmp)/(n+'.sqlite') for n in ('hot','cold')]
        opts=dict(seed=95, backend='cuda', replicas=512)
        prefix = sample_archive(problem, hot, dispatches=64, **opts)
        replay = sample_archive(problem, hot, dispatches=256, **opts)
        repeat = sample_archive(problem, hot, dispatches=256, **opts)
        oracle = sample_archive(problem, cold, dispatches=256, **opts)
        def state(path):
            with closing(sqlite3.connect(path)) as conn:
                row=conn.execute('SELECT state FROM r31_sampler WHERE launches=256').fetchone()
                return json.loads(row[0])
        a,b=state(hot),state(cold)
        for key in ['native','archive','last_collected','solution','experiment_stats']:
            if a[key] != b[key]:
                raise AssertionError(key)
        if replay['rows'] != oracle['rows'] or repeat['rows'] != oracle['rows']:
            raise AssertionError('Published archive changed on continuation')
        return {'scope':opts,'dispatches':[64,256],'prefix_proposals':prefix['new_proposals'],
                'extension_proposals':replay['new_proposals'],'cold_proposals':oracle['new_proposals'],
                'warm_proposals':repeat['new_proposals'],'reused_dispatches':replay['reused_dispatches'],
                'native_arrays_sha256':a['native']['sha256'],'archive_boards':len(replay['rows'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args()
    p=SearchProblem(read_json(args.tiles))
    model=BoardModel(p)
    sample={tuple(ranked_start(model,i)) for i in range(512)}
    result=dict(fingerprint=p.fingerprint,ranked_initial_boards=len(sample))
    if args.cuda:
        result['cuda']=gpu_control(p)
    incumbent=read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes']
    scopes=[]
    for size in (6,24):
        rows=[[x] for x in incumbent]
        for j in range(size):
            rows[j]=sorted({3*(c//3)+r for c in incumbent[:size] for r in range(3)})
        for target in (207,208):
            scopes.append(dict(free_cells=size,target=target,comparison=compare(p,rows,12,4,target)))
    scopes.append(dict(free_cells=160,target=208,comparison=compare(p,None,30,4,208)))
    result['soft_score_scopes']=scopes
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
