"""Finite same-trajectory scorer work, including cache initialization/maintenance.

No search exclusions, unique-board coverage, timing winner or Gold solve claim.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot
ROOT=Path(__file__).resolve().parent
CHUNKS=(1,7,24,64)


def compare(problem,backend,policy,seed,replicas,gold):
    import numpy as np
    import gpu_sampling as g
    from gpu_incremental_sampling_round26 import canonical_labels,metadata_rows
    a=b=None;started=time.monotonic();checked=0;signatures=[]
    try:
        inputs=g._inputs(problem,gold_aware=gold)
        a=g._make_sampler(problem,*inputs,replicas,seed,backend,experiments=policy,_measure_scoring=True)
        b=g._make_sampler(problem,*inputs,replicas,seed,backend,experiments={**policy,'incremental_sampling':True})
        legacy_rows=metadata_rows(g.sampling_flags(policy),b.gold_total)
        for index,chunk in enumerate((0,*CHUNKS)):
            if chunk:
                a.launch(chunk);b.launch(chunk)
                if policy.get('replica_exchange'):a.exchange(index%2);b.exchange(index%2)
            old=a.snapshot_arrays();new=b.snapshot_arrays()
            np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1]);np.testing.assert_array_equal(old[2][:legacy_rows],new[2][:legacy_rows])
            signatures.append(hashlib.sha256(b''.join(x.tobytes() for x in (*old[:2],old[2][:legacy_rows]))).hexdigest())
            for lane in range(replicas):
                for boards,row in ((new[0],1),(new[1],2)):
                    metrics=g.gold_score(problem,boards[:,lane]) if gold else g.score_board(problem,boards[:,lane]);key='gold_objective' if gold else 'objective'
                    if metrics[key]!=int(new[2][row,lane]):raise ValueError('Independent scorer mismatch.')
                    checked+=1
                if gold:np.testing.assert_array_equal(new[2][45:,lane],canonical_labels(problem,new[0][:,lane]))
            if time.monotonic()-started>60:raise RuntimeError('Bounded finite case exceeded its60second guard.')
        baseline=a.experiment_stats();incremental=b.experiment_stats();proposals=sum(CHUNKS)*replicas
        if baseline['incremental_proposals']!=proposals or incremental['incremental_proposals']!=proposals:raise ValueError('Proposal accounting mismatch.')
        return {'policy':policy,'objective':'gold' if gold else 'edges','seed':seed,'replicas':replicas,
                'lane_proposals':proposals,'baseline':baseline,'incremental':incremental,
                'trajectory_sha256':signatures,'independently_checked_boards':checked,'runtime':b.runtime_identity(),
                'state_limits':b.state_limits}
    finally:
        if a:a.close()
        if b:b.close()


def run(data,backend):
    from search_problem import SearchProblem
    from test_constraint_dfs import tetrahedron_tiles
    from search_fixtures import planted_gold
    fixture=planted_gold(2615,size=4,extra_edges=12)
    cases=[('tetra',SearchProblem(*tetrahedron_tiles()),8),('controlled160',SearchProblem(*fixture[:2]),4),('original160',SearchProblem(json.loads(data.decode('utf-8-sig'))),4)]
    policies=[{'ejection_chains':True},{'ejection_chains':True,'obstruction_chains':True,'move_tabu':True,'replica_exchange':True,'lazy_gold_scoring':True}]
    return [{'fixture':name,'fingerprint':p.fingerprint,'cells':p.n,'scope':'fixed proposal work; no solve-speed or coverage claim',
             'pairs':[compare(p,backend,policy,2615,r,gold) for policy in policies for gold in (False,True)]} for name,p,r in cases]


def main():
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=('cuda','webgpu'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data',type=Path,default=ROOT/'data/tiles.json')
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Fresh .json output required.')
    source=source_snapshot(ROOT);runtime=runtime_snapshot();repo=repository_snapshot(ROOT);data=args.data.read_bytes()
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty')is not False):raise RuntimeError('Frozen evidence requires clean committed source.')
    results=run(data,args.backend);after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repo_after=repository_snapshot(ROOT)
    if source!=after or runtime!=runtime_after or data!=args.data.read_bytes():raise RuntimeError('Finite input/source/runtime changed.')
    if args.evidence_stage=='frozen' and repo!=repo_after:raise RuntimeError('Frozen repository changed.')
    report={'family':'F15','backend':args.backend,'evidence_stage':args.evidence_stage,'evidence_status':'passed','results':results,
            'source_before':source,'source_after':after,'runtime_before':runtime,'runtime_after':runtime_after,'repository_before':repo,'repository_after':repo_after,
            'input_sha256':hashlib.sha256(data).hexdigest(),'chunks':CHUNKS,
            'units':{'incremental_domain_terms':'individual old/new domain lookups; a changed-cell difference costs two',
                     'incremental_seam_terms':'individual old/new seam comparisons; an incident difference costs two',
                     'incremental_strand_steps':'endpoint-walker iterations, including accepted cache clear/fill and initialization',
                     'incremental_cache_reads':'cached proper-component label reads; separate from strand traversal',
                     'incremental_component_starts':'trace roots including partial/aborted traces'},
            'limits':['Same proposals and legacy trajectory; no faster solve claim.','Initial score and cache construction are charged; independent host validation costs are separate.','Candidate generation, metadata accesses, parent selection, and duplicate-label comparisons are not included in scorer counters.','Tiny or high-acceptance neighborhoods can do more strand/cache work; all pairs are retained.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:out.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'family':'F15','backend':args.backend,'pairs':sum(len(r['pairs']) for r in results),'passed':True}))
if __name__=='__main__':main()
