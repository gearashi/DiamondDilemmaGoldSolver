"""Finite regional evidence with complete before/after source/runtime identity."""
import argparse,json
from pathlib import Path
from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions

if __name__=='__main__':
    require_assertions()
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=int,choices=(14,15,16,17,18,19,20,21,22,23,24,25));parser.add_argument('--output')
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    before=source_snapshot(root);runtime_before=runtime_snapshot();repo_before=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo_before.get('available') or repo_before.get('dirty') or not repo_before.get('commit')):
        raise RuntimeError('Frozen evidence requires a readable clean Git commit.')
    from test_constraint_regions_round25 import incremental_comparison,durable_comparison,compact_comparison,gold_compact_comparison,andor_comparison,selective_comparison,joint_comparison,algebraic_comparison,isomorphic_comparison,gold_pricing_comparison,multichain_comparison,certificate_comparison
    runners={14:incremental_comparison,15:durable_comparison,16:compact_comparison,17:gold_compact_comparison,18:andor_comparison,19:selective_comparison,20:joint_comparison,21:algebraic_comparison,22:isomorphic_comparison,23:gold_pricing_comparison,24:multichain_comparison,25:certificate_comparison}
    results={str(number):runners[number]() for number in runners if args.candidate is None or args.candidate==number}
    after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if before!=after or runtime_before!=runtime_after:raise RuntimeError('Source or runtime changed during regional comparison; discard this evidence.')
    if args.evidence_stage=='frozen' and repo_before!=repo_after:raise RuntimeError('Git identity changed during frozen evidence collection.')
    report={**results,'_provenance':{'evidence_stage':args.evidence_stage,'sources_before':before,'sources_after':after,
            'runtime_before':runtime_before,'runtime_after':runtime_after,'repository_before':repo_before,'repository_after':repo_after}}
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({**results,'source_runtime_stable':True,'evidence_stage':args.evidence_stage,'source_files':len(before)},indent=2))
