"""Completed/explicitly capped proof-production comparisons, checked on host."""
import hashlib,json,time,sys
from pathlib import Path
from constraint_proof import BoundedDPLL,ProofLimit
from constraint_proof_indexed_round26 import IndexedRUPChecker
from constraint_isolated_proof_round26 import IsolatedRUPProducer

def pigeonhole(holes):
    def v(p,h):return p*holes+h+1
    rows=[[v(p,h) for h in range(holes)] for p in range(holes+1)]
    rows.extend([-v(p,h),-v(q,h)] for h in range(holes) for p in range(holes+1) for q in range(p+1,holes+1))
    return rows

def check_proof(formula,proof):
    checker=IndexedRUPChecker(formula,proof)
    while not checker.finished:checker.advance(ticks=4096)
    if not checker.result:raise RuntimeError('Independent host RUP replay rejected native proof.')
    return checker.stats

def compare():
    owner=IsolatedRUPProducer();rows=[]
    try:
        if owner.argv is None:raise RuntimeError('Configure the isolated proof interpreter first.')
        for holes in (3,4,5,6):
            formula=pigeonhole(holes);baseline=BoundedDPLL(formula);dpll_status='complete';dpll_check=None
            try:
                if baseline.solve():raise RuntimeError('Pigeonhole formula incorrectly accepted.')
                dpll_check=check_proof(formula,baseline.proof)
            except ProofLimit as exc:dpll_status='unknown: '+str(exc)
            prior=dict(owner.stats);end=time.monotonic()+35;proof=None
            while time.monotonic()<end:
                proof=owner.produce_rup(formula,seconds=.05)
                if proof is not None or owner.failed or owner.stats['ready_hits']>prior['ready_hits']:break
            checked=check_proof(formula,proof) if proof is not None else None
            rows.append({'holes':holes,'pigeons':holes+1,'clauses':len(formula),
              'dpll':{'status':dpll_status,'producer':baseline.stats,'checker':dpll_check},
              'isolated':{'status':'complete' if checked is not None else 'unknown','producer':{k:v-prior[k] for k,v in owner.stats.items()},'proof_lines':len(proof) if proof is not None else None,'checker':checked}})
        return rows
    finally:owner.close()

def main(argv=None):
    import argparse
    from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions
    require_assertions();parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink() or args.output.suffix!='.json':raise ValueError('Fresh JSON output required.')
    root=Path(__file__).resolve().parent;source=source_snapshot(root);runtime=runtime_snapshot();repo=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo.get('commit') or repo.get('dirty')is not False):raise RuntimeError('Frozen evidence requires a clean readable commit.')
    rows=compare()
    if any(row['isolated']['status']!='complete' for row in rows):raise RuntimeError('Isolated finite proof comparison did not finish.')
    after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if source!=after or runtime!=runtime_after or repo!=repo_after:raise RuntimeError('Source/runtime/repository changed during comparison.')
    receipt={'scope':'Finite pigeonhole CNFs; optional stronger proof production, no puzzle/runtime superiority claim','rows':rows,
      'evidence_stage':args.evidence_stage,'family':'F05','provenance':{'sources_before':source,'sources_after':after,'runtime_before':runtime,'runtime_after':runtime_after,'repository_before':repo,'repository_after':repo_after}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:json.dump(receipt,out,indent=2);out.write('\n')
    print(json.dumps([{'holes':r['holes'],'dpll':r['dpll']['status'],'native':r['isolated']['status'],'native_decisions':r['isolated']['producer']['native_decisions']} for r in rows]));return receipt

if __name__=='__main__':main()
