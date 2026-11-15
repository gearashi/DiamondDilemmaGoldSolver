"""Certify all learned departures using one unrestricted all-thirteen counterexample query."""
import argparse
import hashlib
import json
from pathlib import Path
from bench_r35_loss_tight import original,problem_scope,InventorySAT
from bench_r35_loss_proof import independent_obligations
from r32_gold_loss import read_json


def export(tiles, directory):
    p,ref,groups,required=problem_scope(tiles);_,_,packed,_,out=original(tiles)
    original_groups=groups[:13];learned=groups[13:]
    seam_check=independent_obligations(p,ref,original_groups,required,read_json('docs/R29_INCUMBENT.json')['codes'])
    assert len(learned)==331 and all(g<=ref and g-required for g in learned)
    model=InventorySAT(p,packed,required,out,len(ref),native=None,additional_groups=original_groups)
    clauses=list(model.clauses)
    top=max(abs(v) for c in clauses for v in c)
    triggers=list(range(top+1,top+1+len(learned)))
    clauses.append(triggers)
    instructions=[]
    for trigger,group in zip(triggers,learned):
        retained=[model.physical_losses[e] for e in sorted(group-required)]
        clauses.extend([[-trigger,-v] for v in retained])
        instructions.append(dict(selector=trigger,group=sorted(group),loss_literals=retained))
    filename=directory/'learned-groups.cnf'
    directory.mkdir(parents=True,exist_ok=True)
    with filename.open('w',encoding='ascii',newline='\n') as f:
        f.write(f'p cnf {triggers[-1]} {len(clauses)}\n')
        for c in clauses:f.write(' '.join(map(str,c))+' 0\n')
    report=dict(reference=len(ref),base_groups=len(original_groups),learned_groups=len(learned),cap=len(ref),variables=triggers[-1],
                clauses=len(clauses),cnf_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                independently_rebuilt_obligations=seam_check,selectors=instructions)
    (directory/'learned-groups.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',required=True);cli.add_argument('--directory',type=Path,required=True)
    args=cli.parse_args();r=export(args.tiles,args.directory);print(r['learned_groups'],r['cnf_sha256'])
