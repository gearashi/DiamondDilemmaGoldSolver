"""Compare rotation graph work with the immutable released R28 oracle in a Git checkout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange_rotations import RotationOracle
from r30_fresh import fresh_start
from search_problem import SearchProblem


PREDECESSOR = 'fb67a56'


def predecessor():
    data = subprocess.check_output(['git','show',PREDECESSOR+':r28_exchange_rotations.py'],cwd=Path(__file__).resolve().parent)
    source = data.decode('utf8').replace('\r\n','\n')
    def at(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Unexpected released oracle at '+old)
        source = source.replace(old, new)
    at('        def star(c):','        def star(c):\n            self.work["threshold_stars"] += 1\n            self.work["threshold_supports"] += len(tables[c])')
    at('        upper = sum(', '        self.work["threshold_stars"] += len(domains)\n        self.work["threshold_supports"] += sum(map(len,local))\n        upper = sum(')
    at('                if key not in lookup:', '                if key not in lookup:\n                    self.work["built_templates"] += 1')
    at('        while active:', '        self.work["plan_builds"] += 1\n        while active:\n            self.work["plan_neighbor_products"] += len(active)+1')
    ns={}
    exec(compile(source,PREDECESSOR+':r28_exchange_rotations.py','exec'),ns)
    return ns['RotationOracle'], hashlib.sha256(data).hexdigest()


def measure(cls, model, cases):
    factors=hashlib.sha256()
    def observe(key,size,product):
        factors.update(repr((key[0],key[1],size,product,len(key[2]))).encode('ascii'))
        for axes,data in key[2]:
            factors.update(repr((axes,len(data))).encode('ascii'))
            factors.update(data)
    oracle=cls(model,observe=observe)
    oracle.work.update(threshold_stars=0,threshold_supports=0,built_templates=0,plan_builds=0,plan_neighbor_products=0)
    outputs=[]
    for board,cap in cases:
        result=oracle.optimum(board,max_entries=cap,ties=True,limit=4)
        if result is None:
            outputs.append(None)
            continue
        score,solutions=result
        witnesses=list(solutions)
        assert witnesses
        for w in witnesses:
            assert int(model.matches(model.legal_board(w)).sum())==score
        outputs.append((score,hashlib.sha256(json.dumps(witnesses).encode()).hexdigest(),len(witnesses)))
    stats=dict(oracle.work,ordered_factor_trace_sha256=factors.hexdigest())
    if hasattr(oracle,'plans'):
        stats.update(built_templates=len(oracle.rotation_supports), retained_plans=len(oracle.plans),
                     retained_plan_bytes=oracle.plan_bytes,plan_budget=oracle.plan_limit)
    return outputs,stats


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--board',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--classic-run',type=Path,help='Optionally verify a real reviewed classic terminal after these dormant edits.')
    args=ap.parse_args()
    model=BoardModel(SearchProblem(json.loads(args.tiles.read_text(encoding='utf-8-sig'))))
    incumbent=model.legal_board(json.loads(args.board.read_text(encoding='utf-8-sig'))['codes'])
    rng=np.random.default_rng(851)
    adjacent=[]
    for count in (1,2,3,4)*3:
        row=incumbent.copy()
        for _ in range(count):
            a,b=rng.choice(model.free,2,replace=False); row[[a,b]]=row[[b,a]]
        adjacent.append(row)
    fresh=[fresh_start(model,rng) for _ in range(6)]
    cases={'incumbent':[(incumbent,59049)],'twelve_independent_exchanges':[(v,59049) for v in adjacent],
           'six_fresh_permutations':[(v,59049) for v in fresh],
           'reused_physical_order_changed_input_rotations':[(3*(incumbent//3)+(incumbent+shift)%3,59049) for shift in (0,1,2)],
           'cap_crossings':[(v,k) for v in (incumbent,fresh[0]) for k in (1,7,81,59049)]}
    previous,sha=predecessor()
    report={'predecessor':PREDECESSOR,'predecessor_file_sha256':sha,'input_sha256':hashlib.sha256(args.tiles.read_bytes()).hexdigest(),
            'incumbent_sha256':hashlib.sha256(json.dumps(incumbent.tolist()).encode()).hexdigest(),'groups':{}}
    for name,rows in cases.items():
        answers,before=measure(previous,model,rows)
        actual,after=measure(RotationOracle,model,rows)
        assert answers==actual,(name,answers,actual)
        assert before['ordered_factor_trace_sha256']==after['ordered_factor_trace_sha256'],name
        report['groups'][name]=dict(queries=len(rows),results=answers,before=before,after=after)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        print(name,Counter(v[0] if v is not None else 'cap' for v in actual),
              before['threshold_stars'],after['threshold_stars'],before['plan_neighbor_products'],after['plan_neighbor_products'],flush=True)

    if args.classic_run:
        import systematic_evidence as proof
        from bench_r35_classic_history import restore
        archive=args.classic_run.resolve()
        marker=proof.read(archive/'systematic/exhaustion.json')
        with tempfile.TemporaryDirectory(dir=args.output.resolve().parent) as tmp:
            base=Path(tmp)
            candidate=base/'candidate'
            dst=base/'checked'
            candidate.mkdir()
            dst.mkdir()
            old=marker['source_sha256']
            origin=proof.run_dir(archive,marker['run_id'])
            for key in old:
                shutil.copyfile(origin/key,candidate/key)
            for key in ('r28_exchange_rotations.py','r35_rotation_graph.py'):
                shutil.copyfile(Path(__file__).resolve().parent/key,candidate/key)
            # Transfer only the real rotation changes; unrelated live edits
            # to a search kernel remain ineligible for this control.
            hashes=proof.capture(candidate,dst,marker['input_sha256'])
            reopened=restore(archive,hashes,dst)
            differences={k:[old.get(k),hashes.get(k)] for k in sorted(hashes.keys()|old.keys())
                         if old.get(k)!=hashes.get(k)}
            assert 'r28_exchange_rotations.py' in differences and 'r35_rotation_graph.py' in differences
            report['classic_reuse']=dict(source_run=marker['run_id'],history=marker['proof_backends'],
                previous_files=len(old)-1,current_files=len(hashes)-1,changed_diagnostics=differences,
                reopened_on_host=reopened['terminal_reopened_on_host'],counters=reopened['counters'])
            args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
