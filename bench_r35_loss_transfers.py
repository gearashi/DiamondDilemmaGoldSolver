import json,sqlite3
from pathlib import Path
from collections import Counter
from r32_gold_bonds import inventory
from r32_gold_loss import read_json,reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem
p=SearchProblem(read_json(Path('../data/tiles.json')))
ref,cert,groups=reference_groups(p,read_json('docs/R29_INCUMBENT.json')['codes'])
q=OutsideSyndromes(p,ref,groups,inventory(p).required)

from r35_integer_phases import IntegerPhaseImages
from r33_gold_refine import _published_core
from r27_incumbent import BoardModel
engine=IntegerPhaseImages.compile(p,q.groups,q.required,q.outside)
hints=[]
for row in read_json('docs/R34_INTEGER_CORES_FINAL.json')['arrivals']:
 val=sum(1<<i for i in row)
 _,_,mask=engine.minimum(row)
 hints.append((mask,val&mask))
hints.append(_published_core(p,q.outside))
unique=sorted(set(hints))
codes=read_json('docs/R29_INCUMBENT.json')['codes']
rot={c//3:c%3 for c in codes}
d=read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']
boards=[('center'+str(i),[3*t+rot[t] for t in c['pieces']]) for i,c in enumerate(d)]
boards.extend(('independent'+str(i),c['codes']) for i,c in enumerate(read_json('docs/R32_GOLD_FOLLOWUPS.json')['cross_board']))

from r35_integer_sat import InventorySAT
from time import perf_counter
all_rows=[]
for name,x in boards:
    rr,_,gg=reference_groups(p,x)
    ps,r,oo=q.layout(rr,gg,inventory(p).required)
    target=set(oo)
    phase=InventorySAT(p,[],r,oo,len(oo))
    literals={bond:v for bond,v in zip(oo,phase.errors)}
    rows=[]
    try:
        for mask,val in unique:
            source={q.outside[i]: bool(val>>i&1) for i in range(len(q.outside)) if mask>>i&1}
            transfer=[(literals[b] if sign else -literals[b]) for b,sign in source.items() if b in target]
            before=phase.solver.accum_stats(); t=perf_counter()
            ans=phase.decide(1.25,transfer)
            elapsed=perf_counter()-t
            core=phase.solver.get_core() if ans is False else None
            work={k:v-before[k] for k,v in phase.solver.accum_stats().items()}
            rows.append(dict(source=len(source),available=len(transfer),status=str(ans),seconds=elapsed,
                            certified=None if core is None else len(core),work=work))
    finally:
        phase.close()
    all_rows.append(dict(name=name,score=len(rr),outside=len(oo),overlap=len(set(q.outside)&target),rows=rows))
    print(name,[(r['status'],r['certified']) for r in rows],flush=True)
Path('docs/R35_LOSS_TRANSFERS.json').write_text(json.dumps(all_rows,indent=2)+'\n',encoding='utf8')
