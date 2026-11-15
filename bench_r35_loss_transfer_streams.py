from pathlib import Path
import json
from collections import Counter
from math import inf
from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r35_integer_phases import IntegerPhaseImages, EnumeratedPatternFilter
from r35_loss_transfer import TransferHints
from search_problem import SearchProblem

p=SearchProblem(read_json('../data/tiles.json'))
codes=read_json('docs/R29_INCUMBENT.json')['codes']
rot={v//3:v%3 for v in codes}
centers=[('center'+str(i),[3*t+rot[t] for t in row['pieces']]) for i,row in enumerate(read_json('docs/R29_EXCHANGE_CENTERS.json')['centers'])]
records=[]
for name,placement in centers:
    ref,_,rows=reference_groups(p,placement)
    for transport in (False,True):
        q=OutsideSyndromes(p,ref,rows,inventory(p).required)
        hints=TransferHints(p,q.required,q.outside) if transport else None
        image=IntegerPhaseImages.compile(p,q.groups,q.required,q.outside)
        if image is None:
            raise AssertionError('Noncompact fixture.')
        count=Counter()
        check=EnumeratedPatternFilter(image,q.outside,17,work=count,transfer=hints)
        try:
            out=q.indexed(17,4,180,start=0,stop=None,accept_errors=check,reject_before_joint=check.excludes,verified_cores=lambda:check.verified)
        finally:
            check.close()
        r=dict(name=name,score=len(ref),transfer=transport,status=out['status'],next=out['next_rank'],work=out['work'],filter_work=dict(count))
        records.append(r)
        print(name,transport,r,flush=True)
Path('docs/R35_LOSS_TRANSFER_STREAMS.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')
