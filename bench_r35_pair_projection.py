"""Compare blank-count projection and bipartite physical mate flows at fixed loss caps."""
import argparse
import json
from pathlib import Path
from random import Random

from bench_r35_loss_boundary344 import problem_scope
from r35_mip_pairing import decide
from verify_r35_contact_wide import read_rows


def run(tiles,output,modes,seconds,cases):
    root=Path(__file__).resolve().parent
    p,ref,base,req=problem_scope(tiles)
    d=root/'docs'
    groups=[*base[:13],*read_rows(d/'R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores'),
            *read_rows(d/'R35_EDGE_CONTACTS.json','edge_perfect_cores')]
    phases={'a':0xcc57eae4e99109f8b073557e8c590b9202465766,'b':0xcc778d82e99c517987f6b57ec15837c21207cf06,
            'c':0xd237aba7e91101e88041413e8c589902021e5126}
    primary=json.loads((d/'R35_MIXED_MIP_DONOR.json').read_bytes())
    donors=[primary,*primary['others']]
    rows=[]
    for case in cases:
        donor,remaining,cap,warm=case.split('/')
        value=phases[donor]
        order=list(range(1,p.n))
        Random(35590+ord(donor)).shuffle(order)
        unfixed=set(order[:int(remaining)])
        pins={i:(value>>i)&1 for i in range(p.n) if i not in unfixed}
        hint=None
        if warm in ('warm','full'):
            found=next((v for v in donors if int(v['phase'],16)==value and len(set(map(tuple,v['pairs'])))==3*p.n//2),None)
            if found is None:
                raise ValueError('No original pairing donor.')
            hint=value,set(map(tuple,found.get('first_pairing',found['pairs'])))
        for mode in modes:
            out=decide(p,ref,groups,req,int(cap),seconds,sphere=True,mode=mode,pins=pins,hint=hint,full_hint=warm=='full')
            rows.append(dict(case=case,mode=mode,status=out['status'],cost=out.get('loss_cost'),phase=out.get('phase'),work=out['work']))
            output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
            print(case,mode,out['status'],out.get('loss_cost'),out['work'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=45)
    ap.add_argument('--cases',nargs='+',default=['a/0/33/cold','a/0/34/cold','b/48/38/cold'])
    ap.add_argument('--modes',choices=('explicit','blank','blank-neighbors','flow','both'),nargs='+',required=True)
    run(**vars(ap.parse_args()))
