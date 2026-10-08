"""Independent gold endpoint evidence from source pixels, without segment pairing."""
from pathlib import Path
import json
import argparse
import hashlib
from collections import Counter
import numpy as np
from PIL import Image


ROOT=Path(__file__).resolve().parent
GROUPS=(('silver',1,32,'S'),('red',33,48,'R'),('blue',81,80,'B'))

def load_gold(data_dir):
    arrays=[]; ids=[]; hashes={}
    for group,start,count,prefix in GROUPS:
        path=data_dir/f'tiles{group}.gif'
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            im=source.convert('RGB')
        for n in range(count):
            x,y=(n%8)*88,(n//8)*80
            a=np.asarray(im.crop((x,y,x+82,y+71))).astype(float)
            contrast=np.minimum(a[:,:,0],a[:,:,1])-a[:,:,2]
            contrast[np.abs(a[:,:,0]-a[:,:,1])>40]=0
            arrays.append(np.maximum(0,contrast))
            ids.append(f'{prefix}{start+n:02d}')
    return np.asarray(arrays),ids,hashes

def sample(arrays,xy):
    xy=np.asarray(xy)
    x,y=xy[...,0],xy[...,1]
    x0,y0=np.floor(x).astype(int),np.floor(y).astype(int)
    x0=np.clip(x0,0,80);y0=np.clip(y0,0,69)
    fx,fy=x-x0,y-y0
    return (arrays[:,y0,x0]*(1-fx)*(1-fy)+arrays[:,y0,x0+1]*fx*(1-fy)
            +arrays[:,y0+1,x0]*(1-fx)*fy+arrays[:,y0+1,x0+1]*fx*fy)

def anchors(params,ps):
    center,apex,half,bottom=params
    vv=np.array([[center,apex],[center+half,bottom],[center-half,bottom]])
    positions=[];normals=[];tangents=[]
    for side in range(3):
        a,b=vv[side],vv[(side+1)%3]
        d=(b-a)/np.linalg.norm(b-a)
        normal=np.array([-d[1],d[0]])
        for p in ps:
            positions.append(a+(b-a)*p/12)
            normals.append(normal);tangents.append(d)
    return np.array(positions),np.array(normals),np.array(tangents)

def evidence(arrays,params,ps=range(1,12)):
    xy,norm,tan=anchors(params,ps)
    values=[]
    for dn in (-.25,0,.25,.5):
        for dt in (-.35,0,.35):
            values.append(sample(arrays,xy+norm*dn+tan*dt))
    return np.max(values,axis=0)

def reverse11(mask):
    return int(f'{mask:011b}'[::-1],2)

def inventory(masks):
    c=Counter(int(x) for row in masks for x in row)
    seen=set();bad=[];maximum=0
    for mask in sorted(c):
        if mask in seen:continue
        other=reverse11(mask);seen.update((mask,other))
        if mask==other:
            maximum+=c[mask]//2
            if c[mask]%2:bad.append({'mask':mask,'reverse':other,'count':c[mask],'reverse_count':c[other]})
        else:
            maximum+=min(c[mask],c[other])
            if c[mask]!=c[other]:bad.append({'mask':mask,'reverse':other,'count':c[mask],'reverse_count':c[other]})
    return {'edge_signatures':len(c),'maximum_possible_matched_edges':maximum,'violations':bad}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data')
    ap.add_argument('--optimize',action='store_true')
    args=ap.parse_args()
    arrays,ids,hashes=load_gold(args.data_dir)
    params=np.array([40.51531927603321,2.7489464741760465,38.28663273115674,68.50667690173671])
    if args.optimize:
        def objective(p):
            on=evidence(arrays,p)
            off=evidence(arrays,p,np.arange(1.5,11,.5)[::2])
            return -float(np.mean(on)-np.mean(off))
        rng=np.random.default_rng(20261007)
        low=np.array([39.5,1,37.5,67.5]);high=np.array([41,3.5,40,70])
        best=objective(params)
        for stage in range(6):
            scale=(high-low)*(.5**stage)
            for attempt in range(120):
                candidate=np.clip(params+rng.normal(size=4)*scale,low,high)
                score=objective(candidate)
                if score<best:params,best=candidate,score
    values=evidence(arrays,params)
    print('parameters',params.tolist())
    summary=[]
    for threshold in (25,50,65,70,75,80,85,90,95,100,105,110):
        present=values>=threshold
        masks=[[sum((1<<p) for p in range(11) if row[side*11+p]) for side in range(3)] for row in present]
        stats=inventory(masks)
        summary.append({'threshold':threshold,'endpoint_count':int(present.sum()),'odd_tiles':int(np.count_nonzero(present.sum(axis=1)%2)),**stats})
        print('threshold',threshold,'endpoints',int(present.sum()),'oddtiles',int(np.count_nonzero(present.sum(axis=1)%2)),'maxedges',stats['maximum_possible_matched_edges'],'violations',len(stats['violations']))
    out={'source_sha256':hashes,'parameters':params.tolist(),'method':'independent boundary gold contrast; threshold90 separates source-pixel classes; segment connectivity needs separate fit','threshold':90,'threshold_stability_interval':[75,105],'minimum_accepted_contrast':float(values[values>=90].min()),'maximum_rejected_contrast':float(values[values<90].max()),'threshold_sweep':summary,'tiles':[]}
    for i,ident in enumerate(ids):
        out['tiles'].append({'id':ident,'endpoints':[[s,p+1] for s in range(3) for p in range(11) if values[i,s*11+p]>=90],'edge_masks':[sum(1<<p for p in range(11) if values[i,s*11+p]>=90) for s in range(3)],'endpoint_evidence':[[s,p+1,round(float(values[i,s*11+p]),4)] for s in range(3) for p in range(11)]})
    target=args.data_dir/'endpoint_audit.json'
    target.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(target)

if __name__=='__main__':main()
