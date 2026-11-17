"""Check held-out score ordering and exactly bound Gold-only pruning in nine-tile restrictions."""
from collections import Counter
import argparse,json
from pathlib import Path

import numpy as np
from bench_r35_incidental_joins import grow
from constraint_pruning import Pruner
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r30_fresh import fresh_start
from r30_gold import search
from r31_fresh import Assemblies,physical_bonds
from r35_pose_contacts import Boundaries,PoseIndex
from r35_speculative_poses import PoseJoins
from r32_fresh import _poses
from validator import reverse_mask
from search_problem import SearchProblem


def partner_opportunities(geometry,raw,groups):
    arrays=[([geometry.describe(x) for x in rows],[raw.describe(x) for x in rows]) for _,rows in groups]
    stronger=0;first=None;total=0
    for j,(_,family) in enumerate(groups):
        a,b=arrays[j];strong=PoseIndex(a,geometry);ordinary=PoseIndex(b,raw);full=(1<<len(a))-1
        for i in range(j):
            for k,signature in enumerate(arrays[i][1]):
                plain=ordinary.allowed(signature,full);extra=plain&~strong.allowed(signature,full)
                total+=len(a);stronger+=extra.bit_count()
                if extra and first is None:first=[groups[i][1][k],family[(extra&-extra).bit_length()-1]]
    return dict(pairings=total,physically_banned_but_edge_matching=stronger,first=first)


class SoftOrder:
    def __init__(self,seed,penalties):
        self.base=np.random.default_rng(seed);self.rows=iter(penalties);self.before=True
    def __getattr__(self,key):return getattr(self.base,key)
    def permutation(self,arg):
        order=self.base.permutation(arg)
        if self.before:
            try:price=next(self.rows)
            except StopIteration:self.before=False
            else:
                assert len(price)==arg
                return sorted(order,key=lambda k:price[k])
        return order


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True);cli.add_argument('--count',type=int,default=64)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));model=BoardModel(p);a=Assemblies(model)
    geo=Boundaries(p);raw=Boundaries(p,False)
    result=dict(original_pairs=partner_opportunities(geo,raw,a.groups),soft=[],public=[])
    for seed in range(8500,8500+args.count):
        candidate=grow(p,model,a,seed)
        if candidate is None:continue
        edges,group=candidate
        rebuilt=_poses(model,group[0],a.bonds|set(edges))
        assert {tuple(sorted(row)) for row in group[1]}=={tuple(sorted(row)) for row in rebuilt}
        if seed in (8504,8511):
            for mode in (False,True):
                check=search(p,bonds=edges,geography=True,joint_poses=mode,phase_anchors='off',seconds=1,max_rounds=1)
                result['public'].append(dict(seed=seed,joint=mode,**{k:check.get(k) for k in ('status','cp_calls','branches','joint_pose_support')}))
        groups=PoseJoins(a).replay(edges)
        trial=object.__new__(Assemblies);trial.model=model;trial.bonds=a.bonds|set(edges);trial.forbidden=a.forbidden
        trial.groups=sorted([(names,rows) for names,rows,_ in groups],key=lambda item:len(item[1]))
        trial.tiles=set.union(set(),*(set(names) for names,_ in trial.groups))
        prices=[];matched=[]
        for names,rows in trial.groups:
            bad=[];hits=[]
            for row in rows:
                codes=dict(row)
                if len(names)>2:
                    checks=[int(p.masks[codes[u],s])==reverse_mask(int(p.masks[codes[v],t]))
                            for u,s,v,t in p.edges if u in codes and v in codes]
                    mismatches=checks.count(False)
                    assert mismatches==raw.describe(row)[3]
                    bad.append(mismatches);hits.append(-sum(checks))
                else:bad.append(0);hits.append(0)
            prices.append(bad);matched.append(hits)
        full=list(trial.groups)
        hint=fresh_start(model,np.random.default_rng(seed));record=dict(seed=seed)
        for name,rng in [('plain',np.random.default_rng(seed)),('soft',SoftOrder(seed,prices)),('first_min_loss',np.random.default_rng(seed)),('first_max_score',np.random.default_rng(seed))]:
            stats=Counter();trial.groups=full
            if name.startswith('first_'):
                tiers=prices if name=='first_min_loss' else matched
                trial.groups=[]
                for (members,rows),tier in zip(full,tiers):
                    least=min(tier)
                    trial.groups.append((members,[pose for pose,amount in zip(rows,tier) if amount==least]))
            try:
                try:codes=trial.seed(hint,rng,work=5000,stats=stats)
                except ValueError:
                    if not name.startswith('first_'):raise
                    stats['fallback_to_full']+=1;trial.groups=full
                    codes=trial.seed(hint,rng,work=5000,stats=stats)
            except ValueError as e:
                if str(e)!='No legal mandatory-bond start found within the pose budget.':raise
                record[name]=dict(work=stats,status='unknown');continue
            joins=physical_bonds(model,codes);assert trial.bonds<=joins and not joins&trial.forbidden
            record[name]=dict(work=stats,status='completed',score=int(model.matches(codes).sum()))
        result['soft'].append(record)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('total cases',len(result['soft']),result['original_pairs'],flush=True)

if __name__=='__main__':main()
