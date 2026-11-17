"""Conditional compatible-partner switches inside physical assembly (#31)."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from validator import reverse_mask
from r27_structure_filters import Work, _vertex_links, domain_bits, strand_graph
from r27_structure_search import _embeddings
from r27_structure_fixtures import planted


def complete_pairing(problem, codes, rng):
    masks=problem.masks[::3].ravel()
    pair=[-1]*len(masks)
    for a,sa,b,sb in problem.edges:
        ta,ra=divmod(codes[a],3)
        tb,rb=divmod(codes[b],3)
        i=3*ta+(sa-ra)%3
        j=3*tb+(sb-rb)%3
        if int(masks[i])==reverse_mask(masks[j]):
            pair[i]=j
            pair[j]=i
    groups=defaultdict(list)
    for i,j in enumerate(pair):
        if j<0:
            groups[int(masks[i])].append(i)
    for m,items in groups.items():
        rev=reverse_mask(m)
        if m>rev:
            continue
        if m==rev:
            if len(items)%2:
                raise ValueError('Odd self-complementary seam class.')
            order=rng.permutation(items)
            left,right=order[::2],order[1::2]
        else:
            left,right=items,rng.permutation(groups[rev])
        if len(left)!=len(right):
            raise ValueError('Incompatible physical seam counts.')
        for i,j in zip(left,right):
            pair[i]=int(j)
            pair[j]=int(i)
    if -1 in pair:
        raise ValueError('Incomplete original seam inventory.')
    return pair


def topology(problem,pair,target):
    seen=set()
    cycles=Counter()
    for start in range(len(pair)):
        if start in seen:
            continue
        node=start
        size=0
        while node not in seen:
            seen.add(node)
            mate=pair[node]
            node=3*(mate//3)+(mate%3+1)%3
            size+=1
        cycles[size]+=1
    graph=strand_graph(problem,[[v] for v in pair])
    seen=set()
    loops=0
    for v in range(len(graph)):
        if v in seen:
            continue
        loops+=1
        nodes=[v]
        seen.add(v)
        while nodes:
            for w in graph[nodes.pop()]:
                if w not in seen:
                    seen.add(w)
                    nodes.append(w)
    return sum((target-cycles).values())+sum((cycles-target).values()),loops


def rewire(problem,domains,codes,seed,proposals,guided=True):
    rng=np.random.default_rng(seed)
    pair=complete_pairing(problem,codes,rng)
    target=Counter(Counter(_vertex_links(problem)).values())
    deficit,loops=topology(problem,pair,target)
    best=(deficit,loops)
    masks=problem.masks[::3].ravel()
    groups=defaultdict(list)
    for i,m in enumerate(masks):
        groups[int(m)].append(i)
    groups=[v for v in groups.values() if len(v)>1]
    work=Work(100000000)
    unique={tuple(pair)}
    embeddings=set()
    validated=set()
    checked=0
    accepted=0
    for _ in range(proposals):
        if not groups:
            break
        group=groups[int(rng.integers(len(groups)))]
        a,c=map(int,rng.choice(group,2,replace=False))
        b,d=pair[a],pair[c]
        if b==c or a//3==d//3 or c//3==b//3:
            continue
        trial=pair.copy()
        trial[a],trial[d],trial[c],trial[b]=d,a,b,c
        k=tuple(trial)
        if k in unique:
            continue
        unique.add(k)
        # Both switched endpoints have the same raw side mask.
        assert int(masks[a])==reverse_mask(masks[d]) and int(masks[c])==reverse_mask(masks[b])
        td,tl=topology(problem,trial,target)
        checked+=1
        best=min(best,(td,tl))
        if td==0:
            for board in _embeddings(problem,domains,trial,work):
                embeddings.add(board)
                report=problem.validate(list(board))
                if report['matched_edges']!=len(problem.edges):
                    raise RuntimeError('Physical embedding seam mismatch.')
                if report['valid']:
                    validated.add(board)
        if not guided or (td,tl)<=(deficit,loops) or rng.random()<.03:
            deficit,loops=td,tl
            pair=trial
            accepted+=1
    return dict(proposals=proposals,checked=checked,accepted=accepted,min_vertex_defect=best[0],
                loops_at_minimum=best[1],embeddings=len(embeddings),validated_gold=len(validated),
                embedding_work=work.used)


def main():
    from search_problem import SearchProblem
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    rows=[]
    problem=SearchProblem(read('data/tiles.json'))
    planted_problem,_=planted(1,10,23,True)
    for name,p,codes in [('original',problem,read('audit/r27/incumbent.json')['codes']),
                         ('planted_multi',planted_problem,list(range(0,30,3)))]:
        for seed in range(3):
            for guided in (False,True):
                record=rewire(p,domain_bits(p),codes,seed,256,guided)
                rows.append(dict(problem=name,seed=seed,guided=guided,**record))
    args.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

