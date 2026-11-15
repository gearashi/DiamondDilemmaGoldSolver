"""Exchange whole alternating cycles between complete physical sewings."""
import argparse
from collections import Counter
import json
from pathlib import Path

from r27_structure_filters import domain_bits
from r28_geo_color import signed_phase_model
from r28_geo_sewing import PosePacking
from r29_gold_phase import EndpointCuts
from r30_gold import embedding
from r31_geographic import ribbon_genus
from r35_blank_projection import BlankProjection,lift
from validator import reverse_mask


def complete(problem,pairs,phases,inputs):
    if len(phases)!=problem.n or any(v not in (0,1) for v in phases):
        raise ValueError('Invalid tile phase vector.')
    result={}
    for a,b in pairs:
        if (a in result or b in result or not 0<=a<3*problem.n or not 0<=b<3*problem.n or a//3==b//3):
            raise ValueError('Expected disjoint physical seams between different tiles.')
        m,c=inputs.sides[a];r,d=inputs.sides[b]
        if reverse_mask(m)!=r or (c^d^phases[a//3]^phases[b//3]^(m.bit_count()&1)):
            raise ValueError('Parent violates signed physical compatibility.')
        result[a],result[b]=b,a
    if len(result)!=3*problem.n:raise ValueError('Incomplete parent.')
    return result


def common_phases(problem,first,second,seconds=5):
    from r35_mixed_sewings import PhaseComponents
    inputs=EndpointCuts(problem)
    a=complete(problem,first['pairs'],first['phases'],inputs)
    b=complete(problem,second['pairs'],second['phases'],inputs)
    pins=(dict(enumerate(first['phases'])) if first['phases']==second['phases'] else None)
    seams={(u,v) for mate in (a,b) for u,v in mate.items() if u<v}
    parity=PhaseComponents(problem.n)
    if not parity.push((u//3,v//3,inputs.sides[u][1]^inputs.sides[v][1]^
                        (inputs.sides[u][0].bit_count()&1)) for u,v in seams):
        return None,'NO_COMMON_PHASE'
    try:
        from ortools.sat.python import cp_model
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split('.')[0]=='ortools':
            return None,'OPTIONAL_CP_UNAVAILABLE'
        raise
    m,variables=signed_phase_model(problem,pins=pins,bonds=seams)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=1
    status=solver.solve(m)
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        return [solver.value(t) for t in variables],'COMPATIBLE'
    return None,'NO_COMMON_PHASE' if status==cp_model.INFEASIBLE else 'UNKNOWN'


def difference(first,second):
    fixed={(i,v) for i,v in first.items() if first[i]==second[i] and i<v}
    unseen={i for i in first if first[i]!=second[i]}
    groups=[]
    while unseen:
        start=min(unseen);v=start
        sides=[[],[]]
        while v in unseen:
            for group,mate in zip(sides,(first,second)):
                u=mate[v]
                unseen.remove(v)
                group.append(tuple(sorted((v,u))))
                v=u
        if v!=start or len(sides[0])<2:raise ValueError('Malformed alternating cycle.')
        groups.append(tuple(tuple(sorted(half)) for half in sides))
    return fixed,groups


class RollbackPaths:
    def __init__(self,n):
        self.parent=list(range(n));self.size=[1]*n;self.log=[]
    def find(self,v):
        while self.parent[v]!=v:v=self.parent[v]
        return v
    def push(self,links):
        for a,b in links:
            a,b=self.find(a),self.find(b)
            if a==b:
                if self.size[a]!=len(self.parent):return False
                continue
            if self.size[a]<self.size[b]:a,b=b,a
            self.parent[b]=a;self.size[a]+=self.size[b]
            self.log.append((a,b))
        return True
    def restore(self,mark):
        while len(self.log)>mark:
            a,b=self.log.pop();self.parent[b]=b;self.size[a]-=self.size[b]
    def frontier(self,nodes):
        labels={}
        return tuple(labels.setdefault(self.find(v),i) for i,v in enumerate(nodes))


class Family:
    def __init__(self,problem,first,second,phases):
        self.problem=problem
        self.inputs=EndpointCuts(problem)
        self.phases=phases
        a,b=(complete(problem,record['pairs'],record['phases'] if phases is None else phases,self.inputs)
             for record in (first,second))
        self.fixed,groups=difference(a,b)
        self.blanks=[g for g in groups if not self.inputs.sides[g[0][0][0]][0]]
        self.groups=sorted((g for g in groups if self.inputs.sides[g[0][0][0]][0]),
                           key=lambda g:(-len(g[0]),g))
        ends={};n=0
        for tile,record in enumerate(problem.tiles):
            for seg in record['segments']:
                for side,k in seg:ends[3*tile+side,k]=n
                n+=1
        self.n=n
        def links(pairs):
            return tuple((ends[a,k],ends[b,12-k]) for a,b in pairs
                         for k in range(1,12) if (a,k) in ends)
        self.base=links(self.fixed)
        self.options=[tuple(links(half) for half in group) for group in self.groups]
        self.frontiers=[]
        remaining=set()
        for options in reversed(self.options):
            remaining.update(v for edge in options[0] for v in edge)
            self.frontiers.append(tuple(sorted(remaining)))
        self.frontiers.reverse()

    def pairs(self,mask,blank_mask=0):
        selected=set(self.fixed)
        for groups,choice in ((self.groups,mask),(self.blanks,blank_mask)):
            for i,group in enumerate(groups):selected.update(group[(choice>>i)&1])
        return selected

    def solve(self,*,max_nodes=1000000,memo=True,geography=False,domains=None,count_after_exclusion=False):
        if self.phases is None:
            from r35_mixed_sewings import mixed_solve
            return mixed_solve(self,max_nodes=max_nodes,memo=memo,geography=geography,domains=domains)
        domains=domain_bits(self.problem) if domains is None else domains
        stats=Counter()
        paths=RollbackPaths(self.n)
        store={}
        cache_words=0
        examples=[]
        gold=None
        full=True
        raw=(1<<len(self.groups))*(1<<len(self.blanks))
        base_allowed=self.fixed.union(*(set(h) for g in self.groups+self.blanks for h in g))
        geo=geography
        certificate=None
        if geography:
            prior=PosePacking(self.problem,domains).check(self.fixed,max_steps=10000)
            if prior['feasible'] is False:
                stats['common_geographic_exclusion']+=1
                geo=False
                core=set(self.fixed)
                genus=ribbon_genus(self.problem,core)
                if genus:
                    for pair in sorted(core):
                        if ribbon_genus(self.problem,core-{pair}):
                            core.remove(pair)
                    certificate=dict(kind='positive_genus',pairs=sorted(core),genus=ribbon_genus(self.problem,core))
                else:
                    certificate=dict(kind='impossible_common_poses',pairs=sorted(core))
            else:
                from r35_mixed_sewings import geographic_roots
                roots=geographic_roots(self.problem,domains,base_allowed)
                if not roots['survivors']:
                    stats['allowed_mate_exclusions']+=1
                    geo=False
                    certificate=dict(kind='no_allowed_geographic_root',**roots)
            if certificate is not None and not count_after_exclusion:
                return dict(complete=True,geographic_status='INFEASIBLE',gold=None,
                            possible_full_sewings=raw,connected_full=None,geographic_certificate=certificate,
                            work=dict(stats))
        # A connectivity-only key cannot carry named geographic exclusions or witnesses.
        use_memo=memo and not geo
        blank=BlankProjection(self.inputs.sides,base_allowed,self.fixed)
        signed=blank.signed(self.phases)
        has_empty=any(not t['segments'] for t in self.problem.tiles)
        def visit(i,mask):
            nonlocal full,gold,cache_words
            if stats['nodes']>=max_nodes:
                full=False;return 0
            stats['nodes']+=1
            if i==len(self.groups):
                if len(examples)<4:examples.append(mask)
                if geo:
                    pairs=self.pairs(mask)
                    if has_empty:
                        for z in range(1<<len(self.blanks)):
                            if stats['nodes']>=max_nodes:
                                full=False;break
                            stats['nodes']+=1
                            mate=dict(self.pairs(mask,z));mate.update({b:a for a,b in list(mate.items())})
                            gold=embedding(self.problem,domains,[mate[d] for d in range(3*self.problem.n)])
                            if gold is not None:break
                    else:
                        nonblank={e for e in pairs if self.inputs.sides[e[0]][0]}
                        gold=lift(self.problem,domains,nonblank,blank,signed,float('inf'),stats)
                return 1
            if use_memo:
                key=i,paths.frontier(self.frontiers[i])
                if key in store:
                    stats['cache_hits']+=1
                    return store[key]
            count=0
            for j,row in enumerate(self.options[i]):
                mark=len(paths.log)
                if paths.push(row):
                    count+=visit(i+1,mask|(j<<i))
                else:stats['proper_partial_loops']+=1
                paths.restore(mark)
                if gold is not None or not full:break
            if use_memo and full and cache_words+len(key[1])<=2000000:
                store[key]=count;cache_words+=len(key[1])+1
            return count
        count=visit(0,0) if self.n and paths.push(self.base) else 0
        if gold is not None and not self.problem.validate(gold)['valid']:
            raise ArithmeticError('Invalid geographic offspring.')
        return dict(complete=full and gold is None,possible_full_sewings=raw,
                    nonblank_cycles=len(self.groups),blank_cycles=len(self.blanks),
                    exchanged_darts=[2*len(g[0]) for g in self.groups],connected_nonblank=count,
                    connected_full=count*(1<<len(self.blanks)),example_masks=examples,gold=gold,
                    geography_requested=geography,geographic_certificate=certificate,
                    geographic_status=('GOLD' if gold is not None else 'INFEASIBLE' if (full or certificate) else 'UNKNOWN')
                    if geography else None,work=dict(stats),memo_entries=len(store))


def main():
    from r28_exchange import read_json
    from search_problem import SearchProblem
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--parents',type=Path,required=True,help='JSON array of complete donors with phases and pairs.')
    cli.add_argument('--max-nodes',type=int,default=1000000)
    cli.add_argument('--geography',action='store_true')
    cli.add_argument('--union-roots',action='store_true',help='Check geographic roots for every matching using only the supplied donor seams.')
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    parents=read_json(args.parents)
    if args.union_roots:
        from r35_mixed_sewings import donor_geography
        print(json.dumps(donor_geography(p,parents),indent=2))
        return
    if len(parents)!=2:cli.error('Provide two donors for a crossover search or use --union-roots.')
    first,second=parents
    phases,state=common_phases(p,first,second)
    result=dict(phase_status=state)
    result.update(Family(p,first,second,phases).solve(max_nodes=args.max_nodes,geography=args.geography))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
