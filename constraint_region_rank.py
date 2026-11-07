"""Algebraic cut-incidence representatives for exact Gold existence (#21).

For a perfect matching M, C[M,S]=1 when every edge stays within a cut S.
Anchor one endpoint outside S. The GF(2) inner product of two such rows is
2**(components(M union N)-1) mod2, hence one exactly for one connected cycle.
A row basis therefore preserves existence against every outside matching.
It does not preserve all placements, counts, or individual matching histories.
"""
from collections import OrderedDict
from constraint_region_incremental import source_binding
from constraint_region_network import joined_signature
from constraint_separator import _Budget

STAT_KEYS=('algebraic_connectivity_queries','algebraic_connectivity_completed','algebraic_connectivity_unknown',
 'algebraic_connectivity_basis_hits','algebraic_connectivity_rows','algebraic_connectivity_kept',
 'algebraic_connectivity_cut_entries','algebraic_connectivity_xors','algebraic_connectivity_joins',
 'algebraic_connectivity_failures','algebraic_connectivity_steps')


def cut_representatives(matchings,check=lambda:None,*,max_endpoints=16,max_rows=4096):
    if any(type(bound)is not int or bound<0 for bound in (max_endpoints,max_rows)) or max_endpoints>16:raise ValueError('Invalid exact cut basis bounds.')
    rows=[]
    for matching in matchings:
        check()
        row=[]
        for pair in matching:
            if len(pair)!=2 or any(type(x)is not int or x<0 for x in pair) or pair[0]==pair[1]:raise ValueError('Invalid endpoint pair.')
            row.append(tuple(sorted(pair)))
        rows.append(tuple(sorted(row)))
        if len(rows)>max_rows:raise _Budget
    rows=tuple(dict.fromkeys(rows));stats={'rows':len(rows),'representatives':0,'cut_entries':0,'xors':0,'cut_columns':0}
    if not rows:return (),stats
    vertices=tuple(sorted(x for pair in rows[0] for x in pair))
    if len(vertices)>max_endpoints:raise _Budget
    if len(vertices)!=len(set(vertices)) or any(tuple(sorted(x for pair in row for x in pair))!=vertices for row in rows):
        raise ValueError('Cut representatives require the same distinct endpoint set.')
    if not vertices:stats['representatives']=1;return rows[:1],stats
    offsets={vertex:i for i,vertex in enumerate(vertices[1:])};anchor=vertices[0];basis={};selected=[]
    stats['cut_columns']=1<<len(offsets)
    for row in rows:
        cuts=[0]
        for a,b in row:
            if anchor in (a,b):continue
            mask=(1<<offsets[a])|(1<<offsets[b]);cuts+= [cut|mask for cut in cuts]
        vector=0
        for cut in cuts:check();stats['cut_entries']+=1;vector|=1<<cut
        while vector:
            check();pivot=vector.bit_length()-1
            if pivot in basis:vector^=basis[pivot];stats['xors']+=1
            else:basis[pivot]=vector;selected.append(row);break
    stats['representatives']=len(selected);return tuple(selected),stats


class AlgebraicConnectivity:
    def __init__(self,owner):
        self.owner=owner;self.binding=source_binding(owner);self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)
    def feasible(self,domains,components,tables,check=lambda:None,max_steps=50000):
        if source_binding(self.owner)!=self.binding:raise ValueError('Algebraic connectivity source changed.')
        if self.owner.target!='gold':raise ValueError('Connectivity basis requires Gold target.')
        if type(max_steps)is not int or max_steps<0:raise ValueError('Invalid connectivity work bound.')
        flat=tuple(c for part in components for c in part)
        if len(components)!=len(tables) or len(flat)!=len(set(flat)) or any(type(c)is not int or not 0<=c<self.owner.problem.n for c in flat):
            raise ValueError('Connectivity regions must be disjoint valid cells.')
        self.stats['algebraic_connectivity_queries']+=1;steps=0
        def tick():
            nonlocal steps
            check();steps+=1;self.stats['algebraic_connectivity_steps']+=1
            if steps>max_steps:raise _Budget
        try:
            reduced=[]
            for table in tables:
                groups={}
                for signature in table:
                    tick();used,masks,pairs,cycles=signature
                    endpoints=tuple(sorted(p for pair in pairs for p in pair))
                    groups.setdefault((used,masks,cycles,endpoints),[]).append(signature)
                selected=[]
                for group in groups.values():
                    self.stats['algebraic_connectivity_rows']+=len(group)
                    if len(group)==1 or not group[0][2] or len(group[0][2])>8:selected.extend(group);continue
                    key=tuple(sorted(signature[2] for signature in group))
                    if key in self.cache:
                        keep=self.cache[key];self.stats['algebraic_connectivity_basis_hits']+=1;self.cache.move_to_end(key)
                    else:
                        representatives,work=cut_representatives(key,tick)
                        keep=frozenset(representatives);tick();self.cache[key]=keep
                        while len(self.cache)>32:self.cache.popitem(last=False)
                        self.stats['algebraic_connectivity_cut_entries']+=work['cut_entries'];self.stats['algebraic_connectivity_xors']+=work['xors']
                    selected.extend(signature for signature in group if signature[2] in keep)
                self.stats['algebraic_connectivity_kept']+=len(selected);reduced.append(selected)
            failed=set()
            def walk(index,members,signature):
                tick();self.stats['algebraic_connectivity_joins']+=1
                if index==len(components):return True
                key=(index,signature)
                if key in failed:return False
                for other in reduced[index]:
                    merged=other if index==0 else joined_signature(self.owner,members,signature,components[index],other,domains,tick)
                    if merged is not None and walk(index+1,tuple(sorted((*members,*components[index]))),merged):return True
                failed.add(key);return False
            result=walk(0,(),None);self.stats['algebraic_connectivity_completed']+=1
            if not result:self.stats['algebraic_connectivity_failures']+=1
            return result
        except _Budget:self.stats['algebraic_connectivity_unknown']+=1;return None
