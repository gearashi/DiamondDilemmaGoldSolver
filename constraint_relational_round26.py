"""Exact semijoins and indexed supports for overlapping complete SAT relations.

Only completed scoped tables may enter. A relation is applicable only when the
whole current domain box lies inside its proving scope. Narrowing may reuse live
row sets; widening restores original rows. Tuple exclusions are never promoted
outside this joint scope, and an unfinished pass publishes no domain change.
"""
from collections import OrderedDict,defaultdict
from itertools import combinations

STAT_KEYS=('relation_index_build_rows','relation_index_slots','relation_dag_nodes',
 'relation_join_queries','relation_join_pairs','relation_join_keys','relation_tuple_prunes',
 'relation_domain_prunes','relation_incremental_reuses','relation_guided_selections','relation_dag_projection_nodes','relation_missing_support_guidance','relation_spurious_guidance')


class TupleIndex:
    def __init__(self,cells,rows):
        self.cells=tuple(cells);self.rows=tuple(sorted(rows));self.dag_projection_nodes=0
        if (len(set(self.cells))!=len(self.cells) or any(type(c)is not int or c<0 for c in self.cells)
                or any(type(row)is not tuple or len(row)!=len(cells) or any(type(v)is not int or v<0 for v in row) for row in self.rows)
                or len(set(self.rows))!=len(self.rows)):raise ValueError('Invalid complete relation shape.')
        self.full=(1<<len(rows))-1;self.literal=defaultdict(int);self.projections={}
        for index,row in enumerate(rows):
            for cell,code in zip(cells,row):self.literal[cell,code]|=1<<index
        # Hash-consed suffix DAG compresses repeated complete tuple suffixes.
        # The row bitsets remain the support index; no count/tuple ambiguity.
        intern={};self.nodes=[]
        def build(level,selected):
            if level==len(cells):return -1
            groups=defaultdict(list)
            for row in selected:groups[row[level]].append(row)
            key=tuple((v,build(level+1,part)) for v,part in sorted(groups.items()))
            if key not in intern:intern[key]=len(self.nodes);self.nodes.append(key)
            return intern[key]
        self.root=build(0,self.rows) if self.rows else None
        self.counts={-1:1}
        for index,node in enumerate(self.nodes):self.counts[index]=sum(self.counts[child] for _,child in node)

    def projection(self,cells):
        cells=tuple(cells)
        if cells not in self.projections:
            if len(set(cells))!=len(cells) or any(c not in self.cells for c in cells):raise ValueError('Projection cells leave complete table.')
            wanted=set(cells);memo={};natural=tuple(c for c in self.cells if c in wanted)
            def visit(node,level):
                if node==-1:return {():1}
                key=(node,level)
                if key in memo:return memo[key]
                self.dag_projection_nodes+=1;result=defaultdict(int);offset=0
                for value,child in self.nodes[node]:
                    for suffix,rows in visit(child,level+1).items():
                        projected=(value,*suffix) if self.cells[level] in wanted else suffix
                        result[projected]|=rows<<offset
                    offset+=self.counts[child]
                memo[key]=dict(result);return memo[key]
            raw={} if self.root is None else visit(self.root,0);order=tuple(natural.index(c) for c in cells)
            self.projections[cells]={tuple(row[i] for i in order):mask for row,mask in raw.items()}
        return self.projections[cells]

    def restrict(self,domains,active):
        for cell in self.cells:
            permitted=0
            for (c,code),rows in self.literal.items():
                if c==cell and domains[cell]>>code&1:permitted|=rows
            active&=permitted
        return active


class SemijoinNetwork:
    def __init__(self,*,limit=32,contexts=4):
        if type(limit)is not int or limit<1 or type(contexts)is not int or contexts<1:raise ValueError('Invalid relational reuse bound.')
        self.limit=limit;self.contexts=contexts;self.pressure={};self.indexes=OrderedDict();self.snapshots=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)

    def filter(self,tables,domains,check=lambda:None):
        before=tuple(domains);applicable=[]
        for (scope,cells),rows in tuple(tables.items()):
            check()
            if len(scope)!=len(before) or any(now&~old for now,old in zip(before,scope)):continue
            key=(tuple(scope),tuple(cells),tuple(rows))
            if key not in self.indexes:
                index=TupleIndex(cells,rows);self.indexes[key]=index
                self.stats['relation_index_build_rows']+=len(rows);self.stats['relation_index_slots']+=len(index.literal)
                self.stats['relation_dag_nodes']+=len(index.nodes)
            self.indexes.move_to_end(key);applicable.append((key,self.indexes[key]))
        while len(self.indexes)>self.limit:self.indexes.popitem(last=False)
        signature=tuple(key for key,index in applicable);self.stats['relation_join_queries']+=1
        live=[index.full for _,index in applicable];prior=self.snapshots.get(signature)
        if prior is not None and all(not now&~old for now,old in zip(before,prior[0])):
            live=list(prior[1]);self.stats['relation_incremental_reuses']+=1
        for i,(_,index) in enumerate(applicable):live[i]=index.restrict(before,live[i])
        overlaps=[(a,b,tuple(sorted(set(left.cells)&set(right.cells))))
                  for (a,(_,left)),(b,(_,right)) in combinations(enumerate(applicable),2)]
        overlaps=[row for row in overlaps if row[2]]
        while True:
            old=tuple(live)
            for a,b,common in overlaps:
                check();self.stats['relation_join_pairs']+=1
                left_index=applicable[a][1];right_index=applicable[b][1];prior_work=left_index.dag_projection_nodes+right_index.dag_projection_nodes
                left=left_index.projection(common);right=right_index.projection(common)
                self.stats['relation_dag_projection_nodes']+=left_index.dag_projection_nodes+right_index.dag_projection_nodes-prior_work
                allowed_a=allowed_b=0
                for key,rows in left.items():
                    check();self.stats['relation_join_keys']+=1;other=right.get(key,0)
                    if rows&live[a] and other&live[b]:allowed_a|=rows;allowed_b|=other
                live[a]&=allowed_a;live[b]&=allowed_b
            if tuple(live)==old:break
        current=list(before)
        for active,(_,index) in zip(live,applicable):
            for cell in index.cells:
                support=0
                for (c,code),rows in index.literal.items():
                    if c==cell and active&rows:support|=1<<code
                current[cell]&=support
        check()
        self.stats['relation_tuple_prunes']+=sum((index.restrict(before,index.full)&~active).bit_count() for active,(_,index) in zip(live,applicable))
        self.stats['relation_domain_prunes']+=sum((a&~b).bit_count() for a,b in zip(before,current))
        for active,(_,index) in zip(live,applicable):
            missing=(index.restrict(before,index.full)&~active).bit_count()
            if missing:
                for cell in index.cells:self.pressure[cell]=min(4096,self.pressure.get(cell,0)+missing)
        self.snapshots[signature]=(before,tuple(live));self.snapshots.move_to_end(signature)
        while len(self.snapshots)>self.contexts:self.snapshots.popitem(last=False)
        changed=current!=list(domains);domains[:]=current;return changed


def guided_cells(domains,helper,arity,max_values,stats=None,network=None):
    eligible={c for c,m in enumerate(domains) if 1<m.bit_count()<=max_values};scores=defaultdict(int)
    # Retained independently proved mask cores identify correlated bottlenecks.
    for core in tuple(getattr(helper,'domain_cores',{}))[-64:]:
        for cell,mask in core:
            if cell in eligible and domains[cell]&mask:scores[cell]+=1
    # Missing semantic supports and impossible joined tuples are selection
    # hints only. Neither becomes a new exclusion without a complete query.
    for source,session in tuple(getattr(helper,'domain_sessions',{}).items())[-16:]:
        if len(source)!=len(domains) or any(now&~old for now,old in zip(domains,source)):continue
        supported=session.get('supported',())
        if len(supported)!=len(domains):continue
        for cell in eligible:
            missing=(domains[cell]&~supported[cell]).bit_count()
            if missing:
                scores[cell]+=missing
                if stats is not None:stats['relation_missing_support_guidance']=stats.get('relation_missing_support_guidance',0)+missing
    if network is not None:
        for cell in eligible:
            pressure=network.pressure.get(cell,0)
            if pressure:
                scores[cell]+=pressure
                if stats is not None:stats['relation_spurious_guidance']=stats.get('relation_spurious_guidance',0)+pressure
    selected=tuple(sorted(eligible,key=lambda c:(-scores[c],domains[c].bit_count(),c))[:arity])
    if scores and stats is not None:stats['relation_guided_selections']=stats.get('relation_guided_selections',0)+1
    return selected
