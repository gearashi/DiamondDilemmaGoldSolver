"""Correlated symbolic inventory and Gold-interface joins (round25 #20).

Each inventory family belongs to one exact seam/strand interface. Compatible
interfaces are joined once; ZDD disjoint products and support queries retain the
inventory correlation without enumerating every cross-region inventory pair.
The current app scope is a bounded two-region partition of the complete board.
"""
from collections import OrderedDict
from constraint_region_incremental import source_binding
from constraint_region_symbolic import DirectRegionSession,DirectRegionDiagram,InventoryZDD
from constraint_region_network import joined_signature
from constraint_separator import _Budget

STAT_KEYS=('joint_region_queries','joint_region_completed','joint_region_unknown','joint_region_admission_skips',
 'joint_region_cache_hits','joint_region_states','joint_region_transitions','joint_region_projection_entries',
 'joint_region_interfaces','joint_region_interface_joins','joint_region_inventory_pairs','joint_region_zdd_nodes',
 'joint_region_zdd_operations','joint_region_support_probes','joint_region_work_ticks','joint_region_prunes')

class JointLimit(Exception):pass

class _Work:
    def __init__(self,limit,check,stats):self.limit,self.check,self.stats=limit,check,stats;self.steps=0
    def tick(self):
        self.check();self.steps+=1;self.stats['joint_region_work_ticks']+=1
        if self.steps>self.limit:raise JointLimit


def complete_relation(owner,cells,domains,work,max_states,max_entries,stats):
    session=DirectRegionSession(owner,cells,domains,max_states,work.limit)
    try:
        while not session.advance(128,work.tick):pass
    finally:
        stats['joint_region_states']+=session.states;stats['joint_region_transitions']+=session.steps
    diagram=object.__new__(DirectRegionDiagram);diagram.__dict__.update(session.result());memo={};entries=0
    def visit(node):
        nonlocal entries
        work.tick()
        if node in memo:return memo[node]
        if node in diagram.terminals:return {diagram.terminals[node]:(0,)*len(cells)}
        level,edges=diagram.nodes[node];result={}
        for code,child in edges:
            for signature,support in visit(child).items():
                work.tick();candidate=list(support);candidate[level]|=1<<code
                if signature in result:result[signature]=tuple(a|b for a,b in zip(result[signature],candidate))
                else:
                    entries+=1;stats['joint_region_projection_entries']+=1
                    if entries>max_entries:raise JointLimit
                    result[signature]=tuple(candidate)
        memo[node]=result;return result
    return {} if diagram.root is None else visit(diagram.root)


def joint_supports(owner,regions,tables,domains,work,stats,*,max_nodes=50000,symbolic=True):
    if len(regions)!=2 or len(tables)!=2 or set(regions[0])&set(regions[1]) or set(regions[0])|set(regions[1])!=set(range(owner.problem.n)):
        raise ValueError('Joint symbolic supports require two disjoint regions covering all cells.')
    supports=[0]*owner.problem.n
    def retain(index,signature):
        for cell,mask in zip(regions[index],tables[index][signature]):supports[cell]|=mask
    if not symbolic:
        for ls in tables[0]:
            for rs in tables[1]:
                work.tick();stats['joint_region_inventory_pairs']+=1
                if joined_signature(owner,regions[0],ls,regions[1],rs,domains,work.tick) is not None:
                    retain(0,ls);retain(1,rs)
        return tuple(supports)
    zdd=InventoryZDD(max_nodes,work.tick);groups=[]
    try:
        for table in tables:
            group={}
            for signature in table:
                work.tick();group.setdefault(signature[1:],{})[signature[0]]=signature
            roots={key:zdd.from_sets(inventories) for key,inventories in group.items()}
            stats['joint_region_interfaces']+=len(group);groups.append((group,roots))
        left,lroots=groups[0];right,rroots=groups[1]
        for li,lrows in left.items():
            for ri,rrows in right.items():
                work.tick();stats['joint_region_interface_joins']+=1
                # Geometry ignores inventory here. Full board coverage leaves
                # no outside resource obligations; the exact ZDD handles all
                # physical disjointness separately within this interface pair.
                if joined_signature(owner,regions[0],(0,*li),regions[1],(0,*ri),domains,work.tick) is None:continue
                if not zdd.disjoint_product(lroots[li],rroots[ri]):continue
                for index,rows,other in ((0,lrows,rroots[ri]),(1,rrows,lroots[li])):
                    for inventory,signature in rows.items():
                        work.tick();stats['joint_region_support_probes']+=1
                        if zdd.exists_disjoint(other,inventory):retain(index,signature)
        return tuple(supports)
    finally:
        stats['joint_region_zdd_nodes']+=len(zdd.nodes);stats['joint_region_zdd_operations']+=zdd.operations


class JointSymbolicPruner:
    def __init__(self,owner,*,symbolic=True,limit=8):
        if type(symbolic)is not bool or type(limit)is not int or not 1<=limit<=32:raise ValueError('Invalid joint symbolic policy.')
        self.owner=owner;self.problem=owner.problem;self.binding=source_binding(owner);self.symbolic=symbolic;self.limit=limit
        self.original=tuple(sum(1<<code for code in set(row)) for row in self.problem.domains)
        self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)
    def prune(self,domains,check=lambda:None,*,max_states=4096,max_entries=8192,max_nodes=50000,max_steps=50000):
        if source_binding(self.owner)!=self.binding:raise ValueError('Joint symbolic source binding changed.')
        limits=(max_states,max_entries,max_nodes,max_steps)
        if any(type(bound)is not int or bound<0 for bound in limits):raise ValueError('Invalid joint symbolic bounds.')
        if len(domains)!=self.problem.n or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(domains,self.original)):
            raise ValueError('Joint symbolic domains escape the original root.')
        if not all(domains):return False
        if self.problem.n<2 or self.problem.n>12 and sum(mask.bit_count()>1 for mask in domains)>8:
            self.stats['joint_region_admission_skips']+=1;return False
        check();key=(tuple(domains),limits)
        if key in self.cache:self.stats['joint_region_cache_hits']+=1;supports=self.cache[key];self.cache.move_to_end(key)
        else:
            self.stats['joint_region_queries']+=1;work=_Work(max_steps,check,self.stats)
            middle=self.problem.n//2;regions=(tuple(range(middle)),tuple(range(middle,self.problem.n)))
            try:
                tables=[complete_relation(self.owner,cells,domains,work,max_states,max_entries,self.stats) for cells in regions]
                supports=joint_supports(self.owner,regions,tables,domains,work,self.stats,max_nodes=max_nodes,symbolic=self.symbolic)
            except (JointLimit,_Budget):supports=None
            check();self.cache[key]=supports
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
            self.stats['joint_region_unknown' if supports is None else 'joint_region_completed']+=1
        if supports is None:return False
        current=[mask&keep for mask,keep in zip(domains,supports)];check()
        self.stats['joint_region_prunes']+=sum((old&~new).bit_count() for old,new in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
