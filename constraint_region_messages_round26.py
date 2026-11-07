"""Correlated symbolic inventory families at every recursive tree join.

Each geometry bucket fixes every seam mask, exposed Gold pairing and closed
cycle count. ZDD products only combine disjoint physical inventories. Counts,
all-value supports and witness provenance are accumulated from compatible
inventory pairs, never inferred from a union of unrelated geometries.
"""
from constraint_region_symbolic import InventoryZDD
from constraint_region_network import joined_signature
from constraint_separator import _Budget
import copy

STAT_KEYS=('recursive_region_zdd_nodes','recursive_region_zdd_operations',
 'recursive_region_symbolic_geometry_tests','recursive_region_symbolic_inventory_pairs',
 'recursive_region_symbolic_class_skips','recursive_region_symbolic_group_rows')

class SymbolicMessageJoin:
    def __init__(self,owner,left_cells,left,right_cells,right,domains,*,max_nodes=50000,state=None):
        if set(left_cells)&set(right_cells):raise ValueError('Symbolic tree interiors must be disjoint.')
        self.owner=owner;self.left_cells=tuple(left_cells);self.right_cells=tuple(right_cells)
        self.left=left;self.right=right;self.domains=domains;self.max_nodes=max_nodes
        self.zdd=InventoryZDD(max_nodes);self.groups=[{},{}];self.roots=[{},{}]
        self.side=0;self.index=0;self.phase='build';self.gi=0;self.gj=0;self.li=0
        self.geometry=None;self.enum=[];self.current_used=None;self.result={};self.links=[];self.complete=False
        self.stats=dict.fromkeys(STAT_KEYS,0)
        if state is not None:self._restore(state)
    def _next_group(self):
        self.gj+=1;self.li=0;self.geometry=None;self.current_used=None;self.enum=[]
        if self.gj==len(self.groups[1]):self.gi+=1;self.gj=0
    def advance(self,check=lambda:None):
        check()
        if self.complete:return True
        self.zdd.check=check;before_nodes=len(self.zdd.nodes);before_operations=self.zdd.operations
        try:
            if self.phase=='build':
                table=(self.left,self.right)[self.side];rows=tuple(table)
                if self.index==len(rows):
                    self.side+=1;self.index=0
                    if self.side==2:
                        self.phase='join'
                        if not self.groups[0] or not self.groups[1]:self.complete=True
                    return self.complete
                signature=rows[self.index];geometry=signature[1:];used=signature[0]
                self.stats['recursive_region_symbolic_group_rows']+=1
                root=self.zdd.union(self.roots[self.side].get(geometry,0),self.zdd.singleton(used))
                # All mutation after a completed ZDD operation; partial cached
                # suboperations remain valid if a callback interrupts above.
                self.roots[self.side][geometry]=root
                self.groups[self.side].setdefault(geometry,{})[used]=signature;self.index+=1
                return False
            if self.gi==len(self.groups[0]):self.complete=True;return True
            lg=tuple(self.groups[0])[self.gi];rg=tuple(self.groups[1])[self.gj]
            if self.geometry is None:
                self.stats['recursive_region_symbolic_geometry_tests']+=1
                merged=joined_signature(self.owner,self.left_cells,(0,*lg),self.right_cells,(0,*rg),self.domains,check)
                if merged is None:self.stats['recursive_region_symbolic_class_skips']+=1;self._next_group();return False
                family=self.zdd.disjoint_product(self.roots[0][lg],self.roots[1][rg])
                if not family:self.stats['recursive_region_symbolic_class_skips']+=1;self._next_group();return False
                self.geometry=merged[1:];return False
            left_masks=tuple(self.groups[0][lg])
            if self.enum:
                node,mask=self.enum[-1]
                if node==1:
                    left_used=self.current_used;right_used=mask^left_used
                    ls=self.groups[0][lg][left_used];rs=self.groups[1][rg][right_used]
                    le=self.left[ls];re=self.right[rs];signature=(mask,*self.geometry)
                    if signature not in self.result:self.result[signature]={'count':0,'supports':{},'witness':{**le['witness'],**re['witness']}}
                    entry=self.result[signature];entry['count']+=le['count']*re['count']
                    for source in (le,re):
                        for cell,domain in source['supports'].items():entry['supports'][cell]=entry['supports'].get(cell,0)|domain
                    self.links.append((signature,ls,rs));self.stats['recursive_region_symbolic_inventory_pairs']+=1
                    self.enum.pop();return False
                self.enum.pop()
                if node:
                    var,low,high=self.zdd.nodes[node]
                    if high:self.enum.append((high,mask|(1<<var)))
                    if low:self.enum.append((low,mask))
                return False
            if self.current_used is not None:self.li+=1;self.current_used=None
            if self.li==len(left_masks):self._next_group();return False
            used=left_masks[self.li]
            family=self.zdd.disjoint_product(self.zdd.singleton(used),self.roots[1][rg])
            self.current_used=used;self.enum=[] if not family else [(family,0)]
            return False
        finally:
            self.stats['recursive_region_zdd_nodes']+=len(self.zdd.nodes)-before_nodes
            self.stats['recursive_region_zdd_operations']+=self.zdd.operations-before_operations
    def snapshot(self):
        result={key:getattr(self,key) for key in ('groups','roots','side','index','phase','gi','gj','li','geometry','enum','current_used','result','links','complete','stats')}
        result['zdd']={'nodes':dict(self.zdd.nodes),'union':dict(self.zdd._union),'product':dict(self.zdd._product),'operations':self.zdd.operations}
        return copy.deepcopy(result)
    def _restore(self,state):
        state=copy.deepcopy(state)
        required={'groups','roots','side','index','phase','gi','gj','li','geometry','enum','current_used','result','links','complete','stats','zdd'}
        if type(state)is not dict or set(state)!=required or any(type(state[k])is not int or state[k]<0 for k in ('side','index','gi','gj','li')) or state['side']>2 or state['phase'] not in ('build','join') or type(state['complete'])is not bool:raise ValueError('Invalid symbolic message cursor.')
        if set(state['stats'])!=set(STAT_KEYS) or any(type(v)is not int or v<0 for v in state['stats'].values()):raise ValueError('Invalid symbolic message counters.')
        if type(state['groups'])is not list or type(state['roots'])is not list or len(state['groups'])!=2 or len(state['roots'])!=2:raise ValueError('Invalid correlated message buckets.')
        z=state['zdd']
        if type(z)is not dict or set(z)!={'nodes','union','product','operations'} or len(z['nodes'])>self.max_nodes or type(z['operations'])is not int or z['operations']<0:raise ValueError('Invalid symbolic inventory DAG.')
        nodes=z['nodes'];unique={}
        for ident,row in nodes.items():
            if type(ident)is not int or ident<2 or type(row)is not tuple or len(row)!=3 or any(type(v)is not int or v<0 for v in row):raise ValueError('Invalid ZDD node.')
            var,low,high=row
            if not high or var>=self.owner.problem.n or any(child>1 and child not in nodes for child in (low,high)):raise ValueError('Invalid ZDD children.')
            if any(child>1 and nodes[child][0]<=var for child in (low,high)) or row in unique:raise ValueError('Invalid ZDD ordering/canonicalization.')
            unique[row]=ident
        if set(nodes)!=set(range(2,len(nodes)+2)):raise ValueError('Noncanonical ZDD identifiers.')
        def root_valid(root):return type(root)is int and (root in (0,1) or root in nodes)
        for cache in (z['union'],z['product']):
            for pair,result in cache.items():
                if type(pair)is not tuple or len(pair)!=2 or not all(root_valid(v) for v in (*pair,result)):raise ValueError('Invalid completed ZDD operator cache.')
        for side in (0,1):
            table=(self.left,self.right)[side]
            if set(state['groups'][side])!=set(state['roots'][side]):raise ValueError('Symbolic roots/buckets differ.')
            for geometry,group in state['groups'][side].items():
                if not root_valid(state['roots'][side][geometry]):raise ValueError('Invalid symbolic root.')
                for used,signature in group.items():
                    if signature not in table or signature!=(used,*geometry):raise ValueError('Symbolic bucket lost full interface correlation.')
        if state['complete'] and (state['phase']!='join' or state['gi']!=len(state['groups'][0]) and state['groups'][0] and state['groups'][1]):raise ValueError('Symbolic completion is premature.')
        if state['phase']=='join' and state['side']!=2:raise ValueError('Symbolic join began before both input families completed.')
        if state['phase']=='build' and (state['side']>=2 or state['index']>len((self.left,self.right)[state['side']])):raise ValueError('Invalid symbolic build position.')
        if state['gi']>len(state['groups'][0]) or state['gj']>=max(1,len(state['groups'][1])):raise ValueError('Invalid symbolic pair position.')
        for node,mask in state['enum']:
            if not root_valid(node) or type(mask)is not int or not 0<=mask<(1<<self.owner.problem.n):raise ValueError('Invalid inventory enumeration cursor.')
        for key,value in state.items():
            if key!='zdd':setattr(self,key,value)
        self.zdd.nodes=dict(nodes);self.zdd.unique=unique;self.zdd._union=dict(z['union']);self.zdd._product=dict(z['product']);self.zdd.operations=z['operations']
