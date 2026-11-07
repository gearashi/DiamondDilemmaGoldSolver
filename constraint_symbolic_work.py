"""Exact reduced multi-valued decision DAGs for finite domain-box sets.

Nodes label disjoint value masks, merge equal children, and skip unrestricted
axes. Terminal 1 means every remaining root value, not a completed search proof.
The caller alone establishes the authority of a completed-region union.
"""
from constraint_work import cube,contains


class DiagramLimit(Exception):pass


class DomainDiagram:
    def __init__(self,root,*,max_nodes=200000):
        self.root=cube(root)
        if type(max_nodes)is not int or max_nodes<2:raise ValueError('Invalid symbolic node cap.')
        self.max_nodes=max_nodes;self.nodes={};self.unique={};self.memo={}
        self.stats={'symbolic_apply_steps':0,'symbolic_cache_hits':0,'symbolic_nodes':0}

    def _level(self,node):return len(self.root) if node<2 else self.nodes[node][0]

    def _make(self,level,edges):
        grouped={}
        for mask,child in edges:
            if mask and child:grouped[child]=grouped.get(child,0)|mask
        if not grouped:return 0
        if len(grouped)==1:
            child,mask=next(iter(grouped.items()))
            if mask==self.root[level]:return child
        key=(level,tuple((mask,child) for child,mask in sorted(grouped.items())))
        old=self.unique.get(key)
        if old is not None:return old
        if len(self.nodes)+2>=self.max_nodes:raise DiagramLimit('Symbolic node cap reached.')
        index=len(self.nodes)+2;self.nodes[index]=key;self.unique[key]=index
        self.stats['symbolic_nodes']=len(self.nodes)
        return index

    def from_cube(self,box):
        box=cube(box,len(self.root))
        if not contains(self.root,box):raise ValueError('Symbolic box escapes root.')
        node=1
        for level in reversed(range(len(self.root))):node=self._make(level,((box[level],node),))
        return node

    def _edges(self,node,level):
        if self._level(node)>level:return ((self.root[level],node),)
        edges=self.nodes[node][1];used=0
        for mask,_ in edges:used|=mask
        missing=self.root[level]&~used
        return edges+((missing,0),) if missing else edges

    def apply(self,operation,left,right,check=lambda:None):
        if operation not in ('union','difference'):raise ValueError('Unsupported set operation.')
        check()
        if operation=='union':
            if not left:return right
            if not right or left==right:return left
            if left==1 or right==1:return 1
            if left>right:left,right=right,left
        else:
            if not left or right==1 or left==right:return 0
            if not right:return left
        key=(operation,left,right)
        if key in self.memo:
            self.stats['symbolic_cache_hits']+=1;return self.memo[key]
        self.stats['symbolic_apply_steps']+=1
        level=min(self._level(left),self._level(right));edges=[]
        for a,x in self._edges(left,level):
            for b,y in self._edges(right,level):
                common=a&b
                if common:edges.append((common,self.apply(operation,x,y,check)))
        result=self._make(level,edges)
        if len(self.memo)>=self.max_nodes*4:raise DiagramLimit('Symbolic operation-cache cap reached.')
        self.memo[key]=result;return result

    def union(self,boxes,check=lambda:None):
        root=0
        for box in boxes:
            check();root=self.apply('union',root,self.from_cube(box),check)
        return root

    def boxes(self,node,*,max_boxes=65536,check=lambda:None):
        if type(max_boxes)is not int or max_boxes<0:raise ValueError('Invalid symbolic output cap.')
        current=list(self.root);out=[]
        def visit(index):
            check()
            if not index:return
            if index==1:
                if len(out)>=max_boxes:raise DiagramLimit('Symbolic cube-export cap reached.')
                out.append(tuple(current));return
            level,edges=self.nodes[index];old=current[level]
            for mask,child in edges:
                current[level]=mask;visit(child)
            current[level]=old
        visit(node);return tuple(out)

    def volume(self,node):
        memo={}
        def count(index,start):
            key=(index,start)
            if key in memo:return memo[key]
            if not index:return 0
            level=self._level(index);factor=1
            for axis in range(start,level):factor*=self.root[axis].bit_count()
            value=factor if index==1 else factor*sum(mask.bit_count()*count(child,level+1) for mask,child in self.nodes[index][1])
            memo[key]=value;return value
        return count(node,0)
