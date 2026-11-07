"""Exact domain-box algebra and portable remaining-work records.

A box is a Cartesian product of orientation domains, before puzzle constraints.
Subtraction is exact even if boxes contain assignments violating inventory.
Completed coverage is trusted local checkpoint state, NOT a portable proof.
Proof imports must be independently checked before their boxes enter a ledger.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def cube(values, size=None):
    result=tuple(values)
    if not result or size is not None and len(result)!=size or any(type(x)is not int or x<=0 for x in result):
        raise ValueError('A work cube requires one positive integer domain per cell.')
    return result


def intersect(left,right):
    left=cube(left);right=cube(right,len(left))
    out=tuple(a&b for a,b in zip(left,right))
    return out if all(out) else None


def contains(outer,inner):
    return len(outer)==len(inner) and all(not b&~a for a,b in zip(outer,inner))


def subtract(source, excluded):
    """Disjoint boxes covering source minus excluded, without enumerating leaves."""
    source=cube(source);excluded=cube(excluded,len(source))
    common=intersect(source,excluded)
    if common is None:return (source,)
    out=[];prefix=list(source)
    for i,(old,inside) in enumerate(zip(source,common)):
        outside=old&~inside
        if outside:
            part=prefix.copy();part[i]=outside;out.append(tuple(part))
        prefix[i]=inside
    return tuple(out)


def subtract_many(sources, exclusions, *, max_boxes=65536, check=lambda:None):
    """Exact subtraction; exceeding the cap returns the original cover unchanged."""
    original=tuple(cube(x) for x in sources);current=original
    for excluded in exclusions:
        check();next_boxes=[]
        for source in current:
            check();next_boxes.extend(subtract(source,excluded))
            if len(next_boxes)>max_boxes:return original,False
        current=tuple(next_boxes)
    return current,True


def disjoint(boxes,check=lambda:None):
    """Exact overlap test using domain-mask posting lists of prior box IDs.

    Unrestricted axes need no index. This is particularly useful for long DFS
    frontiers: most axes are unchanged and a few prefix axes separate cubes.
    No sampling or work cap can turn an incomplete check into acceptance.
    """
    rows=[];universe=None
    for values in boxes:
        check();row=cube(values,None if universe is None else len(universe))
        if universe is None:universe=list(row)
        else:
            for axis,mask in enumerate(row):universe[axis]|=mask
        rows.append(row)
    if len(rows)<2:return True
    postings=[{} for _ in universe];restricted=[0]*len(universe);prior=0
    for index,row in enumerate(rows):
        check();possible=prior
        axes=[(axis,mask) for axis,mask in enumerate(row) if mask!=universe[axis]]
        for axis,mask in axes:
            check()
            # Prior boxes with a full-axis domain overlap any current domain.
            overlapping=prior&~restricted[axis]
            for count,(old,ids) in enumerate(postings[axis].items()):
                if count%256==0:check()
                if old&mask:overlapping|=ids
            possible&=overlapping
            if not possible:break
        if possible:return False
        bit=1<<index
        for axis,mask in axes:
            postings[axis][mask]=postings[axis].get(mask,0)|bit
            restricted[axis]|=bit
        prior|=bit
    return True


def coalesce(boxes):
    """Merge boxes differing on one axis; this changes no represented assignment."""
    rows=list(dict.fromkeys(cube(x) for x in boxes))
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(rows):
            for j in range(i):
                b=rows[j]
                if contains(a,b):rows.pop(j);changed=True;break
                if contains(b,a):rows.pop(i);changed=True;break
                axes=[k for k,(x,y) in enumerate(zip(a,b)) if x!=y]
                if len(axes)==1:
                    k=axes[0];merged=list(a);merged[k]|=b[k]
                    rows[j]=tuple(merged);rows.pop(i);changed=True;break
            if changed:break
    return tuple(rows)


@dataclass(frozen=True)
class WorkScope:
    fingerprint:str
    target:str
    original:tuple
    root:tuple

    def __post_init__(self):
        if not isinstance(self.fingerprint,str) or not self.fingerprint or self.target not in ('gold','edges'):
            raise ValueError('Invalid puzzle or objective scope.')
        original=cube(self.original);root=cube(self.root,len(original))
        if not contains(original,root):raise ValueError('Root escapes original domains.')
        object.__setattr__(self,'original',original);object.__setattr__(self,'root',root)

    @classmethod
    def for_problem(cls,problem,target='gold',root=None):
        original=tuple(sum(1<<code for code in set(row)) for row in problem.domains)
        return cls(problem.fingerprint,target,original,original if root is None else root)

    def accepts(self,other):
        return (self.fingerprint,self.target,self.original)==(other.fingerprint,other.target,other.original) and contains(self.root,other.root)

    def payload(self):
        return {'fingerprint':self.fingerprint,'target':self.target,'original':[hex(x) for x in self.original],'root':[hex(x) for x in self.root]}

    @classmethod
    def restore(cls,value):
        if not isinstance(value,dict) or set(value)!={'fingerprint','target','original','root'}:raise ValueError('Malformed work scope.')
        return cls(value['fingerprint'],value['target'],tuple(int(x,16) for x in value['original']),tuple(int(x,16) for x in value['root']))


def frontier_snapshot(scope,boxes,*,check=lambda:None):
    boxes=tuple(cube(x,len(scope.root)) for x in boxes)
    if any(not contains(scope.root,x) for x in boxes) or not disjoint(boxes,check):
        raise ValueError('Portable frontier overlaps or escapes its root.')
    result={'work_version':1,'scope':scope.payload(),'boxes':[[hex(x) for x in row] for row in boxes]}
    result['digest']=digest(result)
    return result


def restore_frontier(payload,expected,*,check=lambda:None):
    if not isinstance(payload,dict) or set(payload)!={'work_version','scope','boxes','digest'} or payload['work_version']!=1:
        raise ValueError('Unknown portable frontier format.')
    if digest({k:v for k,v in payload.items() if k!='digest'})!=payload['digest']:raise ValueError('Portable frontier checksum differs.')
    source=WorkScope.restore(payload['scope'])
    if not expected.accepts(source):raise ValueError('Portable frontier belongs to another puzzle/objective/root.')
    rows=tuple(tuple(int(x,16) for x in row) for row in payload['boxes'])
    frontier_snapshot(source,rows,check=check)
    return rows


def checkpoint_boxes(checkpoint):
    """Extract pending DFS products AFTER caller validates the native checkpoint."""
    result=[]
    for frame in checkpoint['stack']:
        domains=[int(x,16) for x in frame['domains']]
        if frame['cell']>=0:
            masks=[int(x,16) for x in frame['choice_masks']] if 'choice_masks' in frame else [1<<x for x in frame['choices']]
            remaining=masks[frame['next']:]
            if not remaining:continue
            mask=0
            for part in remaining:
                if mask&part:raise ValueError('DFS choice masks overlap.')
                mask|=part
            domains[frame['cell']]=mask
        if all(domains):result.append(cube(domains))
    return tuple(result)


class CoverageLedger:
    """SQLite-transactional local completed coverage, shared across search policies.

    Scope is mathematical; branch ordering and backend are intentionally absent.
    A narrow/symmetry root cannot prove anything outside that root. Checksums
    detect corruption only. User-edited databases are not external certificates.
    """
    def __init__(self,path,scope,*,symbolic=False,max_symbolic_nodes=200000):
        self.scope=scope;self.path=Path(path)
        self.symbolic=bool(symbolic);self.max_symbolic_nodes=max_symbolic_nodes;self._symbolic_state=None;self._symbolic_additions=[]
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(self.path),timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS closed (identity TEXT PRIMARY KEY, scope TEXT NOT NULL, box TEXT NOT NULL, evidence TEXT NOT NULL, checksum TEXT NOT NULL)')
        self.stats={'coverage_records_added':0,'coverage_records_checked':0,'coverage_hits':0,'coverage_boxes_removed':0,
                    'coverage_symbolic_builds':0,'coverage_symbolic_reuses':0,'coverage_symbolic_nodes':0,
                    'coverage_symbolic_apply_steps':0,'coverage_symbolic_caps':0,'coverage_symbolic_extensions':0}

    def record_completed(self,box,*,status,complete,source_scope=None,evidence=None):
        source=self.scope if source_scope is None else source_scope
        if not self.scope.accepts(source):raise ValueError('Completed source scope differs.')
        if status!='infeasible' or complete is not True:raise ValueError('Only a completed infeasibility proof closes a box.')
        box=cube(box,len(source.root))
        if not contains(source.root,box):raise ValueError('Closed box escapes proved root.')
        body={'scope':source.payload(),'box':[hex(x) for x in box],'evidence':evidence or {'kind':'completed_exact_search'}}
        identity=digest({'scope':body['scope'],'box':body['box']})
        with self.db:
            cursor=self.db.execute('INSERT OR IGNORE INTO closed VALUES (?,?,?,?,?)',(identity,json.dumps(body['scope']),json.dumps(body['box']),json.dumps(body['evidence']),digest(body)))
        self.stats['coverage_records_added']+=cursor.rowcount
        if cursor.rowcount and self._symbolic_state is not None:
            self._symbolic_additions.append(box)
            if len(self._symbolic_additions)>4096:self._symbolic_state=None;self._symbolic_additions=[]
        return bool(cursor.rowcount)

    def exclusions(self,target_scope=None,*,check=lambda:None):
        return tuple(self.iter_exclusions(target_scope,check=check))

    def iter_exclusions(self,target_scope=None,*,check=lambda:None):
        """Stream independently checked local records; no unbounded eager copy."""
        target=self.scope if target_scope is None else target_scope
        if (target.fingerprint,target.target,target.original)!=(self.scope.fingerprint,self.scope.target,self.scope.original):
            raise ValueError('Coverage target differs from bound puzzle/objective.')
        for scope_json,box_json,evidence_json,checksum in self.db.execute('SELECT scope,box,evidence,checksum FROM closed'):
            check();self.stats['coverage_records_checked']+=1
            body={'scope':json.loads(scope_json),'box':json.loads(box_json),'evidence':json.loads(evidence_json)}
            if digest(body)!=checksum:raise ValueError('Completed coverage is corrupted.')
            source=WorkScope.restore(body['scope'])
            if (source.fingerprint,source.target,source.original)!=(target.fingerprint,target.target,target.original):continue
            row=cube((int(x,16) for x in body['box']),len(target.root))
            if not contains(source.root,row):raise ValueError('Stored closed box escapes its root.')
            clipped=intersect(row,target.root)
            if clipped is not None:yield clipped

    def remaining(self,boxes,*,target_scope=None,max_boxes=65536,check=lambda:None):
        source=tuple(cube(row) for row in boxes)
        if self.symbolic:return self._remaining_symbolic(source,target_scope,max_boxes,check)
        result,done=subtract_many(source,self.iter_exclusions(target_scope,check=check),max_boxes=max_boxes,check=check)
        if done and result!=source:self.stats['coverage_hits']+=1
        return result,done

    def _remaining_symbolic(self,source,target_scope,max_boxes,check):
        from constraint_symbolic_work import DomainDiagram,DiagramLimit
        target=self.scope if target_scope is None else target_scope
        if (target.fingerprint,target.target,target.original)!=(self.scope.fingerprint,self.scope.target,self.scope.original):
            raise ValueError('Coverage target differs from bound puzzle/objective.')
        for row in source:
            if len(row)!=len(target.root) or not contains(target.root,row):raise ValueError('Remaining work escapes target root.')
        check()
        version=(self.db.execute('PRAGMA data_version').fetchone()[0],self.db.total_changes)
        key=(target,version,self.max_symbolic_nodes);state=self._symbolic_state;diagram=None;before=0
        try:
            extend=(state is not None and state[0][0]==target and state[0][2]==self.max_symbolic_nodes
                    and state[0][1][0]==version[0] and state[0][1][1]+len(self._symbolic_additions)==version[1])
            if state is not None and (state[0]==key or extend):
                _,diagram,closed=state;before=diagram.stats['symbolic_apply_steps']
                for row in self._symbolic_additions:
                    check();clipped=intersect(row,target.root)
                    if clipped is not None:closed=diagram.apply('union',closed,diagram.from_cube(clipped),check)
                check();self._symbolic_state=(key,diagram,closed)
                self.stats['coverage_symbolic_extensions']+=len(self._symbolic_additions)
                self._symbolic_additions=[];self.stats['coverage_symbolic_reuses']+=1
            else:
                diagram=DomainDiagram(target.root,max_nodes=self.max_symbolic_nodes)
                closed=diagram.union(self.iter_exclusions(target,check=check),check)
                check();self._symbolic_state=(key,diagram,closed);self._symbolic_additions=[];self.stats['coverage_symbolic_builds']+=1
            original_pending=diagram.union(source,check)
            pending=diagram.apply('difference',original_pending,closed,check)
            result=diagram.boxes(pending,max_boxes=max_boxes,check=check);check()
            if pending!=original_pending:self.stats['coverage_hits']+=1
            return result,True
        except DiagramLimit:
            self._symbolic_state=None;self._symbolic_additions=[];self.stats['coverage_symbolic_caps']+=1
            return source,False
        finally:
            if diagram is not None:
                self.stats['coverage_symbolic_apply_steps']+=diagram.stats['symbolic_apply_steps']-before
                self.stats['coverage_symbolic_nodes']=diagram.stats['symbolic_nodes']

    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def exclusion_box(record,root):
    """Convert an ALREADY VERIFIED forbidden conjunction into its product box."""
    out=list(cube(root));seen=set()
    for cell,mask in record:
        if type(cell)is not int or not 0<=cell<len(out) or cell in seen or type(mask)is not int or mask<=0 or mask&~out[cell]:
            raise ValueError('Verified exclusion escapes its stated root.')
        seen.add(cell);out[cell]=mask
    return tuple(out)
