"""Exact domain-box algebra and portable remaining-work records.

A box is a Cartesian product of orientation domains, before puzzle constraints.
Subtraction is exact even if boxes contain assignments violating inventory.
Completed coverage is trusted local checkpoint state, NOT a portable proof.
Proof imports must be independently checked before their boxes enter a ledger.
"""
from __future__ import annotations
from dataclasses import dataclass
from array import array
from sys import getsizeof
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
    if not current:return (),True
    for excluded in exclusions:
        check();next_boxes=[]
        for source in current:
            check();next_boxes.extend(subtract(source,excluded))
            if len(next_boxes)>max_boxes:return original,False
        current=tuple(next_boxes)
        if not current:break
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


def _compact_siblings(rows, check):
    """Coalesce a validated, disjoint portable cover. Native cursors own their rows."""
    if len(rows) < 2:
        return rows
    first = rows[0]
    axes = [i for i, mask in enumerate(first) if any(row[i] != mask for row in rows)]
    short = [tuple(row[i] for i in axes) for row in rows]
    hull = [0] * len(axes)
    volume = 0
    for row in short:
        check()
        size = 1
        for i, mask in enumerate(row):
            hull[i] |= mask
            size *= mask.bit_count()
        volume += size
    hull_size = 1
    for mask in hull:
        hull_size *= mask.bit_count()
    # Disjoint subsets fill their Cartesian hull exactly iff their sizes sum to it.
    if volume == hull_size:
        short = [hull]
    else:
        while True:
            before = len(short)
            for i in range(len(axes)):
                groups = {}
                for row in short:
                    check()
                    rest = row[:i] + row[i + 1:]
                    groups[rest] = groups.get(rest, 0) | row[i]
                if len(groups) < len(short):
                    short = [rest[:i] + (mask,) + rest[i:] for rest, mask in groups.items()]
            if len(short) == before:
                break
    if len(short) == len(rows):
        return rows
    merged = []
    for row in short:
        full = list(first)
        for axis, mask in zip(axes, row):
            full[axis] = mask
        merged.append(tuple(full))
    return tuple(merged)


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


def _validated_boxes(scope, boxes, check):
    boxes=tuple(cube(x,len(scope.root)) for x in boxes)
    if any(not contains(scope.root,x) for x in boxes) or not disjoint(boxes,check):
        raise ValueError('Portable frontier overlaps or escapes its root.')
    return boxes


def _frontier_payload(scope, boxes):
    result={'work_version':1,'scope':scope.payload(),'boxes':[[hex(x) for x in row] for row in boxes]}
    result['digest']=digest(result)
    return result


def frontier_snapshot(scope,boxes,*,check=lambda:None, compact=False):
    boxes = _validated_boxes(scope, boxes, check)
    return _frontier_payload(scope, _compact_siblings(boxes, check) if compact else boxes)


def _read_frontier(payload, expected, check):
    if not isinstance(payload,dict) or set(payload)!={'work_version','scope','boxes','digest'} or payload['work_version']!=1:
        raise ValueError('Unknown portable frontier format.')
    if digest({k:v for k,v in payload.items() if k!='digest'})!=payload['digest']:raise ValueError('Portable frontier checksum differs.')
    source=WorkScope.restore(payload['scope'])
    if not expected.accepts(source):raise ValueError('Portable frontier belongs to another puzzle/objective/root.')
    rows=tuple(tuple(int(x,16) for x in row) for row in payload['boxes'])
    return source, _validated_boxes(source, rows, check)


def restore_frontier(payload,expected,*,check=lambda:None):
    return _read_frontier(payload, expected, check)[1]


def compact_frontier(payload, expected, *, check=lambda:None):
    """Preserve the original transfer identity when its verified rows cannot merge."""
    scope, rows = _read_frontier(payload, expected, check)
    merged = _compact_siblings(rows, check)
    return payload if merged is rows else _frontier_payload(scope, merged)


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


class _CoverageMaskTable:
    """Lossless masks and sparse changes to previously verified dense boxes."""
    anchor_limit = 256
    literal_fallback = True

    def __init__(self, root):
        self.root = root
        self.ids = {}
        self.masks = [0]  # Zero denotes this axis's target-root mask.
        self.anchors = {}
        self.literal_anchors = {}

    def encode(self, row, available):
        if row is None:
            return (None, 16) if available >= 16 else None
        values = [0] * len(row)
        changed = []
        new = {}
        for axis, (mask, root_mask) in enumerate(zip(row, self.root)):
            if mask == root_mask:
                continue
            ident = self.ids.get(mask) or new.get(mask)
            if ident is None:
                ident = len(self.masks) + len(new)
                new[mask] = ident
            values[axis] = ident
            changed.extend((axis, ident))
        if (self.literal_fallback and len(new) >= max(16, len(row) // 2)
                and sum(mask.bit_count() > 1 for mask in new) >= len(new) // 2):
            # Fresh, unrelated wide masks cost more to intern than to store once.
            # Nearby boxes can still share such a row. A bucket only suggests a
            # base; all positions are compared, including on a hash collision.
            stride = max(2, len(row) // 8)
            buckets = (hash(row[::stride]) % 32, 32 + hash(row[1::stride]) % 32)
            base = None
            delta = ()
            for bucket in buckets:
                candidate = self.literal_anchors.get(bucket)
                if candidate is None:
                    continue
                changes = tuple((i, v) for i, v in enumerate(row) if v != candidate[i])
                if len(changes) * 4 <= len(row):
                    base, delta = candidate, changes
                    break
            full = base is None
            if full:
                base = row
                size = getsizeof(row) + sum(map(getsizeof, row))
                size += 128 * sum(bucket not in self.literal_anchors for bucket in buckets)
            else:
                size = getsizeof(delta) + sum(getsizeof(pair) + sum(map(getsizeof,pair)) for pair in delta)
            record = None, base, delta
            size += 16 + getsizeof(record)
            if size > available:
                return None
            if full:
                for bucket in buckets:
                    self.literal_anchors[bucket] = base
            return record, size
        anchor = None
        key = None
        dense = len(changed) >= len(row)
        if dense and len(row) >= 16 and self.anchor_limit:
            # A collision changes compression only. Every coordinate difference is stored.
            key = hash(tuple(values[::max(1, len(row) // 8)]))
            previous = self.anchors.get(key)
            if previous is not None:
                patch = []
                for i, value in enumerate(values):
                    if value != previous[i]:
                        patch.extend((i, value))
                if len(patch) < len(values):
                    dense, anchor, changed = False, previous, patch
        values = values if dense else changed
        largest = max(values, default=0)
        typecode = 'B' if largest < 256 else 'H' if largest < 65536 else 'I' if largest < 2**32 else 'Q'
        packed = array(typecode, values)
        record = dense, anchor, packed
        size = 16 + getsizeof(record) + getsizeof(packed)
        size += sum(104 + getsizeof(mask) + getsizeof(ident) for mask, ident in new.items())
        remember = dense and key is not None and key not in self.anchors and len(self.anchors) < self.anchor_limit
        if remember:
            size += 112 + getsizeof(key)
        if size > available:
            return None
        self.ids.update(new)
        self.masks.extend(new)
        if remember:
            self.anchors[key] = packed
        return record, size

    def matching(self, records, hull, check):
        restricted = [i for i, mask in enumerate(hull) if mask != self.root[i]]
        bad_axes = {}
        decoded = {}
        for record in records:
            check()
            if record is None:
                continue
            dense, anchor, values = record
            if dense is None:
                if values:
                    row = list(anchor)
                    for i, mask in values:
                        row[i] = mask
                    row = tuple(row)
                else:
                    row = anchor
                if all(row[i] & hull[i] for i in restricted):
                    yield row
                continue
            if dense:
                if any(values[i] and not self.masks[values[i]] & hull[i] for i in restricted):
                    continue
                yield tuple(self.masks[v] if v else base for v, base in zip(values, self.root))
                continue

            bad = ()
            key = id(anchor)
            if anchor is not None:
                if key not in bad_axes:
                    bad_axes[key] = {i for i in restricted if anchor[i] and not self.masks[anchor[i]] & hull[i]}
                bad = bad_axes[key]
            repaired = 0
            for j in range(0, len(values), 2):
                axis, ident = values[j], values[j + 1]
                if ident and not self.masks[ident] & hull[axis]:
                    break
                repaired += axis in bad
            else:
                if repaired == len(bad):
                    if anchor is None:
                        row = list(self.root)
                    else:
                        if key not in decoded:
                            decoded[key] = tuple(self.masks[v] if v else base for v, base in zip(anchor, self.root))
                        row = list(decoded[key])
                    for j in range(0, len(values), 2):
                        axis, ident = values[j], values[j + 1]
                        row[axis] = self.masks[ident] if ident else self.root[axis]
                    yield tuple(row)


class CoverageLedger:
    """SQLite-transactional local completed coverage, shared across search policies.

    Scope is mathematical; branch ordering and backend are intentionally absent.
    A narrow/symmetry root cannot prove anything outside that root. Checksums
    detect corruption only. User-edited databases are not external certificates.
    """
    def __init__(self,path,scope,*,symbolic=False,max_symbolic_nodes=200000):
        self.scope=scope;self.path=Path(path)
        self.symbolic=bool(symbolic);self.max_symbolic_nodes=max_symbolic_nodes;self._symbolic_state=None;self._symbolic_additions={}
        # Private, checked prefix for repeated ordinary coverage queries.
        self._coverage_key = None
        self._coverage_prefix = []
        self._coverage_cache_limit = 8 * 1024 * 1024
        self._coverage_bytes = 0
        self._coverage_after = 0
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(self.path),timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS closed (identity TEXT PRIMARY KEY, scope TEXT NOT NULL, box TEXT NOT NULL, evidence TEXT NOT NULL, checksum TEXT NOT NULL)')
        from r34_coverage_index import install
        install(self.db)
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
            self._symbolic_additions[identity]=box
            if len(self._symbolic_additions)>4096:self._symbolic_state=None;self._symbolic_additions={}
        return bool(cursor.rowcount)

    def _coverage_target(self, target_scope):
        target = self.scope if target_scope is None else target_scope
        if (target.fingerprint, target.target, target.original) != (self.scope.fingerprint, self.scope.target, self.scope.original):
            raise ValueError('Coverage target differs from bound puzzle/objective.')
        return target

    def exclusions(self, target_scope=None, *, check=lambda:None):
        target = self._coverage_target(target_scope)
        if self._coverage_cache_limit and not self.db.in_transaction:
            return self._use_checked_coverage(target, target.root, check, tuple)
        return tuple(self.iter_exclusions(target, check=check))

    def iter_exclusions(self,target_scope=None,*,check=lambda:None):
        """Stream independently checked local records; no unbounded eager copy."""
        target = self._coverage_target(target_scope)
        from r34_coverage_index import scope_id
        yield from self._checked_rows(target,scope_id(self.db,target),0,2**63-1,{},check)

    def _checked_rows(self,target,group,after,through,local,check):
        from r34_coverage_index import entries
        for identity,scope_json,box_json,evidence_json,checksum in entries(self.db,group,after,through):
            check()
            if identity in local:
                clipped=intersect(local[identity],target.root)
                if clipped is not None:yield clipped
                continue
            row = self._verified_coverage_row(target, scope_json, box_json, evidence_json, checksum)
            if row is not None:
                yield row

    def _verified_coverage_row(self, target, scope_json, box_json, evidence_json, checksum):
        self.stats['coverage_records_checked'] += 1
        body = {'scope': json.loads(scope_json), 'box': json.loads(box_json), 'evidence': json.loads(evidence_json)}
        if digest(body) != checksum:
            raise ValueError('Completed coverage is corrupted.')
        source = WorkScope.restore(body['scope'])
        if (source.fingerprint, source.target, source.original) != (target.fingerprint, target.target, target.original):
            return None
        row = cube((int(x, 16) for x in body['box']), len(target.root))
        if not contains(source.root, row):
            raise ValueError('Stored closed box escapes its root.')
        return intersect(row, target.root)

    def _selective_remaining(self, source, target, hull, max_boxes, check):
        def subtract_checked(records):
            return subtract_many(source, records, max_boxes=max_boxes, check=check)
        return self._use_checked_coverage(target, hull, check, subtract_checked)

    def _use_checked_coverage(self, target, hull, check, consume):
        from r34_coverage_index import scope_id, version
        # Read the generation and payloads in the same committed WAL snapshot.
        self.db.execute('BEGIN')
        try:
            group = scope_id(self.db, target)
            epoch, through = version(self.db, group)
            if self._coverage_key != (target, group, epoch):
                self._coverage_key = (target, group, epoch)
                self._coverage_prefix = []
                self._coverage_after = 0
                self._coverage_bytes = 0

                self._coverage_table = _CoverageMaskTable(target.root)

            def records():
                yield from self._coverage_table.matching(self._coverage_prefix, hull, check)
                if self._coverage_after == through:
                    return
                sql = """SELECT s.seq,c.scope,c.box,c.evidence,c.checksum
                    FROM coverage_sequence s JOIN closed c ON c.identity=s.identity
                    WHERE s.scope_id=? AND s.seq>? AND s.seq<=? ORDER BY s.seq"""
                cursor = self.db.execute(sql, (group, self._coverage_after, through))
                can_extend = True
                try:
                    for seq, *columns in cursor:
                        check()
                        row = self._verified_coverage_row(target, *columns)
                        # Only cache a contiguous, independently checked sequence prefix.
                        # Sparse deltas reconstruct the complete box in this exact root.
                        if can_extend:
                            encoded = self._coverage_table.encode(row, self._coverage_cache_limit - self._coverage_bytes)
                            can_extend = encoded is not None
                            if can_extend:
                                record, size = encoded
                                self._coverage_prefix.append(record)
                                self._coverage_bytes += size
                                self._coverage_after = seq
                        if row is not None:
                            yield row
                finally:
                    cursor.close()
            rows = records()
            try:
                return consume(rows)
            finally:
                rows.close()
        finally:
            self.db.rollback()

    def remaining(self,boxes,*,target_scope=None,max_boxes=65536,check=lambda:None):
        source=tuple(cube(row) for row in boxes)
        if self.symbolic:return self._remaining_symbolic(source,target_scope,max_boxes,check)
        target = self.scope if target_scope is None else target_scope
        hull = None
        if source and len(source) <= max_boxes and self._coverage_cache_limit and not self.db.in_transaction:
            self._coverage_target(target)
            hull = [0] * len(target.root)
            for row in source:
                check()
                if len(row) != len(hull):
                    hull = None
                    break
                for axis, mask in enumerate(row):
                    if mask & ~target.root[axis]:
                        hull = None
                        break
                    hull[axis] |= mask
                if hull is None:
                    break
        if hull is not None and (self._coverage_prefix or tuple(hull) != target.root):
            result, done = self._selective_remaining(source, target, hull, max_boxes, check)
        else:
            result, done = subtract_many(source, self.iter_exclusions(target_scope, check=check), max_boxes=max_boxes, check=check)
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
        from r34_coverage_index import scope_id,version
        state=self._symbolic_state;diagram=None;before=0
        own_snapshot=not self.db.in_transaction
        if own_snapshot:self.db.execute('BEGIN')
        else:state=None  # An outer transaction can roll back; never cache its proofs.
        try:
            group=scope_id(self.db,target)
            epoch,through=version(self.db,group)
            key=(target,epoch,self.max_symbolic_nodes)
            if state is not None and state[0]==key:
                _,after,diagram,closed=state
                before=diagram.stats['symbolic_apply_steps']
                if through>after:
                    for row in self._checked_rows(target,group,after,through,self._symbolic_additions,check):
                        closed=diagram.apply('union',closed,diagram.from_cube(row),check)
                        self.stats['coverage_symbolic_extensions']+=1
                self.stats['coverage_symbolic_reuses']+=1
            else:
                diagram=DomainDiagram(target.root,max_nodes=self.max_symbolic_nodes)
                closed=diagram.union(self._checked_rows(target,group,0,through,{},check),check)
                self.stats['coverage_symbolic_builds']+=1
            check()
            if own_snapshot:
                self._symbolic_state=(key,through,diagram,closed)
                self._symbolic_additions={}
            original_pending=diagram.union(source,check)
            pending=diagram.apply('difference',original_pending,closed,check)
            result=diagram.boxes(pending,max_boxes=max_boxes,check=check);check()
            if pending!=original_pending:self.stats['coverage_hits']+=1
            return result,True
        except DiagramLimit:
            self._symbolic_state=None;self._symbolic_additions={};self.stats['coverage_symbolic_caps']+=1
            return source,False
        finally:
            if own_snapshot:self.db.rollback()
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
