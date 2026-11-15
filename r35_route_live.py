"""Bound the lifetime of coverage misses as other route workers commit new evidence."""
from collections import Counter,OrderedDict
from time import monotonic


class ReceiptEpoch:
    def __init__(self,db):
        self.db=db
        self.token=None;self.value=0;self.next_poll=0
        self.work=Counter()

    def refresh(self,force=False):
        now=monotonic()
        if not force and now<self.next_poll:return self.value
        self.next_poll=now+.1
        token=self.db.total_changes,self.db.execute('PRAGMA data_version').fetchone()[0]
        self.work['generation_polls']+=1
        if token!=self.token:
            self.token=token;self.value+=1
        return self.value


class EpochMemo:
    """Keep proven successes and invalidate provisional responses at a commit change."""
    def __init__(self,epoch,lookup,*,permanent=bool,limit=8192):
        self.epoch=epoch;self.lookup=lookup;self.permanent=permanent;self.limit=limit
        self.rows=OrderedDict()

    def __call__(self,*args):
        generation=self.epoch.refresh()
        if args in self.rows:
            stamp,value=self.rows[args]
            if stamp is None or stamp==generation:
                self.rows.move_to_end(args)
                return value
        value=self.lookup(*args)
        self.rows[args]=None if self.permanent(value) else generation,value
        self.rows.move_to_end(args)
        if len(self.rows)>self.limit:self.rows.popitem(last=False)
        return value


class RankedCoverage:
    def __init__(self,db,epoch,binding,owner,family,target,root_scope,ranked_scope,free):
        from r32_route_store import PageProofs
        self.db=db;self.epoch=epoch;self.binding=binding;self.owner=owner;self.family=family
        self.target=target;self.root_scope=root_scope;self.ranked_scope=ranked_scope;self.free=free
        self.proofs=PageProofs(db,target);self.work=Counter()
        self.available=EpochMemo(epoch,self._available)
        self.points=EpochMemo(epoch,self._lookup)
        from r35_route_dependencies import CheckedRanges
        self.intervals=CheckedRanges()

    def _available(self):
        import json
        row=self.db.execute('SELECT 1 FROM r30_pages WHERE scope=? LIMIT 1',(self.ranked_scope,)).fetchone()
        if row:return True
        return bool(self.db.execute("""SELECT 1 FROM r31_members m JOIN r30_pages p ON p.scope=m.scope
             WHERE m.domain=? AND m.family=? AND m.owner=? AND m.target<=? LIMIT 1""",
             (self.binding,json.dumps(self.family),self.owner,self.target)).fetchone())

    def covered(self,support):
        if not self.available():return False
        from r32_route_store import colex
        return self.covered_rank(colex(support,self.free))

    def covered_rank(self,point):
        if not self.available():return False
        if self.intervals.get(None,point)>point:return True
        return self.points(point)

    def all_negative(self,pool):
        from itertools import combinations
        from math import comb
        k=sum(self.family)
        if len(pool)<k or comb(len(pool),k)>2048 or not self.available():return False
        return all(self.covered(cells) for cells in combinations(pool,k))

    def _lookup(self,point):
        from r31_route_coverage import inherited_end
        from math import comb
        self.work['ranked_receipt_queries']+=1
        # A known singleton scope was also used by older versions before
        # publication of source membership records.
        row=self.db.execute('SELECT hi,best FROM r30_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1',
                            (self.ranked_scope,point)).fetchone()
        if row and row[0]>point:
            import json
            if row[1]=='null' or json.loads(row[1])['score']<self.target:
                self.intervals.add(None,point,row[0]);return True
        stop=comb(len(self.free),sum(self.family))
        until=inherited_end(self.db,self.root_scope,self.binding,(self.family,),self.target,(self.owner,),
                            point,stop,proofs=self.proofs)
        if until>point:self.intervals.add(None,point,until)
        return until>point

class EndpointReceipts:
    """Consumer of checked scope-matched root and ranked negatives for one physical center."""
    def __init__(self,db,epoch,binding,bound,owner,family,target):
        import hashlib,json
        from r31_route_coverage import register
        from r32_route_store import RootCoverage
        tiles=tuple(map(int,bound.tiles))
        self.key=hashlib.sha256(json.dumps([1,binding,[tiles],(family,),target]).encode()).hexdigest()
        register(db,self.key,binding,(family,),target,(owner,))
        self.ranked=RankedCoverage(db,epoch,binding,owner,family,target,self.key,self.key,
                                   {c:i for i,c in enumerate(sorted(bound.model.free))})
        self.roots=RootCoverage(db,binding,owner,family,target,epoch=epoch)

    def covered(self,cells,order,close):
        return self.ranked.covered(cells) or self.roots.covered(cells,order,close)
