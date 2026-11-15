"""Rank-local, family-typed dependency receipts and checked ranges of completed supports."""
from bisect import bisect_left,bisect_right
from collections import OrderedDict,defaultdict
import json


class Dependencies:
    """Keep the complete family union even when the optional per-rank refinement fills."""
    def __init__(self,limit=4096):
        self.limit=limit
        self.union=defaultdict(set)
        self.by_rank=defaultdict(lambda:defaultdict(set))
        self.count=0

    def add(self,owner,family,source_rank,donor,donor_rank,donor_family):
        key=(owner,tuple(family))
        link=(donor,int(donor_rank),tuple(donor_family))
        self.union[key].add(link)
        if self.by_rank is not None:
            row=self.by_rank[key][int(source_rank)]
            if link not in row:
                row.add(link);self.count+=1
                if self.count>self.limit:self.by_rank=None

    def publish(self,db,scope,start,owners,families):
        for owner in owners:
            aggregate=set()
            for family in families:
                key=owner,tuple(family)
                full=self.union.get(key,set());aggregate.update(full)
                sparse=None if self.by_rank is None else {
                    rank:sorted(row) for rank,row in self.by_rank.get(key,{}).items()}
                data=json.dumps([sorted(full),sparse],separators=(',',':'))
                db.execute('INSERT INTO r35_page_dependencies VALUES(?,?,?,?,?)',
                           (scope,start,owner,json.dumps(family),data))
            db.execute('INSERT INTO r35_page_dependencies VALUES(?,?,?,?,?)',
                       (scope,start,owner,'*',json.dumps([sorted(aggregate),None],separators=(',',':'))))


def selected(raw,rank,lo,hi):
    full,sparse=json.loads(raw)
    if sparse is None:return lo,hi,full
    ranks=sorted((int(k) for k in sparse))
    pos=bisect_left(ranks,rank)
    if pos<len(ranks) and ranks[pos]==rank:
        return rank,rank+1,sparse[str(rank)]
    return max(lo,ranks[pos-1]+1 if pos else lo),min(hi,ranks[pos] if pos<len(ranks) else hi),()


class CheckedRanges:
    def __init__(self):
        self.by_scope=OrderedDict()
        self.count=0

    def get(self,key,rank):
        data=self.by_scope.get(key)
        if data:
            self.by_scope.move_to_end(key)
            at=bisect_right(data,(rank,float('inf')))-1
            if at>=0 and rank<data[at][1]:return data[at][1]
        return rank

    def add(self,key,lo,hi):
        data=self.by_scope.setdefault(key,[])
        self.by_scope.move_to_end(key)
        at=bisect_left(data,(lo,))
        if at and data[at-1][1]>=lo:at-=1
        end=at
        while end<len(data) and data[end][0]<=hi:
            lo=min(lo,data[end][0]);hi=max(hi,data[end][1]);end+=1
        data[at:end]=[(lo,hi)];self.count+=1-(end-at)
        while self.count>4096:
            _,removed=self.by_scope.popitem(last=False);self.count-=len(removed)
