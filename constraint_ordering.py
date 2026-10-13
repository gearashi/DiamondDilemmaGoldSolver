"""Bounded least-constraining placement ordering; never excludes a value."""
from collections import OrderedDict
STAT_KEYS=('value_order_calls','value_order_scored','value_order_cache_hits','value_order_skips','value_order_changed')
class SupportOrdering:
    def __init__(self,tables,limit=128,max_choices=48):
        self.tables=tables;self.limit=limit;self.max_choices=max_choices;self.cache=OrderedDict()
        self.adjacency=[[] for _ in tables.tile_bits]
        for a,sa,b,sb in tables.arcs:self.adjacency[a].append((b,sa,sb))
    def order(self,cell,domains,choices,hinted,rank,check,stats):
        check();stats['value_order_calls']+=1
        fallback=lambda code:(0,hinted[code]) if code in hinted else (1,rank[code])
        if not 1<len(choices)<=self.max_choices:
            stats['value_order_skips']+=1;return sorted(choices,key=fallback)
        neighbors=tuple(sorted({b for b,_,_ in self.adjacency[cell]}))
        key=(cell,domains[cell],tuple(sorted(choices)),tuple(domains[b] for b in neighbors))
        scores=self.cache.get(key)
        if scores is None:
            scores={}
            for code in choices:
                check();support={b:domains[b] for b in neighbors}
                # Parallel seams to one neighbor must be satisfied jointly.
                for b,sa,sb in self.adjacency[cell]:support[b]&=self.tables.compatible[sa][sb][code]
                counts=[value.bit_count() for value in support.values()]
                product=1
                for count in counts:product*=count
                scores[code]=(min(counts,default=0),product,sum(counts))
                stats['value_order_scored']+=1
            self.cache[key]=scores
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key);stats['value_order_cache_hits']+=1
        ordered=sorted(choices,key=lambda code:(0,hinted[code],0,0,rank[code]) if code in hinted else (1,*(-value for value in scores[code]),rank[code]))
        stats['value_order_changed']+=int(ordered!=sorted(choices,key=fallback))
        return ordered
