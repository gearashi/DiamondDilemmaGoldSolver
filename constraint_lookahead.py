"""Bounded private probe reuse and selective two-placement contradiction checks.

A cached entry is a completed consequence of its exact input domains and trusted
search context. Unknown/cancelled probes never publish an entry. The cache is
process-local, disposable and deliberately absent from serialized checkpoints.
"""
from collections import OrderedDict

CACHE_STAT_KEYS=('lookahead_cache_hits','lookahead_cached_probes','lookahead_cache_evictions',
                 'lookahead_subset_hits','lookahead_implication_prunes')
PAIR_STAT_KEYS=('pair_lookahead_queries','pair_lookahead_singletons','pair_lookahead_probes',
                'pair_lookahead_failures','pair_lookahead_parent_prunes','pair_lookahead_learned',
                'pair_lookahead_complete_passes','pair_lookahead_aborted',
                'pair_lookahead_propagations','pair_lookahead_arc_revisions')

class CompletedProbeCache:
    def __init__(self,limit=128,max_slots=32768):
        if type(limit)is not int or limit<0 or type(max_slots)is not int or max_slots<0:
            raise ValueError('Probe cache bounds must be nonnegative integers.')
        self.limit,self.max_slots=limit,max_slots
        self.entries=OrderedDict();self.slots=0

    def run(self,context,domains,selected,probe,check,stats):
        """Run or reuse a complete probe, publishing before a caller Stop poll.

        Older completed supports may be weaker after additional valid learning;
        they remain safe. Context includes the effective root, active decisions,
        and options, and the key retains every input orientation domain.
        """
        check();before=tuple(domains);key=(context,before)
        for name in CACHE_STAT_KEYS:stats.setdefault(name,0)
        cached=self.entries.get(key)
        if cached is not None:
            self.entries.move_to_end(key);stats['lookahead_cache_hits']+=1
            possible,narrowed,_=cached
            if possible:domains[:]=narrowed
            return possible
        # A completed consequence D => R also holds in any narrower D'.
        # Intersect; never replace the narrower caller with the older result.
        # Failure is monotone as well. Exact trusted context equality is still
        # mandatory: a proof in a restricted job/root cannot escape that scope.
        for (scope, source), (possible, narrowed, _) in reversed(self.entries.items()):
            if scope != context or len(source) != len(before):continue
            check()
            if any(now & ~old for now, old in zip(before, source)):continue
            stats['lookahead_subset_hits']+=1
            if not possible:return False
            current=[now & proved for now, proved in zip(before, narrowed)]
            stats['lookahead_implication_prunes']+=sum((a & ~b).bit_count() for a,b in zip(before,current))
            domains[:]=current
            # Reuse is a sound implication, NOT a fixpoint for the narrower
            # problem. Finish propagation there; otherwise pair look-ahead can
            # lose new contradictions that only the stronger input exposes.
            if not all(current):return False
            if current!=list(before):selected=None  # Several cells may change.
            break
        possible=probe(domains,selected)
        if type(possible)is not bool:raise ValueError('A probe must return a boolean.')
        narrowed=tuple(domains) if possible else ()
        if possible and (len(narrowed)!=len(before) or any(type(x)is not int or x<=0 or x&~old for x,old in zip(narrowed,before))):
            raise ValueError('A completed probe may only narrow positive input domains.')
        slots=len(before)+len(narrowed)
        if self.limit and slots<=self.max_slots:
            while self.entries and (len(self.entries)>=self.limit or self.slots+slots>self.max_slots):
                _,old=self.entries.popitem(last=False);self.slots-=old[2]
                stats['lookahead_cache_evictions']+=1
            self.entries[key]=(possible,narrowed,slots);self.slots+=slots
            stats['lookahead_cached_probes']+=1
        return possible


def values(mask):
    while mask:
        bit=mask&-mask;yield bit.bit_length()-1;mask^=bit


def paired_lookahead(domains,cell,choices,singleton_probe,pair_probe,check,on_failure,on_failed_first,stats,*,neighbors=(),max_pairs=16):
    """Prove selected pairs impossible; a full failed row can delete its first value.

    ``on_failure`` must retain the actual ancestor assumptions along with both
    placements. A rejected pair is not automatically an original-root theorem.
    Each second-cell row covers its whole propagated domain before it can prove
    the first placement impossible. A bounded/incomplete row never does so.
    """
    if type(max_pairs)is not int or max_pairs<0:raise ValueError('Pair probe limit must be nonnegative.')
    for name in PAIR_STAT_KEYS:stats.setdefault(name,0)
    before=tuple(domains);near=set(neighbors);used=0;changed=False
    stats['pair_lookahead_queries']+=1
    for code in choices:
        check()
        if not before[cell]&(1<<code):raise ValueError('Pair probe choice is outside its domain.')
        trial=list(before);trial[cell]=1<<code;stats['pair_lookahead_singletons']+=1
        if not singleton_probe(trial,cell):
            check();domains[cell]&=~(1<<code);stats['pair_lookahead_parent_prunes']+=1
            on_failed_first(code);changed=True;continue
        candidates=[other for other,value in enumerate(trial) if other!=cell and 2<=value.bit_count()<=4]
        if not candidates:continue
        other=min(candidates,key=lambda c:(c not in near,trial[c].bit_count(),c))
        survived=False
        for second in values(trial[other]):
            if used>=max_pairs:return changed
            check();used+=1;stats['pair_lookahead_probes']+=1
            candidate=trial.copy();candidate[other]=1<<second
            possible=pair_probe(candidate,other);check()
            if possible:survived=True
            else:
                stats['pair_lookahead_failures']+=1
                on_failure(cell,code,other,second)
        if not survived:
            domains[cell]&=~(1<<code);stats['pair_lookahead_parent_prunes']+=1
            on_failed_first(code);changed=True
    stats['pair_lookahead_complete_passes']+=1
    return changed


GROUP_STAT_KEYS=('group_lookahead_queries','group_lookahead_probes','group_lookahead_failures',
                 'group_lookahead_prunes','group_lookahead_complete','group_lookahead_aborted')


def grouped_constructive_lookahead(domains,cell,masks,probe,check,on_failure,stats):
    """Constructive disjunction over a COMPLETE partition of an arbitrary domain.

    Every surviving branch contributes its support union. Only after all groups
    finish can common deductions be published. An individually failed group is
    removed immediately with its full mask supplied to the scoped explanation
    callback. Exceptions discard incomplete unions, not completed exclusions.
    """
    before=tuple(domains);groups=tuple(masks)
    if (type(cell)is not int or not 0<=cell<len(before) or len(groups)<2
            or any(type(m)is not int or m<=0 or m&~before[cell] for m in groups)
            or sum(groups)!=before[cell]):
        raise ValueError('Grouped probes require a complete disjoint choice partition.')
    union=0
    for mask in groups:
        if union&mask:raise ValueError('Grouped probes require a complete disjoint choice partition.')
        union|=mask
    for key in GROUP_STAT_KEYS:stats.setdefault(key,0)
    stats['group_lookahead_queries']+=1;supports=[0]*len(before);changed=False
    try:
        for mask in groups:
            check();trial=list(before);trial[cell]=mask;stats['group_lookahead_probes']+=1
            possible=probe(trial,cell)
            if type(possible)is not bool:raise ValueError('Grouped probe must return a boolean.')
            if not possible:
                removed=domains[cell]&mask;domains[cell]&=~mask
                stats['group_lookahead_prunes']+=removed.bit_count()
                stats['group_lookahead_failures']+=1;changed=changed or bool(removed)
                on_failure(mask)
            else:
                if (len(trial)!=len(before) or any(type(value)is not int or value<=0 or value&~old
                                                  for value,old in zip(trial,before))
                        or trial[cell]&~mask):
                    raise ValueError('Grouped propagator returned invalid domains.')
                for c,value in enumerate(trial):supports[c]|=value
        check()
    except BaseException:
        stats['group_lookahead_aborted']+=1;raise
    for c,value in enumerate(supports):
        old=domains[c];domains[c]&=value
        stats['group_lookahead_prunes']+=(old&~domains[c]).bit_count()
        changed=changed or old!=domains[c]
    stats['group_lookahead_complete']+=1
    return changed
