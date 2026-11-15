"""Exact unit-cost hitting sets for large conditional physical-bond obligations."""
from functools import lru_cache


def reduced_cover(groups):
    """Return one minimum set of physical labels, proving optimal cost by complete branching."""
    rows=[]
    for row in sorted(set(map(frozenset,groups)),key=lambda r:(len(r),sorted(r))):
        if not row:raise ValueError('A departure obligation has no possible lost bond.')
        if not any(earlier<=row for earlier in rows):rows.append(row)
    labels=sorted(set().union(*rows))
    columns={label:0 for label in labels}
    for i,row in enumerate(rows):
        for label in row:columns[label]|=1<<i
    retained=[]
    # A unit-cost superset covers everything its dominated column could cover.
    for label,bits in sorted(columns.items(),key=lambda item:(-item[1].bit_count(),item[0])):
        if not any(bits&~previous==0 for previous,_ in retained):
            retained.append((bits,label))
    choices=[]
    for i in range(len(rows)):
        choices.append(tuple(j for j,(col,_) in enumerate(retained) if (col>>i)&1))
    if not rows:return frozenset()
    all_rows=(1<<len(rows))-1
    row_order=sorted(range(len(rows)),key=lambda i:(len(choices[i]),-sum(retained[j][0].bit_count() for j in choices[i])))
    touch=[0]*len(rows)
    for i,values in enumerate(choices):
        for choice in values:touch[i]|=retained[choice][0]
    @lru_cache(maxsize=16384)
    def lower(mask):
        # Pairwise column-disjoint obligations each need a separate choice.
        taken=mask;bound=0
        for i in row_order:
            if (taken>>i)&1:taken&=~touch[i];bound+=1
        return bound
    def greedy(mask):
        answer=[]
        while mask:
            chosen=max(range(len(retained)),key=lambda j:(retained[j][0]&mask).bit_count())
            answer.append(chosen);mask&=~retained[chosen][0]
        return answer
    best=greedy(all_rows)
    reached={}
    def search(mask, selected):
        nonlocal best
        if not mask:
            if len(selected)<len(best):best=list(selected)
            return
        if len(selected)+lower(mask)>=len(best) or reached.get(mask,len(best))<=len(selected):return
        if len(reached)>=131072:reached.clear()
        reached[mask]=len(selected)
        i=next(i for i in row_order if mask>>i&1)
        ranked=sorted({(retained[j][0]&mask,j) for j in choices[i]},key=lambda t:(-t[0].bit_count(),t[1]))
        seen=[]
        for hit,j in ranked:
            if any(hit&~old==0 for old in seen):continue
            seen.append(hit)
            search(mask&~hit,selected+[j])
            if len(selected)+lower(mask)>=len(best):return
    search(all_rows,[])
    return frozenset(retained[j][1] for j in best)
