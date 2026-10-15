"""Disjoint branches over a seam endpoint pattern, retaining every orientation.

A choice is a nonempty bitset of orientation codes for one cell. Groups exactly
partition its current domain. No edge pattern is a new puzzle assumption: each
alternative remains pending until searched or independently proved impossible.
"""
from dataclasses import dataclass

STAT_KEYS=('connection_branch_checks','connection_branches','connection_choices','connection_grouped_values')

@dataclass(frozen=True)
class ConnectionBranch:
    cell:int
    side:int
    patterns:tuple
    masks:tuple

class ConnectionBranching:
    def __init__(self,problem):
        self.n=problem.n
        self.groups=[]
        for side in range(3):
            row={}
            for code,masks in enumerate(problem.masks):
                pattern=int(masks[side]);row[pattern]=row.get(pattern,0)|(1<<code)
            self.groups.append(tuple(sorted(row.items())))
        self.neighbors=tuple(tuple(int(v) for v in row) for row in problem.neighbors)

    def choose(self,domains,preferred,check,stats,*,max_cells=12,max_groups=16):
        # A bounded pool favors tight cells but also permits a nearby boundary
        # to discriminate when all choices of the MRV cell have one pattern.
        candidates=sorted((i for i,v in enumerate(domains) if v.bit_count()>2),
                          key=lambda i:(i!=preferred,domains[i].bit_count(),i))[:max_cells]
        best=None
        for cell in candidates:
            check();domain=domains[cell];size=domain.bit_count()
            for side,neighbor in enumerate(self.neighbors[cell]):
                if neighbor<0:continue
                stats['connection_branch_checks']+=1
                groups=tuple((pattern,bits&domain) for pattern,bits in self.groups[side] if bits&domain)
                if not 2<=len(groups)<=max_groups or len(groups)>=size:continue
                # Balanced partitions avoid a near-vacuous first alternative.
                largest=max(bits.bit_count() for _,bits in groups)
                if largest*8>size*7:continue
                key=(largest/size,len(groups),domains[neighbor].bit_count(),cell!=preferred,cell,side)
                if best is None or key<best[0]:best=(key,ConnectionBranch(cell,side,tuple(p for p,_ in groups),tuple(b for _,b in groups)))
        if best is None:return None
        plan=best[1]
        assert sum(plan.masks)==domains[plan.cell] and all(mask and mask!=domains[plan.cell] for mask in plan.masks)
        stats['connection_branches']+=1;stats['connection_choices']+=len(plan.masks)
        stats['connection_grouped_values']+=domains[plan.cell].bit_count()-len(plan.masks)
        return plan

def branch_masks(frame):
    return frame.get('choice_masks') or [1<<code for code in frame['choices']]

def branch_mask(frame,index):
    masks=frame.get('choice_masks')
    return masks[index] if masks is not None else 1<<frame['choices'][index]

def from_native_frame(frame):
    masks=frame['choices'];result={'domains':frame['domains'].copy(),'cell':frame['cell'],
        'choices':[(value&-value).bit_length()-1 for value in masks],'next':frame['next']}
    if any(value.bit_count()!=1 for value in masks):result['choice_masks']=masks.copy()
    return result

def to_native_frame(frame):
    return {'domains':frame['domains'].copy(),'cell':frame['cell'],'choices':list(branch_masks(frame)),'next':frame['next']}
