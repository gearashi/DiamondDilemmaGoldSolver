"""Forward-check rigid assembly placement using occupied cells and relaxed physical resources."""
from collections import Counter

import numpy as np

from constraint_advanced import _complete_matching


def path_cycle_assignment(scores,legal,stats):
    """Choose the maximum-weight perfect matching when all bipartite degrees are at most two."""
    from collections import deque
    rows=[set(np.flatnonzero(choices).tolist()) for choices in legal]
    cols=[set(np.flatnonzero(choices).tolist()) for choices in legal.T]
    if any(not a for a in rows+cols):raise ValueError('Assignment has no legal completion.')
    result=np.full(len(rows),-1,dtype=int)
    used=set()
    pending=deque([(True,i) for i,x in enumerate(rows) if len(x)==1]+
                  [(False,i) for i,x in enumerate(cols) if len(x)==1])
    while pending:
        is_row,i=pending.popleft()
        if (is_row and result[i]>=0) or (not is_row and i in used):continue
        choices=rows[i] if is_row else cols[i]
        if not choices:raise ValueError('Assignment has no legal completion.')
        j=next(iter(choices))
        r,c=(i,j) if is_row else (j,i)
        result[r]=c
        used.add(c)
        stats['forced_weighted_rows']+=1
        for k in rows[r]-{c}:
            cols[k].remove(r)
            pending.append((False,k))
        for k in cols[c]-{r}:
            rows[k].remove(c)
            pending.append((True,k))
    for start in range(len(rows)):
        if result[start]>=0:continue
        options=[]
        for first in sorted(rows[start]):
            r,c=start,first
            chosen={}
            while r not in chosen:
                chosen[r]=c
                other,=cols[c]-{r}
                r=other
                other,=rows[r]-{c}
                c=other
            if r!=start:raise ArithmeticError('Unpeeled component is not an alternating cycle.')
            options.append((sum(int(scores[i,j]) for i,j in chosen.items()),chosen))
        if len(options)!=2:raise ArithmeticError('An unpeeled row must have two alternatives.')
        winner=max(options,key=lambda item:item[0])[1]
        for i,j in winner.items():result[i]=j
        stats['weighted_cycle_components']+=1
        stats['weighted_cycle_rows']+=len(winner)
    return result


def weighted_donors(weights, allowed, stats, *, compact):
    """Solve the fully weighted objective after fixing simultaneously extendible hints."""
    from r27_incumbent import assignment
    scores=weights.max(axis=2)
    legal=allowed.any(axis=2)
    if compact and len(scores)>1 and legal.sum(axis=1).max()<=2 and legal.sum(axis=0).max()<=2:
        stats['path_cycle_assignments']+=1
        return path_cycle_assignment(scores,legal,stats)
    if compact:
        # The constructor's hint bonus exceeds all random tie weights combined.
        # When all surviving original hints can coexist, every optimum keeps them.
        keep,donors=np.nonzero(scores>7)
        if len(keep):
            unused=np.ones(len(scores),dtype=bool)
            unused[donors]=False
            cells=np.ones(len(scores),dtype=bool)
            cells[keep]=False
            free=np.flatnonzero(cells)
            tiles=np.flatnonzero(unused)
            if len(free)!=len(tiles):
                raise ValueError('Hint placements must use unique physical pieces.')
            try:
                sub=np.ix_(free,tiles)
                if len(free) and (not legal[sub].any(axis=1).all() or not legal[sub].any(axis=0).all()):
                    raise ValueError('Surviving hints leave an isolated residual slot or tile.')
                stats['weighted_assignment_rows']+=len(free)
                result=np.empty(len(scores),dtype=int)
                result[keep]=donors
                if len(free):
                    result[free]=tiles[assignment(scores[sub],legal[sub])]
                stats['fixed_hint_rows']+=len(keep)
                return result
            except ValueError:
                stats['compact_fallbacks']+=1
    stats['weighted_assignment_rows']+=len(scores)
    return assignment(scores,legal)


def bits(mask):
    while mask:
        bit=mask & -mask
        yield bit.bit_length()-1
        mask-=bit


class PoseConstructor:
    def __init__(self,assemblies,poses):
        self.groups=tuple((frozenset(tiles),()) for tiles,_ in assemblies.groups)
        self.poses=poses
        self.model=assemblies.model
        self.full=(1<<self.model.n)-1
        self.occupies=[]
        self.incident=[]
        self.max_per_cell=[]
        for group,rows in zip(self.groups,poses):
            masks=[]
            occupied={}
            occurrences=Counter()
            for i,row in enumerate(rows):
                masks.append(sum(1<<int(c) for c,_ in row))
                for c,code in row:
                    occupied[c]=occupied.get(c,0)|(1<<i)
                    occurrences[code//3,c]+=1
            self.max_per_cell.append(max(occurrences.values(),default=1))
            self.occupies.append(masks)
            self.incident.append(occupied)
        self.ordinary=sorted(set(range(self.model.n))-assemblies.tiles)
        domains=self.model.legal.reshape(self.model.n,self.model.n,3).any(axis=2)
        self.ordinary_masks=[sum(1<<int(c) for c in np.flatnonzero(domains[:,t])) for t in self.ordinary]

    @classmethod
    def for_assemblies(cls,assemblies,stats):
        # Pose ordering depends on the seed. Its geometry and hard domains do
        # not, so retain one index across different starts on the same instance.
        source=tuple((frozenset(tiles),tuple(poses)) for tiles,poses in assemblies.groups)
        domains=np.packbits(assemblies.model.legal).tobytes()
        previous=getattr(assemblies,'_constructor_index',None)
        if previous is not None:
            old_source,old_domains,index=previous
            if index.model is assemblies.model and old_source==source and old_domains==domains:
                stats['geometry_pose_rows_reused']+=sum(len(rows) for tiles,rows in source)
                return index
        index=cls(assemblies,[rows for tiles,rows in source])
        assemblies._constructor_index=source,domains,index
        stats['geometry_pose_rows_indexed']+=sum(len(rows) for tiles,rows in source)
        return index

    def feasible(self,eligible,remaining,occupied,policy,stats):
        if policy=='forward':
            return True
        free=self.full^occupied
        size=sum(len(self.groups[i][0]) for i in remaining)
        # Each group's possible slots per tile are at least the eligible pose
        # count divided by that group's maximum per-(tile,cell) multiplicity.
        # If every rigid tile has at least size slots and every ordinary tile
        # has all free slots, their relaxed matching always exists.
        if all(mask&free==free for mask in self.ordinary_masks) and all(
                (eligible[i].bit_count()+self.max_per_cell[i]-1)//self.max_per_cell[i]>=size for i in remaining):
            stats['dense_matching_certificates']+=1
            return True
        rows=[mask&free for mask in self.ordinary_masks]
        for i in remaining:
            locations={t:0 for t in self.groups[i][0]}
            for index in bits(eligible[i]):
                for c,code in self.poses[i][index]:
                    locations[code//3] |= 1<<int(c)
            rows.extend(locations.values())
        counts=Counter(rows)
        if any(count>mask.bit_count() for mask,count in counts.items()):
            stats['identical_hall_rejections']+=1
            return False
        # Ordinary tiles sharing a cell pool can fill its unused slots freely.
        # If each remaining rigid tile stays in one disjoint pool and has at
        # least as many available slots as its rigid peers, greedy matching is
        # sufficient inside that pool. This also handles completed assemblies.
        if policy=='matching':
            ordinary=Counter(rows[:len(self.ordinary_masks)])
            covered=0
            for pool in ordinary:
                if covered&pool:break
                covered|=pool
            else:
                pools=list(ordinary)
                if free&~covered:pools.append(free&~covered)
                peers={pool:[] for pool in pools}
                for mask in rows[len(self.ordinary_masks):]:
                    owner=next((pool for pool in pools if not mask&~pool),None)
                    if owner is None:break
                    peers[owner].append(mask.bit_count())
                else:
                    if any(len(sizes)+ordinary[pool]>pool.bit_count() for pool,sizes in peers.items()):
                        stats['disjoint_pool_rejections']+=1
                        return False
                    if all(not sizes or min(sizes)>=len(sizes) for sizes in peers.values()):
                        stats['disjoint_pool_certificates']+=1
                        return True
        small=[mask for mask in counts if mask.bit_count()<=4]
        unions={a|b for pos,a in enumerate(small) for b in small[:pos] if (a|b).bit_count()<=4}
        if any(sum(n for mask,n in counts.items() if not mask&~union)>union.bit_count() for union in unions):
            stats['subset_hall_rejections']+=1
            return False
        if policy=='matching':
            stats['residual_matching_calls']+=1
            if _complete_matching(rows) is None:
                stats['residual_matching_rejections']+=1
                return False
        return True

    def find(self,complete,budget,policy,stats,preferred=None):
        if policy not in ('forward','hall','matching'):
            raise ValueError('Unknown assembly constructor policy.')
        def place(eligible,remaining,occupied,fixed):
            if budget is not None and stats['pose_nodes']>=budget:
                stats['budget_stops']+=1
                return None
            if not remaining:
                if not self.feasible(eligible,remaining,occupied,policy,stats):
                    return None
                stats['leaf_assignments']+=1
                return complete(fixed)
            group=min(remaining,key=lambda i:(eligible[i].bit_count(),i))
            if not eligible[group]:
                stats['empty_pose_domains']+=1
                return None
            if not self.feasible(eligible,remaining,occupied,policy,stats):
                return None
            rest=tuple(i for i in remaining if i!=group)
            if preferred is None:
                available=bits(eligible[group])
            else:
                mask=eligible[group]
                available=(index for index in preferred[group] if mask&(1<<index))
            for index in available:
                if budget is not None and stats['pose_nodes']>=budget:
                    stats['budget_stops']+=1
                    return None
                next_eligible=list(eligible)
                pose=self.poses[group][index]
                stats['pose_nodes']+=1
                stats['pose_rows_examined']+=1
                for i in rest:
                    forbidden=0
                    for c,_ in pose:
                        forbidden|=self.incident[i].get(c,0)
                    next_eligible[i]&=~forbidden
                winner=place(next_eligible,rest,occupied|self.occupies[group][index],fixed|dict(pose))
                if winner is not None:
                    return winner
            return None
        eligible=[(1<<len(rows))-1 for rows in self.poses]
        return place(eligible,tuple(range(len(eligible))),0,{})
