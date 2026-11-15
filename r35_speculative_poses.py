"""Join rigid pose tables on physical darts and identify proved optional packing conflicts."""
from collections import Counter


class PoseJoins:
    def __init__(self,assemblies):
        self.model=assemblies.model
        self.base=[(frozenset(group),rows,()) for group,rows in assemblies.groups]
        self.cache={}
        self.singletons={}
        self.work=Counter()

    def singleton_rows(self,tile):
        if tile not in self.singletons:
            self.singletons[tile]=[((c,code),) for c in range(self.model.n)
                                   for code in range(3*tile,3*tile+3) if self.model.legal[c,code]]
        return self.singletons[tile]

    def joined(self,left,right,edge):
        a,b=edge
        if a//3 not in left[0]:
            left,right=right,left
        if a//3 not in left[0] or b//3 not in right[0]:
            raise ValueError('A rigid join must cross the specified components.')
        # An ungrouped donor has exactly one candidate at a requested anchor.
        # Keep it on the indexed side and look it up directly in physical domains.
        if left[1] is None and right[1] is not None:
            left,right=right,left
            a,b=b,a
        members=left[0]|right[0]
        history=tuple(sorted((*left[2],*right[2],edge)))
        key=(members,history)
        if key in self.cache:
            self.work['joined_tables_reused']+=1
            return members,self.cache[key],history
        table={}
        for pose in right[1] or ():
            anchor=next((c,v%3) for c,v in pose if v//3==b//3)
            table.setdefault(anchor,[]).append((pose,frozenset(c for c,v in pose)))
        result=[]
        left_rows=self.singleton_rows(a//3) if left[1] is None else left[1]
        for p in left_rows:
            c,r=next((c,v%3) for c,v in p if v//3==a//3)
            dest,side=self.model.problem.board.neighbors[c][(a%3+r)%3]
            self.work['join_probes']+=1
            if dest<0:continue
            slots=frozenset(i for i,v in p)
            rotation=(side-b%3)%3
            if right[1] is None:
                code=3*(b//3)+rotation
                candidates=[(((dest,code),),{dest})] if self.model.legal[dest,code] else ()
            else:
                candidates=table.get((dest,rotation),())
            for q,other in candidates:
                if slots.isdisjoint(other):
                    result.append(tuple(sorted((*p,*q),key=lambda row:row[1]//3)))
        result.sort(key=lambda pose:(pose[0][0],pose[0][1]%3))
        self.cache[key]=result
        self.work['joined_tables_built']+=1
        self.work['right_poses_indexed']+=len(right[1] or ())
        self.work['joined_rows_emitted']+=len(result)
        return members,result,history

    def append(self,groups,edge):
        a,b=edge
        ix=[next((i for i,g in enumerate(groups) if tile in g[0]),None) for tile in (a//3,b//3)]
        if ix[0] is not None and ix[0]==ix[1]:return None
        left,right=((frozenset((tile,)),None,()) if i is None else groups[i] for i,tile in zip(ix,(a//3,b//3)))
        if len(left[0])+len(right[0])>9:return None
        group=self.joined(left,right,edge)
        if not group[1]:return None
        return [g for i,g in enumerate(groups) if i not in ix]+[group]

    def replay(self,edges):
        groups=list(self.base)
        for edge in edges:
            new=self.append(groups,edge)
            if new is None:raise ValueError('A surviving optional component lost its previously legal poses.')
            groups=new
        return groups


def conflicting_groups(groups,budget,stats):
    """Return a proved incompatible group subset. A stopped search has no exclusion."""
    footprints=[tuple(sorted({sum(1<<int(c) for c,v in pose) for pose in rows})) for _,rows,_ in groups]
    consumed=0
    def pack(ids,occupied):
        nonlocal consumed
        if not ids:return True
        choices=[(i,[s for s in footprints[i] if not s&occupied]) for i in ids]
        i,rows=min(choices,key=lambda pair:len(pair[1]))
        if not rows:return False
        rest=tuple(j for j in ids if i!=j)
        for row in rows:
            if consumed>=budget:return None
            consumed+=1
            result=pack(rest,occupied|row)
            if result is not False:return result
        return False
    core=tuple(range(len(groups)))
    result=pack(core,0)
    if result is False:
        for i in core:
            trial=tuple(j for j in core if j!=i)
            if pack(trial,0) is False:core=trial
        stats['packing_cores']+=1
    elif result is None:
        stats['unproved_packing_budgets']+=1
    stats['proof_pose_nodes']+=consumed
    return core if result is False else ()


def rollback_index(extra,groups,budget,stats):
    core=conflicting_groups(groups,budget,stats)
    culprit={edge for i in core for edge in groups[i][2]}
    if culprit:
        index=max(i for i,edge in enumerate(extra) if edge in culprit)
        stats['later_choices_preserved']+=len(extra)-index-1
        return index
    return len(extra)-1
