"""Exact group-drop fallback for reference-loss queries with a large phase image."""
from itertools import combinations
from math import comb
from time import monotonic

from r32_gold_partition import insert


class GroupDropTimeout(TimeoutError):
    def __init__(self, path):
        self.path = list(path)


class _InvalidPath(ValueError):
    pass


def reuse_drops(query, flipped, budget, deadline, work, resume=(), checkpoint=None):
    """Reuse root one-drop bases; an optional cursor names one unfinished branch."""
    discriminator = 1 | sum(2 << i for i in flipped)
    rows = {p: keep | ((rhs & discriminator).bit_count() & 1)
            for p,(keep,rhs) in query._joint_bonds.items()}
    options = []
    for group in query.groups:
        if monotonic() >= deadline:
            raise GroupDropTimeout(resume)
        bonds = sorted(group)
        removable = sorted(group-query.required)
        if not removable:
            return False
        singles = {}
        prefix = {}
        for j, bond in enumerate(bonds):
            if bond not in query.required:
                child = prefix.copy()
                if all(insert(child,rows[b]) for b in bonds[j+1:]):
                    singles[bond] = child
            if not insert(prefix,rows[bond]):
                break
        options.append((bonds,removable,singles))
    options.sort(key=lambda option:len(option[2]))
    floors = [0]*(len(options)+1)
    for i in range(len(options)-1,-1,-1):
        floors[i] = floors[i+1]+(1 if options[i][2] else 2)

    available = budget-len(flipped)
    def solve(base, i, left, path, rest):
        work['nodes'] += 1
        if left < floors[i] or i == len(options):
            if rest:
                raise _InvalidPath
            return left >= floors[i]
        bonds,removable,singles = options[i]
        skip = rest[0] if rest else 0
        ordinal = 0
        def inspect():
            current = path + (ordinal,)
            if monotonic() >= deadline:
                raise GroupDropTimeout(current + tuple(rest[1:]) if ordinal==skip else current)
            if checkpoint is not None:
                checkpoint(current + tuple(rest[1:]) if ordinal==skip else current)
        for initial in singles.values():
            if ordinal >= skip:
                inspect()
                child = initial
                valid = True
                if base:
                    child = base.copy()
                    valid = all(insert(child,row) for row in initial.values())
                child_resume = rest[1:] if ordinal==skip else ()
                if valid and solve(child,i+1,left-1,path+(ordinal,),child_resume):
                    return True
                if not valid and child_resume:
                    raise _InvalidPath
            ordinal += 1
        for size in range(2,min(len(removable),left-floors[i+1])+1):
            total = comb(len(removable),size)
            if ordinal + total <= skip:
                ordinal += total
                continue
            if ordinal < skip:
                from r32_gold_syndrome import ranked_combinations
                ranked = ranked_combinations(len(removable),size,skip-ordinal,total)
                choices = (tuple(removable[j] for j in indices) for _,indices in ranked)
                ordinal = skip
            else:
                choices = combinations(removable,size)
            for omitted in choices:
                if ordinal >= skip:
                    inspect()
                    work['multiple_drop_subsets'] += 1
                    child = base.copy()
                    valid = all(insert(child,rows[b]) for b in bonds if b not in omitted)
                    child_resume = rest[1:] if ordinal==skip else ()
                    if valid and solve(child,i+1,left-size,path+(ordinal,),child_resume):
                        return True
                    if not valid and child_resume:
                        raise _InvalidPath
                ordinal += 1
        if rest and ordinal <= skip:
            raise _InvalidPath
        return False

    if (not isinstance(resume,(tuple,list)) or len(resume)>len(options)
            or any(type(i) is not int or i<0 for i in resume)):
        resume = ()
    try:
        return solve({},0,available,(),resume)
    except _InvalidPath:
        return solve({},0,available,(),())
