"""Bounded matching and forest oracle for a single propagated DFS frame."""
import time

from r27_reason_maxsat import Budget
from r27_reason_search import ScopedSearch


class TailSearch(ScopedSearch):
    def __init__(self,problem,rows,target,max_checks,deadline,check):
        super().__init__(problem,rows,target,max_checks,deadline)
        self.check = check
        self.neighbors = [[] for _ in rows]
        for c,d in self.sides:
            self.neighbors[c].append(d)

    def tick(self):
        super().tick()
        if self.stats['work'] % 64 == 0:
            self.check()

    def reduced(self,rows):
        """Use physical owners and incident seams directly on mostly fixed boards."""
        rows = list(rows)
        while True:
            self.tick()
            fixed = {c:row[0] for c,row in enumerate(rows) if len(row)==1}
            if any(not row for row in rows):
                return None
            used = {x//3 for x in fixed.values()}
            if len(used) != len(fixed):
                return None
            changed = False
            for cell,row in enumerate(rows):
                self.tick()
                if cell in fixed:
                    if any(d<cell and d in fixed and not self.compatible(cell,row[0],d,fixed[d])
                           for d in self.neighbors[cell]):
                        return None
                    continue
                adjacent = [d for d in self.neighbors[cell] if d in fixed]
                keep = tuple(x for x in row if x//3 not in used and
                             all(self.compatible(cell,x,d,fixed[d]) for d in adjacent))
                if not keep:
                    return None
                rows[cell] = keep
                changed |= len(keep)==1
            if not changed:
                return rows


def bounded_tail(problem,masks,target,check,*,max_active=8,max_alternatives=64,max_checks=8000,seconds=.04):
    """Only completion certifies an answer; any limit leaves the caller's exact frame intact."""
    check()
    widths = [m.bit_count() for m in masks]
    active = sum(n>1 for n in widths)
    if active > max_active and widths and min(widths) > 0 and max(widths) <= 2:
        from r28_logic_tail import binary_tail
        return binary_tail(problem,masks,target,check,max_work=max_checks,seconds=seconds)
    if (not all(widths) or not 1 <= active <= max_active
            or sum(n-1 for n in widths)>max_alternatives):
        return None
    rows = []
    for mask in masks:
        row = []
        while mask:
            bit = mask & -mask
            row.append(bit.bit_length()-1)
            mask ^= bit
        rows.append(tuple(row))
    machine = TailSearch(problem,rows,target,max_checks,time.monotonic()+seconds,check)
    try:
        board = machine.explore(rows,'backdoor')
        status = 'infeasible' if board is None else 'witness'
    except Budget:
        board,status = None,'unknown'
    check()
    return status,board,machine.stats['work']

