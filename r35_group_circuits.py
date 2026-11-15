"""Higher-rank grouped phase decisions by disjoint branches on checked XOR contradictions."""

from math import inf
from time import monotonic


class GroupCircuits:
    def __init__(self, query):
        self.group_masks = []
        self.free_to_drop = 0
        self.rows = []
        for group in query.groups:
            members = 0
            for bond in sorted(group):
                bit = 1 << len(self.rows)
                members |= bit
                if bond not in query.required:
                    self.free_to_drop |= bit
                self.rows.append(query._joint_bonds[bond])
            self.group_masks.append(members)

    def solve(self, flipped, budget, deadline, work):
        errors = 1 | sum(2 << i for i in flipped)
        rows = [row | ((symbol & errors).bit_count() & 1) for row, symbol in self.rows]
        limit = budget - len(flipped)
        all_rows = (1 << len(rows)) - 1
        def first_conflict(kept):
            basis = {}
            while kept:
                bit = kept & -kept
                kept ^= bit
                row = rows[bit.bit_length()-1]
                support = bit
                work['insertions'] += 1
                while row > 1:
                    pivot = row.bit_length()-1
                    if pivot not in basis:
                        basis[pivot] = row,support
                        break
                    other, mask = basis[pivot]
                    row ^= other
                    support ^= mask
                else:
                    if row:
                        check = 0
                        witness = support
                        while witness:
                            unit = witness & -witness
                            check ^= rows[unit.bit_length()-1]
                            witness ^= unit
                        if check != 1:
                            raise ArithmeticError('Invalid grouped parity contradiction')
                        return support
            return 0

        def visit(dropped, allowed):
            if monotonic() >= deadline:
                raise TimeoutError
            work['nodes'] += 1
            missing = [group for group in self.group_masks if not dropped & group]
            if len(missing) + dropped.bit_count() > limit or any(not (group & allowed) for group in missing):
                return False
            conflict = first_conflict(all_rows ^ dropped)
            if not conflict:
                return True
            work['circuits'] += 1
            choices = conflict & allowed
            while choices:
                choice = choices & -choices
                choices ^= choice
                allowed ^= choice
                if visit(dropped | choice, allowed):
                    return True
            return False
        return visit(0,self.free_to_drop)
