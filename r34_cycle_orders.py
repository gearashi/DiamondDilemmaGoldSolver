"""Direct iteration and random access for an exact permutation cycle family."""
from collections import Counter
from functools import lru_cache
from itertools import permutations
from math import factorial, perm, prod
from operator import index


FIVE_FAMILIES = ((6,), (2,5), (3,4), (2,2,4), (2,3,3), (2,2,2,3), (2,2,2,2,2))


@lru_cache(maxsize=64)
def family_count(family):
    divisor = prod(size**copies * factorial(copies) for size,copies in Counter(family).items())
    return factorial(sum(family)) // divisor


def blocks(cells, family):
    for length in sorted(set(family)):
        rest = list(family)
        rest.remove(length)
        rest = tuple(rest)
        suffix = family_count(rest)
        yield length, rest, suffix, perm(len(cells)-1, length-1)*suffix


class CycleOrders:
    """The least unused destination owns its cycle. Equal-length cycles are unordered."""
    def __init__(self, family):
        self.family = tuple(sorted(family))
        if not self.family or any(type(length) is not int or length < 2 for length in self.family):
            raise ValueError("Expected nontrivial cycle lengths.")
        self.cells = tuple(range(sum(self.family)))

    def __len__(self):
        return family_count(self.family)

    def __iter__(self):
        order = list(self.cells)
        def visit(cells, family):
            if not cells:
                yield tuple(order)
                return
            for length, rest, _, _ in blocks(cells, family):
                for tail in permutations(cells[1:], length-1):
                    cycle = (cells[0], *tail)
                    for a,b in zip(cycle, (*tail, cells[0])):
                        order[a] = b
                    yield from visit(tuple(c for c in cells[1:] if c not in tail), rest)
        yield from visit(self.cells, self.family)

    def __getitem__(self, rank):
        rank = index(rank)
        if rank < 0:
            rank += len(self)
        if not 0 <= rank < len(self):
            raise IndexError("Cycle family rank out of range.")
        cells = self.cells
        order = list(cells)
        family = self.family
        while cells:
            for length, rest, suffix, count in blocks(cells, family):
                if rank < count:
                    break
                rank -= count
            tail_rank, rank = divmod(rank, suffix)
            unused = list(cells[1:])
            current = cells[0]
            for left in reversed(range(length-1)):
                digit, tail_rank = divmod(tail_rank, perm(len(unused)-1, left))
                following = unused.pop(digit)
                order[current] = following
                current = following
            order[current] = cells[0]
            cells, family = tuple(unused), rest
        return tuple(order)

    def rank(self, order):
        order = tuple(order)
        if len(order) != len(self.cells) or set(order) != set(self.cells):
            raise ValueError("Expected a permutation of the support.")
        cells, family, result = self.cells, self.family, 0
        while cells:
            tail = []
            current = order[cells[0]]
            while current != cells[0]:
                tail.append(current)
                current = order[current]
            for length, rest, suffix, count in blocks(cells, family):
                if length == len(tail)+1:
                    break
                result += count
            else:
                raise ValueError("Permutation has a different cycle family.")
            unused = list(cells[1:])
            for i,cell in enumerate(tail):
                digit = unused.index(cell)
                result += digit * perm(len(unused)-1, len(tail)-i-1) * suffix
                unused.pop(digit)
            cells, family = tuple(unused), rest
        return result
