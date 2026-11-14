"""Batch half-join rows with identical responses to one fixed external exchange."""
from collections import defaultdict


def group_rows(join, left, rights):
    """Group by exact gain and donors across the interface; rights are already disjoint and owned."""
    moved = {c for cycle in left for c in cycle}
    boundary = set().union(*(join.near[c] for c in moved))-moved
    groups = defaultdict(list)
    for gain,bits,cycles in rights:
        response = tuple(sorted((c,donor) for cycle in cycles for c,donor in join.cycles[cycle].items()
                                if c in boundary))
        groups[gain,response].append((gain,bits,cycles))
    return groups.values()
