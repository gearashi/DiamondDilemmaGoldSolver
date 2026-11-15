"""Disjoint physical-placement refinements for interrupted geographic SAT boxes."""


def region(node):
    # Older checkpoints and unsplit nodes store the geographic box directly.
    return node['box'] if isinstance(node, dict) else node


def intervals(rows, pins):
    bounds = [(0, len(row)) for row in rows]
    seen = set()
    for cell, lo, hi in pins:
        if (type(cell) is not int or not 0 <= cell < len(rows) or cell in seen
                or type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= len(rows[cell])):
            raise ValueError('Invalid geographic placement interval in checkpoint.')
        seen.add(cell)
        bounds[cell] = (lo, hi)
    return bounds


def assumptions(rows, variables, pins):
    literals = []
    for cell, lo, hi in pins:
        row = rows[cell]
        if hi - lo == 1:
            literals.append(variables[cell][row[lo]])
        else:
            literals.extend(-variables[cell][code] for code in row[:lo] + row[hi:])
    return literals


def split_placement(box, bounds, pins):
    options = ((hi - lo, cell, lo, hi) for cell, (lo, hi) in enumerate(bounds) if hi - lo > 1)
    choice = min(options, default=None)
    if choice is None:
        return None
    _, cell, lo, hi = choice
    middle = (lo + hi) // 2
    other = [pin for pin in pins if pin[0] != cell]
    # Exactly one orientation belongs to the cell. The two contiguous slices
    # partition its remaining choices and are saved atomically in the DFS stack.
    return [dict(box=box, pins=other + [[cell, middle, hi]]),
            dict(box=box, pins=other + [[cell, lo, middle]])]
