"""Cheap exact support for dense conditional all-different instances."""


def dense_support(rows):
    """Return exact supported tile bits, or None when ordinary matching is needed.

    After fixing singleton owners, choose any remaining row and any of its
    tiles. Each other row loses at most that chosen tile. If sorted row sizes
    pass the bound below, greedily assigning those rows always extends the
    chosen edge to an injection. The argument also holds with extra tile IDs.
    """
    fixed = 0
    free = []
    for index, row in enumerate(rows):
        if row <= 0:
            return None
        if row & (row - 1):
            free.append(index)
        elif fixed & row:
            return None
        else:
            fixed |= row

    sizes = []
    narrowed = None
    for index in free:
        row = rows[index]
        available = row & ~fixed
        if available != row:
            # New singleton chains or contradictions go through the ordinary
            # matcher, which also retains its own checked Hall explanations.
            if available.bit_count() <= 1:
                return None
            if narrowed is None:
                narrowed = list(rows)
            narrowed[index] = available
        sizes.append(available.bit_count())
    sizes.sort()
    if any(size < index + 2 for index, size in enumerate(sizes[:-1])):
        return None
    return tuple(rows if narrowed is None else narrowed)


class CutInventoryCache(dict):
    """Conditional-cut memo that can supply its own exact value on a miss.

    AdvancedPruner queries membership and then reads completed values. Its
    existing cache stores at most 128 row tuples and compares full keys.
    A failed greedy check leaves its matching and support budgets untouched.
    """
    def __init__(self):
        super().__init__()
        self.certificates = 0
        self._last_miss = None

    def __contains__(self, rows):
        if super().__contains__(rows):
            return True
        # _put() repeats the membership check after the ordinary matcher.
        if rows is self._last_miss or len(rows) > 160:
            return False
        supported = dense_support(rows)
        if supported is None:
            self._last_miss = rows
            return False
        # The caller checks cancellation before the lookup. There are at most
        # two passes over the 160 cells and a small sort before it checks again.
        if len(self) >= 128:
            del self[next(iter(self))]
        self[rows] = supported
        self._last_miss = None
        self.certificates += 1
        return True
