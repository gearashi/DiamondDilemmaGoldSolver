"""Use verified weighted departures to constrain the number of lost reference bonds."""
from collections import Counter
from itertools import combinations


def excluded_losses(reference, required, cap, certificate):
    """Infer impossible single and joint losses once the weighted terms are verified."""
    optional = set(reference) - set(required)
    maximum = certificate['maximum']
    slack = cap * maximum - certificate['weight_sum']
    degree = Counter()
    shared = Counter()
    for term in certificate['weighted_cores']:
        bonds = sorted(set(map(tuple, term['bonds'])) & optional)
        weight = term['weight']
        for bond in bonds:
            degree[bond] += weight
        for pair in combinations(bonds, 2):
            shared[pair] += weight
    single = {bond for bond in optional if maximum - degree[bond] > slack}
    paired = {
        (a, b) for a, b in combinations(sorted(optional - single), 2)
        if 2 * maximum - degree[a] - degree[b] + shared[a, b] > slack
    }
    return single, paired


def exact_budget_clauses(certificate, required, optional, ids, cap, top):
    """Encode the integer slack identity for an already authenticated weighted cover.

    Every term needs a lost bond. Each selected bond pays its unused weight
    capacity, and the second and later hits in each term pay the term weight.
    These costs sum to maximum * losses - total_weight.
    """
    from functools import lru_cache
    from pysat.card import ITotalizer
    from pysat.formula import IDPool

    pool = IDPool(start_from=top + 1)
    maximum = certificate['maximum']
    slack = cap * maximum - certificate['weight_sum']
    if slack < 0:
        return [[]]
    degree = Counter()
    terms = []
    for term in certificate['weighted_cores']:
        bonds = frozenset(map(tuple, term['bonds'])) - required
        weight = term['weight']
        for bond in bonds:
            degree[bond] += weight
        terms.append((bonds, weight))

    clauses = []
    costs = {ids[bond]: maximum - degree[bond] for bond in optional if maximum > degree[bond]}
    for bonds, weight in terms:
        if len(bonds) < 2:
            continue
        literals = [ids[bond] for bond in sorted(bonds)]
        limit = min(len(bonds) - 1, slack // weight + 1)
        with ITotalizer(lits=literals, ubound=limit, top_id=pool.top) as total:
            clauses.extend(total.cnf.clauses)
            pool.top = total.top_id
            for extra_hit in total.rhs[1:]:
                costs[extra_hit] = weight
            if len(bonds) > slack // weight + 1:
                clauses.append([-total.rhs[-1]])

    items = sorted(costs.items(), key=lambda item: (-item[1], item[0]))
    tails = [0] * (len(items) + 1)
    for index in range(len(items) - 1, -1, -1):
        tails[index] = tails[index + 1] + items[index][1]

    def imply(prefix, child):
        if child is not True:
            clauses.append(prefix if child is False else prefix + [child])

    @lru_cache(maxsize=None)
    def node(index, remaining):
        if remaining < 0:
            return False
        if tails[index] <= remaining:
            return True
        literal, weight = items[index]
        current = pool.id()
        imply([-current, literal], node(index + 1, remaining))
        imply([-current, -literal], node(index + 1, remaining - weight))
        return current

    root = node(0, slack)
    if root is not True:
        clauses.append([] if root is False else [root])
    return clauses


def applicable_budgets(problem, reference, required, groups, charged, cap):
    """Return only authenticated redundant budgets that were tested for this cap."""
    import json
    from r35_contact_certificate import CERTIFICATE, verify

    if cap < 25 or cap > 30:
        return []
    if cap <= 28:
        candidates = [json.loads(CERTIFICATE.read_bytes())]
    else:
        source = CERTIFICATE.parent / 'R35_CONTACT_MULTIPLIERS.json'
        candidates = json.loads(source.read_bytes())['certificates']
    projected = [frozenset(group) - required for group in charged]
    return [
        candidate for candidate in candidates
        if verify(problem, reference, required, groups, candidate, charged=projected)
    ]
