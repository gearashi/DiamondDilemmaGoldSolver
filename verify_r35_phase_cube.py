"""Independently enumerate and refute signed geographic phase cubes from raw artwork."""
import argparse
from collections import Counter, deque
from functools import lru_cache
import json
from pathlib import Path

from r35_phase_census import census, count_equations, gap_profiles, reverse_side
from r35_phase_cube_rules import certificate_rules, certified_rules
from search_problem import SearchProblem


def physical_labels(raw, phase):
    return [mask ^ (4095 if (phase >> (i//3)) & 1 else 0) for i, mask in enumerate(raw)]


def impossible_family(labels, sides):
    """Exhaust all disjoint necessary mates of one closed physical class."""
    group = set(sides)
    if not group or not all(type(i) is int and 0 <= i < len(labels) for i in group):
        return False
    mates = {}
    for a in group:
        reverse = reverse_side(labels[a])
        row = {b for b,v in enumerate(labels) if a//3 != b//3 and v == reverse}
        if not row <= group:
            return False
        mates[a] = row
    @lru_cache(None)
    def possible(left):
        if not left:
            return True
        a = min(left, key=lambda x: (len(mates[x] & left), x))
        return any(possible(left - {a,b}) for b in mates[a] & left)
    return not possible(frozenset(group))


def failed_roots(problem, labels, anchor):
    """One literal per actual placement. Check every labelled root, without rotations."""
    full = (1 << (3*problem.n)) - 1
    rows = []
    for side in range(3):
        row = {}
        for code in range(3*problem.n):
            t, r = divmod(code,3)
            label = labels[3*t+(side-r)%3]
            row[label] = row.get(label,0) | (1<<code)
        rows.append(row)
    support = {}
    for side in range(3):
        for opposite in range(3):
            options = []
            for code in range(3*problem.n):
                t, r = divmod(code,3)
                key = reverse_side(labels[3*t+(side-r)%3])
                options.append(rows[opposite].get(key,0) & ~(7<<(3*t)))
            support[side,opposite] = options
    def viable(cell, rotation):
        domains = [full]*problem.n
        domains[cell] = 1 << (3*anchor+rotation)
        todo = deque([cell])
        while todo:
            c = todo.popleft()
            d = domains[c]
            if d.bit_count() > 32:
                continue
            for side,(other,back) in enumerate(problem.board.neighbors[c]):
                options = support[side,back]
                candidates = 0
                pending = d
                while pending:
                    bit = pending & -pending
                    candidates |= options[bit.bit_length()-1]
                    pending ^= bit
                new = candidates & domains[other]
                if not new:
                    return False
                if new != domains[other]:
                    domains[other] = new
                    todo.append(other)
        return True
    return sum(not viable(c,r) for c in range(problem.n) for r in range(3))


def check(problem, doc, fixed):
    rules = certificate_rules(doc)
    if not rules or rules != certified_rules(problem):
        raise ValueError("Unrecognized certificate or signed geometry.")
    if (fixed["signed_mesh_sha256"] != doc["signed_mesh_sha256"] or
        fixed["external_result"] != "VERIFIED" or fixed["phase"] != doc["fixed_phase"] or
        fixed["files"]["board.cnf"]["sha256"] != doc["fixed_dimacs_sha256"] or
        fixed["files"]["board.drat"]["sha256"] != doc["fixed_proof_sha256"]):
        raise ValueError("Fixed phase proof manifest does not match.")
    mask = (1 << problem.n) - 1
    raw = gap_profiles(problem.tiles)
    rows = count_equations(problem.tiles)
    leaves = {int(word,16): witness for word,witness in doc["leaves"].items()}
    phase_count = Counter()
    nodes = []
    for row, details in zip(rules, doc["rules"]):
        for complement in (0,1):
            expected = {w ^ (mask if complement else 0) for w in leaves}
            pins = {t:v^complement for t,v in row}
            result = census(rows,problem.n,pins)
            assert result["complete"] and set(result["phases"]) == expected
            assert len(result["phases"]) == len(expected)
            nodes.append(result["work"])
            mismatches = set()
            for h in details["adverse"]:
                word = int(h,16) ^ (mask if complement else 0)
                disagree = [t for t,v in pins.items() if (word>>t)&1 != v]
                assert len(disagree) == 1 and word not in expected
                mismatches.add(disagree[0])
                tally = Counter(physical_labels(raw,word))
                assert all(n == tally[reverse_side(k)] and (k!=reverse_side(k) or not n%2) for k,n in tally.items())
            assert mismatches == set(pins)
    for word, proof in leaves.items():
        for complement in (0,1):
            labels = physical_labels(raw,word^(mask if complement else 0))
            kind = proof["type"]
            if kind == "fixed_phase":
                assert word == int(doc["fixed_phase"],16)
            elif kind == "signed_physical_matching":
                assert impossible_family(labels,proof["sides"])
            elif kind == "geographic_anchor":
                count = failed_roots(problem,labels,proof["anchor"])
                assert count == 3*problem.n
                phase_count["labelled_roots"] += count
            else:
                raise ValueError("Unverified phase leaf.")
            phase_count[kind] += 1
    return dict(cubes=[len(r) for r in rules], census=nodes, gauges=dict(phase_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Run without -O to check all assertions.")
    docs = Path(__file__).resolve().parent / "docs"
    report = check(SearchProblem(json.loads(args.tiles.read_bytes())),
                   json.loads((docs/"R35_PARTIAL_PHASE_CORES.json").read_bytes()),
                   json.loads((docs/"R35_FIXED_PHASE_PROOF.json").read_bytes()))
    print(json.dumps(report, indent=2))
