"""Regression checks for reusing the Gold loss search's suffix table across pages."""
from collections import Counter
from itertools import combinations, product
from math import comb
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from r27_structure_fixtures import planted
from r29_gold_phase import EndpointCuts, _reciprocal
from r32_gold import run_syndrome
from r32_gold_bonds import inventory
from r32_gold_loss import reference_groups
from r32_gold_syndrome import OutsideSyndromes


def fixture(seed=1):
    problem, _ = planted(1, 3, seed, True, True)
    reference, _, groups = reference_groups(problem, [3*i for i in range(problem.n)])
    query = OutsideSyndromes(problem, reference, groups, inventory(problem).required)
    return problem, reference, groups, query


def phase_costs(problem, reference, query):
    """Enumerate all tile phases and price every attainable outside-error set."""
    sides = EndpointCuts(problem).sides
    costs = {}
    for phases in product((0, 1), repeat=problem.n):
        labels = [(mask, color ^ phases[i//3]) for i, (mask, color) in enumerate(sides)]
        counts = Counter(labels)
        reciprocal_counts = []
        for pair, count in counts.items():
            other = _reciprocal(*pair)
            reciprocal_counts.append(count if pair == other else count + counts[other])
        if any(n % 2 for n in reciprocal_counts):
            continue
        failures = {bond for bond in reference | query.required
                    if labels[bond[0]] != _reciprocal(*labels[bond[1]])}
        if failures & query.required:
            continue
        cost = len(failures) + sum(group.isdisjoint(failures) for group in query.groups)
        outside = tuple(i for i, bond in enumerate(query.outside) if bond in failures)
        costs[outside] = min(cost, costs.get(outside, cost))
    return costs


class GoldLookupTests(unittest.TestCase):
    def test_cached_pages_preserve_negative_results_and_positive_witnesses(self):
        for seed in (1, 11):
            p, ref, _, q = fixture(seed)
            costs = phase_costs(p, set(ref), q)
            seen = set()
            for cap, errors in ((7, 4), (9, 4), (6, 3), (7, 4)):
                length = min(errors, 2)
                prefixes = list(combinations(range(len(q.outside)-length), errors-length))
                for rank, prefix in enumerate(prefixes):
                    possible = {indices for indices, cost in costs.items()
                                if cost <= cap and len(indices) == errors and indices[:-length] == prefix}
                    out = q.indexed(cap, errors, start=rank, stop=rank+1)
                    seen.add(out['status'])
                    if possible:
                        self.assertEqual(out['status'], 'PHASE_RELAXATION_FEASIBLE')
                        witness = tuple(q.outside.index(tuple(b)) for b in out['outside_errors'])
                        self.assertIn(witness, possible)
                        self.assertEqual(out['next_rank'], rank)
                    else:
                        self.assertEqual(out['status'], 'INFEASIBLE')
                        self.assertEqual(out['next_rank'], rank+1)
                    if rank:
                        self.assertEqual(out['work'].get('suffix_rows', 0), 0)
                        self.assertEqual(out['work']['suffix_table_reuses'], 1)
            self.assertEqual(seen, {'PHASE_RELAXATION_FEASIBLE', 'INFEASIBLE'})

    def test_disabled_cache_and_empty_interval(self):
        p, ref, groups, cached = fixture()
        uncached = OutsideSyndromes(p, ref, groups, inventory(p).required, reuse_suffix_tables=False)
        rows = comb(len(cached.outside), 2)
        with patch.object(cached, '_build_suffix_table', side_effect=AssertionError):
            empty = cached.indexed(7, 4, start=2, stop=2)
        self.assertEqual((empty['status'], empty['work']), ('INFEASIBLE', {}))
        for start in (0, 1, 1, 2):
            cold = uncached.indexed(7, 4, start=start, stop=start+1)
            warm = cached.indexed(7, 4, start=start, stop=start+1)
            changes = {'suffix_rows', 'suffix_table_reuses'}
            self.assertEqual({k:v for k,v in cold.items() if k != 'work'},
                             {k:v for k,v in warm.items() if k != 'work'})
            self.assertEqual({k:v for k,v in cold['work'].items() if k not in changes},
                             {k:v for k,v in warm['work'].items() if k not in changes})
            self.assertEqual(cold['work']['suffix_rows'], rows)
            self.assertEqual(cold['work'].get('suffix_table_reuses', 0), 0)

    def test_sorted_tail_seeks_keep_all_attainable_error_sets(self):
        p, ref, groups, seeking = fixture()
        scanning = OutsideSyndromes(p, ref, groups, inventory(p).required, seek_ordered_tails=False)
        costs = phase_costs(p, set(ref), seeking)
        skipped = 0
        for errors in range(len(seeking.outside)+1):
            length = min(errors, 2)
            prefixes = combinations(range(len(seeking.outside)-length), errors-length)
            for rank, prefix in enumerate(prefixes):
                for cap in (errors+len(groups)-1, errors+len(groups), errors+len(groups)+2):
                    scan = scanning.indexed(cap, errors, start=rank, stop=rank+1)
                    seek = seeking.indexed(cap, errors, start=rank, stop=rank+1)
                    attainable = any(cost <= cap and len(indices) == errors
                                     and indices[:errors-length] == prefix for indices,cost in costs.items())
                    self.assertEqual(seek['status'] == 'PHASE_RELAXATION_FEASIBLE', attainable)
                    self.assertEqual(seek['status'], scan['status'])
                    self.assertEqual(seek['next_rank'], scan['next_rank'])
                    self.assertEqual(seek.get('outside_errors'), scan.get('outside_errors'))
                    for key in ('prefixes','indexed_tails','joint_queries'):
                        self.assertEqual(seek['work'].get(key, 0), scan['work'].get(key, 0))
                    if seek['status'] == 'INFEASIBLE':
                        visits = scan['work'].get('unowned_tail_visits', 0)
                        self.assertEqual(seek['work'].get('unowned_tail_visits_skipped', 0), visits)
                        skipped += visits
        self.assertGreater(skipped, 0)

    def test_bucket_charges_against_exhaustive_phases_with_and_without_anchor(self):
        p, reference, groups, _ = fixture()
        for ref, packed in ((set(reference), groups), (set(sorted(reference)[:7]), [])):
            required = inventory(p).required
            ordinary = OutsideSyndromes(p, ref, packed, required, use_bucket_price=False)
            prepaid = OutsideSyndromes(p, ref, packed, required, use_bucket_price=True)
            costs = phase_costs(p, ref, prepaid)
            seen = set()
            price_reuse = 0
            for errors in range(len(prepaid.outside)+1):
                length = min(errors, 2)
                prefixes = combinations(range(len(prepaid.outside)-length), errors-length)
                for rank,prefix in enumerate(prefixes):
                    for extra in (0,1,2):
                        cap = len(packed)+errors+extra
                        a = ordinary.indexed(cap, errors, start=rank, stop=rank+1)
                        b = prepaid.indexed(cap, errors, start=rank, stop=rank+1)
                        possible = {s for s,c in costs.items() if len(s)==errors and c<=cap
                                    and s[:errors-length]==prefix}
                        self.assertEqual(bool(possible), b['status']=='PHASE_RELAXATION_FEASIBLE')
                        self.assertEqual(b['status'], a['status'])
                        self.assertEqual(b.get('outside_errors'), a.get('outside_errors'))
                        self.assertEqual(b['work'].get('joint_queries',0), a['work'].get('joint_queries',0))
                        if possible:
                            witness = tuple(prepaid.outside.index(tuple(bond)) for bond in b['outside_errors'])
                            self.assertIn(witness, possible)
                        seen.add(b['status'])
                        price_reuse += b['work'].get('bucket_prices_reused', 0)
            self.assertEqual(seen, {'PHASE_RELAXATION_FEASIBLE', 'INFEASIBLE'})
            self.assertEqual(price_reuse>0, bool(packed))

    def test_timeout_reopens_same_prefix_and_recovers_same_witness(self):
        _, _, _, q = fixture()
        with patch.object(q, 'joint', side_effect=TimeoutError):
            stopped = q.indexed(9, 4, start=0, stop=1)
        self.assertEqual((stopped['status'], stopped['next_rank']), ('UNKNOWN', 0))
        self.assertEqual(stopped['work']['suffix_rows'], comb(len(q.outside), 2))
        resumed = q.indexed(9, 4, start=stopped['next_rank'], stop=1)
        _, _, _, cold = fixture()
        control = cold.indexed(9, 4, start=0, stop=1)
        self.assertEqual(resumed['status'], 'PHASE_RELAXATION_FEASIBLE')
        self.assertEqual(resumed['outside_errors'], control['outside_errors'])
        self.assertEqual(resumed['work'].get('suffix_rows', 0), 0)
        self.assertEqual(resumed['work']['suffix_table_reuses'], 1)

    def test_sqlite_resume_changes_page_size_and_avoids_completed_work(self):
        p, ref, groups, q = fixture()
        with TemporaryDirectory() as folder:
            db = Path(folder) / 'loss.sqlite'
            def stop_after_page(errors, current, total):
                self.assertEqual((errors, current, total), (3, 1, 4))
                raise InterruptedError
            with self.assertRaises(InterruptedError):
                run_syndrome(p, ref, groups, 6, [3], db, 0, page=1, emit=stop_after_page)
            resumed = run_syndrome(p, ref, groups, 6, [3], db, 0, page=2)
            cold = run_syndrome(p, ref, groups, 6, [3], ':memory:', 0, page=2)
            self.assertEqual(resumed['status'], cold['status'])
            self.assertEqual(resumed['status'], 'INFEASIBLE')
            self.assertEqual(resumed['completed_prefixes'], cold['completed_prefixes'])
            self.assertEqual(resumed['work']['prefixes'], 3)
            self.assertEqual(resumed['work']['suffix_rows'], comb(len(q.outside), 2))
            self.assertEqual(resumed['work']['suffix_table_reuses'], 1)
            with patch.object(OutsideSyndromes, '__init__', side_effect=AssertionError):
                replay = run_syndrome(p, ref, groups, 6, [3], db, 0, page=3)
            self.assertEqual(replay['status'], 'INFEASIBLE')
            self.assertEqual(replay['new_pages'], 0)
            self.assertEqual(replay['work'], {})


if __name__ == '__main__':
    unittest.main()
