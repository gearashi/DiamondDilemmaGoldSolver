"""Compare family quotient costs with explicit matchings, subsets and renamed scopes."""
from itertools import combinations
import random
import unittest

from r35_family_frontiers import exact_frontier, relaxed_frontier, _local_exact, _physical_exact
from r35_mate_price import _matching_prices, _relaxed_prices


def all_pairings(vertices, allowed):
    if not vertices:
        yield frozenset()
        return
    first,*rest=vertices
    for i,other in enumerate(rest):
        edge=tuple(sorted((first,other)))
        if edge in allowed:
            for sub in all_pairings(rest[:i]+rest[i+1:],allowed):
                yield sub|{edge}


def record_price(best,missing,covers):
    tag=0
    for edge in missing:
        tag |= covers[edge]
    best[tag]=min(best.get(tag,len(missing)),len(missing))


class LocalPriceTests(unittest.TestCase):
    def test_every_matching_and_independent_dropout_in_generated_families(self):
        for seed in range(80):
            rng=random.Random(seed+83102)
            n=(2,4,6,8)[seed%4]
            family=rng.sample(range(480),n)
            choices={tuple(sorted(pair)) for pair in combinations(family,2) if rng.random()<.7}
            named=set(rng.sample(sorted(choices),min(seed%9,len(choices))))
            masks=[0,1<<51,8,(1<<51)|8,6|8,6]
            covers={edge:rng.choice(masks) for edge in named}
            reference={}
            for m in all_pairings(family,choices):
                record_price(reference,named-m,covers)
            result=exact_frontier(_matching_prices,family,choices,named,covers)
            self.assertEqual(dict(result),reference,(seed,n))
            drop={}
            for count in range(len(named)+1):
                for subset in combinations(named,count):record_price(drop,subset,covers)
            self.assertEqual(dict(relaxed_frontier(_relaxed_prices,named,covers)),drop)
            self.assertIsInstance(result,tuple)

    def test_physical_and_obligation_renaming_reuses_the_exact_subset_search(self):
        _local_exact.cache_clear()
        _physical_exact.cache_clear()
        def run(sides,tags):
            choices={tuple(sorted(pair)) for pair in combinations(sides,2)}
            covers={tuple(sorted((sides[0],sides[1]))):tags[0],tuple(sorted((sides[1],sides[2]))):tags[1]}
            actual=dict(exact_frontier(_matching_prices,sides,choices,set(covers),covers))
            self.assertEqual(actual,_matching_prices(sides,choices,set(covers),covers))
        run((2,9,17,55),(5,2))
        run((205,45,309,7),(64,384))
        # Use the same two disjoint obligation columns with different labels.
        before=_local_exact.cache_info()
        run((109,28,39,11),((1<<35)|(1<<43),128))
        after=_local_exact.cache_info()
        self.assertEqual(after.misses,before.misses)
        self.assertEqual(after.hits,before.hits+1)
        # Changing which named bonds satisfy each condition must still reprice.
        run((109,28,39,11),(32,32))

    def test_large_tables_use_the_complete_direct_fallback(self):
        edges={(2*i,2*i+1) for i in range(10)}
        covers={e:1<<i for i,e in enumerate(sorted(edges))}
        self.assertEqual(dict(relaxed_frontier(_relaxed_prices,edges,covers)),_relaxed_prices(edges,covers))
        family=tuple(range(20))
        self.assertEqual(dict(exact_frontier(_matching_prices,family,edges,edges,covers)),{0:0})


if __name__=='__main__':
    unittest.main()
