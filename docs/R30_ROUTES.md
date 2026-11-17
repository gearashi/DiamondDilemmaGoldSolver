# Exact four-swap searches around the 207-edge placements

The release closes the two six-cell families on each of the three known 207-edge centers. Each center has 847,730,166,400 endpoints made from two directed three-cycles and 1,907,392,874,400 endpoints made from a four-cycle and a swap. None reached 208. Every physical piece could rotate, regardless of whether it moved. The five-cell family was already exhausted. A four-swap improvement therefore requires seven or eight moved physical pieces at these centers.

| Starting center | Joint star queries, two triples | Joint star queries, four-cycle plus swap | Exact rotation solves |
|---|---:|---:|---:|
| 207/52, d297c8f5 | 1,503,983 | 2,734,721 | 1 in each family |
| Original, 51c26e4e | 802,589 | 1,446,115 | 0 |
| Third, 840affdf | 864,836 | 1,542,237 | 0 |

At the 207/52 center the two-triple star survivor had optimum 205, and the four-cycle-plus-swap survivor had optimum 206. Rebuilding both witnesses gave the same physical identities and independently scored seams. The figures above include intersections with other tested centers. Keep separate counts for endpoints covered, cycle prefixes streamed, and complete star evaluations. [Machine-readable evidence](R30_ROUTES_RESULTS.json)

## Why the native join covers each endpoint

A directed cycle begins at its smallest cell in the outer catalogue. The inner cycle is anchored at its smallest cross-contact cell for a center whose cross-cycle contact rule has been proved. Otherwise it is anchored at its smallest cell. Equal-size cycles are ordered by their smallest cells. Each unordered pair of disjoint directed cycles therefore has one owner. The release's independently auditable bound requires contact between different cycles only on the listed three centers, with the matching physical domains and threshold at least 208. Elsewhere the join searches all cell relationships. [Contact certificate](R30_CONTACT_RESULTS.json)

Each anchored inner catalogue is sorted by its standalone gain in doubled star score. For disjoint cycles A and B, bound the star score of applying both using their individual gains and an upper bound on interactions. An interaction occurs only where inputs of both cycles occur in the same star. Replacing one neighbor changes the star maximum by at most one. Replacing its central tile changes it by at most three. The mixed difference of two replacements is therefore at most twice the smaller of those limits, regardless of all other inputs. Telescoping one replacement at a time in each cycle bounds their total interaction by the sum over such pairs. An entire sorted suffix below threshold is omitted. Remaining candidates get a simultaneous exact star calculation and eligible survivors get global rotation optimization. Independent tests compare every native survivor and its multiplicity with full portable evaluation on ten-cell problems, including small output buffers and non-release domains.

For workstream 1, the portable rule first prices isolated cycles against the same exterior rotations as their undone predecessors. For a remaining candidate, `--deep` tests each way to swap back two cells in the same cycle. It compares the current two rotations against every orientation of the swapped-back pair with the shared neighboring rotations fixed. The bounded joint relation enforces shared rotation choices across these tests. Exceeding the joint budget retains the candidate. On the two real star survivors, the four-cycle endpoint failed this stronger test; the triple endpoint survived it. They had two and five cross-cycle contacts respectively. These complete six-cell runs also close every zero- or one-contact subcase at 208. The seven- and eight-cell families have a second independent certificate. With exactly one cross-cycle cell pair at distance at most two, at most two star factors couple cycles. Each shared star can increase the combined doubled gain over the separate gains by at most two. Enumerating all standalone swap/triple gains, then joining only gain-qualified disjoint cycles, left 9,400 joint star evaluations across the six larger scopes. Zero reached the 208 star threshold. A 208 endpoint in either remaining family therefore requires at least two cross-cycle contact pairs. They can involve different cycle pairs. A synthetic planted target with precisely one contact tests that the same audit retains a valid solution when a stronger certificate is unavailable. [Seven/eight-cell geometric evidence](R30_SINGLE_CONTACT.json)

A separate exhaustive component audit priced every potentially sufficient contacting swap pair and three-cycle/swap pair. Components with no cells within distance two have exactly additive star gains. For three swaps plus an isolated fourth, the earlier native triple-swap engine catalogued all threshold-eligible triples with previous-center exclusions disabled. At the original and third center 182 and 204 triple-star candidates respectively passed its deliberately rounded-down threshold. The middle center had zero. Joining every component to its allowable isolated cycles produced zero full threshold candidates. Consequently every cycle must belong to the same connected component in each remaining family. A triple and two swaps require at least two cycle-to-cycle contacts; four swaps require at least three. The 448,406 two-cycle component star queries in this follow-up are separate from the 9,400 one-contact endpoint queries above. [Connected-component audit](R30_CYCLE_GROUPS.json)

## Running the searches

For a different center or target, use the project Python environment and the two-family native solver. A local GCC or Clang builds the small C library on first use. Use one invocation per family. A fresh install recognizes a matching shipped closure and reports it immediately. To recompute a published family add `--recheck` and choose a fresh database. For an unrelated center, a whole-swap-budget defect packing can close the family before the native catalogue is built. The program commits deterministic prefix pages along with newly solved rotation optima in SQLite. All commands below are run from the solver directory.

```text
python r30_route.py --incumbent docs/R29_INCUMBENT.json --family 33 --target 208 --database runtime/four-swaps.sqlite --seconds 0
python r30_route.py --incumbent docs/R29_INCUMBENT.json --family 42 --target 208 --database runtime/four-swaps.sqlite --seconds 0
```

For workstreams 5, 6 and 10, a portable restricted-pool command accepts the actual physical cells to search. All other physical identities stay fixed and all rotations remain free. The `--halves` option joins one triple with an ordered pair of swaps, or two pairs of swaps. Half catalogues store their occupancy, standalone gain and interaction-boundary membership; bitset intersections and sorted suffix bounds remove entire partner ranges. The four-swap half join keeps a single split of four ordered swaps. The explicit catalogue is intended for small pools. Use the ranked multi-center runner for full-board seven- and eight-cell searches.

```text
python r30_cycle_joins.py --family 322 --cells 18 30 42 44 51 62 118 122 146 155 --halves --target 208
python r30_cycle_joins.py --family 2222 --cells 18 30 42 44 51 62 118 122 146 155 --target 208
```

Across two real ten-cell pools the three-cycle/two-swap search used 87 and one individual star queries against 25,200 with ordinary donor enumeration. The indexed half join and the direct typed search produced the same survivors. On the first pool the half join excluded partner ranges covering 25,113 endpoints. On the second it excluded 23,952 before considering the remaining partners individually. The four-swap search required zero star queries in both pools against 4,725 with ordinary enumeration. The half join spent more pair-price lookups on one four-swap pool, so it is an optional choice. These pools are partial seven/eight-cell results.

For workstream 7 the family-specific support bound restricts a directed donor assignment to the specified cycle lengths. Four transpositions use maximum-weight perfect matching. On 2,544 support/family comparisons, the restriction closed eight additional cases at score 208. It closed one additional case for four-cycle/swap, three for two triples, one for triple/two swaps and three for four swaps. All of the seven/eight-cell cases still open after generic assignment closed in that sample.

On eight planted exact-score searches, ordering children by admissible gain reduced prefixes to the first validated witness from 241 to 154 in aggregate. It improved six cases, tied one and increased one from 12 to 31. Each ordering needed one exact rotation solve in each case. An unrestricted witness search may cross any score floor on its way to the endpoint. A `--path` request on the native command reconstructs a shortest path only after obtaining a threshold hit.

To reproduce the cold production measurements use a new database. A rerun on the same completed database reads its counters and reports the shorter time for reopening it.

```text
python -B -m unittest -v test_r30_routes
python -m benchmarks.bench_r30_routes --tiles data/tiles.json --database fresh-four.sqlite --output routes.json
python bench_r30_single_contact.py --tiles data/tiles.json --output contact-regeneration.json
python bench_r30_cycle_groups.py --tiles data/tiles.json --output component-regeneration.json
```
