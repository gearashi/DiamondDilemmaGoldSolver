# Round 34 physical and geographic Gold experiments

Base commit 8e010cc8 (v1.14.0). Use `python bench_r34_physical.py --tiles ../data/tiles.json` from the checkout. The finite regression command is `python -m unittest -q test_r34_physical test_r28_geometry test_r30_exactgold test_r31_gold test_r32_gold test_r32_fresh`. All 40 tests passed on the project Python 3.12 environment with OR-Tools.

## 16. Mandatory paths and mating support

The Gold physical inventory, domain partner projection, and restricted GoldMatching CP model can contract forced sewings into weighted open strand paths. Test all loose mates for premature closure and repeat matching after removing any. Closing a cycle containing every strand is permitted. A bounded matching query retains unresolved choices. Independently tracing the original tile segments agreed with the contracted path test across four planted positive controls, four depths each.

On the real 160-piece puzzle the root still has 3,036 possible pairs, eight required, 24 forbidden. Across ten reproducible distinct samples per pin count from the 207 board's matched physical bonds, compare to perfect-matching support on exactly the same restrictions.

| Fixed bonds | Already infeasible by matching | Additional residuals closed by paths | Additional mates removed on surviving residuals |
| --- | ---: | ---: | ---: |
| 4 | 0 | 0 | 5 |
| 12 | 0 | 0 | 18 |
| 24 | 1 | 0 | 18 |
| 48 | 2 | 0 | 45 |
| 96 | 5 | 1 | 37 |
| 144 | 9 | 1 | 0 |

## 17. Sparse nonbipartite mate classes above 18 darts

The 46 blank sides form the only self-reciprocal class above the old matching threshold on this puzzle. Its root graph already has a density certificate. A screen of the 60 pin samples above and 21 incumbent geographic restrictions (4, 8, 16, 32, 64, 96 and 128 free cells, seeds 34-36) found zero sparse, even nonbipartite classes above 18. For an explicit residual of the same 46 physical darts, retain two odd cycles of 23 darts each and one eligible crossing. The old sparse fallback admitted all 47 edges. Exact per-edge support now retains the 23 edges of the only possible perfect matching after 267 memoized matching states. Incomplete queries retain unverified choices. The common dense and bipartite cases keep their existing certificates.

## 18. Odd boundaries in large fractional mate classes

The fractional pricing option can now propose odd-set inequalities using minimum cuts. An odd number of darts cannot all match internally. The installed inequality counts all crossing candidate bonds with integer coefficient one and is re-priced against the unrounded LP solution. The 46-dart, 47-edge residual above raised the LP crossing lower bound from zero to one after one 23-dart cut. On 160 generated 8/10-dart fractional covers the minimum returned by the cut-tree screen agreed with direct enumeration of every odd subset (26 violated cases). A 12-cut unrestricted root run produced exactly the same cuts with large-class pricing enabled and disabled, with zero large-class cuts selected. This improvement applies on restricted LP graphs.

## 21. Geographic channel on a fresh board

The geographic CP option retains its ternary table for small queries and uses shared physical partner variables and mesh element constraints when the table overflows. It checks all 240 mesh seams and enforces piece uniqueness. At the unrestricted real root the old seam loop would visit 55,296,000 pose pairs, above the 120,000-entry limit. The alternative built and validated a model with 115,200 partner array entries, 5,135 integer/Boolean variables and 11,048 constraints. Independent positive and negative planted controls at two seeds agreed with this encoding. Six further geographic searches (seed 34, 3/6/8 movable tiles, single versus split loop) reached the same Gold or infeasible verdict in one native CP call each under the same ten-second ceiling. On the positive 6/8-mover controls the small ternary model used 638/5,365 branches and the partner model used 2,560/6,043. The small ternary model remains preferred when it fits. This is a build result; a full-root proof or witness has not been obtained.

## 22. Pose packing

Restricted roots choose the tile with the fewest candidate poses. The packing routine splits independent groups by their possible cell footprints and only memoizes used cells reachable by later groups. Eighty exhaustive small assignment comparisons gave identical feasibility results. On a 10-cell residual with independent 3-tiles/4-cells and 7-tiles/6-cells components, the earlier complete-packing recursion visited 267 nodes. Splitting and projecting returned the same infeasible verdict in 67 nodes. On the planted seed 34 fixture, five two-tile controls reduced candidate roots from twelve poses to one and each produced one compatible footprint.

## 24. Early Hall tests on restricted parity skeletons

Maintain a current assignment of physical pieces to cells. When a skeleton decision removes its assigned piece, look for an alternating path to repair the matching. After widening a row during backtracking, the narrower matching remains valid. Full unrestricted domains still use the active-piece count. All 50 complete ten-cell controls produced the same six surviving patterns, reducing total states from 3,951 to 2,337, with 627 earlier rejections. On the real 160-piece puzzle, restrict a seed-34 parity placement and widen the first 16, 32 or 48 cells with four alternative poses each. Every comparison finished exhaustive enumeration and found exactly one identical skeleton. States fell from 1,178 to 203, 2,652 to 205, and 17,682 to 793 respectively.

## 25. Conditioning vertex colors on the required active count

The new `r32_fresh.py --conditioned-contour` mode chooses mutually nonadjacent free vertices, fixes the others at the sampled density, and samples a completion with exactly 103 active cells on the real puzzle whenever one exists. The two fixed apices stay pinned. Fresh checkpoints distinguish this mode from the original sampler. For seeds 0-31 the old sampler used 868 coloring trials to reach 32 distinct complete parity placements. The conditioned sampler used 40 for 32 distinct complete parity placements. Both ran 32 assignment queries; neither raw placement met all eight mandatory physical bonds. Best raw score was 19 for the old sampler and 18 for the conditioned sampler. Exact counts improve contour construction, with a further complete physical lift unverified. Finite exhaustive free-color assignments check reachable counts, retained apices and returned completions.

## Remaining work

- 19. Infer a mate when every surviving pose of a physical owner implies it. Test reversible ownership and clause watching with backtracking and exhaustive geographic positives.
- 20. Integrate confirmed physical restrictions in the separate ordinary SAT/CP circuit builders and compare the edges-only path.
- 23. Build independent phase cubes with validated complement scope and contraction-aware branching.
- 24. Condition mandatory component poses on the current partial parity skeleton, and measure additional closures over early Hall matching alone.
- 25. Compare physical lift outcomes on equal search budgets and test whether count-preserving local flips add independent useful patterns.
- 26. Factor signed donor rows by physical dart masks and phase constraints and compare cross-seed and cross-rotation reuse.
- The next independent implementation is #19. Start with `python -m unittest test_r32_gold.GoldTests.test_sewing_certificate_transport_and_guarded_loss`. Build a fixture where an owner has two possible poses that both force the same mate, while both cells remain nonsingleton. Replay the old and new watcher before restriction, after the two-pose restriction, and after relaxing to the sibling. Compare exact survivors with `r27_structure_fixtures.independent` before integrating a changed-owner index.
