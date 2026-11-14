# Physical exchange and rotation bounds

The threshold in these experiments is 208 matched seams. A full Gold solution requires 240 matched seams and one loop. The published physical centers, topology and legal domains identify where a certificate applies. Each calculation allows all rotations unless the supplied puzzle restricts their domains.

## 3. Physical defects and movement budgets

A seam with no compatible physical side must fail. Another failure is forced when a group of seams incident to one cell cannot all use the same rotation there. The compiler packs witnesses with disjoint **seams** and collects their incident cells. If 33 failures persist, reaching 208 requires changing at least one physical placement in that set. Distinct packings yield alternative movement clauses. Subsumption removes redundant clauses. A partial choice can be rejected if the minimum number of remaining cells needed to hit every unfulfilled clause exceeds the move budget. Compilation limits weaken pruning and preserve coverage.

| Center | Retained movement clauses | Cells per clause | Six-cell supports ruled out by their union |
|---|---:|---:|---:|
| Original | 2 | 60 | 1,263,575,544 |
| 207/52 | 4 | 61 | 1,259,815,824 |
| Third | 2 | 60 | 1,263,575,544 |

The seven- and eight-cell counts are in the evidence. They overlap other filters. On each center, 25–26 of 384 seeded six-cell samples were rejected before asking for a donor assignment bound. The previous support bound could close all 384 after doing that work.

The v1.10.0 and current portable runners were also compared on identical seven-cell colex prefixes at the 207/52 center, with fresh in-memory state and prior ball reuse disabled. On the published first twelve supports, the movement clauses saved one of twelve support assignments. On the first 1,024 they saved 36, lowering assignment solves to 988. Both versions rejected all 215,040 possible endpoints in that longer prefix and needed zero full rotation solves. Times were 0.579 s and 0.687 s respectively. This comparison supports retaining the reduction in search work. [Seven-cell comparison](R30_BOUNDS_SEVEN.json)

## 4 and 18. Common rotations across multiple stars

Each star allows its three neighbors to choose independent rotations. When two or more centers are priced together, the score must also agree on rotations on their common internal edges. The solver computes integer discounts and packs groups with disjoint **star centers**. Shared neighbors outside those centers are safe because their rotations remain relaxed. A partial exchange keeps a discount only while every physical identity in its dependency footprint stays fixed.

On the three physical centers, pair agreement lowered the doubled bounds 415, 416 and 415 to **414** each. Adding short paths and rings of up to six vertices tied the cheaper pair filter. The portable four-swap search therefore applies the pair version.

A separate bound partitions seams into acyclic subgraphs and independently prices both trees. Any seams outside these forests stay in the upper bound through a physical compatibility check. Opposite integer rotation prices cancel on any common legal assignment. Each upward tree message is cached by its input contributions and ordered edge matrix. The best bound over two decompositions remains safe even if cost shifting ends early. On 55 selected nearby permutations at the 207/52 center, a single round improved the star bound on nine. Exchanging cells 30 and 118 reduced 208 to 207. Four rounds tied one on the three unchanged centers. The route tries one round on an uncached endpoint that survives earlier bounds, before exact rotation optimization. The standalone command exposes the longer diagnostic.

## 8. Entire clusters with unrestricted distant choices

A family specifies one or more interacting moved cells. Any additional moved cells have disjoint closed stars, both with each other and with the specified cluster. The solver computes the interaction terms within the cluster, creates price rows for every possible distant substitution and obtains a checked integer assignment upper bound. A moved-row penalty bounds the number of additional movers. Relaxing some donor correlations preserves an upper bound. A donor refinement fixes part of the cluster only if it needs another decision. The overlap mode can instead promote a previously unaccounted interaction into the cluster and recurse. Exhausting its allowance returns a bound that still covers the unresolved cases.

On every one of the 710 nearby pairs in each center, the benchmark allowed arbitrary donors and up to four additional distant movers. It closed **684, 685 and 684** pair families before choosing the remaining cells. The first open pair in each center reached its 16-node donor refinement cap. The release records the completed qualifying pairs for six-piece search. A candidate needs its *only* close pair to match a recorded pair to consume this certificate. A 207/52 residual with moved cells 0 and 1 and possible further changes at 2, 3, 18 and 44 had a doubled upper bound of 412 after the overlap decomposition visited five cases. This residual allowed interactions outside the initially specified pair.

Removing the movement penalty and allowing an arbitrary number of distant, mutually isolated movers still closed **682, 685 and 684** pair families. These certificates apply at any swap distance. The portable route uses the union of applicable records for distances of at least four. For a rotated center, it maps both the complete physical inventory and the certified pair through a verified, domain-preserving proper rotation. The mapping of every anchor was checked on all ten rotations of all three centers. In an isolated filter benchmark on the original 207/52 orientation, a real six-piece cursor page with support 0, 1, 57, 91, 110 and 129 consumed the saved proof and excluded all 130 cycle orders on that support. The final release also recognizes the entire six-piece completion on the ordinary route. [Unrestricted distant-mover certificates](R30_BOUNDS_ALL_SIZES.json)

## 9. Remaining-swap potential

For a known physical midpoint, every unaffected witness in a disjoint defect packing must still fail. At most twice the remaining swap count of distinct cells can change. The solver bounds the largest number of witnesses those cells could repair. On small restricted regions, exact coverage or bounded enumeration of distinct swap endpoints strengthens the estimate. Previously seen physical endpoints are reused during enumeration. A node cap falls back to the analytic defect bound. The BFS option `--max-swaps` scopes terminal decisions to the requested depth and shares exact physical scores across depths. Omitting it retains the existing unlimited-distance walk and its saved frontier.

On the real 207/52 center, four arbitrary swaps had upper bound 222 from this potential. Restricting movers to the 99 cells outside one movement clause tightened it to 207. On the seven-cell residual in the evidence, one- and two-swap endpoint tables improved the respective bounds from 211 and 215 to 208. On a planted ten-cell board, a one-swap walk with six displaced pieces went from 16 closed centers and 16 exact optimizations to **one closed center and one exact optimization**. Both proved that the goal was unreachable within one swap. The screened walk reported its best visited witness at 8 matched edges and the exhaustive control at 11. The depth-limited search is a target decision procedure. Only a reached threshold or a completed frontier establishes its corresponding result.

The portable four-swap route also calculates this bound on the entire physical shell before transferring a support to the expensive filters. On all eight saved CPU/CUDA fresh starts, the seven-cell cursor went straight to complete with zero evaluated donor orders, rotation solves or hits. The corresponding bounds were 191, 189, 188, 189 (CPU) and 184, 193, 191, 188 (CUDA). Each ordinary seven-cell cursor contained 466,251,591,520 supports. This closure retains the committed counts of a partial run. Reopening a completed cursor does zero further budget evaluations. Portable controls can disable this check with `--no-cluster-screen`. [Fresh center data](R30_FRESH_RESULTS.json)

## 13. Contextual donor scoring

The scorer groups donors by the set of allowed local rotations and their ordered Boolean response vectors to every possible rotation of every candidate neighbor identity. It keeps individual physical IDs in each group. Neighbor identity changes refine the signature before reusing prices. The score cache includes the widths of the neighbor domains so boundaries between different physical sides stay distinct. At the 207/52 center, 25,600 destination/donor combinations had **3,655** destination-local groups, evaluating **545** distinct scalar response patterns across the board. A warm replay required zero new representative evaluations. Each of the 25,600 individual answers matched the ungrouped scorer. This count excludes constructing the signatures. The method is available for repeated regional requests. The portable route already vectorizes most one-off star computations.

## Reproduce

Run from the project directory. The commands take the installed piece data and the bundled 207/52 reference. The family and overlap queries can also be used on a different incumbent.

```text
python r30_bounds.py cores --incumbent docs/R29_INCUMBENT.json --target 208
python r30_bounds.py family --incumbent docs/R29_INCUMBENT.json --cells 0 1 --extra 4 --nodes 128
python r30_bounds.py overlaps --incumbent docs/R29_INCUMBENT.json --cells 0 1 --candidates 2 3 18 44 --extra 6 --nodes 256
python r30_bounds.py budget --incumbent docs/R29_INCUMBENT.json --candidates 0 1 18 30 44 118 146 --swaps 2
python r30_bounds.py rotation --incumbent docs/R29_INCUMBENT.json
python r30_bounds.py responses --incumbent docs/R29_INCUMBENT.json --cells 17 18
python bench_r30_bounds.py --tiles data/tiles.json --output bounds.json
python bench_r30_bounds.py --tiles data/tiles.json --followups --output followups.json
python bench_r30_bounds.py --tiles data/tiles.json --unrestricted-families --output all-sizes.json
python bench_r30_bounds.py --tiles data/tiles.json --seven-baseline ../v1.10.0-checkout --output seven.json
python -B -m unittest test_r30_bounds test_r29_exchange test_r29_plateau
```

[Bound measurements and pair certificates](R30_BOUNDS_RESULTS.json) · [Rotation, donor and walk comparisons](R30_BOUNDS_FOLLOWUPS.json)
