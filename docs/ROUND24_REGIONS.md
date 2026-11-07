# Round 24 regional and residual reasoning

These are opt-in implementations and bounded research variants. The recorded
finite checks do not establish an original Gold solution or a universal
reduction in search work. An interrupted or capped calculation is unknown.

## Implemented mechanisms and integration

| Proposal | Implementation | Scope and integration |
| --- | --- | --- |
| 8 | `constraint_region_oracles.ResidualProofCache` | Keys include the exact remaining cell domains, physical inventory, fixed boundary masks, contracted outside path pairing, objective, puzzle fingerprint and explicit effective-root scope. Different fixed interior histories may share a completed UNSAT result. Arbitrary residual graph isomorphism is not implemented. `group()` supports equivalent-query batching. |
| 17 | `constraint_region_symbolic.DirectRegionDiagram` | Direct memoized frontier construction retains exact labelled assignment histories. DAG-to-DAG overlap joins recover supports without expanding every row. `DirectRegionSession` retains unfinished construction in the same process; this is not an on-disk continuation format. `RegionNetwork` retains four exact-query sessions across cancellation. |
| 18 | `InventoryZDD`, `inventory_supports` | Exact symbolic families of used-piece subsets, disjoint products and all-region inventory support. `symbolic_inventory` activates it in the ordinary regional network. |
| 19 | `constraint_region_elimination` | The full actual seam and physical-piece constraint graph is compiled. Exact bucket-tree inward/outward messages recover every unary support in one pass. Resource lifetimes end only after the final possible occurrence variable is eliminated. Constant factors are removed only when their relation equals the complete Cartesian product. Full and selected-region modes have explicit budgets; cutset conditioning is available as a bounded exact variant. |
| 20 | `eliminate(..., width=...)` | Mini-bucket splitting creates an optimistic relaxation of the complete factor system. Completed negative answers remain sound. |
| 21 | `relaxed_diagram_possible`, `refined_diagram_possible` | Bounded-width state merging unions possibilities conservatively. Still-possible relaxed queries are refined through larger widths; completed exact or negative answers stop the sequence. |
| 22 | `representative_matchings`, `representative_component_feasible` | A GF(2) basis of connectivity rows preserves existence for every possible outside matching within an identical inventory and boundary class. Limited to ten exposed endpoints for the reduction; larger groups retain all states. This is an existence-only oracle, not unconditional support/count compression. |
| 23 | `subset_convolution`, `subset_inventory_supports` | Exact integer ranked zeta/Mobius convolution joins disjoint inventory families. Only present cardinality layers are calculated. The generic transform accepts at most 16 pieces; integrated support queries default to ten. |
| 24 | `LazyRegionMaster` | Regional realizations are generated only when demanded. Completed failed inventory/strand interfaces exclude all equivalent internal histories. This is a bounded existence query; a capped parent never becomes a failed interface. |
| 26 | `CertifiedRelaxationPruner`, `priced_inventory_master` | GLOP proposes dual vectors, but exact rational inequalities authorize every infeasibility conclusion. The pricing prototype uses dynamic programming over regional DAGs to generate missing columns, and requires completed pricing before closing an infeasible inventory master. Feasible relaxations are not Gold witnesses. |
| 27 | `LabelledRouteOracle`, `LabelledRoutingPruner` | Paths maintain compatible cell orientation and physical-piece labels. A two-route query enforces shared labels and actual port capacity, allowing multiple strands in one triangle. Only Gold targets use this necessary connectivity condition. |

`RegionalResearchPruner(problem, target, options)` integrates exact elimination,
mini-buckets, relaxed diagrams, representative existence, subset inventory joins,
lazy interfaces, certified LP and labelled routing with explicit admission,
completed-query and unknown counters. It uses the full constraint system on
small or deeply narrowed problems and bounded optimistic selected-cell queries
otherwise. The full-constraint API remains available for capped original-board
experiments. Default app controls remain off until the combined benchmarks are
reviewed.

## Reproduction

```
python -B -m unittest test_constraint_region_symbolic test_constraint_region_reasoning_round24 test_constraint_region_network
python -B benchmark_regions_round24.py --output audit/optimization-round24/regions.json
```

Optional `--original-tiles /path/to/tiles.json` adds a 50,000-operation capped
full-original factor query. A cap is reported as unknown, not a result about
solvability. The release audit runner should bind the exact final source files,
interpreter and outputs; development outputs are not substitutes for that final
manifest.

The small independent checks compare full solution/support sets, integer joins,
all endpoint matchings in the declared range, certificate corruption, changed
objective/root scopes, cancellation and resumed work. Python profiles without
OR-Tools explicitly skip backend-dependent tests; pure symbolic algorithms
remain available.

## Development discrimination results

- Seven-cell ring: all 5,040 concrete histories agree; eager enumeration performs
  60,620 candidate trials, direct construction 889 trials and 128 states. Sliced
  construction repeats zero candidate trials. This counts candidate trials,
  not every lower-level helper operation.
- Six regional inventories over twelve pieces: both algorithms support all 66
  inventories per region; baseline recursive allocation tries 135,102 inventory
  alternatives, versus 19,033 ZDD primitive operations. The primitive types differ
  and this is not an elapsed-time ratio.
- Synthetic 160-cell ring with long-range paired physical resources: one exact
  support pass uses 2,080 factor operations and live-resource width two. A
  full-inventory frontier compiler exceeds 4,096 states. Eighty-one independently
  validated global witnesses cover every retained placement. These are synthetic
  pieces, not the original Gold input.
- Connectivity: 105 eight-endpoint matchings reduce to eight representatives,
  checked against all 105 possible outside matchings. Support multiplicity and
  history counts are deliberately not claimed preserved.
- Ten-piece homogeneous inventory join: 63,504 explicit pair tests become 16,384
  integer transform operations, with exactly the same 252 joined histories.
- Lazy eight-cell realization requests two regional rows; eager local tables
  contain 3,360 rows. The resulting full Gold arrangement is independently
  validated.
- Pricing generates 24 columns from a ninety-history feasible relaxation. A
  separate shortage is closed only after complete pricing and exact rational
  certificate checks.
- An optimistic connected graph admits individual routes but no pair of routes
  with consistent orientation labels; the bounded labelled oracle rejects that
  obstruction. Exhaustive small real-puzzle tests preserve every Gold solution.

Original-instance gains, best combinations and default-policy decisions must
come from the separate final frozen-source benchmark report.


## Exact-search integration and owned roots

`residual_reuse` now queries contracted completed-problem keys before DFS
propagation and stores only consumed failed nodes or fully exhausted Python
frames. Timeout, cancellation, unfinished probes and discarded backjump frames
never create negative entries. Native strong-pruning callbacks use the same
completion hook; native basic-only rejects and native parent exhaustion are
conservatively not cached. Cache statistics distinguish queries, hits and live
completed entries, with warm-slice deltas for counters.

A public integration regression exposed an older ownership bug: importing a
restricted DFS root could learn an empty local clause under the broader puzzle
scope. Checkpoints now persist `owned_root_domains`; every local nogood, SAT/LCG
helper and mask-proof verifier binds to that root. Legacy v2 checkpoints adopt
their actual first-frame domains, keeping pending search but dropping optional
old local-mask claims whose original scope was not explicit. Splitting work
creates new child roots and restricts parent mask records into unverified child
claims. Conjunctions already impossible in a child are omitted; parent-root
proof certificates are discarded. Every remaining claim must be independently
verified under the child root before pruning. Reusing
a mutable learning dictionary with a different root raises instead of silently
transporting local proofs. Immutable worker support caches now include owned
root in their keys, which may reduce cross-job table reuse. No search coverage
is discarded by this safety boundary.

The separate coverage-ledger regression similarly prevents a restricted
standalone checkpoint from certifying exhaustion of the full original puzzle.
`test_constraint_residual_reuse_round24.py` and
`test_constraint_coverage_scope_round24.py` cover these public solver paths,
legacy migration, split roots, stop/cap handling, native-controller scope and
independently validated Gold solutions outside a completed child region.
