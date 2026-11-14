# Round 34 repair and 208 experiments

Baseline is v1.14.0 (8e010cc). The candidate numbers refer to the solver-wide register.

## 10. Ranked five-swap routes

The multicenter runner accepts all seven five-transposition cycle families. Each support has 120 (6), 504 (2,5), 420 (3,4), 1,260 (2,2,4), 1,120 (2,3,3), 2,520 (2,2,2,3), or 945 (2,2,2,2,2) endpoints. Only the requested family is generated. The least remaining cell owns its directed cycle. Repeated cycle lengths have one owner. A size formula counts a whole support without generating an order, and mixed-radix indexing seeks any individual endpoint. Four-swap cuts retain their original order. Nine- and ten-cell ownership uses the actual requested support size. Global defect-budget keys include the number of swaps.

Checked every rank in both directions, compared the original distance generator, and independently compared sets to all permutations on 6 through 10 labels (4,037,760 permutations). Tests on nine- and ten-mover scopes with two distinct centers recovered the planted optimal board and reopened the database with zero new pages. Example: `python r30_multicenter.py --family 2,2,2,3 --database runtime/five-swap.sqlite --target 208 --page 64`. The checkpoint counts finished support ranges. An interruption inside a support page can still repeat that incomplete page.

## 11. Disconnected five-swap endpoints

A cycle with mutually nonadjacent moved cells and an unchanged neighbor at every incident seam can be undone when the sum of its largest incident-score gains across every exterior orientation is nonpositive. This comparison maximizes only over legal rotations at each changed cell. The remaining cycles must also identify an actual negative predecessor (shipped physical shell or checked negative ranked support). Missing, skipped or positive predecessors cannot exclude a candidate. Tests exercise a neutral removable cycle with and without completed four-swap coverage, plus positive five-swap fixture searches.

Opportunity measurement on the published 207 center used 250 uniformly sampled support/order pairs per family, one Python Random stream seeded 6401, families in the order above, and only the published predecessor shells. The component rule independently excluded 0, 93, 80, 160, 133, 54 and 0, respectively (520/1,750). Other search filters can overlap these exclusions. Full-campaign first-rejection counts and larger interacting component comparisons remain open.

## 5. Shared defect-cover certificates

The stream of typed donor prefixes now retains checked infeasible CP assumptions as sets of seam-failure clauses. A clause can transfer only when the current obstruction involves the same geographic failure edges and a subset of its eligible movers. Both the mover and failure budgets must be no larger. This permits reuse after changing fixed donors, center, pool or target when the actual obligations still imply the cut. The cache retains up to 1,024 cores in each run. Positive and interrupted solves add no negatives. Finite cross-center tests compare every answer with a fresh solver and cover missing rows and both changes of budget direction.

Across all 80 published prefix records, three configurations per record used the full movable pool (3 extra movers, target 208), cells 0-79 (3, 208), and the full pool (2, 209). Reusing checked cores avoided 98 of 240 CP calls. The 142 remaining calls learned 4,105 clauses in total; 195 of 240 optimized queries closed. With a four-second CP limit, nine paired controls contained at least one UNKNOWN and all other pairs agreed with fresh solving.

## 6. Exact mover-quota multipliers and reduced costs

The isolated-extra family bound now finds the minimum of the full Lagrangian envelope using integer ratios at assignment-line intersections. The assignment dual at that ratio bounds every forced donor arc. Child branches reuse the parent ratio and can close before another assignment query. Random finite tests compare the rational optimum and each forced-arc ceiling with every permutation on two through six rows, including sparse matrices without perfect assignments.

On the published 207 center, 80 independent two-active-cell cases used a 22-cell sampled additional pool and extras 0 through 4 (Random seed 20706). Checked 99 assignments. Two root bounds tightened by one doubled-score unit each. Neither crossed 208. At twenty matched local thresholds, the same one case closed within 35 nodes, using 233 visited donor branches in aggregate against 393 in v1.14.0. Dual certificates cut 164 donors before recursive re-solving.

## 7. Multiple contacting cells in joint repair

The fresh engine accepts `--contacts 2` or `--contacts -1` for an unrestricted induced pool. `--disjoint-contacts` requires each changed cell to have at most one changed neighbor. New options have distinct trajectory identities. The original adjacent and nonadjacent configurations retain their previous semantics. Four-cell optima match enumeration of every donor and orientation with two contacts, two disjoint contacts, unrestricted contacts and unrestricted disjoint contacts.

Eighteen connected six-cell pools on independent fresh original-puzzle boards used seed 734 and the same CP deterministic budget 0.12 per call. One contact gained 41 seams in total; two contacts gained 46, two disjoint contacts 43, and unrestricted contacts 56. They improved on one contact in 5, 2 and 14 cases respectively; all 72 calls proved their local optima. CP branches were 30,528 / 42,117 / 32,552 / 41,388. These totals compare locally improved seams across different boards. Broader trajectory and width-directed pool selection remain open.

## 8. Small and bipartite swap conflict components

With `r31_fresh.py --exact-packing`, connected positive swap conflicts of up to 22 vertices use exact independent-set search. Larger bipartite components use a weighted minimum cut when OR-Tools is available. In the remaining large components the same highest-priority greedy swap is fixed and the residual is split again. Independent tests enumerate all subsets on twelve-vertex problems and demonstrate a larger bipartite cut beating greedy. On 48 random original-puzzle starts (seed 834, alternating full board and 30-cell pools; last 24 received twelve independent-set repairs), 49,841 positive swaps were priced by each version. The new pack gained eight extra matched seams across eight cases, and never decreased any case's score.

## 9. Gain from contacting nonpositive cycles

`r31_fresh.py --pair-cycles 2` or `3` evaluates all legal rotations of local cycles up to that size. Two disjoint cycles can be retained when at least one is independently nonpositive and their shared physical seams make the joint change strictly better than either. The packer compares with its singles-only result before applying a board. Direct exhaustive rotation tests and full-board rescoring verify the signed correction.

For each length limit, 32 connected eight-cell neighborhoods used independent original-puzzle boards with RNG seed 934 across both groups; half were polished for eight independent-set steps. At length two, 2,313 joint prices gave four additional matched seams in four boards versus positive singles. At length three, 33,737 joint prices gave 14 additional seams in 13 boards. Neither mode lost a seam in any matched comparison. Trajectory comparisons are still needed before choosing defaults.

## 13. Incremental exterior elimination across supports

The 208 connected-pool runner with `--exterior` now shares immutable fixed-edge tables and exact separator responses between supports of the same bound. Each cached result binds the eliminated variable, remaining axes, widths and all exact input factor bytes including multiplicities. Changed factors force recomputation of their dependent responses. The cache bounds response storage, evicts by recent use and rejects another center. Width and conditioning limits still apply.

Compared cold/warm construction on the published 207 center for 16 distinct six-cell supports. Half shared a dense five-cell prefix (0,1,2,3,4); half a dispersed one (0,28,73,106,132). The sixth cell was sampled with seed 1334. All 16 compiled. All 64 comparisons at order ranks 0,17,68,89 agreed. Warm compilation evaluated 5,652 table entries versus 37,368 cold, reusing 1,880 separator responses and 3,351 original edge tables. Response storage was estimated at 208,692 bytes. Independent small-board tests also check the exact score and witness under cache eviction and reject a foreign center.

## Remaining work and next experiment

- **12, adaptive fresh centers.** Read-only load unrelated finished trajectories, extract distinct verified board/defect sets, calculate the optimistic score headroom and overlap. First compare three deterministic strategies (score only, defect diversification, union novelty in sampled physical endpoints) on the same small ranked intervals using r30_multicenter.run. Increase the center union on the same checkpoint and compare old coverage, unique endpoints, first witness and exact solves with independent fresh runs. Freeze the ranked control seed and retain the physical board hashes.
- **14, failed-seam 208 master.** Build a restricted exact donor and rotation subsolver with assumptions on geographic seams. Solve a master hitting set of allowed failures, validate each subsolver core independently, and only publish a negative on exhaustion. Compare feasible placements and exact calls with brute donor/rotation controls both above and below the true maximum before testing an incumbent pool.
- **15, component swap budgets.** Enumerate component boundary responses indexed by local physical inventory and exact swap distance, max-plus convolve compatible rows, and handle cross-component donation by an explicit upper relaxation or fail-open. Test a crossing-donor positive and an impossible bound for which independent component scores add incorrectly without inventory.

Open follow-ups on tested items: sweep whole fresh trajectories under matched per-seed budgets and report threshold rounds for #7–9 before changing defaults; compare width-aware geographic pool selection with random pools on the same seeds. For #11, separately measure alternative multicenter/partial-checkpoint predecessor coverage and stronger comparisons for directly touching cells. The 520 endpoint exclusions measure a rule alone. For #5, if nine time-limited pairs are used as evidence of exact equivalence, rerun them to completion first. No full 208/Gold solution or complete five-swap shell was found in this batch.
