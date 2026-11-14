# Physical exchanges and rotation repair

The 208-edge threshold was tested around three distinct 207-edge physical permutations. Every comparison allows all 160 tiles to rotate, including those outside the exchange. Independently scoring endpoint sets confirms each reported candidate. Both research commands optimize matched seams. Their threshold of 240 accepts any edge-perfect board; the independent Gold flag also requires one connected loop.

| Physical pieces moved | Permutations per center | Original center reaching 208 | 52-component center reaching 208 | Third plateau center reaching 208 |
|---|---:|---:|---:|---:|
| 2 | 12,720 | 0 | 0 | 0 |
| 3 | 1,339,840 | 0 | 0 | 0 |
| 4 | 236,649,240 | 0 | 0 | 0 |
| 5 | 36,096,897,408 | 0 | 0 | 0 |

The unchanged physical permutations also peak at 207. A 208-edge board must move **at least six physical pieces from each of these three centers**. Neighborhoods from different centers overlap. Adding their counts would overstate the number of distinct physical permutations excluded globally.

The physical-center fingerprints in table order are `51c26e4e47abf28f`, `d297c8f5cbdf792d`, and `840affdf318119d3`. These are the first sixteen hex digits of SHA-256 over the Python list representation of the zero-based physical tile indices in cell order.

## How to continue from a saved board

Run from the solver directory using the project Python environment and prepared tiles. Both commands accept `--seconds`. Zero runs until the shell or connected plateau finishes, or until the plateau's center budget is reached.

```sh
python r28_exchange.py --incumbent runtime/best.json --distance 6 --target 208 --backend cuda --database exchanges.sqlite
python r28_exchange_plateau.py --incumbent runtime/best.json --floor 206 --target 208 --centers 1000 --database plateau.sqlite
```

The shell has a NumPy CPU backend for any physical distance from 2 through the number of cells, and a CuPy CUDA implementation for distances 2–6 on unrestricted boards. Its canonical ordering enumerates all four-cycles and pairs of disjoint swaps at distance four, and includes every cycle partition without fixed points at higher distances. Direct rank lookup resumes large combination and permutation cursors. A batch transaction records both coverage and any exact score optima. After interruption the unfinished transaction can be replayed. The cache includes the full allowed placement domains, tile and topology fingerprint, and physical permutation. It also works across different shell centers. A missing fallback CP-SAT dependency or unresolved exact solve leaves the transaction unfinished.

The plateau walk expands one representative per physical permutation because it globally reoptimizes rotations before considering the next swap. The 207-floor connected component of all three centers closed after **38,160** distinct center/swap queries. The star bound rejected 38,034, 125 fixed permutations received exact solves, and one exact score was reused. All three 207 centers have been expanded. A 206 floor opens new paths. Its configured center budget and remaining frontier are always reported.

At the 2026-11-14T17:09:43+00:00 pilot snapshot, the six-piece shell on the 52-component center had covered 68,217,733,120 of 5,616,212,352,400 physical permutations with zero 208 hits. The 206-floor walk had expanded 52 centers and had 238 queued. These two scopes are partial.

## Evidence for workstreams 1–9

1. **Physical shells.** An upper bound for each center rotation counts how many of its incident seams could match, allowing each neighbor to rotate independently. The best of these local rotations upper-bounds one cell's contribution. Summing all cell bounds and dividing by two upper-bounds the entire board. A small exchange changes only the participating cells and their immediate neighbors. Disjoint neighborhoods add directly. Overlapping neighborhoods use a simultaneous full local recomputation. All distance-three and distance-five candidates failed this test. At distance four, only four queries across the three centers needed an exact rotation optimum. Each was at most 206.
2. **Rotation messages.** For fixed permutations, test each rotation against the current score threshold by changing its star and incident neighbors. Eliminate restricted variables exactly and cache the resulting factor tables with all input axes, dimensions, and factor entries. Cached argmax masks reconstruct tied optima. On 24 real physical swaps the threshold filter removed 8,515 rotation choices and closed 27 stronger trial thresholds. The elimination steps computed 420 table entries and reused 4,545. Small exhaustive tests checked all optimal orientation ties. The caller falls back to CP-SAT when elimination exceeds the factor budget.
3. **Partial exchange upper bounds.** The benchmark applies independent cell maxima, a physical assignment dual, distinct neighbor matching, and unequal endpoint weights to incomplete exchange plans. All four agreed on the five real connected neighborhoods screened. The fixed-permutation screen is the retained shell default. The more expensive partial variants are available for reproducible experiments.
4. **Strict dominance.** Minimum summed substitution gain removes a rotation only if the alternative wins on every combination of neighbor assignments. The real samples yielded zero strict single-rotation deletions. Threshold conditioning still reduced those domains. Exact same-inventory two-cell substitution dominated 8 of 9 tested local rotation combinations when the exterior was fixed. Scope this rule to seam optimization. The joint connectivity model retains separate component checks.
5. **Tied independent-set assignments.** A verified integral dual identifies all zero-cost assignments and rotations. Integer slack layers generate ranked alternatives; an existing dual is reused only after checking feasibility and complementary tightness. Three tested blocks of 8, 16 and 57 cells each had one optimum at 207/52. The 12-cell test returned one 207 followed by nine 206 alternatives at slack one.
6. **Connected and forest blocks.** Exact joint placement uses all physical pieces belonging to the selected block union and the fixed exterior. Connected blocks of 2, 3, 4, 6, and 8 cells and nonadjacent controls of equal size all closed the 208 threshold on the selected positions. Two- and three-dimer shared-inventory unions also closed it. The broader defaults retain their existing settings.
7. **Rigid patches.** Proper orientation propagation enumerates congruent placements and checks cyclic seam order and every induced adjacency. Tests of 256 two-dimer and 512 three-dimer cyclic trades peaked at 203 and 202. Four tests releasing an additional two-cell cut after rigid translation closed the 208 threshold. No new incumbent resulted.
8. **Seam and component thresholds.** The CP-SAT model enforces matched-seam and actual strand-component limits simultaneously. It contracts fixed connections, then asks for a sufficiently large acyclic subset of enabled segment joins. A separately decoded board checks both thresholds. At 207 with at most 51 components, the tested 2-, 3-, 4- and 6-cell neighborhoods were infeasible. The eight-cell query reached its short benchmark timeout. A longer follow-up closed 207/51, 206/51 and 205/50 on that patch in under ten seconds each. A complete small-board enumeration independently checked the joint decisions. Score and connectivity thresholds can be swept independently.
9. **Plateau pricing and commuting moves.** Changing placements only invalidates neighboring assignment price rows. The benchmark rebuilt 442 rows and reused 3,718, with a cold matrix reference test. Two disjoint swaps produced 132 orderings but only 66 distinct endpoint transformations. The reduction preserves read/write dependencies and disables reordering under intermediate acceptance or caps. The exact physical plateau uses persistent deduplication by physical permutation.

The machine-readable record includes the exact shell, plateau, and patch counters. [Exchange results](R28_EXCHANGE_RESULTS.json)

To reproduce the controls, run `python -m unittest -v test_r28_exchange`. To benchmark a new incumbent run `python bench_r28_exchange.py --tiles data/tiles.json --incumbent runtime/best.json --output exchange-comparison.json`. The counters measure threshold coverage and work performed on the stated incumbents.
