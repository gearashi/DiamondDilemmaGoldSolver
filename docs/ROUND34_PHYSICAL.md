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

## 19. Watched physical mates with nonsingleton owners

For each checked conflict, a tile side is forced to a particular mate if every viable remaining pose of the owner gives exactly that mate across its geographic seam. Both incident owners can force it. Rechecking only pairs whose owners occur on changed cells or their neighbors lets the same watcher handle backtracking. A two-pose-per-cell ten-tile branch closed before any singleton appeared. A second rotation removed a unit placement earlier, and both independently validated Gold boards remained in its sibling. Replaying identical input performed zero further pair checks. A seven-mate certified proper circuit on a split planted puzzle also closed two-pose branches. Imported checked conflicts with multiple bonds can post these derived domain units in SAT/CP.

## 20. Physical restrictions in ordinary SAT and CP Gold engines

The application's Gold option gives direct SAT, persistent SAT, CP-SAT, the direct CP-SAT circuit, and classic CP their geometric implications for forbidden and mandatory physical partners. The released root generates 11,520 forbidden placement pairs and 7,680 required implications across 240 seams. On seed 34's five-mover split-loop control, SAT decisions fell from 61 to 7, CP-SAT branches from 125 to 0, and classic CP branches from 43 to 4. The direct circuit proved infeasibility with zero branches both ways. All four engines returned edge-perfect for the edge-only controls and accepted the planted single-loop positive. Persistent SAT keeps its original proof base and validates saved native learning against that base separately.

## 23. Independent phase cubes

`r32_fresh.py cube --independent-cubes` and the corresponding fresh option use independent coordinates of the mandatory-bond GF(2) system, anchoring physical tile zero to remove color complement. Branch variables maximize the number of immediately determined phases, then the number of shortened small equations. On the full puzzle 114 dimensions remain. Width 32 fixed 48 physical tile phases; numbered tile selection fixed 35 and only increased rank by 31. An exhaustive ten-tile geographic lift compared the entire numbered and independent coordinate spaces. It found the same single verified Gold assignment with 16 geographic solves in place of 512. Independent cubes have a distinct fresh-search policy identity; geographic negatives remain scoped to actual tile pins.

## 24. Mandatory physical poses during skeleton enumeration

`r32_fresh.py skeleton --mandatory-poses` carries each mandatory component's live pose bitset as its cells acquire parity patterns and includes that state in memoization. On the 16/32/48-mover real residuals above, each of the sole parity patterns failed a required component before any skeleton state expanded. The Hall-only searches had visited 203/205/793 states. A planted ten-tile Hamiltonian ring had two required bonds. On the seed-34 three-alternative residual its search shrank from 46 states and two parity patterns to 19 states and one pattern, retaining the independently validated Gold board. This is a Gold-specific geographic compatibility check.

## 25. Count-preserving local flips and physical lifts

`r32_fresh.py skeleton --contour-walk --limit 16` walks one- and two-vertex moves, accounts for shared triangles, skips previously emitted contours, and reassigns only the affected donor cells. On the actual puzzle, 16 different seed-zero walk contours cost 340 assigned cells. Sixteen independent conditioned contours cost 2,560. Mean pattern distance from the first contour was 56.3 versus 121.1 cells for the independent starts. Three geographic lifts per method under identical two-second budgets all returned UNKNOWN. Six equal-budget probes on a planted ten-piece ring instead found three Gold lifts via the walk and two from independent seeds. The sampler's use is for inexpensive nearby alternatives.

## 26. Signed donor prices shared across phases

Signed repair retains the existing versioned SQLite table for completed three-side prices. Cold rows now combine phase-independent dart compatibility templates with one signed vector per exterior dart and phase configuration. Seed 0-7, 16 full-data repair rounds each, produced identical board codes and independently recomputed scores at every round relative to the pre-change solver. The 7,852 response rows previously computed 23,556 signed vectors; the same 128 assignments needed 3,832. A total of 480 physical templates were built and 3,352 later phase-specific builds reused one of them. Template vectors occupied 230,400 bytes and the final phase's vectors occupied 229,920 bytes, excluding Python container overhead. Changing the forbidden set rebuilds scoped templates.

Run `python bench_r34_physical_followups.py --tiles ../data/tiles.json --root-lifts` to regenerate [the measured results](R34_PHYSICAL_FOLLOWUPS.json). The two physical test modules, ordinary SAT/CP and persistent SAT, and related geometry, Gold and fresh suites passed 72 tests. The maintained public board remains at 207 matched edges; these exact controls and scoped reductions do not establish a higher score.

## 48. Resume published heuristic checkpoints

A real v1.14.0 optimizing crossover snapshot previously failed its source binding before resuming its first saved node. The release's transitive heuristic manifest had 46 dependencies. Round 34 adds the coverage-index module and changes three older source images. The historical heuristic only used the original digest function and did not execute either physical Gold filter. The heuristic compatibility table binds each reviewed current image to the exact corresponding historical LF/CRLF hashes. Its v1.14 layouts preserve the two historical Gold files, and v1.12 retains its original two-file omission. The Git layout uses LF for the 49 Python files in the full controller identity and CRLF for its native C file. Each source addition or image change must still match its pinned identity. Exact-search coverage identities have separate checks.

The regression uses a live one-node snapshot exported by revision `8e010cc8b716f93b1567211cf04aa0318e3fb4b4`, plus independently collected headers for the two uniform text layouts. Each of the three restored controllers preserved its in-progress search model, offered the same candidates, and reached the same verified Gold board after two more proposals at total node count three. Wrong policy, narrowed domains, altered current or saved hash, and incomplete historical source sets all failed restoration. The previous v1.7/v1.12 compatibility checks also pass. Run `python -m unittest -q test_r34_checkpoint_compat test_r27_incumbent test_gpu_recombination_round26` (30 tests passed). [Historical control](R34_V114_CROSSOVER.json)

### Exact terminal evidence sources

The constraint runner now snapshots the transitive local Python imports and referenced C, CUDA and WGSL resources. The integrated checkout archived 206 source files, including the physical Gold matcher, geographic mate encodings, coverage index, GPU lane resizer, both domain CUDA engine modules and the generic CUDA kernel. A runner fixture injected an infeasible terminal result and checked its archived raw bytes and digests against the source. Removing one physical dependency and recomputing the record's self-digest still made the runner reject the exhaustion marker before launching a solver. An independent file fixture verifies changes to nested shader bytes and a missing optional engine. The runner/checkpoint suites passed 54 tests. Run `python -m unittest -q test_constraint_search test_r34_checkpoint_compat`.

### Classic GPU prefix replay and DX12

The full puzzle has three forbidden physical mates which match their edge masks and do not themselves close a short loop. On the RTX 2060, all three were restored as consumed children above the ownership floor on CUDA, Vulkan and Direct3D 12. For each API the static-order and dynamic-order cases unwound exactly three children, kept all three owned prefixes active, and retained all sibling cursors, shifts, prefix identities and random states. The dynamic child had actual rank 1 and static rank 37. The independent Gold GPU suite passed 18 tests with CUDA and Vulkan enabled, including a finite-tail oracle and legacy 4→1→3 lane restoration. Reproduce each full-puzzle control with `python bench_r34_physical_followups.py --tiles ../data/tiles.json --gpu-prefix CUDA`, then with `Vulkan` and `D3D12`.

With the native Windows FXC compiler, the original v1.14.0 WGSL at `8e010cc` failed to compile on D3D12 in both static and dynamic order with error X3511. The pre-fix Round 34 shader reproduced both failures. Spelling out the triangle's three neighbor checks and the three local Hall tile slots eliminated the nested unroll while preserving candidate scan order and the full dispatch budget. The production-sized shader compiled on that same FXC with static flags, Hall, dynamic ordering, connectivity, and all GPU-side options together. The three real prefixes described above also passed under FXC without a Windows SDK compiler override.

Under the default D3D12 compiler, the hardware Gold, reasoning and WebGPU suites ran 32 tests in 514 seconds, with 25 passed and seven CUDA-only/interop cases skipped in this Direct3D-only invocation. The finite-tail checks covered ordinary matched-edge enumeration, per-policy Gold counts and combined reasoning with a restored checkpoint. The separate real-puzzle D3D12 classic comparison reproduced 41 candidate tries and 26 placements before the physical filter, then zero tries and zero placements with all three owned prefixes exhausted. Set `WGPU_BACKEND_TYPE=D3D12` and `DIAMOND_TEST_WEBGPU=1`, then run `python -m unittest -q test_gpu_gold.GoldWebGPUTests test_gpu_reasoning test_webgpu` to reproduce.
