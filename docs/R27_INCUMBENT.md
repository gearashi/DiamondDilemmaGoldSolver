# Round 27 incumbent repair and construction

The original board used here has 160 distinct physical pieces, 240 seams, 207 matched seams and 54 strand components. All 160 placements belong to the active domains. A Gold solution needs all 240 seams and one closed strand. All published scores use the original tile masks. The independent validator also checks strand components for every recorded board.

## Retained repair

Enable **Incumbent polish** in the durable GPU sampling and DFS hybrid to run up to eight CPU refinement rounds on a legal saved full board. The option records completed rounds and candidates in the hybrid checkpoint. The assignment and rotation passes run on the CPU, including when the next phase uses a GPU. Subsequent exact DFS keeps its existing coverage. A separate command supports work on a copy of any full-board save.

```shell
python r27_incumbent.py data/tiles.json runtime/best.json --seed 0 --rounds 18 --output improved.json
python r27_incumbent.py data/tiles.json runtime/best.json --rotations-only --alternatives 2 --seconds 15 --output fixed-permutation.json
```

For proposal 4, the solver chooses nonadjacent movable cells, gives each physical tile and cell its best legal rotation score against the frozen neighbors, then solves the resulting assignment exactly. A random secondary weight can select equally good assignments without changing the optimum seam count. The OR-Tools assignment engine and the existing checked Hungarian implementation support the normal and Python 3.15 dependency profiles respectively.

For proposal 5, the rotation pass maximizes seam factors by eliminating orientation variables and reconstructing a consistent assignment. Large intermediate tables trigger conditioning on the supplied orientation of a cell. The result reports whether the optimization was globally exact for this permutation, a safe upper bound, the conditioned cells and the cost of the tables. The optional CP-SAT command optimizes the whole fixed permutation and excludes each returned orientation before finding the next one. Check each returned bound and exact flag when a time limit applies. This command needs the Python profile with OR-Tools.

On the original 207/240 board, CP-SAT proved an upper bound of 207 and found exactly one orientation attaining it. The best alternative scored 206 with a matching upper bound. Reaching 208 requires a different physical permutation. Across four seeds the alternating passes retained 207 while reducing the 54 components to 52. The existing CP patch finished each original-board comparison at 207 and 54 components. No tested method reached 208.

## Comparisons

The main screen used seeds 0 through 3 and each of three starts per seed. The starts were the saved board, that board after 18 legal edit proposals, and a freshly sampled legal assignment. Each entry used its corresponding identical start. Local operators received 18 rounds; the existing patch had up to 18 calls, 0.2 seconds per call and a five-second total cap. Population comparisons used eight lanes and 1,024 proposed boards. Construction methods have different amounts of work per proposal. Results include table prices, propagation work, proposals and wall time where available. Use the operation counts appropriate to each method.

| Method | Saved board, mean matched / components | Perturbed, mean matched | Fresh, mean matched |
| --- | ---: | ---: | ---: |
| Independent-set assignment | 207 / 53.25 | 187.75 | 152.25 |
| Assignment followed by rotation repair | 207 / 52 | 189.25 | 151.25 |
| Rotation repair only | 207 / 54 | 162.5 | 38.5 |
| Existing bounded CP patch | 207 / 54 | 157.25 | 57.25 |
| Replica exchange, host reference | 207 / 54 | 151.5 | 20 |
| Damped beliefs with decimation | 207 / 54 | 176.75 | 59 |
| Fractional assignment and projection | 207 / 54 | 148.75 | 104.5 |

### Closed experimental candidates

| Proposal | Experiment and disposition |
| --- | --- |
| 3, fresh GPU lane seeding | Initialized three quarters of fresh lanes from the incumbent and deterministic perturbations around broken seams. Kept the remaining random controls, original RNG streams, per-lane bests and Gold cache labels. Four paired CUDA runs of 262,144 proposals each yielded 207 in seeded lanes versus 72–76 in fresh lanes. Both treatments retained the saved 207/54. Four further pairs from the same polished 207/52 board used 12,582,912 proposals per seed per treatment. Fresh native bests reached 115–118 and seeded lane bests stayed at 207. Neither improved the independently retained board. The benchmark implementation supports replay. The standard dashboard omits this treatment. |
| 25, belief propagation | Damped cavity messages and physical-tile prices guided projection, with an optional decimation pass. Decimation reached 59 on fresh boards versus 38.75 without it. On five smaller residuals, support ordering and projection gave 13, 11, 15, 13, 15 out of 15. Beliefs gave 13, 13, 15, 15, 15. Exact block assignment had substantially better original-puzzle construction. |
| 26, recurring defects | Decaying seam penalties and periodic resets accompanied move tabu. A second comparison charged four exact block assignments per perturbed repair and 700 repair epochs per seed. Both control and treatment ended at 207/52 on all four saved-board runs. |
| 27, quality-diversity archive | Capacity eight for both archives. Loop, open-path and mismatch-sector bins reached a fresh mean of 24.75 versus 22.5 for the released Hamming archive. On perturbed boards they reached 152 versus 155.5. Extra graph traversals are counted. |
| 28, population annealing | Resampled descendants systematically using their Boltzmann weights at each temperature change. In four held-out, 4,096-proposal fresh starts it reached 44, 46, 50, 47 versus 34, 40, 43, 34 for equal-budget replica exchange. This was a real fresh-start benefit. Exact block repair provided the stronger construction available to the app. No improvement appeared from the incumbent. |
| 28, flat histogram | Visitation penalties over seam score produced 31, 30, 30, 29 on those four held-out fresh starts. |
| 28, cross entropy | Projected smoothed elite tile/cell/orientation preferences to legal assignments. The held-out fresh scores were 39, 33, 39, 35. The additional projection work is recorded. |
| 29, fractional quadratic assignment | Frank–Wolfe updates stayed in the convex hull of legal assignment vertices. Each step maximized the exact line quadratic. Rounded assignments then received rotation repair. The 104.5 fresh mean exceeded beliefs but trailed 152.25 from a direct block assignment. Only integral independently validated boards counted. |
| 30, half-turn crossover | Used actual verified orientation-preserving half-turns, paired-cell orbit exchange, connected and scattered boundaries, fixed-domain checks and canonical rotation deduplication. Sixteen attempted children per start received fixed-permutation repair. An independently perturbed, block-repaired parent had the stronger perturbed and fresh means of 176.25 and 110.25 versus 152.25 and 43.5 for geometric crossover. Independent-parent preparation costs three extra block rounds. |
| 31, seam-preserving physical rewiring | Completed the original unmatched mask classes and switched partners within a compatible class. Scored oriented vertex links and physical-strand components, then attempted the exact embedding checker when vertex counts permitted it. Three seeds each on the original and a planted multi-loop assembly, 256 proposals per treatment. Neither random nor topology-guided switching produced a new confirmed embedding. Derived maps were kept outside the board search. |

A follow-up explored a best-first plateau of exact nonadjacent two-cell swaps. It expanded 4,000 board states and tracked 38,869 distinct arrangements, retaining 207/52. All local exclusions applied only to that neighborhood.

## Reproduction

The reproducible scripts accept or read a *copy* of the incumbent at `audit/r27/incumbent.json` and the prepared tiles at `data/tiles.json`. The full candidate code vectors and local GPU snapshots stay in the audit directory. The checked-in results include the input code digest, seed-level scores, operation counts and native state hashes. [Machine-readable results](r27_incumbent_results.json)

```shell
python -m unittest test_r27_incumbent
python bench_r27_incumbent.py --seeds 4 --rounds 18 --proposals 1024 --output audit/r27/incumbent-screen.json
python bench_r27_incumbent_followups.py
python bench_r27_incumbent_perturb.py --epochs 700 --seeds 4 --output audit/r27/incumbent-perturb.json
python bench_r27_incumbent_rewiring.py --output audit/r27/rewiring.json
python bench_r27_incumbent_trades.py --boards 4000 --output audit/r27/trade-record.json
python bench_r27_incumbent_gpu.py --backend cuda --replicas 128 --dispatches 32 --seeds 4 --restore --output audit/r27/native-seeded.json
python bench_r27_incumbent_gpu.py --backend cuda --replicas 1024 --dispatches 192 --seeds 4 --from-polished --output audit/r27/native-seeded-deep.json
python bench_r27_incumbent_gpu.py --backend webgpu --replicas 32 --dispatches 2 --seeds 1 --gold --restore --output audit/r27/native-webgpu-restore.json
```

Both CUDA and Vulkan WebGPU ran on an RTX 2060. Split/resume tests compared the entire native state hash and all collected observations with continuous execution. Small finite tests also checked exhaustive assignments, restricted rotations, private domains, k-best enumeration, checked CPU fallback, malformed inputs, rotation-bound validity and replay of every tested population controller.

