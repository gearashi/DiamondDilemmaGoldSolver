# Gold solver v1.8.0

Thirty-five proposals were implemented as production changes, bounded alternative engines, or comparative experiments. The original input still has no verified Gold solution. The best board in this round matches 207 of 240 seams, uses every piece once, and has 52 line components. The pre-release incumbent matched 207 seams and had 54 components.

## Retained settings

For an existing board on the tested i7-4790 and 6 GB RTX 2060, choose **Experimental GPU sampling + repair + DFS**, enable **Optimize the saved board by reassigning pieces and rotations**, and resume from your incumbent. The first eight repair rounds run on the CPU. Leave duration **Unlimited** for a continuing search. The experiment established a benefit from this repair. It did not establish an optimal replica count. Exact-search users can keep their existing DFS configuration. CUDA domain users can enable **lane balancing** to share propagated assignments during donation. This is a different engine from compact GPU DFS.

For a separate, inspectable output without starting the dashboard, run `python r27_incumbent.py data/tiles.json runtime/best.json --seed 0 --rounds 18 --output improved.json`. Use the full Python 3.12 profile for native SAT, CUDA and the optional CP-SAT rotation certificate.

## Proposal register

| # | Mechanism | Outcome |
| ---: | --- | --- |
| 1 | Physical-side matching | Recovered seven known forced bonds and no new exclusions at the original root within the recorded cap. |
| 2 | Connected complementary regions | Implemented dual-region reachability, sealed pockets and conditional passage checks. Kept bounded diagnostics. |
| 3 | Incumbent-seeded GPU replicas | Tested fresh, perturbed and restored lanes. The longer 100,663,296-proposal comparison produced no improvement over 207/52. Benchmark only. |
| 4 | Reassign nonadjacent pieces exactly | Retained in hybrid repair. Each assignment maximizes the seam sum against frozen neighbors while using each selected piece once. |
| 5 | Optimize rotations of a fixed permutation | Retained with explicit bounds. CP-SAT proved that the original incumbent permutation has one 207-edge rotation vector. Its best alternative scores 206. |
| 6 | Seam-first tables and tries | Available in the model comparison. Building seam choices and then recovering physical pieces added work on the original root. |
| 7 | Sew pieces before embedding | Implemented oriented vertex links, partial embedding checks and complete oriented embedding verification. Kept for bounded experiments. |
| 8 | Trace the physical tour | Implemented grouped side bonds, successor choices, empty sides and embedding. The wide original scope exhausted the pilot budget. |
| 9 | Mandatory-strand separators | Tested all 66,430 pairs of strands on the original optimistic graph. No additional separator exclusion was found. |
| 10 | Tour positions modulo 5, 73 and 365 | All three original moduli were exercised. The fixed incomplete board was decidable. Full-root model construction hit its work cap. |
| 11 | Colored exact cover | Tested reversible exact cover and traversal continuation against independent enumeration. Retained the comparative implementation. |
| 12 | Count-pattern search and exact lifting | Implemented coarse branching, exact lifts and rejection of failed patterns. No full-root default change. |
| 13 | Euler bounds on the two regions | Tested local noncrossing structure and attainable regional totals. Root screening removed zero values. |
| 14 | Noncrossing interface words | Retained in the regional counter. Face, strip and hemisphere cases held the same four solutions with measured Python pairing-key sizes of 56, 140 and 224 bytes, against 1,360, 3,976 and 6,208 for pair lists. Shared inventory and active-port data are excluded. |
| 15 | SAT vertex elimination | Retained a bounded sparse residual solver. It used 175 SAT decisions on the four-cell mixed case against 192 for lazy checking. |
| 16 | Whole-board MaxSAT | Full-board threshold 208 remained unresolved. The bounded CLI reports the witness and the proven upper bound separately. |
| 17 | Unavoidable defects | Retained verified lower bounds and incremental seam projections. Caching reduced 13,736 seam-pair evaluations to 5,868 on the measured queries. |
| 18 | Small decision sets | Retained matching and forest remainder solvers, including an optional bounded tail in dashboard DFS. A six-cell ring fell from six DFS nodes to one. Held-out visits fell from 301 nodes to 254, with extra work to identify the structure. |
| 19 | Dependency-directed dynamic backtracking | Retained a scoped solver. It reduced assignment operations on held-out cases and incurred extra reason bookkeeping. |
| 20 | Residual symmetry | Retained verified orbit representatives for Gold existence. One fixture fell from 11 visits to two. Mapping checks were substantially more expensive there. |
| 21 | Lifted LP and polynomial certificates | Retained bounded degree-two experiments with exact verification. Both detected an odd-cycle contradiction missed at degree one. |
| 22 | Determinant counting | Tested exact cofactor counts against enumeration on bounded joins. Direct connectivity checking remains cheaper on the measured cuts. |
| 23 | Two-child probing and learned splits | Exact probing cut held-out visits from 301 to 258 and more than tripled counted operations. Learned selection took 383 visits and stays out of the default path. |
| 24 | Concurrent lookahead and SAT | Helper supplied checked witnesses or proved units. Extra native calls outweighed the saved owner calls across the five measured cases. Benchmark only. |
| 25 | Belief-guided construction | Damping and decimation improved some starts. Direct block assignment produced stronger fresh boards. |
| 26 | Recurring-defect penalties | Four saved-board trials tied the unpenalized control after 700 repair epochs per seed. |
| 27 | Geometry-diverse archive | A small fresh-start gain did not generalize to all starting conditions. Kept as an experiment. |
| 28 | Population annealing, flat histogram, cross entropy | Tested separately. Annealing beat replica exchange on four held-out fresh starts. Block repair produced better fresh boards in the larger comparison. |
| 29 | Fractional quadratic assignment | Projected fractional solutions to legal boards before scoring. Fresh mean was 104.5 against 152.25 for direct assignment. |
| 30 | Half-turn crossover | Compared canonicalized orbit children to independently prepared parents. Independent-parent repair had stronger fresh and perturbed means. |
| 31 | Seam-preserving partner swaps | Checked candidate physical assemblies for orientation-preserving embeddings. Physical seam compatibility alone did not yield a new verified Gold board. |
| 32 | Share propagation at CUDA donations | Retained with native lane balancing. On a planted fork, the children reused ten assignments and nine strand joins each. Post-donation joins fell from 24 to six. Both modes enumerated the two solutions once, and device state occupied 53,232 bytes in either case. |
| 33 | Batch lanes by next operation | Grouped dispatch matched native state after every dispatch in twelve comparisons with 1, 8 or 128 lanes. It increased runtime in all twelve. Three paired, 128-lane runs on the original board also performed the same 392,320 support checks with identical final states and longer grouped dispatch times. Benchmark entry point only. |
| 34 | White-solution controls and orientation seeds | Both Silver boards, Red and Blue validated as single loops of 48, 48, 102 and 179 segments. A wrong rotation and a single-tile reversal were detected for each. On 32 matched full-board permutations per Silver layout, seed patterns averaged 6.41 and 6.13 Gold seams against 6.63 and 6.56 for shuffled orientations. |
| 35 | Source sensitivity | Refit the eight lowest-margin pairings. All left the original board at 207 seams. One changed the component count to 53. Seven deliberate endpoint shifts lowered the score to 206. The candidates had weaker image support. The prepared Gold input was preserved. |

## Reproduction and limits

The [structural report](R27_STRUCTURE.md) describes the tested limits and alternatives for proposals 1, 2, 6–15 and 22. The finite comparisons include independently validated outputs, adverse cases and scoped interruption tests. A full-root work cap, timeout or sampled failure leaves the unresolved branch open. Statistics from planted or constrained fixtures apply to those fixtures. For the smaller diagrams, one globally bijective assignment resolves competing number matches. Five path readings needed manual review against the white-only boards.

- Structural suites and full-input caps are in [the finite results](r27_structure_finite.json) and [the full-size results](r27_structure_full.json). Run `python bench_r27_structure.py --help` for the independent and planted cases.
- Reasoning algorithms, bounds and reproduction commands are in [the scoped reasoning report](R27_REASONING.md).
- Per-seed scores and native resume checks are in [the incumbent experiments](R27_INCUMBENT.md).
- Native dispatch and donation evidence is in [the CUDA measurements](r27-gpu-evidence.json). Run `python benchmark_r27_cuda.py` on CUDA.
- After preparing the piece sheets, run `python r27_diagrams.py --fetch` on Python 3.12 to reconstruct the four controls. Results include the paired permutations for the [white and Gold orientation controls](r27-white-evidence.json). Original source images are available on [Jaap’s puzzle page](https://www.jaapsch.net/puzzles/diamdil.htm).
- Run `python r27_source_sensitivity.py runtime/best.json` on a copy of an incumbent to reproduce the [source sensitivity checks](r27-source-evidence.json). Diagnostic changes live only in memory.

