# Round 24 results and starting settings

**The original Gold puzzle remains unsolved.** These results demonstrate specific reductions in repeated reasoning and finite search work; they do not establish a fastest overall solver. A valid Gold solution needs all 160 tiles, all 240 matched seams and one loop. See the [complete 42-proposal register](ROUND24.md) and [machine-readable evidence index](round24-candidates.json).

## Completed finite comparisons

The frozen `finite-final-02` runs used the implementations and source hashes recorded in their reports. The fitted-ordering values below supersede the earlier 133/119 development comparison.

| Mechanism | Measured result | What the result establishes |
| --- | --- | --- |
| SAT witness pool | 874 → 72 native calls across 60 contexts | Same supported-domain digest; less repeated querying. |
| SAT support-cover queries | 874 → 965 calls alone; 72 with the witness pool | Workload-dependent; batching alone regressed this corpus. |
| Joint conservation, affine/PB filtering, obstruction templates, interface formulas | Zero extra deletions after the prior fixpoint in those 60 contexts | Isolated discriminators show capability, but this corpus shows no added pruning. |
| Direct regional DAG | 60,620 → 889 candidate trials; all 5,040 histories preserved | Less enumeration; 191 continuation slices repeated zero candidate trials. |
| Symbolic inventory | 135,102 explicit inventory trials versus 19,033 ZDD operations | Identical supported inventories; different operation units, not a timing ratio. |
| Full factor elimination | 2,080 operations, live-resource width 2 on a synthetic 160-cell fixture | Exact seam/inventory support pass; 81 validated witnesses cover the retained values. This is not the original tile set. |
| Representative connectivity | 105 rows → 8 representatives | Existence preserved against all 105 outside matchings; not support/count preservation. |
| Subset join | 63,504 pair tests versus 16,384 integer-transform operations | Same 252 joined histories; operation types differ. |
| Lazy regional realization | 3,360 eager rows versus 2 requested local rows | The small resulting Gold arrangement was independently validated. |
| Regional pricing | 24 generated columns versus 90 eager histories | A separate shortage closes only after complete pricing and exact rational verification. |
| Frozen learned ordering | **136 → 120 DFS nodes** across 12 solved held-out cases | 7 wins, 2 ties, 3 losses on ten-cell fixtures; every Gold board independently checked. |

The six additional forty-cell ordering pairs all reached their 1.5-second query caps. They remain inconclusive. Existing time-bounded optional reasoning can cause small node-count variation between runs. The original 160-cell factor query also capped during its 50,000-operation construction/reasoning allowance: it removed zero values and returned unknown.

Details and independent checks are in [learning/proofs](ROUND24-learning.md), [regional reasoning](ROUND24_REGIONS.md), and [ordering](ROUND24_ORDERING.md). Frozen records: `audit/optimization-round24/finite-final-02/{learning,regions,ordering}.json` and `manifest.json`. Reproduction scripts are [learning](../benchmark_learning_round24.py), [regions](../benchmark_regions_round24.py), and [bundled ordering](../benchmark_learned_ordering_round24.py) (`--evaluate-bundled`).

## Whole-board comparisons

The recorded run used Windows, Python 3.12.14, four CPU workers and the RTX 2060. Its report confirms unchanged source hashes during that run. The six CPU profiles were `prior`, `reuse`, `sat-witness`, `algebra`, `regional`, and `branching`; their exact definitions are in [the benchmark script](../benchmark_round24.py).

- Six 160-cell planted cases: seed 817, 30 seconds per CPU profile.
- Twelve original-puzzle CPU cases: seeds 71 and 197, 60 seconds per profile, reversing profile order for the second seed.
- Two original-puzzle CUDA-domain cases: 512 lanes, seeds 71 and 197, 60-second requested budgets.
- **All 20 cases timed out.** None produced a Gold solution or completed a full search, including the planted cases whose solutions are known to the fixture generator.

| CPU profile | Planted nodes, seed 817 | Original nodes, seed 71 | Original nodes, seed 197 |
| --- | ---: | ---: | ---: |
| Prior | 46 | 60 | 63 |
| Reuse | 44 | 59 | 65 |
| SAT witness/relations | 44 | 61 | 67 |
| Algebra | 40 | 76 | 76 |
| Regional | 44 | 60 | 64 |
| Branching | 46 | 66 | 68 |

These are **unfinished-search counters, not a ranking of steps needed to solve**. A lower count at timeout may mean more work per node. CPU wall time was about 30.4–30.5 seconds for planted cases and 60.4–60.6 seconds for original cases. Five original CPU profiles removed almost the same raw Cartesian volume at the displayed scale; the branching profile removed slightly less. This does not establish a reliable whole-puzzle winner.

The raw-space measure counts Cartesian orientation assignments, including repeated-piece and otherwise illegal assignments. It is not a percentage of legal boards searched, a probability of success or an estimate of remaining solve time. CPU and GPU node counters have different meanings and must not be compared as if they were the same work unit.

### CUDA checkpoint follow-up

The initial 512-lane runs reported 5,341 and 5,311 GPU nodes and took **116.97 and 116.33 seconds wall time**, despite 60-second requested budgets. Duplicate checkpoint preparation was identified for correction. Those timings describe the pre-fix implementation and cannot serve as post-fix throughput or Stop-latency evidence.

After the fix (`7eda890`), comparable runs took **75.06 and 75.86 seconds**, with 5,311 and 5,256 decisions. The repeated-state regression proves that the same snapshot is built once, invalidated after a new dispatch or ownership change, and protected from callback mutation. Sixteen affected CPU/CUDA regression tests passed. The roughly 35% wall reduction is a checkpoint-work improvement, not evidence of fewer search decisions to Gold. A requested time limit still includes a final checkpoint overrun; these are whole-run times, not an independently measured Stop-button latency. Large frontiers remain expensive to save.

The whole-board source/input manifest and all 20 records are in `audit/optimization-round24/whole-final/comparisons/report.json`. Original input SHA-256: `81e5e40ab23afb135cafe913af3e9d6579d68eea4c096c60d2ddd57a1c8cfe79`.

## Sampling comparisons: two complete cooling cycles

All eight configurations started from fresh seeded boards and ran serially on CUDA
with 4,096 replicas, seeds 71/197,
ordinary and loop-aware scoring, and **8,192 proposals per replica**. This covers
two complete 4,096-step cooling cycles for the baseline schedule: 33,554,432
nominal proposals per case, **1,073,741,824 across all 32 cases**. The earlier
256-step warm-up comparison was too short to judge cooling behavior and is not
the primary evidence.

| Configuration | Ordinary edges, seeds 71 / 197 | Loop-aware edges, seeds 71 / 197 | Loop-aware measured seconds, seeds 71 / 197 |
| --- | ---: | ---: | ---: |
| Baseline | 64 / 66 | 69 / 66 | 12.02 / 11.94 |
| Ejection chains | 60 / 60 | 61 / 60 | 52.88 / 53.49 |
| Move tabu | 62 / 61 | 64 / 62 | 52.98 / 53.86 |
| Chains + tabu | 61 / 60 | 61 / 60 | 55.77 / 54.25 |
| Replica exchange | 108 / 112 | 108 / 110 | 58.12 / 55.50 |
| Chains + exchange | 99 / 98 | 100 / 100 | 57.22 / 55.70 |
| Tabu + exchange | 111 / 111 | 111 / 112 | 56.73 / 55.72 |
| All three | 99 / 98 | 100 / 98 | 55.53 / 56.91 |

Every reported candidate score was independently recomputed. None of these runs
found an edge-perfect or Gold solution. Replica exchange improved partial-board
edge scores against the baseline in the tested contexts. Standalone ejection
chains did not; tabu outcomes depend on the combination. These remain optional
heuristics. A higher partial score does not prove fewer steps to a Gold solution,
and a sample is not exhaustive branch coverage.

The reported seconds include GPU dispatch, transfers and independent candidate
checking, excluding separately recorded setup. Ejection chains change multiple
tiles and rescore the board, so equal proposal counts are not equal primitive
work. The archive-repeat counters in the JSON only observe reported best boards;
they do not count all repeated proposals. See the [full measured records](round24-results.json)
and [sampling methodology](ROUND24_GPU_HYBRID.md#reproducing-the-evidence).

## Starting settings: i7-4790 and RTX 2060 6 GB

**Preserve existing progress first.** If a search already has substantial saved work, keep its engine, output folder and compatible policy and use Resume. Changing to a different engine or policy without a valid frontier transfer can start a separate search. The settings below are a reasonable starting point for new work, not a measured fastest configuration.

| Setting | Starting choice | Reason |
| --- | --- | --- |
| Python | 64-bit Python 3.12 with full dependencies | Enables CUDA, native SAT/CP-SAT and repair as well as DFS. |
| Primary exact engine | Parallel CPU DFS, 4 workers | Matches the tested CPU configuration; keeps jobs disjoint. |
| CPU core | Automatic/compiled | Uses the available compiled propagation core. |
| Duration | Unlimited; Resume | Avoids unnecessary restarts; Stop still saves pending work. |
| Baseline pruning | Keep the existing defaults | Whole-board tests do not justify replacing them with every experimental option. |
| Work preservation | `shared_coverage`, `queue_pruning`, `resumable_reasoning`, `incremental_hall` | Completed coverage, queued exclusions and partial reasoning reuse address repeated work directly. These remain opt-in. |
| Alternative GPU experiment | CUDA domains, initially 128 lanes | Full reversible reasoning state differs from compact GPU DFS. |

The new GPU domain engine permits up to 512 lanes. Its 512-lane search state is 482,469,888 bytes (about 460 MiB), plus tables/runtime, so 6 GB VRAM does not require lowering it on memory grounds. However, 512 lanes have not been shown to minimize total solve work or time, and large checkpoints still have a visible saving cost. The old compact GPU engine's 131,072-lane setting is a different architecture; those replica counts are not interchangeable.

Leave the broad algebra/regional bundle, learned ordering, support-cover batches and new sampling heuristics **off initially**. They remain available for controlled experiments. Use `residual_reuse` when its completed-cache hits justify its cost. If trying SAT assistance, `witness_pool` has a clear finite reuse discriminator; its original-puzzle benefit is still unestablished. Persistent standalone SAT and SAT theory propagation cannot currently be combined.

For engine changes, stop the exporting run and preserve its checkpoint. Compatible portable frontiers can move unfinished work between parallel DFS, persistent plain SAT and CUDA domains when puzzle, objective and original domains match. Use `portable_frontier`/`--work-frontier` as documented. The legacy compact GPU checkpoint is not a general portable-frontier receiver. Do not run simultaneous copies of the same frontier when avoiding duplicate work; explicitly disjoint partitions are needed. Completing an imported subset alone never proves the full original puzzle impossible.

## Validation status and release gate

| Recorded environment | Reported outcome | Qualification |
| --- | --- | --- |
| Local full-dependency Python 3.12 | 1,275 tests, 74 skipped; successful exit | Historical suite before the final narrow checkpoint fixes; counts include skipped tests. |
| Local limited-dependency Python 3.12 | 1,276 tests, 204 skipped; successful exit | **This was Python 3.12, not a local Python 3.15 run.** |
| GitHub Python 3.15.2, Windows | 1,281 tests, 204 skipped; success | Verified interpreter in the job log. |
| GitHub Python 3.15.2, Ubuntu and both macOS architectures | 1,281 tests per job, 201 skipped; success | CPU/package checks do not establish Apple/AMD GPU hardware coverage. |

The final code preflight [GitHub workflow](https://github.com/gearashi/DiamondDilemmaGoldSolver/actions/runs/42327782447) completed **23 successful jobs** at commit `7eda890fed69d66ae58bb12689860b12869009dc`: all eight CPU jobs ran 1,281 tests, both software-Vulkan jobs ran 92 tests, and four platform source archives were built. Local NVIDIA CUDA/D3D12 hardware validation ran 113 tests with one optional skip; the subsequent 16 affected checkpoint/frontier tests passed without skips. The public [evidence record](round24-results.json) distinguishes these exact source versions.

**Final release gate:** CI must succeed on the exact commit referenced by the release tag, and the recorded benchmark artifacts must match their manifests. The earlier preflight run is identified separately from the final tag gate, whose exact run is linked in the release notes. Finite tests, package builds and unsuccessful bounded searches do not prove universal correctness or that no further optimization exists.
