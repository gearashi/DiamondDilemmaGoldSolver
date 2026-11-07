# Round 25 frozen finite results

All 37 proposals have retained, default-off implementations with bounded useful evidence. **The original unrestricted Gold puzzle remains unsolved.** Stronger deductions and avoided repeated work matter separately from elapsed time; these figures do not establish a universally fastest configuration.

The frozen source is `1b82e48930ed8bba12d31d01ee7c3c0a5132dfe4`. The serial finite coordinator completed all 42 commands. The [machine-readable results](round25-results.json) retain exact measurements, full SHA256 values, limitations and a reproduction command for every proposal. The [37-proposal register](ROUND25.md) and component reports describe implementation scope.

Validation at this source:

- Local Python 3.12 CPU/native suite: **1,839 tests, 159 skipped**, no failures/errors, plus the separate CPU scorer script.
- Local hardware suite: **291 tests, 1 skipped**, no failures/errors, plus six separate legacy GPU correctness scripts. CUDA and native WebGPU ran on an RTX 2060 with 6 GB VRAM; this is not physical AMD/Apple GPU coverage.
- [Cross-platform CI 42370266775](https://github.com/gearashi/DiamondDilemmaGoldSolver/actions/runs/42370266775): all **26 jobs passed**, including the Python 3.12/3.15 matrix. This preflight is not the final release-tag gate.
- Installed legacy DFS/hybrid compatibility: **four read-only checks passed**, all **six protected-file hashes unchanged**. Durable adoption retains old DFS progress instead of restarting sampling.

Standalone scripts are separate from unittest counts. Correctness-run timings are incidental, not performance recommendations. Proposal 5 uses successful exact-source CI on supported producer platforms; the Windows producer was not run. Its older detailed decision counts remain historical.

## Finite results and retained tradeoffs

Arrows compare the same stated counter, never unlike units. Finite success does not justify enabling every option for the original puzzle. Hashes below are abbreviated; the JSON gives full SHA256 values. Report paths are relative to `audit/optimization-round25/frozen-candidates-final01/`. They identify local reproducible artifacts, not files automatically shipped in source archives.

| ID / control | Frozen finite result or mechanism | Additional cost or boundary | Reproduce / report SHA256 prefix |
| --- | --- | --- | --- |
| 1 `sat_core_lifting` | SAT queries 12 -> 2, including verification. | One extra verifier model; decisions are zero in both cases. | `L(1)`; `01-learning/report.json` / `a5a583be40e7` |
| 2 `shared_sat_assistant` | 20 jobs: models 20 -> 1; support calls 400 -> 43. | Same-process worker reuse; native state is not persisted. | `L(2)`; `02-learning/report.json` / `c0330d00f99a` |
| 3 `resumable_proof_checking` | 32 leases: retained checker completes in 172 visits; restart uses 288 and remains incomplete. | Warm-process cursor only; baseline has not completed. | `L(3)`; `03-learning/report.json` / `3c2988ffbd63` |
| 4 `native_explanations` | Same 152 solutions/245 nodes; Python checker work 650 -> 281. | Native trace production is additional work. | `L(4)`; `04-learning/report.json` / `9822cdf0705e` |
| 5 `native_sat_persistence` | Verified clause-subset restart mechanism; three supported-platform CI jobs pass. | Windows producer disabled; older numerical comparisons remain historical. | `P`; `proposal05-platform-reference.json` / `a344f97e1026` |
| 6 `symbolic_coverage` | 176,800 flat checks versus 509 DAG operations; 12 insertions avoid 78 rereads. | Different units; Cartesian assignments are not legal-board counts. | `L(6)`; `06-learning/report.json` / `7debebabe724` |
| 7 `mixed_proof_dag` | Two refutations: checked proof nodes 41 -> 21. | Producer/other checker work is additional; bounded proof rules. | `L(7)`; `07-learning/report.json` / `d9521da63d7e` |
| 8 `higher_order_relations` | Compilation included: SAT calls 51 -> 28; decisions 178 -> 162. | Adds 2,040 table checks and more witness checks. | `L(8)`; `08-learning/report.json` / `fa8e8a0c80e9` |
| 9 `incremental_affine` | 13 contexts: equations 1,222 -> 376; row operations 3,282 -> 1,168. | Construction reuse; no extra pruning measured. | `L(9)`; `09-learning/report.json` / `1b96ea5a04fc` |
| 10 `conflict_pb` | Two extra candidate deletions; repeat filtering adds no cut combinations. | Costs 65,960 combinations and 1,084,925 scored terms. | `L(10)`; `10-learning/report.json` / `513aed57c470` |
| 11 `integer_lattice` | Three extra deletions preserve all 6 edge/4 Gold completions. | Adds 2,650 integer operations; puzzle gains also follow from rational span. | `L(11)`; `11-learning/report.json` / `99484e3d10dc` |
| 12 `hall_rematch` | After rematching, SCC vertices revisited 6 -> 2. | Matching-edge tests remain 2 in both policies. | `L(12)`; `12-learning/report.json` / `f3b3bf540b34` |
| 13 `reversible_gold_graph` | 16 contexts: seam tests 5,280 -> 561. | Still 234 support updates, 10,491 edge materializations and full traversals. | `L(13)`; `13-learning/report.json` / `01a76e1eadf0` |
| 14 `incremental_regions` | Compiler transitions 10,206 -> 889; counted edge work 15,141 -> 6,412. | Also 1,792 copied counts and 15 binding checks; unchanged outside scope required. | `R(14)`; `14-regional/report.json` / `8fac0fcc4ff9` |
| 15 `durable_regions` | Three restarts: attempts 1,303 -> 889; no completed transitions repeated. | Four writes/247,901 bytes; trusted local state, not an imported proof. | `R(15)`; `15-regional/report.json` / `676fea7be943` |
| 16 `compact_factors` | Controlled 160 cells: direct compile completes in 504 transitions; pair-table construction caps. | 2,160 table entries and 504 transitions are different units; edge/inventory only. | `R(16)`; `16-regional/report.json` / `538d20857a9e` |
| 17 `gold_elimination` | Controlled Gold retained; rewired two-loop 160-cell case rejected after 213 transitions. | Bounded whole-board elimination on controlled inputs. | `R(17)`; `17-regional/report.json` / `0dd8601ba91d` |
| 18 `separator_search` | Component builds 4 -> 2, transitions 84 -> 42, ticks 609 -> 487. | Same bounded separator planner; no general engine dominance. | `R(18)`; `18-regional/report.json` / `46f97a43edb8` |
| 19 `selective_refinement` | Prefix reuse: transitions 52 -> 34; puzzle fixture removes 12 choices. | 24 prune applications repeat those 12 removals; positive relaxation is not Gold. | `R(19)`; `19-regional/report.json` / `ddeb9a4b23d5` |
| 20 `joint_symbolic_regions` | Nine-cell symbolic join: counted ticks 38,328 -> 33,166. | Seven-cell regression 4,388 -> 5,467; two whole-board regions only. | `R(20)`; `20-regional/report.json` / `a67bc8c63fe2` |
| 21 `algebraic_connectivity` | App basis/join work 1,176 -> 142 ticks. | Keeps 11 rather than 5 rows; existence only; shared compilation excluded. | `R(21)`; `21-regional/report.json` / `27fa716bfeb8` |
| 22 `residual_isomorphism` | Three 160-cell ring residuals: compilations 3 -> 1; ticks 5,956 -> 2,955. | Synthetic ring topology; seven-cell regression 448 -> 507 ticks. | `R(22)`; `22-regional/report.json` / `848b2d060f74` |
| 23 `gold_pricing` | Gold-aware master rejects a case left possible by inventory-only pricing. | 208 pricing edges plus 88 compile transitions; bounded fractional relaxation. | `R(23)`; `23-regional/report.json` / `1b75cef6bfa4` |
| 24 `multichain_routing` | Joint linkage rejects a contradiction missed by individual-chain checks. | 12 route steps plus 83 graph ticks; positive linkage is only necessary. | `R(24)`; `24-regional/report.json` / `55d1ffec0d9c` |
| 25 `region_certificates` | Import avoids 889 compiler and 889 export transitions. | Still verifies all 889 transitions; first production pays all three stages. | `R(25)`; `25-regional/report.json` / `1c09289f5791` |
| 26 `gpu_joint_support` | Two finite cases reduce decisions 13 -> 9 and 3 -> 0. | Other cases are neutral with added Hall/support work. | `D(26)`; `26-cuda-domain/report.json` / `9aad78b5f5c4` |
| 27 `gpu_propagated_forks` | Prepared-parent reuse reduces strand joins 44 -> 17. | Same case adds checks 46 -> 60; another adds 621 -> 643. | `D(27)`; `27-cuda-domain/report.json` / `2280ea1fbb00` |
| 28 `gpu_event_propagation` | Four cases reduce candidate visits; controlled 160 cells: 326 -> 171. | Decisions unchanged; queue/index operations are additional. | `D(28)`; `28-cuda-domain/report.json` / `dab7af1574ca` |
| 29 `gpu_precise_reasons` | Six-cell prism: decisions 21 -> 17; learned clauses 20 -> 16. | Small-cache saturation loses gain; hash/support costs remain. | `D(29)`; `29-cuda-domain/report.json` / `ab1aaca78aa1` |
| 30 `gpu_shared_clauses` | Recipient decisions 4 -> 2 and checks 87 -> 64; same four completions. | Producer unfinished; costs 5 decisions/54 checks plus 2 verifier queries. | `D(30)`; `30-cuda-domain/report.json` / `afb281eae778` |
| 31 `gpu_lane_balance` | Four budgets use fewer dispatches; budget 64: 10 -> 7. | Budget 17 increases decisions 21 -> 23 and checks 898 -> 947. | `D(31)`; `31-cuda-domain/report.json` / `0d1f779aa2ff` |
| 32 `gpu_incremental_checkpoint` | One snapshot supplies encoding/frontier/stats; controlled 160 cells omit 3,793,836 padding bytes. | Small compressed payloads grow; retained generations and filesystem overhead cost space. | `D(32)`; `32-cuda-domain/report.json` / `97166d8414eb` |
| 33 `durable_hybrid` | Completed ordinary repair prefix: 54 versus 72 native calls; controller work also survives. | Interrupted native queries retry; local completion does not prove global infeasibility. | `Q`; `33-repair/report.json` / `1d8511d3c687` |
| 34 `obstruction_chains` | Each synthetic fixture: guided solves 12/12 versus 9/12 by the cap. | Adds 6,054/2,688 orientation checks; successful-pair totals exclude capped runs. | `G(chains)`; `34-cuda/report.json` / `aa5517840fd3`; `34-webgpu/report.json` / `9fd7062fd4e2` |
| 35 `cycle_crossover` | Supplied parents: two parent plus two child scores recover Gold with zero GPU proposals. | Parent discovery unmeasured; planted examples and bounded history only. | `G(crossover)`; `35-cuda/report.json` / `0ae6dcdeccbf`; `35-webgpu/report.json` / `ed7b8d9db864` |
| 36 `lazy_gold_scoring` | Original input, 2,176 proposals: base passes 4,352 -> 2,176; loop passes 4,352 -> 2,022. | Same trajectory, not fewer proposals; passes are not total instructions or solve-time evidence. | `G(gold_scoring)`; `36-cuda/report.json` / `25f93420960f`; `36-webgpu/report.json` / `8f92a67dd387` |
| 37 `adaptive_temperatures` | 24 pairs/backend: 20 versus 21 solve; both-solved proposals 966,656 -> 667,648. | 7 wins/4 losses/9 ties; adverse fixture 126,976 -> 139,264; host work extra. | `G(temperatures)`; `37-cuda/report.json` / `5c22f93561bc`; `37-webgpu/report.json` / `46e75bf9815d` |

## Reproduction

Use an unchanged clean checkout, required dependencies and fresh output paths. `--evidence-stage frozen` rejects changed source/runtime/repository identity; optimized Python is rejected. The JSON expands these entries into exact per-proposal commands:

- `L(n)`: `python -B benchmark_learning_round25.py --candidate n --evidence-stage frozen --output audit/new-learning-n.json`.
- `R(n)`: `python -B benchmark_regions_round25.py --candidate n --evidence-stage frozen --output audit/new-region-n.json`.
- `D(n)`: `python -B benchmark_gpu_round25.py --proposal n --evidence-stage frozen --output audit/new-domain-n.json` (CUDA hardware).
- `P`: `benchmark_sat_capabilities_round25.py` and `benchmark_sat_persistence_round25.py`, each with `--evidence-stage frozen --output PATH`, only on supported CPython 3.12/python-sat 1.9.dev15 Linux/macOS producer environments. Do not run the disabled native proof logger on Windows.
- `Q`: `python -B benchmark_repair_round25.py --evidence-stage frozen --output audit/new-repair.json` (CPU/OR-Tools).
- `G(name)`: `python -B benchmark_gpu_name_round25.py --backend cuda --evidence-stage frozen --output audit/new-name-cuda.json`; repeat with `webgpu` and another output. Substitute `chains`, `crossover`, `gold_scoring`, or `temperatures` for `name`.

Prepared original puzzle data is required by original-input scoring and continuation checks; source packages prepare it locally. Native explanation checks require matching locally compiled CPU libraries. Hardware commands are not generic CPU installation tests.

Separate original-input fresh-process sampler comparisons preserve final state for baseline and combined policies on both CUDA and WebGPU: 18 dispatches, 147,456 proposals per configuration, and zero completed dispatches replayed after a 9+9 split. This establishes same-runtime continuation, not Gold completion or faster search. `benchmark_hybrid_round25.py --resume-comparison` reproduces this family; exact commands and artifact identities are in the bound coordinator manifest.

## Evidence revisions and the hybrid handoff fix

The 42 finite commands and the CPU/SAT, CUDA-domain and latency measurements above retain their actual frozen source `1b82e48930ed8bba12d31d01ee7c3c0a5132dfe4`. A later two-file patch at `4af1d003bf14284050bf980e14e33a57ff0ee64e` fixes the durable hybrid controller handing coordinator-only metadata into the exact DFS cache. It preserves actual cache/store objects and mirrors DFS cleanup; the DFS input whitelist remains strict. This integration fix is not an additional search-efficiency gain.

An independently reviewed source bridge confirms all 333 other tracked files unchanged, with raw/LF bindings for the executable/fixture inputs. Fresh affected evidence at the patched revision includes 34 focused hardware tests (0 skipped), six repair comparisons, and 4 full original-input hybrid cases with the repaired handoff observed. Capped solver results remain incomplete; no old report is relabelled as a patched-source rerun. The JSON records every report revision and bridge hash. Final release validation/CI applies to the release commit separately.

## CPU and SAT performance on the test machine

These results use the frozen solver at `1b82e48930ed8bba12d31d01ee7c3c0a5132dfe4`, on an Intel Core i7-4790 (4 physical cores/8 logical processors), about 16 GiB RAM, and an RTX 2060 with 6 GiB VRAM. All recorded source, runtime and input guards passed. The numerical companion is `round25-results.json`.

The CPU baseline already includes the existing DFS pruning and compiled propagation. “Baseline” does not mean unpruned search. The first comparison used six matched finite cases: two 10-cell puzzles and two seeds each of 160-cell planted puzzles with 8 or 24 free cells. Every returned solution was independently checked for unique pieces, matching seams and one complete Gold loop.

| CPU profile | Solved within 30 s | Nodes on those completed cases | Baseline nodes on the same cases |
|---|---:|---:|---:|
| Baseline |6/6|128|128|
| Local/shared reuse bundle |6/6|130|128|
| SAT-assisted DFS bundle |6/6|128|128|
| Arithmetic bundle |3/6|71|28|
| Earlier regional bundle |3/6|63|29|
| New regional bundle |6/6|140|128|
| All combined |3/6|21|27|

The combined row excludes three unfinished cases; its smaller subtotal is not an overall win. New regional reasoning did reduce the two residual-8 cases from 17 to 6 nodes, but increased residual-24 from 90 to 116. In the 60-free-cell comparison, no bundle improved the completed baseline result of 72 nodes. All profiles timed out on the other seed. The broader tests therefore support keeping optional research bundles selectable, rather than enabling all of them by default.

Follow-up configuration checks isolated local reuse and the compiled search loop. Local reuse totaled 128 nodes versus 129 baseline; an earlier baseline repeat itself used 128, so that one-node difference is not robust evidence. The native-loop/explanation variants totaled 142–143 nodes versus 128 baseline. These ablation names are report labels installed by audit-only adapters, not additional profile names accepted by the shipped CLI. Their exact option dictionaries are included in the numerical companion; the adapters are not bundled. Native reason checking has a documented dedicated finite benefit, but it did not establish a general winner in these larger combinations.

One full-reuse residual run fragmented its frontier into 2,877 jobs and produced a 290 MB checkpoint. Its 60-second search budget required 81.187 seconds through final process completion. Disabling only `queue_pruning` does not eliminate this mechanism: shared coverage still subtracts completed boxes from queued jobs. The local-only ablation disables shared coverage and avoids that coordinator path. This is a reason to avoid the full bundle as a default, while preserving its separately measured scoped capabilities.

### Four or eight CPU workers

Two 120-second original-puzzle runs per worker count all remained unfinished. Four workers performed 193 controller evaluations in total; eight performed 280. Eight workers closed slightly more committed Cartesian orientation volume in both seeds—approximately 37.28306% versus 37.28041%, a difference of only 0.002656 percentage points. These are not counts of legal boards or a prediction of solution time. Four workers used about 870 accumulated propagation worker-seconds versus 1,591 at eight; the separate work counters must not be confused with elapsed wall time.

The successful 261-sample memory capture covered all four runs. Minimum observed available memory was 4,785 MiB with four workers and 4,623 MiB with eight, with low sampled paging. These are system-wide observations, not process peak RSS or long-run cache-growth bounds. A separate earlier capture failed and is excluded.

For this machine, use **four workers when resource efficiency and responsiveness matter**, or **eight for a dedicated CPU search with greater measured bounded throughput**. Neither setting is proven to solve the puzzle in fewer steps. Resume an existing DFS checkpoint first; keep the Gold target as 240 matched edges in one loop. Unlimited duration is appropriate for an intentional long run, while the short tests do not establish a long-run memory bound. The corresponding dashboard controls are **Search algorithm → Parallel CPU DFS**, **CPU workers → 4 or 8**, **CPU propagation core → Automatic** (compiled when available), and **Run duration → Unlimited**. Preserve the current checkpoint rather than restarting the puzzle to adopt a profile.

### Persistent SAT

Both SAT configurations solved all six finite cases. The baseline used 157,207 native SAT decisions in total; the lifted/symbolic bundle used 62,878, but won only two of six pairs and lost four. Residual-24, seed 817 improved from 152,432 to 9,041 decisions; seed 291 regressed from 4,584 to 53,234. SAT decisions and DFS controller nodes are different units and must not be directly ranked.

This was a **bundle comparison, not an isolated core-lifting result**. Both received the same 32-box initial partition, but symbolic coverage recompressed the lifted configuration to the union—one original-root box—even with no exclusions. Core candidates, independent core verifications and core-removed boxes were zero throughout this matrix. The difference reflects changed decomposition and solver configuration; it cannot be credited to verified core lifting alone. The extra symbolic construction/apply work is retained in the companion report.

The baseline’s 32.844-second process time on residual-24, seed 817 also includes publication: the reported SAT advance found its valid solution at 28.266 seconds, before constructing and writing a 40.2 MB checkpoint containing 6,295 Gold cuts. The elapsed field is recorded before snapshot construction. Original-puzzle tests used two warm slices and one model per case; there the elapsed field describes only the last slice, so whole-run comparisons use worker/process wall time.

All four 45-second original-puzzle SAT runs remained unfinished and closed no full SAT cube. Internal SAT learning is not represented by that cube count. Persistent SAT remains an alternative with mixed finite behavior, not a demonstrated original-puzzle winner. On Windows this benchmark kept native clause logging disabled; warm native state survives slices, while a process restart can repeat internal work inside the unfinished cube.

## GPU exact search and hybrid measurements

These comparisons use the RTX 2060 with 6 GB VRAM. Feature and finite evidence remains at source `1b82e489`; the full hybrid rerun uses `4af1d003`. The latter changes only the hybrid cache handoff and its tests. A reviewed source bridge verified 333 other tracked files unchanged; it is not a claim that an untested code revision inherited every result. Report identities, hashes, scopes and adverse observations are preserved in [the machine-readable results](round25-results.json).

### Exact GPU search

The finite matrix used 128 lanes, a 30-second cap and six generated planted fixtures per profile, including 160-cell boards with 8 or 24 unrestricted cells. These fixtures use synthetic tiles, not the original piece set.

| Profile | Gold solutions /6 | Decisions on the five cases solved by every profile | Candidate checks on those five |
|---|---:|---:|---:|
| Baseline |5|6,624|3,108,163|
| Joint support |5|6,512|3,096,666|
| Reuse |6|7,479|1,491,148|
| Shared learning |5|6,612|3,104,433|
| Combined |6|7,113|1,446,352|

All 27 returned solutions were independently validated. The three capped cases remain unknown. On all six cases solved by both reuse and combined, combined used 10,293 decisions versus 11,158 and 2,312,890 candidate checks versus 2,420,571. It also took 17.343 solver-seconds versus 16.344 in aggregate and was slower in every pair. Thus it reduced this measured search work while adding overhead. The shared-learning profile produced no verified imports in its capped case; no benefit is claimed for sharing from that case.

On the original puzzle, all six completed 128-lane runs timed out without a solution or an exhausted lane. Baseline and reuse removed approximately 18.4010% of the initial raw Cartesian domain volume; combined removed 19.7983%. This volume includes duplicate-piece and invalid assignments: it is **not the percentage of legal boards exhausted or the percentage of the puzzle solved**. Combined made 263 decisions in each repeat, versus 1,063/1,043 baseline, but also performed approximately 2.09 billion added joint-support word scans. Fixed-budget decision counts are not decisions required to solve.

The original 128-lane solver durations were 35.42 to 35.83s baseline, 36.44 to 37.31s reuse, and 33.05 to 33.11s combined under a 30 s soft budget. Complete worker durations were 38.80 to 54.61s, including checkpoint/validation overhead. Retained checkpoint stores were 20.15 to 20.35MB for reuse and 14.13 to 14.14MB for combined; their 445-byte reference files are only pointers, not whole checkpoints. The baseline final checkpoints were 26.40 to 26.50MB. Saved references were reopened and verified without resuming search or decoding native arrays.

A separate 512-lane baseline exceeded the 120 s whole-worker watchdog. It had written a 93.94 MB checkpoint about 72 s after starting; the records do not isolate whether later solver work, publication, validation or serialization consumed the remaining time. This is an incomplete observation, not evidence of a CUDA driver or kernel timeout. 512 lanes are not recommended on the strength of this run.

### Hybrid pipeline

The six paired planted fixtures all solved during GPU sampling. Baseline charged 2,359,296 lane proposals in total and combined 393,216, but combined also performed 30,962,931 guided orientation evaluations and was slower in five pairs (one tie). All combined cases were observed at the first eight-dispatch collection; neither crossover, adaptive-temperature changes, repair nor DFS was exercised there. Initial-population Gold was not measured separately by this harness, so charged proposals bound first discovery from above. Multiple settings and the sampling objective changed together.

The first full original-puzzle run exposed a real handoff bug: the durable controller passed its own metadata into DFS's strict cache validator. The two-file fix filters exactly those controller keys, preserves cache identities and propagates newly created cache owners and cache removals even on exceptions; unknown/malformed DFS state remains rejected. The corrected revision passed 34 focused hardware/controller checks with no skips, reran all six bounded repair comparisons, and completed the four original pipeline runs below.

These original hybrid runs start from fresh seeded populations: the benchmark supplies no saved best board or candidate hints. The application can reuse an existing board. These scores do not supersede an existing saved result and are not a reason to restart retained exact work.

All four use 4,096 replicas, a 200 s whole-run cap, soft allocations of 30 s sampling and 60 s repair. Every run reached exact DFS and then returned an ordinary timeout. No original Gold solution or infeasibility proof was obtained.

| Profile / seed | GPU proposals | Sample best edges /240 | Repair best /240 | Ordinary repair native calls | Exact DFS nodes | Solver seconds |
|---|---:|---:|---:|---:|---:|---:|
| baseline / 71 | 2,438,987,776 | 70 | 134 | 65 | 101 | 200.125 |
| round25-combined / 71 | 29,884,416 | 155 | 163 | 16 | 81 | 200.610 |
| baseline / 197 | 2,451,570,688 | 69 | 135 | 65 | 129 | 200.140 |
| round25-combined / 197 | 29,884,416 | 155 | 160 | 15 | 58 | 200.641 |

Combined guided orientation evaluations were 3,832,398,855, 3,832,177,830 for seeds 71 and 197 respectively. These are additional to its lane proposals and exclude other scan work. Recorded temperature adjustments were 7, 7. Crossover independently scored 240, 240 children and injected 0, 0 boards, in the same seed order.

The repair-call column counts ordinary LNS only. Each combined run additionally made three experimental oracle calls, all returning UNKNOWN, and two master attempts. Those operations are separate from the table counts; no global exclusion follows from UNKNOWN.

Higher edge scores are useful heuristic outcomes, not proof of being closer to one complete Gold loop. Likewise, fewer DFS nodes within equal time do not prove a smaller complete search. GPU proposals, orientation evaluations, local repair queries and DFS nodes are different units. These two-seed, multi-setting comparisons do not identify a universally best solver or isolate the effect of each feature. CUDA_PATH discovery warnings did not prevent actual CUDA execution and are distinct from the fixed handoff failure.

### Replica count, Stop latency and memory

A separate bounded calibration measured two early 64-move dispatches per population with original pieces and the combined Gold sampler. Times below include synchronized native launch overhead. Stop was requested during the second dispatch; the last column also includes durable snapshot encoding, JSON serialization and a flushed write.

| Replicas | Normal dispatch(s) | Stop request to return(s) | Stop request to written checkpoint(s) |
|---|---:|---:|---:|
| 128 | 0.219 | 0.126 | 0.259 |
| 4,096 | 0.263 | 0.154 | 0.468 |
| 8,192 | 0.304 | 0.173 | 0.665 |
| 16,384 | 0.551 | 0.376 | 1.242 |

These are two-dispatch observations, not worst-case Stop or steady-state guarantees. Exchange/adaptation and regular collection were not yet due, so their unobserved costs are unknown, not zero. The driver reported kernel timeouts enabled; no driver setting was changed. 32,768 and 131,072 replicas were not latency-tested. The 4,096-replica full-pipeline runs provide a separate later-phase check.

For the 29-row combined sampler, current/best boards and metadata occupy 1,396 bytes per replica, plus approximately 0.369 MiB shared tables for the 160-cell input: about 5.82 MiB of primary payload at 4,096 replicas and 174.87 MiB at 131,072. These are allocation estimates, not measured total VRAM. They exclude CUDA/WebGPU context, compiler, temporary buffers, memory pools and local spills; checkpoint encoding can use several additional host copies. Memory capacity alone does not justify selecting the largest population. The exact GPU-domain engine has a different, much larger lane state and must not be sized using sampler figures.

The finite per-proposal evidence also includes exact CUDA/WebGPU sampler continuation across a fresh process: 18 dispatches versus 9+9 at 128 replicas matched native/controller signatures with zero completed dispatches replayed. Completed local repair work was retained across restarts, while an interrupted OR-Tools search retries its exact pending query; native OR-Tools internal state is not restored. All new controls remain optional and default off.

## Starting settings for this hardware

The measured machine is an Intel Core i7-4790 (4 cores/8 threads), about 16 GiB RAM, and an NVIDIA RTX 2060 with 6 GiB VRAM. Use the existing **64-bit Python 3.12 environment** for the full CUDA/SAT/repair feature set. These are evidence-based starting settings, not a proven fastest route to Gold.

| Purpose | Dashboard settings |
| --- | --- |
| Continue established exact progress | Resume the existing compatible DFS search; **Run duration: Unlimited**, **CPU propagation core: Automatic**. Keep the original puzzle and saved-search directory. |
| Use the CPU more fully | **Parallel CPU DFS**, **4 CPU workers**, **Unlimited**, **Automatic** propagation. On its first use it imports existing unfinished DFS progress into its own queue. Use **8 workers** for a dedicated run if throughput matters more than resource efficiency. |
| Try GPU guidance | **Experimental GPU sampling + repair + DFS**, **NVIDIA CUDA**, **4,096 GPU replicas**, **Unlimited**, with **Resume GPU sampling and completed repair work** enabled to preserve the sampling/repair controllers and completed work. Existing saved DFS progress resumes without sampling again. |
| Try the newer exact CUDA engine | **CUDA domains + learned backjumping**, **128 lanes**. Treat it as an experimental alternative; it does not inherit another engine's completed coverage. |

Leave the other new experimental controls off initially. All 37 are available for targeted experiments, but the broader comparisons did not support enabling every CPU bundle. More replicas are not an exhaustion guarantee: hybrid sampling may revisit boards, and it eventually hands work to CPU repair and DFS. The newer CUDA-domain lanes and hybrid replicas have different state sizes and work units.

Four CPU workers used less accumulated worker time; eight produced greater bounded throughput. Neither has established fewer steps to a Gold solution. The 512-lane CUDA-domain benchmark exceeded its 120-second watchdog after publishing a roughly 94 MB checkpoint; the precise delayed stage was not identified. That result does not justify recommending 512 lanes on this machine.

For the combined hybrid sampler, two early dispatches at 4,096 replicas measured about 0.26/0.22 seconds, with the injected Stop request reaching a saved checkpoint in about 0.47 seconds. At 16,384 the corresponding checkpoint interval was about 1.24 seconds. These are small calibrations, not worst-case Stop guarantees or evidence that either population finds better boards. Counts above 16,384 were not calibrated in this round. Do not select 131,072 merely because it fits VRAM.

Resume compatible saved progress rather than restarting to adopt a new preset. When intentionally changing engines, preserve each engine's separate state. Wait for **Stopped** before changing settings; Unlimited still stops on your request, a verified one-loop Gold solution, or proven exhaustion of that engine's encoded work. No production search was started by these benchmarks.

## Release gate

These measurements bind the frozen source shown above. Publication requires all required CI jobs on the exact final main commit and `v1.6.0` tag; the [release notes](https://github.com/gearashi/DiamondDilemmaGoldSolver/releases/tag/v1.6.0) retain that exact gate receipt. Feature-source preflight is not a substitute. Unknown, interrupted and capped computations never certify exhaustion; Gold witnesses require independent validation.
