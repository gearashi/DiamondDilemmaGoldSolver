# Round 23: candidate register and search controls

This release evaluates the 52 requested proposals and reopens two earlier proof-reuse candidates. An improvement qualifies through additional sound deductions, fewer search decisions on a completed comparison, or avoided repeated reasoning. Lower nodes per second alone is not grounds for rejection. A bounded timeout, a deeper partial placement, or two incomplete searches with different trees cannot establish total solve efficiency.

All new controls are optional. Existing compatible checkpoints and the previous default policy remain available. No original Gold solution was found in these tests. The actual puzzle has **160 tiles and 240 seams**; the solution must also contain one loop.

## Candidate disposition

“Retained” means the finite checks demonstrate the stated deduction or reuse. “Experimental” means sound behavior is checked but practical benefit is workload-dependent or unestablished. It does not mean every option should be enabled together. The tests named below are included in this repository; the local audit also retains source snapshots, contracts and raw results.

| # | Proposal | Implementation and disposition |
| --- | --- | --- |
| 1 | SAT mask cores | Retained: domain assumptions yield scoped impossible conjunctions; imported claims are proved again. |
| 2 | Resumable SAT candidate scans | Retained: a capped scan continues at its next candidate, with scope changes invalidating obsolete cursors. |
| 3 | SAT model support witnesses | Retained: a valid model supplies supports for other candidates, avoiding repeated queries. |
| 4 | Implications from completed probes | Retained: narrower domains can reuse wider completed results, followed by propagation to a new fixpoint. |
| 5 | Conditional Hall certificates | Retained: a small physical-piece shortage rejects subsequent boundary combinations satisfying its scope. |
| 6 | Smaller sufficient explanations | Retained: bounded independent replay removes an assumption only after proving the weaker context impossible. |
| 7 | Clause vivification | Retained: completed replay can shorten learned mask clauses. |
| 8 | Clause resolution | Retained: complementary pivot masks eliminate a pivot without assuming it is fixed. |
| 9 | Persistent root proofs | Retained: saved mask claims are bound to the exact root and independently reverified. |
| 10 | Verified cross-worker mask sharing | Retained: transport is bounded, incremental and reverified; unknown claims never prune. |
| 11 | Rotated Gold clauses | Retained: transport proper-loop witnesses through verified mesh rotations, then validate each transformed witness. |
| 12 | Grouped constructive look-ahead | Retained: probe complete partitions by connection pattern and keep common deductions only after all alternatives finish. |
| 13 | Three-cell path consistency | Retained: bounded binary-relation checks remove pair supports lacking a third-cell support; independent enumeration checks solution preservation. |
| 14 | Three-side signature branching | Experimental: disjoint pattern groups change the exact search partition without omitting placements. |
| 15 | Learned impact priorities | Experimental: prioritize cells using observed domain reductions; priorities never become exclusions. |
| 16 | Solution/conflict ordering | Experimental: selectable ordering goal; fixture-specific outcomes are reported separately. |
| 17 | Frontier restarts | Experimental: reorder intact unfinished jobs with fairness instead of discarding their search cursors. |
| 18 | Worker portfolios | Experimental: disjoint workers vary ordering, while retaining the same proof rules and ownership checks. |
| 19 | Deduction-yield budgets | Retained: completed deductions and reusable proofs influence expensive regional-check budgets and cooldowns. |
| 20 | Limited discrepancy | Experimental: prioritize intact jobs by excluded preferred choices, preserving eventual service of other jobs. |
| 21 | Regional network fixpoint | Retained: live summary compatibility removes an extra orientation beyond repeated pair filtering on an eight-cell fixture. |
| 22 | Triple-region joins | Retained: a three-region join removes four additional orientations on the fixture. |
| 23 | Overlapping regions | Retained: exact shared assignments remove three choices while preserving all three fixture Gold solutions. |
| 24 | Global regional inventory | Retained: joint physical-piece allocation detects a shortage missed by pairwise checks. |
| 25 | Outside-chain conditioning | Retained: fixed outside strands distinguish otherwise identical regional boundary masks. |
| 26 | Multi-port separators | Retained: a necessary component/cut bound rejects a new obstruction; independently constructed cycles remain accepted. |
| 27 | Wider seam cuts | Retained experimentally: exact memoized boundary checks admit up to six live seams; a capped check reports unknown. |
| 28 | Frontier-width regional admission | Retained: narrow frontiers allow completed queries larger than 16 cells, including controlled 40- and 160-cell examples. |
| 29 | Decomposition-plan cache | Retained: reuse verified query plans; a whole-puzzle search gain is unestablished. |
| 30 | Conflict-guided region selection | Experimental: bounded priorities select regions; additional benefit remains unresolved. |
| 31 | Regional decision diagrams | Retained: exact shared-assignment representation compresses a 720-history fixture to fewer than 100 internal nodes. |
| 32 | Component AND/OR joins | Retained: the fixture removes one extra orientation beyond pair/triple/inventory filtering. |
| 33 | GPU board symmetry | Retained experimentally: use only verified proper rotations; the selected policy has separate saved coverage. |
| 34 | Remote GPU forward checks | Retained experimentally: reject placements leaving a remote cell unsupported. |
| 35 | GPU small-set Hall checks | Retained experimentally: candidate rotations share physical-piece inventory, exposing small shortages. |
| 36 | GPU forced propagation | Retained experimentally: a unique supported orientation is committed through the exact resumable stack. |
| 37 | GPU future connectivity | Retained experimentally: a conservative possible-connection graph can prove a Gold branch disconnected. |
| 38 | GPU dynamic MRV | Retained: dynamically select constrained cells while storing the selected cell and cursor in each frame. See the clean hardware comparison. |
| 39 | GPU shared Gold cuts | Retained experimentally: keep up to 32 independently verified proper-loop exclusions from incoming prefixes or submissions. |
| 40 | CPU probes of GPU jobs | Retained experimentally: CPU seam/Hall propagation can reject the exact current subtree only after a completed proof; unknown retains the GPU job. |
| 41 | CPU endgame ownership | Retained experimentally: freeze the handed-off GPU lane, persist CPU state, and return a verified solution through normal candidate handling. |
| 42 | Connectivity-aware GPU sampling | Retained experimentally: a heuristic for seed quality; sampling does not prove coverage or permanently fix guesses. |
| 43 | Piece-dependency repair | Retained experimentally: expand repair patches along displaced-piece dependencies; a controlled full-board two-piece swap is repaired. |
| 44 | SAT partial Gold propagation | Retained experimental engine: native SAT callbacks reject proper loops before a complete board. |
| 45 | Direct SAT connectivity | Tested implementation remains research-only; excluded from the dashboard because the tested encoding did not justify adoption. |
| 46 | Alternative SAT encodings | Tile/orientation channeling is an optional engine. The tested cardinality-network variant remains research-only. |
| 47 | CP-SAT regional alternatives | Retained experimental engine: use completed regional alternatives; capped table generation falls back without dropping solutions. |
| 48 | Global seam conservation | Experimental: exact quantities are sound, but no additional deductions over the existing full propagator were observed in sampled planted contexts; practical benefit remains unestablished. |
| 49 | Piece/orientation equivalence | Experimental generic reduction using full strand pairings. The original input has no equivalent orientations or physical pieces; no benefit is claimed for it. |
| 50 | Persistent regional tables | Retained: bounded, source-bound completed local queries survive process restart. Incomplete tables are never saved as proofs. |
| 51 | Endpoint supply bounds | Retained experimentally: inventory-aware capacity prevents counting one piece twice; some deductions overlap older seam/Hall checks. |
| 52 | Portable certificates | Retained: independently check Hall shortages and proper-loop witnesses against explicit puzzle/root scopes. |
| 53 | Dirty-neighborhood proof reuse | Reopened and retained experimentally: unchanged local inputs reuse completed certificates, including after global cache eviction. |
| 54 | Parity-root reuse | Reopened and retained experimentally: share repeated disjoint-set lookups inside one unchanged parity pass. |

Finite regressions are in `test_constraint_learning_round23.py`, `test_constraint_transport_round23.py`, `test_constraint_lease_round23.py`, `test_constraint_model_round23.py`, `test_constraint_deep.py`, `test_constraint_integrated_round23.py`, `test_constraint_region_network.py`, `test_constraint_reopened_round23.py`, `test_gpu_reasoning.py`, and `test_gpu_hybrid_reasoning.py`. Existing checkpoint, cancellation, native/Python equivalence, loop-validator and ownership tests run alongside them. The shader tests need an explicitly enabled compute backend.

## Controls and prerequisites

The dashboard exposes the new controls under **Exact-search optimizations** and **Experimental GPU reasoning**. CPU controls also apply to the exact phase of hybrid search. `smart_learning` groups mask-core reuse, verified sharing and bounded clause strengthening; SAT-dependent features require the full Python 3.12 profile. `extended_regions` enables the regional network, overlap, larger-region and persistent-table work. Compiled search and First-UIP controls remain separate.

GPU DFS uses a JSON `gpu_features` object with these keys: `remote_check`, `hall`, `forced`, `dynamic_mrv`, `connectivity`, `symmetry`, `shared_gold`, `cpu_probe`, `cpu_endgame`. All default to false. Every non-default combination gets its own `runtime/systematic-<policy-hash>/` directory. Re-selecting the same combination resumes that saved frontier. The all-false policy keeps `runtime/systematic/` unchanged. Switching policy does **not** transfer completed coverage between those folders.

Example using the portable launcher (replace `python` with the installation's virtual-environment interpreter):

```bash
python platform_launcher.py start --unlimited --algorithm gpu-dfs --backend cuda --replicas 131072 --gpu-features '{"dynamic_mrv":true}'
```

Windows PowerShell users can set the same controls in the dashboard, avoiding shell-specific JSON quoting. New native model identifiers are `sat-propagator`, `sat-channel`, and `cp-sat-regions`. Native model checkpoints retain validated cuts and hints; they do not preserve a native solver's internal cursor.

## Correctness and durability boundaries

Optional explanation work runs after committing the consumed search node. Stopping during that work preserves the pending frontier; stopping after the last rejection records exhaustion instead of an empty invalid checkpoint. Worker claims are independently proved under their bound root. A local symmetry assumption cannot escape into a wider sibling or reused cache.

Regional files are trusted local performance caches, bound to exact inputs and source. Their checksums detect accidental corruption; they do not make a maliciously rewritten cache a portable mathematical certificate. Only the separately verified Hall and proper-loop formats provide that certificate behavior. SAT timeouts, partial region tables and interrupted unions are always unknown.

CUDA and WebGPU receive the same optional rules and checkpoint metadata. Hardware execution on this machine covers NVIDIA CUDA and Windows WebGPU, not physical AMD or Apple hardware. Cross-platform package/CPU tests and software Vulkan shader tests provide separate coverage.

## Retesting and scope

All performance comparisons from the overlapping `n5_runner` session were invalidated. The release comparison reruns GPU work and CPU work sequentially after the user confirmed that program was stopped or paused. Old raw outputs remain in the local audit, explicitly marked unsuitable for recommendations.

The regional and learning tests use small complete enumerations where completeness is the useful measurement; full-board tests use all 160 pieces where that is necessary. Unlimited mode is available for actual searches. Discriminating benchmarks have explicit bounds so a timed-out candidate is classified as inconclusive rather than consuming unbounded resources.

Closing this register does not prove that no other improvement exists. Historical tested candidates are recorded as closed or deferred with their evidence; they should not be presented again as new proposals without a changed implementation or new reason to retest.


## Clean GPU comparison on the original input

Hardware: RTX 2060 (6 GiB), i7-4790 (4 cores / 8 threads), approximately 16 GiB RAM, Windows, Python 3.12.14 and CUDA. A short screen compared four replica counts and eight feature profiles. The selected dynamic-MRV policy was then compared with the baseline in **baseline / MRV / MRV / baseline** order for 60 seconds each, using the identical 131,072-prefix frontier and equal warmup. CPU benchmark jobs did not overlap these runs.

| Measured work | Baseline repetitions | Dynamic MRV repetitions |
| --- | --- | --- |
| Main candidate attempts | 9.52B / 9.41B | 1.39B / 1.38B |
| Final exhausted length-3 prefixes | 1,837 / 1,837 | 4,192 / 4,192 |
| Final exhausted length-4 prefixes | 4,112 / 4,106 | 6,936 / 6,935 |
| Deepest placement | 91 / 91 tiles | 103 / 103 tiles |
| Newly excluded raw subspace relative to paired baseline | 1 / 1 | **25.393 / 25.396** |

The warmup had already exhausted 1,745 length-3 and 1,856 length-4 prefixes in every run; the comparison subtracts those. A prefix of length L covers `(160-L)! * 3**(160-L)` raw labelled completions. Length-3 prefixes therefore carry 471 times the raw weight of length-4 prefixes. These are **subspaces proved infeasible for Gold**, not full boards individually enumerated. Support scans are additional work outside the main-candidate counter. This comparison establishes neither a full-puzzle completion percentage nor a 25-fold solve-time guarantee.

Dynamic MRV allocated about 252 MiB of device arrays; display/driver memory is additional. GPU utilization was generally over 90 percent. The largest measured MRV dispatch was about 242 ms; Stop also waits for host work and checkpoint saving. More allocated VRAM by itself is not useful search progress.

For this GPU, the measured starting recommendation is **GPU DFS**, **NVIDIA CUDA**, **131,072 replicas**, **Unlimited**, with only **Choose the most restricted cell** enabled under Experimental GPU reasoning. The remaining new GPU switches are available for experiments; the best combination has not been established across all workloads. Selecting this new policy creates its separate resumable search folder and preserves the old default-policy checkpoint. Returning to the old settings resumes the old frontier.


## Completed CPU fixture comparisons

The clean native-model panel completed 84 cases: 70 validated Gold solutions and 14 infeasibility results. Independent checks also correctly classified 72 complete edge-matching fixture boards under two connectivity encodings (144 classifications). These fixtures test the encoded Gold condition; they are not original-puzzle solves.

Repeated SAT assistance on one finite context required **16 instead of 60 solver calls**, and reached 12 additional orientation removals when the old restarted short scan did not. A complete support scan reached the same narrowed domains using **20 instead of 33 calls**. Scoped cores and complete model witnesses account for the reused reasoning.

All 90 ordering comparisons completed with independently validated Gold solutions. On the ten repeated-mask cases, solution-oriented ordering used 100 main nodes and 255 propagations versus the baseline's 104 and 286. It won on five cases, tied three and lost two. Conflict-oriented ordering used 112 nodes but only 239 propagations; signature grouping used 120 and 305; impact priorities tied the baseline at 104 and 286. On the five ordinary cases, all six policies tied at six total nodes. These results support optional workload-specific priorities, not an all-on default.

A full-original-input 60-second SAT-propagator run additionally learned 55 independently checked partial Gold-loop exclusions before finding any complete edge-perfect candidate. This demonstrates actual-input early Gold reasoning, while leaving the full solution unresolved.


## Full-board CPU observations and superseded panels

The full-board CPU measurements are separated by source version and scheduling
policy in the [compact report](round23-results.json). The initial parallel
combined-pruning panel contained two startup errors: optional learned claims
were imported during a canonical-root transition under the wrong temporary
scope. The scope handling was repaired and both cases reran normally. Those
errors remain historical diagnostics; they are not search-performance results.

The repaired runs and the older 120-second parallel panel still used short fixed
worker leases. A worker could repeatedly prepare the same unfinished node when
its lease expired, increasing reported evaluations without advancing its saved
frontier. These panels retain sound interrupted checkpoints, but their node
counts do not establish newly expanded search or support a worker-count
recommendation. They are superseded for that purpose by a fresh adaptive-lease
comparison using a new frozen source capture.

The corrected 120-second runs reached longer leases but still completed only the canonical-root preparation transition, with no worker branches or completed jobs. A separate **300-second pair**, in reversed eight-worker/four-worker order, used the same frozen source, original input and seed 23071. Eight workers committed one native branch; four committed three. Their frontier-change counters were two and four respectively because each also includes the canonical transition. Neither run retired a job or made net cut-domain deletions.

Eight workers performed more internal conditional-support work (17.40M support steps, 425,682 internal removals, 87 completed cut passes) than four (14.43M, 254,792, 68), but that did not yield more committed search. Internal support removals can overlap or repeat and are not net board-domain reductions. The scheduler now increases a stalled job's allowance up to 60 seconds, preserves its scheduling hint across resume, and keeps cooperative Stop available. The longer run demonstrates that this lets workers get past the observed preparation bottleneck; it does not eliminate all repeated preparation or establish total solve efficiency.

For a CPU alternative on this i7-4790, start with **Parallel CPU DFS, four workers, Automatic/Compiled CPU, Unlimited**, retaining the existing default optimizations. Four workers is also the cautious starting count if experimenting with the combined heavy profile, based on this one bounded pair. Keep the new expensive controls opt-in: the combined profile remains experimental, and no fastest CPU configuration has been established. The GPU recommendation above has stronger measured pruning evidence on its tested frontier.

All new proposals have now been implemented and evaluated in their stated scopes. Finite fixtures establish additional deductions, solution preservation or reuse where reported; they do not establish that every option benefits the broad original root. The compact report preserves superseded diagnostics separately and distinguishes them from the corrected measurements.

## Release validation

The final Windows source passed the complete CI-listed CPU suite on both supported Python profiles: **1,121 tests on Python 3.12** (58 skipped) and **1,121 on Python 3.15** (167 skipped), with zero failures or errors. Skips cover platform-specific, opt-in hardware, and unavailable optional-engine checks; the compiled CPU core was required in both runs. A separate **44-test CUDA/Windows-WebGPU audit passed without skips**, including finite solution preservation, Gold validation, checkpoint resizing and CPU-owned endgames.

These are regression and bounded mathematical checks, not an original Gold solution or a proof that no defect remains. The GitHub workflow separately checks Windows, Linux, Intel macOS and Apple Silicon macOS, both Python profiles, setup scripts and software-Vulkan shaders before producing the four source archives. Physical AMD and Apple GPU execution remains untested.
