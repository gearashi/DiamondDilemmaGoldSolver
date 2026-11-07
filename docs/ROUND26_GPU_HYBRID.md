# Round 26 GPU and hybrid workstreams

This document tracks the complete F14-F17 scope, including every lifecycle and backend variant. The authoritative 81-item register is the round26 scope record. All new controls remain optional; an implemented path is not called retained until its independent correctness and work comparison pass.

## F14: exact GPU reasoning

The current implementation includes bounded support-witness reuse, watched clause state, global inventory matching, bounded structural proof production and an alternative warp execution layout. Independent correctness review is clear; the speed-only cooperative execution disposition still awaits the final controlled comparison. Caches are hints: every hit checks current domains and immutable compatibility, and every miss uses the existing complete path. Cache sizing, saturation, eviction, root rebasing, lane replacement and checkpoint validation are part of this family. Global matching repairs retained matches, rebuilds dirty old components, and removes unsupported physical-piece choices using alternating reachability. Every matching stage is resumable. Structural proof claims must pass the independent CPU checker; missing learned-clause premises and caps yield no proof. Cooperative execution parallelizes seam-word checks while retaining identical logical trajectories, and still awaits its speed comparison.

The exact domain engine currently runs on CUDA. Existing compact CUDA/WebGPU solvers remain distinct engines; unsupported domain-engine/backend combinations must continue to reject before search mutation. Any added per-lane allocation must be included in the authoritative memory estimate.

## F15: incremental sampling scores

Scope includes every move class and both CUDA/WebGPU. Seam deltas and strand-component changes must agree exactly with the existing full scorer. Acceptance, RNG, best-board updates, injection/exchange, Stop/resume and full-score fallback are acceptance requirements. The implementation and finite work gate passed on both backends; its bounded evidence and adverse case are recorded below.

## F16: recombination

Scope includes interacting parental cycles, rotation differences, bounded multi-parent/enlarged neighborhoods, reusable optimization progress and adaptive pair/neighborhood allocation. Any heuristic outcome remains separate from exact search exclusions. The host optimizer and both native integration paths passed independent correctness review; source-bound final evidence remains part of the release gate. See [complete recombination scope](ROUND26_RECOMBINATION.md).

## F17: retained repair

Scope includes guarded models for changing patches/outside assignments and objective bounds, all applicable repair modes, completed bound/core reuse, policy allocation, rebuild/eviction and durable pending queries. Interrupted native searches may require retry; native-state portability is not assumed. The ordinary repair service, Gold feasibility pool and minimum-change/fragment master reuse have passed their focused correctness and independent lifecycle reviews. The final frozen work comparison remains a release gate.

## Validation contract

Use independent tiny enumeration and stage-specific discriminators, then representative completed residual work, lifecycle interactions and the final frozen integrated matrix. Record added construction/checking work separately from saved search decisions. A timeout is unknown, and neither an unfinished optimizer nor an unverified proof can remove global work. No original Gold solution is claimed.

## Current F14 evidence and limits

Development checks preserve independently enumerated solutions with support caching, watches and matching; include learning disabled, repeated-neighbor seams, word boundaries, every matching stage, and a controlled160-cell Gold witness. A genuine deficient-inventory fixture is rejected at zero decisions. Component fixtures demonstrate fewer rebuilt rows than rebuilding every row on every matching pass. These are finite mechanism checks, not a result for the original Gold challenge.

The alternative CUDA policy uses one warp per lane and column-major state. Scalar and warp kernels agree on logical native arrays and complete finite answers; parallel misses inspect all support words, while scalar misses may stop early. Counter differences are charged explicitly. No speed winner has been established. Actual fork, selected-lane replacement, untouched-column preservation and factored checkpoint reconstruction have been exercised. Native local checkpoints remain trusted process state, not proof imports.

The complete F14 correctness suite passed 47 tests, including actual clause-ring overwrite, proof transport, all seven matching-stage import barriers, fork/refill, policy changes, backend rejection and authoritative memory sizing. A subsequent bounded proof-cache refinement has its own host and finite transport checks. Independent support/watch/matching/proof and cooperative-layout review is clear. The family remains open only for the cooperative speed-only disposition; no speed claim is inferred from correctness timings.

An isolated development snapshot contains 358 copied source/native/data/workflow files with before/after original hashes and exact copied hashes. Its 22-row finite matrix passed with unchanged snapshot sources/runtime. Support scans fell 1016 to 503, 1477 to 298 and 781 to 345 on three fixtures, with unchanged search decisions and separately charged witness reads. Tetrahedron clause literal checks fell 58 to 7, adding 36 watch checks; the prism case was neutral. Nonlocal inventory deficiency fell from two decisions to zero; a second fixture removed three unsupported candidates while preserving four solutions. Matching construction/reachability work is charged separately. Cooperative variants preserved decisions/solutions but read more support words, so their retention still requires the coordinated speed comparison. Structural proof transport, scope minimization and restart replay used zero SAT queries on the bounded discriminator; CPU replay work is reported separately from native search and SAT model/query work.

F15 is implemented for CUDA and WGSL with independent lifecycle review clear. F16 host optimization and native integration are likewise independently reviewed; F17 implementation and independent lifecycle review are also complete; the final frozen family comparison remains pending.


## F15 development evidence and lifecycle

The native implementation passes the 14 new CPU/CUDA/WGSL checks: all 32 combinations of proposal policy and score objective per backend, exact old metadata/RNG/board trajectories, independent physical-component labels, fixed/no-free inputs, actual160-cell injection and exchange, scratch-only native restore validation, malformed-cache/objective rejection before writes, 18-dispatch versus9+fresh-restore9 continuation, and 64-bit counter carry. Two existing Gold-host checks and seven existing sampler-host checks also pass. These checks were run in an isolated375-file source/data/native snapshot; the first command named one supplemental class incorrectly, and that loader-only error was corrected and the seven-test class run separately.

Twelve finite paired comparisons per backend preserve the same native proposals, acceptance and RNG. Both backends give the following bounded original160-cell Gold counts at384 lane proposals, including initialization and accepted cache maintenance:

| Policy | Domain lookups, old to new | Seam comparisons, old to new | Strand walk steps, old to new | Added component-label reads |
|---|---:|---:|---:|---:|
| Ejection |123520 to3558|185280 to9632|580337 to19223|6116|
| Guided/tabu/exchange/lazy |62080 to3996|93120 to10886|301989 to20483|6010|

The tiny tetrahedron combined case is adverse: domain evaluations3104 to5220, seam comparisons4656 to9078, strand steps4515 to5492, plus3301 cache reads. A changed term charges both its old and new evaluation. These counters omit candidate generation, metadata traffic and duplicate-label comparisons; they establish reduced repeated scorer work in the stated cases, not fewer proposals, a universal timing improvement, or a Gold solution.

The cache uses immutable physical segment IDs and current-board proper-loop labels. A rejected proposal leaves it untouched; accepted moves clear affected old loops using a virtual old board and fill affected new loops. Exchange swaps labels with current boards. Arbitrary host injection performs a full selected-lane reconstruction. Both objectives and every native move class share this path. Restore rebuilds current and best scores and current labels in scratch before writing live state; additional verification work is reported separately in `restore_validation_work`, not hidden inside preserved trajectory counters. Failed partial device writes poison the session.

`state_size`/`validate_capacity` are the authoritative native-state estimate. At160 cells and365 segments, incremental Gold sampling uses228 metadata rows and2192 payload bytes per replica. The256MiB durable snapshot limit therefore permits122461 replicas;131072 is rejected before backend selection. This is a payload/checkpoint bound, not a measured VRAM peak or a recommendation to maximize replicas. Device buffers, temporary restore buffers and driver memory require additional space. New native scorer counters use paired uint32 words; unrelated existing legacy counters retain their documented modular scope.


## F16 native integration and continuation

The optimizing crossover is shared by CUDA and WebGPU. It uses the independently rescored exact-distinct parent pool, then sends a validated child through the existing atomic native injection path. The sampler commits completed device collection before optimizer work and saves `crossover_due` when cooperative cancellation interrupts that work. Resume finishes the pending optimizer without rereading GPU candidates or advancing the sampler. Other exceptions poison the session, preserving the last durable checkpoint. A pending Gold child requires independent full validation; optimizer bounds never become global search exclusions.

Six integration checks passed: host legacy forwarding and pending-optimizer continuation, actual injected Gold children on both native backends, and combined incremental scoring/temperature/exchange/optimizer160-cell18 versus9+9 dispatch continuation. Native arrays and retained optimizer/frontier/history state agree. Restored static-model construction and source-verification work are deliberately charged separately. The host neighborhood suite and native integration received independent review clearance. This does not establish an original-puzzle timing or solution gain.

## F17 retained model and continuation boundary

The ordinary retained service builds a global soft-edge CP-SAT model once, guards each patch/outside/objective context explicitly, and negates previous query guards. Completed exact-context results, independently validated Gold cuts, feasible incumbent hints, valid objective bounds and unfinished query identities survive bounded local checkpoints. A new native CP-SAT solve starts for every unfinished attempt; native learned clauses and active native search are not serialized. Guard exhaustion rebuilds the model. Model-size exhaustion returns an inconclusive `MODEL_BUDGET` result, never a deduction.

Gold cuts can invalidate an earlier heuristic board without invalidating an upper bound established before the new cut. The service therefore stores the best heuristic board separately from an incumbent feasible under every current cut. A new cut invalidates affected incumbents and completed optimum claims, preserves valid upper bounds, and reopens the exact query. Only a currently feasible incumbent can become a hint; a Gold-mode partial board does not supply a lower bound that could exclude a better Gold arrangement. Exact-context upper bounds are applied under the new query guard after rebuild. Returned heuristic boards remain independently validated puzzle arrangements, but are not presented as current-model optimum certificates.

The service also owns the retained Gold feasibility pool and minimum-change/fragment master models. These reuse checked cores, witnesses and completed semantic results across their applicable contexts. Ordinary improvement, core repair, fragment completion and scoped feasibility all use the retained service when selected; loop trades need no native model. See [Gold repair models and evidence](ROUND26_GOLD_REPAIR.md). Adaptive patch/mode selection rewards completed outcomes only, reserves exploratory budget, and saves the selected order and allocations with the pending controller. UNKNOWN does not become a completed result.

Limits are explicit: 64 guarded queries before ordinary model rebuild, 128 ordinary completed-result records, 256 Gold cuts, a 64 MiB model-text cap and a 32 MiB ordinary service snapshot cap. Pool limits and their independent counters are documented separately. Eviction loses optional reuse, not puzzle coverage. Direct calls close their own native owners; application-owned services stay warm until application cleanup. Strict source, domain, geometry and physical-segment bindings reject incompatible reuse. The checkpoint reader normalizes JSON sequence representation while independently rescoring saved archive entries.

The final combined host regression command passed 103 tests with no skips. It covers new ordinary and Gold-pool checks, retained benchmark guards, regional proof integration and earlier ordinary/resumable/hybrid controller regressions. The two cooperative benchmark host checks were rerun after removing their dependency on prepared puzzle data and passed. Independent review found no remaining ordinary-service or Gold-pool lifecycle blocker. This is correctness evidence, not the final frozen work comparison. The release coordinator will run `benchmark_repair_round26.py --output <fresh.json> --evidence-stage frozen`, which charges model construction, native calls, branches/conflicts, restore and verification separately.

## Regional reasoning to independently checked learning

Regional deduction transport uses the same original-root guarded chain checker as GPU claims. F11 certificates replay their complete DSU transitions and dead children; F12 reasons replay their precise connectivity checks and retain the Gold-only target restriction. F09/F10/F13 results are transported through a fresh bounded semantic re-proof using a complete DSU certificate. They are not treated as compact proofs of the original compiler, overlap or integer algorithm, and their saved internal completion flags never authorize learning.

Each candidate exclusion retains a singleton candidate guard. The optional queue keeps at most 16 claims and 64 attempted identities; fresh re-proof defaults to 2048 states and 20000 transitions, with a 1 MiB certificate bound. Caps discard optional learning, not search work. The proof queue retains a yielded certificate until its consumer finishes admission, so cancellation does not consume an unadmitted claim. Nine focused checks passed, including every regional stage, malformed/under-scoped certificates, Gold-only rejection, independent solution preservation, and the actual public DFS with all regional options and smart learning. The latter interrupts admission after the search trail is committed, resumes safely, and separately verifies public-store admission in an uninterrupted run. Independent review cleared this bridge with the bounded re-proof limitation stated here.

## Final cooperative execution comparison

The speed-only alternative remains undecided until the controlled frozen run. Use `benchmark_gpu_round26.py --mode cooperative --output <fresh.json> --evidence-stage frozen --repeats 5 --dispatches 32 --quantum 16 --replicas 128`. The runner alternates scalar/warp order after separate constructor and primitive warmup, pairs exact initial/final logical state and work, reports wider support-word scans separately, and times synchronized dispatches including their status copy. Allocation and independent validation are separate measurements. The bounded comparison includes a word-boundary fixture and original puzzle input; it does not measure a complete search, worst-case Stop latency or time to Gold.
