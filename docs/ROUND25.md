# Round 25: sequential search-reuse evaluation

Baseline: `v1.5.0` (`30483db81894359ef9030493e45d192a36f66c98`).

This register tracks the 37 requested proposals in their original order. Implementation and individual correctness evaluation proceed numerically; final frozen comparisons test useful combinations. Fewer search decisions and repeated reasoning take priority over elapsed time. Unknown, interrupted and capped computations never establish infeasibility. A Gold witness requires all 160 tiles, 240 matched seams and exactly one loop.

The machine-readable status is in [round25-candidates.json](round25-candidates.json). Pending entries are not implemented or validated claims. Any unavailable or rejected capability must retain its exact reason and evidence. Longer benchmarks are reserved for discriminating unresolved candidates, with identical inputs, explicit seeds and preserved source hashes.

The current implementation checkpoint covers proposals **1–25**, each with focused tests and independent review. A combined 96-test regional, UI, capability, benchmark-accounting and worker-process check passed after proposal 25. The earlier committed first-15 preflight passed all 26 cross-platform CI jobs. Proposals 26–37 are still pending; these checkpoints are not the final release gate. Final comparisons will rerun with frozen source and runtime provenance. [Regional scope and measured tradeoffs](ROUND25_REGIONS.md) cover proposals 14–25.

| ID | Proposal |
| --- | --- |
| 1 | Use one SAT failure to eliminate more branches |
| 2 | Keep one SAT assistant across related jobs |
| 3 | Resume interrupted proof verification |
| 4 | Generate explanations directly inside the native CPU solver |
| 5 | Preserve native SAT learning across application restarts |
| 6 | Represent completed and unfinished search regions symbolically |
| 7 | Share a common graph of verified deductions |
| 8 | Retain relationships involving several choices |
| 9 | Update the parity-equation basis incrementally |
| 10 | Learn weighted constraints from actual conflicts |
| 11 | Use exact integer divisibility constraints during search |
| 12 | Repair only affected piece-matching components |
| 13 | Maintain the CPU's possible Gold connections reversibly |
| 14 | Incrementally restrict and restore regional diagrams |
| 15 | Save unfinished regional calculations |
| 16 | Build compact constraint factors directly |
| 17 | Include Gold connectivity throughout whole-problem elimination |
| 18 | Make automatic decomposition drive the search |
| 19 | Refine diagrams where a false candidate exposes missing information |
| 20 | Compress inventory and boundary connectivity together |
| 21 | Construct connectivity representatives algebraically |
| 22 | Reuse proofs across verified equivalent residual problems |
| 23 | Extend regional pricing to seams and the single Gold loop |
| 24 | Check several unfinished chains jointly |
| 25 | Certify the completeness of regional results |
| 26 | Combine seam and inventory support in CUDA domains |
| 27 | Fork GPU branches after propagating their common parent |
| 28 | Make CUDA propagation respond only to affected constraints |
| 29 | Track more precise GPU failure explanations |
| 30 | Share verified general GPU clauses between lanes |
| 31 | Refill individual lanes in the newer CUDA domain engine |
| 32 | Use incremental, shared checkpoint representations |
| 33 | Resume sampling and repair phases exactly |
| 34 | Choose ejection chains from the actual obstruction |
| 35 | Combine parent boards through inventory-preserving cycles |
| 36 | Reject hopeless sampling moves before tracing every loop |
| 37 | Adapt replica-exchange temperatures from measured behavior |

## Evaluated scope through proposal 7

Proposals 1–7 have useful finite discriminators and are retained as optional features. Proposal 5 preserves a bounded, independently verified native clause subset on the tested Linux x86-64 and macOS x86-64/ARM64 environments, with CPython 3.12 and python-sat 1.9.dev15. Actual interrupted logging, fresh-process replay and all 72 finite puzzle solutions were checked. Windows proof logging remains disabled; saved subsets can still be independently verified and imported with logging off. This is not full native solver-state persistence.

Proposal 7 combines independently checked Hall shortages, proper Gold loops, affine contradictions, exact-integer weighted inequalities and RUP deductions in a bounded proof graph. Every node is scoped to immutable puzzle data, original domains and objective. Changed strand pairings invalidate live cached Gold deductions even if edge masks remain equal. Imported hashes identify content but never replace semantic checking; partial checks cannot admit an exclusion. New loop witnesses enter the live graph as they are found.

On the finite two-cell discriminator, independent enumeration finds 18 edge arrangements and no Gold loop. Two related certificates require 41 cold node checks versus 21 with shared completed nodes. This counts proof checking, not full original-puzzle search or total CPU work. Thirty focused proof/PB regressions pass, including a real floating-point cancellation forgery that is now rejected by exact-integer validation. The benchmark command is `python -B benchmark_learning_round25.py --candidate 7 --output report.json`.

A separate 156-test regression run for proposals 1–6 passed. These are developmental checks; final frozen-source validation remains outstanding.

## Proposal 9: incremental affine basis - retained, opt-in

`incremental_affine` implies `affine_reasoning`. A componentwise narrower domain box can copy a cached ancestor's exact GF(2) basis and add the complete affine-hull equations of only changed cells. Every ancestor equation remains valid under narrowing. Exact snapshots support rollback; an uncovered widening rebuilds. Candidate checks use private basis copies, and interrupted construction does not publish an unfinished basis. The cache remains bounded to four live contexts.

Thirteen nested consistent four-cell contexts reduce equation insertions from 1222 to 376 and Gaussian row-work steps from 3282 to 1168. This isolates basis construction; it excludes binding checks and dictionary copies and is not a speed ratio. Ten tests pass, including 180 differential narrowing/widening contexts, ancestor immutability, cancellation, scope mutation and actual DFS routing. No original-puzzle solve or branch-count improvement is claimed. Reproduce with `python -B benchmark_learning_round25.py --candidate 9 --output report.json`.

Proposal 5 is integrated as default-off `native_sat_persistence`, which enables a persistent SAT session. Its final supported-platform run passed 24 tests on each of Linux, Intel Mac and Apple Silicon. Completed fresh-process inventory decisions fell from 8957 to 4473 and from 1724 to 1509, while separate RUP verification cost 732969 and 3966599 clause visits. Complete finite puzzle enumeration retained all 72 boards but increased from 495 to 596 decisions with the imported auxiliary clauses. These mixed results justify an optional mechanism, not a universal speedup or an original Gold performance claim. [Detailed scope, limits and evidence](ROUND25_LEARNING.md) include the monitored native-log threshold and separate verification counters.


## Proposal 11: exact integer seam and inventory lattice - retained, opt-in

`integer_lattice` uses exact signed seam and inventory equations over every cell. Integer-span nonmembership can exclude a branch or candidate; a fixed candidate removes its own cell's difference family from the relaxation. Resource caps and interruptions remain unknown. Defaults bound 4096 placements, 512 active equations, 250000 integer operations, 128-bit coefficients, 32 candidate trials and four cached contexts.

The finite puzzle test removes three additional choices after ordinary, GF(2) and static-PB fixpoints, preserving six edge and four Gold completions. A planted near-complete 160-cell Gold witness remains valid; this is not the original puzzle's solution. Validation includes 4000 independent exact `Fraction` membership comparisons. In a separate 3000-context follow-up, 2129 contexts survived the old fixpoint and all 965 exclusions also followed from rational-span inconsistency: **no divisibility-only puzzle gain has been demonstrated**. A determinant-three arithmetic test establishes integer-specific strength separately. Ordering #11 before #10 saves two PB trials and 70064 scored terms on this fixture while adding 2793 integer operations; there is no universal dominance or speed claim. [Detailed scope and evidence](ROUND25_LEARNING.md) distinguish these counts.
