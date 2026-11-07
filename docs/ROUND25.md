# Round 25: sequential search-reuse evaluation

Baseline: `v1.5.0` (`30483db81894359ef9030493e45d192a36f66c98`).

This round implements and evaluates the 37 proposals below in their original order. Fewer decisions and repeated reasoning take priority over elapsed time. Each proposal must preserve valid solutions, handle cancellation and limits conservatively, and demonstrate a useful bounded effect before it is retained. A completed Gold witness still requires all 160 tiles, all 240 matched seams and one loop. **The original unrestricted Gold challenge remains unsolved.**

## Current checkpoint

Proposals **1–37** are implemented, have focused correctness tests and finite discriminators, and have passed independent review. They remain optional; a successful finite case does not establish a useful default for the original puzzle. Several mechanisms trade extra checking or setup work for fewer repeated queries, and some have measured adverse cases.

The first-34 [cross-platform preflight](https://github.com/gearashi/DiamondDilemmaGoldSolver/actions/runs/42366157306) passed all 26 jobs at `effb29b1431b14fb364b6ec248ccc3ff3f9ecd1a`, including Python 3.15 and the four source packages. The first-25 source checkpoint is `b3048ecf2df6b8eb4528b8985307e4459660eb66`. Its [cross-platform preflight](https://github.com/gearashi/DiamondDilemmaGoldSolver/actions/runs/42348120844) passed all 26 jobs, including the Python 3.12/3.15 CPU matrix, package installation, supported native SAT proof checks, WGSL checks and four source packages. The earlier [first-15 preflight](https://github.com/gearashi/DiamondDilemmaGoldSolver/actions/runs/42342504337) passed at `89a724e815f907cca2404b63f87a8d00d720f866`. These development checkpoints are **not the final release gate**.

All **37 individual implementation and evaluation gates are complete**. The [GPU report](ROUND25_GPU.md) records the completed focused gates for 26–37, including their additional verification and bookkeeping work. Final frozen-source comparisons, combined-feature regressions, measured hardware recommendations and the release build are also outstanding. The [machine-readable register](round25-candidates.json) is the per-proposal status authority; pending entries are not implemented or validated claims.

## Implemented scope through proposal 37

| Proposals | Retained mechanism | Boundary that matters |
| --- | --- | --- |
| 1–3 | Broader independently verified SAT exclusions, worker-level SAT sharing, resumable proof checking | Job assumptions remain scoped. Unfinished verification cannot exclude work; proof cursors currently survive only within the process. |
| 4–5 | Native CPU explanations and checked native SAT clause subsets across restarts | CPU explanations require a matching native build. SAT persistence is a bounded verified clause subset, not the native solver's complete state, and its logging producer is platform/version gated. |
| 6–8 | Symbolic work coverage, shared verified deduction graphs, complete higher-order projected relations | Cover changes preserve unfinished regions. Only fully checked proofs and completely enumerated projected relations can prune. |
| 9–13 | Incremental affine equations, conflict-guided weighted cuts, integer lattice reasoning, local Hall-component repair, reversible possible-Gold graph construction | Integer-lattice puzzle gains so far also follow from rational inconsistency; no divisibility-only puzzle gain is claimed. Reversible graph construction does not make global connectivity traversal incremental. |
| 14–15 | Reversible regional diagrams and saved unfinished regional computations | Complete support results alone prune. Unfinished saved cursors are trusted local continuations, not independently checked imported proofs. |
| 16–18 | Compact whole-constraint factors, exact Gold-aware elimination, automatic separator search | All cells and resources participate in whole-board modes. Width/work caps remain unknown; original-board tests use controlled domains, not a solution of the unrestricted challenge. |
| 19–21 | Counterexample-guided diagram refinement, correlated symbolic inventory/interfaces, algebraic connectivity representatives | Refinement positives are finite-factor witnesses only. Symbolic joins currently cover two complete regions. Representative bases preserve existence, not every history or placement support. |
| 22–24 | Verified residual graph equivalence, Gold-aware rational regional pricing, joint fixed-chain linkage | Isomorphism requires a complete checked mapping. Pricing requires exact Farkas and omitted-column checks. Positive chain linkage may omit optional strands and is only necessary. |
| 25 | Independently verified complete regional DAG certificates | Every code transition and legal dead child is checked. Source hashes bind inputs but cannot establish completeness. Completed certificates are separate from proposal 15's trusted unfinished cursors. |
| 26–30 | Joint GPU support, propagated child roots, event-driven propagation, precise GPU explanations and independently verified cross-lane clauses | Conservative fallbacks and complete original-domain verification preserve exact search. Preparation and independent verification costs remain separate. |
| 31–32 | Refill/donate CUDA domain work and retain shared checkpoint chunks | A branch closes only after all its descendants finish. Checkpoint storage is bounded, single-writer, and falls back to complete snapshots when needed. |
| 33 | Durable sampler trajectory and completed repair/controller state | Same-runtime native sampling can resume exactly. An interrupted native CP-SAT query retries; completed scoped queries and controller progress survive. Heuristic progress never proves exhaustion. |
| 34 | Build ejection chains from actual conflicts | Small independently enumerated fixtures need fewer proposals, with additional donor/rotation checks. This remains a heuristic, not pruning or a demonstrated original-puzzle improvement. |
| 35 | Inventory-preserving cycle crossover and bounded exact parent/child history | Gains are conditional on useful supplied parents. A separate eight-parent pool preserves nearby boards; every child is independently scored, and partial native writes poison the sampler rather than allowing a checkpoint. |
| 36 | Recover current scores exactly and avoid unnecessary loop traces | Same experimental proposals, RNG and boards in paired native tests. Bounded original-puzzle runs use fewer full score passes; this is not fewer proposals or a timing/solution result. |
| 37 | Bounded adaptation of replica-exchange temperature spacing | Native and host temperature state resume together. Paired synthetic cases include wins, losses and capped runs; the feature remains optional and does not establish improvement on the unrestricted puzzle. |

The [CPU learning report](ROUND25_LEARNING.md) records access flags, dependencies, tested configurations and exact scope for proposals 1–13. The [regional report](ROUND25_REGIONS.md) does the same for 14–25, including adverse comparisons and separate compilation, export and verification costs. Their numerical examples are **development evidence**, not final release measurements. The separate full hybrid pipeline benchmark records GPU proposals, repair queries and DFS decisions without adding those different units or claiming exhausted Cartesian volume.

## Ordered proposal register

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

## Validation and remaining work

A capped, interrupted, unavailable or uncertified computation returns unknown and preserves unproved work. Positive relaxation results do not become Gold solutions. Imported deductions remain bound to the puzzle, original domains, objective and applicable query scope. Full witnesses are independently validated before the app reports a solution.

The native SAT persistence producer has been checked with CPython 3.12 and python-sat 1.9.dev15 on Linux x86-64 and macOS x86-64/ARM64. Windows logging remains disabled; ordinary independently verified import is distinct from logging support. Gold-aware pricing needs OR-Tools/GLOP, and the app rejects an unavailable explicit selection with an explanation. These capabilities must not be inferred merely from general OS or Python compatibility.

The completed first-25 gates cover independent finite enumeration, narrowing and widening, scoped reuse, malformed proofs, cancellation, resource caps, public solver integration, and relevant spawned-worker paths. Final comparisons will test retained combinations on identical inputs and seeds, with all setup, checking and replay work reported separately where the operation types differ. Longer runs are reserved for unresolved discriminators; extra runtime cannot turn an unknown result into a proof.

Reproduction entrypoints are `benchmark_learning_round25.py --candidate N --output PATH` for CPU learning evidence and `benchmark_regions_round25.py --candidate N --output PATH` for regional evidence. The regional runner's `--evidence-stage frozen` requires a readable clean Git commit, unchanged repository identity, and matching before/after source and runtime provenance. Development receipts remain in the ignored audit directory until the release-wide frozen rerun supplies final evidence.

After proposals 33–37 finish, the remaining gates are frozen benchmarks and combination decisions, full platform/setup checks, measured starting settings for the target hardware, and CI on the exact proposed release tag. Completion of this documented candidate set cannot establish that no future algorithmic improvement exists.

### Benchmark accounting

The comparison tools reject optimized Python because their finite discriminators require assertions. Frozen runs require a clean committed checkout and unchanged source, dependency and input snapshots. GPU and persistent-SAT original-input repeats use deterministic search; their seed labels are repetition identifiers, not independent randomized searches. SAT comparisons start from the same checked partition of up to 32 original-root cubes.

GPU comparisons warm the exact kernel specialization without search and report that cost separately. Every GPU policy uses the same bounded asynchronous checkpoint publisher. `solver_wall_seconds` includes solver work and publication drain; worker/coordinator times additionally include artifact copies, validation, dependency hashing and process setup. Published JSON bytes and retained store bytes describe logical contents, not physical SSD writes. Saved references and compressed dependencies are reopened and checked, while native continuation is tested separately.
