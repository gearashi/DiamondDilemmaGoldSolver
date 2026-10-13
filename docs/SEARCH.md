# Search, checkpoints, and validation

## Board and orientation model

The board is a closed pentagonal bipyramid: ten triangular faces, each divided into sixteen small triangles. It has 160 tile cells and 240 shared edges.

Each physical tile is labelled. An orientation code is `3 * tile_index + rotation`, with rotation in `0, 1, 2`. Reflections are not allowed. Endpoints use eleven positions on an oriented edge; adjacent sides match with the endpoint order reversed.

The raw labelled space is `160! × 3^160`, approximately `1.03 × 10^361` arrangements. New unrestricted CPU DFS searches (including the exact phase of hybrid) and rebuilt native SAT/CP models retain one representative of each group of ten proper whole-board rotations. Existing DFS checkpoints retain their original coverage; native engines already restart their internal search. Reflections and identical-looking physical tiles are not merged. The scale makes exhaustive completion uncertain; neither a timeout nor an unchanged best depth establishes impossibility.

## Gold-only constraint search

The dashboard exposes `dfs`, `parallel-dfs`, `cp`, `cp-sat`, `cp-sat-circuit`, `sat`, and `hybrid`, implemented by `constraint_search.py`. All seven require every tile once, all matching seams, and one complete gold loop. An edge-perfect arrangement with multiple loops is rejected and search continues.

Strong DFS uses bitset domains, minimum-remaining-values cell ordering, seam arc consistency, global physical-tile matching, restricted local tile-competition checks, forced-neighbor clusters, face-boundary inventory and mesh-parity checks, bounded learned conflicts, and early rejection of closed proper loops. A rejected partial assignment discards all its impossible completions. It stores each pending branch and its next choice, so normal resume does not re-enumerate consumed complete assignments. Periodic checkpoints are written between 30-second search slices; Stop also saves a checkpoint.

SAT uses Glucose through PySAT. CP-SAT and traditional CP use OR-Tools with tile uniqueness and seam constraints. Independently detected closed components generate exclusions of the conjunction of placements preserving that component. This is a sound exclusion of impossible Gold completions, not a heuristic score threshold. Native engines save these validated exclusions and hints; their internal search, propagation, and learned clauses restart after a process restart. A checkpoint timestamp for these engines does not mean an exact native search cursor was saved.

### Experimental direct Gold circuit

Select **CP-SAT direct Gold circuit (experimental)** in the dashboard, `--algorithm cp-sat-circuit` in the portable launcher, or `-Algorithm cp-sat-circuit` in `StartSolver.ps1`. Ordinary `cp-sat` remains available. This mode uses one CPU solver worker; the GPU replica setting does not apply.

The direct model chooses each physical tile exactly once and includes every segment of its chosen rotation. It connects occupied seam endpoints with their order reversed. A single optional-node circuit constraint forces all occupied ports into one loop; unused ports take self-loops. Drawn crossings remain separate strands. Consequently the native solver can reason about the one-loop condition before producing a complete arrangement. Every returned board still passes the independent Gold validator.

Identical support sets share Boolean literals where present, and fixing the traversal direction of tile 1's first segment removes the two equivalent directions around a finished loop. This direction convention does not restrict where the tile is placed. These are encoding reductions, not a demonstrated real-puzzle speedup. The current full input has no support groups available for sharing.

Construction is limited to 160 cells, 80,000 candidate placements, 200,000 directed circuit arcs, and 1,000,000 linking terms. The current unrestricted input, after the existing proper-rotation reduction, fits these limits: 76,368 placements, 5,280 possible port nodes, and 147,840 arcs. It is still a large model. Reaching a cap reports a visible unsupported-model error and an incomplete result; it never records exhaustion. Stop is checked during construction and interrupts native search. Resuming rebuilds the model and restarts native search, so this mode does not offer the exact pending-branch continuation of DFS.

The shared solver API also accepts an `edges` target for tests: in that case this engine explicitly uses ordinary CP-SAT without a single-loop constraint. The dashboard always requests Gold. See [the comparison and finite validation scope](ALGORITHM_COMPARISON.md#direct-circuit-validation).

GPU sampling uses a fused local move delta and keeps per-launch counters in local variables; WebGPU gathers selected candidate boards in one dispatch before reading them. These preserve sampling semantics and do not make stochastic proposals exhaustive.

The experimental hybrid runs native CUDA or WebGPU sampling, then CPU neighborhood repair with CP-SAT, then the complete smart DFS tree. It independently scores the saved best permutation and GPU samples, using the better candidate for repair. Each patch frees both mismatched cells and nearby matching cells, preserves the physical-piece inventory and fixed placements, and tries several sizes. A patch timeout or infeasible patch never proves the puzzle impossible. The first 30-second search slice budgets about six seconds for sampling and nine seconds for repair; unused time goes to DFS. The standalone hybrid API caps their requested budgets at 30 and 60 seconds respectively; GPU initialization or native calls can overrun, and elapsed time is deducted before DFS. A missing optional repair dependency is reported and DFS still runs.

Neither sampling nor repair fixes guessed placements in the exact search. Returned boards guide value ordering only. A certificate from either phase is accepted only after full independent Gold validation. GPU sampling keeps up to eight distinct, independently scored candidates with nearby scores, favoring different placements. Repair varies patch size, switches seeds when stalled, and excludes independently identified closed proper loops. It skips an identical local query only after a conclusive local solve; a timeout is never an exhaustion record. Samples can still repeat, and the local cache lasts only for that repair invocation. Once DFS begins, resumes continue its stack without repeating the heuristic phases. A fresh hybrid search considers the saved best board automatically; an existing hybrid checkpoint keeps its original search order.

### Smart exact pruning

- **Possible-loop connectivity (Gold only):** build an overapproximation of seams that can carry gold. If this cell graph is disconnected, a single loop cannot visit every nonempty tile. Any bridge must carry a positive even number of endpoints, so incompatible orientations are removed. Parallel seams remain separate graph edges. The edge-only target does not use this deduction.
- **Incremental work and exact caches:** reuse identical seam-filter inputs, complete physical-tile matching results, and immutable cluster embeddings. A new in-memory child of a fully propagated parent can start with only affected seam checks; resumed and interrupted frames rebuild their queues. Caches are bounded and keyed to their exact inputs; eviction only loses speed, never coverage.
- **Faster loop checks:** precompute endpoint-to-segment tables and cache exact seam joins. The independent final validator still follows explicit endpoint connections. A fixed table supplies all 2,048 possible eleven-bit mask reversals.

- **Proper board rotations:** for a fresh unrestricted search, verify all ten orientation-preserving mesh automorphisms and restrict tile 1 to face 1. All sixteen small cells and all three tile rotations within that face remain possible. This removes a factor of ten in rotation-equivalent full arrangements, not a guaranteed factor of ten in elapsed time. Fixed placements or narrowed input domains disable it. Checkpoints explicitly record the root restriction; legacy checkpoints are never silently changed.
- **Faster equivalent propagation:** group placements by edge signature when domains are broad, use individual support tables when sparse, deduplicate queued seam checks, and reuse immutable tables across search slices. The physical tile on both sides must still differ.
- **Mesh parity:** seam crossing parities must be consistent with binary values at mesh vertices. For the 365-segment input, the two apex values must differ. Forced parities can reject inconsistent branches or remove orientations before all exact edge masks are known. This deduction is enabled only on the verified bipyramid topology.
- **Local tile competition:** for a proposed center placement, its neighbors must have distinct supporting physical pieces. Parallel seams to the same neighbor must be supported by one orientation. Default checks are restricted to center domains of at most 24 orientations to control cost.

- **Forced neighbors:** count original physical edges by their eleven-position endpoint masks. On a closed board, an edge with exactly one compatible different physical edge forces that adjacency. Every possible embedding of its forced cluster is retained; impossible anchors and rotations are removed. The rule is disabled on open fixture boards.
- **Global tile competition:** find a perfect matching between cells and physical tile IDs, then remove choices that belong to no such matching. Three cells competing for only two pieces fail immediately. Rotations never increase the inventory.
- **Face-boundary inventory:** internal seams consume reversed pairs of edge masks, including empty masks. The unpaired surplus must fit the region boundary (12 small edges for a full face). Apply this when a region's candidate inventory fixes its set of physical tiles, and prune only consequences proved from that inventory. An unrestricted face does not have a known tile group to reject.
- **Learned conflicts:** cache bounded, proven-impossible conjunctions of placements and use them to reject or prune later branches. An applicable existing clause can cause an immediate nonchronological backjump. Otherwise, bounded contradiction probes can remove unnecessary assumptions, starting from effective initial domains. Only proved exclusions discard branches; timeouts never create conflicts. Generic conflicts remain private to the job and survive only in memory; independently verified proper-loop exclusions may also persist as described below. Existing v2 DFS checkpoints remain accepted.

These are sound reductions for the encoded puzzle. Bounded tests measure some propagation gains, but do not establish a faster real-puzzle solve; see [optimization evidence and limits](OPTIMIZATIONS.md). The dashboard reports forced bonds, removed choices, rejected face groups, learned conflicts, and repair attempts separately. These counters are not unique full-board counts.

Each mode saves under `runtime/searches/<algorithm>-gold/`. A completed exhaustion record is bound to the input, algorithm, and source hashes, and is reused without restarting completed work. It is recorded solver evidence, not an independently checkable UNSAT proof. A saved solution is always independently validated again. These checkpoints are separate from the existing GPU pool and from each other. Search counts cannot be added across algorithms as unique coverage. Parallel CPU DFS accepts a selectable worker count; the other CPU modes use one worker. GPU replica counts affect GPU work only. See [the algorithm comparison](ALGORITHM_COMPARISON.md).

## Systematic GPU search

The legacy `systematic_search.py` backend fixes a cell order and partitions the search into prefix jobs. Refining a job replaces it with **all** unused-tile orientations compatible with the already assigned seams. A cancellation or resource cap preserves an unsplit parent rather than dropping its remaining children.

The resulting frontier is prefix-free: no job duplicates another job or contains another job as a descendant. Each active GPU lane owns one job. Surplus lanes stay idle. An available lane can continue a paused job from its saved cursor or receive a previously unassigned job.

Within a branch, DFS stores its candidate-list cursor at every entered depth. Candidate ordering can vary, but a child is tried once under that partial assignment. Edge contradictions reject all full arrangements extending that partial board. Backtracking revisits ancestors; this does not repeat a complete arrangement.

The CPU checks a complete candidate independently. The configured stopping condition is the first full edge match, not continued enumeration of all edge matches. Multiple closed gold loops satisfy the edge check but do **not** solve the Gold Challenge.

## Progress is not a completion percentage

| Display | Meaning |
| --- | --- |
| Tiles placed | Occupied cells in the live partial branch |
| Deepest placement | Greatest branch depth reached in this run |
| Active branches | Assigned jobs on the current GPU lanes |
| Paused branches | Unfinished jobs retained off the GPU, with their search positions saved |
| Waiting branches | Paused jobs plus jobs that have never been assigned |
| Completed branches | Jobs whose entire subtree was exhausted |
| GPU nodes | Candidate attempts, not globally unique full boards |
| Historical best | A previously saved complete placement |

Jobs have unequal subtree sizes. Completed-job counts, depth, and node throughput cannot be converted into a reliable percentage of the whole search or time remaining.

Only exhaustion of the complete, correctly bound frontier can establish that no edge-compatible full arrangement exists for the encoded input. Finite tests and a bounded search run do not establish that result.

## Durability and stopping

The systematic checkpoint stores active and paused DFS stacks, candidate cursors, fixed prefixes, job IDs, and the allocation cursor in **one atomic NPZ file**. Restore validates the DFS state and binds every owned branch back to its immutable frontier job.

To change the replica count, Stop and wait for the final checkpoint, select the new count (128–131,072), then Start. Reducing the count moves excess unfinished branches into a host-side paused bank without restarting or completing them. Increasing the count makes more GPU lanes available to continue saved branches. This reduces active GPU work when fewer replicas are selected; paused state still occupies host memory and checkpoint storage.

A coherent host snapshot is captured between bounded GPU launches. One background writer saves it while searching continues. Checkpoints are uncompressed to reduce compression latency; the default periodic interval is 30 seconds after the preceding save finishes.

Stop waits for current work and the required final save. An older write may need to finish before the newest state is written. Shutdown speed depends on the state size and storage load.

Normal Stop/Resume retains cursors and ownership. A crash can replay work performed after the last completed checkpoint; the program does not claim exactly-once execution across a crash. Never discard, replace, or roll back a checkpoint if preserving its coverage is required.

For legacy GPU DFS, `--seconds 0` means Unlimited. It does not disable checkpointing or automatic stopping at the first 240-edge candidate. The new constraint modes instead stop automatically only for a verified single-loop Gold solution or exhaustion.

## GPU compute backends

The systematic runner accepts `--backend auto`, `cuda`, or `webgpu`. CUDA targets NVIDIA GPUs on Windows/Linux. WebGPU runs a native compute kernel through Metal on macOS or Vulkan/DX12 on compatible PCs; the browser only displays and controls the search.

CUDA and WebGPU share the systematic checkpoint schema, including paused branches. Bounded interchange tests on an RTX 2060 passed transfer in both directions and continued search after restore. The puzzle data, geometry, cell order, and immutable frontier must agree; retain the entire output directory and its input snapshots when changing `--backend`. The stochastic and standalone drivers remain separate formats.

Both Intel and Apple Silicon Macs use the same Python source with native dependencies for their CPU architecture. Driver support and adapter memory/compute limits still determine whether a particular machine can run the requested population. Package imports and CPU fixtures are separate from testing a kernel on actual GPU hardware.

## Separate backends

Use separate output directories to avoid mixing status files or certificates.

```powershell
# Default systematic method.
.\venv\Scripts\python.exe systematic_search.py --resume --seconds 0 --replicas 4096 --output runtime

# Optional stochastic full-board search.
.\venv\Scripts\python.exe solve.py --resume --seconds 3600 --replicas 4096 --output stochastic-runtime

# Standalone DFS driver for experimentation.
.\venv\Scripts\python.exe dfs_gpu.py --seconds 120 --replicas 3840 --output dfs-runtime
```

On Linux/macOS, replace `.\venv\Scripts\python.exe` in these commands with `venv/bin/python`. Forward slashes work for data and output paths on all platforms.

The dashboard and launcher select the requested algorithm within the main runtime; command-line launchers keep the legacy GPU default for compatibility. The other drivers have their own checkpoint behavior; inspect their `--help` before use. To request their stop, write a `stop.request` file in that driver's output directory. The standalone DFS driver may require removal of that request before restarting.

The stochastic backend's recent-board cache is bounded and evicts old entries. Different replicas have separate caches, and old placements can recur. Its persistent validation cache avoids repeating CPU report calculations; it is not a ledger of exhaustive search coverage.

## Validation and evidence

A Gold solution requires all of the following:

1. Every physical tile appears exactly once, with an allowed rotation.
2. All 240 shared edges have matching gold endpoints.
3. Every gold segment belongs to exactly one closed connected loop.

Segments may join two points on the same tile side. Geometric crossings of drawn strokes are not graph junctions.

Native Vulkan checks on an RTX 2060 enumerated the expected 54 synthetic arrangements while exercising resizing, frozen pending candidates, and counters. A comparison using actual puzzle masks and a varied cell order matched every CUDA device-state field after each of 20 launches, each with a 32-node budget per lane across three lanes. CUDA-to-WebGPU and WebGPU-to-CUDA checkpoint restoration and continued execution passed. An isolated 128-replica systematic run also saved, restored, and advanced its saved search state. These finite results establish the tested compatibility cases, not correctness on every GPU or every possible search state.

CPU tests exercise topology, small exhaustive prefix covers, checkpoint ownership, terminal validation, and control behavior. Synthetic host arrays test runner scheduling without CUDA. Separate GPU tests check actual kernels. Bounded WebGPU tests run on an NVIDIA Vulkan adapter; CI also executes finite shader checks through Mesa lavapipe with an explicit test-only software-adapter opt-in. Neither establishes AMD or macOS Metal hardware correctness. Source-image extraction checks require locally prepared diagrams and data. None of these test categories substitutes for independently validating a complete candidate.


## Parallel CPU search and advanced controls

`parallel-dfs` uses spawned CPU worker processes and a separate saved queue. Each job owns a disjoint part of the unfinished DFS tree. The coordinator partitions only committed pending choices, validates incoming job checkpoints, and retains unfinished jobs on cancellation or worker failure. The persisted queue includes active leases, so an intermediate-snapshot resume explicitly records potential replay of uncommitted work. Queue size can grow when a job is split; it is not a completion percentage.

The dashboard's **Exact-search optimizations** controls apply to `dfs`, `parallel-dfs`, and the exact phase of `hybrid`. CLI callers can supply `--search-options` as a JSON object with these Boolean fields:

| Option | Effect |
| --- | --- |
| `failure_guided` | Prefer variables implicated in observed failures when choosing among tied domains. |
| `lookahead` | Try a bounded number of singleton assumptions before opening a real branch. |
| `strand_connectivity` | Check connectivity of possible individual Gold endpoints. |
| `multi_seam_cuts` | Check correlated orientation choices across small seam cuts. |
| `regional_cache` | Enumerate and cache entire eligible 2–4-cell regions with fixed piece inventory. |
| `biconnected` | Reject candidates that cannot fit one possible cyclic endpoint block. |
| `separator_regions` | Build bounded frontier summaries for regions of up to eight cells. |
| `backjumping` | Reuse proved conflicts, minimize them with bounded rechecks, and skip excluded intervening branches. |
| `incremental` | Skip completed unchanged propagation dependencies, including rotation-only changes irrelevant to piece matching. |
| `adaptive` | Temporarily defer expensive unproductive optional checks and explanation probes, with periodic retries. |
| `persistent_learning` | Import, share, and save independently verified proper-loop placement exclusions. |

All these fields default to `true`. `native` selects `auto`, `python`, or `compiled`; it defaults to `auto`. Parallel search also accepts `--workers 1..64`. The portable launcher accepts these controls; Windows `StartSolver.ps1` supports `-Algorithm parallel-dfs -Workers 4` with the default optimization settings. Options change future search choices and scheduling; saved sibling lists remain intact.

Failure guidance changes future branching priorities. Look-ahead uses private domain copies and removes a value only after a completed contradiction. Strand and cut deductions are Gold-specific. Regional support results come from complete bounded local enumeration and may be reused only for exactly matching subproblem keys. A budget limit is unknown, never an impossibility result. These caches do not replace the persistent unfinished DFS tree.

The compiled backend runs the seam-consistency queue in bounded native batches and polls Stop between batches. It uses the same exact integer support relations as the reference Python engine. Remembered per-value supports are reused only after checking both static compatibility and membership in the current neighbor domain; backtracking and domain widening cannot turn stale support into a deduction. A missing remembered support triggers an ordinary support search. All complete candidates still pass the independent Gold validator.


### Conflict explanations and propagation scheduling

A contradiction first commits the failed frame, so cancellation during subsequent optional explanation work leaves a valid ordinary DFS stack. Explanation minimization tries at most four assumption deletions under a shared cooperative 10 ms budget. Each deletion must be reproved from the effective initial domains using sound constraints. Already proved applicable clauses can jump without another proof. A jump retains the responsible ancestor frame's complete sibling list and next-choice cursor; it discards only continuations excluded by the clause. This is bounded conflict minimization, not a full lazy-clause-generation engine.

Scheduling caches at most eight completed unchanged dependency signatures per phase. Matching depends on physical-tile availability; cluster checks include relevant orientations and outside-piece presence; inventory checks include boundary orientations; parity and connectivity include the endpoint categories read by those deductions. Domain widening, backtracking, changed settings, and changed inputs cannot reuse an incompatible signature. An interrupted or resource-limited pass is not recorded as a completed unchanged result.

Adaptive scheduling observes elapsed cost and reductions. Optional propagators can wait up to sixteen calls before a retry; expensive unproductive explanation probes can wait up to thirty-two failed frames. A useful pass resets its cooldown. Skipped checks produce no deduction, and explanation proof passes bypass scheduling shortcuts. Required seam consistency, tile ownership, and independent final validation still determine whether a candidate is accepted.

### Biconnected loops and regional dynamic programming

The biconnected filter uses the optimistic graph of possible segment connections and occupied seam endpoints. A single closed Gold cycle must lie within one cyclic biconnected block. Each orientation must fit all of its own segment and seam edges within an eligible block, and every cell must have a candidate for a common block. Unused optional endpoints do not need to be in that block. Drawing crossings are still not graph junctions.

The `separator_regions` filter extends regional search beyond the earlier closed-inventory 2–4-cell queries. It processes a connected region in frontier order and merges histories only when their used physical pieces, boundary masks, exposed-endpoint pairings, and closed-cycle counts agree. Supported placements can then be unioned without losing the distinctions needed for the remaining local search. Adjacent outside cells must each jointly support all incident region seams with an unused physical piece; compatibility among different outside cells remains a relaxation.

Limits are eight region cells, 2,048 states per layer, 20,000 transitions per query, and two automatically chosen queries per propagation pass. Automatic selection requires at most nine orientation values per participating cell. The cache holds at most 64 completed queries and 8,192 final summaries. Its exact key includes region domains and all adjacent outside-cell domains. Cancellation or an exceeded budget returns **unknown** and adds no completed negative cache entry. These in-memory regional summaries do not certify completion of the rest of the puzzle and do not survive a process restart.

### Shared, persistent Gold-loop knowledge

DFS, parallel DFS, and hybrid use `runtime/knowledge-gold.json` when `persistent_learning` is enabled. The file is bounded to 256 records and 262,144 bytes (256 KiB). Every record is a conjunction of legal placements proving a closed Gold component that contains fewer than the puzzle's total number of segments. Such a component cannot later connect to the remaining strands, so its conjunction is impossible in every Gold completion of the same original problem.

Imports verify puzzle, original-domain, and target bindings, then reconstruct the claimed component directly from original segment endpoints and seam incidence. Generic conflicts learned under a restricted job are never exported; their scope may depend on that job. A checksum detects damage but does not substitute for witness validation. Rejected saved knowledge produces a warning and leaves the saved search stack or queue authoritative. Removing this optional knowledge file loses reusable deductions, not committed branch coverage.

Each parallel worker retains at most two job-local learning states. Only immutable support tables with matching input, target, and symmetry bindings can be reused across jobs; mutable native state, generic clauses, domains, and pruning caches stay job-local. Up to four consecutive slices favor a warm job before waiting alternative work is selected. The coordinator independently validates shared loop records and checkpoint ownership. Worker crashes preserve the last committed queue state and explicitly report possible replay after that checkpoint.


### Optional deduction reasons and SAT help

The advanced controls `reason_tracking`, `sat_assist`, `hierarchical_regions`, `value_ordering`, and `shared_conflicts` default to false. They apply to DFS, parallel DFS, and hybrid exact search. The dashboard exposes them as experimental checkboxes; command-line users can pass `--search-options '{"reason_tracking":true,"sat_assist":true,"hierarchical_regions":true}'` (adapt quoting to the shell).

Reason tracking resolves recorded domain dependencies to active placement decisions for conflict learning. Seam and tile-ownership rules have direct antecedents; global/native batches use conservative antecedents. Resume and look-ahead rebuild conservative reasons when necessary. Generic clauses never leave their exact job. The graph is not saved to disk; coverage remains in the ordinary stack.

SAT assistance incrementally builds a private Glucose model, then reuses it across worker slices. Only a completed UNSAT assumption core can exclude a branch; SAT is not a Gold solution. Each scheduled construction step has a 5 ms budget and each query a 10 ms/256-conflict budget, with cooperative Stop. Missing optional python-sat, partial construction, limits and interruptions cannot prove impossibility. The helper caps model size at 160 cells, 80,000 placements and one million clauses and caches 64 conclusive assumption queries. It can consume extra CPU and memory per warm job. Restarting the process rebuilds its model without losing committed DFS coverage.

Hierarchical regions combine exact child summaries in a balanced separator tree, up to 16 cells. Joined summaries retain used pieces, boundary masks, endpoint pairings and closed-loop counts. Joins are indexed by seam masks and physical inventory. Budgets are 4,096 states per table, 20,000 created states and 200,000 transitions, one automatically selected query per pass; candidate cells have at most six orientations. An interrupted query makes no deduction. See [optimization evidence](OPTIMIZATIONS.md) for measured behavior and the larger synthetic stress tests.

For ordinary use, keep the default checkboxes and Automatic CPU core. The added compiled Hall-matching and cyclic-block helpers accelerate existing deductions. Python reference mode remains available for comparison. Bounded tests did not establish that enabling all experimental checks speeds up the full puzzle.


The `value_ordering` experiment prefers placements with more jointly compatible neighboring choices, retaining GPU hints first and seeded tie-breaking. It changes only newly expanded frames; previously saved sibling lists and cursors remain intact. Scoring is bounded to domains with at most 48 orientations.

The `shared_conflicts` experiment offers short local failures to a separate SAT verifier built from the original puzzle domains, physical-piece inventory and seam constraints. A worker, coordinator or disk reader independently proves each received claim before applying or republishing it. Job-specific restrictions and symmetry choices are never assumed by this verifier. Its bounded verified cache is saved separately as `conflicts-gold.json`; incomplete or interrupted proofs are not saved as exclusions. Proper-loop witnesses still use their separate existing store. Neither file is a list of every board visited.

Intermediate regional tables now survive overlapping parent queries within a running job. Their keys include physical inventory restrictions, boundary masks, and Gold strand connectivity; partial tables are not reusable. Cache eviction affects speed only. Native seam propagation reuses its allocated workspace and synchronizes changed rows, including widening after backtracking.
