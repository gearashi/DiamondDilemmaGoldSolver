# Solver improvements and practical settings

Choose **DFS with constraint pruning**, set **Run duration: Unlimited**, and resume your saved DFS search. This preserves completed branch coverage and uses the new propagation and pruning automatically. Only a fully validated, single-loop Gold solution stops it successfully. CPU search does not use GPU replicas.

To try GPU guidance, choose **Experimental GPU sampling + repair + DFS**, **Automatic**, and **4,096 replicas** as a starting point. It has separate saved state. A fresh hybrid search uses your saved best board and GPU candidates, repairs local patches on the CPU, then starts exact DFS. Resuming an existing hybrid checkpoint continues its DFS directly. More replicas do not establish a faster solve.

## New exact-search improvements (round 5)

Six more features now apply to DFS, Parallel CPU DFS, and the exact phase of hybrid search. Their switches are under **Exact-search optimizations**; the defaults enable them. They strengthen or schedule exact search without treating a high-scoring guessed board as a fixed placement.

| Addition | Implemented scope and limit |
| --- | --- |
| Smaller conflicts and backjumping | Reuse an applicable proved conflict immediately, or try removing assumptions with independent contradiction rechecks. At most four probes share a cooperative 10 ms budget. A proved conflict can discard irrelevant intervening branches while retaining the responsible ancestor's pending sibling choices. |
| Dependency-based propagation | Physical-piece matching watches tile availability rather than rotations. Cluster checks watch relevant piece placements; inventory checks also watch boundary orientations; parity and connectivity watch their endpoint categories. Completed unchanged signatures are cached, so unrelated changes do not rerun those checks. |
| Adaptive check scheduling | Measure optional-check cost and whether it removes choices. Repeated expensive passes with no reduction receive bounded cooldowns, then retry. Unproductive conflict minimization also backs off. Skipping or interrupting a check never proves a branch impossible. |
| Biconnected strand filtering | Require every segment and occupied seam of a candidate to fit a cyclic block that can also host a candidate from every other cell. This detects articulation bottlenecks left by ordinary connectivity. Optional ports are not all required to participate. |
| Larger regional summaries | Solve bounded regions by frontier dynamic programming. Summaries retain used physical pieces, exact boundary masks, exposed-strand pairings, and completed-cycle counts; only equivalent histories merge. Unlike the earlier 2–4-cell cache, the region can draw from a larger possible piece inventory. |
| Persistent, shared loop exclusions | Independently reconstruct a closed proper Gold component before saving or sharing its placement conjunction. Reuse these verified exclusions across CPU jobs and restarts. Generic branch conflicts remain private to their originating job. |

Regional dynamic programming handles at most **8 cells**, **2,048 states per layer**, and **20,000 attempted transitions per query**. Automatic selection considers cells with at most nine orientations and runs at most two regional queries per pass. The cache holds at most 64 queries and 8,192 frontier summaries. A resource limit or cancellation means **unknown**: it does not publish a completed cache result or remove candidates. Boundary support is optimistic about outside-to-outside constraints and outside strand pairing; a locally feasible region is not a global Gold certificate. Completed small-region results remain in memory, not an SSD ledger of all board arrangements.

The shared file is `runtime/knowledge-gold.json`, bounded to **256 records and 256 KiB**. Imported records are checked against the puzzle, original domains, and Gold target, then their claimed closed components are reconstructed from the original segments and seams. A matching checksum alone is insufficient. Invalid saved knowledge is ignored with a status warning, while the authoritative search checkpoint remains usable. Disabling this optimization stops its import/export without deleting saved search coverage.

Parallel workers now keep at most two job-local learning caches each and separately reuse bound immutable seam-support tables. Mutable domains, generic conflicts, native state, and pruning state are not shared between jobs. A worker can favor the same warm job for four consecutive slices; it then selects waiting alternative work. Process failure still retains the last committed job checkpoint and can replay its uncommitted tail.

The compiled seam engine also remembers a previously found support. It rechecks that support against the static compatibility relation and the current neighbor domain before reuse; an outdated support triggers a fresh search. This remains safe when domains widen or a search backtracks. Results were mixed in five alternating trials per variant, with twenty cycles in the warm runs: a sequence of five real saved frames changed from **8.29 to 6.68 ms warm** (about 19% less time), but **7.53 to 7.78 ms cold**. A planted 160-cell workload changed from **7.04 to 7.08 ms warm** and **7.14 to 9.32 ms cold**. These are bounded propagation timings; they do not establish an end-to-end search improvement.

Finite checks demonstrate the new operations, rather than a universal speedup:

- A controlled four-tile search retained exactly the same **49 nodes and nine rejected leaves**, while dependency scheduling reduced physical-piece matching checks from **25 to one**.
- An eight-cell open-inventory ring fixture represented **362,880 local completions with nine final summaries**, creating 511 DP states. This demonstrates merging of equivalent regional histories; it is not a count of exhausted real-puzzle boards.
- Independent tiny-puzzle enumeration, actual-tile backjump/Stop/resume checks, optional-check cancellation, projected-domain changes, and validation of imported loop witnesses are covered by focused tests. Evidence and bounds are under `audit/optimization-round5/`.

Final local Windows validation for round 5: **490 CPU tests ran: 467 passed and 23 skipped**, with no failures. Three native CUDA sampling checks also passed. Isolated app runs exercised compiled DFS, a two-worker parallel search resumed with one worker, and CUDA sampling followed by compiled DFS. A separate parallel Stop check completed in **0.30 seconds** and its saved queue validated without advancing it. Production puzzle data and prior checkpoints were preserved. These are bounded tests on this host, not new AMD or Apple hardware coverage.

These changes have not solved the actual Gold puzzle or established a time to completion. More pruning can cost more per node; leave adaptive scheduling enabled initially and compare bounded runs before interpreting a node-rate change as an improvement.

## Previous exact-search improvements (round 4)

All seven additions are available in DFS, and the exact phase of hybrid search. **Parallel CPU DFS** runs the same deductions in separate processes. Expand **Exact-search optimizations** to inspect or disable each new deduction; **Automatic** selects the verified compiled propagation library when installed, otherwise Python.

| Addition | Implemented scope |
| --- | --- |
| Failure-guided branching | Break variable-ordering ties using observed contradictory decisions; retain every pending sibling. |
| Singleton look-ahead | Trial at most three candidates of a 2-4-value cell, with a cooperative 20 ms budget. Remove a candidate only after a proven contradiction. |
| Gold-strand connectivity | Track possible individual endpoint connections, including optional ports, to reject orientations that cannot belong to one complete loop. |
| Correlated seam cuts | Check joint orientation choices across cuts of at most three live seams, rather than treating all seam choices as independent. |
| Entire regional subproblem cache | Fully enumerate eligible connected 2-4-cell regions with a closed physical-piece inventory. Cache the complete local support union and completion count, including an exact zero count when proved. |
| Parallel exact CPU search | Divide unfinished DFS branches into disjoint jobs with a durable queue. Resume with a different worker count without losing committed branches. |
| Compiled CPU propagation | Execute exact AC-3 support filtering in a C99 library with bounded batches and the same Python fallback. |

The regional cache key contains the cells, all regional domains, physical-piece inventory, and every adjacent outside-cell domain. A changed boundary or inventory cannot reuse an incompatible result. The local query enforces all internal seams, distinct pieces, and joint boundary compatibility per outside cell. It does not assert that the remainder of the board forms a Gold loop. Only fully completed local queries enter the cache; cancellation and budget limits never mean impossibility. Limits are 128 cached regions, 4,096 enumeration nodes per query, four new enumerations and sixteen regional checks per propagation call. This is caching an entire small regional subproblem, not caching arbitrary full faces or completed whole boards. No eligible closed-inventory region occurred in the five real saved frames tested; a controlled fixture verifies complete enumeration and a cache hit.

Caches survive slices within a running DFS worker, but are rebuilt after restarting the process. Cumulative exact coverage is retained separately in the DFS stack or parallel queue checkpoint. Graceful Stop commits worker progress. A killed or crashed worker may replay work since its last committed checkpoint, which is reported; uncommitted work cannot safely be declared exhausted.

For this computer, start with **DFS with constraint pruning**, **Automatic** CPU core, and all new deductions enabled. Use **Parallel CPU DFS**, initially **2 workers**, to try concurrent exact branches while leaving CPU capacity for the dashboard. Raising workers can increase memory and setup costs; more workers is not established to shorten the real puzzle's solution time. The first parallel resume can import the existing DFS checkpoint into its own saved queue. The original DFS checkpoint stays intact; subsequently choose the same parallel mode to continue that queue. Running both checkpoints separately can revisit their shared starting coverage.

The compiled backend was measured with extra heuristics disabled to isolate the engine. Three trials per variant resumed the same real checkpoint and reached identical pending stacks. Median solve-call time was **0.555 seconds for the previous build, 0.528 seconds for current Python, and 0.292 seconds for compiled propagation (1.90x versus the previous build)**. A planted 40-tile workload improved 1.47x; a short 160-tile root workload had no gain because setup dominated. These are bounded tests, not a full-puzzle solve-time estimate. The installed native library is built locally from the checked-in source; source/binary hashes and compiler details are recorded in `.native/build.json`.

Final Windows validation: **428 CPU tests ran: 405 passed, 23 optional/platform tests skipped**. Subsequent backend-label and stopped-state display fixes passed all **15 parallel tests** and **36 dashboard tests**; three native CUDA sampler/integration tests also passed. Real-data app checks resumed a copied DFS checkpoint, imported it into two-worker parallel search, resumed that queue with one worker, and ran a 128-replica CUDA hybrid through compiled exact DFS. All saved valid time-limit checkpoints. A separate two-worker Stop check saved a valid queue in **0.29 seconds** after the request. The production puzzle, best board, and search checkpoints were not used as writable test outputs.

Detailed scripts, initial failures, final results, source hashes, and bounds are under `audit/optimization-round4/`. Linux/macOS compilation is included in the CI matrix but was not executed locally in this Windows verification. No actual Gold solution or impossibility proof was produced by these checks.

## Implemented

| Improvement | Purpose and correctness boundary |
| --- | --- |
| Grouped seam supports and sparse fallback | Perform the same arc-consistency check with fewer Python operations; preserve tile inequality. |
| Deduplicated propagation queue | Avoid processing a pending seam repeatedly before its next turn. |
| Reused immutable support tables | Avoid rebuilding equivalent tables in every search slice; bind the cache to the effective input domains. |
| Physical-tile bitset extraction | Gather tile IDs from three orientation bits efficiently without changing the matching problem. |
| Global tile matching | Remove choices that cannot participate in any one-piece-per-cell assignment. |
| Forced neighbor clusters | Retain every legal embedding of proven neighbor relationships; the actual input contains seven forced bonds. |
| Face boundary inventories | Reject a fixed group of pieces when its unpaired edge masks cannot fit the region boundary. |
| Mesh parity | Propagate exact mod-two constraints, including the odd total segment count and the two apices. |
| Restricted neighborhood matching | Reject a center orientation whose neighboring cells cannot use distinct compatible physical pieces. |
| Ten-way proper-rotation symmetry | On fresh eligible searches, keep one macroface representative of each whole-board rotation orbit. All legacy checkpoint branches remain valid. |
| Bounded learned conflicts | Reuse only proven contradictory placement combinations; never learn impossibility from a timeout. |
| Early closed-loop rejection | Reject a closed gold component that prevents the required single complete loop. |
| Diverse GPU candidate archive | Keep several independently scored, distinct candidates for repair and ordering. |
| Adaptive neighborhood repair | Vary freed patch sizes, retain piece inventory, try alternate seeds, and add validated Gold component exclusions. |
| Conclusive local-query cache | Skip identical repair queries only after a conclusive local result; this does not establish global exhaustion. |
| Independent final validation | Require tile uniqueness, every matching seam, and one connected closed loop before saving a Gold certificate. |

The dashboard exposes pruning counters, rotation-symmetry status, and repair progress. These counts are not unique full-board coverage or a completion percentage. Heuristic results never become fixed assumptions in exact DFS.

## Previous optimization round (3)

This round compares against the already improved version installed immediately before it. Improvements remain specific to the tested workloads; the CPU and GPU ratios must not be multiplied into an estimated Gold completion time.

| Additional change | What it avoids or proves |
| --- | --- |
| Cached physical-tile projections | Repeatedly scanning all three rotations when checking tile ownership. |
| Dirty seam scheduling | Revisiting every seam when a new child changes one cell; interrupted/resumed frames rebuild safely. |
| Bounded seam-filter memoization | Repeating exactly the same two-domain support calculation. |
| Revalidated matching and Hall-result reuse | Rebuilding an unchanged physical-tile matching graph; cached results bind the entire graph. |
| Bitset strongly connected components | Repeated Python adjacency traversal in matching propagation. |
| Cached forced-cluster and region results | Rechecking unchanged cluster placements or known inventory consequences. |
| Gold connectivity and bridge cuts | A disconnected possible-seam graph cannot hold one loop; a bridge must have a positive even endpoint count. |
| Precomputed segment ports and seam joins | Reconstructing point dictionaries for every partial-loop check. |
| Complete mask-reversal lookup | Eleven Python bit operations/conversions on every endpoint-mask reversal. |
| Symmetry in all native models | SAT, CP-SAT and CP also retain one proper-rotation representative when the unrestricted input permits it; native resumes already rebuild their models. |
| Fused GPU proposal calculation | Duplicate neighbor reads and writes of proposals that will be rejected. |
| Local GPU launch counters | Repeated device-memory reads/writes for the current score, best score and iteration. |
| Packed WebGPU candidate gathering | Thousands of individual buffer-copy commands when reading selected 160-cell boards. |

The native CP regression also exposed and fixed a partial-hint crash. Only variables present in the assignment are now passed to `DecisionBuilderFromAssignment`; its fallback still searches every variable. This matches the [OR-Tools implementation](https://raw.githubusercontent.com/google/or-tools/v9.15/ortools/constraint_solver/search.cc), which queries an assignment value for each supplied variable. Hints remain preferences, never fixed assumptions.

Final Windows checks on 2026-10-12:

- **Combined CPU search:** three trials per revision resumed the same copied real checkpoint. All six reached exactly the same pending stack after eight new nodes. Median solve-call time changed from **0.911 seconds to 0.557 seconds (1.63×)**. Process imports and `SearchProblem` construction are excluded; solver setup is included.
- **Loop checks:** a warm-cache workload of 57 partial/full/mutated boards ran **2.49×** faster with identical component results. Independent graph traversal also checks the detector.
- **Pruning helpers:** selected saved-frame and synthetic workloads ran **2.1–4.2×** faster with identical filtered domains. These overlap CPU work and are not additive speedups.
- **CUDA sampling:** warmed proposal throughput improved **1.32× at 128 replicas**, **1.43× at 4,096**.
- **Vulkan WebGPU sampling:** warmed proposal throughput improved **1.22× at 128 replicas**, **1.16× at 4,096**. Selected-board readback improved from about **20 ms to 1.2–1.4 ms**.
- **Correctness:** 367 CPU tests ran in 39.399 seconds: **344 passed, 23 skipped** for platform or optional hardware conditions. The integrated native sampler then passed **3 CUDA and 3 Vulkan WebGPU checks**. Tests check score drift through 8,192 moves per lane, fixed placements, narrowed domains and counter wrap; separate same-backend comparisons retained identical trajectories through 1,280 moves per lane.
- **App integration:** a three-second isolated copy of the actual DFS search advanced 54 nodes, published connectivity counters, and saved a resumable checkpoint on its time limit. The production search was not advanced.

Detailed sources, bounds, raw trials and hashes are under `audit/optimization-round3/`. Setup and compilation are excluded from the warmed GPU timings. CUDA candidate-readback timing at 4,096 replicas did not improve; the large readback improvement applies to WebGPU. No experiment here solved the actual Gold puzzle or established a best completion-time algorithm.

Further seam-filter crossover thresholds and an inline sparse loop were benchmarked on saved frames and generated dense/sparse domains. Their results were mixed, so the existing policy was retained. Clause indexing was not a significant cost in the measured profiles. No further small change demonstrated a broad benefit in this investigation. Larger architecture changes remain experiments requiring their own correctness and performance evidence.

## Earlier evidence and limits

All experiments use isolated fixtures or in-memory copies of saved state. Production checkpoints and puzzle data are preserved. Detailed local evidence is under `audit/optimization-round2/` and is ignored by Git because it includes input-dependent records.

Previous-round local checks on 2026-10-12 (Windows, i7-4790, RTX 2060 6 GB):

- **Same-work CPU comparison:** three serial trials per version resumed an in-memory copy of the same real checkpoint and stopped after three partial-loop checks. Both versions used the frozen pruning module to isolate equivalent seam-propagation changes. All six runs reached exactly the same pending stack after eight new nodes. Median elapsed time was **2.344 seconds before, 0.860 seconds after (2.73×)**. This is one short partial-search benchmark, not a full-solver or time-to-solution speedup.
- **CPU regression:** 339 tests ran in 35.654 seconds; 317 passed, 22 platform-specific or optional hardware checks skipped. No failures.
- **Native GPU regression:** two checks passed on CUDA and two on Vulkan WebGPU, using 128 replicas on the RTX 2060. This does not establish AMD or Apple hardware coverage.
- **Loop-repair diagnostic:** eight seeded, single-attempt trials on a two-piece test puzzle improved from **0/8** to **8/8** independently verified Gold solutions.
- **Pruning cost control:** the default local competition rule removed 3 and 17 orientation choices on two saved frames, while the expensive unrestricted version was left out of the default. Mesh parity removed two choices on one saved frame in a separate sub-millisecond pass.

Commands, per-run results, input hashes and source hashes are saved in `audit/optimization-round2/dfs-manifest.json`, `gpu/manifest.json`, `pruning/report.json`, and `full-suite.log`. The version-1 evidence manifests record provenance and bounds but do not enforce operating-system resource limits. These are bounded implementation checks, not a Gold solution, an impossibility proof, or proof of the fastest algorithm for this instance.

Two equal-budget 15-second local-repair trials on the saved board both retained **207/240**. The enhanced trial accepted an alternative after changing seeds, but did not improve the global score. Targeted fixtures verify the new repair operations; real-puzzle gains remain unestablished.

## Evaluated without enabling indiscriminately

- Unrestricted neighborhood matching removed some choices but was expensive. The default checks only center domains of at most 24 candidates; the implementation retains broader checks for controlled comparisons.
- Merging identical Gold patterns or identical edge triples gives no reduction on the current 160-piece input: none were found. Physical pieces remain individually identified.
- Raising GPU replicas increases simultaneous heuristic work, but does not strengthen exact pruning. The existing selectable range remains unchanged.
- Saving every heuristic board does not provide a practical exhaustive cursor. Exact DFS stores unfinished decisions and skips entire contradictory subtrees instead.

Larger separator decompositions are now available as the bounded optional experiment described below. Different search portfolios and further connectivity encodings remain possible experiments. No finite investigation can establish that every useful optimization has been found.


## Round 6: compiled Hall and Gold graph helpers

The optional C99 library now also computes exact physical-piece AllDifferent supports and cyclic edge blocks in the optimistic Gold endpoint graph. Hall filtering repairs a checked perfect matching, then uses integer-bitset reachability to retain exactly those edges that belong to some perfect matching. The cyclic-block helper uses iterative Tarjan traversal, preserving parallel edge identity and avoiding recursion limits on the 5,280-port board.

The **Python** CPU option keeps both Python implementations. **Automatic** uses a matching local native build when available and otherwise falls back. **Compiled** requires that build and reports a build error if it is unavailable. Run `python build_native.py` after updating the C source. Hall supports at most 160 cells; the graph helper bounds its input at 5,280 vertices and 262,144 edges. Both helpers poll cancellation during native work. An interrupted result is discarded before it can enter a pruning cache or change a domain.

Profiling five copied real DFS frames identified cyclic-block analysis and Hall matching as worthwhile native targets. Three alternating Python/native trials on those same five frames produced identical filtered orientation domains. Median times for the complete five-frame sequence were:

| Helper | Python, cold result caches | Compiled, cold result caches | Python, repeated sequence | Compiled, repeated sequence |
| --- | ---: | ---: | ---: | ---: |
| Physical-piece Hall filtering | 21.77 ms | 3.42 ms | 2.31 ms | 0.89 ms |
| Gold biconnected filtering | 261.65 ms | 265.22 ms | 152.24 ms | 131.05 ms |

The clearest improvement is Hall filtering: 6.36x for cold result caches and 2.60x for the repeated sequence. Biconnected filtering improved about 14% on repeated use and showed no cold-cache gain. Constructor time was measured separately and excluded from these columns; native construction added approximately 4 ms per five Hall helpers and 13 ms per five graph helpers. These are bounded helper measurements on this Windows computer, not whole-search speedups or Gold completion-time estimates.

The final focused check ran **81 native, pruning and advanced-filter tests, all passing**. Independent oracles enumerate all 1,024 simple graphs on five vertices and all 512 physical-piece matching graphs with three cells. Additional checks cover parallel edges, deep graphs, randomized 4/10/64/160-cell domains, stale matching hints, widening after backtracking, and cancellation inside C. The Python fallback remains covered; Linux/macOS native execution was not run locally for this round. Evidence, source hashes, compiler metadata, copied-frame bounds and raw trials are under `audit/optimization-round6/native/`. No production search or checkpoint was modified by these checks.


## Round 6: hierarchical regional summaries

The optional `hierarchical_regions` control solves selected connected patches of up to **16 cells**, including complete macrofaces and ring patches. It recursively proposes balanced cuts and chooses a split using child frontier width, crossing seams and domain volume. This is a bounded separator heuristic, not a claim of optimal treewidth. Children are solved separately; joins are indexed by shared seam masks and compatible physical-tile inventories instead of testing every pair of assignments.

Each exact summary retains the used physical tiles, all frontier masks including empty edges, the pairing of exposed gold endpoints, and the number of closed cycles. Compatible child history counts multiply. Histories merge only when that complete summary agrees, and their supported placements are unioned. A closed proper Gold component is rejected; edge-only queries retain their cycle state. Adjacent outside cells must individually have an unused compatible placement, but their mutual inventory constraints and outside strand connections remain deliberately optimistic. A completed local result is therefore a necessary-condition filter, not a certificate of global completion.

Automatic selection requires at most six candidate orientations per cell and runs at most one larger query per pass. Each query allows at most 4,096 states in one table, 20,000 created states in total, and 200,000 inventory/summary transition probes. The cache holds at most 32 complete queries and 8,192 final summaries. Cancellation or a work limit publishes no incomplete support table and makes no negative deduction.

The option remains **off by default**: five copied real search frames produced four completed queries, 196 states and no additional removals, taking a median **54.86 ms** for the five-frame sequence over three trials. A planted 16-cell macroface retained its known Gold completion and removed 32 injected alternatives in **12.68 ms** median. This does not establish that these removals are unavailable to other propagators.

Broader uniform-ring stress cases exercise the actual hierarchical composition and inventory indexing. Their domains exceed the automatic six-candidate threshold:

| Local query | Exact histories represented | Created summary states | Summary merges | Median query time |
| --- | ---: | ---: | ---: | ---: |
| Complete 12-cell ring | 479,001,600 (12!) | 3,137 | 38,059 | 694 ms |
| 12 cells with 13 available physical tiles | 6,227,020,800 (13!) | 5,057 | 79,807 | 1.20 s |

The focused check passed **22 hierarchical and sequential-region tests**. Independent tiny enumeration checks completion counts and supported placements for both targets. Other tests distinguish inventories, boundary masks, strand pairings and cycle counts; cover parallel seams and changed outside domains; preserve planted Gold faces; and interrupt partially completed joins. The version-2 bounded computation manifest, exact command, source/input hashes and raw trials are under `audit/optimization-round6/regions/run/`. These observations do not establish an end-to-end speedup, an actual Gold solution, or exhaustion of the puzzle search.


## Deduction reasons and persistent SAT assistance (round 6)

`reason_tracking` records immutable domain facts and their antecedents in a private DFS dependency graph. Seam revisions and owned-tile removals use their direct dependencies; native and global batches conservatively depend on every pre-call domain fact. Resolving a contradiction yields a forbidden conjunction of active placement decisions, usable by the existing backjumping code. Resumed restrictions and look-ahead changes receive conservative explanations. The graph is rebuilt above an 8,192-fact threshold at node entry; this is not a strict peak-memory limit. It is not a full first-UIP lazy-clause-generation solver. Generic learned clauses and reasons stay private to their exact job.

`sat_assist` retains an optional incremental Glucose model of initial domains, tile uniqueness and matching seams across slices within one running job. Model construction is incremental (5 ms per scheduled attempt); queries use placement assumptions and at most 256 conflicts / approximately 10 ms. A completed UNSAT core can prune a branch. SAT only means edge feasibility and never accepts a Gold solution. Partial construction, budget exhaustion and Stop are inconclusive. A 64-entry cache reuses conclusive assumption queries. Models are capped at 160 cells, 80,000 placements and one million clauses. Missing python-sat leaves assistance unavailable without changing exact coverage. This model is rebuilt after process restart; the exact stack remains the persistent coverage record.

Both controls, and `hierarchical_regions`, start **off**. Two alternating 2-second trials from the same copied real checkpoint processed median 22.5 additional nodes with defaults, 18 with reasons, 24 with SAT assistance, 20 with larger regions, and 18.5 with all three. These short runs included setup and do not demonstrate a solve-time benefit. SAT construction was incomplete in this short observation; its small node-count difference is not evidence of stronger pruning. Independent small-board enumeration and interruption/resume checks validate the optional paths.

Use **DFS with constraint pruning**, **Automatic CPU core**, and the default checkboxes first. Parallel CPU DFS with two workers is a reasonable alternative for this four-core machine. **CP-SAT with direct Gold circuit** is an additional experimental search mode, enforcing the single-loop requirement inside its model. Its native search restarts after Stop; choose the same DFS mode when preserving exact branch coverage is the priority. No full Gold solution or exhaustion proof has been found by these bounded checks.


## Round 7: reuse, ordering, explanations and verified sharing

Five additions address distinct costs:

1. **Native workspace reuse.** Seam propagation retains its domain, queue, dirty-row and counter buffers. It compares current rows with an exact synchronization shadow, transfers changed inputs, and reads changed outputs. Widening, siblings and interrupted callbacks invalidate or update that shadow safely. The previous C exports remain available.
2. **Support-based placement order.** The optional `value_ordering` strategy scores at most 48 candidates by the minimum, product and sum of their compatible neighbor counts. Parallel seams to the same neighbor are intersected jointly. Hints keep priority; seeded ranks break ties. This only reorders the complete choice set for a new frame.
3. **Reusable intermediate regional tables.** Completed child tables are indexed independently of their parent query, with supports in sorted child-cell order. Keys retain adjacent outside orientation domains and the relevant physical-piece restrictions of other parent boundary cells. Bounded cache eviction does not lose exact search coverage.
4. **More precise explanations.** Native seam revisions publish ordered events; learned unit clauses cite their forcing cells. Hall removals may use independently checked tight cell/piece subsets, and regional deductions cite their query dependencies. These facts feed existing conflict learning and backjumping. Other rules retain conservative explanations; this remains a bounded dependency graph, not a full first-UIP solver.
5. **Reverified shared conflicts.** The optional `shared_conflicts` mode proves candidate clauses against an original-domain SAT model without job prefixes or symmetry assumptions. Only completed UNSAT cores become reusable records. Each receiving process and the persistence consumer reverify transported claims. Limits are 128 verified records, 128 pending claims, 32 placements per record and 128 KiB serialized output. Unknown results and cancellation never prove impossibility. A valid empty core means the original edge model itself was proved unsatisfiable.

Three additional changes remove smaller repeated costs: an UNSAT core also answers stronger assumption queries without another native SAT call; batch explanation outputs share an immutable antecedent tuple; and regional lookup uses a cheap node/domain index before constructing projected outside constraints. Subsumption, limits and exact problem bindings apply throughout.

The native workspace is used automatically when the current compiled core is available. Placement ordering and shared conflicts join the existing optional experiments and start off. Larger-region caching and precise explanations are active when their corresponding existing experimental controls are enabled. These experiments have additional cost, so enabling every box is not a recommendation for the fastest real-puzzle search.


Round 7 measurements on the same Windows/i7-4790/RTX 2060 computer:

- Five alternating trials of saved-frame, fixed-point and sibling/restoration seam sequences took **21-28% less time** in the workspace helper, with identical domains and status. These are helper timings, excluding construction.
- Regional child reuse reduced medians from **745.6 to 696.4 ms** on four overlapping 7-10-cell queries, **45.6 to 42.1 ms** on changing planted 16-cell faces, and **40.1 to 38.0 ms** on five static real frames. Counts, summaries and support sets matched both the frozen baseline and the uncached implementation. An initial eager-key implementation regressed the real-frame case; the retained lazy index removed that overhead in this bounded comparison.
- Three alternating complete-solver trials stopped after the same ten new nodes and retained identical pending stacks. Median time was **0.886 seconds before and 0.920 seconds after**. This short experiment does **not** demonstrate an overall speedup; observed timings varied substantially.
- Two alternating 2-second copied-checkpoint trials processed median **23.5 nodes with defaults**, **26 with ordering**, **21 with sharing**, **19 with reasons**, **21 with regions**, and **19 with all experiments**. Ordering can change traversal, and SAT verification was still preparing in these short runs. Node counts are not search coverage or a completion-time forecast.
- The integrated CPU suite ran **599 tests: 576 passed, 23 skipped**, with no failures. Three native CUDA checks also passed. Independent tests cover all 512 three-cell inventory graphs, raw-mask seam events, saved sibling cursors, stale cache keys, missing SAT dependencies, forged transported conflicts, and cancellation during proofs.

Local reproducible scripts, input/source hashes and bounded component manifests are under `audit/optimization-round7/`. The aggregate manifest uses the historical version-1 provenance format; linked regional and shared-conflict runs use version 2. These checks do not establish a full Gold solution, a proof of exhaustion, a universal correctness proof, or validation on macOS/AMD hardware.

The next round investigated repeated construction of immutable Pruner tables across DFS slices. A current Pruner object cannot be reused directly because it retains scheduler and cancellation closures; the implementation below separates immutable construction tables from fresh per-call state.


## Round 8: measured construction and graph costs

Each DFS slice now creates fresh mutable pruning state while reusing completed immutable puzzle tables. Exact bindings cover masks, topology, regions and parity inputs; narrowed job domains are not part of these puzzle-only deductions. Completed neighborhood support profiles join the immutable record only after construction finishes. A cancelled partial build is discarded. Cancellation callbacks, scheduler wrappers, matching hints and per-pass caches remain private to the current call.

Native cyclic-block checks accept a bounded immutable signed-32-bit edge buffer. The caller uses range-checked packing, the wrapper owns an aligned copy, and C checks every endpoint before indexing. The original validated tuple API and Python fallback remain available. No graph deduction changes.

Proper-loop knowledge imports still validate the full payload, integrity, target and original-domain binding. Exact records currently retained in the receiving store may reuse that store's earlier independent proof. Novel or evicted records are proved again. The complete incoming batch must validate before records or counters are committed; checksums never establish a loop proof.

Bounded measurements on the same Windows/i7-4790/RTX 2060 computer:

| Workload | Before | After | Scope |
| --- | ---: | ---: | --- |
| Real 160-cell Pruner construction | 27.37 ms | 2.89 ms warm | Seven alternating trials; three constructors per mode. Cold current median was 23.68 ms; planted-fixture cold timing was mixed. |
| Five real frames, cold cyclic-block checks | 346.37 ms | 199.80 ms | Three alternating trials; constructor setup excluded. |
| Five real frames, warm cyclic-block checks | 118.63 ms | 61.39 ms | Every graph edge and full filtered domain matched. |
| Four resumed DFS slices | 3.020 s | 2.188 s | Three alternating sequences per revision; all retained the same 31 new nodes and four pending stacks. |
| Forty repeated imports of one actual two-cell loop witness | 42.33 ms | 40.27 ms | Includes validation and snapshots; redundant graph proofs fell from 40 to zero. |

The DFS observation is **27.5% less elapsed time for this identical short workload**, not a Gold completion-time forecast. One of six hybrid exact-stage sequences differed under clock-limited optional probes, so its timing comparison is inconclusive. A synthetic forty-record knowledge workload improved repeated-import median from 234.13 to 123.07 ms; its cold-import timing was noisy and slower, so no cold benefit is claimed.

A sparse breadth-first replacement for the strand union-find was also implemented in an audit-only discriminator. Its partitions and deductions agreed, but it was 14% slower cold and 27% slower warm on the five-frame sample, including adjacency construction. The solver retains union-find. Port-to-code bitset mapping is deferred because only 10 of 800 sampled cells used that fallback. A new native strand-component API is deferred until a complete-search profile demonstrates enough remaining benefit to justify it. These decisions are specific to tested workloads, not claims that further optimization is impossible.

Commands, source snapshots, input hashes, exact results and route decisions are retained locally under `audit/optimization-round8/` and `audit/optimization-program/`. The source distribution excludes local audit inputs and private search data. Component records include both version-2 bounded runner manifests and explicitly labeled legacy version-1 provenance. Exact integer deductions are checked against independent small enumerations and frozen implementations; bounded tests cannot prove correctness for every possible puzzle or hardware platform.


The runner also avoids enqueueing a terminal checkpoint that its final consumer will immediately save again. Continuing slices still publish checkpoints; Stop always retains the final authoritative flush. A Stop or time-limit race during enqueue can intentionally retain a duplicate write. This removes ordinary redundant final I/O, not search work: a copied 528 KB checkpoint took a median 7.8 ms per atomic write in the bounded diagnostic.

A further worker-wide immutable-table pool was measured but deferred. With two resident jobs and four-slice affinity, the existing cache needed two builds and fourteen reuses over sixteen slices, with no eviction. A hypothetical pool saved about 21 ms against twelve seconds of nominal slice time. Forced completion churn saved about 39 ms; those construction-only observations do not establish whole-search speedups. The extra ownership lifecycle is not justified by this limited benefit. Revisit if real cache-eviction and table-build counters show materially greater churn.

Release checks now run the sampler's finite WGSL tests through software Vulkan in CI as well as the existing GPU DFS checks. Software adapters are enabled only through an explicit private test argument; the app's hardware requirement and supported Metal/Vulkan/DX12 backends are unchanged.
