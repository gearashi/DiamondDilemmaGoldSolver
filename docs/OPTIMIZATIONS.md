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


## Round 9: four additional search engines and policies

Four independently selectable options are available in DFS, parallel DFS and the exact stage of hybrid search. They initially remain off so an existing saved run keeps its established policy. The dashboard exposes each option and its counters.

| Option | Implemented behavior | Exactness and resource boundary |
| --- | --- | --- |
| Gold subtree CP-SAT | Calls the existing direct single-loop circuit model on a narrowed DFS domain state. A validated solution finishes the search; a conclusive infeasibility result rejects that submitted subtree. | At most 24 open cells, 1,024 placement choices and four attempts per slice; 0.5-second cooperative construction-and-solve budget per call. The first Python dependency import is not preemptible. Exact infeasibility cache only; timeout and model limits leave the branch pending. |
| Connection-pattern branching | Groups one cell's possible orientations by a real seam endpoint pattern, then searches every group. | The nonempty groups partition the complete current domain. Checkpoints retain every pending group and can resume after disabling this option. Parallel splitting uses the same groups. Old releases cannot consume this new grouped checkpoint representation. |
| First-UIP lazy clause generation | Builds literal-level implications and resolves a conflict to one highest-level implication point. Learned nogoods can contain intermediate finite-domain mask facts, beyond complete placement equalities. Unit clauses remove incompatible masks. | Sound rule antecedents are required; global rules conservatively retain all input domain facts. Stores bind the exact effective root and target. Default bounds are 8,192 explanation facts, 256 clauses and 4,096 retained literals. Exhausting a budget loses learning only. Clauses and proof graphs are process-local; ordinary DFS sibling cursors retain coverage. |
| Compiled control loop and reversible trail | A separate C99 library controls branch iteration, domain rollback, node consumption and pending frames. It calls the shared strong propagators and branching policy through Python callbacks. | Disjoint mask choices, bounded memory, cooperative polling, and exact checkpoint conversion. Depth/trail capacity falls back to Python with the remaining frontier intact. Independent final validation remains mandatory. |

`python build_native.py` now builds both the propagation library and the search-control library. The new control loop uses **Automatic** or **Compiled CPU**; **Python reference** selects Python control. A missing optional library is reported as a fallback in Automatic mode.

Finite independent checks include 5,265 assignments across 33 LCG fixtures, every generated clause checked against the valid tiny solutions, all 16 combinations of the four new controls, disjoint checkpoint partitioning, malformed partitions, Python/native cross-resume, fact and native memory caps, and Stop after a committed solution. These are bounded implementation checks, not a proof of correctness for every puzzle.

Component evidence is under `audit/optimization-round9/`. On planted residual problems, a two-second helper budget solved all three tested 160-cell cases. The actual saved frame has 122 open cells and 23,538 placements and is therefore not sent to the oracle. The compiled loop's inexpensive-callback microbenchmarks were slower than Python because conversion and callback overhead dominated. Neither result establishes a full-puzzle speedup.


Three alternating copied-checkpoint trials compared v1.1.0, current defaults, each new option, and all four together. Two bounded slices per trial stopped after three partial-loop checks each. Baseline, defaults, the gated oracle, and LCG retained exactly the same 16 additional nodes and pending stacks. Median solver times were **2.38, 2.79, 2.47 and 3.33 seconds**, respectively. Baseline trials ranged from 2.23–3.99 seconds and defaults from 2.21–3.10 seconds, so the overlapping short observations do not establish a reliable default performance change. LCG learned ten clauses but had higher observed overhead. The oracle made no native query because the saved domains exceeded its gate.

Connection branching produced three grouped branches and a different pending partition. The native controller retained a different interruption representation, although independent tiny coverage and cross-resume checks pass. These runs are therefore not used for a same-work speed comparison. Enabling all four also changes the pending partition. The detailed report separates dependency import, input setup, cold/warm solver time and subprocess wall time; all source snapshots, native build hashes and untouched canonical input hashes are retained under `audit/optimization-round9/performance/`.

Keep the existing default controls for normal searching. The four requested experiments are implemented and can be enabled individually, but these bounded measurements do not justify enabling all of them as a faster default. Regional caches and existing default pruning continue to work alongside them.


The final local CPU suite passed **668 tests with 23 skipped** (691 discovered), including compiled-worker backend reporting, all 16 option combinations, missing dependencies, checkpoint corruption, interrupted deductions, and native depth/trail fallback. Tests use independent finite enumerations where a deduction or branch partition can change the set of solutions. This round was tested locally on Windows; the updated cross-platform CI definition is included, but no new hosted-platform run is claimed here.


Six isolated application runs covered default DFS, two-worker parallel search and one-worker resume, plus all-four DFS, parallel DFS and a **128-replica native CUDA hybrid**. Each saved a checkpoint, and every all-four run reported the compiled control loop. A separate Stop check returned in **0.65 seconds** and its saved parallel queue validated without advancing. One five-second default parallel resume spent its budget preparing and committed no new nodes; it preserved its queue. The six protected canonical puzzle/best/checkpoint files remained unchanged.


## Round 10: less work inside the three optional engines

The compiled search loop now propagates matching seams and forced physical-piece inventory before crossing into Python. Rejected branches never allocate a Python domain list or call the strong-rule evaluator. Surviving branches keep the same complete Python checks, and final Gold validation remains independent. A bounded residue table remembers a supporting orientation for each directed arc/value; every reuse checks the current neighbor domain, including after widening or rollback. Interrupted work stays uncommitted. Branches rejected entirely in C currently skip optional clause learning, trading explanation work for fewer callbacks. The original eight-word native statistics export is preserved; additional counters use a separate export.

First-UIP bulk explanations now share immutable pre-rule parent references and expand only reached reasons during conflict analysis. Root consequences need no resolution, and nested facts for one cell retain only their strongest mask. The learned-clause store also maintains a cell-to-clause index: unchanged clauses can skip evaluation, while affected outputs remain pending for the next fixpoint. Repeated inputs, widening, clause eviction and interrupted calls invalidate or update this state explicitly. The learned clauses are still private to their exact root; they do not replace the saved branch stack.

Before a Gold subtree CP-SAT call, forced pieces and seams reduce the domains, occupied physical tiles leave the remaining inventory, and fixed degree-two chains become mandatory connections between unfinished-cell ports. Separate crossing strands and parallel paths remain separate. A fixed proper loop is a contradiction; an all-fixed complete loop still returns the full assignment for independent validation. Immutable rotated segment tables are cached by exact geometry. A completed UNSAT query can also answer a new query only when every new cell domain is a subset of the proved domain, within the same immutable root. Timeouts and unknown results never enter that negative cache.

These are improvements to the existing experimental controls, which continue to start off. Additional dashboard counters expose early native rejections, support reuse, skipped unchanged clauses, contracted endpoints and reused impossible-subtree results. These are computation counters, not unique full-board counts or estimates of how much of the puzzle is exhausted.


Five alternating synthetic trials retained identical native ordered leaves and learned-clause outputs. Native prefiltering reduced the median from **1.735 to 1.355 seconds** for 30 five-cell problems (13,336 committed nodes and 8,172 leaves). The disabled-fast-path control measured 2.091 seconds; this is a noisy component comparison, not a real Gold throughput claim. For 20 synthetic 160-cell explanation graphs, shared reasons and demand-driven first-UIP reduced the median from **24.906 to 0.886 seconds**. Separate Python allocation tracing measured 1,012,952 versus 165,160 peak bytes; it does not measure total process memory. Event-indexed learned-clause checks reduced a 1,000-call stable-frontier median from **0.201 to 0.081 seconds**, with identical final domains and clauses.

Three alternating full/contracted model trials on planted 160-cell boards measured construction plus solve:

| Residual fixture | Full circuit | Contracted circuit | Model reduction |
| --- | ---: | ---: | --- |
| Ordinary, eight unfinished cells | 152.5 ms | 17.0 ms | Forced seam/inventory preprocessing completes the remaining board. |
| Equal seam masks, eight unfinished cells | 626.2 ms | 171.6 ms | 960 endpoint nodes become 48. |
| Equal seam masks, sixteen unfinished cells | 1,059.0 ms | 554.6 ms | 960 endpoint nodes become 96. |
| All pieces fixed | 214.2 ms | 14.7 ms | Checks the forced complete loop without an optional circuit. |

All 24 model runs produced independently validated Gold witnesses respecting their submitted domains. Another bounded check reused one completed UNSAT result for 100 narrower queries without rebuilding its model. These fixtures contain known planted solutions; they do not establish progress on the unsolved actual puzzle.

Three alternating before/after trials used an immutable copy of the actual saved checkpoint, with identical settings within each comparison. Two slices per run stopped after three completed partial-loop checks per slice. Each comparison retained the same sixteen new nodes and exact pending stack hashes. Import, input setup and process wall times are recorded separately from the following solver medians:

| Settings | Previous round | This round |
| --- | ---: | ---: |
| Existing defaults | 1.467 s | 1.685 s |
| First-UIP learning enabled | 2.126 s | 1.407 s |
| Compiled search loop enabled | 1.629 s | 1.252 s |
| All four experimental controls | 2.275 s | 2.312 s |

The LCG and native options took **33.8% and 23.2% less median solver time** for these short equal-work comparisons. Defaults and the combined configuration did not improve. Timings varied substantially: baseline native trials ranged from 1.195 to 3.841 seconds, for example. The oracle remained gated off on this broad real checkpoint. These measurements do not justify enabling every experimental checkbox or predict completion time; existing defaults remain unchanged.

Independent checks include 108 tiny residual solution-set comparisons, 6,885 LCG assignments, 34 learned clauses tested against all valid fixture assignments, 480 indexed/full-scan propagation comparisons, and 45 random finite native problems. A separate clause-store stress review checked 48,020 domain histories, including 2,165 interrupted calls. Full application regression and installation results are recorded below. Reproducible commands, immutable before/after snapshots, exact input hashes, raw timing trials and peer reviews are retained locally under `audit/optimization-round10/`.


The final CPU suite ran **720 tests: 697 passed, 23 skipped, no failures or errors**. Installed-source hashes are checked against that run. Tests cover all sixteen option combinations, both reason systems with native prefiltering, old statistics ABI buffer bounds, native Stop/rollback/capacity fallback, malformed checkpoint partitions, interrupted clause passes, fixed-loop circuit edge cases, and cache narrowing versus widening. Local CPU evidence is from Windows; updated CI includes the new residual-model tests, but no new hosted-platform run is claimed.


Six isolated app runs passed: DFS, two-worker parallel DFS and one-worker resume, plus experimental DFS, parallel DFS and a **128-replica CUDA hybrid**. All saved checkpoints; experimental runs reported compiled search control. Stop returned in **1.16 seconds**, and its saved parallel queue validated without advancing. The canonical puzzle, best board, DFS/hybrid checkpoints and systematic GPU frontier remained unchanged. The main search was left stopped.


## Round 11: the four requested improvements and follow-up checks

| Improvement | Implemented behavior | Coverage safeguard |
| --- | --- | --- |
| Skip repeated basic propagation | A completed native seam/inventory pass certifies only the immediate Python callback input. | Later deductions resume ordinary propagation; checkpoints never carry the marker. |
| Native event queue | A fixed-capacity deduplicated FIFO and incoming-arc adjacency replace global pending/arc scans. | Reconfiguration and imported frames start with all arcs; interrupted scratch is discarded. |
| Asserting first-UIP backjumps | The learned mask is negated at an eligible pending ancestor; independently excluded deeper pending rectangles are removed. | Unvisited choice unions, selected ancestor masks and earlier siblings are retained. Ineligible jumps fall back to ordinary learning. |
| Reusable residual CP-SAT templates | A private ancestor encoding is cloned and narrowed for a compatible query. | Exact puzzle/root binding, four entries and an 8 MiB representation cap. No shared mutation, incomplete proof, or native search-state reuse. |

Follow-up improvements retain native fixpoint information privately across live branches, choose sparse inventory enumeration only for sufficiently sparse large rows, compact conclusive UNSAT proofs to remove redundant subsets, and check discardable backjump tails once rather than repeatedly. Empty trusted local nogoods immediately consume the remaining local frontier. The first unconditional inventory prototype regressed dense cases and was replaced with density gating.

These changes preserve the established default search policy. The native search loop, first-UIP engine and Gold oracle remain experimental opt-ins. Finite coverage checks and component timings do not establish a real Gold solve-time estimate or a claim that every possible optimization has been exhausted.


Five alternating native component trials on Windows/i7-4790/RTX 2060 compared the frozen round-10 source with the new queue and density-gated inventory logic:

| Bounded workload | Before | After |
| --- | ---: | ---: |
| 30 small DFS problems | 0.9681 s | 0.7492 s |
| 500 sparse 160-cell fixpoints | 60.82 ms | 17.34 ms |
| 500 medium-density fixpoints | 71.87 ms | 41.78 ms |
| 500 dense fixpoints | 258.5 ms | 222.2 ms |

The DFS comparison retained the same 13,336 nodes and 8,172 ordered leaves. Arc checks fell from 148,560 to 50,128; 13,306 nodes used private incremental seeds. These component measurements exclude any claim about completing the real Gold puzzle. A subsequent wrapper-only change removes repeat library loading during transactional backjumps; it has separate failure/cancellation tests and was not part of those native component timings. Local scripts, exact input/source hashes, rejected prototypes and peer-review records are retained under `audit/optimization-round11/`.


The template comparison used three planted 160-cell residuals and four ancestor/sibling queries per sequence, with three alternating trials. At a two-second per-query budget, warm construction medians changed from **6.20 to 1.58 ms**, **18.16 to 1.97 ms**, and **36.91 to 2.32 ms**. Cold template admission added **4.77, 4.75, and 8.18 ms**. Complete four-query medians were mixed: **61.63 to 50.38 ms**, **372.99 to 384.69 ms**, and **1138.29 to 1051.69 ms**. Short equal-mask queries remained UNKNOWN. These results support construction reuse, not a claim that every query or Gold search is faster.

A further single-pass endpoint-support table builder reduced the median actual-board constructor time from **20.87 to 10.36 ms** over seven alternating trials, with exact support equality. A different candidate that scanned unique segment supports for local connectivity was rejected in the measured scope: the five-frame/800-cell sample slowed from **42.64 to 54.24 ms** cold and **26.35 to 33.79 ms** with warm tables. Exact partitions agreed. The existing local union-find remains in use.


The final asserting-backjump discriminator used a physically Gold-infeasible four-tile fixture, seven seeds and five alternating trials. Early loop filters were disabled only in this test to expose independently rejected leaves. Python medians were **10.22 to 10.32 ms** (inconclusive); native medians were **20.31 to 17.48 ms**, with total processed nodes **995 to 800**, 105 useful asserting jumps and 800 skipped duplicate basic passes over 35 current native runs. A prior version wasted work rebuilding state for non-jumping assertions or already-empty tails; the retained version defers ordinary units and requires real pending-work reduction. Full invalid Gold leaves now use the existing certified proper-loop witness for a tighter LCG conflict. No claim is made that these small-fixture gains predict the public puzzle.


Three alternating trials on an isolated copy of the real saved checkpoint compared two bounded slices per configuration. LCG, compiled control, and all four experimental controls retained identical pending-stack hashes and node counts across revisions. Median solver times were **1.501 to 1.341 s**, **1.240 to 0.928 s**, and **1.131 to 0.963 s**, respectively. The all-four current trials ranged from **0.883 to 2.645 s**, showing substantial timing variation. The default comparison is inconclusive: one baseline trial committed an extra node under clock-limited optional probes, so its pending work differed. The public checkpoint still exceeds the oracle trigger gate; these observations do not measure oracle solving on that frame or forecast a Gold solution. Canonical input/checkpoint hashes remained unchanged.


Final local validation discovered **749 CPU tests: 726 passed, 23 skipped, no failures**. Six isolated application runs covered default DFS, two-worker parallel search, one-worker resume, and all-four DFS/parallel/CUDA hybrid search with 128 replicas. Each wrote a resumable checkpoint. A separate parallel Stop check returned in **0.41 seconds** and its saved queue validated without advancing. The six protected canonical puzzle, best-board and checkpoint files remained unchanged. Hosted build results are linked from the release; these local checks alone do not validate physical Apple or AMD GPUs.


### Python 3.15 compatibility follow-up

Regular 64-bit Python 3.15.0 now has a separate DFS/WebGPU setup profile. The full Python 3.12 profile remains available for CUDA, SAT, CP, CP-SAT, and their optional assistance/repair. Setup preserves existing environments, and the dashboard reports installed engine availability before starting a search.

The final local regression runs each discovered 766 tests: Python 3.12 passed 739 with 27 skips; Python 3.15 passed 648 with 118 explicit dependency/platform/hardware skips. Both had no failures. The reduced profile retains DFS, native control, first-UIP, checkpoint, and cancellation coverage; missing native solver packages are not reported as tested. All 27 opted-in Python 3.15 WebGPU/sampling checks passed on the RTX 2060 through Vulkan, including comparison with Python 3.12 CUDA checkpoints. Six additional isolated Python 3.15 application runs covered compiled DFS, parallel resume, and WebGPU hybrid search, preserving the canonical saved searches. These checks establish bounded compatibility, not a Python-version performance advantage or a Gold solution.


## Round 12: graph processing, forced groups and GPU bookkeeping

The optional compiled CPU library now prepares immutable rotated-segment tables once and computes local union forests and vertex-biconnectivity certificates on cache misses. The certificates retain the same ordered edges and used ports as the Python implementation. Whole-board strand connectivity also batches its ordered unions through C, including input packing in the performance checks. Python remains the fallback when the native library is unavailable. Each calculation is bounded, checks cancellation, and publishes a cached result only after completion; no native workspace enters a checkpoint.

Forced-neighbor clusters now transpose their finite embeddings into per-cell bitsets. Intersecting supported embeddings computes the same exact placement union. An unrelated tile remains allowed at a cell only when at least one viable embedding omits that cell. Immutable indices are reused across slices, while the existing eight-entry cache still keys completed results by exact projected domains. This does not merge physical tiles or discard unexplored branches.

First-UIP learned clauses retain bounded cell signatures and mask maps to avoid rebuilding dictionaries during subsumption checks. Stores with fewer than four clauses keep the previous direct scan. Clause meaning, eviction order, root binding and pending search coverage stay unchanged.

The persistent GPU DFS runner avoids refilling when no free lane and queued job can be paired. It still validates ownership after every completed launch and at restore/checkpoint boundaries. CUDA aggregates counters before transferring them to the CPU. CUDA and WGSL candidate checks now test only the occupied third side when the lookup bucket already guarantees the first two occupied sides. Candidate order, random state and checkpoint formats are unchanged.

Seven alternating trials on a copied real checkpoint with 4,096 replicas measured CUDA kernel launches at **1.167 to 0.736 ms** and WebGPU/Vulkan launches at **2.633 to 1.993 ms** on an RTX 2060. Every one of the twelve device fields matched the previous kernels bit for bit at work budgets 1, 7, 64 and 512. A separate 131,072-replica counter test measured **1.564 to 0.204 ms**, including transfer and summation, with exact unsigned-64 overflow parity. These component measurements do not predict time to a Gold solution.

Candidates that did not justify more complexity remain out of the production path: ordinary learned-conflict event indexing regressed small stores and dense changes; staged regional boundary caches gave no consistent benefit; lazy local-certificate edge construction slowed the Python fallback; sparse WebGPU gathering has no measured benefit for the common contiguous live reads. Native frontier snapshots remain a small fraction of the measured work. These are workload-based decisions, not a claim that future optimization is impossible.


Three alternating before/after trials resumed the same immutable copy of the actual saved checkpoint, with two slices stopping after three completed partial-loop checks each. Within every settings group the old and new versions committed the same sixteen nodes and retained identical pending stack hashes. Solver medians (excluding dependency import and input setup) were:

| Settings | v1.2.0 | Round 12 | Reduction |
| --- | ---: | ---: | ---: |
| Existing defaults | 0.923 s | 0.752 s | 18.4% |
| First-UIP enabled | 0.976 s | 0.803 s | 17.7% |
| Compiled search loop enabled | 0.794 s | 0.631 s | 20.5% |
| All four experimental controls | 0.870 s | 0.725 s | 16.6% |

These are short bounded comparisons on one checkpoint. They support retaining the changes but do not establish a universal speedup, rank the algorithms across the full puzzle, or predict when a Gold solution will be found. Complete raw trials, source/native-library hashes, separate process/import/setup times and exact pending-state comparisons are retained in the local round-12 audit records.


The final CPU suites each discovered 793 tests: Python 3.12 passed **764 with 29 skips**, and Python 3.15 passed **673 with 120 skips**, with no failures. Skips explicitly identify unavailable engines, platforms or hardware rather than claiming coverage. Version-2 bounded-run manifests verified the exact source and compiled-library inputs. Ten opted-in Python 3.12 GPU tests passed, and Python 3.15 passed 28 WebGPU/sampling checks with its direct-CUDA test intentionally skipped. Six isolated Python 3.15 application runs saved valid checkpoints and exited cleanly, including two-worker to one-worker resume and WebGPU hybrid search. Canonical puzzle/best/checkpoint hashes remained unchanged. Physical AMD and Apple GPU coverage remains unverified; the local GPU was an RTX 2060.


## Round 13: neighborhood reuse and cheaper graph bookkeeping

Neighborhood pruning retains up to 512 completed local support results per private propagator on boards with at least 32 cells. The exact key includes the center domain, every grouped neighbor domain and immutable seam profiles. Changed rotations or parallel-seam constraints cannot reuse an incompatible result. Incomplete work is never cached, zero-support results remain valid cache entries, and FIFO eviction only causes recomputation. Smaller boards keep the previous per-call cache. No cache is serialized as exhausted search coverage.

Compiled global strand and biconnectivity graphs now collect ordered integer endpoints directly before converting to immutable native buffers. This removes intermediate edge tuples while retaining native range validation, cancellation checks and Python fallback. Biconnectivity uses dense cell/port seam indices and per-cell block rows; missing seams retain a distinct sentinel. Separator dynamic programming computes cut edges using immutable incident-edge bitsets. XOR removes internal seams, preserves separate parallel seams and cancels self-loops. Ascending edge IDs preserve frontier order, tie-breaking and exact local histories.

Seven alternating component trials on the existing Windows/i7-4790/RTX 2060 setup measured:

| Bounded workload | Before | After |
| --- | ---: | ---: |
| Neighborhood filter, 25 contiguous saved frames | 20.70 ms | 17.18 ms |
| Neighborhood filter, 100 repeated frame calls | 79.40 ms | 22.50 ms |
| Native forest graph construction and traversal, five frames repeated ten times | 191.44 ms | 134.80 ms |
| Native certificate graph construction and traversal, same bounds | 212.98 ms | 145.91 ms |
| Complete separator filter, five saved frames | 13.55 ms | 2.81 ms |

Feasible neighborhood workloads without cache reuse incurred 1.3-2.8% helper overhead; the retained cache measured roughly 283-312 KB in recursive Python object sizes, not process RSS. A separate full biconnectivity-filter discriminator measured tuple-key seam lookups versus the selected dense port-block rows at 416.5 to 395.0 ms in Python and 157.4 to 133.1 ms compiled (five saved frames repeated four times). These component gains overlap and must not be added together.

Three alternating whole-search comparisons resumed an immutable copy of the actual checkpoint. Two slices stopped after three completed partial-loop checks each. Defaults, first-UIP and all four experimental controls each preserved their exact pending stack hashes and sixteen new nodes across revisions. Their median solver times were respectively **0.685 to 0.622 s**, **0.768 to 0.682 s**, and **0.687 to 0.633 s**. The native-search-only comparison was inconclusive because one clock-bounded trial followed different pending work; it is not reported as a speedup. Import, input setup and process times are recorded separately. Existing search-policy defaults remain unchanged.

Independent checks cover local-star enumeration, widening, rotations, parallel seams, eviction, cancellation, exact packed graph sequences, non-32-bit array fallback, finite Gold witnesses, and 4,392 small graph/subset/domain ordering comparisons. Raw trials, immutable source/input hashes and peer reviews remain locally under `audit/optimization-round13/`. These bounded results do not establish a Gold solution, a completion-time forecast, or universal exhaustion of future optimization ideas.


A follow-up physical-tile projection cache was rejected. Although the complete inventory callback on a recorded access sequence improved from 67.16 to 40.40 ms, sparse miss-heavy input regressed from 15.40 to 16.28 ms. Nine alternating full runs with identical twelve nodes and ordered frontiers measured 683.14 to 696.28 ms. The simpler existing per-propagation cache remains. This decision uses total search behavior, not the favorable inner measurement alone.


Port-enumeration reuse was also rejected after a follow-up comparison. A broad tuple cache regressed sparse filters; an inline shortcut for exactly all 33 ports improved a compiled filter sample, but complete equal-pending search medians were 0.681 to 0.741 s with defaults and 0.795 to 0.835 s with first-UIP. All-four improved 0.723 to 0.697 s, and the native-only comparison was inconclusive because pending work differed. The release retains the original streaming enumeration. All trials remain recorded; no additional change met the full-workload acceptance gate in this review.

Final source matches the previously completed bounded regression manifests exactly: **802 tests per environment**, with **773 passed / 29 skipped on Python 3.12** and **682 passed / 120 skipped on Python 3.15**, no failures. Eight targeted Python 3.15 WebGPU/sampling hardware checks passed, including checkpoint interchange through the preserved Python 3.12 CUDA environment. Isolated compiled DFS Stop/resume and 128-replica WebGPU hybrid resume saved valid checkpoints. The observed DFS Stop latency was 0.174 s in this run, not a guaranteed maximum. Unavailable engine/platform tests remain explicit skips. Native libraries and canonical puzzle/best/checkpoint files remained unchanged during validation.


## Round 14: complete native strand filtering and regional split costs

The optional C library now performs the complete optimistic strand filter in one prepared workspace: local forest lookup, graph assembly, ordered union, candidate-to-component support mapping, and common-component intersection. The Python boundary still publishes only completed supports and independently validates final solutions. The workspace belongs to one propagator, has bounded storage, and never becomes checkpoint coverage. Unsupported native exports retain the existing Python-orchestrated path. Tiny boards keep that path because native workspace initialization cost more in measured cold controls.

Hierarchical regional joins calculate proposed cut widths with exact incident-edge bitsets instead of repeatedly scanning every puzzle edge. The selected cuts, tie-breaking, completed summaries, physical inventories, endpoint pairings, cycle counts and placement supports remain unchanged. This accelerates the existing optional hierarchical-regions setting; it does not enable it by default or drop any part of a regional table.

Two proposed state-growth split predictors were implemented as audit prototypes and rejected against the faster exact-order control. Their extra estimation generally increased complete regional-query time. A separate learned-conflict event index was tested on genuinely learned stores rather than only synthetic lookups. Even after delaying activation until two unchanged-store passes, a complete planted Gold search with 244 retained clauses and 515 nodes slowed from **0.514 to 0.575 s** across five alternating trials; all five indexed trials were slower. It rebuilt 83 times but used indexed propagation only 16 times. Exact solutions, pending frontiers, clauses and non-timing counters agreed, and 10,273 finite index comparisons passed. The existing ordinary conflict scan remains in production. Prototype subclass overhead is included in these measurements; they do not rule out a different future implementation.


The native filter's seven alternating component trials included input packing, workspace setup and Python application. Five recorded depths repeated three times measured **111.28 to 29.21 ms** with reused helpers and **85.92 to 68.92 ms** with cold helpers. Cold 10-cell and 40-cell controls also improved; 1-, 2- and 4-cell controls motivated the size gate. An alternative cache-hash finalizer was rejected: actual complete-call timings were nearly flat, while dense and sparse misses regressed roughly 2-3%.

A further regional improvement maintains the set of adjacent unselected candidate cells during connected growth. It preserves the original split sequence, including disconnected-region fallback. Seven alternating trials against the incidence-mask implementation measured actual warm queries **31.61 to 30.49 ms**, planted warm queries **51.05 to 44.29 ms**, and miss-heavy warm queries **20.60 to 17.11 ms**. Cold tiny queries regressed **4.62 to 4.82 ms**, and cold ring queries **90.86 to 92.39 ms**. These are complete bounded query sequences, including helper setup, not a promise that every query improves.

The combined three-trial saved-search comparison retained the same sixteen nodes and ordered pending stacks with defaults: **0.610 to 0.537 s**, about **11.9% lower elapsed solver time**. Clock-bounded optional proof work made most other configurations incomparable; those measurements remain recorded rather than being reported as wins. A follow-up regional-enabled comparison with look-ahead/adaptive scheduling disabled still had timed conflict minimization active and showed substantial variation, including a saved-case regression from **1.082 to 1.641 s**. It is retained as contrary evidence, not discarded. An additional isolated regional comparison disables conflict minimization and requires all non-timing counters as well as pending states to agree.


In that isolated follow-up, only `constraint_regions.py` differed; both sides used the same previous native engine, with look-ahead, adaptive scheduling and conflict minimization disabled. All non-timing counters and pending states matched across five alternating trials per case. Median full-search time changed **0.764 to 0.737 s** on the saved case (3.6%) and **0.592 to 0.528 s** on a fresh restricted ancestor subproblem (10.8%). This supports retaining the regional changes within the tested scope; the contrary combined trial and all raw trials remain in the audit record.

The release retains the native strand filter, incidence-mask scoring and incremental regional candidate frontier. The predictive split heuristics, ordinary conflict index and cache-hash variant remain audit-only. Further native biconnectivity fusion is a possible next project: the post-change profile identifies it as material, but it has not been implemented or credited with a speedup here. No exhaustive claim is made about future optimization ideas or the full Gold solution space.


Final local suites discovered **815 tests per environment**: Python 3.12 passed **786 with 29 explicit skips**, and Python 3.15 passed **695 with 120 skips**, with no failures. Both exact-input version-2 manifests validated. Eight opted-in Python 3.15 WebGPU/sampling checks passed on the RTX 2060, including checkpoint comparison with the preserved Python 3.12 CUDA environment. Four isolated application runs validated compiled DFS Stop/resume and 128-replica WebGPU hybrid resume. Observed Stop latency was 0.180 s in this run, not a guaranteed maximum. All six protected canonical puzzle/best/checkpoint hashes remained unchanged. These checks do not establish physical AMD or Apple GPU execution.


## Round 15: native cyclic blocks and GPU readback/selection

The compiled CPU path now performs the complete biconnectivity filter in a private bounded workspace: local certificates, ordered seam assembly, cyclic blocks, candidate support mapping, and common-host intersection. It returns the same ordered supports or first failing cell as the Python path. Exact domain keys bind each completed certificate to immutable piece geometry; interrupted global results are discarded. Boards smaller than ten cells and older native libraries retain the existing path. The new C workspace uses at most about 3.1 MB, plus immutable tables and transient traversal storage; it is never checkpoint data.

Both native connectivity filters now poll during candidate mapping after at most 256 code/endpoint work units, alongside the existing cell, seam, traversal and output checks. This removes excess Python callbacks on cheap candidates and adds checks within unusually long segment lists. Every callback position in successful and failing small filters, and in long-segment fixtures, is tested for cancellation and recovery.

Seven alternating full-filter trials measured five saved depths repeated three times at **99.66 to 38.43 ms** for the new biconnectivity pass, including input/output conversion and applying supports. Cold helpers measured **97.55 to 73.91 ms**. A separate polling comparison measured combined connectivity passes at **52.37 to 41.80 ms** on reused actual states and **126.05 to 115.75 ms** on dense misses. Sparse reused calls regressed **26.35 to 27.15 ms**, while sparse cold calls improved **170.34 to 162.07 ms**. The polling change's complete fixed-work search comparison measured **304.76 to 268.52 ms**, with six of seven paired trials faster and identical pending states and non-timing counters. These overlapping component gains must not be added together.

The WebGPU systematic runner can combine dispatch with its required lane-state read when the actual active pool has at most 4,096 replicas. That read completes the launch, avoiding a separate four-byte fence; ordinary `step()` retains its existing synchronous contract. Ledger changes happen only after a successful read, and a failed read preserves the last trusted checkpoint. Sampling uses threshold partitioning for large pools, then sorts only the selected candidates. Boundary ties retain descending lane-ID order exactly. Small pools retain the original stable sort.

Final-source ordinary WebGPU runner iterations on copied real lanes measured **1.630 to 1.241 ms** at 128 replicas (seven of seven paired wins) and **2.868 to 2.448 ms** at 4,096 (six of seven). At 32,768, an earlier prototype's small apparent gain did not hold: final-source medians regressed **16.31 to 21.06 ms**, with substantial variation and only four of seven paired wins. The runner therefore retains the existing path above 4,096 actual lanes, including after checkpoint resize. The larger-pool regression remains in the audit record. Selecting and gathering the best 32 from 131,072 replicas, including independent CPU scoring and archive/diversity selection, measured **29.137 to 24.004 ms** with CUDA and **27.686 to 23.370 ms** with WebGPU. Hardware measurements used one RTX 2060; they do not establish AMD or Apple hardware performance.

Further prototypes were rejected when they did not improve the whole operation. A conditional subtraction in GPU candidate indexing failed to reproduce its initial gain; reusing CUDA timing events had no net benefit. A density-gated byte-table iterator halved its recorded inner workload but changed the complete saved search from **334.23 to 339.75 ms**; another restricted case improved, but only three of seven paired trials were faster in each case. The original iterator remains. All 480 supplied oriented tiles also have distinct endpoint-pair multisets and side-mask triples, so exact interchangeable-piece or rotation reduction does not apply to this input. This finite scan says nothing about full-board symmetries or the existence of a Gold solution.


Forced-cluster reconstruction, per-rotation lookups, sparse embedding reconstruction, skipping zero projected domains, and an allocation-free Hall-of-three helper were also evaluated. The final two shortcuts failed the strict compiled whole-search gate: a saved prefix measured **356.15 ms** unchanged, **353.85 ms** with zero-domain skipping and **369.37 ms** with the Hall helper; a fresh restricted ancestor measured **230.67, 238.19 and 234.75 ms** respectively. All non-timing counters and pending states agreed. Earlier component measurements used the Python Hall fallback while the staging native library was stale; they are not reported as compiled-engine gains. None of these normal-pruning changes is retained.

The profile still identifies forced-cluster propagation as a possible deeper native implementation project. It requires a separate bounded workspace, complete embedding-support oracle, interruption checks and setup-inclusive measurements. That route remains open; this release is a measured optimization milestone, not a claim that every future improvement has been exhausted.


A further sampling improvement builds fresh membership sets once per collection for batches of at least 32 candidates, converting only domains with at least 64 choices and sharing equal domain contents within that collection. The unchanged raw-mask CPU scorer still checks physical inventory, fixed placements, seam matches and domain violations for every board. The temporary view is discarded after collection; later domain changes are observed. Short batches and narrow domains use ordinary membership. A CPU reversal-table variant was slower and was rejected.

Setup-inclusive collection measurements for this additional change were **26.42 to 17.58 ms** for the actual 160-piece input, **12.42 to 10.75 ms** at the 64-choice boundary, **14.12 to 11.75 ms** for mixed-width domains and **27.88 to 21.63 ms** with fixed placements. Exact metrics and selected archives agreed. Disabled-path controls showed substantial clock variation despite using the same scorer; their differences are not attributed to the optimization. These timings overlap the earlier sampling measurements and must not be summed.


The combined CPU comparison used three alternating trials for each of three cases: the copied saved checkpoint and fresh restricted subproblems from ancestor frames 5 and 15. Two slices each stopped after three partial-loop checks. Fixed-policy defaults, first-UIP, compiled search and hierarchical regions disabled look-ahead, adaptive scheduling and timed conflict minimization. All twelve case/settings combinations retained identical full pending stacks and every non-timing counter. Solver medians improved **8.9–22.0%**. On the saved checkpoint the four medians were **0.705 to 0.639 s**, **0.921 to 0.792 s**, **0.649 to 0.533 s** and **0.680 to 0.616 s**, respectively. Import and input setup were recorded separately. Normal timed defaults were also run, but their internal work differed and those comparisons are inconclusive.

The exact as-run sources, native binaries, inputs and result are archived with a validated version-2 manifest. The later replica-size gate changes only the GPU systematic runner and its tests, outside these CPU search paths. A test-only verifier fix also removes a scheduling assumption: model preparation can consume an advance budget without completing a query. The revised regression keeps real monitor-thread interruption and checks each unknown attempt, while a separate test confirms that preparation alone does not consume a retry. Ancestor cases are independent restricted workloads, not newly exhausted canonical coverage. These measurements do not predict time to a Gold solution or justify changing the default search policy.


Final release-source suites discovered **838 tests per environment**: Python 3.12 passed **807 with 31 explicit skips**, and Python 3.15 passed **716 with 122 skips**, with no failures or errors. Both exact-input version-2 manifests validated. Ten opted-in WebGPU/sampling hardware checks passed on the RTX 2060, including comparison against the preserved Python 3.12 CUDA environment. Four isolated application runs validated compiled DFS Stop/resume and 128-replica WebGPU hybrid resume. The observed Stop latency was **0.108 s** in this run, not a guaranteed maximum. All six protected canonical puzzle/best/checkpoint hashes remained unchanged. Two test-only robustness fixes distinguish completed verifier attempts from preparation time and allow PowerShell to wrap an expected diagnostic without changing its meaning. The initial failed runs remain recorded. These tests do not establish physical AMD or Apple GPU execution.


## Round 16: physical-piece projections and WebGPU checkpoint capture

CPU pruning now gathers dense orientation-domain bits into physical-tile bits with staged integer masks, and reverses those stages when expanding tile inventories. Sparse domains use the existing set-bit strategy with its loop inlined. Short domains and inputs wider than the 160-piece model retain the original lookup-table paths. Both exact-key LRU caches still hold at most 8,192 entries. The stages only change the representation calculation: they retain all physical-piece identities, all three rotations and the same pruning results.

Independent per-tile oracles checked exhaustive small inputs, every single-bit boundary through 1,024 bits, seeded mixed-width cases and fallback paths on Python 3.12 and 3.15. Longer fixed-work comparisons stop after exactly 32 partial-loop checks and require the entire pending stack, result fields and all non-timing counters to agree. Look-ahead, adaptive scheduling and timed conflict minimization are disabled for this comparison. Nine alternating prototype trials measured Python 3.12 saved and restricted-ancestor samples at **1.848 to 1.671 s** and **1.550 to 1.441 s**; Python 3.15 measured **1.722 to 1.667 s** and **1.371 to 1.308 s**. The earlier four-loop samples were mixed and are retained as contrary evidence.

A final-source comparison used five alternating trials, fresh solver setup and equally cleared projection caches. The saved sample measured **1.997 to 1.783 s**, and the restricted ancestor **1.659 to 1.441 s**, with five of five paired wins in each case. Import and input construction are outside these solver timings. These bounded samples support retaining the change; they do not predict solution time or guarantee a gain for every state or interpreter.

WebGPU checkpoints now read each of the six mutable physical buffers once and immediately decode independently owned, frozen logical arrays. The first synchronous read fences prior dispatches; the search thread submits no intervening work. Existing ownership validation still precedes capture, and the file writer only receives a complete snapshot. Counter limb order, logical dtypes, array order, metadata and checkpoint schema are unchanged. This removes repeated maps and intermediate copies while keeping one physical mapping at a time.

Seven alternating final-source tests included four launches, ledger validation, worker creation, serialization, flush and atomic checkpoint replacement:

| Replica pool | Previous complete Stop/save | New complete Stop/save | Paired wins |
| --- | ---: | ---: | ---: |
| 128 | 23.50 ms | 19.75 ms | 7/7 |
| 4,096 | 93.36 ms | 68.00 ms | 7/7 |
| 32,768 | 924.70 ms | 440.01 ms | 7/7 |

These are checkpoint-operation gains on an RTX 2060 using Vulkan, not per-node search gains. The prototype's traced Python allocation peak at 32,768 replicas fell from 106.0 to 74.1 MB; this excludes driver and native staging memory. Final integrated checks covered independent scalar field decoding, signed values, 64-bit counter boundaries, frozen ownership, every map-failure position, actual Stop/resume/resize and CUDA checkpoint interchange. Four bounded manifests record the final-source evidence.

Other implemented audit prototypes were rejected. Compact native forced-cluster tables remained slower even after deferring duplicate Python setup and reusing immutable packed inputs: the final refinement regressed actual complete filtering **11.8%** and fixed-work search **2.3%**. Flat neighborhood cache keys reduced some cache storage but regressed longer full searches **1.8% and 3.4%**. Both retained exact domains, counters and pending states; neither enters the application. All prepared candidates in this round are resolved. Further profiling can identify future work, but no unmeasured optimization is credited as a gain.

Final local CPU suites discovered **847 tests per environment**: Python 3.12 passed **816 with 31 explicit skips**, and Python 3.15 passed **725 with 122 skips**, with no failures or errors. Both bounded-run manifests validated against the exact final implementation. Skips identify unavailable dependencies, platforms or hardware; they are not counted as passes. Final integrated WebGPU hardware and application capture/Stop/resume checks also passed. Physical AMD and Apple GPU performance remains unverified.


## Round 17: measured status reporting and rejected pruning experiments

WebGPU pools of at least 32,768 replicas now reduce the three progress counters on the device. Two bounded compute passes add low/high 32-bit limbs with exact carry and modulo-64-bit overflow, matching the existing NumPy/CUDA counter semantics. Only the engine's complete counter sum uses this path; other array operations and smaller pools keep their existing behavior. Each engine lazily owns its reduction resources and releases them at close. Every-launch branch-state reads, paused/retired accounting, search transitions and checkpoint representation remain unchanged.

Seven alternating final-product trials included complete status construction, JSON serialization and atomic file replacement. On an RTX 2060 through Vulkan, 32,768 replicas measured **6.523 to 2.458 ms**, and 131,072 replicas **16.511 to 2.580 ms**, with seven of seven paired wins at each size. Counter readback at the largest pool falls from **3,145,728 to 24 bytes** per query. The first enabled publication, including shader compilation, took **130.1 ms**; the later 131,072-lane cold publication took **5.04 ms**. Startup cost is separate from the steady measurements. Both small-pool controls retained the old path and allocated no reducer.

Full fixed-work trials retained identical device fields and ordered ownership ledgers. Their median changes were approximately 1.1% and 2.1% faster at the two large pools, but only four and three of seven pairs improved. An unchanged small-pool control varied by 7.9%, and short trials straddled the one-second reporting boundary. **These full-search speed results are inconclusive.** The retained benefit is the complete status operation, not a demonstrated reduction in time to a Gold solution. Physical AMD and Apple GPU performance remains untested. CUDA already reduces these counters on the device and is unchanged.

Two correct pruning prototypes were not retained:

- **Cost/usefulness scheduling:** 18 independent policy and finite-solution tests passed. Twenty of 22 predeclared decision tasks completed in both variants; the other two timed out and remain inconclusive. Across seven alternating trials of all 20 completed tasks, aggregate median time increased **3.71%**. All four substantial tasks that exercised the extra cooldown also slowed down. The existing scheduler remains.
- **Dirty local neighborhoods:** nine independent support/lifecycle tests passed on each of Python 3.12 and 3.15. All 20 fixed-work search runs preserved the pending frontier. The saved case changed **1.904 to 1.907 s**, while a restricted ancestor improved **1.582 to 1.489 s**. Cold, dense-change and planted workloads also showed regressions. This mixed result did not justify adding the bookkeeping to normal pruning.

The rejected prototypes and all bounded-run evidence remain in the local audit directory. Tests on restricted domains do not exhaust the canonical puzzle, and no new Gold solution is claimed.

Final local regression suites discovered **856 tests per environment**: Python 3.12 passed **824 with 32 explicit skips**, and Python 3.15 passed **733 with 123 skips**, with no failures or errors. Eight final-source host checks, nine opted-in host/hardware checks and two actual Stop/resume/resize tests also passed. Source-bound manifests validated; all six canonical puzzle/best/checkpoint hashes stayed unchanged. The solver was left stopped. Skipped platforms and dependencies are not counted as tested.

The first Python 3.12 suite encountered Windows socket abort 10053 in an existing dashboard security test. Its unchanged isolated retry and complete-suite rerun passed; the initial failed report remains preserved. Security checks and their assertions were not changed.


## Round 18: Windows startup and repeated branch validation

Windows launchers now test port availability with a short-lived exclusive loopback bind. On the affected computer, a closed TCP port took about 2.1 seconds to report refusal, exceeding the previous 500 ms probe timeout. The new probe checks the operation the server needs and preserves socket error details. The actual Windows HTTP server also binds exclusively, and launchers still verify the dashboard's installation identity. Unix keeps its existing TCP probe and server reuse behavior. A browser bookmark requires the local service to be running; after restarting Windows, launch `OpenDashboard.cmd` again without reinstalling.

The systematic CUDA and WebGPU runners now retain a private immutable copy of a successfully checked branch-ID array when at least **4,096 jobs are active**. Exact array equality permits reuse of only the uniqueness result. ID bounds, current lane states, completed counts, paused overlap and retired counter shape/type remain checked on every call. In-place changes, reordering or replacement require revalidation. The certificate is never serialized; resume/resize constructs and validates fresh ledgers. Pools with fewer active owners retain the original uniqueness check, even if the allocated replica capacity is large.

Seven alternating trials per case compared complete GPU launch, lane-state read and ledger retirement/validation at the application's default budget of 32 nodes per launch. Every device field and ordered checkpoint field matched after each trial. The frozen actual checkpoint copy retained unchanged ownership during these timed runs; this is the measured regime. Medians on the local RTX 2060:

| Backend | Allocated replicas | Previous iteration | Cached iteration | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 4,096 | 1.091 ms | 0.603 ms | 7/7 |
| CUDA | 32,768 | 9.828 ms | 5.581 ms | 7/7 |
| CUDA | 131,072 | 46.945 ms | 21.956 ms | 7/7 |
| WebGPU (Vulkan) | 4,096 | 1.642 ms | 1.133 ms | 7/7 |
| WebGPU (Vulkan) | 32,768 | 13.472 ms | 8.210 ms | 7/7 |
| WebGPU (Vulkan) | 131,072 | 58.928 ms | 28.400 ms | 7/7 |

At 131,072 replicas, 128,854 jobs were active. These are bounded iteration gains, not predicted times to a Gold solution. CUDA used Python 3.12 and WebGPU used Python 3.15. The cache adds up to about 1 MiB of host storage at the supported maximum. Physical AMD and Apple GPU performance was not measured.

There is a tradeoff when ownership changes frequently: exact comparison and copying add work. A host-only Python 3.15 control retiring/refilling one eighth of the pool each transition at 32,768 lanes regressed **8.980 to 10.315 ms**, with only one of seven paired wins. Other mutation controls were mixed; no general speedup is claimed for them. A sparse pool with only 128 active owners also motivated gating by active jobs rather than capacity. Earlier gate variants and contrary measurements remain in the audit record.

Two CPU prototypes were tested and rejected. Reusing branching topology/cardinalities preserved decisions but had inconsistent paired results and a **2.39% slower aggregate median**. Reusing completed parity roots improved a restricted ancestor workload **5.43%**, but slowed the saved workload **6.82%** and failed the declared retention gate. All 40 fixed-work comparisons retained the complete pending frontier and all non-timing statistics; 24 correctness tests passed on each Python profile. Neither CPU source change enters the app.

Final regression suites discovered **871 tests per environment**: Python 3.12 passed **839 with 32 explicit skips**, and Python 3.15 passed **748 with 123 skips**, with no failures or errors. Actual application checks passed on CUDA/Python 3.12 and WebGPU/Python 3.15: both 128→64→192 and 32,768→2,048→32,768 Stop/resume sequences preserved all device/checkpoint arrays, counters and branch ownership. The large sequence exercised cache hits, and the smaller resumed pool used the uncached path. These checks used private output directories and copied puzzle inputs.


## Round 19: stronger pruning and adaptive work scheduling

The acceptance criterion for this round prioritizes sound elimination of search choices and resource efficiency. A reduction in nodes per second does not disqualify a change that removes more impossible branches. Finite fixtures and bounded original-puzzle runs remain separate from any claim about total solution time.

1. **Gold-aware GPU DFS.** CUDA and WebGPU follow explicit paired endpoints, including same-side turns, and reverse seam positions exactly. A closed component containing fewer than all Gold segments proves the branch cannot solve Gold. The new placement is rejected, or a restored branch is unwound to its closing placement while preserving sibling cursors. Restored-prefix validation is split across bounded dispatches and blocks new search until complete. Its disposable scan cursor is rebuilt after restore; only this validation work can repeat. The systematic app defaults to Gold, continues after multi-loop 240-edge candidates, and still uses independent CPU validation for every accepted solution.
2. **Adaptive GPU work budgets.** High-resolution synchronized launch timings adjust budgets between 1 and 512 work units, with growth limited to doubling and immediate reduction after a slow launch. The target is 50 ms, not a hard latency guarantee. `--fixed-nodes` retains explicit fixed-budget experiments. Work scheduling does not certify or discard branches.
3. **Adaptive CPU worker slices.** Parallel DFS grows requested slices from 0.75 to 1.5 to 3 seconds while work is available. An idle worker can request a cooperative checkpoint from a busy worker; the coordinator splits only the validated committed frontier. Larger unfinished-work estimates receive split priority. Exact partition validation, bounded affinity, global Stop, and crash recovery remain authoritative. A crash can replay a longer uncommitted slice.
4. **Dynamic small-cut discovery.** In addition to neighboring cell pairs and complete faces, the solver finds other regions with two or three potentially occupied boundary seams. XOR of incident seam sets gives the complete boundary of each discovered subtree. Removing a traversal edge leaves its original incidence bit in the boundary computation. Exact topology keys cache at most eight discoveries; each contains at most 128 boundaries, prioritizing two-seam cuts. Every retained boundary is sound even when discovery is capped. The existing mask-correlation checker supplies necessary support; interrupted or capped checks never establish impossibility without a complete result.
5. **Useful learned-clause retention.** When First-UIP learning is enabled, clauses earn bounded retention credit for actual deductions or conflicts. Credit ages as new conflicts arrive. Eviction changes only which valid clauses are retained; it never removes unexplored frontier work. Proof binding and indexes remain unchanged, and metadata leaves with evicted clauses. First-UIP remains opt-in through the existing control.
6. **Ownership-cache turnover handling.** Repeated short-lived uniqueness certificates trigger eight ordinary full validations before probing caching again. The only shortcut remains exact equality with a frozen, validated array. Every bounds, lane-state, paused-overlap and completion check still runs. The private policy and certificate are never checkpointed.

Checkpoint version 3 requires an explicit Gold-connectivity marker. Versions 1 and 2 can migrate forward; a Gold-pruned checkpoint cannot be used by an edge-only engine, and older applications reject version 3. Downgrading the application therefore requires a compatible older checkpoint or a separate search output directory.

Pruning evidence:

- An independently enumerated eight-cell fixture retained all valid Gold solutions while dynamic cuts removed four orientation values, reducing raw orientation combinations from **6,561 to 1,296**. The easy fixture itself still solved in three DFS nodes in both variants.
- A bounded clause-retention fixture enumerated the identical valid solution set in **3 rather than 7 DFS nodes**.
- A copied original-puzzle checkpoint, stopped after 32 completed partial-loop checks, showed **100 additional orientation removals**, with **80 new search nodes in both versions**. Diagnostic elapsed times varied across runs; no timing conclusion follows.
- In bounded actual-data GPU checks, both backends exhausted 746 prefix branches. An independent CPU connectivity check certified **741 of those prefixes already contain a closed proper Gold loop**. The other five are explicitly unclassified by this root-level cycle check. This is evidence of sound branch elimination, not original-puzzle exhaustion.
- A 32,768-owner turnover control retained identical complete ledger fields while reducing certificate copies from **128 to 13** across 64 retire/refill transitions. Scheduling and cache changes make no claim of reducing mathematical search nodes.

Both GPU backends preserved exact saved device arrays and disjoint ownership through a **32,768 → 128 → 32,768** Stop/resume sequence. Adaptive work budgets grew 4 → 8 → 16 in those bounded checks. Mid-scan resume, legacy paused-bank migration, true legacy v2 input, missing v3 metadata, independently validated full Gold loops, and exhaustive two-cell completion tails are covered by regressions. CPU scheduling tests enumerate every one of 18 fixture leaves exactly once after a demand-triggered yield and split. Physical AMD and Apple hardware execution remains untested.

Final frozen-source suites discovered **906 tests per environment**: Python 3.12 passed **860 with 46 explicit skips**, and Python 3.15 passed **769 with 137 skips**, with no failures or errors. CUDA/Python 3.12, WebGPU/Python 3.12 and WebGPU/Python 3.15 each passed seven hardware and four host checks; the seven opposite-backend cases were skipped. Both final-source application resize sequences used checkpoint version 3. The earlier full-suite run was superseded because the checkpoint-format safeguard changed its source inputs during execution; its tests passed, but its manifest correctly records `inputs-changed`.


## Round 20: shared deductions, conditional inventory and native failure learning

This round accepts additional sound pruning even when the extra reasoning costs more than it saves in elapsed time. Main search nodes, look-ahead propagation and native replay checks are reported separately; fewer nodes alone are not a claim of less total computation.

1. **Constructive look-ahead.** For a selected cell with two to four orientation alternatives, private probes compute each surviving alternative's consequences. Only after every alternative finishes does the solver retain the union of their supported values. Values outside that union cannot appear in any completion. A capped or interrupted partition cannot publish shared deductions; already proved individual failures remain valid. Parent deduction reasons use the actual pre-probe facts, not hypothetical singleton assumptions. The existing 20 ms probe budget and Limited look-ahead control remain.
2. **Conditional cut inventory.** Each complete boundary-mask combination now requires an injective physical-tile assignment across its at most six endpoint cells. Exhaustive Hall-subset checks reject impossible combinations and remove tile choices that cannot participate in any such matching. All orientations of a physical piece share one inventory identity. This strengthens the existing Multi-seam region checks, including discovered cuts. Partial checks never publish unsupported exclusions.
3. **Learning from native failures.** The compiled search loop captures a seam/inventory contradiction only after committing the consumed node. With First-UIP learning enabled, bounded Python replay starts from the original effective root and active mask decisions, independently deriving the contradiction and its reasons. Captured carried domains alone are not proof. A completed explanation can yield a reusable clause and a safe asserting backjump. Incomplete replay learns nothing. Stop preserves the committed native frontier; neither captured failures nor disposable replay state enter checkpoints. This requires both experimental First-UIP conflict learning and Compiled search loop and undo trail controls.

The dashboard adds shared look-ahead removals, probe passes, cut tile conflicts, native replay clauses and replay checks. Shared removals are a subset of total look-ahead removals. Conditional matching support counts describe work inside alternative mask combinations; only the ordinary cut-removal counter counts net parent-domain removals.

Finite pruning evidence:

- Constructive look-ahead preserved all **3 Gold solutions among 2,916 enumerated boards** on a cube fixture. It removed **3 additional orientation values**, with the same 4 main nodes and 6 probe propagations. An initial copied original-puzzle checkpoint diagnostic completed 7 partitions and removed 1,215 common values, using 61 versus 65 main nodes and 38 versus 39 probe propagations. The final-source repeat completed 6 partitions but removed no common values; it used 61 versus 64 main nodes and 38 versus 39 probe propagations. Both stopped after 32 partial-loop checks. The 20 ms probe cap changes which partitions finish, so these are variable bounded diagnostics, not evidence of a consistent original-puzzle gain or a whole-puzzle work comparison.
- Conditional inventory rejected a fixture with **48 orientation supports** surviving previous cut checks, ordinary seam support and broad global Hall checks. A satisfiable companion retained all **768 Gold solutions**. Tests also compare 168 small random inventory graphs with brute-force injective assignments. On copied original-puzzle frames after the previous cut fixpoint, the new check removed **5 additional parent-domain orientations across 4 frames**. Its 77 conditional support removals are not 77 net parent removals.
- A complete four-cell native-engine fixture preserved the identical **152 solutions**, each enumerated once, while reducing native nodes **247 to 245** and transitions **575 to 571**. The new reasoning used **650 replay checks and 11 learned clauses**; the saving in native nodes is not a net speedup claim. Independently enumerated random fixtures check learned clauses against all valid completions. Restricted/replaced roots, nested mask decisions and Stop after a committed failure are covered. No reduction in the original puzzle's full search length has been established.

These are finite correctness and pruning results. They neither solve the original Gold challenge nor establish its solvability. Existing canonical puzzle data, best result and checkpoints are preserved.

Final frozen-source suites discovered **933 tests per environment**: Python 3.12 passed **887 with 46 explicit skips**, and Python 3.15 passed **796 with 137 skips**, with no failures or errors. The native backend was required for its integration tests. Both source-bound regression manifests and the bounded pruning manifest completed successfully. These CPU reasoning changes do not modify the GPU engines or native C libraries; unchanged hardware tests were not repeated this round.


## Round 21: conditional inventory, placement pairs and stronger assistance

The retention criterion remains independently verified pruning or reuse of completed reasoning, even when additional checks increase elapsed time. Main DFS nodes do not count all computation; matching steps, actual probe passes, cache reuse and SAT work remain separate counters.

1. **Whole-board inventory under cut choices.** After local cut matching, every surviving boundary-mask vector must admit an injective physical-tile assignment across the entire board. A completed augmenting-path search can reject a vector that reserves pieces needed by cells outside the boundary. The cut cache binds every outside inventory row, and a previous matching is only a rechecked hint. Work is limited to 200,000 matching steps per cut query. A cap or interruption discards the incomplete support union, while independently completed inventory facts may remain cached. Existing Multi-seam region checks enable it.
2. **Scoped pair probing.** Selected placement pairs receive private propagation and Gold-loop checks. Proved incompatible pairs retain their ancestor assumptions in learned exclusions. They are not promoted to global rules merely because they failed in a narrowed branch. A first placement can be removed only after all second-cell alternatives in a selected row fail. Grouped-mask paths without a complete singleton explanation skip this pair-learning path. The separate Test pairs of placements checkbox defaults on and requires Limited look-ahead.
3. **Verified Gold rules in SAT assistance.** A matching knowledge store supplies candidate loop exclusions, each independently revalidated against original tile segments before entering the SAT model. Clauses queued during construction install before the model becomes ready. Strengthening invalidates cached SAT answers; existing proved UNSAT cores remain valid. Imports, retained records and total clauses are bounded. Edge-only searches do not import Gold exclusions, and SAT success still cannot certify a Gold solution. Enable both Persistent SAT assistance and Remember verified loop exclusions; the optional SAT dependency is required for actual SAT execution.
4. **Completed-probe reuse and adjustable reasoning time.** A bounded private cache retains only completed probe results for exact domains, effective root, target, active assumptions and options. Stop or incomplete propagation never publishes an unfinished result. Completed entries can survive same-process checkpoint slices; they are not stored on SSD and are lost on restart. The per-node wall-time budget is now configurable from 1 to 1,000 ms, default 100 ms. Stop is polled within the checks. An internally expired partition still lets ordinary DFS advance; this does not resume an unfinished propagator or guarantee every partition will finish.

The dashboard reports conditional matching steps and conflicts, pair tests and actual pair propagation passes, completed probes reused, and verified Gold clauses in SAT. These counters describe reasoning activity; they are not a percentage of the full board space or a count of globally unique arrangements.

Finite evidence so far:

- A four-cell cut/inventory fixture removes **6 additional orientation values** beyond the previous cut and global-Hall fixpoint, preserving all **132 independently enumerated cut/inventory completions**. The check uses 3 whole-board matching calls and 44 matching steps, rejecting one vector. A separate companion preserves all **768 Gold solutions**. Matching tests compare 210 finite random graphs with independent injection enumeration. Copied original-puzzle checkpoint frames yielded **no additional removals** in the bounded comparison.
- Real Glucose and a separate complete-board oracle checked all **18 edge-compatible completions** of a two-cell fixture. Six independently verified loop clauses reject all **6 multi-loop completions**, preserve all **12 Gold solutions**, and invalidate 18 formerly cached SAT answers. This demonstrates a stronger helper model, not a measured reduction in original-puzzle DFS nodes.

- On a four-cell fixture with fixed tile positions and all **81 rotation combinations**, pair probing reduced complete infeasibility proof from **49 to 19 main DFS nodes**, with total propagation passes **127 to 119**. The satisfiable companion has **42 independently enumerated Gold solutions**; learned exclusions preserve every one, and a successful search used **5 rather than 7 nodes**, but increased propagation from **19 to 39 passes**. Both comparisons used the actual 100 ms setting without probe aborts. A demonstrated pair contradiction survived both singleton propagation checks and was legal in four original-root solutions under different ancestor choices, making the retained ancestor assumption essential. A surviving singleton probe is not a proof that the placement has a completion. The edge-only control retains all **81 edge-compatible boards**.
- In actual Python and compiled-loop Stop/resume tests, two completed probes remained in private memory. Resuming with that cache needed **4 new singleton propagation passes versus 6** without it, producing the same three shared removals and retaining all three original fixture Gold solutions. This is reuse of completed work in one process; internal budget expiration still allows ordinary search to proceed.

Final frozen-source suites discovered **961 tests per environment** on Windows: Python 3.12 passed **915 with 46 explicit skips**, and Python 3.15 passed **823 with 138 skips**, with no failures or errors. Actual SAT execution was verified with Glucose on Python 3.12; Python 3.15 used the independent board oracle and recorded unavailable optional-SAT cases as skips. Native CPU integration was required in both suites. The combined final-source pruning run reproduced the results above, and all three source-bound manifests completed successfully. No native C or GPU kernels changed; unchanged GPU hardware tests were not repeated.


## Round 22: conditional supports, SAT domains and regional pairs

Three extensions strengthen deductions rather than promising a higher node rate:

1. **Conditional matching support.** Each complete cut-mask alternative now filters all whole-board matching-supported physical-piece choices, including outside cells. The complete alternative union is required before applying cut deductions. Outside cache masks contain all rotations of supported pieces and are intersected with the current domains. Matching and support analysis each have a separate 200,000-step cap; unfinished support analysis falls back to the R21 relaxation, and unfinished matching remains unknown. The two finite comparison fixtures each remove two more orientation choices than the actual R21 source plus ordinary Hall propagation at fixpoint. They preserve all 32 independently enumerated cut/inventory completions. A further 851 finite matching graphs and 4,079 Gold solution/context pairs check support correctness and preservation. These conditional completion counts are not original-puzzle solutions.
2. **SAT domain deductions.** The helper temporarily assumes every current domain restriction, so it can find unit consequences and disprove individual placements that placement-only queries cannot exclude. Calls share a 10 ms deadline and test at most 16 candidates by default. Unknown, timeout and Stop do not establish new exclusions; returned deductions retain all input-domain dependencies. No conditional core is published as a global clause. A narrowed four-cell fixture retains 15 edge-perfect arrangements, including six Gold solutions; a complete 48-candidate-budget diagnostic removes 12 choices beyond ordinary seam/inventory propagation. The 16-candidate cap also yields additional deductions when allowed sufficient test time. This comparison does not guarantee completion inside the default 10 ms budget. An additional fixture verifies a forced placement from unit propagation alone.
3. **Cross-region support.** Two completed, disjoint regional tables are compared through piece inventory, shared seam masks and endpoint pairings. A summary contributes placements only if it has a compatible partner, and domain reductions publish only after the bounded pair scan completes. Local cached tables are not mutated, and this is pairwise placement filtering rather than persistent multi-table consistency. The four-cell comparison removes six additional orientations beyond independent R21 regional fixpoint while retaining all four Gold solutions. A seeded 144-query check compares supports with independent exhaustive enumeration, retaining 664 solution occurrences. A separate 16+16-cell test preserves two known Gold completions without constructing a 32-cell table; it does not enumerate the full 32-cell solution set.

Multi-seam checks use the matching extension automatically. SAT assistance and hierarchical regions keep their existing optional controls, both off by default. Cross-region scanning is bounded to 20,000 pair comparisons and one selected disjoint pair per pass. Completed regional query and child-table supports share 65,536 support slots and 8,192 cached summaries. Work counters distinguish conditional support rejections, SAT removals/forced placements, and regional comparisons/removals; they are not full-board coverage counters.

Reproducible finite experiments, source snapshots and version-2 manifests are stored under `audit/optimization-round22/`. Tests cover both CPU engines, scoped learning, cache widening and changed boundaries, cancellation, edge-only semantics, and Stop/resume with SAT assistance disabled. Actual Glucose runs use the Python 3.12 profile; Python 3.15 tests exercise graceful optional-dependency absence. No GPU kernels or native ABI changed. These results justify retaining stronger reasoning on the tested finite cases; they do not establish fewer total steps or a faster solution for the original Gold puzzle.


## Round 23 candidate evaluation

See the [complete 52-proposal register, two reopened candidates, controls and validation scope](ROUND23.md). New comparisons distinguish actual deductions and reused proofs from raw node throughput.
