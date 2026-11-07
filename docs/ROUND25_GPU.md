# Round 25 GPU and hybrid proposals

Proposals are implemented and independently reviewed in order. Features remain opt-in. Finite discriminator results do not establish a speedup or a solution for the original unrestricted Gold puzzle.

## 26. Joint neighboring seam and piece support

Implemented and accepted after independent review. The `gpu_joint_support` option applies to the experimental CUDA domain engine (`gpu-domains`). The compact GPU DFS and hybrid sampler are separate engines.

For a candidate orientation, all seams to the same neighboring cell must be supported by the same neighboring orientation. The candidate's physical tile is excluded from every neighboring support set. The engine checks Hall's condition for each nonempty subset of the at most three distinct neighbors. A failed subset removes the candidate. Explanations union the participating neighbors' domain-removal reasons; the permanent lane cube remains part of the inference scope.

A neighbor support set can stop after finding D distinct physical tiles, where D is the number of distinct neighbors. Every subset containing such a set already has at least D possible tiles and contains at most D cells. Smaller sets remain exact. This preserves the local Hall answer, including repeated seams and rotations spanning a 32-bit domain word. An independent host Cartesian-matching check covers all 4,368 systems of up to three support sets over four physical tiles.

Five durable 64-bit counters report joint candidate checks, Hall subsets, exclusions, compatibility words examined, and supported orientations visited: `joint_support_checks`, `joint_subset_checks`, `joint_exclusions`, `joint_word_scans`, and `joint_tile_visits`. These are separate work units. Enabled state adds 40 bytes per lane; 160-cell state grows from 942,324 to 942,364 bytes per lane. Explicit state and VRAM budgets still apply.

Disabled native state retains the v1 layout. Enabled state uses `diamond-domain-cuda-v2` and binds `joint_support: true`. Identical-policy resume preserves exact arrays and counters. Changing the feature or reducing replicas reconstructs scratch from validated disjoint remaining cubes; it retains search coverage but not the inference cursor. `gpu_domain_portable_restart` reports that distinction. Old runner checkpoints without the new counters remain accepted with zero initial values.

Completed finite CUDA comparisons on an RTX 2060:

| Controlled fixture | Solutions, both modes | Decisions off -> on | Existing candidate visits off -> on | Added joint word / orientation / Hall work |
| --- | ---: | ---: | ---: | ---: |
| Four-cell neighboring inventory | 6 edge solutions | 13 -> 9 | 154 -> 117 | 315 / 489 / 735 |
| Two-cell conjoined repeated seams | 0 | 3 -> 0 | 18 -> 3 | 3 / 0 / 3 |
| Twelve-cell tile-word boundary | 18 edge solutions | 24 -> 24 | 640 -> 640 | 706 / 649 / 622 |
| Controlled 160-cell planted Gold | 1 Gold solution | 0 -> 0 | 326 -> 326 | 14,400 / 960 / 2,240 |

The 160-cell fixture uses actual polyhedron geometry and a synthetic known single loop (`planted_gold`, seed2526). All but three cells have singleton domains; those three allow their tile's rotations. It tests full-size ownership/connectivity preservation, not the original puzzle's difficulty. Extra work in neutral cases is an explicit tradeoff; short-check wall times are not performance conclusions.

Validation includes independent complete domain-product enumeration followed by the raw board validator, edge and Gold objectives, learning and strand-maintenance combinations, disjoint remaining-solution checks after every primitive, native save/restore, 64-bit counter carry, legacy policy-change resume, and the controlled 160-cell witness. Reproduce the finite report with `python benchmark_gpu_round25.py --proposal 26 --output audit/optimization-round25/gpu26-finite.json`.

## 27. Propagate parents before GPU forks

Implemented and accepted after independent review. `gpu_propagated_forks` applies only to `gpu-domains` and is independent of joint support. A single parent advances through bounded CUDA inference primitives to a depth-zero fixed-point barrier. The runner partitions its remaining domain cube and clones established piece ownership, assignments, and strand union history into the children. Each child cube becomes its permanent root boundary; old removal trails and decision reasons are reset. Parent counters are accumulated once, never copied into every lane.

The preparation cursor survives native Stop/resume. A stop after a fork preserves child arrays exactly. A policy change uses the validated portable remainder and reports a scratch restart. An unfinished parent and its children are never simultaneously queued as separate work. No per-lane state is added. Cloning temporarily retains one parent allocation alongside the bounded child batch; the existing free-VRAM check still applies.

`fork_parents`, `fork_children`, `fork_root_removals`, `fork_root_checks`, `fork_reused_assignments`, and `fork_reused_unions` distinguish successful prepared roots, allocated children, preparation work, and root facts copied into children. Preparation-specific work counters cover successful fixed-point preparations; failed or unfinished preparation is still included in the ordinary cumulative counters. Preparation work is a subset of the cumulative ordinary counters, not an additional quantity to sum with them. Reused assignments/unions count copies made into children, not measured avoided operations. `gpu_domain_preparing` indicates unfinished root propagation. None of these is a count of exhausted complete positions. Preparation is serial per queued parent; dynamic refill is a later proposal.

Completed finite comparisons have joint support enabled in both configurations. The baseline cold-splits the original domains; the reuse configuration pays for parent preparation once, then enumerates all children. Reported work includes both parent and child execution:

| Controlled fixture | Solutions, both modes | Decisions off -> on | Candidate visits off -> on | Strand joins off -> on | Domain removals off -> on |
| --- | ---: | ---: | ---: | ---: | ---: |
| Twelve-cell single-loop root reuse | 2 Gold | 0 -> 0 | 46 -> 60 | 44 -> 17 | 2 -> 2 |
| Four-cell neighboring inventory | 6 edge | 6 -> 6 | 103 -> 98 | 0 -> 0 | 25 -> 18 |
| Twelve-cell tile-word boundary | 18 edge | 22 -> 22 | 621 -> 643 | 0 -> 0 | 43 -> 43 |

Saved strand work is useful even though extra preparation scans can increase other work. The root-reuse fixture is a synthetic twelve-cell loop with ten fixed pieces and two interchangeable remaining pieces. These are complete enumerations against independent raw-board validation, not original-puzzle speed measurements. Tests also preserve a controlled full160-cell Gold witness, exercise both strand modes and joint support settings, validate every preparation primitive's remaining solutions, reject incomplete fork partitions, and restore rebased native state.

Reproduce with `python benchmark_gpu_round25.py --proposal 27 --output audit/optimization-round25/gpu27-finite.json`. Hardware correctness tests require `DIAMOND_TEST_CUDA=1`; ordinary CPU CI runs host contract tests without a device.

## 28. Event-driven CUDA domain propagation

Implemented and accepted after independent review. `gpu_event_propagation` is a separate default-off `gpu-domains` option. A persistent dirty-cell bitset deduplicates dependency notifications. Domain removals schedule seam neighbors (including joint-star consumers); a new singleton also schedules itself. New physical-tile ownership schedules cells still containing that tile. A static index built from the original domains skips cells that could never consume the tile. Every valid lane and restored domain remains a subset of those original domains.

When the dirty set empties after domain changes, learned clauses receive another pass before the engine declares a fixed point. Branch creation and rollback conservatively schedule every cell. Queue contents, active-cell candidate cursor, and clause cursor survive native checkpoints. The v3 native binding includes event propagation; changing it reconstructs scratch from portable work. Restore rejects dirty padding bits and a claimed fixed point that still has dirty cells, an active scan, or a pending changed flag. Child forks start with their own complete dirty set and retain inherited root facts.

Five 64-bit counters separate the added operations: `event_enqueues` (new dirty bits, excluding constructor initialization), `event_cell_scans` (cells taken from the set), `event_inventory_checks` (potential consuming cells tested), `event_queue_word_scans`, and `event_inventory_words` (static consumer-index words read). Queue state and counters add60bytes/lane at160cells; the shared static index adds3200bytes per engine. With joint support enabled, state is942424bytes/lane. The existing512-lane ceiling and512MiB state budget remain valid.

Completed finite comparisons hold joint support enabled and perform complete enumeration in each mode:

| Controlled fixture | Solutions, both modes | Decisions, both modes | Candidate visits off -> on | Added consuming-cell / index-word / queue-word checks |
| --- | ---: | ---: | ---: | ---: |
| Twelve-cell single-loop root reuse | 2 Gold | 2 | 64 -> 46 | 4 / 14 / 47 |
| Four-cell neighboring inventory | 6 edge | 9 | 117 -> 108 | 35 / 16 / 100 |
| Twelve-cell tile-word boundary | 18 edge | 24 | 640 -> 346 | 24 / 34 / 355 |
| Controlled160-cell planted Gold | 1 Gold | 0 | 326 -> 171 | 0 / 800 / 499 |

The implementation removes repeated candidate reasoning; it does not reduce decisions on these fixtures and introduces explicit queue/index work. The controlled full-size fixture is synthetic and near complete. These measurements establish neither original-puzzle gain nor universal wall-time improvement. An earlier version scanned every cell on ownership notifications; the static consumer index removed that avoidable work before retention.

Validation passed112 combinations across seven fixtures, both objectives, joint support, learned explanations, and incremental strand modes, each compared against independent full domain enumeration. Additional checks cover every-primitive remaining ownership and native-array resume, policy changes, prepared forks, full160-cell connectivity, original-domain index word boundaries, malformed checkpoints, and legacy runner behavior. Final focused suite:22tests passed, plus the subsequent fixed-point changed-flag regression. Reproduce with `python benchmark_gpu_round25.py --proposal 28 --output audit/optimization-round25/gpu28-finite.json`.

## 29. Bounded precise reasons for GPU support failures

Implemented and accepted after independent review. `gpu_precise_reasons` is a default-off `gpu-domains` option, independent of proposals26-28. Seam and failed joint-star explanations consider only originally possible supporting orientations that have actually been removed. A bounded per-lane hash cache records the decision-bit reason for each cached deletion. A missing entry uses the existing conservative whole-cell reason. The feature cannot infer stronger conclusions from a cache miss, saturated table, or exceeded probe bound.

Default capacity is the next power of two covering twice the possible cell/code pairs, capped at2048slots; occupancy is at most half and each lookup/insertion inspects at most32slots. Root-domain masks fix the support universe. A separate bitmap retains zero-decision deductions under that immutable root, preventing these facts from consuming cache capacity or acquiring unrelated later decision reasons. Undo invalidates each reverted deletion; insertion also invalidates an old same-key record before any possible dropped insertion. Tombstones preserve collision chains. Fresh child forks create fresh root masks, zero bitmaps, and caches while retaining proved assignments/connectivity.

For the ordinary app, search stops at its first validated solution. Low-level enumeration can acknowledge solutions, so its later reasons and zero bits are relative to the root plus those acknowledged exclusions. Such checkpoints carry delivered history; the app rejects them as infeasibility searches, and root forking rejects delivered histories. These local native arrays remain trusted process state, never portable proof authority. Native v4 restores exact cache state and checks root identity, zero-mask containment, active removal identity, unique keys/counts, hash reachability, and reason padding. Policy changes use the portable remainder and rebuild scratch.

Eight 64-bit counters distinguish hash probes, cached hits, conservative fallbacks, stored entries, invalidations, support words, support values, and dropped insertions: `precise_probes`, `precise_hits`, `precise_fallbacks`, `precise_stores`, `precise_invalidations`, `precise_support_words`, `precise_support_values`, `precise_dropped`. The160-cell default adds68420state bytes/lane. With joint support and event propagation, total state is1010844bytes/lane;512lanes require517552128bytes (about493.58MiB), within the512MiB state budget. Static tables and a temporary preparation parent remain separate allocations covered by the existing free-VRAM check.

An independent seeded CPU reference search found a six-cell triangular-prism discriminator (seed2529, trial789). Direct raw-board enumeration checks34560Cartesian assignments,730with unique physical inventory, and finds zero edge or Gold completions. In one explanation, coarse decision bits111 become101: unrelated cell1's decision is absent from every relevant missing seam-support reason. Actual CUDA matches the independently predicted improvement:

| Controlled fixture | Solutions, both modes | Decisions off -> default cache | Candidate visits off -> default cache | Learned clauses off -> default cache |
| --- | ---: | ---: | ---: | ---: |
| Six-cell prism, edges | 0 | 21 -> 17 | 509 -> 458 | 20 -> 16 |
| Six-cell prism, Gold | 0 | 21 -> 17 | 497 -> 446 | 20 -> 16 |
| Twelve-cell tile-word boundary | 18 edge | 24 -> 24 | 640 -> 640 | 6 -> 6 |
| Controlled160-cell planted Gold | 1 Gold | 0 -> 0 | 326 -> 326 | 0 -> 0 |

The edge discriminator pays718hash probes and63support-word checks, with148cached reason hits. Deliberately restricting it to8slots produces119dropped inserts and34fallbacks, preserves the result, and returns to baseline21decisions/509candidate checks. Neutral fixtures incur extra bookkeeping without fewer decisions. No original-puzzle gain or general wall-time speedup has been established.

Validation covers48 complete-enumeration feature combinations plus8 learning/strand-disabled combinations. At every primitive of a satisfiable fixture, independent completions verify cached reasons and zero deductions relative to the still-undelivered solution set. Deliberate collisions, saturation, backjumping, uncached re-deletion, native restore, malformed identities, portable policy changes, rebased forks, and controlled full160Gold preservation pass. All25focused tests passed; one entailment test was rerun after clarifying enumeration scope. Reproduce the finite report with `python benchmark_gpu_round25.py --proposal 29 --output audit/optimization-round25/gpu29-finite.json`.

## 30. Independently verified sharing between CUDA lanes

Implemented and accepted after independent review. `gpu_shared_clauses` is a default-off `gpu-domains` option requiring python-sat. Lane-local learned clauses are untrusted claims outside their own owned cube. The coordinator offers bounded samples to `VerifiedConflictStore`, whose separate CPU SAT model uses the original puzzle domains, physical inventory, and seams. Only completed original-root UNSAT answers are eligible for another lane. A Gold-only conflict that this edge model cannot prove is left unshared; SAT is never accepted as a Gold solution.

Harvesting reads at most four lanes and16claims per synchronization, with a durable lane-and-slot cursor so a busy lane cannot starve later clause slots. Claims longer than32placements are skipped. Existing pending claims receive a verification slice first; harvesting is limited by free pending capacity, so new claims do not evict old unfinished verification. A slice allows one query,10ms requested wall budget, and256native conflicts. These are cooperative limits. The verifier model and native query work remain separate from GPU decisions.

At most eight short verified clauses refresh into each lane's existing32-clause ring. Duplicate clauses are not inserted; unchanged learned generations and verified records skip redundant refreshes. Ordinary local learning may later evict these clauses, and a subsequent refresh can restore them. This adds no per-lane persistent arrays. New batches receive already verified clauses before their first search step. Injection restarts clause processing in live scan/branch phases; a pending failure completes rollback first. Enumeration exclusions are never harvested as puzzle failures.

A verified empty original-root conjunction is handled explicitly by the coordinator: all remaining work is cleared, an empty frontier is saved, and the original scope may be recorded as infeasible with independent-verification evidence. It is not interpreted as an unused device clause slot. `gpu_verified_root_unsat` distinguishes this outcome.

The app's existing verified-conflict lifecycle is reused. Runner checkpoints separately retain verified-record transport and unverified pending/discarded work. The new pending API preserves maximum retry counts and bounded discarded markers across restarts; neither can publish a clause. It validates complete payloads before mutation, with128pending and128discarded records, at most32literals each, and a separate1MiB serialized cap. The original verified snapshot API and131072-byte bound are unchanged. FIFO eviction may eventually allow optional work to be rediscovered. Imports are independently revalidated, and an interrupted native SAT query may need to run again; this is not native SAT-state persistence.

Progress, checkpoint, and final events expose detached verified snapshots. Immediate-stop resumes retain saved sharing telemetry. Unknown or cancelled verification never changes GPU domains. Native lane checkpoints still contain previously admitted local/global clauses as trusted process state; they are not proof imports. A cold resume can repeat clause comparisons, while identical native arrays and remaining ownership remain intact.

Work counters are `gpu_claims_harvested`, `gpu_claims_offered`, `gpu_claims_oversize`, `gpu_shared_imports`, `gpu_shared_comparisons`, `gpu_shared_verification_calls`, `gpu_shared_verified`, `gpu_shared_rejected`, and `gpu_shared_unknown`. Gauges report verified entries, pending claims, and verifier model clauses. GPU decisions, CPU query counts, and model-clause counts are different units and are not added together as a node total.

A completed finite experiment uses genuinely disjoint producer and recipient regions. The producer prefix costs5decisions/54candidate checks and remains unfinished. Its two claims are independently checked with2CPU queries and an88-clause model: `(cell0,code0)` is globally impossible, while `(cell1,code0)` is impossible only under the producer restrictions and is rejected for sharing. The three recipient cubes exhaust exactly the same four independently enumerated edge completions:

| Recipient work | Without reuse | With verified reuse |
| --- | ---: | ---: |
| Decisions | 4 | 2 |
| Candidate visits | 87 | 64 |
| Domain removals | 17 | 14 |
| Imported lane clauses | 0 | 3 |

Producer-prefix and verification setup costs are reported separately in the receipt; reuse is not free work. This is completed recipient enumeration, not completion of the still-owned producer region or the original Gold puzzle. It demonstrates fewer repeated recipient decisions, not a general wall-time gain.

The final26-test GPU/runner suite passed, including actual SAT rejection of a local-only claim, verified whole-root contradiction before GPU allocation, fair harvesting, durable unknown budgets/tombstones, cancelled verification, phase0-3 injection, Gold completion preservation, controlled full160Gold/native resume, app initial injection, and saved snapshot ownership. The pending-store API also passed its separate29-test new/legacy suite. Reproduce the development finite report with `python benchmark_gpu_round25.py --proposal 30 --output audit/optimization-round25/gpu30-finite.json`. It records python-sat1.9.dev15 and explicitly hashes the verifier/store/encoder dependencies; frozen release evidence is a separate later gate.


## 31. Exact lane refill and remaining-work donation

Implemented and accepted after independent review. `gpu_lane_balance` is a default-off CUDA `gpu-domains` option. It does not require propagated forks. A physical lane is a temporary worker; each input root has a durable owner record that remains referenced by every unfinished live or queued descendant. An owner enters the coverage ledger only after all those descendants exhaust. A propagated parent keeps its original pre-propagation owner across forks, donations, and queued fragments.

Finished slots retire their work counters exactly once and clear only the native counter rows. Queued work fills idle slots first. If the queue is empty, at most one active donor is selected per scheduling pass. Its exact remaining frontier replaces the donor task atomically: one box stays in the donor lane, one moves to an idle lane, and any additional boxes remain queued under the same owner. A single nonsingleton box can be bisected. Donation-created queued fragments are capped at4096; larger imported queues remain accepted and are drained before donation is attempted.

Donation reconstructs the selected donor's trail, assignments, connectivity, and reason cache from its exact remaining cubes. It does not preserve that donor's native stack. Every untouched lane remains byte-for-byte unchanged. No GPU search kernel runs between ownership transfer and replacement commit. Existing verified original-root clauses are explicitly refreshed into replaced lanes. Constructor-only initialization still performs no search.

Native resume saves owner roots, lane and queue ownership, retired slots, counters, and all native columns. Changing the balancing policy, propagation policy, or reducing the lane capacity preserves the validated portable remainder and reconstructs scratch; broader owner-completion claims are conservatively abandoned. Trusted local checkpoints remain process state, not imported proofs. Stop does not declare unfinished donors or parents exhausted.

Counters are `gpu_lane_refills` (queued tasks placed in slots, including independent extras during a prepared fork), `gpu_lane_donations` (completed ownership transfers), `gpu_lane_rebuilt_roots` (fresh roots replacing existing columns), `gpu_lane_discarded_frames` and `gpu_lane_discarded_removals` (donor depth and live removal-trail entries discarded during reconstruction), `gpu_lane_owners_completed`, `gpu_lane_donation_cubes`, and `gpu_lane_balance_deferrals`. Gauges report idle slots, unfinished owner groups, and peak queued fragments. Retired GPU counters are folded once into cumulative totals; a task count is not a count of distinct complete board arrangements.

A completed six-cell prism experiment starts with12 disjoint queued roots and4 hardware lanes. Independent raw enumeration proves there are no edge completions. With fixed per-dispatch primitive budgets, all four configurations complete the same input scope:

| Primitive budget | Dispatches without/with balancing | Decisions without/with | Candidate checks without/with |
| --- | ---: | ---: | ---: |
| 1 | 559 / 393 | 21 / 18 | 898 / 985 |
| 8 | 70 / 53 | 21 / 19 | 898 / 971 |
| 17 | 34 / 26 | 21 / 23 | 898 / 947 |
| 64 | 10 / 7 | 21 / 18 | 898 / 896 |

This demonstrates less unused scheduling capacity on this finite input. Additional repeated propagation can offset the gain, and one configuration takes more decisions. It is not an original-puzzle speed claim. The reproducible development receipt command is `python benchmark_gpu_round25.py --proposal 31 --output audit/optimization-round25/gpu31-finite.json`.

Validation includes the73-test related CUDA/runner/checkpoint suite plus9final focused tests after stricter version/counter validation. Focused checks cover exact complete-solution preservation through donation, untouched columns, once-only retirement, durable ownership, mixed prepared-parent/independent roots, Stop and policy resume, delayed parent coverage, queued refill, actual verified-clause reapplication, and controlled full160Gold. The independent reviewer verified all19source hashes in the final eight-case finite receipt.


## 32. Shared live-state checkpoint generations

Implemented and accepted after independent review. `gpu_incremental_checkpoint` is a default-off CUDA `gpu-domains` option. One frozen GPU-to-host native snapshot supplies checkpoint encoding, statistics, and portable remaining work. The legacy path copies the full native array separately for encoding and frontier reconstruction. Checkpoint serialization remains synchronous with search ownership changes; asynchronous app consumers receive immutable generation references.

The codec saves active removal/assignment/union logs and decision frames, while keeping domains, coarse reasons, DSU state, counters, dirty queues, cursors, all32clause lengths and each ring slot's declared literals. Precise-cache keys and tombstones remain exact; reason words for inactive cache slots and unused log/frame tails normalize to zero. This preserves future search behavior and live state, not stale padding bytes. Bounds and structural validation remain active independently of Python assertions. Every-primitive tests compare canonical state and continuation through phases0-4; controlled160-frame tests preserve the exact remaining volume after two acknowledged leaves.

Each lane's packed state is compressed and addressed by SHA256. Unchanged packed chunks reuse host compression work, and already stored chunks are not written again. A manifest binds the entire matching runner state: queue, ownership groups, cumulative counters, pending verifier work, portable frontier, and native chunk identities. Publication writes immutable chunks, then the complete manifest, then atomically replaces a small HEAD journal. Consumers validate reference, manifest, size, scope and chunk hashes before a native restore can mutate device arrays. Frontier export reads manifest metadata only.

All old objects remain retained, including objects referenced by delayed app callbacks. The app process lock provides one writer per run directory; direct API callers must also serialize writers because this store does not add an interprocess creation/cap lock. There is no garbage collection in this version. The cap is512MiB of logical file bytes plus32768files, with a reservation for atomic metadata writes; filesystem allocation blocks and journaling can occupy additional disk space. Unreachable objects left by a failed publication also count toward the cap. On cap or I/O failure, the current complete checkpoint falls back to the existing self-contained native format and reports `legacy_fallback` with its reason. Existing durable references remain intact. The one-snapshot path still applies after fallback, although live-prefix compression reuse stops for that run.

Fresh runs never inherit an unrelated old HEAD. A writer chains only from its explicitly validated resume origin or its own last successful generation. Normal app resume is strict: a corrupt or missing object is an error. The optional recovery API follows only the named previous-generation chain in that same store, at most eight links, and restores the entire older host/native state. It reports that work after the recovered generation may repeat. It never selects an unrelated or newer HEAD. Recovery markers are consumed once when new checkpoints save their cumulative rollback counter.

The checkpoint directory is provisioned for all domain-engine runs so feature32 can be disabled while resuming its earlier checkpoint. A self-contained fallback remains usable without its chunk directory. A referenced checkpoint requires its matching directory; copying only checkpoint.json is insufficient. Older legacy checkpoints remain readable. The stored native state is still trusted local process state, not a proof import.

Cumulative telemetry reports `gpu_checkpoint_host_copies`, `gpu_checkpoint_reused_chunks` (host compression reuse), `gpu_checkpoint_fallbacks`, and `gpu_checkpoint_rollbacks`. Latest-size gauges report `gpu_checkpoint_packed_bytes`, `gpu_checkpoint_compressed_bytes`, `gpu_checkpoint_omitted_bytes`, and `gpu_checkpoint_store_bytes`. The current-generation retained-byte gauge can update after that generation's saved metadata was constructed. Status and fallback notes are separate from integer work counters.

Finite CUDA evidence includes two consecutive snapshots for a partially exhausted six-cell prism and a controlled160-cell Gold instance. Every decoded snapshot independently completes exactly its remaining solution set. The controlled160-cell case with four lanes changes4,043,216native bytes into249,766packed bytes; compressed native content falls from27,695 to22,791bytes. Its next generation reuses one of four compression chunks and writes146,789logical bytes, compared with165,028bytes for the equivalent legacy JSON payload. The small prism's extra manifest/chunk metadata makes its files larger, despite less live data and one reused chunk. These are bounded storage/repeated-work measurements, not a claim of original-puzzle search acceleration.

The final37-test focused suite passes: delayed consumers, capacity preservation, failure between manifest and HEAD, corrupt/missing chunks, strict and explicit prior-generation recovery, same-directory fresh-run lineage, actual lane donation/refill and verified-clause restoration, Stop and disabled-feature resume, controlled full160Gold, and all160active frames. Reproduce the finite development receipt with `python benchmark_gpu_round25.py --proposal 32 --output audit/optimization-round25/gpu32-finite.json`.


## 33. Durable hybrid sampling and repair

Implemented and retained opt-in after independent review. `durable_hybrid` applies to the hybrid engine and supports both CUDA and native WebGPU. It preserves the existing AMD/macOS WebGPU path; the local physical-device checks used an RTX2060 through CUDA and native WebGPU, not physical AMD or Apple hardware.

Hybrid checkpoint version3 records the current phase, once-allocated phase budgets, sampler/controller state, repair state, checked candidate archive and exact DFS remainder. App duration0 means unlimited overall search; heuristic phases still receive at most30seconds of sampling and60seconds of repair. Repeated30second app slices do not allocate those heuristic budgets again. Every completed phase publishes a detached checkpoint before the next phase starts native work. A saved version3 checkpoint is adopted even if the feature switch is subsequently disabled. Older version1/2 checkpoints retain their exact-DFS-only resume behavior.

The sampler saves both current and best boards, all RNG/temperature/tabu metadata, global dispatch count, exchange parity schedule, archive and collection cursor. Collection uses a deterministic eight-dispatch cadence; a final partial cadence is collected once. Resume skips population initialization and continues the same next64-move dispatch. An unfinished sampler requires the same replica count, policy, backend/device/runtime, input tables and source/kernel identity. After sampling ends, irrelevant GPU settings can change. This is same-runtime trajectory persistence, not cross-device checkpoint portability. Returned hints and solutions are independently scored; native state itself is trusted local state and has no coverage or proof authority. The native byte payload is bounded at256MiB before compression. A closed session rejects further advancement or snapshots.

Ordinary local repair saves its RNG, selected patch and exact outside-board context, pending query, cumulative attempt limits, adaptive schedule, candidate archive and bounded completed-query cache. A completed query can be reused after restart. If an interrupted query has already generated independently validated Gold-loop cuts, those witnesses are reconstructed in its new CP-SAT model rather than rediscovered. The four earlier experimental repair controllers also continue: core repair retains completed cores and pending master/query state; fragment crossover retains master conflicts and pending selection; scoped repair retains its exact pending query; loop trades retain their search cursor. Interrupted native OR-Tools search itself is retried. Native CP-SAT internal state is not saved.

The default ordinary controller retains at most64 logical query attempts and8 archive boards, with at most512 Gold witnesses for its one pending query. The direct API bounds attempts at4096, input archive at64 boards and published/restored repair snapshots at64MiB. Capacity errors do not silently discard pending work. Scope/source changes, invalid target certificates and malformed native snapshots fail before publication. Completed repair queries and local failure reports never establish global puzzle exhaustion. Repair assumption cores are offered to the independently verifying mask store once per live store and scoped report; a fresh app process requeues them even when the app has already constructed an empty store.

Sampling telemetry separates `initial_boards`, `move_proposals`, `dispatches`, `sampler_restores`, `archive_checks` and repeated archive observations. These count proposed or inspected work, not globally unique arrangements. Repair `attempts` count outer local queries/model attempts; `repair_native_calls` counts actual CP-SAT solve calls, which may exceed attempts because a query can add several Gold cuts. Model-build attempts, successful model builds, pending-query retries, restored cuts and completed-cache hits remain separate. The legacy repair `repair_completed_work_reused` counter counts reuse events across restores, not a count of unique avoided calls.

Actual finite native tests preserve arrays, archives and experiment counters for18 dispatches versus9 plus snapshot/restore plus9, across CUDA/WebGPU, base and combined ejection/tabu/exchange policies, edge/Gold objectives and fixed placements. An actual hybrid Stop at9 dispatches followed by a version3 resume with the feature switch disabled also matches18 uninterrupted dispatches on both backends. The focused suite has28 passing checks, including simultaneous lifecycle, phase-transition, model/cut, scoped-cache and malformed-state cases. A separate fresh-process development check on the original160-cell input, two policies and both backends also matched exactly:147456 move proposals, no completed dispatches replayed. This validates continuation, not progress toward a Gold solution.

The six-case finite repair harness independently enumerates the small fixture's allowed Gold outcomes and compares continued work against discarding the saved prefix and restarting:

| Completed comparison | Continued work | Discarded-prefix restart |
| --- | ---: | ---: |
| Ordinary completed query prefix | 54 native solves | 72 native solves |
| Ordinary interrupted query with saved Gold cuts | 54 native solves | 55 native solves |
| Core-guided repair | 4 native calls | 5 native calls |
| Fragment repair | 4 native calls | 5 native calls |
| Scoped repair | 1 native call | 2 native calls |
| Loop-trade controller | 5 completed transitions retained | 5 transitions repeated |

All six outcomes agree independently. The two-cell ordinary fixture has18 edge-perfect arrangements and no Gold solution; its local repair remains explicitly inconclusive about global feasibility. Reproduce with `python benchmark_repair_round25.py --output audit/optimization-round25/repair-new.json`; use `--evidence-stage frozen` only from a clean committed checkout. `benchmark_hybrid_round25.py` separately provides fixed-proposal and fresh-process sampler comparisons. Checkpoint construction, validation and model reconstruction are required overhead. No timing winner or original-puzzle acceleration is claimed.


## 34. Obstruction-guided ejection chains

Implemented and retained opt-in after independent review. `obstruction_chains` applies to both legacy and durable hybrid sampling, implies `ejection_chains`, and is available in CUDA and native WebGPU. It is off by default. This is a proposal-selection heuristic: every proposed board still passes the existing complete domain/seam/Gold score and acceptance rule. It introduces no exact-search pruning or coverage claims.

Three quarters of enabled proposals use guidance. A cyclic scan starts at a random movable position and selects an actual illegal-domain or unmatched-seam cell when one is present. For fully matched legal boards with proper Gold loops, it starts on a segment-bearing cell. The remaining proposals use the previous random ejection policy. A guided move relocates3–6 distinct physical pieces, or all movable pieces when only1–2 exist. For each successive receiver it evaluates all3 rotations from at most16 eligible donor positions, using domain legality and incident seams against the unchanged board. The final receiver takes the first displaced tile, closing the inventory-preserving cycle. Already selected and fixed cells cannot be donors. Random cyclic orders break ties reproducibly; only a strictly better local score replaces the first maximum.

The local placement score is neither a prediction nor a bound on the completed move: other moved neighbors may change. The full completed permutation is scored afterward, and may still be rejected. Guidance therefore trades extra candidate inspection for potentially fewer random proposals. On a closed Gold board, segment-bearing cells belong to the improper components; the guidance does not prove which multi-piece trade will merge them.

Enabled native metadata grows from18 to22 rows; feature-off layouts and RNG proposal draws remain unchanged. The four extra32-bit words cost16bytes per replica. Same-runtime durable snapshots include all new counters and state. `guided_proposals` counts guided cycles; `guided_candidate_checks` counts donor/closing orientation evaluations (at most243 per guided cycle); `guided_obstruction_starts` counts starts where an obstruction was identified; `guided_base_improvements` counts proposed increases in the domain/seam objective, including proposals later rejected by tabu. Candidate checks exclude the obstruction scan and neighbor comparisons, so they are not a total primitive-work measure.

Seventeen focused checks pass across the CPU reference and both actual native backends. Actual compiled CUDA and WGSL helpers agree with an independent unpacked CPU implementation for512 lane cases over1/2/3/6-cell chains, both objectives, fixed placements, multiple loops and the40-cell donor-cap case. Complete moves and exchanges retain independently correct scores for every tested lane. A controlled160-cell Gold-capable input checks22-row scores and exact4-dispatch versus2-plus-restore-plus2 continuation. Native software-adapter fallback is exposed only through the explicit CI test path. Four additional host tests validate the finite benchmark's paired initialization, zero-cost initial Gold, cap handling and exhaustive fixture enumeration.

The finite comparison uses the previous ejection policy as baseline,8 replicas, seeds1–12, and at most64 one-proposal rounds. Both policies start with exactly equal boards, best boards, RNG state and scores. Every proposal round checks all replicas independently. Initial Gold populations cost zero proposals. Both native backends produce the same recorded results:

| Independently enumerated fixture | Gold solutions | Solved within cap, baseline/guided | Lane proposals on the9 pairs solved by both, baseline/guided | Extra guided orientation evaluations over all12 pairs |
| --- | ---: | ---: | ---: | ---: |
| Four-cell tetrahedron | 4 | 9 / 12 | 1640 / 272 | 6054 |
| Same board with singleton domains | 1 | 9 / 12 | 2048 / 152 | 2688 |

Two tetrahedron initial populations already contain Gold and correctly cost0 moves. The other three baseline pairs in each fixture hit the cap; they are not counted as solved or folded into the completed-pair proposal totals. A successful round still charges every replica's proposal. These are finite synthetic puzzle results, not unique-board coverage, a total-work or timing win, or an original-puzzle improvement claim. Production archive collection normally occurs less frequently than this diagnostic's per-proposal checks.

Reproduce with `python benchmark_gpu_chains_round25.py --backend cuda --output audit/optimization-round25/guided-cuda-new.json` and the corresponding `--backend webgpu` command. The driver records full source/runtime/Git identity, bounds, complete paired outcomes and independent solution-set hashes; `--evidence-stage frozen` requires a clean committed checkout. Development receipts are `gpu34-cuda-finite-01.json` and `gpu34-webgpu-finite-01.json` in the round25 audit directory.


## 35. Inventory-preserving parent-cycle crossover

Implemented and retained opt-in after independent review. `cycle_crossover` is a default-off hybrid option for CUDA and native WebGPU, with no required sampling experiment. Both legacy sampling and the durable controller invoke it during actual archive collection. It is a host heuristic and adds no GPU metadata rows or exact-search exclusions.

For two complete parent permutations, map each cell to the location in the first parent of the physical tile used by the second. This is a permutation of cells. Copying the second parent's oriented codes over a complete cycle preserves physical inventory; copying any union of disjoint cycles does too. Singleton physical cycles allow orientation mixing. Fixed placements remain unchanged because both parents have the same fixed code there. Each proposed child is independently scored, must satisfy every original domain, and must improve both parents under the selected objective or be an independently validated Gold solution.

The existing diversity filter deliberately rejects very similar hint boards. That would discard useful parents close to a solution: a controlled160-cell test caught this during integration. Crossover therefore has a separate pool of at most8 exact-distinct parents, with deterministic quality/tie ordering. It retains nearby boards while the existing diverse hint archive continues to serve repair and DFS. Restored parent scores are independently recomputed.

A collection examines at most4 round-robin parent pairs and scores at most16 new children by default. Bounded per-pair cycle/direction cursors continue through interrupted batches. Recent exact child boards have a256-entry FIFO history; eviction can permit later repeated work. Parent departure can evict its cycle cursor. The history is optional heuristic reuse, not a global visited-board set or proof of exhaustion. Controller snapshots are capped at8MiB; direct API limits are28pairs,256candidate scorings and4096history entries. Counters and policy/scope validation remain active without Python assertions.

A useful nonsolved child can replace one lowest-current-score native lane, provided it does not lower that lane's current objective. An even better personal-best board remains intact. Only that lane's stale move-tabu fields are reset; its RNG, iteration/temperature schedule and native work counters are preserved, as are every other lane's bytes. The child is independently checked again before any native write. A Gold child is returned and saved without needing injection. Final phase collection also retains useful children as hints without injecting state that will immediately be discarded.

If an injection write or synchronization fails after mutation begins, the engine and owning sampler session cannot continue or publish a snapshot. Recovery requires a fresh session from the last durable checkpoint; partial device writes are not claimed to roll back. Cooperative Stop has no callback inside the bounded injection operation. Crossover decisions, writes, counters and the collection cursor complete before a progress/checkpoint callback can run.

Enabled sampler substate version2 saves the parent pool, pair/cycle scheduler, exact bounded history and counters together with native state. Feature-off sampler state retains version1 and makes no extra GPU reads for crossover. Both formats retain the existing exact input/source/runtime binding; they are not cross-version or cross-device trajectory transport. A finished or stopped resumed collection neither reinjects an admitted child nor scores its saved child history again. Interrupted native CP-SAT state remains outside this feature.

Host telemetry is separate from GPU move proposals: `cycle_crossover_rounds`, `cycle_parent_pairs`, `cycle_components`, `cycle_candidates`, `cycle_duplicate_skips`, `cycle_domain_rejections`, `cycle_improvements`, `cycle_gold_solutions`, `cycle_history_evictions`, `cycle_injections`, `cycle_parent_scores`, `cycle_parent_duplicates`, `cycle_pair_cursor_evictions` and `cycle_offers`. Parent rescoring and native-injection verification still cost work. A duplicate-history hit avoids child scoring, not every operation associated with visiting that parent pair.

The27-test focused suite passes. It includes all576 pairs of four-piece permutations and every union of their relative cycles, bounded scheduling/history and malformed-state checks, actual CUDA/WebGPU score/inventory validation,4/18/22-row injection layouts, current-score ties, preservation of a better personal best, worse-child no-write cases, simulated partial-device-write poisoning, completed injection plus Stop/native resume, and actual160-cell near-parent Gold recovery. Four additional host tests check the finite evidence contract and accounting.

The finite native comparison deliberately supplies parents derived from known synthetic Gold targets. Their discovery is not measured. Each engine's initial random population is explicitly replaced by the same prepared parents, with equal native boards, RNG and scores; that discarded setup is disclosed. The results are conditional on those supplied parents:

| Supplied-parent fixture | Crossover result | Previous ejection baseline |
| --- | --- | --- |
| Four-cell complementary physical cycles,4seeds | 4 Gold recoveries;2 parent scorings and2 child scorings each;0 GPU proposals | 2 seeds finish with212 summed lane proposals;2 hit the64-round cap |
| Four-cell complementary orientation cycles,4seeds | 4 Gold recoveries;2 parent scorings and2 child scorings each;0 GPU proposals | All4 hit the64-round cap |
| Controlled160-cell nearby parents | The independently validated planted Gold witness recovered with2 parent and2 child scorings;0 GPU proposals | No random-search baseline run or enumeration claim |

Both native backends give the same recorded results. The two small fixtures have complete independently enumerated Gold sets. The160-cell case verifies one planted witness and is unrelated to solving the supplied Diamond Dilemma tile set. Finished resume allocates no new GPU; replaying the same parents after controller restore scores0new children but still rescores2parents. CPU child evaluations and GPU proposals are different work units, so these results are not a timing or total-primitive-work comparison.

Reproduce with `python benchmark_gpu_crossover_round25.py --backend cuda --output audit/optimization-round25/crossover-cuda-new.json` and the corresponding WebGPU command. Full source/runtime/Git guards, exact prepared inputs, caps, independent certificates and bounded replay evidence are recorded. `--evidence-stage frozen` requires a clean committed checkout. Development receipts are `gpu35-cuda-finite-01.json` and `gpu35-webgpu-finite-01.json`. No original-puzzle gain, global deduplication or new exact pruning is claimed.


## 36. Exact lazy Gold scoring

`lazy_gold_scoring` is optional for hybrid sampling on CUDA and native WebGPU. The app enables `gpu_gold_sampling` and `ejection_chains` with it; the chain dependency keeps the same experimental proposal generator. Direct sampler callers retain their explicit objective, including the zero-segment score case. The feature changes repeated scoring work, not proposed boards or acceptance.

With segment count `T`, domain/seam score `B` and proper closed-loop count `L`, the stored objective is `S = B*(T+1) + T-L`, where `0 <= L <= T`. Integer quotient/remainder recover the old base score and loop count exactly. Every candidate still receives a complete base evaluation. A tabu candidate skips loop tracing if its integer upper score `Bnew*(T+1)+T` cannot strictly exceed the personal best. Equality rejects and consumes no RNG. For a non-tabu negative-base candidate with exchange disabled, the existing Metropolis input depends only on that base difference. The identical draw can precede tracing: rejection skips the trace; acceptance traces once and reuses the draw. Uncertain aspiration and exchange acceptance retain complete candidate scoring. There is no floating probability-bound shortcut.

Enabled native state has 28 metadata rows; other policies keep their 4/18/22-row layouts. Checked crossover injection, exchange and native/controller resume preserve exact stored scores. Counters under `gpu_sampling.sampling_experiments` are `gold_old_base_scans_saved`, `gold_old_loop_traces_saved`, `gold_candidate_loop_traces`, `gold_tabu_traces_skipped`, `gold_metropolis_traces_skipped` and `gold_early_metropolis_draws`. Loop counters increment only for `T>0`. Complete loop passes have varying segment-visit counts, so these are not all primitive operations.

Twenty focused checks pass: ten host/native, eight independent control-flow and two evidence-accounting tests. Native checks compare boards, best boards, RNG, iteration, tabu and every old counter after each chunk for all eight tabu/exchange/guidance combinations with Gold scoring on/off. Further checks cover temperature wrap at4096, fixed/no-free/one-cell cases, actual160-cell injection/exchange and both native and controller Stop/resume. The pure reference covers10,000 control-flow cases, strict aspiration equality, one-draw success/failure, best-tie RNG, zero uniforms, threshold/underflow, high integer scores and `T=0`. Pure float tests do not claim cross-backend exp identity; native equivalence is checked within each backend.

Source-bound comparisons use equal native initial populations and fixed chunks of1,7,64,64 proposals per lane. Both backends produce these results:

| Input and paired cases | Lane proposals per policy | Base passes baseline to lazy | Loop passes baseline to lazy |
| --- | ---: | ---: | ---: |
| Four-cell synthetic,8cases | 17,408 | 34,816 to17,408 | 34,816 to11,287 |
| Planted160-cell,4cases | 2,176 | 4,352 to2,176 | 4,352 to1,973 |
| Supplied original160-cell tiles,4cases | 2,176 | 4,352 to2,176 | 4,352 to2,022 |

Baseline counts follow one old and one candidate pass per proposal; initialization is excluded. All old native state matches after every chunk, and current/best scores pass independent checks. Fixed work deliberately continues after any synthetic Gold. This demonstrates fewer repeated score passes on bounded original-input samples as well as synthetic inputs, not fewer proposals, faster time to Gold, a solved original puzzle or exact pruning.

Native experimental counters remain32-bit per lane; host sums are modular snapshots, not unlimited cumulative counters. Guidance adds at most243 orientation checks per proposal; each new scoring counter adds at most one. The app sampling phase remains capped at30seconds even when the whole run is unlimited. Direct long-running callers must account for counter wrap; all finite evidence is well below it.

Reproduce using `python benchmark_gpu_gold_scoring_round25.py --backend cuda --output audit/optimization-round25/gold-scoring-cuda-new.json` and the WebGPU equivalent. `--evidence-stage frozen` requires clean committed sources. Development receipts `gpu36-cuda-finite-01.json` and `gpu36-webgpu-finite-01.json` bind complete source/runtime/Git identities and original input bytes. Independent source, correctness and evidence review cleared this bounded implementation.


## 37. Measured replica-exchange temperature spacing

`adaptive_temperatures` is an optional hybrid sampler policy for CUDA and native WebGPU, and implies `replica_exchange`. It is available in both ordinary and durable hybrid runs. Direct callers need at least eight replicas; the app minimum already exceeds this. It remains off by default because the finite results include worse cases.

After every 16 completed 64-move dispatches, the host reads adjacent exchange attempts and acceptances across the eight-temperature islands. This covers both exchange parities. Seven rates redistribute the seven log-temperature gaps with diminishing gain `0.5/sqrt(window+1)`. Endpoints stay fixed; iterative clamp/rescale preserves positive gaps with a mathematical floor of0.05, followed by checked float32 rounding. Equal rates leave the ladder unchanged. Missing observations remain unknown and do not invent zero acceptances. There is no absolute acceptance target: a common target term would cancel under fixed-endpoint normalization. After eight eligible windows the ladder freezes and the sampler performs no further feedback reads. This is a bounded heuristic, without a stationary-distribution or exact-search claim.

Native metadata has29rows when enabled. Row28 stores float32 temperature bits and is explicitly excluded from work-counter aggregation. Both local Metropolis acceptance and replica exchange use the same stored ladder. Other policies retain their existing4/18/22/28-row layouts. The host reads only the two exchange-counter rows and eight temperature values at a feedback boundary, rather than all boards. Counter deltas use modulo2^32 and are bounded by32768 per edge/window, with accepted no greater than attempted. Native32-bit work-counter sums remain modular as described for proposal36; the bounded feedback deltas separately handle wrap correctly.

The sampler's enabled substate is version3, distinct from the outer hybrid version3 envelope. It saves native ladder bits, exact dispatch progress and a bounded16KiB, source/policy-bound controller snapshot. Restoring checks cadence, every island's ladder and host/native ladder equality. Checkpoints do not repeat completed observations. A failed native ladder write poisons the session; it cannot dispatch, report or save until a fresh session restores the last complete checkpoint. Updates consume no RNG. Before the first adjustment, native trajectories match the static ladder within each backend. Legacy and durable routes expose all seven counters: `temperature_feedback_windows`, `temperature_eligible_windows`, `temperature_adjustments`, `temperature_empty_windows`, `temperature_freezes`, `temperature_feedback_attempts` and `temperature_feedback_accepts` under `gpu_sampling.sampling_experiments`.

Twenty-six focused checks pass. Eleven pure controller tests cover160 split/resume windows, equal-rate identity, bounded clamp/rescale, fixed endpoint bits, modular wraps, unknown rates, idempotent observations, source mutation and malformed checkpoints. Native CUDA/WebGPU tests cover pre-adjustment trajectory equality, ladder-only writes, invalid per-island restore, actual partial-write poisoning, and full160-cell15-dispatch Stop plus3-dispatch resume matching18 uninterrupted dispatches. Host checks cover early small-replica rejection, exact feedback aggregation, frozen no-read behavior, ordinary/durable forwarding and live ordinary telemetry; two harness tests protect caps/adverse accounting.

The finite comparison predeclares three synthetic10-cell Gold fixtures and eight random seeds each, with eight replicas and a256-dispatch cap. The planted witness certifies each fixture but is never supplied to the sampler. Random initial populations and pre-adjustment native states match. The real16-dispatch adaptation and8-dispatch collection schedule is used; every burn-in proposal is charged. Initial Gold has zero proposal cost. Later detection is a collection-boundary upper bound, not an exact first-solution move index. CUDA and native WebGPU return the same complete results:

| Fixture seed | Solved, static/adaptive | Both-solved pairs | Both-solved lane proposals, static to adaptive | Adaptive wins/losses among both solved |
| --- | ---: | ---: | ---: | ---: |
| 2537 | 7/7 | 7 | 491,520 to208,896 | 4/0 |
| 91 | 7/8 | 7 | 348,160 to319,488 | 2/2 |
| 17 | 6/6 | 6 | 126,976 to139,264 | 1/2 |

Across all24pairs, static solves20 and adaptive21. On20pairs both solve, proposal totals are966,656 and667,648, with seven wins, four losses and nine ties. One additional adaptive run solves while its static partner caps, and three pairs both cap. All eight positive cases occur after actual adjustments. Fixture17 is an aggregate regression and remains in the report. CPU feedback/ladder calculations and transfers are extra work reported separately from GPU proposals. These synthetic results justify optional access, not a universal benefit, an original-puzzle solution or exact coverage.

Reproduce with `python benchmark_gpu_temperatures_round25.py --backend cuda --output audit/optimization-round25/temperatures-cuda-new.json` and the corresponding WebGPU command. `--evidence-stage frozen` requires clean committed sources. Development receipts are `gpu37-cuda-finite-01.json` and `gpu37-webgpu-finite-01.json`; they contain every seed, cap, adverse result, native identity and source/runtime/Git before/after guards. Independent controller, native integration and evidence review cleared this bounded opt-in implementation.
