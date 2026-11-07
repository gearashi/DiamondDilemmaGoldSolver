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
