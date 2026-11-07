# Round 26 regional workstreams

F09-F13 implement the 25 agreed regional subscopes below. The five controls are opt-in: `regional_compiler`, `recursive_regions`, `regional_proof_reuse`, `connectivity_obligations`, and `integer_regional_master`. They are integrated into DFS; the direct regional DAG encoding also runs in SAT, CP-SAT, CP-SAT circuit and persistent SAT. No original Gold solution or universal speed improvement is claimed.

The implemented mechanisms have passed bounded correctness and lifecycle checks. Final frozen evidence, whole-board ablations, hardware settings and release validation remain separate gates. A cap means that optional reasoning is unknown; it never marks the owned search region exhausted.

## F09: reusable Gold-aware compiler

| Scope | Implementation and usable effect | Correctness boundary |
| --- | --- | --- |
| F09.01 | Choose among natural, reverse and greedy orders using live pieces, seam alternatives and exposed Gold endpoints. | Heuristic order selection is not a claim of globally minimum width. |
| F09.02 | Apply GF(2) connectivity representatives while constructing each frontier layer, within identical resource/seam/endpoint classes. | Existence only: this mode publishes neither exact counts nor placement supports. |
| F09.03 | Compile inside-only envelopes; filter completed interfaces for changing outside domains and mandatory fixed chains. | Inside domains may narrow within the envelope; outside changes do not alter its compiled relation. |
| F09.04 | Encode exact DAG flow directly in SAT and CP-SAT; do not expand all histories. | Existing global inventory, seam and Gold constraints remain in force. An incomplete region adds no constraint. |
| F09.05 | Retain completed transitions, construction cursor, exact backward supports and validated witnesses; persist trusted local construction. | Exact and representative results have different types/fields. Native models replay flow encoding after rebuild, and that cost is counted. |

Implementation: [compiler](../constraint_region_compiler_round26.py), [parametric interfaces](../constraint_region_interfaces_round26.py), [native model adapter](../constraint_region_models.py).

The seven-cell, five-strand parity fixture completes infeasible under all three compiler orders. The development comparison reduces transition work from natural to selected order and again after early rank reduction. The receipt includes order evaluation, basis entries and XOR work; fewer transitions alone do not establish lower elapsed time.

## F10: recursive decomposition and correlated messages

| Scope | Implementation and usable effect | Correctness boundary |
| --- | --- | --- |
| F10.01 | Repeated separator conditioning and descendant re-decomposition, with balanced exact fallback. | Every separator case must complete before a negative result. Separator discovery and tasks are explicitly bounded. |
| F10.02 | Durable AND/OR task graph, completed descendant reuse and resumable leaf/join work. | Shared physical inventory and Gold endpoints remain explicit dependencies; disconnected cells are not automatically independent. |
| F10.03 | Multiway joins group exact seam masks, Gold pairings and closed-cycle count, then combine disjoint inventory ZDD families. | Inventory families never mix different geometric interfaces. Exact local counts, supports and witness provenance follow compatible pairs; partial messages do not certify global extendibility. |
| F10.04 | Paired-DAG overlap consistency and propagated full-join supports; query keys invalidate changed messages while unchanged inside envelopes remain reusable. | Supports come from completed correlated joins, never intersected marginal supports alone. |

Implementation: [recursive controller](../constraint_region_tree_round26.py), [symbolic joins](../constraint_region_messages_round26.py), [overlap/envelopes](../constraint_region_interfaces_round26.py).

Six-cell message tests compare the symbolic join against complete pair enumeration and independent full Gold enumeration. Split/resume execution retains identical completed symbolic work. The finite comparison separately records geometry tests, compatible inventory pairs, ZDD nodes/operations, controller steps and restores. ZDD construction can outweigh the saved geometry tests on small cases.

## F11: independently checked regional failure reuse

| Scope | Implementation and usable effect | Correctness boundary |
| --- | --- | --- |
| F11.01 | Build small inventory/Gold failure cores; a separate DSU/strand verifier checks every legal transition including dead children. | Hashes bind input bytes; they do not establish mathematical validity. |
| F11.02 | Generalize unnecessary inside assumptions and index admitted proofs by a sufficient assumption anchor. | Every remaining assumption is checked against the new domains before reuse. |
| F11.03 | Reuse under monotone domain inclusion, including smaller active residuals. | Domain widening beyond any sufficient assumption disables that proof. No equality-sized isomorphism is required. |
| F11.04 | Contract forced outside placements only after reconstructing every fixed strand, inventory obligation and boundary seam. | Arbitrary active graph contractions are not admitted: an unproved completion mapping is insufficient. This assessed restriction is intentional. |
| F11.05 | Detached proof export/import, bounded persistence and eviction; already verified identical bytes avoid repeat verification. | Novel imports always replay the independent verifier. Eviction can repeat optional work but cannot change search coverage. |

Implementation: [regional proof store](../constraint_region_proof_round26.py), using the independent [regional certificate checker](../constraint_region_certificate.py).

The finite discriminator reuses one independently verified two-cell inventory failure across four outside contexts, retaining the same proof scope while reducing repeated checker work. Producer generalization and first verification are included separately. Callback-origin exceptions propagate, including `InterruptedError`, `OSError` and `ValueError`; they are not treated as corrupt-cache fallbacks.

## F12: connectivity obligations

| Scope | Implementation and usable effect | Correctness boundary |
| --- | --- | --- |
| F12.01 | Exact maximum-weight injective cell-to-piece endpoint bounds with independently checked integer duals. | Piece uniqueness is preserved in the relaxation. All allowed assignment edges must satisfy the dual. |
| F12.02 | Exact joint component-demand DP with orientation-correlated vectors; a separate recursive check replays negative demands. | Componentwise maxima from incompatible orientations are not combined. |
| F12.03 | Refine optimistic chain routes when a selected orientation implies omitted strands. | Each refinement strengthens a necessary condition; it cannot turn a merely positive route into a Gold proof. |
| F12.04 | Joint labelled routing activates all strands of every chosen orientation. | Distinct strands of one chosen tile may coexist; the algorithm does not impose an unsound one-chain-per-tile capacity. |
| F12.05 | Reversible possible-graph connected components, bridges and cyclic-block certificates; recompute only affected components. | Unchanged components are reused only after exact edge-delta checks; snapshots restore exact graphs. |
| F12.06 | Scoped reason records with exact replay entrypoints for assignment, joint demands, blocks and all-strand routes. | Diagnostic dictionaries are not automatically trusted imports. The route checker reconstructs and exhausts the same bounded necessary-condition problem. |

Implementation: [connectivity obligations](../constraint_connectivity_round26.py).

Independent tests cover 100 small assignment systems, 120 changing/restored graphs and all completions of small actual puzzle models. A labelled fixture is feasible for the old chain relaxation but infeasible when every strand of the selected orientation is required. The stronger test can perform more work while rejecting an extra branch. Positive relaxations never satisfy the full Gold objective.

## F13: integer regional branch-and-price

| Scope | Implementation and usable effect | Correctness boundary |
| --- | --- | --- |
| F13.01 | Integral branching over multiple exact regional interfaces, retaining physical inventory and full Gold pairings. | A complete full-board witness is independently validated; fractional feasibility is not a solution. |
| F13.02 | Restricted-master column generation at integer nodes; exact branch-compatible DAG pricing and conditional Gold connectivity separation. | Floating-point LP values only propose candidates and duals. Negative decisions require exact rational Farkas checks plus complete omitted-column pricing. |
| F13.03 | Reuse validated columns, exact completed bound results, dual proposals and weighted/conditional boundary cuts. | Columns are rechecked against each branch; dual hints are not proofs; connectivity cuts condition on occupied anchors. |
| F13.04 | Candidate fixing from completed certified bounds; durable native-result and pricing cursors. | Interrupted native calls may retry and are counted. Completed native output is committed before cooperative cancellation. |
| F13.05 | Independent full-board witness validation and a separate forward signature closure audit for exhaustive integral failure. | Incomplete pricing, capped closure or missing LP support is unknown. No timeout proves infeasibility. |

Implementation: [integer master](../constraint_region_integer_round26.py), [branch-and-price](../constraint_region_branch_price_round26.py), [LP proposer](../constraint_region_master.py).

The conditional Gold cut is `cross(W) - 2*occupied(p in W) - 2*occupied(q outside W) >= -2`. Both anchors occupied require at least two crossings; absent anchors do not accidentally forbid solutions that use different ports. Tests check the cut against every independently enumerated Gold completion of a finite fixture.

OR-Tools/GLOP is optional. If unavailable or inconclusive, the integral master continues with exact regional branching and weighted bounds. The finite comparison includes that strong fallback, native model/call counts, generated/reused columns, exact pricing edges and proof replay. Native solver objects are not serialized.

## Integration, persistence and evidence

`RegionalWorkstreams` exposes `prune`, `solution`, detached cumulative `stats`, `configure_cache_dir`, `exported_certificates` and `import_certificate`. Parent DFS checks any witness against the current frame and independently validates the full objective. Statistics distinguish compilation, checking, native construction/retries, completed reuse, cap skips and storage failures; none is an estimate of legal boards exhausted.

Trusted local continuation directories are `regional-compiler-v1`, `recursive-regions-v1`, `integer-regions-v1` and `native-regional-dags-v1`; independently checked failure artifacts use `regional-cores-v1`. Atomic writes may replace an older cursor with another valid cursor; this can repeat work after a crash but cannot omit required transitions. Local continuation records are bounded and source/puzzle/domain/policy bound. They are not portable proof authority. The transitive source manifest follows local imports and adjacent C implementations, so changed ZDD, envelope, separator, pruning or verifier algorithms invalidate saved construction.

The app can stop an optional reasoning lease and move to a child before that optional query finishes. This does not imply exhaustion of the optional query. Warm sessions and configured local storage preserve it for an identical later query; bounded cache eviction or a crash before saving can repeat work.

Independent bounded mechanism reviews cleared the conditional LP cut, omitted-column replay, ZDD correlation, rank-only authority and scoped failure reuse after the source-binding and callback fixes. These reviews complement the tests; they do not certify every possible configuration.

Development validation: 59 combined tests passed before the final transitive-manifest addition (39 new regional, 8 current integration and 12 affected legacy tests). Four targeted source-binding/continuation tests then passed, including the new source test, giving 40 regional tests in the final module. Three standalone evidence-guard tests pass. These are scoped correctness results, not the final release-wide test totals.

The reproducible finite runner is:

```text
python benchmark_regions_round26.py --output audit/optimization-round26/regions-finite.json --evidence-stage frozen
```

Use `--family F09`, `F10`, `F11`, `F12` or `F13` for a single workstream. It requires a fresh JSON destination, enabled Python assertions, stable source/runtime/Git identity, and a clean commit for frozen evidence. All comparisons must finish and satisfy their independent finite oracle before a passed report is written. Final frozen receipts and original-board measurements are still required for retention/settings decisions.
