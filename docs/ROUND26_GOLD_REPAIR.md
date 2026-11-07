# Retained Gold repair models and masters

This is the Gold portion of F17, composed with the ordinary retained repair
service. `retained_repair` enables it through the repair controller; all other
existing repair modes keep their default behavior. This component does not
retire any global search region merely because a repair budget ends.

`GoldRepairPool` keeps one original-domain, assumption-free Gold circuit model
across outside assignments and across minimum-change, fragment and scoped
feasibility queries. Each call installs its own assumptions. Completed Gold
witnesses are independently checked before reuse; a proved assumption core can
answer any stronger matching query. Retained cores also become globally valid
placement-disjunction clauses in the original Gold model. Compatible validated
witnesses provide hints, which impose no extra constraints.

The minimum-change master retains its release variables and accumulated core
clauses. The fragment master retains its placement conflicts, selection variables
and learned exclusions. A completed optimum remains a valid lower bound for a
stronger minimum-change problem, or an upper bound for a stronger fragment
problem. If its chosen solution still satisfies the added constraints, it is
already optimal and needs no new native solve. Otherwise the proved bound and a
compatible feasible hint survive. Removing constraints discards the narrowed
model and its bounds before rebuilding. Base seed, fragment placements, fixed
cells and original domains are bound separately from changing constraints.

Native CP-SAT search state and native learned clauses are not serialized. Every
native solve starts native search again; UNKNOWN and interrupted queries retain
the exact semantic context and explicitly count a retry. Completed outer master
choices and repair-query results remain in the existing resumable controller.
The pool additionally preserves completed cross-controller results, model
constraints, optimum bounds, hints and pending query identities across JSON
restart. Native models are reconstructed lazily and that work is counted.

Snapshots are bounded, strictly checked trusted-local continuation records.
Their digest detects accidental corruption; it is not a mathematical proof.
Exported assumption cores continue to require independent receiving-side
verification. Positive witnesses are independently revalidated on restoration
and use. Live puzzle/source changes, fixed placements, dependency versions and
policy changes invalidate the pool. Callback errors propagate; interruption and
model limits remain unknown. Closing the pool releases its native model owners.

Defaults retain eight master contexts and 256 results/cores/pending contexts,
with a 64 MiB text-size limit per native model and a 32 MiB snapshot limit.
FIFO eviction can cause later repeated work, which is counted. Accumulated Gold
clauses have a separate bounded lifetime: reaching the retained-core limit
causes a model rebuild with the currently retained valid core family. There is
no unlimited native clause accumulation.

Counters separate native calls, branches/conflicts, UNKNOWN and interrupted
attempts, retries, cache hits, model builds/reuses/evictions, initial variable and
constraint construction, added core/master clauses, hint/bound reuse and
independent witness/master validation. Model-build attempts include aborted
construction; partial Gold-circuit term counts cannot be recovered from a
constructor that raised before returning and are not claimed as known.

## Current evidence and limitations

`test_retained_gold_repair_round26.py` checks independent Gold completions and
minimum/maximum master optima, changing contexts, widened master resets,
strict snapshots and live source changes, native UNKNOWN/Stop retries, bounded
clause/model eviction, actual controller dispatch and actual 160-cell geometry.
`test_gpu_probe_round25.py` preserves the default repair-controller behavior.
`benchmark_gold_repair_round26.finite()` is included in the common frozen F17
repair receipt; it is not a standalone timing recommendation.

The bounded development comparison completed 49 tiny Gold contexts: 49 cold
native calls versus 36 with retention and one restart. Replaying all 49 used no
new native calls. Five hitting-set contexts used five versus four native calls;
four fragment contexts used four versus three. All optima/completions were
independently checked. The actual 160-cell geometry case fixes 157 cells and
checks all 162 remaining physical/rotation assignments; it is a correctness
fixture, not progress on the original unsolved Gold puzzle. Final frozen
receipts and whole-application interaction results are recorded separately.

The ordinary retained service and adaptive patch/policy allocation are owned by
the rest of F17. These Gold-specific results do not establish the best overall
solver settings or a wall-clock improvement on the original puzzle.
