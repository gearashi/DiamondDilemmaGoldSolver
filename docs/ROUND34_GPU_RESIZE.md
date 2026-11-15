# Native GPU domain lane resizing

**Workstream 33 retained.** Lowering the requested replica count on a compatible native CUDA domain checkpoint restores that many lane columns to the GPU. The remaining native columns wait in compressed host archives. Their decision frames, rollback trails, learned clauses, scan cursors and unfinished siblings survive. Raising the request again can combine these columns into a larger batch. The existing GPU memory and replica limits still govern allocations.

The portable frontier includes every parked remainder. Native counters include work already performed by those lanes, exactly once. Donated owner groups retain their parked descendants. A propagated parent is eligible for coverage publication after its final child batch finishes. A pause between batches can have no current GPU engine and still contain parked work. Changing search policy takes the existing portable restart path. A checksum or mechanical state error rejects corrupted native columns.

## Original-puzzle comparison

The RTX 2060 reduced four original-board lanes to one. Each lane was run separately to the same independently recomputed propagation fixed point as a fresh portable restart. The native path avoided **81,748** support checks at a checkpoint with no removals and **38,294** at a later checkpoint with 437 removals. All eight matched fixed-point comparisons agreed. These are counts for recovering existing progress, not a measurement of faster unconstrained Gold search or a prediction of time to solution.

## Exact controls

A 4→1→2→8 sequence recovered all six finite edge solutions, both Gold solutions after a propagated fork, and no solution on the fixture with an actual donor transfer. Every finite result set was checked by independent enumeration. A factored incremental generation recovered the same six solutions after resizing. Unchanged native layouts preserved state bytes. Factored generations preserve the live state and normalize unused padding. Switching to joint propagation preserved the portable frontier and accumulated decisions. Host tests also pause immediately after a batch retires, reopen the parked-only checkpoint, and check corrupt counters, wrong bindings and altered native checksums.

Run `python bench_r34_lane_resize.py --tiles ../data/tiles.json --output docs/R34_NATIVE_LANE_RESIZE.json` on a CUDA host to reproduce the [per-lane counts and results](R34_NATIVE_LANE_RESIZE.json). No additional dashboard setting is required. Use the existing replica selector when resuming a compatible domain search.
