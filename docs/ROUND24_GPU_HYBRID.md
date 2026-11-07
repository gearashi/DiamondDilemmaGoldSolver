# Round 24 CUDA domains and hybrid experiments

These additions are opt-in. Finite correctness and discriminator results do not
establish a faster solution of the original Gold puzzle. Every saved Gold board
still needs all 240 matched seams, unique physical tiles, and one complete loop.

## Durable CUDA domain engine

Select **CUDA domains + learned backjumping** in the dashboard. The default is
128 lanes. Unlike compact GPU DFS, each lane holds complete domains, removal and
reason trails, reversible strand connectivity, and scoped learned clauses.
The maximum is 512 lanes within a 512 MiB state budget and 75% of free VRAM. On the
160-cell puzzle, 128 lanes use 120,617,472 bytes; 512 use 482,469,888 bytes of search
state, plus the small immutable tables and CUDA runtime. This experimental
engine currently requires NVIDIA CUDA. The original CUDA/WebGPU engines remain
available for AMD, Apple and larger compact-lane counts.

```
python constraint_search.py --algorithm gpu-domains --replicas 128 --seconds 0 --resume
```

Zero means unlimited in the application runner. Stop saves native
GPU domains, unfinished scans, rollback state, choices and learned clauses.
Checkpoint JSON contains a bounded, checksummed compressed array. Native local
checkpoints and completed coverage are trusted progress, not external proofs.
A checksum does not authenticate an adversarially edited checkpoint.

Changing to fewer lanes restores exact remaining domain cubes and reconstructs
the native scratch; this preserves work ownership but redoes current inference.
A batch retains its lanes until that batch finishes. Faster lanes are not yet
refilled while another lane in the same batch remains active. The UI reports
active/completed lanes instead of implying that every requested lane is busy.
All lanes stop when the first independently validated Gold board is returned.
No solution is acknowledged as an infeasibility exclusion in application mode.

The checkpoint's `frontier` is also written to `remaining-work.json`. Import it
into a separate output directory using `--work-frontier PATH`. Supported
receivers are `gpu-domains`, `parallel-dfs`, and `sat` with `persistent_sat:true`.
CPU↔GPU transport uses disjoint Cartesian domain cubes with matching puzzle,
objective and original domain scope. Imported unfinished work is a task scope;
finishing an imported subset alone does not prove the entire original puzzle
impossible. CPU exports use `constraint_portable.export_frontier`.

`shared_coverage:true` records completed infeasible lane roots in a transactional
local SQLite ledger. Incomplete lanes and any lane with enumerated solutions do
not enter that ledger. Enumeration acknowledgements exist only for finite tests.

## Features and evidence

| Proposal | Implementation and finite evidence | Scope/limitation |
| --- | --- | --- |
|10|Reversible per-lane strand union state; same complete tiny Gold solution sets as rebuild mode, fewer strand joins.|New CUDA domain engine; WebGPU compact engine unchanged.|
|28|Physical-piece present/absent domain partitions; independently enumerated completions belong to exactly one child.|Ordering/partition policy; no assumption that rarer pieces always solve sooner.|
|29|Continuation branches distinguish internal strand destinations even when every side mask is identical.|Exact exhaustive mask partitions; optional CPU DFS.|
|30|Small-region exact completion counts reorder choices; counts match independent enumeration.|Capped counting retains every uncounted choice; no pruning.|
|31|Five-feature ridge ranking implementation and validation.|Frozen weights fitted on 46 completed child searches. Disjoint tiny held-out tier: 133 baseline versus 119 learned nodes; larger capped fixtures inconclusive. Optional only; see ROUND24_ORDERING.md.|
|33|Lane-local reason masks, learned placement clauses and nonchronological backjumping.|Finite Hall instance: 42 decisions / 757 support checks with learning versus 166 / 2852 without; both complete UNSAT. Clauses stay within their lane root.|
|34|Complete durable domains, code-removal trail, reason snapshots and resumable propagation scans.|Memory-bounded alternative architecture; not 131072 full-domain lanes.|
|35|Bounded CPU-probe ranking by piece-shortage evidence.|160-cell discriminator isolates four cells competing for three pieces; selected CPU query independently proves failure. Ranking itself never prunes.|
|36|Assumption-core hitting-set repair finds minimum changed cells.|Tiny exhaustive comparison proves minimum Hamming distance. Unknown/capped native solves never establish infeasibility.|
|37|Exact patch trades preserve outside placements while changing internal seams and reducing loop components.|Two-cell discriminator merges three loops into Gold. Requires an edge-perfect legal input.|
|38|Matched fragment set packing followed by exact inventory and Gold completion.|Tiny crossover uses fragments from both parents and independently validates Gold.|
|39|Repair queries are feasibility-only under explicit outside assumptions; stored cores are scoped.|With `smart_learning`, repair claims enter independent original-root verification before DFS use. A regression queues one true and one false claim; only the true claim is admitted.|
|40|Closed 3–6-cell relocation chains with independently chosen rotations.|Heuristic sampling, no coverage proof.|
|41|Short move-attribute tabu memory with aspiration for a better lane best.|Heuristic sampling, may still revisit boards.|
|42|Metropolis exchange of current states between fixed temperature lanes.|Temperature, RNG and iteration belong to the lane; current board/score move together. Best archives remain per-lane.|

All 7 nonempty sampling option combinations, with Gold scoring on and off, passed
native CUDA and D3D12 tests. Every current and best board was checked for physical
inventory, fixed cells, and an independently recalculated score after 1, 7 and 64
steps. Equal-energy exchanges were checked for actual board swaps without moving
the temperature-lane state. These are correctness results, not measured claims
that each heuristic improves original-puzzle progress.

Repair cores from the pinned OR-Tools version are requested with presolve off,
because a tested circuit case returned presolved rather than original assumption
IDs with presolve on. Any unmappable returned core conservatively falls back to
the entire proved-infeasible assumption set; it never guesses literal IDs.

## Reproducing the evidence

`test_gpu_domain_round24.py` fully enumerates tiny independent reference cases,
checks every primitive's disjoint remaining work, resumes native checkpoints,
and checks 160-cell-deep pending branch volumes. The host checkpoint audit also
rejects cycles, invalid rollback pointers, padding bits and removal-trail
inconsistency. `test_gpu_domain_runner.py` covers Stop, native JSON resume,
changed lane counts, CPU→GPU→CPU transport, coverage reuse and failed-restore
cleanup. Hardware tests require `DIAMOND_TEST_CUDA=1`;
sampling WebGPU tests additionally require `DIAMOND_TEST_WEBGPU=1`.

`benchmark_hybrid_round24.py` compares all 8 sampling configurations using paired
seeds and equal proposed moves. Example:

```
python -B benchmark_hybrid_round24.py --data data/tiles.json --output audit/sampling.json --steps 1024 --replicas 128 --seeds 1,2,3,4 --backend cuda
```

Repeat with `--gold` to include loop-aware sampling scores. Its archive-repeat
counter samples reported best boards; it is not a count of all repeated proposed
boards. Experimental chain paths currently rescore the entire board, while the
legacy sampler has a delta-score path. Equal proposals therefore do not mean
equal computational work. Report both proposals and time, and compare solution
quality/Gold success across seeds before claiming a heuristic gain.

Original 512-lane startup/Stop smoke is recorded in
`audit/round24-domain-original-smoke.json`. Dispatches through 1024 primitives
completed without driver timeout. Host preparation improved from 19.81 s total to
0.079 s partitioning plus 2.274 s construction by caching root masks, using a heap
for partition splitting, and vectorizing immutable compatibility-table packing.
These timings were correctness/startup smoke under concurrent CPU activity, not
a controlled throughput benchmark. The full 512-lane checkpoint took 2.80 s and
restore 4.02 s in that smoke. Live preview copies only board rows, not the whole
460 MiB reasoning state.
