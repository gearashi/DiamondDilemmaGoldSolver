# Search algorithm comparison

The target is the actual Gold Challenge: all 160 physical tiles used once, all 240 seams matching, and one closed gold loop. Drawn line crossings do not create connections. A high edge score or several closed loops is not a solution.

## Choosing a mode

The dashboard defaults to **DFS with constraint pruning** because it combines puzzle-specific early contradiction checks with exact saved branch cursors. This is a practical default, not a claim that it is the fastest solver of the real Gold instance. The bounded comparison has not established a speed winner. All contenders remain selectable.

| Mode | What helps it | Main cost or limitation | Resume behavior |
| --- | --- | --- | --- |
| Strong DFS | Minimum-domain choice; seam propagation; forced clusters; global tile matching; face budgets; learned conflicts; early loop pruning | Python propagation can cost more per node; difficult branches may remain huge | Exact pending branches and choice cursors |
| Traditional CP | Native finite-domain propagation, AllDifferent, seam tables, minimum-domain search | Gold connectivity is checked on complete candidates, then excluded by component cuts | Validated cuts/hints survive; internal search restarts |
| CP-SAT | Native integer constraints, presolve and learned conflicts | Model and presolve overhead; global loop cuts require additional solves | Validated cuts/hints survive; internal search restarts |
| Experimental CP-SAT direct Gold circuit | One native circuit constraint includes every chosen segment and seam join; one-loop reasoning during search | Larger model; construction and presolve cost; explicit resource caps | Rebuilds the model; native search restarts |
| SAT (Glucose) | Boolean conflict learning; compact seam-signature support encoding | Boolean model construction and cardinality constraints; additional loop cuts | Validated cuts/hints survive; learned native state restarts |
| Experimental GPU sampling + repair + DFS | GPU and saved-board seeds feed CPU neighborhood repair, then smart DFS tries their choices first | Heuristic startup consumes time; hints can mislead ordering; repair is not exhaustive coverage | Exact DFS stack, with original hints; no resampling on resume |
| Existing GPU DFS | Many disjoint branches; inexpensive local seam checks | Static cell order and weaker pruning; the app retains its first-240-edge stopping rule | Existing GPU frontier and active/paused stacks |

The new Gold-only modes reject edge-perfect boards containing multiple loops. Their exclusions preserve completeness: DFS rejects a closed proper loop; ordinary native engines forbid the conjunction of placements that preserves that same closed component. The experimental direct circuit engine instead enforces connectedness in its model and independently validates every returned board. GPU samples never remove domain values or force placements. The hybrid can also accept a sampled board immediately if independent validation proves it is already a Gold solution.

## Bounded measurements

The October 12 optimization checks measured an earlier 2.73× median gain for equivalent seam propagation and a subsequent 1.63× same-work gain for the combined CPU solver over its already improved predecessor. Bounded sampler throughput also improved 16–43% across the tested native backends and replica counts. These are separate workloads and their ratios must not be multiplied. They do not identify a fastest complete solver. See [the exact scope and optimization results](OPTIMIZATIONS.md).

These measurements predate the smart-pruning and neighborhood-repair changes. They remain historical evidence for the earlier implementation and do not benchmark the enhanced modes.

Recorded on 2026-10-09 with an Intel Core i7-4790 and NVIDIA RTX 2060 (6 GB), Windows, Python 3.12.14, NumPy 2.5.3, OR-Tools 9.15.6755, PySAT 1.9.dev15, CuPy 14.2.0, and WGPU 0.32.0. Two search seeds (71, 197), three full-board instances, and six engines gave **36 runs**. Every engine timed out on all six of its runs; none produced a solution or an infeasibility result. There were no worker errors. **No speed winner was established.**

The requested engine budget was 10 seconds per run. Actual wall times include setup and any native-call overrun. These timeout durations are not solve-time comparisons. The generated boards each have 366 gold segments; the real encoded puzzle has 365.

| Engine | Validated solves | Timeouts | Actual wall seconds (min–max) |
| --- | --- | --- | --- |
| gpu-dfs | 0 / 6 | 6 | 10.72–10.98 |
| dfs | 0 / 6 | 6 | 10.56–10.72 |
| cp | 0 / 6 | 6 | 10.62–11.42 |
| cp-sat | 0 / 6 | 6 | 9.64–12.36 |
| sat | 0 / 6 | 6 | 11.09–16.69 |
| hybrid | 0 / 6 | 6 | 10.67–10.97 |

[Per-run timings and input hashes](search-comparison-results.csv) are included in the source repository. The full local evidence record is `audit/search-comparison/manifest.json`; load-bearing source hashes were unchanged during the run. The small independent brute-force tests exercise successful solutions and exhausted cases; the bounded full-board experiment did neither.

The reproducible harness is `benchmark_search.py`. It runs engines serially in fresh subprocesses on the real encoded tiles and two generated 160-tile instances whose planted single-loop solutions are independently validated. It uses the same target, seeds, limits, and 128 replicas for both GPU contenders, with one worker per CPU engine. In the harness, legacy GPU DFS continues after multiloop boards so its objective matches the other engines.

Elapsed wall time includes process startup, model creation, GPU initialization, sampling, and validation. Engines receive a search budget; an external guard adds 30 seconds for initialization/shutdown and records a hard timeout if exceeded. A native call may return after its requested time limit, so actual wall time is recorded. Node counters differ by engine and cannot be ranked as equivalent throughput. Generated tile distributions differ from the actual puzzle; even a synthetic solve winner would not prove which engine solves real Gold fastest.

Run locally after preparing data and installing the selected GPU packages:

```bash
venv/bin/python benchmark_search.py --data data/tiles.json --seconds 10 --seeds 71 197 --replicas 128
```

On Windows use `venv\Scripts\python.exe`. Results, input hashes, fixture seeds, software versions, source hashes, and the computation manifest are saved under ignored `audit/search-comparison/`. They are separate from all existing search checkpoints. A timeout is inconclusive, not proof of impossibility. Integrity digests detect accidental checkpoint damage; a modified checkpoint is not an independently verifiable infeasibility certificate.

## Direct circuit validation

The selectable `cp-sat-circuit` engine was added after the earlier six-engine comparison and has not been established as faster than DFS or ordinary CP-SAT. Its optional endpoint nodes preserve separate strands at crossings, force every selected segment into the circuit, and connect opposite edge positions in reversed order. OR-Tools then requires a single cycle over active ports. It retains the existing proper whole-board rotation reduction and removes the circuit's two equivalent traversal directions.

Finite checks compare complete solution sets on eleven tiny fixtures against independent brute-force arrangement validation, including fixed pieces, open boundaries, internal pairings with equal edge masks, and six seeded four-tile variations. A disconnected multiloop fixture is proved infeasible directly, before any complete candidate or lazy Gold cut. Planted 10-, 40-, and 160-cell certificates are accepted and independently validated; the 40- and 160-cell tests restrict each cell to two orientations of its assigned physical tile. Those tests validate large-board encoding, not unrestricted solving speed.

The current real input builds a valid model with 76,368 placement candidates, 5,280 potential port nodes, 147,840 directed arcs, 267,616 variables, and 122,977 constraints after the existing root symmetry restriction. No actual Gold search was run as part of this model-construction check. Identical or complementary support groups can share literals on synthetic inputs; the current real input yielded zero such reuse and no model-size reduction from that option. Exploratory construction timings overlapped other CPU checks and are not used to claim a gain.

Model construction caps are 160 cells, 80,000 placements, 200,000 arcs, and 1,000,000 linking terms. Exceeding any cap reports unsupported/incomplete, and interrupted work is never an infeasibility result. The separate saved folder is `runtime/searches/cp-sat-circuit-gold/`. Like other native modes, it restarts internal search on resume. In the API's edge-only mode it explicitly omits the single-loop requirement; app runs always target Gold.

The bounded evidence and reproduction script are stored locally under `audit/optimization-round6/circuit/`; the script records source and input hashes. The cross-platform CPU test suite includes the direct circuit tests. The benchmark harness also accepts `--methods cp-sat-circuit` for future controlled comparisons.

## Sources and implementation scope

- [Jaap's puzzle description](https://www.jaapsch.net/puzzles/diamdil.htm) defines the Gold objective.
- [OR-Tools CP-SAT guide](https://developers.google.com/optimization/cp/cp_solver) documents integer constraints and the distinction between feasible, infeasible, and unknown outcomes.
- [OR-Tools pinned AddCircuit source and contract](https://raw.githubusercontent.com/google/or-tools/v9.15/ortools/sat/python/cp_model.py) specifies one active circuit with self-loops for inactive nodes.
- [OR-Tools traditional CP Python reference](https://or-tools.github.io/docs/python/namespaceortools_1_1constraint__solver_1_1pywrapcp.html) documents the native constraint solver.
- [PySAT solver API](https://pysathq.github.io/docs/html/api/solvers.html) documents Glucose and interruption support.

These sources describe capabilities; the speed conclusions here depend on the recorded bounded experiments, not on general claims that an algorithm is always best. No Eternity II solver files are used.

## Additional exact-search engines and controls

DFS now includes optional failure-guided ordering, bounded singleton look-ahead, strand connectivity, correlated seam cuts, and complete local enumeration caches for closed-inventory 2-4-cell regions. A source-verified C99 propagation library accelerates the same exact filtering when available. These controls also apply to the exact phase of hybrid search.

Parallel CPU DFS partitions pending branches into a durable disjoint queue and supports changes in worker count on resume. This is an additional execution mode, not a demonstrated winner in the earlier equal-budget comparison. Its tests verify finite coverage and crash recovery. Use a modest worker count initially; repeated process setup and memory costs can dominate short jobs. See [implementation bounds and measured results](OPTIMIZATIONS.md).
