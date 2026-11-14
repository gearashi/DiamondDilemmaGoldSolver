# Gold solver v1.9.0

The best verified incumbent used in this release matches 207 of 240 seams and has 52 line components. The target is 240 matched seams and one closed loop. Work counts below refer to the stated board, search region, or synthetic control.

## Continue the one-swap route

The exact search exhausted physical distances two through five around each of three 207-edge permutations, including 36,096,897,408 five-piece permutations per center. Every tile could rotate. Reaching 208 requires moving at least six pieces from each of these centers. The exact one-swap walk also explored sideways moves. At a floor of 207, all three reachable physical permutations completed their one-swap neighborhoods, for 38,160 transitions across the three centers. A continuing chain of one-swap moves from this plateau must first drop to 206 or below to reach 208. Run the following from the project directory with the full Python 3.12 environment.

```text
python r28_exchange_plateau.py --incumbent runtime/best.json --floor 206 --target 208 --centers 100 --database runtime/plateau-206.sqlite --output runtime/plateau-206-best.json
```

Each invocation can expand another 100 physical permutations. The same database preserves the frontier, closed centers and rotation optima. Each pair may exchange positions and all pieces may rotate. Target 208 asks for a new edge record. The independent validator reports the component count of any result.

For a different saved board, the following examines every three-piece cycle with arbitrary rotations. Reuse the same database to resume a shell or share previously computed permutation optima.

```text
python r28_exchange.py --incumbent runtime/best.json --distance 3 --target 208 --backend cuda --database runtime/exchange.sqlite
```

Use `--backend cpu` on other GPUs. On Windows, `.\venv\Scripts\python.exe` selects the installed application environment. On Linux or macOS, use `./venv/bin/python`.

## Workstreams and retained features

- **1–9.** Exact exchange shells, rotation-factor reuse, assignment bounds, dominance, tied assignments, adjacent blocks, rigid trades, component thresholds and plateau ordering are described in the [exchange measurements](R28_EXCHANGES.md).
- **10–17 and 27–30.** Physical color phases, Euler and complementary-region constraints, vertex-star census, sewn cluster packing, boundary supply, sign and switch tests, and regional join methods are covered by the [geometry measurements](R28_GEOMETRY.md). A set of 15 matched physical seams from the 207/52 board already contradicts the signed-inventory conditions for Gold. Every Gold solution must break at least one of those bonds.
- **18–26 and 35.** The existing DFS easy-remainder option now handles larger binary residuals. The logic experiments include min-cut rotation bounds, overlapping seam cores, abstraction refinement, regional interpolants, independent support, dominance, formula factoring, component caching, and justified propagation replay. See the [logic measurements](R28_LOGIC.md).
- **31.** Durable ordinary CPU repair can transplant a proven optimum onto a different frozen exterior. The key checks the excluded inside vector, physical inventory, incident masks and whether a full match remains possible. For a Gold-capable exterior, it also compares the contracted outside strand paths and accepts a fully validated Gold witness. With an unavoidable outside mismatch the frozen score is a constant offset. An original-board four-cell query required one native solve cold and zero on reuse, with the same independently checked optimum. Selectors and external cuts from the retained-service engine continue to use its own cache.
- **32.** The factor cache supports a byte budget and eviction by work saved per byte. On 24 actual nearby permutations it saved 4,486 factor-table evaluations at 16 KiB versus 4,474 for LRU. At 64 KiB both saved 4,531. The budgets count serialized operands and numeric responses. A separate SQLite spill replay and small exact offline cache oracle establish limits for the tested traces. The main elimination cache keeps its existing larger capacity. Bounded local variable-order trials also preserve pending, finished and actively leased work through conversion and restore. Four successive trials reduced a live eight-axis decision diagram from 26 reachable nodes to 17. The conversion work is reported separately.
- **33.** With **Reuse GPU seam supports** enabled in the scalar CUDA domain engine, lanes with identical seam-table rows and neighbor-domain words can share both positive and negative lookups. On 64 original-board lanes, three dispatches took 1,464,064 support-word scans independently and 45,752 with sharing. All 196,160 candidate checks and every search-state field outside the instrumentation matched. A one-lane control saved zero scans.
- **34.** **Incremental GPU checkpoints** compare exact device pages before host transfer. Snapshots still use the existing checked manifests and restore path. Across the initial capture, three dispatches and a rollback, 64 lanes transferred 60,403,848 bytes against 301,561,600 for full copies. The optimization needs a second device image when memory permits. The existing full snapshot handles insufficient VRAM. A planted lane-donation test recovered both solutions once after restore.
- **35.** A native SAT worker can reconnect to a new controller and retain learned state. The installed Glucose4 interface exposes no full CDCL-state import/export after the worker process exits. The exact saved frontier supports recovery there. See the [native SAT measurements](R28_LOGIC.md).
- **36.** Endpoint-displacement tie ordering tied the control at 207/52 on each of six saved-board seeds. The optional rejection-free sampler samples exact holding times and updates only the affected swap scores after an accepted transition. On the 207/52 board, 10,000,000 clock proposals at temperature 0.05 required 1,670,724 move evaluations and accepted 139 swaps. The final score was 207. Virtual clock ticks represent the sampling law and do not certify additional boards.

The [policy measurements](R28_POLICY_RESULTS.json) and [CUDA measurements](R28_DEVICE_RESULTS.json) include work units and control runs. Reproduce these with `bench_r28_policy.py` and `bench_r28_device.py`. Use `r28_sampling.py --help` for the rejection-free sampler or `r28_swap_walk.py --help` for the faster heuristic that rotates only the two exchanged pieces. Exact physical coverage is reported by `r28_exchange.py` and `r28_exchange_plateau.py`.
