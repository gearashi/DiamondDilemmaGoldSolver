# Fresh Gold assembly and regional experiments

The 160-tile experiments use the released physical tiles on the actual mesh. Seeds 4–7 were held out from implementation tuning. Every reported physical board passed the independent placement and seam scorer. Gold additionally requires every seam and exactly one closed strand component.

## Recommended fresh search

Run this from the installed project folder with full Python 3.12. Each CPU seed receives 64 repair rounds. Zero starts allows unlimited seeds. The existing bonded constructor now keeps the eight required physical bonds and removes the 24 forbidden physical pairings, including the 21 pairs that seal small loops. Each repair returns a board respecting those restrictions. The new bonded trajectory identity starts a compatible checkpoint while preserving the earlier search data.

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --rounds 64 --starts 0 --database runtime/fresh-eight.sqlite --output runtime/fresh-eight-best.json
```

On the i7-4790 this search finished at 170, 167, 174 and 173 matched seams. The released comparison protecting seven bonds finished at 171, 175, 175 and 169 but kept 2, 4, 2 and 3 forbidden physical pairings. On the two RTX 2060 sampler starts, the eight-bond runs finished at 172 and 174 after 64 repair rounds. The seven-bond results were 175 and 169. Neither reached 208. The CUDA comparisons used 512 lanes, 256 dispatches and the same sampled archive within each seed. Concurrent host work makes timing comparisons noisy. [CPU results](R32_FRESH_CPU.json) · [CUDA results](R32_FRESH_CUDA.json)

## 23. Phase-first exact cubes and independent starts

A phase cube fixes tile zero's color convention and the first few other physical tile color bits. Width two gives four disjoint cubes covering all assignments modulo global complement. Each start solves the integer signed inventories and Euler equation, keeps the mandatory bonds and separates disconnected endpoint types. All four cubes supplied a connected phase assignment on the real tiles. These are necessary conditions for Gold.

The `cube --lift` command tests the entire supplied cube geographically, including all free phases and actual strand connectivity. A negative decision is shrunk by removing pins and rerunning the exact lift under the same physical bonds and cell domains. Only completed negative decisions enter the SQLite cache. On a planted geographic board the correct branch yielded Gold in one call. The opposite branch took three calls including shrinking and subsequently closed in zero calls. A fresh heuristic start that cannot satisfy its optional cube may retry without its optional pins. Such a start cannot close a cube.

```powershell
.\venv\Scripts\python.exe r32_fresh.py cube --cube-bits 2 --seed 1 --lift --seconds 15 --database runtime/phase-cubes.sqlite
.\venv\Scripts\python.exe r32_fresh.py fresh --backend cpu --cube-bits 2 --row-cache runtime/phase-rows.sqlite --rounds 64 --starts 0 --database runtime/phase-starts.sqlite --output runtime/phase-best.json
```

## 24. Alternate phase fitting and physical repair

The signed repair prices an independent set of geographic cells and finds the optimal distinct donor assignment for that set. Every candidate keeps the mandatory parts together. The price gives credit for two matching sides only when their colors agree under one common inventory-feasible phase vector. A bounded phase refit can improve that vector without moving any tiles. It releases optional cube pins, enforces the necessary global phase equations and proposes only connected endpoint-type assignments. The result labels the best feasible phase score and its relaxation ceiling separately. Use `--refit 8` to retry every eight rounds.

The single-phase physical repairs on seeds 4–7 reached **129, 135, 125 and 142** simultaneous signed matches, with raw totals of 132, 135, 127 and 143. At this effort, seven refit attempts per seed added no improvement to any of these four boards. Phase-first repair is available as a separate comparison mode. The stronger raw scores and eight-bond guarantee support continuing to use the circulation constructor as the recommended fresh profile.

## 25. Build the 103 odd-seam contours first

At an edge-perfect placement, exactly 103 cells of this inventory use two odd seams and the remaining 57 use zero. A frontier search places a three-bit odd-side pattern on each cell, records the unresolved geographic seams and tracks the remaining count. On restricted domains it tests a perfect physical tile matching at the end. For the odd-arc noncrossing sphere it also requires odd crossing parity on a fixed mesh path between the exceptional apices. This permits multiple disjoint parity contours. Memoization merges fully failed continuations with the same exterior seam bits, unresolved inventory obligations and apex crossing parity.

On the original inventory the search with merging reached eight distinct projected skeletons in 9,670 states. Without merging it found zero in 100,000 states under the same order. Both searches were interrupted by their respective limits. Sampling vertex colors gave four new legal parity-constrained placements in 39, 6, 24 and 28 trials, using one physical matching each. Each satisfied all geographic parity seams before bonded construction. Growing these into bonded 64-round physical candidates using `fresh --contour` resulted in 137, 139, 135 and 125 signed matches. The seed constructor and subsequent repair may change the starting skeleton. To keep its geographic restrictions throughout exact lifting use `skeleton --lift`. [Geometry and lift results](R32_FRESH_GEOMETRY.json)

```powershell
.\venv\Scripts\python.exe r32_fresh.py skeleton --nodes 100000 --limit 8
.\venv\Scripts\python.exe r32_fresh.py skeleton --contour --seed 4 --nodes 2000 --lift --seconds 5
```

## 26. Project the six reciprocal slot classes on geographic cuts

Position p on one physical side meets 12−p on the other. The slot constraint tracks separate parities for reversal orbits (1,11) through central singleton (6). Around any region the XOR of its chosen tile charges equals its boundary parity after including parity differences on failed internal seams. One Boolean shared across every projected layer pays for each failed seam in a soft-score search. Additional regional equations use those same failures.

A finite frozen placement's tight threshold agreed with an independent count of failed projected seams. On the original residual with the first 24 cells freely permutable and rotated, the one-layer and six-layer models were both inconclusive at 207 and 208 in 0.1 deterministic units. At 240 both ruled out the residual, including the control without explicit geographic equations. The projection remains an explicit residual query. A JSON request accepts per-cell legal code lists in `domains` and optional `regions` listing the corresponding cell IDs. A feasible projected assignment is lifted back to a legal board and its actual seam count is reported separately.

```powershell
.\venv\Scripts\python.exe r32_fresh.py slots --layers 0 1 2 3 4 5 --target 208 --request residual.json --effort 1
```

## 27. Reversible speculative assemblies

The constructor orders optional phase-compatible mates by available partner counts and a seeded tie breaker. It computes every geographically legal pose of a proposed joined component, tries to pack those poses with the required components and completes the donor permutation. Failure can release optional joins in reverse order, and all optional joins are free to break in future repair rounds. Seed variation changes the order. The eight globally mandatory bonds are protected throughout.

Four optional joins via `fresh --merges 4` reached signed counts of 135, 129, 130 and 134. This improved two seeds and regressed on two relative to phase repair alone. Keep speculative merges disabled in the recommended fresh profile. Exact geography can consume a saved candidate independently.

## 28. Reuse after convergence

A signed-repair price row covers every potential donor and rotation for a given ordered triple of exterior physical sides. Its key binds the exact puzzle, phase vector, prohibited mates and price policy. The cell's actual placement domain and donor inventory enter after retrieval, as does the present-pose tie breaker. Changing the seed or RNG cannot affect a row. The optional `--row-cache` SQLite file can be shared across workers.

Four independent starts at 24 rounds produced **5,249** destination-row requests. The cold shared store computed **1,260** rows and reused 3,989. Repeating the four starts computed zero rows and produced the identical boards as the uncached oracle in all 96 rounds. On the first pass no complete board recurred across different seeds, and the four seeds had no exterior row keys in common. Measured savings on that pass came from revisiting local boundaries within each trajectory. Reusing whole stochastic continuations across different seeds had no measured opportunity. The existing checkpoint saves the RNG and policy so a legitimately resumed seed can still continue at a repair batch. [Cold and warm counts](R32_FRESH_CACHE.json)

## Reproduction

The benchmark runs against an unmodified prepared inventory. Use a fresh database for each cold comparison. The CUDA sampler reuses the same prefix within that comparison and reports the corresponding proposal counts. Microbenchmarks use the same 64-round horizon. Short deterministic subsolver allowances stay visible as unknown when they expire.

```text
python -B -m unittest -v test_r32_fresh test_r31_fresh
python -B bench_r32_fresh.py cpu --tiles data/tiles.json --database runtime/check-fresh.sqlite --output cpu.json
python -B bench_r32_fresh.py cuda --seeds 4 5 --tiles data/tiles.json --database runtime/check-cuda.sqlite --output cuda.json
python -B bench_r32_fresh.py geometry --tiles data/tiles.json --database runtime/check-geometry.sqlite --output geometry.json
python -B bench_r32_fresh.py cache --rounds 24 --tiles data/tiles.json --database runtime/check-rows.sqlite --output rows.json
```
