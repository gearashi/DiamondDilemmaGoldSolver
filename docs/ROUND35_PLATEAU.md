# Resumable one-swap plateau searches

The fixed-permutation rotation solver scores a physical tile swap over its allowed rotations. A plateau route keeps every intermediate optimum at or above its score floor. The objective here is matched edges. The independent validator reports separately whether a 240-edge placement forms one circuit.

## Candidate checkpoints, #42

The normal plateau command now checkpoints 512 pair-star bounds per page and each completed swap candidate. A fenced SQLite transaction records the candidate cursor, exact optimum and discovered directed edges together. Each center owns one band at a time. Its minimum covered floor advances only after the complete band. A resumed process can use another depth limit or a lower floor. Historical score-only optima recover a witness on demand. A historical closed center proves membership at its recorded floor.

On the shipped 207/52 center at floor 207, the benchmark killed each child process inside its eighth completed rotation, immediately before publication. The old single-transaction expansion replayed all eight optima and all 12,720 pair stars. The candidate checkpoint retained seven optima and all star pages. It replayed one optimum. The cursor still reported seven committed candidates and an open band. Total rotations were 95 and 88 respectively. Both completed runs yielded 87 independently scored candidate neighbors, including the same two physically distinct neighbors at floor 207. Their ordered graph digest is recorded in R35_PLATEAU.json.

The cutoff regression places a single-loop target on the last allowed pair. Resuming with one remaining candidate returns that target in one swap. Other controls interrupt a partially priced star vector, revoke ownership inside a rotation and race two distinct tied witnesses with the same score. Partial snapshots resume at the saved boundary. The stronger witness and tie count survive a late write. A cold root deadline returns the independently scored starting board with initialization pending. The measured 0.05, 0.25 and 1-second calls returned in 0.063, 0.265 and 1.625 seconds on the test machine. Model construction, an active numerical factor, native solver calls and SQLite lock acquisition can extend a soft deadline.

A separate experiment prefetched two or four same-depth neighbors while serializing parent publication. All four planted starts returned the same physical shortest path and witness. The serial, two-worker and four-worker runs solved 51, 61 and 72 exact rotations respectively across those starts. Serial evaluation remains the production scheduling choice.

## Lossless star transport, #43

Each new center can import its saved predecessor's exact doubled pair deltas outside the two-hop neighborhood of the exchanged positions. Within that fringe it recomputes the affected stars. Doubling preserves parity when adding the child's changed baseline. New vectors and partial pages store signed-byte deltas. A previous floor-only vector remains usable at its own center. When the predecessor has no compatible lossless vector, the child computes its own.

The original 207/52 center had 71 distinct one-swap neighbors scoring at least 206. Twelve parents/children spread across that neighbor list gave 152,640 independently recomputed pair bounds. The saved parent eliminated 118,328 pair calculations and recomputed 34,312. All exact doubled values agreed. Transferring floored differences would misprice 16,047 pairs across four odd-baseline children. The twelve new vectors used 68,939 compressed bytes. The legacy floor-only vectors used 67,134 bytes and the explicit floor-plus-parity encoding used 86,346. The parent vector was 5,721 compressed bytes, read as one indexed blob per new child. These counts exclude SQLite page/index overhead.

The earlier two-center in-memory support method independently agreed on all 1,536 sampled original pairs, including 1,163 whole-support reuses. It priced each of the 1,536 parent supports for the two-center comparison; the durable representation can open that saved parent directly on a later run. Four further planted parent/child swaps, one pinned tile and a forbidden donor independently checked the saved vectors over all 741 movable pairs. Rotation ties have the same physical key.

## Incremental widest-floor routes, #44

Pass `--incremental-floors` to retain shortest paths across descending floors. The command indexes directed arcs by exact destination score, wakes a center when a lower star band or scored edge can enter, and sends a newly shorter or lexicographically earlier path through its affected descendants. A shortened path restores the associated remaining swap budget. Old scored arcs are indexed on first access, and later arcs enter the index transactionally. A candidate hit can be marked feasible before a shorter pending path has been settled.

On the original 207/52 center, a one-swap search from floor 207 through 205 found 443 qualifying neighbors and no 208. Both algorithms evaluated 2,143 candidate swaps and 2,144 exact optima including the root. The incremental traversal reduced center closures from 519 to 446, directed neighbor rows read from 5,506 to 443, and stored distance rows from 519 to 444. The current fixed-floor traversal also stages component ties before committing parents, which reads its candidate list a second time. Counting only one of its two scans gives 2,753 rows. Both completed resumes processed zero additional centers or candidates.

Four independent planted seeds at depth one reduced closures from 256 to 31 and directed rows read from 440 to 24, for the same 28 exact optima. The eight depth-two and unrestricted positive runs returned the same validated two-swap witnesses, with no change in exact rotation count. A separate ten-tile fixture evaluated all 24 free tile permutations and their 81 rotations per permutation. A newly enabled bridge shortened one existing shortest route from four swaps to two. Every reachable physical vertex and canonical shortest path agreed with independent BFS under caps 0, 2, 3 and 4, including interrupted one-center-at-a-time resumes.

## Running the route

From the installed project in PowerShell, use this bounded one-swap comparison or set a larger swap limit for a different route scope.

```powershell
.\venv\Scripts\python.exe r28_exchange_plateau.py --incumbent docs/R29_INCUMBENT.json --database runtime/plateau-207.sqlite --incremental-floors --floor 205 --target 208 --max-swaps 1 --ties 1 --centers 1000 --seconds 0
```

Repeat the same command to resume. `--candidate-limit N` limits newly committed swap candidates in a call. `--cold-stars` supplies the parent-transport ablation. The original floor-by-floor queue is available through `--widest`. Fresh database paths give independent benchmark runs.

`bench_r35_plateau_pages.py` performs the forced process kill. `bench_r35_plateau_vectors.py` rebuilds all sampled original vectors. `bench_r35_plateau_controls.py` compares two-center pricing and measures stop behavior. `bench_r35_plateau_frontier.py` compares speculative worker counts. `bench_r35_plateau_floors.py` produces the actual and planted traversal comparisons. Each script provides its path arguments through `--help`. The quantitative receipts are in `docs/R35_PLATEAU.json`. Run `python -B -m unittest test_r29_plateau test_r35_plateau_pages test_r35_plateau_floors` for the integration and finite proof controls.
