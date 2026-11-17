# Shortest physical routes to 208

All three known 207 centers require **at least four swaps** to reach 208. A swap exchanges two physical pieces. The search allows unrestricted rotations of all 160 pieces and any intermediate score. Each completed three-swap shell below contained **317,898,812,400** physical endpoints before pruning. Neighborhoods of different centers overlap.

| Center SHA-256 prefix | Full triples considered | Remaining after physical-seam bound | Exact rotation solves needed | 208 hits | Cold wall time |
|---|---:|---:|---:|---:|---:|
| d297c8f5cbdf792d (207/52) | 41,199,139 | 38,300,529 | 0 | 0 | 7.69 s |
| 51c26e4e47abf28f | 7,505,395 | 6,829,355 | 0 | 0 | 3.00 s |
| 840affdf318119d3 | 9,399,450 | 8,516,945 | 0 | 0 | 3.28 s |

The endpoint with six moved cells at swap distance three must consist of three disjoint two-cycles. For each six-cell set there are 15 such permutations. Every other endpoint within three swaps moves at most five cells and falls under the published exhaustive bounds, including the unchanged center. The lowest intermediate score can fall freely because every terminal permutation is considered regardless of path. The remaining matched-edge threshold of 208 is distinct from the Gold objective of 240 and one connected loop. [Earlier radius-five evidence](R28_EXCHANGES.md) · [New machine-readable results](R29_EXCHANGE_RESULTS.json)

## How the screens work

1. For each possible swap, count how many physical seams could match if each incident tile could rotate independently at each seam. A seam depends on at most two disjoint swaps. Its exact contribution to this relaxed score is thus a constant, two single-swap gains and one pair correction. Classify a triple by the number of positive corrections. With zero positive corrections, sort the three single gains. With one, start at that pair. With two or three, traverse the common vertex's adjacency list in descending pair price. For a positive triangle only the least-indexed vertex owns the triple. Every discarded suffix has an upper bound below the requested score.
2. A second bound allows the three neighbors of each central tile to rotate independently but gives the central tile a consistent rotation on all three seams. Twice the actual score cannot exceed the sum of these star bounds. Price single substitutions once. Wherever multiple changed cells affect the same star, subtract their individual prices and recompute that star using all changes simultaneously. This handles intersecting pairs and larger connected groups. In the three real searches the second bound and previously proved regions together closed every survivor of the first bound.
3. Across a group of changed cells with disjoint incident stars, all substitution prices add exactly. A global assignment relaxation with verified integer duals had a doubled upper bound of 416 at each of the three centers. For each center there were just three assignments needing an exact rotational optimum. Their values were 206, 206 and 207. Thus any improvement from these centers must move at least two cells that are adjacent or share a neighbor. The certificate generator also explores positive integer slack when needed and leaves the region open when a leaf budget is reached. Center permutations, complete domain fingerprint and duals are stored in the release. The benchmark recomputes them. [Checked centers](R29_EXCHANGE_CENTERS.json)
4. The portable search assigns each support to its lexicographically first interacting pair where that separation proof applies. It prices an entire support before generating any donor order. For a star hit by several substitutions, each possible central donor chooses its best of three rotations against optimistic neighboring donors. A checked assignment dual then bounds all internal donor permutations at once. On seven tested support sizes from 3 to 15, this closed 448/448 randomly sampled supports and 447/448 connected supports. A paired four-swap, seven-cell control needed 2,520 donor bounds on twelve supports with the screen disabled. The screen closed all twelve supports before any donor pricing.
5. A completed negative search owns its neighborhood for the same or stronger threshold and exactly matching physical domains. The native search can also skip a two-swap prefix when even changing two more cells leaves the result inside an owned ball. The final runs skipped 1,931, 1,211 and 1,206 such prefixes respectively. The portable search checks ownership for an entire support and again for each remaining donor assignment. Ownership extends to proper rotations of the mesh that preserve every orientation domain. Compatible exact rotation results reuse the v1.9.0 physical-permutation cache. Each SQLite transaction commits a page and its exact results together.

## Running from the solver directory

These commands use the project's Python environment and prepared tiles. The native three-swap search takes roughly 170 MB for swap-pair corrections on the standard board. It builds a local C99 library with GCC or Clang. On a computer without a compiler, pass `--backend cpu` for the portable version. Larger physical supports use the portable backend.

```sh
python r29_exchange_route.py --incumbent runtime/best.json --swaps 3 --moved 6 --target 208 --backend native --database routes.sqlite
python r29_exchange_route.py --incumbent runtime/best.json --swaps 4 --moved 6 --target 208 --backend cpu --database routes.sqlite --path
python r29_exchange_route.py --incumbent runtime/best.json --swaps 4 --moved 7 --target 208 --backend cpu --database routes.sqlite --path
python r29_exchange_route.py --incumbent runtime/best.json --swaps 4 --moved 8 --target 208 --backend cpu --database routes.sqlite --path
```

For a different center, also search the five-moved-cell support at four swaps. At the three published centers the radius-five proof already covers it. Each `--moved` value is its own scope. Exhausting one support size alone does not exhaust four swaps. The certificate permits a generic center to be searched with full support enumeration when its fingerprint or domain differs. Set `--seconds 0` for unlimited time or supply a positive budget for a resumable session. Search cursors count owner swaps, interacting-pair pages or ordinary support ranks as stated in the result. They do not count endpoints individually. `--path` reconstructs an exact shortest swap route to a verified endpoint and maximizes its minimum intermediate rotation optimum.

To regenerate the evidence, supply a fresh database path to the benchmark. The planted controls compare complete native survivor sets with independent full-star evaluation, including ownership and a deliberately undersized output buffer. The suite checks exact-cost cycle generation, cluster bounds, truncated certificates, resumption and path recovery.

```sh
python -m unittest -v test_r29_exchange
python -m benchmarks.bench_r29_exchange --database fresh-r29.sqlite --output r29-results.json
```
