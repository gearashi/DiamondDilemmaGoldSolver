# Exact engines on solvable boards that look like the real puzzle

This round asked two questions about the exact engines: does a different cell ordering or a different SAT model make them faster, and how far are they from solving a full board? To answer them the app gained a test board that is known to be solvable and has exactly the real puzzle's supply of side signatures. [All measured runs](ROUND37_EXACT_BENCHMARKS.json)

**Nothing here decides whether the original puzzle has a Gold solution.** Every number is a timing on a finite test instance for the edge target (all 240 seams matched, loop ignored).

## The test board

`search_fixtures.planted_inventory(source, seed)` builds a full-size board from any tile set. It takes the 240 pairs of (side signature, matching signature) that the source tiles can put on seams, deals one pair to every seam, joins the endpoints inside each cell into segments at random, and then shuffles and rotates the tiles. The new tile set has the same number of every side signature as the source, and one board with all 240 seams matched is known. With the original tiles as source the engines see the real inventory: 83 side signatures, 103 seams with an odd number of endpoints, 23 blank seams.

A tile has an even number of endpoints, so the seams with an odd-sized signature must form closed curves. The generator places them on the seams that separate a random set of mesh vertices from the rest, which gives every such arrangement. The loop structure of a planted board is arbitrary, so it is a control for seam matching only.

`benchmarks/bench_r37_planted_inventory.py` runs the engines on these boards with a chosen number of planted cells given as fixed, and on the matching negative control: the same number of fully matched cells of a near-miss board on the real tiles.

## How many given cells the engines need

Six planted boards (seeds 1 to 6), one core per run, 60 second limit. "Given" cells are picked at random from the planted board.

| Given cells | Constraint search (DFS) | SAT | SAT with tile channels | CP-SAT |
|---|---|---|---|---|
| 8 | 6 of 6, median 0.7 s | 6 of 6, median 0.8 s | 6 of 6, median 0.7 s | 6 of 6, median 10 s |
| 6 | 2 of 6 | 5 of 6, median 2.1 s | 5 of 6, median 1.2 s | 6 of 6, median 14 s |
| 4 (five boards) | 0 of 5 | 1 of 5 | 1 of 5 | 1 of 5 |
| 0 | none in 60 to 120 s | none of 3 in 25 minutes | | none in 60 to 120 s, none in 10 minutes on four cores |

Every board found matches all 240 seams and shows the planted side signatures in all 160 cells, so with 6 or 8 cells given the planted board is the only solution up to exchanging tiles that look alike. A full board of this kind is pinned down by very few cells, and the engines finish as soon as they have them. Without them none of the engines finished.

More cores did not change this. CP-SAT with four workers did not finish the three boards with 4 cells given that every engine had missed (200 seconds each), nor one board with 2 cells given or with none (10 minutes each).

The negative control behaves the same way. With 8 fully matched cells of the 216-seam board fixed on the real tiles, both engines proved in under a second that no 240-seam board contains those 8 cells (six samples, five of them by propagation alone). With 6 cells SAT finished all six samples (median 1.0 s) and the constraint search five.

## Cell ordering in the constraint search

Eight settings on the six boards with 6 cells given: the default, `ordering_goal` solution and conflict, `value_ordering`, `native_search`, a 10 ms lookahead, `signature_branching` and `impact_branching`. Each solved 2 or 3 of the 6 boards in 60 seconds. `value_ordering` and `impact_branching` each solved one board that the default did not, and no setting solved a board in clearly less time than the others on the boards they shared. The search visits about 50 nodes per second in every setting, because each node runs the full propagation. No ordering is recommended over the default on this evidence.

The morning's first exact scripts were much weaker than the app. The constraint search refutes all 81 fixed-rotation rules and the two-ring completion instances at the first node, where the scripts needed millions of nodes. The one place where the scripts were still in use was the annealing engine's completion check, which now hands over to the constraint search (see below).

## SAT models and SAT solvers

The app's SAT model has one "exactly one" group per cell and per tile and one shared literal per seam signature. It has no counting constraint over colours, so there was nothing to remove. (A script written outside the app had one, and it made a proof take 134 seconds against 14 without it.)

Encodings, boards with 4 cells given, 60 seconds: the sequential model, the tile-channel model, the cardinality-network model and CP-SAT solved 1, 1, 2 and 1 of 5. With 6 cells given the tile-channel model was a little faster than the sequential one (median 1.2 s against 2.1 s, both 5 of 6), and CP-SAT was the only engine to finish all six, at 13 to 17 seconds each.

Solvers, the same clauses (the sequential model, eight boards with 5 cells given, 60 seconds):

| Solver | Solved |
|---|---|
| Glucose 4 (the app's choice) | 4 of 8 |
| CaDiCaL 1.5.3 | 4 of 8 |
| CaDiCaL 1.9.5 | 3 of 8 |
| Glucose 4.2 | 2 of 8 |
| MergeSat 3 | 2 of 8 |
| Lingeling | 2 of 8 |
| MapleChrono | 0 of 8 |

No solver or encoding is reliably faster than the current default. The differences are within what a change of seed produces.

## What the seam score says

One default cooling run of the annealing engine (1.5 billion moves) on each planted board scored 212, 215, 212, 213, 216 and 217 of 240. On the real puzzle twelve such runs scored 210 to 216. Its best board on a planted board agreed with the planted solution in 0 or 1 of the 160 cells.

So a 216-seam board on the real puzzle is what a solvable puzzle with this inventory would also give, and such a board is not a partial solution: it shares essentially nothing with the full board. The seam score does not tell a solvable tile set from an unsolvable one, and high-scoring boards are not useful starting points for the exact engines.

## Change to the completion check

`anneal_core.exact_engine_check` asks the annealing engine's completion question of the constraint search: keep every fully matched cell outside the chosen ring, search the rest. For the two 216-seam boards it settles rings 0, 1 and 2 (136 and 140 of 160 cells free) in about 0.1 seconds each at the first node. The annealing runner now adds this two-ring check to every new best board and uses the constraint search whenever its own search runs out of budget. Three rings (152 and 158 cells free) stayed undetermined after 30 seconds.

## Limits

One machine with four cores, six to eight boards per row, 60 second limits: the tables separate "finishes at once" from "does not finish", and nothing finer. The planted boards share the real inventory of side signatures but not the real tiles, and their loops are arbitrary.
