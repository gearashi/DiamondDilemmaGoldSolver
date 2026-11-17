# Oriented reference distance and physical tile ownership

Every geographic Gold board must change at least 95 oriented tile placements from the authenticated 160-cell reference. Each proper rotation therefore permits at most 65 exact reference poses. The geographic search now uses the one-copy-per-tile rule to detect violations even when every cell still allows multiple orientations. This check is part of `python r30_gold.py embed --reference-bound`. It does not apply the distance theorem to an ordinary 208-edge score search.

The strongest independent original case used two overlapping donor banks. Even after full-width physical seam propagation, compiled seam propagation, complete ordinary tile matching, direct pose clauses and image-pair counts reached a fixed point, its narrowest cell retained 95 codes. Its assigned-tile lower bound was 75, ten over the limit. Before the public geographic search received the owner filter, the same query remained open after 327,049 branches in eight seconds. The enabled owner filter rejected it before native search. Seven of fourteen separate one-bank pools survived the full existing fixed point and were subsequently excluded by the distance bound. Six needed only the donor-prefix count. One needed a weighted assignment. A held-out two-bank pool remained feasible in this relaxation and lost ten candidate codes at a lower bound of 63. The pools impose reproducible restrictions on the real original piece inventory; they do not represent unrestricted boards or Gold solutions. [Full controls and scope](R35_POSE_MATCHING.json)

## How the bound works

For each reference image, a cell can cost one only when assigned its own reference tile in its exact reference orientation. If another orientation of that same tile is legal, choosing the alternative costs zero. Rows that can never force a reference pose are identified before loading the proof on broad inputs. A count of distinct available donor tiles in the cheapest row prefixes can already prove that too many reference poses must stay.

For the remaining cases, each allowed cell-to-tile edge receives weight one for a zero-cost move or zero when the reference pose is required. An integer assignment dual bounds the maximum possible total weight. Subtracting that value from the number of assigned cells gives the minimum number of retained poses in this relaxation. A row with at least as many zero-cost tiles as remaining rows can always extend an assignment of the others and is omitted. Independent connected row/tile components are priced separately. Fixing a candidate consumes its tile column and, within an active component, its row as well. The corresponding dual prices give a lower bound conditional on that individual oriented code.

The solver retains 16 recent immutable image/domain responses and 64 exact numeric assignment matrices of at most 256 entries. Large matrices and temporary matching graphs are released. On a tight 22-component case, two distinct matrices and 41 dual checks sufficed; twenty congruent components reused their checked answers. The 20 prior joint-contact, branch-child and contour states used zero weighted assignments. The broad full-tile and two-orientation controls also used none.

The outer geographic loop reapplies full physical seam and tile matching support after any removal. On a held-out tight case, a duplicate inner pass removed ten more codes immediately. Both continuations already eliminated the case on the next ordinary propagation pass. The duplicate inner pass was removed. Exact alternating-path prices at positive slack reduced small artificial pools, but all affected sparse pools already failed compiled seam propagation. Of 42 dense additional comparisons, 34 survived the compiled and ordinary matching controls, and none gained a further code from enabling the more costly unbounded margin calculation. Single-image searches keep the checked dual and use the shared geographic propagation loop after changes.

## Reproducing the controls

From the installed project, run the following with prepared original tiles and the native CPU library available in `.native`.

```text
python bench_r35_pose_matching.py --tiles data/tiles.json --native .native --ac-only --output owner-controls.json
python bench_r35_pose_matching.py --tiles data/tiles.json --labels chain-bank-0-5 chain-bank-7-5 nested-tight-25 --seconds 8 --output owner-native.json
python -m unittest test_r35_pose_matching
```

The independent finite tests enumerate complete physical assignments and allowed rotations on 300 seeded domains. The shared alternating-path utility is checked against every injection of small square and rectangular matrices. Empty domains, duplicate and malformed images, alternative permitted rotations of a tile, modified puzzle masks and loss of the authenticated theorem have separate controls.

## Dense conditional inventory on CPU DFS

Single-process Gold DFS started with `--reference-pose-cuts` also uses an injective tile-assignment certificate on each bounded seam cut. Each tile fixed by a singleton cell is unavailable elsewhere. The receiver removes those tile bits and sorts the remaining row sizes. Fixing an arbitrary candidate in one free cell consumes at most one choice from every other row. When the i-th smallest of the remaining row sizes has at least i+1 choices before that consumption (one-based), every candidate extends greedily to a legal assignment. Empty or newly singleton residual rows and unmet size bounds use the ordinary matcher. The receiver stores only complete results, using the existing 128-entry cache and one previous miss. A failed check consumes zero units from either matching work budget. Cancellation is checked by the caller immediately before the bounded lookup.

Across 24 original scopes, standard and wide cut budgets returned exactly the same narrowed domain at every cell and recorded the same 54 cut prunes. The certificate supplied 1,467 exact results. It reduced matching callbacks from 1,558,781 to 808,535, SCC support callbacks from 22,619,669 to 15,228,646, and support-budget skips from 2,145 to 798. Sixteen scopes improved. The eight deeper negative controls at depths 20 and 23 had zero certificates and unchanged matching and support callbacks. The twelve held-out depth-5/12/20 scopes independently reduced support callbacks from 12,595,176 to 8,393,381. Testing the row-size criterion without removing already-owned tiles gave only 17 hits across the full sample. These measurements count work within the cut propagator. The experiment did not demonstrate fewer DFS branches or a higher board score.

On a fresh single-process run over the saved depth-5 original DFS domain, the public entry stopped at its fourth certificate with one search node, zero SCC support callbacks and zero memo-table hits. Cold resume preserved the stack and all four counts. The two installed legacy cursors carried no conditional-policy marker. Under both CLI policy choices the zero-budget public resume preserved all 25 DFS frames at 21,184 visited nodes and all 26 inner hybrid DFS frames at 1,348 nodes. Existing unmarked cursors keep the prior matching policy even when reference-pose clauses are requested. A freshly opted-in cursor carries a digest-covered version marker. Switching that cursor off uses the previous matcher and retains eligibility for a later opt-in. Portable, symbolic, and hybrid worker searches keep their own policy. Exact terminal source validation and the byte-fingerprinted original conditional matcher remain intact.

The standalone tests compare all certified results to every injection of 8,457 finite graphs, including square, rectangular and undersupplied universes. They also exercise cancellation, tiny budgets, cache eviction, disjoint high tile IDs, malformed and legacy checkpoints, and resumed planted Gold. Reproduce the original-frame comparison from the project folder when those saved DFS and hybrid stacks are available.

```text
python -m benchmarks.bench_r35_cut_inventory --tiles data/tiles.json --checkpoints runtime/searches --output cut-inventory-controls.json
python -m unittest test_r35_cut_inventory
```
