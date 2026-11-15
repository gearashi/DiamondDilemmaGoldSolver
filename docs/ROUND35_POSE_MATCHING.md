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
