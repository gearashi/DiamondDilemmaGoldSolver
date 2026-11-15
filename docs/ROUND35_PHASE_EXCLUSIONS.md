# Geographic exclusions from signed tile phases

A phase bit specifies which of the two colors touches the starting corner of a tile in the region coloring induced by a Gold loop. Flipping every color preserves all matches. Fixing a phase leaves the placement and rotation of every tile free.

For geographically embedding a proposed signed arrangement, use `r30_gold.py embed --phase-anchors rules`. The JSON input accepts a `pins` object mapping physical tile indices to 0 or 1. The solver rechecks a stored exclusion on the current tile data, mesh and base domains before sharing it across requests. When enough pins are known, it can finish before allocating CP-SAT. With fewer pins it posts a short clause for each color convention.

The two shipped exclusions use 21 and 18 of the 160 signs. SAT unit propagation independently rejected every one of 480 placements of the selected anchor for each exclusion. They applied to 19 and 43 of the 54 nearby phases of their respective donors. In those sets, 7 and 20 covered phases respectively passed both earlier Euler and physical matching tests. In a separate panel of 105 phases neither short rule applied. Directly probing five scarce tiles rejected 48 on that panel, all already ruled out by at least one earlier test. These measurements support reusing the rules for related signed queries. [Paired tests](R35_HELDOUT_PHASE_ANCHORS.json)

For new phases, `--phase-anchors roots` tests the placements of scarce tiles first. `--phase-anchors shared` also removes poses excluded in every surviving branch of a completed split. Each policy uses verified board rotations only when the domains preserve them. The independent enumeration checked 768 pruning calls, complementary sign assignments and positive Gold layouts. [Independent audit](ROUND35_PHASE_BOARD_AUDIT.md)

A complete fixed-phase search can also use signed poses directly. For the third original donor, testing seven anchor tiles first removed 12,291 poses, reducing the native solve from 472,912 to 299,387 decisions and from 64,935 to 33,433 conflicts. The prefilter used 4,183 root probes. Both native runs finished with an infeasibility result. To reproduce, install the SAT search dependencies and run the command below. A zero solve budget runs to completion. [Counters](R35_FIXED_PHASE_COMPARISON.json)

```powershell
.\venv\Scripts\python.exe bench_r35_phase_board.py --tiles data/tiles.json --phase 0xd237aba7e91101e88041413e8c589902021e5126 --anchor 25 --probe-tiles 25 159 23 14 22 37 90 --seconds 0 --output runtime/fixed-phase.json
```

The unrestricted Gold search can explore phases from entirely different boards. Each exclusion applies only to its verified signed conditions.
