# Geographic search using the 207 board

The 207/240 arrangement used for the cell-distance proof has 52 line components. Any Gold solution must differ in tile or rotation at 95 or more of its 160 cells. A second published 207 arrangement has 54 components. It differs at exactly two cells, so its certified lower bound is at least 93 changed cells. [Second arrangement](R29_GOLD_RESULTS.json)

Each bound also holds after rotating its entire board into any of its ten legal orientations. The geographic solver can now use these facts on a different starting board whenever its remaining choices would preserve too much of the reference. A score search toward 208 can still explore every board allowed by its own constraints. [Independently checked distance proof](ROUND35_REPAIR_DISTANCE.md)

## Search counts

| Original-artwork query | Comparison |
| --- | --- |
| Five restricted Gold queries using the geographic API without its optional placement channel | The previous search remained open after 340,866–387,703 CP branches per query. The new search excluded all five before creating the CP model. |
| Same API with one tile fixed, six separate rotations and three native rounds | The previous solver took 147,326 branches in each. Full propagation reduced two to 144,201 and 142,705, raised one to 149,165, and left three unchanged. All six remained open. |
| Public embed command with its normal placement channel | In four comparisons the new preflight avoided building the 115,200-entry channel and saved one CP call. Two predecessors had already proved infeasibility during presolve. Two remained open at the time limit. |
| Three mixtures of different rotated poses | Isolating the seam stage, the geographic API went from three CP calls and 147,326 branches per case to zero on each. |
| Four actual 95-cell repair proposals tested against the 4,225 main clauses | Four additional small conflicts now reject these proposals during the first clause pass. Full geographic propagation can independently reject all four before CP. |
| Sixteen different calls on one problem object | Proof-file validation and decoding fell from 16 times to one. The solver authenticated the current mesh and artwork on every call. All sixteen outcomes matched. |

The five early exclusions comprise actual two-, three- and multi-cell pose conflicts and two distinct ways of holding 66 placements. A search that keeps at most 65 placements can still fail to make a Gold loop for other reasons. An independent test reconstructs the ten orientations from all 480 oriented root choices and checks every translated clause. On changed artwork, the identical tile and rotation codes formed a separately validated 240-edge single loop in every orientation. The solver recognized that the original artwork's proofs did not apply. [Comparison receipts](R35_POSE_DEPARTURES.json) · [Independent transfer checks](../test_r35_pose_transfer_authority.py)

## Combining seam and physical-tile constraints

The preflight first checks the stored pose clauses and the one- and two-orientation limits. It then removes placements without a compatible adjacent tile and cell/tile choices absent from every complete assignment. It alternates these two checks until neither removes another choice. A weighted assignment checks how many exact reference poses must remain, even when no cell has fixed such a pose. Its deductions go back through the same checks. [Weighted ownership cases](ROUND35_POSE_MATCHING.md) · [Two-orientation and cycle controls](R35_POSE_IMAGE_PAIRS.json)

Five widths were compared on 132 original-puzzle scopes, including 51 archived DFS and hybrid frames. Restricting seam checks to domains with at most ten poses excluded 46 scopes. Adding complete tile matching at that width excluded 50. Full-width seam checks with tile matching excluded 60. The full-width check permits all 480 tile/rotation codes. It is enabled on the restricted geographic reference-bound path. An unrestricted board begins with no fixed reference poses.

The queue now schedules a cell only once until its latest domain has been processed. Across the same corpus, scheduled entries fell from 73,139 to 39,192 and entries processed fell from 46,226 to 31,832. Every surviving domain and every infeasible verdict matched. Another short check can certify that each allowed cell/tile pair extends to a complete matching by greedily assigning the remaining rows in order of size. This avoided 35 of 79 full support computations on these inputs. The check is independently tested on all cell-row multisets of every square graph up to four tiles.

A separate exact SAT projection included all applicable pose clauses, image capacities and ownership restrictions on 24 crossover pools. It excluded seven, admitted sixteen and left one unresolved. Ordinary seam propagation had already excluded all 24. The heavier two-image assignment and alternating-cycle checks also yielded no additional geographic exclusion at the final propagation strength. [Complete scope and comparison data](R35_POSE_DEPARTURES.json) · [Two-image dispositions](R35_POSE_IMAGE_PAIRS.json)

## Comparing different 207-edge boards

The two public arrangements use the same 160 physical tiles. They agree on 158 cell poses and 205 matched physical bonds. At most 175 of the 52-component board's named bonds can appear in any geographic Gold solution. The 54-component board has only two matched bonds outside that set, so at most 177 of its 207 can survive. It must lose at least **30 physical bonds**. Its two changed cells similarly give a bound of **at least 93** cell changes. The 95-cell master assignment includes both differences. The test establishes lower bounds. The exact distances to Gold remain open. [Source board](R29_INCUMBENT.json) · [Alternate board](R29_GOLD_RESULTS.json)

The public geographic loss command now checks this overlap against the particular board and the physical joins already fixed in its branch. On the two boards and one partition of each, the previous configuration entered CP-SAT and returned UNKNOWN within eight seconds. The new comparison ruled out loss caps of 31 and 29 respectively before constructing the 115,200-entry placement channel. At the exact lower bounds of 32 and 30, both controls remained open. The optimization is restricted to full geographic Gold search. Signed-only inventory queries continue to test the weaker model they represent. [Compared queries](R35_REFERENCE_CENTERS.json)

## Running a targeted comparison

Use the full Python 3.12 environment for the geographic CP search. A case file can specify the allowed tile/rotation codes in each cell. The CPU DFS and CUDA domain engines can also use the pose clauses on restricted frontiers. Their unrestricted policy follows separate native branch measurements. [Targeted DFS and CUDA commands](ROUND35_POSE_FRONTIERS.md) · [Unrestricted settings](ROUND35_RESULTS.md)

```sh
python r30_gold.py embed --tiles data/tiles.json --input case.json --reference-bound
python bench_r35_pose_departures.py --tiles data/tiles.json --output comparison.json
python bench_r35_pose_departures.py --tiles data/tiles.json --compare-seams --cases pool-3-120 pool-7-40 pool-10-80 --rounds 3 --output seam-comparison.json
python bench_r35_pose_seams.py --tiles data/tiles.json --checkpoints runtime/searches --output widths.json
python -m unittest test_r35_pose_departures test_r35_pose_transfer_authority
```
