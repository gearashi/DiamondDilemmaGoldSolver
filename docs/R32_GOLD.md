# Gold on a different physical board

**Every Gold solution of the encoded puzzle loses at least 17 of the 207 matched physical side pairs on the saved 207/52 board.** A pair names two tile sides and stays intact when those tiles move together. The bound applies at any geographic arrangement. A physical swap or tile rotation may destroy several of the named bonds. The 208 matched-edge experiments use a separate objective.

**Eight specific physical bonds are compulsory in every Gold board.** There are 3,060 reciprocal, between-tile mask matches. Twenty-one of them complete a proper closed strand entirely on their two tiles. With those pairs excluded, B145 side 1 can only mate to S19 side 0 (zero-based physical side IDs 433 and 54). Exact perfect-matching support excludes three other options involving side 54. The resulting physical matching model uses 3,036 candidate pairs. Each of the eight required bonds and each exclusion is recomputed from the input. On meshes with an open side, the solver confines this filter to proper local loops.

## How the loss bound works

The 207 reference has seven published, mutually disjoint signed-inventory contradictions. Three proper loops have disjoint physical bond sets outside all seven contradictions. Their physical side pairs are (96,392), (245,352), and the six pairs (171,376), (173,374), (192,373), (194,266), (265,350), (349,377). Keeping every bond of any one group would preserve a phase contradiction or a proper closed loop. All ten groups are disjoint, contain 80 distinct reference bonds, and require at least one physical loss each. The other 127 reference bonds include seven of the eight compulsory bonds.

Each physical tile has one phase bit representing its choice of complementary color. A retained bond implies an XOR equality between the phases of its two tiles. An outside phase error indicates a broken bond. Some broken bonds can still have compatible phases, so the error count is a lower bound on the actual outside losses. There are 120 variable outside bonds after accounting for the compulsory ones. Any assembly losing at most 16 reference bonds has at most six outside errors. Each such assembly belongs to exactly one outside phase-error set.

For each packed group, exact elimination of phase bits gives an XOR syndrome for the chosen outside errors. A table records the smallest number of phase mismatches in the group that could realize the syndrome. Charge a group the greater of this minimum and one. Sum the ten group charges and the outside error count. When the sum permits a solution at the requested budget, a joint check enforces one assignment for shared phases by explicitly branching over the permissible lost bonds and solving binary linear equations. Removing a bond releases its equation. The procedure allows any release with a compatible phase. Every survivor of the joint test remains a relaxation candidate for full geographic and one-loop verification.

The direct outside-set iteration and an indexed version agreed on the full four-error layer at budget 14, including all 58 joint calls. The indexed query split each six-or-fewer-error pattern into an ordered prefix and a suffix of at most two errors. At budget 15, all six error layers were infeasible. The five-error layer represents **190,578,024** possible error sets. It used 266,916 prefixes, 7,140 suffix catalogue rows, and 1,046 joint checks. The seeded index selection controls which exact table is used first and cannot eliminate a valid choice. At budget 16 the six-error layer represented 3,652,745,460 possible sets, covered by 7,673,835 ordered prefixes. Its nonoverlapping intervals and all lower layers closed. A separate direct deletion audit exhausted all 8,001 possible two-outside-loss sets at budget 12. Small controls independently enumerate every tile phase, classify min-cost group losses, enumerate every legal Gold placement in narrow scopes, and check indexed and ordinary traversal. A relaxation witness or timeout cannot rule out a physical range. [Full measurements and physical side IDs](R32_GOLD_RESULTS.json)

## Results by proposal

| Proposal | Measured outcome and integration |
|---|---|
| 13. Initial physical pairs | Seven compulsory bonds became eight. All 24 excluded pairings correspond to 432 directed placement-support entries in the production CPU tables across side orientations. The Gold matching model, Gold-aware forced assemblies and exact CUDA domain search can consume the deductions. The ordinary edges objective keeps full mask-matching support. |
| 14. Exact loss budgets | Checked disjoint loss layers and durable prefix intervals. A completed 12-budget search reopened with zero pages and no outside queries. A geographic trial at exactly 16 losses and one outside physical loss closed the first and last branch of the six-bond loop in one CP call each. The other branches and farther boards remain eligible. |
| 15. Minimum losses over all 207 | XOR, signed integer, Euler and physical mating cost models encode the complete reference. All four two-second global CP runs at budget 16 returned unknown. The projected XOR index subsequently closed budget 16 with a checked finite calculation. Feasible costs reported by the phase or mating models apply to those relaxations. |
| 16. Repeated separation | A master chooses released bonds, checks the retained set for closed loops or parity conflicts, then handles geographic pose packing and complete physical pairing. Every transported conflict is replayed. The original bounded comparison generated additional reference clauses and finished unknown. The planted comparison found a feasible sewing; the geographic search independently validated that board. Master results are optional diagnostics. |
| 17. Cost of replacement mates | The mating stage requires one physical mate per side, a common phase per tile and the checked exclusions. Requiring broken sides to find replacements raised the short full-board CP lower bound to three from two. The projected index supplies a stronger reference bound. Retain simultaneous mating for restricted branches. |
| 18. Growing physical loop conflicts | Given eight compulsory mates, 40,000 optional-extension probes found ten distinct proper-loop conflicts of size two or three. A finite planted test checked each emitted clause against independent solutions. Probes are explicit. Truncating a catalogue supplies only the conflicts actually found. |
| 19. Placement propagation | On six narrowed contexts from the original reference, the physical watcher removed the final remaining realization of a forbidden pairing. Rollback uses the actual domain vector. The CUDA watcher rechecks a physical core and translates its witness to a placement conjunction. On the RTX 2060, 13 replayed clauses entered the verified store and 1,024 clause copies entered 128 lanes. The 32-second run made 964 decisions and saw zero clause units, so this remains an explicit comparison. Full production domain tables carry the unary mate exclusions on every lane. |
| 20. Cross-board sharing | The independent 183 CPU and 188 CUDA reference boards supplied 23 and 46 replayable parity clauses. None of those additional clauses fully occurred on the original 207 board. The union held 82 physical clauses, of which 13 applied there. Keep physical transfer explicit for related boards. |
| 21. Disjoint nested partitions | Unary losses propagate automatically. The first nontrivial split on this reference has six children. Each child holds the preceding bonds and loses the next bond, so each admissible Gold placement has one owner. Completed geographic queries persist under their full physical scope, loss interval, domains, pins and branch. A planted Gold query reopened with zero CP calls. Timeouts leave the branch open. |
| 22. Changed pose bound | Hitting the checked groups in oriented tile space proved that at least nine tile poses must change from the reference. Restricting allowed changes to cell zero made the covering relaxation infeasible. This counts altered tile poses and permits both rotations and swaps. |

[Follow-ups on the independent boards and geographic branches](R32_GOLD_FOLLOWUPS.json)

## Commands

From the prepared installation, the unrestricted dashboard exact Gold search and the independent constructor are appropriate for finding a completely different board. The loss command answers a different question: how many of the current reference's bonds a Gold board could keep. It saves exclusions between runs. On the tested i7-4790 use CPU for the loss index. Zero seconds removes its time limit. Use one database per experiment or rerun the same command to continue its saved scope. The first open certified budget in this release is 17.

```powershell
.\venv\Scripts\python.exe r32_gold.py inventory
.\venv\Scripts\python.exe r32_gold.py loss --upper 17 --seconds 0 --page 8192 --database runtime/r32-loss.sqlite
.\venv\Scripts\python.exe r32_gold.py geographic --lower 17 --upper 17 --outside 1 --branch 0 --database runtime/r32-geo.sqlite --seconds 30
.\venv\Scripts\python.exe r32_gold.py poses
```

In loss mode, `--outside` is an exact count of outside phase errors. Omitting it searches every eligible error layer. In geographic mode it instead counts reference bonds actually absent outside the packed groups. Supplying one branch or one layer scopes the result to that branch or layer. A geographic `GOLD` result has 160 distinct pieces, 240 matched seams and one independently validated closed line.

Use `r32_gold.py dropout --stage integer` or `--stage mating` for the cost models, `master` for separation, `loops --limit 40000` to probe short physical cycles, and `cores --output claims.json` to export replayable conflicts. Repeating `--claims` loads prior physical clauses and verifies them from the current inventory. `watch --claims claims.json` enables them in placement search; `watch --backend cuda --replicas 128` exercises their checked transport to CUDA. A case JSON can supply a reference `codes` array, restricted `domains` or mode-specific pins. Gold-specific clauses do not prune searches for 208 seams.

```powershell
.\venv\Scripts\python.exe -B -m unittest -v test_r32_gold
.\venv\Scripts\python.exe -B -m benchmarks.bench_r32_gold --output gold-results.json --max-budget 15 --layer-seconds 420
.\venv\Scripts\python.exe -B -m benchmarks.bench_r32_gold_followups --fresh runtime/fresh-cpu.sqlite --fresh runtime/fresh-cuda.sqlite --output followups.json
```
