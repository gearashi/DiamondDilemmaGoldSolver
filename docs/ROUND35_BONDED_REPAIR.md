# Mandatory assembly repair in Round 35

## 37. Enforce forbidden physical mates during ordinary fresh repair

The original Gold inventory has eight mandatory bonds and 24 forbidden physical side pairs. Across twelve independent 320-round starts, the released end-of-round cleanup ran 1,401 times. The cleanup succeeded on all those rounds. Its direct cost was 3,605 matched seams summed over separate intermediate boards, including repeatedly returning to the same conflict. The released eight-bond profile, the regional-certificate profile and the individual-cycle profile each supplied four starts.

`--forbidden-repair guarded` excludes a donor orientation when it would create a forbidden physical mate with a fixed neighbor. This applies to noncontacting full-board swaps and circulation, independent assignments, and the local CP model. CP checks both sides when both cells can move. The sparse list of physical pairs is shared by local seams with the same oriented sides. Incremental price invalidation now considers neighbor identity in this mode, including substitutions that leave the facing mask unchanged. Whole-cycle rollback handles a touching move, tie repair or rotation stage that would otherwise introduce a conflict. Every resulting board retains the mandatory assemblies and passes the physical validator.

Rollback assigns each complete permutation cycle and each changed singleton orientation to its earlier or later state. It fixes uninvolved changes, finds the independent geographic components of possibly forbidden interactions, and optimizes the physical score on components of up to 14 decision groups. Larger components use a compatible greedy pass with a legal earlier state. The `project` option uses this as a final alternative to released cleanup and keeps the better witness. Its reference is taken after mandatory assemblies relocate, so rollback cannot tear apart a newly placed assembly. The `prices` option confines the new restrictions to separable donor/independent assignment prices. Its regional certificate uses the same unrestricted no-contact residual as its subsequent CP fallback.

On 240 distinct saved transitions with a physically compatible earlier state and a newly forbidden later mate, projection beat the released cleanup 150 times, tied 78 times and lost 12 times. Retaining the better legal witness gained 242 total seams over always using released cleanup. The projection enumerated 494 local binary choices. All 240 projected and released witnesses were checked against full physical mate sets, legal tile codes and complete edge scores. A separate regression starts with one forbidden edge, shows that undoing just one rotation creates a different forbidden edge, then verifies the coupled correction. Exact regional CP scores match all eligible permutations and rotations with zero, one or unlimited contacting moves.

| Paired profile | Released cleanup count / summed seams lost | Guarded final cleanup count | Released 320-round scores | Guarded 320-round scores |
|---|---|---:|---|---|
| Exact packing and two-bound regional repair, seeds 3500–3503 | 536 / 1,432 | 0 | 180, 176, 184, 175 | 181, 179, 181, 175 |
| Released full circulation, seeds 3504–3507 | 267 / 647 | 0 | 180, 182, 175, 181 | 183, 174, 176, 176 |
| Individual profitable cycles, seeds 3508–3511 | 598 / 1,526 | 0 | 178, 183, 179, 175 | 184, 183, 185, 178 |

On the first four starts, prices-only finished at 176/181/181/175 and final projection at 178/175/180/178. Each profile used identical starts, its own 16/64/320-round prefixes, and a separate cold 64-round replay on seed 3500. In a further sixteen one-round physical controls, five had no collision anywhere on the released path. Their guarded scores tied once and improved four times; different physically legal ties and candidate lists can alter the path even before the winning sequence has a collision.

**Decision.** Keep the guarded mode for avoiding repeated forbidden-mate cleanup and for individual-cycle exploration. Keep final projection as an optional witness comparison, including the tested local transitions. Prices-only remains a narrower option for separable search. The regional score distributions still favor different policies on different seeds. The guarded individual-cycle run reached 185 on seed 3510 at round 245. These are fresh candidate scores, below the established 207 board and the one-loop Gold target of 240.

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --contact-group 1 --forbidden-repair guarded --rounds 320 --starts 0 --database runtime/fresh-guarded-cycles.sqlite --output runtime/fresh-guarded-cycles.json
```

[Physical and trajectory receipts](R35_FORBIDDEN_REPAIR.json). Reproduce stage incidence with `bench_r35_forbidden_repair.py`, exact local rollbacks with `bench_r35_mate_rollback.py`, and the three matched regional trajectories with `bench_r35_trajectories.py`.

## 38. Coordinate whole mandatory assemblies and price displaced pieces

The original eight rigid dimers each have 480 legal poses. Single relocation samples up to eight poses of one dimer and uses a random legal assignment for the displaced pieces. The optional `--assembly-relocation priced` keeps that candidate and scores every legal displaced-piece permutation and rotation for each sampled pose, up to four displaced pieces. `reciprocal` adds all legal pose pairs that exchange two entire dimer footprints. That is 28 pairs and 112 possible oriented exchanges on the original board. `mutual` also repairs sampled poses that partially overlap another assembly. Each mode keeps the best fully legal witness and checks its physical edge score. The search keeps the released sampler�s random-number progression.

| Independent controls | Better displacement orientations | Plus reciprocal exchanges | Plus partial-overlap exchanges |
|---|---:|---:|---:|
| One trial on each of 24 starts | +3 seams on 3 starts | +17 on 11 | +1 on 1 |
| Eight trials on the same 24 starts | +4 seams on 4 starts | +3 on 3 | +2 on 2 |

Half the starts had already completed eight repair rounds. No individual local comparison lost a seam. A planted two-dimer blockade reached its known single-loop Gold only when both dimers could move. Separate exhaustive rotated displacement controls reproduced full board scores under forbidden physical mates.

| 320-round profile | Seed 3500 | 3501 | 3502 | 3503 |
|---|---:|---:|---:|---:|
| Released regional repair | 180 | 176 | 184 | 175 |
| Priced singles | 179 | 180 | 175 | 176 |
| Reciprocal | 180 | 175 | 181 | 176 |
| Partial overlap | 180 | 175 | 181 | 175 |
| Guarded ordinary repair, released relocation | 181 | 179 | 181 | 175 |
| Guarded, priced singles | 179 | 173 | 182 | 175 |
| Guarded, reciprocal | 177 | 180 | 184 | 184 |
| Guarded, partial overlap | 177 | 180 | 184 | 184 |
| Guarded, individual cycles, released relocation | 178 | 179 | 174 | 175 |
| Guarded, individual cycles, partial overlap | 177 | 180 | 177 | 177 |

Each profile used 16/64/320-round prefixes and a cold 64-round repeat on seed 3500. On the unguarded reciprocal runs the family cache priced 3,429 distinct-in-cache queries, reused 32,411, and bypassed 129,644 oriented exchanges. The three unguarded policies returned the same physical boards and logical-work counters on all 36 prefixes as their earlier per-pose cache runs. Receipts include process-local physical cache operations separately. The bounded cache key contains the exact footprint codes, facing masks and the physical boundary darts that may participate in a forbidden mate. Results are placed back into the current board, which preserves remote edits.

**Decision.** Retain the displacement and coordinated search as optional moves. Guarded reciprocal is the smallest tested variant to attain 184 twice on this seed group. Partial overlap added verified local moves and had mixed trajectory effects. Use fresh independent starts for this comparison. The incumbent remains 207 and Gold requires 240 seams on one loop.

[Assembly controls, physical hashes and trajectories](R35_ASSEMBLY_RELOCATION.json). Use `bench_r35_assembly_relocation.py` for independent one-move comparisons and `bench_r35_trajectories.py` for saved prefixes.
