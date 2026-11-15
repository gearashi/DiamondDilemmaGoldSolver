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
