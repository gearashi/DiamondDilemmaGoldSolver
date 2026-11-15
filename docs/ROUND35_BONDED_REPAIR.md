# Mandatory assembly repair in Round 35

## 37. Enforce forbidden physical mates during ordinary fresh repair

The original Gold inventory has eight mandatory bonds and 24 forbidden physical side pairs. Across twelve independent 320-round starts, the released end-of-round cleanup ran 1,401 times. The cleanup succeeded on all those rounds. Its direct cost was 3,605 matched seams summed over separate intermediate boards, including repeatedly returning to the same conflict. The released eight-bond profile, the regional-certificate profile and the individual-cycle profile each supplied four starts.

`--forbidden-repair guarded` excludes a donor orientation when it would create a forbidden physical mate with a fixed neighbor. This applies to noncontacting full-board swaps and circulation, independent assignments, and the local CP model. CP checks both sides when both cells can move. The sparse list of physical pairs is shared by local seams with the same oriented sides. Incremental price invalidation now considers neighbor identity in this mode, including substitutions that leave the facing mask unchanged. Whole-cycle rollback handles a touching move, tie repair or rotation stage that would otherwise introduce a conflict. Every resulting board retains the mandatory assemblies and passes the physical validator.

Rollback assigns each complete permutation cycle and each changed singleton orientation to its earlier or later state. It fixes uninvolved changes, finds the independent geographic components of possibly forbidden interactions, and optimizes the physical score on components of up to 14 decision groups. Larger components use a compatible greedy pass with a legal earlier state. The `project` option uses this as a final alternative to released cleanup and keeps the better witness. Its reference is taken after mandatory assemblies relocate, so rollback cannot tear apart a newly placed assembly. The `prices` option confines the new restrictions to separable donor/independent assignment prices. Its regional certificate uses the same unrestricted no-contact residual as its subsequent CP fallback.

On 240 distinct saved transitions with a physically compatible earlier state and a newly forbidden later mate, projection beat the released cleanup 150 times, tied 78 times and lost 12 times. Retaining the better legal witness gained 242 total seams over always using released cleanup. The projection enumerated 494 local binary choices. All 240 projected and released witnesses were checked against full physical mate sets, legal tile codes and complete edge scores. A separate regression starts with one forbidden edge, shows that undoing just one rotation creates a different forbidden edge, then verifies the coupled correction. Exact regional CP scores match all eligible permutations and rotations with zero, one or unlimited contacting moves.

| Paired profile | Released cleanup count / summed seams lost | Guarded final cleanup count | Released 320-round scores | Guarded 320-round scores |
|---|---|---:|---|---|
| Exact packing and two-bound regional repair, seeds 3500â€“3503 | 536 / 1,432 | 0 | 180, 176, 184, 175 | 181, 179, 181, 175 |
| Released full circulation, seeds 3504â€“3507 | 267 / 647 | 0 | 180, 182, 175, 181 | 183, 174, 176, 176 |
| Individual profitable cycles, seeds 3508â€“3511 | 598 / 1,526 | 0 | 178, 183, 179, 175 | 184, 183, 185, 178 |

On the first four starts, prices-only finished at 176/181/181/175 and final projection at 178/175/180/178. Each profile used identical starts, its own 16/64/320-round prefixes, and a separate cold 64-round replay on seed 3500. In a further sixteen one-round physical controls, five had no collision anywhere on the released path. Their guarded scores tied once and improved four times; different physically legal ties and candidate lists can alter the path even before the winning sequence has a collision.

**Decision.** Keep the guarded mode for avoiding repeated forbidden-mate cleanup and for individual-cycle exploration. Keep final projection as an optional witness comparison, including the tested local transitions. Prices-only remains a narrower option for separable search. The regional score distributions still favor different policies on different seeds. The guarded individual-cycle run reached 185 on seed 3510 at round 245. These are fresh candidate scores, below the established 207 board and the one-loop Gold target of 240.

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --contact-group 1 --forbidden-repair guarded --rounds 320 --starts 0 --database runtime/fresh-guarded-cycles.sqlite --output runtime/fresh-guarded-cycles.json
```

[Physical and trajectory receipts](R35_FORBIDDEN_REPAIR.json). Reproduce stage incidence with `bench_r35_forbidden_repair.py`, exact local rollbacks with `bench_r35_mate_rollback.py`, and the three matched regional trajectories with `bench_r35_trajectories.py`.

## 38. Coordinate whole mandatory assemblies and price displaced pieces

The original eight rigid dimers each have 480 legal poses. Single relocation samples up to eight poses of one dimer and uses a random legal assignment for the displaced pieces. The optional `--assembly-relocation priced` keeps that candidate and scores every legal displaced-piece permutation and rotation for each sampled pose, up to four displaced pieces. `reciprocal` adds all legal pose pairs that exchange two entire dimer footprints. That is 28 pairs and 112 possible oriented exchanges on the original board. `mutual` also repairs sampled poses that partially overlap another assembly. Each mode keeps the best fully legal witness and checks its physical edge score. The search keeps the released sampler’s random-number progression.

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

## 39. Forward-check fresh placements and complete their weighted donor assignment

The optional `--assembly-constructor` modes order the remaining groups by viable poses. Occupancy bitsets remove overlapping candidates, and a run-level geometry index survives new seed-specific preferences. Matching and compact modes first test identical and small overlapping cell sets for a resource deficit. Disjoint ordinary slot pools can supply a certificate when each remaining rigid tile belongs to one pool and has enough alternatives for all its rigid peers. Where these tests leave uncertainty, a complete alternating-path matching checks the residual physical resources.

`compact` preserves the constructor's randomly weighted objective. A surviving hint outweighs all random tie terms combined. If the surviving hints simultaneously extend, the maximum-weight assignment needs only the unused rows and donors. When every physical donor and cell has at most two legal partners, forced paths and the two alternatives on each alternating cycle determine the exact optimum. The solver falls back to full weighted matching when needed. Equal weighted optima may produce different physical starts.

| Physical domain | Policy | Complete legal starts | Pose nodes | Weighted leaves visited | General weighted assignment rows |
|---|---|---:|---:|---:|---:|
| unrestricted | released | 32/32 | 256 | 32 | 4,608 |
| unrestricted | forward | 32/32 | 256 | 32 | 4,608 |
| unrestricted | hall | 32/32 | 256 | 32 | 4,608 |
| unrestricted | matching | 32/32 | 256 | 32 | 4,608 |
| unrestricted | final-compact | 32/32 | 256 | 32 | 241 |
| two-extra | released | 5/16 | 26,150 | 16,589 | 2,388,816 |
| two-extra | forward | 5/16 | 25,722 | 16,910 | 2,435,040 |
| two-extra | hall | 16/16 | 4,551 | 521 | 75,024 |
| two-extra | matching | 16/16 | 284 | 16 | 2,304 |
| two-extra | final-compact | 16/16 | 284 | 16 | 0 |
| five-extra | released | 0/8 | 16,000 | 12,888 | 1,855,872 |
| five-extra | forward | 0/8 | 16,000 | 13,228 | 1,904,832 |
| five-extra | hall | 1/8 | 15,707 | 1,413 | 203,472 |
| five-extra | matching | 8/8 | 794 | 8 | 1,152 |
| five-extra | final-compact | 8/8 | 794 | 8 | 0 |

The 32 unrestricted starts each used eight obligatory placements. On those starts, compact assignment reduced 4,608 weighted rows to 241 and retained 4,367 hinted placements. The repeated-static-index comparison rebuilt 3,840 physical poses once and reused 119,040 later pose rows, preserving all 32 cold board witnesses. Changes to physical domains or to the actual pose list force a rebuild.

Each restricted input admits a separately checked complete physical witness. Two or five additional candidate poses per mandatory group compete for scarce remaining cells. All five constructor policies received the same hint and seed and a 2,000-node budget. A bounded failure records a missed starting board. In all 24 sparse final lifts, exact path/cycle assignment handled the remaining 144 pieces without a general weighted matching. Full-resource matching recovered every known feasible restricted start; cheap Hall checks alone missed some five-extra cases.

The Round 35 contour comparison emitted 35 source records representing 31 exact physical domain tuples. All 31 constructed complete legal starts on eight mandatory bonds. The disjoint-pool certificate replaced all 279 full matching checks at the same 248 pose nodes. Compact assignment reduced 4,464 ordinary weighted rows to 289. The separate four-cell conclusions on these contours retain their original pinned scope. Independent small tests enumerate every feasible labelled pose tuple, inspect every three-by-three graph of maximum degree two, account for failed hint extensions, and retry an interrupted constructor.

**Decision.** Keep dynamic occupancy when repeated colliding pose rows are present, full relaxed matching for restricted placement, compact weighted leaves, and checked index reuse across independent starts. Cheap Hall and unrestricted forward checks alone have narrower effects. The constructor seeds heuristic discovery; its failure cannot close an exact Gold branch.

[Constructor, contour and repeated-start receipts](R35_POSE_CONSTRUCTOR.json). Reproduce the conditional controls with `bench_r35_pose_constructor.py`, adding `--contours --work 100` for the published contour scopes or `--warm` for 32 repeated starts.

## 40. Join optional rigid components on the assumed physical mate

`r32_fresh.py fresh --merges 8 --incremental-merges` joins verified component pose tables on one physical dart pair. Each left pose dictates the only possible anchored cell and rotation on the right. An ungrouped piece uses a direct domain lookup. When both sides are assemblies, the join indexes complete right poses by the anchored piece and rejects overlapping footprints. Tables bind the component tiles and the assumed physical edges within the current construction. Their sorted rows match full root traversal. Rebuilding a surviving component can reuse its table after a different optional edge is removed.

| Requested optional edges per start | Independent starts | Released physical neighbor probes | Joined neighbor probes | Right poses indexed | Legal rows produced by both |
|---:|---:|---:|---:|---:|---:|
| 3 | 16 | 49,920 | 23,040 | 0 | 23,040 |
| 8 | 16 | 141,120 | 61,440 | 480 | 61,440 |
| 16 | 16 | 321,580 | 122,880 | 3,840 | 122,860 |

All 48 original phase-compatible runs retained every requested join, yielding 432 optional placements. The independently traversed and incrementally joined models returned identical complete physical boards on each run, with the same mandatory and forbidden mate validation. A second comparison bypassed 203,040 singleton index rows within the joined algorithm. At depth 16 the resulting initial boards had 27–35 seams before repair.

When the bounded assembly constructor fails, a separate occupancy search can prove an incompatible subset of rigid groups. It tries removing groups while the remaining conflict can still be proved. Rollback then removes the latest optional edge in the conflict and rebuilds other surviving groups. The planted conflict preserves a later, unrelated merge that suffix rollback would discard. A budget-limited proof yields only the ordinary retry. A compatible later arrangement survives both a stopped attempt and a completed packing check. Original unrestricted starts showed no rollback incidence in this sample.

**Decision.** Keep incremental joins for optional phase-compatible assembly experiments. Conflict-directed removal applies only to a proved occupancy conflict. It has a planted positive and produced no additional full original starts at the tested merge depths. Reopen expansion of conflict explanations only after recording a recurring original packing failure where this occupancy proof leaves a relevant earlier choice unidentified.

[Joined and full traversal receipts](R35_SPECULATIVE_JOINS.json). `bench_r35_speculative_poses.py` reproduces both builders. `test_r35_speculative_poses.py` compares sorted poses on unrestricted and restricted physical domains with reversed join order.
