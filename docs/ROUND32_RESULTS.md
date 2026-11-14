# Diamond Dilemma Gold Solver v1.13.0

The best independently scored original board remains **207/240, with 52 line components**. The release covers all 28 workstreams. Its two searches answer different questions. A local exchange can improve the seam score to 208 even if it creates small loops. Unrestricted Gold search can start anywhere and requires all 240 seams to match in one loop.

## What the search ruled out

- **Four swaps from the known 207 boards.** The installed checkpoint gained 65,536 seven-piece position sets and 16,384 eight-piece sets. These represent 41,287,680 and 5,160,960 center-labelled permutations respectively, before the independent choice of rotations. No 208 appeared in those ranges. The database now covers **706,816 of 466,251,591,520** seven-piece sets and **16,384 of 8,917,061,687,820** eight-piece sets. Reopening checked intervals added zero pages and zero search work. [Checkpoint measurements](R32_CHECKPOINT.json)
- **Gold from a different board.** Any one-loop solution must break **at least 17 of the 207 specific matched physical bonds** in the released 207/52 reference. Eight physical bonds are required in *every* Gold board, and 24 pairings can be excluded. The loss bound applies to reference bonds that disappear, regardless of where tiles move. The proof closes all admissible phase-error layers for a budget of 16 losses. The linked separate budget-15 replay includes its input hashes. [Gold calculations, controls and follow-ups](R32_GOLD.md) · [Independent run and input hashes](R32_LOSS_REPLAY.json)
- **Repeated work.** On six original-puzzle intervals the ranked filter skipped **10,381 of 24,576** position sets before generating them. An overlapping-defect test ruled out 57 of 80 previously open prefix/center cases. In 96 signed repair rounds, shared price rows reduced 5,249 calculations to 1,260 on the first run and zero on replay, with identical boards. [Ranked filters](R32_ROUTES.md) · [Fresh repair measurements](R32_FRESH.md)

## Recommended settings for the i7-4790 and RTX 2060 6 GB

**For Gold, start here.** In the dashboard select **CUDA domains + learned backjumping (experimental)**, **128 lanes**, **scalar propagation** and **Unlimited**. Use the full linked matching profile, including **Check global GPU piece matching**, **Reuse GPU seam supports**, **Watch GPU learned clauses**, **Share independently verified GPU conflicts** and **Transport checked GPU deductions**. Leave **Use cooperative GPU propagation** off for the scalar layout. Resume your current search. The native runner can migrate a v1.12 Gold checkpoint through its saved frontier. A checkpoint replay recovered all four independently validated fixture solutions. On the released original puzzle, a short paired GPU run reached no decisions in either version. On the closed-loop fixture the new mate filters cut support-word scans from 108 to 72. The hardware evidence supports resuming the existing profile, and gives no time-to-Gold estimate. [GPU checks](R32_GPU_RESULTS.json) · [Dashboard control settings and comparison](ROUND26_RESULTS.md)

**For 208 near the three known boards**, use the exact multicenter search on CPU with pages of **1,024**. Begin with seven pieces. Switch the family to `2,2,2,2` for eight using the same database. Across four paired intervals, all page sizes and both CPU and CUDA made the same 3,659 assignment solves. CPU pages of 1,024 used eight commits and 1,118 range probes. Pages of 256 used 32 commits and 1,186 probes and limit the amount redone on a hard interruption. The timer value of zero is unlimited. [Hardware measurements](R32_ROUTE_SETTINGS.json)

```powershell
.\venv\Scripts\python.exe r30_multicenter.py --family 2,2,3 --backend cpu --page 1024 --target 208 --seconds 0 --database runtime/four-swap-union.sqlite
```

**For independent fresh boards**, use full Python 3.12, CPU repair, cycle budget **3** and **64 rounds per seed**. This searches unrelated placements while keeping all eight mandatory bonds and avoiding the forbidden pairings. Four held-out starts finished at 170, 167, 174 and 173. Other builders can explore different regions. Their scoped results and settings are in the [fresh-search experiments](R32_FRESH.md).

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --rounds 64 --starts 0 --database runtime/fresh-eight.sqlite --output runtime/fresh-eight-best.json
```

Use the commands from the installed project folder. Each command has its own search objective and checkpoint format.

## All workstreams and follow-ups

| Workstreams | Experiments and measured dispositions |
|---|---|
| 1–7 | [Rank skipping, restricted donor assignments, two-cycle joins, ownership buckets, bridge pricing, overlapping defects and multicenter traversal](R32_ROUTES.md) |
| 8–12 | [Dependency-checked page reuse, partially completed roots, attachment catalogues, shared exterior rotations and equivalent half-join responses](R32_REUSE.md) |
| 13–22 | [Loss budgets, conflict packing, physical assembly, geographic restrictions and checked conflict reuse](R32_GOLD.md) |
| 23–28 | [Disjoint phase starts, signed repair, parity contours, geographic slot cuts, reversible assemblies and trajectory reuse](R32_FRESH.md) |
