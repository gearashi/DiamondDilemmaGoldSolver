# Diamond Dilemma Gold Solver v1.12.0

The best verified original-puzzle score in these experiments is still **207/240**, with 52 line components. Both seven- and eight-piece four-swap families around the known 207 placements remain open. The unrestricted solver can place every piece anywhere its physical domain permits. Gold requires 240 matched seams and one closed loop.

## Settings for the i7-4790 and RTX 2060 6 GB

**Unrestricted exact Gold.** In the dashboard select **CUDA domains + learned backjumping (experimental)**, **128 lanes**, **scalar propagation**, and **Unlimited**. Enable the previously measured global matching, seam-support reuse, conflict sharing and checked deduction transport profile. Resume the existing checkpoint. This profile took 6,084 decisions across six solved comparison fixtures. The new physical deductions have separate entry points for assembly and restricted residuals. [Full dashboard settings and hardware comparison](ROUND26_RESULTS.md)

**Fresh board with all seven mandatory Gold bonds.** Use the full 64-bit Python 3.12 installation. Start with CPU repair, a whole-board donor-cycle budget of three per round and 64 rounds per independent seed. This continuously creates unrelated boards while preserving the necessary bonds. It saves the best board during the run. Each command below assumes the current directory is the installed project root.

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --rounds 64 --starts 0 --database runtime/r31-bonded.sqlite --output runtime/r31-bonded-best.json
```

For an independent **soft-score** search, omit `--bonds` and use separate checkpoint and output paths. The unbonded whole-board cycle search reached 160 matched edges in 36 total rounds on four held-out seeds. The released block repair needed 101. Each new round can examine substantially more donor combinations. For GPU-initialized starts, `--backend cuda --replicas 512 --dispatches 256 --sampler-cache runtime/r31-sampler.sqlite` reuses the expensive native sampling prefix when extending dispatch budgets. A full replay performed zero new proposals. The bonded CPU runs on the four held-out seeds reached 160 in 45 rounds total and finished at 177, 175, 170 and 178 with all seven mandatory bonds. The 320-round bonded CUDA starts finished at 179, 183, 180 and 178. The matching unrestricted comparison reached 181, 184, 180 and 181; the previous best independent CUDA board still holds 188. Fresh-search checkpoints and exact DFS checkpoints measure separate work. [Fresh measurements](R31_FRESH_HOLDOUT.json) · [Checkpoint results](R31_DISTRIBUTION.md)

**Exact four-swap search for 208.** Start with the seven-piece family, CPU, page size 256, and the existing four-swap union database. This preserves older committed pages. In the new default order, contact count and physical defects precede exact cycle connectivity and score bounds. The eight-piece search can use the same database with `--family 2,2,2,2`. Zero seconds is unlimited.

```powershell
.\venv\Scripts\python.exe r30_multicenter.py --family 2,2,3 --target 208 --backend cpu --page 256 --seconds 0 --database runtime/four-swap-union.sqlite
```

On one held-out interval of 1,024 eight-cell sets, cycle connectivity lowered support assignment solves from 30 to six. The other three compared intervals tied in assignment calls. On the dense pool of cells 0–9, bounding the last attachment lowered complete star evaluations from 25,074 to one for seven pieces and from 4,710 to zero for eight. These star counts compare the connected traversal with its own pricing disabled. The released small-pool half join also reached one and zero. The pool experiments close their stated cells only. [Scoped route measurements](R31_ROUTE_RESULTS.json)

## The physical departure required for Gold

Seven disjoint sets of bonds in the 207/52 arrangement each conflict with the integer signed inventory needed for Gold. They contain 8, 8, 9, 9, 10, 11 and 17 matched bonds. Replaying them in the independently implemented integer CP model, with every tile phase free and the Euler constraint disabled, returned infeasible seven times. Thus any Gold solution must break **at least seven of those 207 specific physical bonds**. Moving a bonded pair together preserves its bond, and a single moved piece can break several. This proof concerns the one-loop objective. The 208 soft-score search allows all seven sets. [Certificate data](R31_GOLD_RESULTS.json)

```powershell
.\venv\Scripts\python.exe verify_r31_departure.py --tiles data/tiles.json
```

## Experiment register

| Proposals | Results and follow-ups |
|---|---|
| 1–10. Connected exchanges, final attachment bounds, donor defects, six-piece comparisons and coverage reuse | [Route results and commands](R31_ROUTES.md) |
| 11–16 and 18. Bonded construction, candidate races, checkpoint forks, whole-board donor cycles, tied repairs, intermediate boards and rotation messages | [Fresh-board experiments](R31_FRESH.md) |
| 17 and 19. Independent seed jobs, GPU-prefix reuse and geographic mismatch budgets | [Measurements, commands and scoped outcomes](R31_DISTRIBUTION.md) |
| 20–28. Geographic conflicts, physical seam support, unavoidable bond losses, tour cuts, cut paths, complementary regions, corner-ring pricing, signed phases and bundled loop switches | [Gold assembly results and reproduction](R31_GOLD.md) |

Elapsed time, complete threshold decisions, and steps to the first witness are recorded separately. Benchmarks with all 160 cells present and only a few free placements concern those restricted domains. Production defaults use deductions with demonstrated gains. More expensive diagnostics and inconclusive variants remain explicit experiments for reproduction.
