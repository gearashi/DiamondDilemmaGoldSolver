# Gold solver v1.10.0

The strongest result is a lower bound on how many swaps separate the three known 207-edge placements from 208. For each center the solver exhausted **317,898,812,400** arrangements made by three disjoint swaps. None can reach 208, allowing independent rotations of all 160 pieces. Any sequence of at most three swaps with a repeated cell moves at most five pieces, which the previous release already exhausted. Therefore **any route from these centers to 208 requires at least four swaps**, including routes that temporarily lose matches. The three neighborhoods overlap. The best verified board in this release matches **207/240** with **52 line components**. Gold needs 240/240 and one loop. [Exchange proofs and controls](R29_EXCHANGES.md) · [Starting board](R29_INCUMBENT.json)

## Settings for the RTX 2060 6 GB and Core i7-4790

**To look for 208 on the installed Windows machine**, use the full Python 3.12 environment and the following PowerShell command in `C:\DiamondDilemmaGoldSolver`. It follows single swaps from the measured 207/52 board. The short 207-floor component closes first. In the measured continuation, floor 206 reached 69 closed centers and 167 queued. The checkpoint below resumes that work on the test machine. On a fresh installation it starts from the supplied board. The search returns a shortest route at the highest reachable floor it has established, if it finds 208. Keep using the same database and starting board. The budget of 1,000,000 centers is per invocation and zero seconds means unlimited duration. Ctrl+C rolls back only the unfinished center. The dashboard replica setting has no effect on this CPU command.

```powershell
.\venv\Scripts\python.exe r28_exchange_plateau.py --incumbent docs/R29_INCUMBENT.json --database runtime/plateau-206.sqlite --floor 206 --target 208 --widest --ties 1 --centers 1000000 --seconds 0 --output runtime/plateau-route.json
```

**To certify the four-swap shell**, use the exchange command below. Exactly four swaps can move five, six, seven or eight pieces. The five-piece case is closed at these centers. Repeat with `--moved 7` and `--moved 8` to cover the other open cycle types. All three commands can share the database. The support screen is enabled by default. The native three-swap kernel is useful on other centers whose three-swap search remains open.

```powershell
.\venv\Scripts\python.exe r29_exchange_route.py --incumbent docs/R29_INCUMBENT.json --database runtime/four-swaps.sqlite --target 208 --swaps 4 --moved 6 --backend cpu --batch 64 --seconds 0 --path
```

**For an exact Gold dashboard run**, resume the existing checkpoint with **Unlimited** duration. On this machine, the scalar **CUDA domains + learned backjumping** configuration with **128 lanes** and global matching used the fewest decisions in the six-fixture hardware comparison, 6,084. Use the matching, seam-support reuse, shared conflict and proof transport settings in the linked profile. For CPU-only exact search use **Parallel CPU DFS**, **Automatic** propagation and **8 workers** on a dedicated machine, or **4** while using the desktop. Enable **Solve small remaining branches with matching and forest search** for the improved binary tail. The standalone plateau and regional experiments evaluate different objectives. No measured run establishes the fastest configuration to reach Gold on the unrestricted puzzle. [Exact settings and hardware trials](ROUND26_RESULTS.md) · [DFS tail option](ROUND27_RESULTS.md)

## Results from the fourteen workstreams

| Workstreams | Measured outcome and use |
|---|---|
| 1–3. Local interactions, swap distance and cross-center ownership | The three-swap proof priced 41,199,139 complete triples at the 52-component center, 7,505,395 at the original center and 9,399,450 at the third. Group bounds cover the rest. Released exclusions are specific to domain, center and threshold. For larger exchanges, the portable route can reject a whole group of moved cells before generating donor orders. [Measurements](R29_EXCHANGES.md) |
| 4. Score-floor reuse and shortest paths | Lowering the floor from 207 to 206 on the same real center reused 12,720 star bounds and needed 436 additional exact score solves. A cold 206 pass needed 524 solves and 12,720 new bounds. Both returned the same 71 neighbors and 207/52 best. Rotation scores and sampled tie quality are stored separately. Shell and plateau commands share rotation optima when given the same database. [Cold and warm counts](R29_PLATEAU_RESULTS.json) |
| 5. Physical movement cores | Two checked cores exclude 501 of 12,720 two-piece supports and 209,209 six-piece supports at this center before choosing donors. Each core allows all rotations and arbitrary permutation of the pieces outside it. CP timeouts retain the corresponding assumption. These cores are sufficient and can still have redundant cells. Use the command below to recheck and add them to the route or CPU shell database. A different center may reuse a core if it retains every physical identity in that core. [Core inputs](R29_PHYSICAL_CORES.json) |
| 6–7. Endpoint connectivity and parity | Extending from pairs to triples of connected endpoint types produced 2,418 phase rules, including a measured rejection beyond the pair rules. The rank-44 inventory parity system rejects the earlier 15-bond conflict without a CP call. Start with width two in explicit phase checks, then try width three on surviving restrictions. [Gold tests](R29_GOLD.md) |
| 8–9. Geographic inventory and color capacity | The second landmark lowered a six-free-cell incumbent residual's score bound from 208 to 207. Partial color-capacity reasoning rejected a planted residual whose optimistic connection graph passed. Geography is available for restricted residuals, with unsigned inventories for soft scores. [Residual scopes](R29_GOLD.md) |
| 10–11. Binary cuts, inventory and implication reuse | On 391 feasible four-cell controls the priced, contracted cut certified the optimum in all 391 versus 186 previously. On 500 loop restrictions the two-choice implication search used 13,654 work units versus 20,781. The full-width incumbent binary restriction still peaked at 207. [Binary controls](R29_REGIONAL.md) |
| 12. Persistent regional tables and prefix exclusions | On the same 108-solution fixture, five-step pages required 167 compilation visits cold and zero on warm replay, down from 2,672. Changing one cell rebuilt only its region. Use paged regional search with a checkpoint and table cache. [Paging and restart checks](R29_REGIONAL.md) |
| 13. Proof-derived boundary constraints | A checked proof produced two reusable boundaries in five in-process SAT calls on a parity conflict. The cube method needed 98. The full-width 160-cell proof experiment reached its work limit. [Proof methods and limits](R29_REGIONAL.md) |
| 14. Cut&Count | Connected witnesses passed independent validation. The trie required less work on the tested controls, and the 160-cell Cut&Count trial exhausted its budget. Automatic search keeps the trie. [Witness experiments](R29_GOLD.md) |

```powershell
.\venv\Scripts\python.exe r29_move_cores.py --database runtime/four-swaps.sqlite --target 208 --verify-release --seconds-per-query 30
```

A four-swap pilot on the 52-component center with seven moved pieces completed 83,392 distinct supports in 90 seconds. The support bound excluded all 17,512,320 corresponding permutations before any donor ordering needed individual scoring. The full seven-piece scope at this swap distance contains 97,912,834,219,200 permutations.

The measured benefit is the work eliminated in each stated scope. The four-swap frontier, the 206-floor component, and the unrestricted Gold problem remain open.
