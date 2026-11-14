# Gold solver v1.11.0

The best verified board still matches **207 of 240 edges with 52 line components**. Gold requires all 240 edges and one closed loop. The new exact search closed both six-piece four-swap families at all three known 207-edge placements. At each center, that covers **2,755,123,040,800** physical permutations with every legal rotation available. Across the six runs, only two permutations needed exact rotation optimization. Their best scores were 205 and 206. A four-swap route to 208 from these centers would have to move seven or eight pieces. Other starting placements and longer routes remain open. [Six-piece runs](R30_ROUTES.md)

## Best settings on the Core i7-4790 and RTX 2060 6 GB

**For the actual Gold puzzle**, use the unrestricted dashboard search. Choose **CUDA domains + learned backjumping (experimental)**, **128 lanes**, **scalar propagation**, and **Unlimited**. Enable global piece matching, seam-support reuse, verified conflict sharing and checked deduction transport using the full measured profile. Resume the same exact-search checkpoint. This search permits entirely different placements. A verified 240-edge single loop saves the solution and ends the search. On six solved comparison fixtures this profile needed 6,084 decisions, the lowest reported count in that hardware comparison. Time to Gold on the full input is unknown. For CPU-only Gold use **Parallel CPU DFS**, **Automatic** propagation, and **Solve small remaining branches with matching and forest search**. Use eight workers on a dedicated machine or four while using the desktop. [All exact-search toggles and hardware measurements](ROUND26_RESULTS.md)

**For a different board that might beat 207**, the independent-start command seeds fresh complete placements, samples on the GPU if requested, then reassigns nonadjacent pieces and optimizes rotations. Each completed seed is saved to SQLite. Set `--starts 0` for unlimited independent starts. Use the full Python 3.12 environment for CUDA. Run this in PowerShell from the installed solver folder.

```powershell
.\venv\Scripts\python.exe r30_fresh.py --backend cuda --replicas 512 --dispatches 256 --rounds 320 --starts 0 --database runtime/fresh-cuda.sqlite --output runtime/fresh-cuda-best.json
```

Using 512 GPU replicas, 256 dispatches and 64 CPU repair rounds gave scores of 167, 177, 174 and 172 on four independent seeds. Extending each to **320 rounds** raised them to **179, 188, 185 and 183**. Four CPU-only seeds rose from 175, 172, 171 and 172 to 183, 180, 183 and 182. The extension reused all eight initializations, all 512 earlier repair rounds and all 33,554,432 GPU proposals. It performed 2,048 additional repair rounds. A second 320-round run reused all completed work. The benchmark selected 320 because every seed improved at that setting. It did not compare every possible budget or population.

The eight boards at 64 rounds were at least 148 swaps away from the closest known center in any proper board rotation. Counting unavoidable defects proved that four additional swaps from those placements could reach at most 184–193, depending on the board. The route solver can skip those whole neighborhoods. The fresh search checkpoints the actual board, RNG state and best board after the sampler and every 16 repair rounds. Raising the round count on the same database resumes the same trajectories. Independent starts are heuristic and have no exhaustive-coverage count. Changing the backend to CPU removes GPU sampling. A run exports its own best board separately from the dashboard incumbent. [Validated boards and scope checks](R30_FRESH_RESULTS.json)

**For an exhaustive four-swap proof around the known 207 boards**, start with seven pieces. Contact, movement, shared star and cycle-type assignment screens are enabled. Use the same database on subsequent runs. The seven- and eight-piece rank spaces are separate. Passing `--family 2,2,2,2` selects eight pieces. In a separate CUDA-only filter trial the device processed large geometry and core batches accurately. The CPU setting below avoids device setup for small pages. Neither family has been exhausted.

```powershell
.\venv\Scripts\python.exe r30_multicenter.py --family 2,2,3 --target 208 --database runtime/four-swap-union.sqlite --backend cpu --page 256 --seconds 0
```

To continue the existing one-swap walk through scores of 206 and above, use the [previous plateau command and checkpoint](ROUND29_RESULTS.md). It can take more than four swaps. The walking floor constrains the path, so an improving route with a lower intermediate score remains possible.

## Workstreams and follow-ups

| # | Implemented and tested | Outcome |
|---|---|---|
| 1 | Undo a cycle, split it into predecessors and require common exterior rotations | Nonpositive isolated-cycle rules cover at least 652,374,718,073,587 four-swap endpoints at 207/52. The more expensive joint split check is optional. |
| 2 | Require contact between cycles, then test whether one contact suffices | At all three centers, all three cycles on seven pieces or all four cycles on eight pieces must be connected by cross-cycle contacts at mesh distance two or less. That requires at least two or three contacts respectively. |
| 3 | Unavoidable defects, compact movement clauses and partial hitting budgets | Four 61-cell clauses at 207/52 together exclude 1,259,815,824 six-cell supports. |
| 4 | Shared rotations on adjacent pairs, paths and rings | Pair discounts lower the doubled upper bound to 414 at each center. Paths and rings tied the cheaper pair check there. |
| 5 | Canonical directed-cycle joins | Both six-piece types closed. Restricted seven- and eight-piece joins agreed with exhaustive donor ordering. |
| 6 | Indexed partial joins for a triple plus two swaps and four swaps | Tested one-plus-two and two-plus-two joins on the same pools. The four-swap timings varied by pool. The half join stays optional. |
| 7 | Assignment bounds constrained by the remaining cycle lengths, including weighted matching | Eight of 2,544 tested support/family queries closed at 208 beyond the general assignment bound. Used on survivors. |
| 8 | Bound a partial cluster plus isolated movers, multiple clusters and interaction refinements | At the 207/52 center, 685 of 710 nearby pair anchors close even with arbitrarily many distant isolated movers. The evidence includes domain-preserving rotations. |
| 9 | Remaining-swap defect budgets and regional tables, including bounded plateau walks | A planted one-swap decision dropped from 16 exact solves to one. On eight unrelated 160-piece boards, a single budget bound ruled out every four-swap endpoint at 208. |
| 10 | Witness ordering and comparison with proof exhaustion | Gain ordering reduced total prefixes to the first checked planted witness from 241 to 154. One case took more steps. |
| 11 | Cross-center ownership, parity, mesh rotations and disjoint rank intervals | The two outer shells overlap on 668,394,211,290 eligible endpoints after prior exclusions, before local filters. |
| 12 | Share unaffected support prices and individual local terms | On one matched six-cell range the CPU computed 1,416 support assignments in place of 1,536, with 2,699 mixed terms reused. |
| 13 | Group donors by responses, refine under changing neighbors and preserve physical IDs | At 207/52, 545 scalar response evaluations supplied all 25,600 individual destination/donor scores. Warm replay needed zero new evaluations. |
| 14 | Learn sufficient donor subsets and independently verified numeric caps | Two donor choices covered five distinct permutations on a restricted query. All five repeated queries closed without solving again. |
| 15 | Exact history merging, relaxed widths and refinement | Same-score exact merges saved transitions on six-, seven- and eight-cell controls. Kept optional for hard residuals. |
| 16 | First-exclusion accounting, CPU/CUDA page reuse and filter ordering | The two-contact screen reduced assignment solves from 1,366 to 79 on the measured seven-cell range. The three-contact screen cut the eight-cell count from 1,660 to 394. Reopening completed pages on the other backend generated zero pages. |
| 17 | Independent CP-SAT model with exact cycle lengths and all rotations free | Both six-cell families closed on the restricted real pool during presolve. Full-family five-second CP trials remained open. |
| 18 | Rotation-consistent forest bounds with reusable messages and integer checks | A one-round bound tightened nine of 55 selected permutation bounds, including a 208-to-207 exclusion. |
| 19 | Discover arbitrary endpoint-type cuts and reuse necessary two-crossing conditions | The Gold phase entry point now rejects two real restrictions and both complements before any CP call or fixed-width subset enumeration. The rule is independent of the physical starting board. |
| 20 | Require multiple crossings to coexist in one side matching and check geographic embedding | Five coupled cuts ruled out a ten-cell control with four distinct legal side matchings. A planted geographic control recovered Gold. The full matching root stayed open in the bounded trial. |

The mathematical counts overlap. The first-exclusion counters measure incremental benefit within individual runs. Benchmarks with fixed pools, restored checkpoints and planted positive controls support different conclusions and retain their scopes in the linked evidence.

[Exact cycle joins and witnesses](R30_ROUTES.md) ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· [Physical bounds and matched seven-cell control](R30_BOUNDS.md) ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· [Ownership, diagrams and CPU/GPU coverage](R30_SHARED.md) ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· [Gold and independent exact checks](R30_EXACT_GOLD.md) ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· [Original contact verifier output](R30_CONTACT_RESULTS.json) ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· [Second-contact verification](R30_SINGLE_CONTACT.json) Ã‚Â· [Connected-cycle verification](R30_CYCLE_GROUPS.json)
