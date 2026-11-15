# Regional seam SAT audit and interrupted hint reuse

The independent finite control in `test_r34_seam_audit.py` uses the planted seed-19 ten-cell, fifteen-seam mesh with overlapping physical-piece domains. Of 288 Cartesian assignments, 80 use every physical tile once. They produce 38 distinct three-region defect vectors and checked edge scores from 3 through 11. The oracle reverses the eleven mask bits explicitly for every seam and cross-checks the resulting score with the board validator.

## #55 One-call restarts

A genuine three-core source query at target 5 populated a fresh database. A deterministic interruption returned UNKNOWN only when the independently replayed, broader core was checked. Normal narrow geographic boxes continued to use the native solver. Before the fix, six consecutive target-6 opens with a one-call limit attempted the same hint six times. The query's checkpoint generation stayed at zero.

The solver now records each inconclusive or unsuccessful replay under the exact regional formula and target query while holding its publication lease. This only suppresses future optional checks. The box stays open for a normal native solve. Independently verified exclusions remain available, including those learned later by another query. On the same control the three hints took three calls total, and the next four one-call opens produced a validated witness. Target 7 could still independently try the same source hints. The project stores deferrals in a separately indexed SQLite table.

## Falsification controls

| Check | Result |
|---|---|
| Every defect cap 0–15, fresh counter vs replayed cap-15 counter | Across 5,482 empty, one-region and two-region signed requests, all 10,964 native decisions agreed with physical enumeration. Each unpartitioned cap also matched the enumeration. |
| Native automatic transfer between attainable targets | Targets 5, 6 and 7 found validated witnesses. Target 6 independently checked three nonempty transferred cores and skipped one box. Target 7 skipped four boxes using verified cores. None of the compatible physical boards satisfied any stored core under its own budget. |
| Mixed-polarity nonempty core | Region 1 at most one defect and region 2 at least two defects were each individually satisfiable. Together those assumptions were UNSAT at budgets 10 and 9. Both native solvers returned both literals in the core. After budget-9 verification a budget-8 child reused it, and budget 11 received no exclusion. |
| Partial publication | Interrupting the structural builder after three checkpoints left zero cached encodings. Interrupting after one regional counter left only the complete structural encoding. Reopening reused it and rebuilt the counter. |
| Scope and concurrency | Four concurrent opens at three target/width settings agreed with enumeration, including two owners of the same query. The duplicate returned the same checked witness. Changed donors or narrowed domains used new structures, and permuting or repeating a domain's codes preserved its key. |

Reproduce with `python -m unittest -v test_r34_seam_audit test_r34_query_reuse test_r31_distribution` in an environment with python-sat and OR-Tools. All 17 tests passed. The computational scope was the finite domains described above.

## #56 Disjoint placement restarts

When a native SAT call returns UNKNOWN at the last permitted call or wall deadline, a fully narrowed defect box can still contain many piece arrangements. The solver now bisects a cell's remaining ordered placement choices and saves both children in the fenced geographic stack. An unsplit geographic search can use the same fallback. Each child stores indices into the original scope's canonical domain. For a finished singleton assignment the physical inventory, score, and all regional counts are checked directly. A placement-dependent UNSAT core closes only its child. A transferable regional proof must be independent of all placement assumptions.

The splitter requires an observed native decision or conflict. Expiry during construction, clause replay, or before native search retains the original box. Conflict slices within an uninterrupted solve keep the same native instance and learned clauses. Legacy stacks containing bare geographic boxes open directly.

The paired test used the planted seed-19, 40-cell, 60-seam problem, all placements allowed, target 60, width 1, and copies of a single prebuilt cap-zero database. Each process-equivalent open had a 0.35-second limit, 2,000-conflict slice and one native call. The repeat control made exactly 2,054 conflicts and 22,097 decisions on each of 16 reopens. With placement refinement the 16 opens used 25,036 conflicts and 275,820 decisions total, against 32,864 conflicts and 353,552 decisions in the control. Fourteen disjoint splits and two independently closed children left thirteen pending branches. Both configurations were open at the measurement boundary. Native effort per time-limited call depends on the machine and load. Run `python bench_r34_band_placements.py` to repeat the comparison. The recorded calls are in `R34_BAND_PLACEMENTS.json`.

In a separate finite control the first placement sibling was UNSAT and the second produced an independently checked 11/15 board after three one-call opens. No regional core was published from the excluded sibling. Further regressions exercised all eight leaves of an infeasible domain through repeated restarts, complete coverage of a 15-leaf interval partition, zero-decision interruptions, read-only reopens, and recovery from an existing bare box. The four suites including `test_r34_band_placements` passed all 22 tests.
