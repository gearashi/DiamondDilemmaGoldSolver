# Binary residuals, regional paging and checked boundary reuse

These routines take explicit cell domains. Their certificates cover the supplied domains and objective. A complete Gold witness also passes the independent puzzle validator.

## 10. Inventory-priced cuts and tied solutions

`python r28_logic.py priced-cut --data data/tiles.json --domains binary.json` accepts one or two distinct legal orientation codes per cell. The output contains an upper bound on matched edges, a checked board when available, work counters, and integer inventory prices. Pass `--prices previous.json` to initialize the prices on a neighboring residual. Every price adjustment vanishes on a legal full tile permutation.

The routine charges a competing piece in quarter-defect units and uses at most eight rounds. Positive residual flow arcs describe all minimum source sides. The solver checks up to 128 tied cuts per round for legal inventory and one-loop Gold, and only strengthens a bound from exhausted ties after certifying every optimum cut was illegal. Every completed cut is a valid bound. Each returned board is rescored. The reported `optimal` status certifies the soft matched-edge objective. Read `gold` for the one-loop result.

## 11. Reusable implications, cyclic inventory choices and frustration

The existing binary DFS tail and `scc` command now build the two-choice implication graph once per binary scope. A Gold-loop rejection branches on changes to that loop. Each branch accumulates forced implications, reuses the parent graph, and can reproduce individual contradiction or unit paths. The full model is independently checked after propagation.

The priced cut also contracts choices linked by compulsory inventory and alternating cycles. It conditions on up to three representative variables that cover all remaining sign-inconsistent arcs after rephasing. The resulting subproblems admit exact minimum cuts. Larger cutsets use the previous doubled-graph upper bound.

Of 600 seeded four-cell physical restrictions, exhaustive enumeration found 391 inventory-feasible cases. The released cut gave the exact upper in 186. The new method gave the exact upper in all 391, including all 59 with frustrated signed interactions, and returned an optimal board in 390. Without contraction, pricing and tie enumeration tightened 163 bounds over a one-cut, one-witness control. Taking additional tied cuts in the first round recovered an optimal witness in 86 cases where the first cut missed it. The incremental Gold implication search consumed 13,654 work units over 500 two-piece residuals. The earlier graph rebuilds consumed 20,781, with matching outcomes in every case.

Around the installed 207 incumbent, separate scopes with 8, 32 and 160 binary cell domains all attained and certified 207 under both methods. The new cut reduced these to 4, 12 and 54 free variables respectively. The full-width scope used the doubled-graph fallback. These scope-specific controls support enabling implication reuse by default and using the priced cut for residuals with competing inventory.

## 12. Durable table compilation and empty boundary prefixes

`python r28_geometry.py regions --method paged --input regions.json --checkpoint runtime/region.json` stores the join cursor as JSON and the tables in an adjacent SQLite file. `--table-cache PATH` shares that database with another checkpoint. Each file must have its own path. The default `--table-work 1000000` caps compilation transitions per region per call. `--max-work` caps join cursor steps per page. Reinvoke the same command to continue.

Each regional table is bound to the physical puzzle, region cells and their local domains. DFS path, next choice, partial multiplicities and first row witnesses commit together. Complete tables and seam indexes survive restarts. Sorting complete row patterns supplies stable row identifiers. A domain change recompiles only the affected regions. Proven empty join prefixes retain complete boundary ports, strand pairings and inventory for all selected and remaining rows, permitting reuse when those logical patterns recur.

On a ten-cell, three-region five-step-page control, both releases returned 108 labelled solutions in 12 complete joins. Cold compilation dropped from 2,672 node visits and 9,984 strand visits to 167 and 624. A deliberately tiny 17-transition table allowance finished over 54 calls. A warm pass used zero compilation nodes. Narrowing one cell recompiled one region and gave the same 90 solutions as an independent cold join. A split-loop case reused an empty prefix both on warm replay and on the related narrower domain. The regression suite also interrupts an in-flight database transaction and verifies the recovered count.

## 13. Resolution proof interpolation and hierarchy

`python r28_logic.py proof-interpolate --data data/tiles.json --domains scope.json --cells 0 1 2 3 --subregion 0 1` builds a refutation for a physically encoded regional conflict. It reconstructs and independently checks resolution steps from unit implication conflicts or the existing isolated native RUP producer where available. A bounded elimination fallback handles small scopes. `--proof-work` controls the reconstruction limit. Exhausting it returns `unknown`.

Each requested nested region derives a boundary from the same checked proof. Two independent SAT queries verify the implications on either side of that boundary. Results include the circuit, private gates and output literal. `--prior-boundary previous.json` rebuilds the circuit with fresh private variables and rechecks it against the current pair of formulas before skipping a join.

On a six-variable parity conflict, boundary-cube enumeration used 98 SAT calls and 32 boundary clauses. The new path used five in-process SAT calls for two checked boundaries from one portable proof, including their semantic checks. Rechecking a boundary on a related region used two calls. On the real 8- and 32-cell binary restrictions, each proof supplied two checked boundaries. At 160 cells, a 250,000-unit proof allowance returned unknown. Use bounded proof interpolation on repeatedly encountered regional conflicts.

## Reproduce

```text
python -B -m unittest -v test_r29_regional test_r28_logic test_r28_geometry
python -B -m benchmarks.bench_r29_regional --samples 600 --tiles data/tiles.json --incumbent runtime/best.json
```

The benchmark uses deterministic synthetic data and an optional read-only 160-cell reference. OR-Tools and python-sat are required for the full benchmark. Core DFS and durable regional joins also run in the portable dependency profile.
