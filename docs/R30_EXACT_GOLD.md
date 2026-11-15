# Exact exchange verification and Gold connectivity cuts

The exact route target counts matched seams with free tile rotations. The Gold methods require a single loop. Each method reports the domain and objective it checked through its inputs, and a bounded solve may return `UNKNOWN`.

## 14 and 17. Cycle-constrained threshold and conditional donor bounds

The CP-SAT verifier represents each moved cycle by distinct actor cells. Successive actors assign their center pieces to the preceding actor. Every other donor is fixed to its own cell. Ordering by least cell removes equivalent cyclic shifts and permutations of equal-length cycles. All cells retain their legal rotations, including those outside the candidate pool. The solver independently decodes the resulting permutation, cycle lengths and matched seam score. A returned full board also gets the separate one-loop validity flag.

For the measured 207/52 center, try a restricted certificate from the project directory.

```powershell
.\venv\Scripts\python.exe r30_exact_routes.py --cycles 4 2 --cells 18 30 44 51 62 146 --target 208 --seconds 5
.\venv\Scripts\python.exe r30_exact_routes.py --cycles 3 3 --cells 18 30 44 51 62 146 --target 208 --seconds 5
```

Both six-cell tests closed at 208 during presolve. The maximum for the first restricted family was at most 205. Its 204 query returned an independently validated witness. Enabling contact and predecessor constraints admitted the same two negative decisions in the measured pool. A deliberately tiny solve budget returned `UNKNOWN`. The independent full-family model with all 160 candidate cells returned `UNKNOWN` for each family with five solver seconds each. Use the specialized physical route command for whole-family work and this verifier for difficult residuals and cross-checks. A ten-cell planted control recovered independently verified Gold boards in both families. A second control enumerated every permutation of six cells to compare the optimized scores.

Supply `--learn --pins pins.json --database runtime/typed-rules.sqlite` to extract a sufficient donor condition for an infeasible target. Pins use zero-based `[destination_cell,source_cell]` pairs relative to the center permutation. Each rule retains the legal-domain, physical-center, movable-pool and cycle-type binding. The builder shrinks an infeasible assumption set, asks for a numeric upper bound and independently rebuilds a fresh unguarded threshold model to confirm the bound. A new query reuses applicable rules automatically when given the same database. It can close an already covered arrangement before a solver call. The existing domain-checked physical movement cores can also load from that database. `--contact` explicitly enables the release-center inter-cycle proof for these four families at 208 or higher. `--predecessors` separately adds the release-center rule for an isolated changed cycle. If none of its cells has a changed neighbor, its total donor gain must be positive to improve on the verified three-swap predecessor. Every actor participates in the adjacency test. The rule is skipped when the domains, physical center, target or cycle class fall outside the proof. Both options are off for an independent trial.

On the 204 control, keeping the donor from cell 30 at cell 18 and the donor from 51 at 44 implied an upper bound of 203. These two literals cover five arrangements in the tested pool. All five repeated queries closed without a solver call. The generalization took six CP calls including fresh confirmation. Retain the rule for recurring scopes. Pre-generating rules at every node adds work on a single pass through this example.

## 19. Separate the endpoint partitions a surviving phase assignment exposes

`r30_gold.py types` uses the signed inventory and Euler model. For each complete phase assignment, it builds the occupied endpoint-type graph. It finds disconnected components of arbitrary size. A connected closed tour meeting both sides crosses the cut at least twice. The cut adds a guarded sum of crossings over individual tile phases. The occupancy guards ensure a phase change that empties one side remains legal. A surviving phase assignment is a relaxation result, labeled `CONNECTED_TYPES`. It supplies no geographic board.

The old five-pin and twelve-pin exclusions each needed two CP calls and two discovered cuts cold. The previously tested fixed-width generators visited 801 and 7,628 candidate subsets respectively. The lazy solver used cut widths 2 and 140 in the first case, 3 and 140 in the second. Reusing those cuts needed one CP call with zero branches in each case. To retain the discovered partitions, provide `--output` and then pass that result through `--reuse` on the next call. Each cut is rebuilt from physical tile endpoints.

```powershell
.\venv\Scripts\python.exe r30_gold.py types --input phases.json --seconds 5 --output runtime/endpoint-cuts.json
.\venv\Scripts\python.exe r30_gold.py types --input phases.json --reuse runtime/endpoint-cuts.json
```

The existing `r29_gold.py phase` command now tries the two small measured partitions and their global color complements on restricted branches. It derives any additional fixed tile phases from signed parity, proves that both sides must contain a strand endpoint and sums the maximum cut crossings available separately on each tile. When that upper bound is less than two, the branch cannot be Gold. It reconstructs this test from the physical tiles. The check requires no incumbent or center permutation, and applies to unrelated starting boards. In both published branches and their two phase complements, it proves an upper bound of zero before generating any subsets or running CP. A phase run with no pins or bonds skips this step. Use `r29_gold.py phase --input phases.json` to reproduce through the existing solver.

An input for the five-pin regression is `{"pins":{"49":0,"62":0,"77":0,"89":1,"126":1}}`. Pins index physical tiles and their internal color phases. They do not set rotations.

## 20. Require compatible bridges to coexist and verify the geographic embedding

The matching model activates one reciprocal physical side pair for every tile side, including empty sides. It channels the tile phases and joins the individual physical strands at their reversed slot positions whenever that side pair is active. An initial set of chosen side pairs, forbidden pairs and per-cell placement domains can restrict the search. Each complete matching induces a disjoint union of cycles. For every proper cycle found, the next matching must supply at least two crossing strand connections on its cut. One seam activation can supply several strand crossings. All cuts constrain a single matching simultaneously. In `embed` mode the solver tries all legal root poses, propagates the entire sewing to the actual mesh, requires consistent injective placements, and sends any candidate to the independent Gold validator. A geographically impossible side matching is excluded as one unit.

Use `r30_gold.py matching --input residual.json` to find a connected side sewing, or `embed` to require actual placements. On a restricted ten-cell fixture the optimistic union of possible connections is connected. Independent enumeration found four compatible complete matchings and zero single loops. Five simultaneous physical cuts closed the scope in two CP calls. A separately planted geographic test returned Gold. The unrestricted real board has 3,060 candidate physical seam activations. The five-second pilot left that relaxation open. Keep this mode for restricted physical work.

A request may contain `bonds` and `forbidden` arrays of zero-based physical side pairs, with a side numbered `3*tile + side`. `domains` contains one list of legal placement codes per board cell. For `capacity`, supply `strand_cuts` containing proper sets of physical strand IDs, in tile and local segment order. The solver maximizes attainable crossings on each cut under all preceding two-crossing requirements. In the four-matching fixture the successive maxima were 4, 2, 4, 2 and 0. Every cut had at least two possible crossings if considered alone. The combined model rejects the scope. `core --output runtime/matching-core.json` tests weakened restrictions and confirms the final core before recording it. `matching --reuse runtime/matching-core.json` can skip a covered query on the same puzzle and cell domains.

For sewings with blank physical sides, the optional [deferred blank-side search](ROUND35_BLANK_PROJECTION.md) can infer the remaining joins from a connected nonblank tour using `r30_gold.py embed --project-blanks`.

## Reproduction

```powershell
.\venv\Scripts\python.exe -B -m unittest -v test_r30_exactgold
.\venv\Scripts\python.exe -B bench_r30_exactgold.py --tiles data/tiles.json --include-root-gold --output exact-gold-results.json
```

These CP-SAT methods use the full Python 3.12 profile. Time-limited root counters vary with machine load. [Recorded results](R30_EXACT_GOLD_RESULTS.json)
