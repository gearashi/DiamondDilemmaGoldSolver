# Extra seams implied by a retained geographic path

Keeping specific named side pairs fixes the relative placement of their tiles once a root tile and its orientation are chosen. Other sides of those tiles can touch. If every internally edge-compatible root places the same two sides together, every edge-perfect board retaining the path must join them. This implication also holds when the goal requires one connected loop.

On the 207-edge reference, 202 of 344 screened paths implied 245 additional individual joins. Seven independent donor sewings supplied more paths. The separate verifier reconstructed the pose equations from side steps and tested the endpoint slots on every induced contact. It checked all 577 implication occurrences across the eight sources. Some occurrences represent the same path in different donors. [Source counts and public inputs](R35_PATH_MATES.json) · [Independent replay](R35_PATH_MATE_VERIFICATION.json)

For six source-labelled queries (five distinct restrictions), the recorded path was required and an inferred mate was forbidden. The sparse geographic baseline explored 159,183–276,306 CP branches in five seconds per query and returned unknown. The geographic path check rejected each input before CP. In the public embedding command, use `--path-mates units` to add common and impossible internal joins. The `table` option also posts the permitted joint patterns where the model represents them. The search recomputes implications against the current geometry and placement domains. These are edge-perfect geographic conditions.

```text
python bench_r35_path_mates.py --tiles data/tiles.json --limit 2 --length 10 --sources reference raw contract near0 near1 trade cp contact --output runtime/path-screen.json
python verify_r35_path_mates.py --tiles data/tiles.json --screen runtime/path-screen.json --output runtime/path-verified.json
python -m unittest -v test_r35_path_mates
```

## A forced seam can close a proper loop

If a path forces seam E, and E together with other named seams closes a circuit shorter than the puzzle's 365 segments, a Gold board must omit at least one seam from the path or from those other seams. The solver rechecks the geometry and the physical loop before using each such restriction. Of 32 distinct substitutions, 19 remain after removing restrictions already implied by a smaller group. The retained groups contain five to nine bonds each. [Public claims and signed costs](R35_PATH_CIRCUITS.json) · [Independent strand, placement and optimizer replay](R35_PATH_CIRCUIT_AUDIT.json)

On four fixed signed assignments, replaying the 19 groups with 194 geographic and 985 existing contact groups changed the minimum number of lost reference bonds from 31 to 33, 33 to 35, 33 to 33, and 37 to 38. These are exact prices for the fixed signs in the stated relaxation. Every full solution in that scope obeys those lower bounds. They give no new lower bound across all signs.

Pass `--reuse docs/R35_PATH_CIRCUITS.json` to `r30_gold.py embed` or `--claims docs/R35_PATH_CIRCUITS.json` to `r32_gold.py loss`, `integer-bound`, or `phase-sewing`. Other claim files can be supplied alongside it in the loss tools. The retained-bond costs depend on which additional claims are supplied.


The same 19 clauses reduced native SCIP nodes on the full 13 + 194 + 985 scope at caps 31, 32, and 33, using the same initial phase, compact Euler/partner constraints, worker and stopping rule. The paired counts were 1,257 to 753, 5,167 to 2,199, and 1,519 to 1,052. Caps 31 and 32 were infeasible in both runs. Both cap-33 runs found a feasible phase in this signed relaxation. Applying all 258 reachable single-mate implications simultaneously gave the same four fixed-phase prices as the 19 clauses. The implied pairs all belong to the existing reference or mandatory inventory. [Simultaneous implication controls](R35_PATH_CLOSURE.json)

The independent optimizer verified all 16 signed costs on the 13- and 344-group scopes. Fixed signed donor sewings were priced separately. With the 19 claims the best single-loop losses increased from 35 to 37 on the first new donor and from 37 to 40 on the older one. The other new donor remained at 39. The resulting connected sewings all have positive genus.

When translated into actual cell assignments, the 19 obligations offer 9,120 root poses. Of these, 600 cannot place the retained bonds, 80 already mismatch an internal edge, and all other 8,440 explicitly contain the original proper loop. The compact clauses contribute earlier in the search when only physical mates are known. No additional GPU placement table is needed for these roots.

A second screen of 1,200 branched components on five sources found four components with correlated optional seams, excluding eight joint values beyond the individual possibilities. Two original cases posted a nontrivial CP table. In matched one-round sparse tests one saved ten branches while another used 1,026 more. Two projected-blank cases used ten and 150 more branches. The full geographic channel also showed mixed work. The default remains unchanged. Use `units` for conditioned queries and the optional `table` to experiment with correlated restrictions. [Branched component screens and public comparisons](R35_BRANCHED_PATH_MATES.json)

```text
python bench_r35_path_mates.py --tiles data/tiles.json --public-controls --output runtime/path-public.json
python bench_r35_path_circuits.py --tiles data/tiles.json --screen runtime/path-screen.json --output runtime/path-circuits.json
python verify_r35_path_circuits.py --tiles data/tiles.json --claims runtime/path-circuits.json --output runtime/path-circuit-audit.json
python bench_r35_path_closure.py --tiles data/tiles.json --screen runtime/path-screen.json --output runtime/path-closure.json
python bench_r35_path_patterns.py --tiles data/tiles.json --repeats 40 --seconds 6 --output runtime/branched-paths.json
```
