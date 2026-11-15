# Extra seams implied by a retained geographic path

Keeping specific named side pairs fixes the relative placement of their tiles once a root tile and its orientation are chosen. Other sides of those tiles can touch. If every internally edge-compatible root places the same two sides together, every edge-perfect board retaining the path must join them. This implication also holds when the goal requires one connected loop.

On the 207-edge reference, 202 of 344 screened paths implied 245 additional individual joins. Seven independent donor sewings supplied more paths. The separate verifier reconstructed the pose equations from side steps and tested the endpoint slots on every induced contact. It checked all 577 implication occurrences across the eight sources. Some occurrences represent the same path in different donors. [Source counts and public inputs](R35_PATH_MATES.json) · [Independent replay](R35_PATH_MATE_VERIFICATION.json)

For six source-labelled queries (five distinct restrictions), the recorded path was required and an inferred mate was forbidden. The sparse geographic baseline explored 159,183–276,306 CP branches in five seconds per query and returned unknown. The geographic path check rejected each input before CP. In the public embedding command, use `--path-mates units` to add common and impossible internal joins. The `table` option also posts the permitted joint patterns where the model represents them. The search recomputes implications against the current geometry and placement domains. These are edge-perfect geographic conditions.

```text
python bench_r35_path_mates.py --tiles data/tiles.json --limit 2 --length 10 --sources reference raw contract near0 near1 trade cp contact --output runtime/path-screen.json
python verify_r35_path_mates.py --tiles data/tiles.json --screen runtime/path-screen.json --output runtime/path-verified.json
python -m unittest -v test_r35_path_mates
```
