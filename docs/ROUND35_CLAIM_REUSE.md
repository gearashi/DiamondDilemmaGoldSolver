# Reusing verified geographic claims in a fresh process

The original mesh now recognizes 205 unconditional geographic obstructions, 988 edge-perfect contact obstructions and 19 geographic implications that create a smaller physical loop. The 205 include the original 194, ten further residuals and the 29-bond obstruction. The first family fails even plain placement. The second can be placed, but every placement has a mismatching seam. The third uses the actual strand pairing inside tiles to rule out a single Gold loop. Each proof type has a separate authenticated claim catalogue. [Independent evidence and corpus hashes](R35_CLAIM_CATALOGUE.json)

The independent verifier packed every geographic claim from scratch. Scalar enumeration found 358,200 geometric placements across the 988 contact claims and zero with all incidental edges matched. The 19 circuit premises collectively have 8,520 edge-matched placements. Every one contains its claimed extra seam and every corresponding circuit seals a proper subloop. The verifier checks the unrestricted original board so its exclusions also hold after fixing tiles or narrowing cell domains. A warmed narrower query can still return its own smaller conflict. [Raw verifier](../verify_r35_claim_catalogue.py)

| Fresh public query | Prior material checks | Current material checks | Prior elapsed | Current elapsed |
| --- | ---: | ---: | ---: | ---: |
| Signed integer, feasible at 33 | 194 packing, 985 contact | 0 | 36.79 s | 1.84 s |
| Ranked signed, infeasible at 31 with Euler | 194 packing, 985 contact | 0 | 35.06 s | 0.82 s |
| Weighted certificate, infeasible at 30 | 195 packing, 985 contact | 0 | 32.33 s | 1.00 s |
| Physical sewing, infeasible | 194 packing, 985 contact, 19 path | 0 | 35.90 s | 0.53 s |
| Geographic embedding, unrestricted cells | 194 packing, 985 contact, 19 path | 0 | 26.23 s | 0.49 s |
| Geographic embedding, tile 0 fixed in cell 0 | 194 packing, 985 contact, 19 path | 0 | 20.26 s | 0.49 s |

The two embedding queries impose the same previously verified set of six required seams. Each returns that same impossible set before a native CP call. Each compared public result has an identical canonical hash after removing the new reuse counter. The table reports local wall times under other project load. Every timed case used a fresh interpreter. [Complete work counts and result hashes](R35_CLAIM_COLD_RESULTS.json)

The public `r32_gold.py loss`, `integer-bound`, and `phase-sewing` commands apply the lookup automatically for supplied claim files. The `r30_gold.py embed --reuse` command does the same for saved claims. Both live and published paths check the actual destination reference and compulsory bonds before converting raw geographic groups to a loss clause. Unknown or mixed claim families go through live certification. Reordered claims, physical pairs written in reverse order, or duplicate supplied rows preserve the same proof. Modified strand pairs, masks or mesh block the shortcut.

For independent claim replay, use `--independent-native` with the integer-bound or phase-sewing command, or `--phase-anchors off` with geographic embed. Direct claim verifier functions also use their existing live checks. The two measured independent command controls each performed all 1,198 packing/contact/path checks.

```text
python verify_r35_claim_catalogue.py --tiles data/tiles.json
python bench_r35_claim_catalogue.py --tiles data/tiles.json --output runtime/claim-lookup-results.json
python -m unittest -v test_r35_claim_catalogue
```
