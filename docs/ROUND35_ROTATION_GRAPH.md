# Round 35 rotation thresholds and elimination plans

The original 160-piece inputs were priced against the released R28 oracle at `fb67a56`. Thirty queries covered the incumbent, twelve independently exchanged boards, six fresh placements, three initial rotation patterns on a fixed physical order and eight width-cap queries. Both oracles returned identical scores or cap results, numeric message-entry counts, and reconstructed witnesses in the same order up to the first four ties. Each returned witness was separately scored on its physical seams. The incumbent remained at 207. [Absolute work counts, input hashes and witness hashes](R35_ROTATION_GRAPH.json)

| Query group | Star maxima, previous → current | Seam support vectors, previous → current | Neighbor products, previous → current |
| --- | ---: | ---: | ---: |
| Incumbent | 2,880 → 800 | 8,640 → 3,360 | 13,040 → 160 |
| Twelve nearby exchanges | 55,932 → 10,916 | 167,796 → 51,591 | 169,401 → 2,635 |
| Six fresh placements | 72,504 → 1,916 | 217,512 → 25,386 | 326,000 → 2,719 |
| Same physical order, changed starting rotations | 8,640 → 2,400 | 25,920 → 10,080 | 39,120 → 160 |
| Two placements at four width caps each | 48,656 → 4,432 | 145,968 → 29,496 | 194,887 → 572 |

## Threshold propagation

Strictly lower a rotation domain only when its conditioned upper bound falls below the requested score. Fixing one rotation changes support on incident seams. For a single seam into a neighbor, the neighbor's maximum drops by one exactly when every previously maximizing rotation loses that support. Parallel seams have a joint price calculation. Upper bounds, maximizing rotation sets and two-hop loss records update after a domain changes. A fall in the global sum can also invalidate a distant rotation, so cells whose loss exceeds the remaining slack wake even across disconnected components.

Each lower target starts from the original domains. Intersecting these domains with a previous stricter target would be unsound. Support tables use one of 512 possible three-by-three binary patterns. Conditioned losses can be reused when the entire relevant two-hop domain matches. A seven-variable finite fixture exhibits three remote domain restrictions missed by local-only scheduling, reducing the first component from 3/3/2 choices to 2/1/1. Independent full sweeps and exhaustive scoring cover all 256 pairs of two-rotation path tables as well as multigraphs, disconnected components and changing thresholds. [Tests](../test_r35_rotation_graph.py)

## Structural plans

The oracle keeps the same minimum-product order and cell-index tie break. Only the remaining neighbors of the eliminated cell acquire new neighborhoods. Widths and the nonadditive pair graph key the reusable plan. Numeric factor bytes and actual rotation labels still independently determine scores and reconstructed ties. A cap-crossing plan retains the remaining neighborhoods and unchanged products. Raising the cap extends it from that point.

The six fresh boards reused twelve plans and built thirteen. Twelve nearby exchanges built thirteen and had zero complete-plan hits, while local neighborhood updates still saved work. The cap control had eleven hits and two extensions. Charged structural storage reached 372,776 bytes in these groups, within the default 512 KiB structural limit. With an explicit factor-cache byte budget, numeric factors take priority and both caches share that budget. Interrupting a numeric observation and retrying produced the same full set of tied witnesses. The numeric cache's work remained 1,639 evaluated entries across the six fresh placements and 221,189 across the twelve nearby exchanges. [Bench harness](../bench_r35_rotation_graph.py)

A former mixed CUDA/WebGPU classic checkpoint also reopened on the host across the two actual rotation source changes. Its complete diagnostic archive grew from 194 to 195 files. Its independently checked terminal counters remained 230,400 / 0 / 480. [Controlled source crossover](R35_ROTATION_GRAPH.json)
