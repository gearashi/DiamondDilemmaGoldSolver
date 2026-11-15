# Alternating exchanges with a phase for each child

The directed endpoint labels of a selected physical seam fix one XOR between the phases of its two tiles. Two complete parents can have contradictory tile phases even when a subset of their seams gives a valid child. The ordinary crossover command now falls back to child-specific phase equations when the common spherical phase query fails or its optional CP backend is unavailable. The common-phase CP path retains its spherical Euler constraint. Physical counts in mixed mode require signed endpoint compatibility and a single strand circuit. A geographic claim also requires an embedding on the actual board and a complete Gold validation.

The search takes one whole half of each alternating cycle, including cycles containing blank sides. It rolls back both the relative tile phases and the components of the labelled strands. A bounded counting cache includes the signs on the remaining tiles, the remaining strand components and the identity of cycles assigned by lookahead. Geographic searches keep the named seam choices until the board has been validated. Exhausting a node allowance returns a partial physical count and an unknown geographic result if the roots survived.

## Original donor families

The five connected original donors have one circuit through all 365 labelled strands. They comprise the original raw and contracted tours, the CP-generated alternative to the contracted tour, and two tours with 183 and 182 retained reference seams. A sixth donor, the synthetic long-cycle trade, has eight circuits. All six have complete 240-seam matchings.

The table gives exact signed, connected offspring for thirteen pairings. The five mixed pairs that triggered this work are listed first. Each reported count agrees between cached variants and direct endpoint traces on bounded independent slices at both parental anchors. Every measured complete mixed family admits one of its two original parental phases on every connected child.

| Parents | Connected full children | Memoized nodes | With forced-cycle lookahead |
|---|---:|---:|---:|
| Raw / contracted | 652 | 5,876 | 798 |
| Raw / 183 | 4,492 | 7,969 | 2,537 |
| Raw / 182 | 5,560 | 12,990 | 1,558 |
| Contracted / 183 | 7,251 | 17,995 | 5,820 |
| Contracted / 182 | 3,000 | 10,216 | 2,653 |
| Raw / synthetic trade | 28 | 2,267 | 164 |
| Raw / alternative CP | 1,500 | 2,468 | 765 |
| Contracted / synthetic trade | 15,464 | 5,682 | 3,436 |
| 183 / 182 | 12 | 16 | 14 |
| 183 / synthetic trade | 25,702 | 31,754 | 12,297 |
| 183 / alternative CP | 40,136 | 76,807 | 30,390 |
| 182 / synthetic trade | 4,245 | 10,194 | 3,645 |
| 182 / alternative CP | 10,254 | 25,550 | 10,361 |

For contracted / alternative CP and synthetic trade / alternative CP, a 100,000-node diagnostic counted at least 220,447 and 180,649 connected children, respectively. Geographic search resolved both entire families. The benchmark preserves the partial flags on their abstract counts.

Looking ahead 1, 2, 4, 8, 16 and 48 cycles preserved the exact counts. One forward pass through up to 48 later cycles gave the fewest or tied for the fewest entered nodes among these windows. Each check rejects a half only for a signed contradiction or an already closed proper strand loop. A second pass to a fixed point reduced entered nodes again in twelve of the thirteen counts, but repeated additional individual half tests. The default uses one forward pass. Separating the two donor phases from a residual phase search increased entered nodes in all eleven mixed families. The single combined phase search is the default. [Deterministic counts and controls](R35_MIXED_SEWINGS.json)

## Geographic closure for repeated exchanges

The union of the five connected parents contains 714 distinct physical seams. All six together contain 773. A geographic board using these seams must place tile zero in at least one of 480 cell/rotation combinations. On each assumed root, the solver revises neighboring pose domains using all allowed mates of a physical side. It can reject the root when a cell runs out of poses. For the five-donor union 11,280 cell revisions eliminated all 480 roots. The six-donor union required 10,860 revisions.

An independent Glucose 4.2 encoding creates one literal for each of 76,800 cell/pose combinations. For each directed board edge it lists every allowed supporting neighbor pose. Each cell needs a pose, and selecting a tile-zero root removes other poses at that cell. Unit propagation on these 460,480 clauses rejected all 480 root assumptions for each union. The relaxed encoding omits tile uniqueness away from the root, phase, and circuit constraints, making the exclusion valid for every geographic placement restricted to those seams. All fifteen pairings also passed an independent enumeration of concrete unique-tile partial placements at their 480 roots. The six-donor exclusion applies to any number of successive exchanges or combinations between them because none can introduce a seam outside the union.

A subsequent contact-priced physical pairing contributes 19 new seams. Its three circuits contain 8, 99 and 258 strands. Both independent methods reject every root for the extended 792-seam union. Two subsequent cost-preserving exchanges reconnect that donor into a single 365-strand tour. It adds one further seam. Both methods again eliminate all 480 roots for the eight donors' combined 793 seams. Subsequent complete pairings from the 117- and 985-contact portfolios expanded the union to 852 seams. All 480 roots still fail in 12,150 revisions and in the independent SAT check. This conclusion depends on the named seams. Allowing all signed mates under each donor phase leaves 430, 460 and 460 roots in the three respective phase tests. Their 2,782-seam union leaves all 480 roots. [Contact-priced donor](R35_MIXED_CONTACT_DONOR.json)

On ten-triangle planted Gold instances, direct enumeration at seeds 103, 224, 310, 384 and 392 each found a connected child requiring a phase admitted by neither parent. At seeds 2164, 3521 and 5388 a child of this kind also embeds on the intended board. The production geographic search recovered all three. The tests read the named physical seams back from each returned arrangement, check both parental assignments fail and validate the complete geographic circuit. A zero-node search on viable domains reports unknown. Empty domains report geographic exclusion. Ordinary matched-edge scoring is independent of these Gold criteria. [Regression and held-out controls](../test_r35_mixed_sewings.py)

Run `python r35_alternating_sewings.py --parents parents.json --union-roots` to screen any number of complete donors together. Use `python r35_alternating_sewings.py --parents parents.json --geography` for a new pair of complete physical sewings. On the documented six donors, another physical count cannot improve the geographic decision. A new donor with a seam outside the extended union can be screened by adding its seams and checking the geographic roots before opening a new physical count. The unrestricted Gold puzzle and board scores above 207 remain open.

With prepared original tile data and the optional Python SAT and CP packages installed, `python bench_r35_mixed_sewings.py --tiles data/tiles.json --output replay.json` reproduces the full original matrix and both independent union checks. The planted phase and Gold regressions also run on the Python 3.15 NumPy-only profile. The independent SAT test is optional there.
