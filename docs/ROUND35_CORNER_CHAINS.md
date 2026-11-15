# Vertex-ring constraints on physical joins

A physical side pair determines two successors in the permutation of triangular corners around mesh vertices. The original mesh has five degree-four vertices, two degree-five vertices, and 75 degree-six vertices. A directed path meeting the same triangle twice, exceeding every remaining vertex degree, or closing at a degree with no remaining vertex cannot be embedded. Undoing a join restores both paths and all degree counts.

The physical embed CLI accepts `--corner-mode chains`, `packing` or `positions`. The first checks open and closed chains, propagates newly compulsory mates and can reject a completed nonblank circuit before embedding. Packing also assigns the lengths of disjoint open chains to the remaining degree slots. A capped packing query keeps the candidate. Positions adds a direct CP representation with a common vertex representative, degree and cyclic position on each selected corner successor. A cycle's unique position-zero corner is its smallest physical ID, and the three corners of a triangle must have different representatives. The signed physical-only matching API keeps accepting nongeographic sewings. The geometric rules apply to an oriented closed sphere with three distinct mesh vertices per triangle.

## Original pieces

At the unrestricted root, the physical matching inventory allowed 3,036 pairs before the corner check and 3,035 afterward. Forced bond (318,364) prevents (319,365), since the two bonds would join the same two physical tiles. The partial ribbon genus is zero. Independent rooted placement enumeration ruled out those two bonds alone. This exclusion also follows from simple mesh adjacency and counts once across the two workstreams.

Use the same 60 original reference-bond residuals as the Round 34 physical matching study. Of 41 scopes that passed matching support and forced-strand tracing, corner chains closed ten more; the remaining scopes lost 107 additional candidate mates. All 686 distinct sets of premises involved in trial or fixed-bond rejections were checked independently within their own scopes using side ownership, ribbon genus or direct geographic pose packing. The 107 exclusions on surviving scopes also passed independent checking directly from the incoming fixed mates and the proposed pair. Whole fixed-bond sets alone sufficed for two of the ten new closures. The other scopes require the sequence of proved exclusions and newly forced choices.

| Reference pairs pinned | Already closed | Additional scopes closed | Additional pairs removed on survivors |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0 | 13 |
| 12 | 0 | 0 | 24 |
| 24 | 1 | 0 | 38 |
| 48 | 2 | 6 | 32 |
| 96 | 6 | 4 | 0 |
| 144 | 10 | 0 | 0 |

Length packing added zero exclusions across these original residuals. On an independently enumerated ten-tile control, nine fixed joins survived both rules. A tenth passed the individual path limits but failed joint length packing. All 30 eligible complete physical matchings under those ten pairs independently failed the mesh ring inventory. Keep packing available for such restricted sewings.

The 217-pair original connected sewing from [the blank-side experiment](ROUND35_BLANK_PROJECTION.md) contains an overlong corner chain. Each mode closed it before any CP branching. The earlier genus and pose checks also decide that physical input. The full original position model built and validated with 7,534 variables and 28,708 constraints for 6,070 guarded successors. This model represents all rings without generating complete ring columns up front.

## Positive and adverse controls

All 10,395 pairings of twelve tetrahedral sides agreed with a separate complete successor-cycle census, including rollback to the empty map after every pairing. Three hundred independent small bin assignments agreed with joint length packing. Twelve perturbed ten-tile option sets agreed with R28's exhaustive ring-column oracle on feasibility. Alternate insertion orders, partial rollback, interrupted packing and incompatible mesh degree inventories are covered.

Four three-mover planted scopes used seeds 35, 71, 103 and 157. The position and existing coupled geographic models both found an independently validated Gold witness in one call and zero branches in all four. Chain-only took 10 calls/85 branches on seed 35 and one call/zero branches on the others. Original nonchannel embedding took 13/1,198, 34/8,575, an open 40-call/10,805-branch run, and 9/180 respectively. The existing geographic channel remains suitable on small placement domains. Retain chains for verified mate pruning. Packing and full positions remain explicit restricted options.

```text
python r30_gold.py embed --corner-mode chains --input restrictions.json --seconds 60
python bench_r35_corner_chains.py --tiles data/tiles.json --output runtime/corner-comparison.json
python -m unittest -v test_r35_corner_chains
```

[Measured comparisons](R35_CORNER_CHAINS.json)
