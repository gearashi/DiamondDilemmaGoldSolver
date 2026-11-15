# Complete responses for disconnected rotation components

The rotation oracle removes additive edge terms into per-cell unaries. The remaining nonzero pair terms partition the fixed-permutation problem into disconnected components. Their maximum seam scores add. A cached response binds every residual pair entry, every unary contrast, all domain widths and all outgoing trace axes. The current unary constants are added back. The solver restores its original global elimination and rotation enumeration order when producing boards.

Isolated unaries have direct maximum and tie-mask answers. Larger component responses reuse all conditional optimal-rotation masks. For two or three cells the engine can reorder vertices and rotation ranks. It defers generating complete optimal tuples until another numeric labeling needs them, then maps those tuples into the current elimination separators. Score offsets, cell and rotation permutations and ties all move together. The full numeric key checks equality after canonicalization. Component storage shares an explicit byte budget with numeric factors and structural plans. A cap that disallows a factor still returns unresolved. A factor observer receives the original complete ordered trace, and an independent cancellation callback remains active when components hit.

## Work and scope

Across 58 queries the ordinary factor-cache and structural-plan path built 18,105 numeric factor keys. The retained policy built 2,064 plus 1,946 whole-component keys and 264 small-frame canonicalizations. Numeric grid entries fell from 593,183 to 592,579. These counts cover two matched seeds, each with twelve adjacent original exchanges, six incumbent-free permutations, three rotations of the incumbent and eight factor-cap crossings. Every result retained the exact optimum, first witness and up to four ordered witnesses where available, and the same unresolved outcomes.

| Queries | Numeric factor keys before | After | Whole-component keys |
| --- | ---: | ---: | ---: |
| Twelve adjacent exchanges | 2,868 | 467 | 128 |
| Twelve held-out adjacent exchanges | 2,689 | 536 | 109 |
| Six fresh permutations | 4,000 | 345 | 701 |
| Six held-out fresh permutations | 3,520 | 350 | 560 |
| Both repeated-incumbent cohorts | 960 | 0 | 0 |
| Both cap-crossing cohorts | 4,068 | 366 | 448 |

On a complete original floor-207 one-swap neighborhood, both implementations returned identical independently verified physical witnesses for all 88 optima and the same two qualified neighbors. The normal command built 784 numeric factor keys and 363 component keys, compared with 14,646 numeric keys alone. The numeric grid total fell from 628,257 to 628,166. The best remained 207/52.

## Extensions and adverse controls

The screen compared singletons alone, component limits of four through 160 cells, absolute and offset-normalized unaries, rank relabeling and vertex reordering, directional masks and complete tuple projections. Enumerating extra indistinguishable rotation-rank permutations through three cells made the same numeric equivalence classes as the stable sorting rule. Exhaustively permuting small vertex orders captured a few further hits at several times the canonicalization work. The stable variable ordering retains the measured drop in factor work.

Lazy materialization encoded 153 complete tied tuples across the main comparison instead of 446 eagerly. Extending vertex transport to four cells saved eight further factor keys and 15 grid entries, with 61 additional canonicalizations and two more tuples encoded. Five through eight cells yielded zero further factor or grid savings. Three cells remain the normal vertex-transport limit. Larger components still have exact directed-mask reuse without vertex reindexing. Highly connected boards whose cost comes from one wide factor can retain most of their numerical work.

The independent finite checks enumerate all physical assignments for relabeled chains, independent components, additive cross terms, tied rotations and changed operands, including 65,536-valued control entries. They cover interrupted work, very small byte budgets, all and first-only witnesses, a stricter factor cap after a successful warm solve, and the original callback behavior. Run `python -B -m unittest test_r35_rotation_components test_r35_rotation_graph` for these checks. `bench_r35_rotation_components.py` reproduces the original and held-out profiles. `--plateau-only DIRECTORY` compares all frozen physical witnesses on the shipped complete one-swap neighborhood. The optional rank and vertex comparator scripts are `bench_r35_rotation_frames.py` and `bench_r35_rotation_vertex_response.py`.

The optimization is active automatically for ordinary cached rotation solves, including ranked physical routes and the plateau command. `RotationOracle(model,components=False)` is the factor-only ablation.
