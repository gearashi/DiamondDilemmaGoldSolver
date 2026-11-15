# Geographic phase anchors and fixed-phase board SAT

Run the independent checks from the repository root using the project's Python environment. The full audit uses NumPy, python-sat and OR-Tools. The anchor-only check runs with NumPy on Python 3.15 as well.

```text
python verify_r35_phase_board.py --output phase-board-audit.json
python verify_r35_phase_board.py --checks anchors
```

The [recorded receipt](R35_PHASE_BOARD_AUDIT.json) contains counts and SHA-256 digests of the audited source files with LF line endings. The auditor reads raw tile strands, toggles color at each endpoint around the physical tile boundary, and compares all twelve open intervals in reverse order at a seam. It builds its own finite boards from those intervals.

| Check | Result |
| --- | --- |
| All-different placement enumeration on four ten-face fixtures | 15,680 whole-board occurrences, 50 signed edge-perfect, 31 with one physical loop |
| Required and forbidden mate subsets and pinned tile phases | 36 positive scopes and 60 empty scopes, with 768 anchor calls |
| Projected solutions of the exported DIMACS formula | All 116 formulas agreed with the interval oracle. There were 20 positive formulas, 96 empty ones and 88 projected witnesses across them |
| Complementary signed phases | 50 equalities checked over every phase on the two-cell and four-cell spheres |
| Roots constrained to a macroface | Eight comparisons with different anchor tiles, face choices, gauges and loop structures. Each restored all ten distinct unrestricted layouts from one representative |
| Public search | 12 Gold returns, six infeasible returns and four matched invalid-input errors between baseline and anchored search |

**Anchor scope.** Each physical tile must occupy one cell. Placing a tile partitions the residual by its cell and orientation. A candidate that survives in a legal board always has a neighbor on every incident seam, so deleting unsupported neighbor values preserves that board. If a cell's surviving orientations all belong to one tile, only that cell can use that tile. A proper rotation preserves these statements, including physical tile phases and named mates. The implementation reuses branches only under rotations that also preserve the input domains. Rejecting a whole failed orbit or intersecting pose exclusions across every surviving orbit image leaves these symmetries intact for later splits. The audit checks roots with trivial and order-two stabilizers and 24 deterministic deadline points, including four interrupted splits following an already completed reduction.

**SAT scope.** Every cell and every physical tile has an exactly-one constraint. At each mesh edge, the equality of the two selected signed side labels is equivalent to matching all twelve colored intervals. Enumerating models while blocking only the selected placement literals compares entire board sets, independently of how the auxiliary variables are assigned. The auditor checks the serialized DIMACS header and clauses, empty cell domains, a missing tile, duplicate fixed tiles, and five malformed domain inputs. The input phase remains fixed under simultaneous rotation of board and pieces. A macroface restriction is complete for the unrestricted canonical bipyramid because its proper rotations act freely and transitively on the ten macrofaces. A satisfying arrangement may have several gold loops. The normal validator determines the loop count before the solver reports Gold. These finite cases give no verdict on existence of a solution on the full 160-tile puzzle.
