# Regional reasoning experiments

The `extended_regions` option enables bounded exact regional reasoning. It is
experimental and off by default. It can remove choices that survive independent
regional queries; it is not a claim that the original Gold puzzle is solved.

A regional summary contains physical inventory, every boundary mask, exact
pairing of exposed strand endpoints, and completed-cycle count. Per-cell support
unions alone are insufficient for overlapping regions: the overlap path instead
uses exact assignments represented by a reduced multi-valued decision diagram.
Only completed stages publish deletions. A timeout or state/frontier budget
means the unfinished stage is unknown; earlier complete deductions remain valid.

| Proposal | Implementation and tested scope |
| --- | --- |
| 21 | Scoped live summary sets propagate across a network to fixpoint. An eight-cell ring retained all independently enumerated Gold completions while removing one extra orientation beyond repeated two-region domain filtering; six incompatible summaries were removed. |
| 22 | Three-region joins compose the complete endpoint pairing. A six-cell fixture removes four orientations beyond pairwise compatibility while retaining all Gold completions. |
| 23 | Overlapping regions agree on exact shared assignments. A four-cell fixture removes three orientations beyond independent region fixpoints and retains all three Gold completions. |
| 24 | Whole-inventory allocation is solved jointly across disjoint regional alternatives. A four-region case proves a shortage that pairwise compatibility leaves unresolved. |
| 25 | Singleton outside chains participate in compatibility checks. A fixture with identical local masks but a forbidden global loop is rejected; its valid counterpart is retained. |
| 26 | Two- and three-port separator tests count only mandatory components in an optimistic strand graph. Removing two vertices from the tested K2,3 obstruction leaves three mandatory components. Independently constructed cycles and small puzzle Gold completions survive. |
| 28 | Region size can grow beyond 16 when frontier/state budgets permit. A 40-cell ring query completes with two histories; a planted 160-cell query completes with a wider configured frontier budget. |
| 29 | Several separator plans are compared and exact selected plans cached by cell set and domain sizes. Tests verify repeated plan reuse and agreement with exact finite results. |
| 30 | Recorded regional conflicts affect later region ranking. This changes priorities only; a search-step advantage has not been established. |
| 31 | A reduced exact assignment diagram shares identical suffix relations. All 720 assignments of a six-cell ring survive, using fewer than 100 internal diagram nodes. |
| 32 | Component alternatives join only after inventory and strand-interface compatibility are established. A four-component fixture removes one extra orientation beyond pair/triple consistency and joint inventory allocation. |
| 50 | Completed local tables can persist to a bounded JSON cache. Scope includes puzzle fingerprint, target, exact domains, format version, and producer-source hashes. Repeated tests avoid recomputation; changed scope and corrupt records are rejected. |
| 51 | Endpoint-supply bounds combine cell capacity with distinct-piece capacity. A fixture rejects a demand that would require using one high-capacity piece twice. This bound may overlap existing seam/Hall deductions. |

Proposal 27 (wider seam cuts) is implemented separately in `constraint_advanced.py`.

Disk tables are a **trusted local computation cache**, like local search state.
Checksums detect accidental corruption; they do not authenticate deliberately
modified tables whose checksum was recomputed. Do not import outside regional
cache files as proof certificates. The separate certificate format independently
checks its supported Hall and proper-loop witnesses.

The reproducible tests are `test_constraint_region_network.py`, together with
`test_constraint_regions.py`, `test_constraint_region_pairs.py`, and
`test_constraint_advanced.py`. They cover exact small-board enumeration, random
domain restrictions, target separation, cancellation, caps, persistence,
correlation preservation and planted full-board retention.

The clean CPU ordering rerun used an immutable current-source snapshot after
an exclusive benchmark window was confirmed. All 90 runs completed and every
returned Gold board passed separate validation. On ten repeated-mask ten-cell
fixtures, solution ordering used 100 total search nodes and 255 propagations,
versus 104 nodes and 286 propagations for the baseline. It won on five instances,
tied on three, and lost on two. Impact ordering tied baseline work in this corpus.
Conflict ordering used more nodes (112) but fewer propagations (239). Signature
branching used 120 nodes, winning once and losing seven times; combined ordering
used 121 nodes. All configurations tied on five ordinary planted fixtures.

These are finite completed-search comparisons, not an original-puzzle solve-time
claim. Fresh records are in
`audit/optimization-round23/regions/ordering-clean-rerun-final/`. Earlier records
from the overlapping competing-program session remain preserved but are excluded
from release performance evidence.
