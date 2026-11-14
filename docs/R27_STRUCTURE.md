# Structural proposals in round 27

The new opt-in tools cover exact SAT connectivity on sparse residuals and compact regional Gold counting. The existing dashboard search remains the recommended broad-board entry point. These experiments did not find a Gold solution on the published tile set.

## Running the retained alternatives

Run from the project directory after normal setup. A domain file is a JSON array of 160 arrays. Each inner array lists the allowed codes for that cell, where a code equals three times the zero-based tile index plus its clockwise rotation (0, 1 or 2). All omitted restrictions default to the original puzzle domains when the entire file is omitted.

```shell
python r27_structure.py sparse-sat --domains residual.json --max-work 1000000
python r27_structure.py regional-count --domains residual.json --region 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

Sparse SAT requires python-sat. It blocks the selected placements after each answer, so auxiliary SAT assignments never duplicate a board. It applies minimum-degree elimination and switches to checked projected-board enumeration if an elimination step encounters more than 12 neighbors. Regional counting needs a single simple geographic cut. Its default region is the first triangular macro face. The JSON result has a completion flag, a labelled solution count and a checked witness when found. Exit 2 signals a partial search. Only a complete count of zero proves the supplied residual infeasible. These commands do not change dashboard checkpoints.

## Evidence

The [finite comparisons](r27_structure_finite.json) include an existing mixed four-cell residual, noncrossing planted puzzles, repeated tile signatures and a sealed multiple-loop counterexample. The [160-cell comparisons](r27_structure_full.json) include the actual unrestricted input, a fixed private 207-seam arrangement represented only by aggregate results, a planted four-solution residual and a separate positive 365-strand control. The tests also exercise five nested latitudes and random domain restrictions against an independent permutation enumerator. Search counters are compared only when they measure the same operation. Released DFS stops on its first Gold witness. The research enumerators count all labelled boards.

| ID | Experiment and outcome |
|---|---|
| 1 | Computed support for individual physical side pairs across all 480 original darts. Small classes use subset matching, larger bipartite classes use alternating reachability, and the dense even class meets the degree condition for a perfect matching after pinning every edge. Exactly seven bonds were forced and no additional root pair was excluded. A narrowed 160-cell planted board lost placements before ordinary propagation. Post-propagation controls did not demonstrate incremental benefit. Kept as a dependency of assembly experiments. |
| 2 | Chord intervals form complementary disk patches even on same-side arcs. Union of all allowed patch connections detects a sealed third component. Checked local passage assignments by wiring class on narrower domains. The proof detects the deliberately split and nested arrangements. The unrestricted real input admitted all complementary connections. Bounded per-wiring probing exhausted its allowance there. No default or persistent cut cache was justified by measured cross-branch reuse. |
| 6 | Compared explicit seam-label branching and precompiled side-mask tables against placement branching. Both retained the same exact tile multiplicities and physical strand checks. On the mixed finite residual seam branching visited 30 nodes versus 27 for placement branching. Both table representations followed the same decisions. No promotion. |
| 7 | Assembled labeled side involutions and checked oriented vertex-link lengths. Each fully sewn map was embedded in every compatible root placement. Testing the newly joined component for a partial oriented embedding made the 160-cell planted comparison finish with all four answers in 332 nodes. Final-embedding-only assembly exhausted 500,000 work units. Cell search finished in seven nodes. The embedding primitive is available to the assembly repair experiment. |
| 8 | Compared two physical-tour orders, a scarce frontier on the anchored strand component and a fixed directed walk from strand zero. Each joining decision fixes every port of that physical side, including side pairs with no ports. Both hit their 500,000-work limit on the planted 160-cell residual. Neither replaced placement search. |
| 9 | Tested every single- and two-strand separator in the optimistic original 365-node graph, including all 66,430 unordered pairs. The low-link implementation agrees with explicit deletion on 225 random finite graphs. A two-vertex obstruction can survive articulation checking. No additional published-root cut was found. More elaborate degree-two bounds had no observed original-root obstruction to refine. |
| 10 | Binary successor ranks modulo 5, 73 and 365 were tested on a known positive 365-strand cycle and the fixed original incumbent. All preserved the positive cycle and rejected the incumbent. On the four-cell control the small and full divisors each reduced projected candidates from 15 to 6, while requiring 308 and 289 decisions. Lazy validation took 192 and vertex elimination took 175. The released contracted circuit also recovered all six. No default change. |
| 11 | Colored cover uses both cell and tile columns and canonically directed seam colors. It checks closed subloops. A bound deterministic choice-trail cursor permits restarting on the next unprocessed row. Tests concatenate two-node pages, check full solution coverage and reject a cursor bound to other domains. On the planted 160-cell residual cover used 169 nodes versus seven placement nodes. Retained as a reproducible comparison. |
| 12 | Branched over oriented edge-population triples, propagated cyclic type inventories, and lifted each coarse pattern to physical identities and wirings. Added bounded failed-lift contexts only after every child completed. The real root took 28 coarse alternatives within its work cap and did not finish. Finite cases had no reusable failed-lift advantage. No production switch. |
| 13 | Summed attainable Euler contributions from cell patches, subdivided edges and mesh vertices, with the parity equation for the two odd-degree apices. Rejected the sealed split, preserved planted Gold and admitted the five nested latitudes. The latter illustrates why two global Euler totals cannot certify Gold. The root had attainable totals after 382,560 sum operations. There was no root exclusion to strengthen with geometric turning bounds. |
| 14 | Encoded noncrossing disk-cut pairings as parenthesis words, alongside exact inventory, active seam ports and multiplicities. Crossing boundary matchings fall back to explicit pairs. Face, two-face strip and hemisphere controls each counted the same four boards. Measured Python pairing-key objects totaled 56/140/224 bytes versus 1,360/3,976/6,208 bytes respectively, excluding the separately shared active-port and inventory data. Kept the compact encoding for regional enumeration. |
| 15 | Guarded min-degree elimination joins live neighbors through an eliminated mandatory strand. On the mixed residual it visited 6 projected boards in 175 decisions versus 15 boards/192 decisions for lazy validation and 6 boards/18,441 decisions for released SAT reachability. The contracted CP circuit took 595 branches with a different engine. Elimination used 1,036 decisions versus 802 lazy on a repeated-signature case. Kept as an opt-in sparse residual solver. |
| 22 | Computed exact integer Laplacian cofactors over grouped regional states, preserving physical inventory and multiplying by the number of independent lifts. Contracting internal paths makes a connected cut-port cycle contribute its number of ports; every disconnected cycle cover contributes zero. Counts agreed with ordinary unions at all three geographic cuts. Hemisphere unions cost 1,805 total work units versus 10,499 with determinant evaluation. The existing regional union test remains selected. |

The original-root comparator used bounded work; full-board time-to-solution was not measured. The 160-cell planted residual frees eight cells and the remaining cells have singleton domains. Actual input and planted-board outcomes are recorded separately in the evidence.

```shell
python -m unittest test_r27_structure -v
python bench_r27_structure.py --section finite
python bench_r27_structure.py --section full
```

With an original local best-board JSON, the full benchmark also accepts `--incumbent PATH`. It reports only counts and the incumbent score. The two baseline test engines in the finite benchmark require the optional SAT and OR-Tools dependencies.

