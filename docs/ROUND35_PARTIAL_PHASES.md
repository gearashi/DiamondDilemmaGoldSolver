# Partial phase exclusions for geographic edge matching

Two verified assignments on 48 and 49 physical tile signs exclude signed edge-perfect boards on the published mesh. Each global color complement also excludes every board in its scope. The [certificate](R35_PARTIAL_PHASE_CORES.json) contains the signed tile literals, all 22 inventory-feasible phases per assignment, the individual obstructions and a signed-inventory counterexample for deleting each pin.

The two assignments contain 44 common pins. They have the same 22 completions under exact signed inventory. Their partial queries differ. When the 49-pin assignment is supplied to the existing parity projector, testing the 48-pin rule still leaves five relative signs to decide. In the other direction six signs remain. Both clauses therefore short-circuit actual partial inputs. The earlier fixed-phase rule and the two earlier geographic clauses left both direct conditions open, with and without the eight known compulsory mates. With the new rules toggled off, the public embedding search completed four CP rounds per gauge and returned unknown after 91,186 and 91,170 branches on the 48-pin inputs, and 62,789 and 62,795 branches on the 49-pin inputs. Enabling the rules returned infeasible in all four cases with zero native calls.

Nineteen of the 22 full phases leave a closed class of physical sides without a perfect matching. Two further phases leave no possible board position for a physical anchor. The last one is the independently checked [fixed-phase DIMACS and DRAT proof](R35_FIXED_PHASE_PROOF.json). Every obligation uses unsigned side positions and signed boundary colors on the original mesh. These exclusions also apply when full physical mating, a single loop, more pins or narrower geographic domains are required.

Run the independent verifier with the project's Python and prepared tile data.

```text
python verify_r35_phase_cube.py --tiles data/tiles.json
python -m unittest -v test_r35_phase_cubes
```

The verifier paints every boundary interval directly from the raw strand endpoints. It derives integer balance for reversed side profiles and even multiplicity for self-reversing profiles. Bounds and Gaussian elimination prune the census. It checks both gauges independently in 373 nodes per gauge for the 48-pin assignment and 43 per gauge for the 49-pin assignment. For the two geometric leaves it tests every orientation of the anchor in every cell, covering 1,920 labelled roots across both gauges. It links the final leaf to the signed-mesh, board-CNF and DRAT hashes in the fixed-phase manifest. The proof archive named in that manifest supports a separate native DRAT replay.

The public rule loader authenticates the signed mesh and the certificate before treating either conjunction as a proved exclusion. The tests cover a different board, altered artwork, a damaged edge list, tampered rule data, budget exhaustion and an independent enumerated positive. The original census and complete leaf verification passed under Python 3.12 and 3.15. No claim about unrestricted Gold existence follows from these partial signs.

## A third exclusion verified by a board proof

A separate search freed the signs of tiles 5, 9, 15, 36, 38, 80, 89 and 141 in the same original phase. Exactly six of their 256 settings can balance the signed physical sides. Five of the six were outside the previous two rules, and four of these five also pass both the Euler and physical matching prerequisites. A labelled board SAT formula allowed all 160 tiles and all three rotations. Proper rotations of the whole puzzle allow tile 25 to occupy any cell on one chosen macroface. The formula includes an independently enumerable 22-clause restriction on the eight free signs. CaDiCaL refuted the resulting 272,720-variable, 793,278-clause formula. An independent run of drat-trim checked 817,211,113 resolution steps. [Formula, DRAT hashes, and downloadable asset name](R35_PARTIAL_PHASE_PROOF.json)

Exact enumeration from the raw endpoints then found all the same six phases with only 50 signed tiles fixed. The independent verifier visits 775 nodes in each color convention. An inventory-valid neighboring phase witnesses each of the 50 single-pin deletions. The older public search finished four CP calls with 104,009 and 106,340 branches on the two conventions. The verified rule rejects both partial inputs before the first CP call. [Public search and coverage counts](R35_PARTIAL_PHASE_DIRECT_COUNTS.json)

To regenerate the DIMACS with python-sat installed, run the command below. The file is identical under the two tested Python hash seeds. The release proof archive also includes the original DIMACS so that drat-trim can directly check the DRAT against it.

```text
python bench_r35_partial_phase_board.py --tiles data/tiles.json --phase 0xd237aba7e91101e88041413e8c589902021e5126 --anchor 25 --mode labels --variable 5,9,15,36,38,80,89,141 --inventory --output source.json --formula board.cnf
drat-trim board.cnf board.drat
```
