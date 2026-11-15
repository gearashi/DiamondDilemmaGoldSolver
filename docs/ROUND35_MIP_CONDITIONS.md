# Spherical and physical mate pruning in the signed integer solver

The original 207-bond reference needs **33 departures in the signed-inventory model with spherical Euler weights**, when the model includes all 13 original, 194 geographic and 985 induced-contact departure groups. Two different signed assignments achieve 33 and each passes the reciprocal mate checks for families of up to 16 physical sides. The earlier 31-loss signed witness has spherical weight −69. The admissible raw weights are −5, −3, −1, 1, 3 and 5. These signed assignments give no full geographic placement. [Per-run work, raw witnesses and independent finite census](R35_MIP_CONDITIONS.json)

The optional SCIP backend now posts the exact Euler sum before solving. With mate support enabled it posts short missing-partner constraints and replays the local matching failures of known reference phases on the current input. On the original data this starts with 249 distinct native clauses. If another phase lacks support, the checker can exclude every completion of a smaller verified assignment. The mate-cost checker keeps each price exclusion in the active loss scope. The solver can stop as soon as it finds one eligible candidate within the requested cap.

| Original query with Euler and mate support | Prior whole-phase rejection | Native partial constraints | With 19 additional geographic path-to-loop obligations |
| --- | ---: | ---: | ---: |
| Loss ≤31 | 2,643 nodes, 1 rejected phase, infeasible | 1,257 nodes, 0 rejected phases, infeasible | 753 nodes, infeasible |
| Loss ≤32 | 6,010 nodes, 10 rejected phases, unknown at 600 seconds | 5,167 nodes, 0 rejected phases, infeasible | 2,199 nodes, infeasible |
| Loss ≤33, starting with the earlier 37-loss donor | — | 1,519 nodes, cost 33 | 1,052 nodes, cost 33 |

Euler alone refuted 31 in 1,523 nodes and 32 in 4,730. For the older 326-contact scope, posting the short partner and Euler rows refuted 31 in 1,477 nodes. The same build accepted the known 37-loss donor at zero branch nodes when given that phase. All three original unrestricted negative comparisons include the same 43 independently checked redundant signed cycles. The two optional physical constraints have separate meanings. A positive in the partner-support relaxation can still cost more when individual physical mates must be chosen consistently.

The first 33-loss phase, `0xcc57eae4e99109f8b073557e8c590b9202465766`, has a width-12 matching price of 34 and admits a complete physical pairing at 34. Its connected sewing costs 35. The held-out 33-loss phase, `0xcc778d82e99c517987f6b57ec15837c21207cf06`, has a width-12 price of 37, a full physical pairing at 38 and a connected sewing at 39. Both phases have Euler weight −5. The second still has signed loss 33 after adding all 19 independently checked geographic path-to-loop departure groups. Public signed searches validate the actual charged groups before accepting either phase.

Four authentic outside-error patterns each have 16 sign assignments after parity elimination. Exactly one assignment per pattern satisfies integer inventory. At loss 39 the compact native model and the independent enumeration agreed on all 16 combinations of source pattern and requested conditions. On the 21-loss source, the ten-pin Hall obstruction refutes a whole family of phases. Each of the ten single-pin deletions restores a matching in that relaxed local graph. The known supported 37-loss phase passed all four conditions. Three ten-tile planted boards also underwent exhaustive sign enumeration, including a zero-loss Gold fixture and a board with no accepted assignment.

Installing all 714 gauge-reduced missing-partner clauses or explicitly allocating 649 possible mate edges over 39 classes trades stronger propagation for more native work. The eager linked-price model missed a known narrow-scope cost-24 positive within 15 seconds. The smaller model accepted it at zero branch nodes. In the 13-group diagnostic without contact premises, the sphere-only question at cap 21 remained open after 39,911 native nodes and 750 seconds. It has its own scope and stays outside the broad reference-loss result.

Run the following from the installed project with the optional native CPU dependencies and prepared original tiles. The first command requests the wide relaxation. The second runs a matched comparison without published decisions.

```text
python r32_gold.py integer-bound --all-groups --claims docs/R35_GEOGRAPHIC_RESIDUALS.json --claims docs/R35_EDGE_CONTACTS.json --integer-backend scip --sphere-euler --mate-support --upper 33 --seconds 600
python bench_r35_mip_conditions.py --tiles data/tiles.json --scope complete --cases 32/em/31 33/em/37 --modes whole warm --seconds 600 --first-feasible --output native-signed-comparison.json
```
