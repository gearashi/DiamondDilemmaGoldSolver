# Shared support work and checkpoints

The multi-center search works on physical permutations at exactly four swaps. It optimizes every legal tile rotation on an endpoint that survives the upper bounds. Each rank denotes one set of moved cells. The different permitted donor cycles over that set form its endpoints. A returned `remaining_supports` counts full sets still to process under the selected scope. It counts neither the entire puzzle space nor time to a witness.

## 11. Ownership of overlapping endpoints

The solver orders centers by their physical placement vectors. Each eligible endpoint has an owner in that order. It checks the relative permutation and its cycle lengths, including domain-preserving rotations. An odd difference between centers rules out an intersection of their even four-swap shells in that orientation. Earlier certified radii can also own an endpoint. The combination-rank checkpoint belongs to the union of the selected centers and families. An interval looked at in isolation may defer an endpoint to its owner's support in another interval. Negative coverage of the entire union requires finishing every interval or consuming a checked whole-family certificate.

The two outer centers share 2,848,460,420 eligible six-cell endpoints, 76,683,186,580 seven-cell endpoints and 588,862,564,290 eight-cell endpoints after the earlier radius-five and three-swap exclusions. Their sum is 668,394,211,290. The counts use the actual three-cell center difference. For every possible intersection pattern with those three cells, the benchmark enumerates the relative cycle types and weights the other cells by their binomial multiplicity. These are mathematical overlap counts. A core or contact check can already have eliminated the same endpoint.

## 12. Shared local prices

When the support is disjoint from the distance-two neighborhood of every changed identity between two centers, their local price changes agree. Each center still uses its own starting star total. For a partial overlap the solver caches contributions by the ordered physical identities entering each star, and recalculates changed terms. On 100 independently sampled supports of each size, three stand-alone center queries cost 300 assignment bounds. Shared queries used 213, 228 and 228 at sizes six, seven and eight. Every shared price was checked against the independently computed price.

## 15. Decision diagrams

The exact max-sum diagram keys a partial assignment by used donors, boundary choices of still-open star factors, directed path endpoints and lengths, and the remaining cycle partition. It retains the best prefix price for each key. A width-limited relaxation can drop cycle history while retaining inventory and all active boundary choices. When loss of history leaves a possible threshold hit, the exact diagram refines it. Every emitted exclusion is an upper bound. Full rotation optimization remains the authority for threshold hits.

Across three sampled supports per family, exact merging reduced transitions from 2,694 to 2,026 for the two six-cell types together, 5,409 to 3,686 for seven cells, and 4,488 to 3,678 for eight. The complete scores matched exhaustive distinct donor orders. Smaller transition counts within this control do not make the diagram cheaper than the existing assignment filter on easy supports. Add `diagram` after the other filters when investigating hard supports.

## 16. Measure coverage once

The default filter order is `contact,core,star,family`. Cells within distance two provide a quick upper bound on how many contacts between different cycles a support can supply. A single vector operation discards a support when every center requires more contacts than it has. A surviving support gets the applicable movement, assignment and cycle-type bounds. Endpoint geometry and predecessor tests are stronger once actual donors are known. Only certified centers use their corresponding previously established rules. A four-swap defect budget can close a complete scope at an unrelated center before constructing any support table.

The same 1,024 ranks starting at 200,000,000,000 were compared with the contact rule first and with it disabled. At seven cells, adding the verified two-contact requirement cut support assignment solves from **1,366 to 79**, with 595,980 of 645,120 center-labelled endpoints first ruled out by geometry. At eight cells, the verified three-contact requirement cut the solves from **1,660 to 394**, with geometry first ruling out 238,770 of 322,560. Both orders closed every tested endpoint. On a separate matched set of 360 center queries, core-first needed 287 star bounds versus 360 with star-first. That comparison deliberately included 24 supports sampled outside a known core.

An immediate SQLite transaction records the disjoint page interval, first-exclusion counters, independently checked witnesses and new exact optima. A failed page can be retried. Committed pages are skipped even when filter order, page length or backend changes. During a long CLI run, a progress line reports committed supports remaining, the next support rank and exact solves so far. A CPU-to-CUDA replay of the seven- and eight-cell ranges generated zero new pages. Queries sharing one database serialize their page writes. The CUDA core filter agreed with CPU on 98,304 center queries, including 2,346 exclusions. Both devices also agreed on the geometry of all 32,768 eight-cell supports, of which 17,464 lacked two contacts and 26,619 lacked three. GPU filter-only timings are recorded separately from end-to-end search steps.

```powershell
.\venv\Scripts\python.exe r30_multicenter.py --family 2,2,3 --database runtime/four-swap-union.sqlite --backend cpu --seconds 0
.\venv\Scripts\python.exe bench_r30_shared.py --tiles data/tiles.json --cuda --output shared-results.json
```

Use repeated `--incumbent` arguments for different starting boards. Omit them to use the three release centers. Use `--start` and `--stop` for half-open rank intervals. `--recheck` bypasses the shipped six-cell completion, and a fresh database gives cold measurements. [Raw measurements](R30_SHARED_RESULTS.json)
