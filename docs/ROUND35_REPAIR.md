# Fresh repair and swap packing in Round 35

## 32. Which three-cycle repairs improve fresh boards?

The new `r31_fresh.py --contact-group 1` prices individual profitable two- and three-piece cycles and packs mutually independent moves. Set `--pair-cycles 2` as well to restrict the maximum cycle length to two. The familiar `--pair-cycles 3` alone retains the v1.15.0 mixed-pair policy. Contact-group values 2 and 3 additionally price two or three touching, disjoint cycles. These options have separate trajectory identities.

Four held-out independent CPU starts used seeds 3500-3503, no candidate hints, the original domains and all eight mandatory physical Gold bonds. Every policy began from identical validated physical codes within that seed, and the four seeds produced four distinct starts. The initial scores were 11, 12, 12 and 16. Each standard repair used a ten-cell randomly chosen pool, whole-board cycle budget three, and the same independent full-board validation. Every best board retained the required mates and excluded the 24 forbidden pairs.

| Policy | Best at 64 rounds per seed | Best at 320 rounds per seed | Rounds to 160 | Rounds to 180 |
|---|---|---|---|---|
| Released eight-bond baseline | 174, 170, 169, 173 | 179, 176, 177, 174 | 11, 23, 13, 13 | unreached on all four |
| Released `--pair-cycles 3` | 175, 170, 169, 173 | 181, 181, 177, 174 | 11, 14, 13, 13 | 120, 236, unreached, unreached |
| Individual cycles, raw or factored | 175, 170, 169, 173 | 181, 181, 177, 174 | 11, 14, 13, 13 | 120, 236, unreached, unreached |

The individual-cycle arms reproduced the released pair policy's exact best codes, components, milestone rounds, unique-board counts, orientation entries and CP branches at both 64 and 320 rounds. Across 1,280 repair iterations with each policy, the released option priced 364,800 legal cycles, and 601 mixed cycle pairs. Seven pair candidates improved their local comparison but the trajectory gained nothing from evaluating those pairs on these starts. Individual cycles required 44,646 bounded CP branches downstream versus 42,162 for the original baseline, while full-board assignment relaxations fell from 2,304 to 2,024 and rotation entries from 9,333,108 to 7,480,170. Bounds/descent calls, repair iterations and rotation entries are different work units. Timings were collected under concurrent CPU work and were excluded from selection. Every 16-to-64 and 64-to-320 comparison continued its own prefix. Independent cold 64-round replays on seed 3500 matched their split-path boards and work.

The factored individual-cycle kernel prices each allowed rotation of each candidate donor at each position against unchanged neighbors once. It shares those values through all permutations of local cycles. For moved cells sharing a seam it corrects the double-count using the exact oriented interaction and compares the witness with a full rescore. The 1,280 matched repairs would otherwise materialize 8,812,800 allowed orientation tuples in individual cycles. The new code evaluated 384,000 individual destination/donor/rotation slots, separated 346,353 whole cycles, bounded away 18,422 more and examined 243 entries in jointly coupled orientation grids. It retained the same four physical 320-round results. Positive controls, missing orientations and pinned pieces are included in the finite tests.

**Decision.** Use `--contact-group 1` for sustained independent CPU starts when prioritizing the measured seam thresholds on this hardware. The inexpensive eight-bond baseline is still available. The released mixed-contact correction has a separate use on preselected interacting neighborhoods. A changed score threshold/deeper search without materially different physical opportunities is insufficient to reopen its general-setting comparison.

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --contact-group 1 --rounds 320 --starts 0 --database runtime/fresh-independent-cycles.sqlite --output runtime/fresh-independent-cycles.json
```

## Contacting pairs and triples

When two cycles each improve the board independently, their best rotations can still conflict on their common seam. Earlier code omitted all these positive-positive pairs. The new pricing indexes disjoint cycles by actual shared seams and retains the best legal orientation for each exact interface response. When three cycles form a contacting path, their two leaf orientations are independent given the central cycle's orientation. The algorithm maximizes both leaves separately before selecting the best central orientation. Every triangle still sees every compatible combination of compressed interfaces. Tied maximizing responses retain the earliest original witness. The packer keeps the baseline packed score if adding alternatives makes its large-component greedy result worse.

Each row below covers sixteen original-board pools, eight from arbitrary legal fresh placements and eight after eight independent-set repair steps. The numbers sum gains across different boards and do not denote one placement's score.

| Pool | Released gain | With positive-positive pairs | With connected triples | Pools with a further strict gain from pairs / triples |
|---|---:|---:|---:|---:|
| Six geographically connected cells | 26 | 29 | 29 | 2 / 0 |
| Eight geographically connected cells | 35 | 40 | 46 | 5 / 6 |
| Ten scattered cells | 52 | 53 | 54 | 1 / 1 |

Across those 48 pools the physical seam index considered 60,057 disjoint, actually touching cycle pairs, out of 826,960 raw pair combinations. Pair response compression reduced eligible orientation-product entries from 16,556,967 to 2,762,253. Triple compression narrowed 145,824,057 raw products to 69,039,900. Path factorization then used 3,188,073 pairwise leaf entries and 204,942 central rows for the path cases; 56,894,832 complete compressed triple entries remained on triangles. These counts describe local pricing work and exclude index construction. Independent finite controls enumerate legal physical rotations, verify every connected triple occurs once, and check exact witnesses including ties and restricted domains.

A 64-round eight-cell-forest trajectory used the released, expanded pair and triple policy on the same four seeds. All three finished with identical best board hashes (178, 168, 172, 169). Applying each local stage before the full-board circulation instead produced 176/171/170/174, 177/171/169/173 and 177/170/171/173 respectively. This earlier order shortened the time to 160 on some seeds and lowered some final scores. Keep the new two/three-cycle rules as opt-in exact local repair capabilities; reject the early reordering and their promotion over individual cycles for ordinary sustained search. No method in this comparison reached 208 or Gold.

[Per-seed scores, witness digests, work and 48 local comparisons](R35_CYCLE_REPAIR.json). Reproduce with `python -m benchmarks.bench_r35_trajectories --help` and `python -m benchmarks.bench_r35_cycle_contacts --help`; use an isolated output directory, the installed physical tile file, and seeds 3500 through 3503 for the trajectory arms.

## 33. Select neighboring swaps by their physical seam gains

Each candidate exchanges two nonadjacent cells in its individually best legal orientations. Different candidates may touch one another. The new packing stage retains one move per cell and computes changes to shared seams directly from both replacement codes. A neighboring change can make a previously rejected move profitable, so only candidates touching newly changed cells are reconsidered. `--touch-packing priced` keeps the better of a zero-correction pack and a signed marginal pack. `complete` also solves each connected component of up to 18 candidate swaps exactly, including all signed pairwise corrections. A seam touches at most two disjoint selected moves, so these pair corrections determine the exact physical objective for a fixed list of oriented swaps. The original witness wins ties.

The `cp` profile adds bounded whole-list CP-SAT when a component exceeds that limit and there are at most 512 candidates. It encodes one Boolean per fixed move, at most one move per cell, and the actual edge mask comparison on every seam. It starts with the feasible previous pack, keeps that pack after UNKNOWN or a tie, and checks any improved outcome against the full board. The candidate-count gate separates the tested moderate pools from the large, low-yield controls. Its numeric cutoff is an engineering bound.

Each of 24 additional original-board starts was tested over all free cells and a 30-cell sample. Twelve starts were raw placements and twelve were polished with eight independent repairs. The sums below add strictly comparable one-step gains on separate boards. All 48 returned legal physical arrangements.

| Scope | Zero correction | Signed | Signed plus exact small components | New cases helped by the last stage beyond signed |
|---|---:|---:|---:|---:|
| Full board, summed extra seams over released exact packing | 259 | 262 | 265 | 2 cases, 3 extra seams |
| 30-cell pool, summed extra seams | 22 | 23 | 23 | 0 |

On a 12-seed subset, bounded CP added 12 further seams across all six raw 30-cell cases. Nine of the 12 pool models proved optimal and three returned feasible improvements. None of the 12 full-board calls improved the existing witness. The six large raw cases returned UNKNOWN. Increasing deterministic effort from 0.2 to 2.0 on two raw and two polished held-outs confirmed all four pool optima, including 30 matches versus 27 and 28 versus 26 on the raw boards. Both large raw full-board calls found a feasible incumbent only and gained zero. The four starts at higher effort are a subset of the first 12.

A planted triangle of three mutually touching exchanges gains only 2, 1 and 3 seams individually. Applying them together recovers the known 15-edge one-loop solution from 3 matched edges. Independent physical enumeration also checks the small-component and CP optima, including legal orientations, endpoint conflicts and retained ties.

## 34. Split nonbipartite conflict components with bounded residual work

`--separator-packing` first uses the released exact routines for components of up to 22 candidates and for bipartite graphs. For an unresolved component of up to 64 candidates it branches on including or excluding an articulation vertex, or a vertex on a recovered odd cycle. It combines disconnected residuals and retries the released exact routines. A depth-three limit leaves unproved components to the released packer. An exact optimum tied with the existing witness retains that witness. Exactness here applies to the noncontacting subset of the stated fixed candidate list.

Across the 48 physical controls, the extension attempted 62 bounded nonbipartite components and proved nine. Eight matched the released witness value. One 30-cell case gained one seam; there was no gain on the 24 full-board cases. The additional local seam disappeared when signed packing was already active. Independent 25-vertex positives for both articulation and odd-cycle branches verify gains over the old fallback; a tied odd cycle verifies identical returned move IDs. Leaving equal-weight ties to the branch solver changed two held-out 320-round bests. Retaining the released tie policy restored one of those results.

## Sustained comparison and settings

All variants below used four independently validated eight-bond starts with the same codes by seed, no archive hints, CPU sampling, full-board cycle budget three, a ten-cell repair pool, and rounds 16, 64 and 320. The 64- and 320-round runs reused the previous deterministic prefix. A separate cold 64-round replay for seed 3500 reproduced every variant's best board and work. Counts for a threshold are rounds per start, and a dash means that start did not reach the threshold.

| Packing policy | Best at 320 rounds, seeds 3500–3503 | Rounds to 180 |
|---|---|---|
| Released exact packing | 171, 182, 181, 180 | –, 155, 315, 210 |
| Zero-correction contacts | 176, 176, 180, 176 | –, –, 220, – |
| Signed contacts | 176, 176, 183, 178 | –, –, 147, – |
| Signed plus small exact components | 178, 176, 181, 178 | –, –, 204, – |
| Signed, small exact and bounded CP | 177, 177, 171, 181 | –, –, –, 105 |
| Separators, released ties preserved | 171, 182, 181, 179 | –, 155, 315, – |
| Separators, counterfactual arbitrary ties | 171, 182, 178, 179 | –, 155, –, – |
| Released exact packing plus individual cycles | 183, 176, 181, 175 | 90, –, 273, – |
| Exact small signed components plus individual cycles | 178, 176, 174, 178 | –, –, –, – |
| Bounded CP, separators and individual cycles | 179, 177, 174, 180 | –, –, –, 253 |

**Decision.** Retain signed, component-exact, bounded-CP and tie-preserving separator modes for the tested local objectives. Reject arbitrary tie replacement, unconditioned full-board fixed-code CP, and enabling all the options as the ordinary sustained profile. On these four starts the released `--exact-packing` reached 180 most often. The individual-cycle mode reached 160 within 11–14 rounds on all four and reached 180 on two. Adding individual cycles to exact packing attained the earliest 180 (round 90) and a peak of 183 but also missed 180 on two starts. None reached 208 or Gold. A materially different repeated physical repair distribution or an independently verified target improvement would warrant a new comparison.

For independent discovery at the measured 180 threshold:

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --exact-packing --rounds 320 --starts 0 --database runtime/fresh-exact-swap.sqlite --output runtime/fresh-exact-swap.json
```

[Physical pools, branches, per-seed milestones and board digests](R35_SWAP_PACKING.json). Reproduce with `benchmarks/bench_r35_packing.py`, `benchmarks/bench_r35_fixed_packing.py` and `benchmarks/bench_r35_trajectories.py` using the original tile file. Fixed packing optimizes only the supplied oriented candidates. A feasible CP result establishes its checked score. A proof of optimality on that scope never closes a full unexamined board.

## 35. Retain profitable swap candidates when the board changes

The standard fresh repair now keeps an in-memory move catalog per trajectory. A candidate is reconsidered when one of its two endpoint codes changes, an active endpoint sees a different facing mask, or a previously protected endpoint becomes eligible. Removing a cell from the allowed set also removes its candidates. Dirty/dirty and dirty/clean pairs form disjoint sets. Existing positive rows keep their order, and newly priced pairs merge by gain, then by their original cell order. A second stage reuses the noncontacting packing only when the ordered endpoints, gains, packing policy and geographic model agree. It puts the current replacement codes on the current board and reprices signed contacts there. It keeps at most 32 exact graph keys and 256 KiB of key bytes per trajectory.

Four standard exact-packing trajectories, 320 rounds each, represented 13,178,880 possible unordered pairs. The released calculation built every current gain matrix. Strict selective maintenance reconsidered 1,699,827 pairs, of which 1,672,625 needed nonadjacent donor pricing. A trial that switched back to a full rebuild once 60% of pairs were dirty reconsidered 1,737,456. All 16,376 positive ranked entries agreed with full reconstruction, including replacement codes and tie priority. A completely unchanged ranking applied on 331 rounds. The active set and original domains are part of the test.

| Original-board edit, 32 calls per control | Available unordered pairs across the calls | Pairs reconsidered |
|---|---:|---:|
| Rotate one movable piece | 407,040 | 32,094 |
| Exchange two physical pieces | 407,040 | 50,980 |
| Apply four exchanges | 407,040 | 141,435 |
| Replace the eight protected cells | 367,232 | 46,615 |
| Independently regenerate the whole board | 407,040 | 407,040 |
| Repeat an unchanged board | 407,040 | 12,720 |

Under standard exact packing, a one-entry history recovered 269 nontrivial graph packings. Capacities of 2, 4 and 8 recovered 278, 283 and 289. Capacities 16, 32, 64 and 320 recovered the same 289. The case with no recurring nontrivial graph was retained. Across four actual 320-round matching trajectories per policy, bounded reuse avoided 289 standard graph solves, 505 separator graph solves, and 172 graph solves before signed CP. Empty and single-candidate lists are excluded from these gains. The largest observed charge for graph keys was 48,120 bytes. Positive counts sum work across independently run policies.

**Decision.** Retain selective candidate and noncontacting-graph reuse transparently. A full price pass is appropriate for a fresh state or when every available pair has a dirty endpoint. All twelve sustained production runs reproduced the frozen pre-catalog best codes and logical search counters. Cold or resumed workers reconstruct the volatile catalog from their legal board on first use; existing saved trajectory formats and work receipts are compatible. No new search setting is required. Reopen only with evidence of useful nontrivial cross-process reuse or a distinct dependency omitted by the present masks.

[Direct comparisons, physical controls and per-seed outputs](R35_SWAP_CATALOG.json). Reproduce with `bench_r35_swap_catalog.py`. The finite regression checks every legal pair of rotations against an independent full score, including an unchanged swap invalidated by rotating its excluded neighbor.

## 36. Disjoint donor assignments and regional optimality certificates

For a conflict between changed positions (i) and (j), the two children now describe “(i) stays” and “(i) changes, (j) stays”. A required change may rotate the original tile in place. If its alternate rotations are illegal, that row must take a different donor. `--cycle-partition neighbors` also pins every local neighbor of a required change. `warm` keeps tight, surviving arcs and the feasible parent row and column duals, then augments the unmatched rows. It checks the completed assignment's legality, distinct donors, dual feasibility and primal/dual equality.

On 36 original-board regions of 8, 12 or 22 cells, all four policies proved the same physical optimum for every region. The released overlapping search needed 6,194 assignment solves and produced 74 repeated relaxed physical boards. Disjoint branching used 4,977, neighbor propagation used 4,438, and warm matching used 4,459. The three new branches produced zero repeats. With the smaller initial budgets the old search completed 30 regions, compared with 32 for each of the new variants. These certificates cover only the indicated regional cells and fixed exterior.

Independent cold and warm Hungarian solvers were evaluated on each identical matrix along the bounded warm search. Across 1,677 regional calls, reused assignments needed 2,428 augmentations versus 34,464 cold and traversed 25,276 alternating columns versus 176,176. Eight full-board budget controls needed 1,411 versus 17,280 augmentations, and 60,386 versus 667,186 alternating columns. Full-board controls with just 3 or 24 relaxation calls per region remained incomplete. Neighbor propagation sometimes found a weaker early primal under those small budgets. On complete small controls, independently enumerated actual rotations and donor permutations agreed with the certified results.

A second use applies to the ordinary local CP-SAT stage when it allows zero contacts. Its geographic slots, donor inventory, changed-code definition, legal rotations and boundary objective are then the same as the regional nonadjacent circulation. `--regional-cycle-budget N` uses a proved regional witness immediately. If the budget ends, it runs the ordinary model on the original base and retains the better verified board. Other contact settings keep the ordinary CP model. `--regional-only` makes the bounded partial witness the explicit fallback for a no-contact run. It is labeled feasible and gets a separate checkpoint identity, including the installed assignment backend. This supports the Python 3.15 NumPy profile without local CP.

The unchanged assignment has score delta zero. When potentials `p` satisfy `p[i] - p[j] >= gain[i,j]` on every legal arc, row potentials `p` and column potentials `-p` prove an assignment upper bound of zero. The current legal board attains that bound. In 1,280 actual local calls, the zero potentials already certified 1,241. One update certified 27 more and two updates certified eight. Four calls required the ordinary assignment solver. The dual was also checked against all permutations on independent small matrices with unavailable arcs and positive cycles. Among the 36 harder regional pools, it replaced 11 matching calls and preserved all 36 complete optima.

| Policy, four independent 320-round starts | Best matched seams per start | Local CP branches | Regional assignment solves after identity proof |
|---|---|---:|---:|
| Released full-board exact packing and ordinary local CP | 171, 182, 181, 180 | 57,280 | – |
| Disjoint whole-board branches and ordinary local CP | 171, 182, 181, 180 | 57,280 | – |
| Whole-board forced-neighbor branches and ordinary local CP | 178, 173, 179, 183 | 54,738 | – |
| Whole-board warm branches and ordinary local CP | 178, 177, 173, 178 | 52,614 | – |
| One local assignment bound before CP | 180, 176, 181, 175 | 1,007 | 5 |
| Two local assignment bounds before CP | 180, 176, 184, 175 | 0 | 5 |
| Two local bounds plus individual cycles | 183, 176, 181, 175 | 0 | 0 |

The two-bound profile reached 180 on seeds 3500 and 3502 at rounds 180 and 202. Budgets 8, 32 and 128 returned the same codes and search work before the identity shortcut. Independent 16-to-64 and 64-to-320 extensions and a cold 64-round replay returned the checked codes and counters for each policy. At budgets 8, 32 and 128 on the separate harder regional distribution, 23, 27 and 30 of 36 pools, respectively, were proved before direct CP. All 12 CP-proved optima agreed with completed regional scopes; CP returned UNKNOWN on the other 24 at its usual effort.

**Decision.** For fresh discovery with the priority on avoiding repeated optimization, use the existing full-board `--exact-packing` and add `--regional-cycle-budget 2`. The earlier 180-attainment comparison still favors the ordinary local CP profile on three out of four starts. The new profile skipped all 1,280 local CP models and attained 184 on one fresh board. Adding `--contact-group 1` eliminated all local assignment solves as well, with different final scores. Keep unrestricted CP for requested contacting neighborhoods and tune the regional budget upward only for demonstrably harder regional pools. Keep neighbor propagation and warm matching for regional closure or repeated assignment work; their ordinary full-board trajectories did not improve the default recommendation.

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --exact-packing --regional-cycle-budget 2 --rounds 320 --starts 0 --database runtime/fresh-regional.sqlite --output runtime/fresh-regional.json
```

For the NumPy-only Python 3.15 installation, add `--regional-only` and use a separate database. The actual 3.15.0 installation with NumPy 2.5.3, eight mandatory bonds and no OR-Tools reached 155 after 16 rounds, then 161 after 32 with 16 reused and 16 new rounds. The retained regular Python 3.12 and portable Python 3.15 options return physical heuristic boards; 208 and one-loop Gold remain open.

[Regional optima, matched assignments, trajectory receipts and native/portable controls](R35_CIRCULATION.json). Reproduce with `benchmarks/bench_r35_cycle_branches.py`, `benchmarks/bench_r35_identity_duals.py` and `benchmarks/bench_r35_trajectories.py`.
