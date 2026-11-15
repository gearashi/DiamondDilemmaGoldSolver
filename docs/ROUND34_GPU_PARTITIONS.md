# GPU domain partition comparison

Workstream 61 adds an opt-in CUDA domain search control, **Split the smallest remaining tile domain first**. Its CLI search-option key is `gpu_small_domain_split`. At each split it selects the pending cube with the largest product of domain sizes, picks the smallest cell domain with at least two choices, and bisects that cell’s ordered orientation codes. The two children are disjoint and together cover the original cube. The option also works with the existing propagated-parent fork. Changing the flag during resume preserves the live native columns, search counters, and owned pieces of the frontier.

## Completed finite work

The RTX 2060 exhausted 36 configurations under eight split/fork combinations. The 16 previously measured Round 34 residuals covered four prism restrictions, four restricted original-puzzle boards, and eight planted 10- or 160-cell boards, at 128 requested lanes. The remaining 20 configurations varied lane count and propagation on five held-out finite fixtures. Independent enumeration supplied each expected answer set. The comparison includes both zero-witness proofs and searches that continued after finding the first validated witness.

| Split policy | Native decisions to exhaustion | Native support checks to exhaustion | Decisions by first witness across positive cases | Checks by first witness |
|---|---:|---:|---:|---:|
| Existing newest-first tie break | 196 | 10,485 | 64 | 6,484 |
| Oldest-first tie break | 201 | 11,031 | 65 | 6,514 |
| Largest cube, widest cell | 201 | 11,004 | 65 | 6,491 |
| Largest cube, smallest unresolved cell | 142 | 9,664 | 38 | 6,576 |
| Existing split after shared root propagation | 196 | 5,661 | 64 | 2,268 |
| Smallest unresolved cell after shared root propagation | 134 | 4,894 | 35 | 2,365 |

The smallest-domain split improved at least one exhaustion counter with no increase in the other on 14 configurations, tied on 18, worsened on three, and traded fewer decisions for more checks on one. For the eight-lane plain prism, decisions fell from 22 to 15 and support checks from 683 to 444. With four lanes and full matching on the Hall fixture, decisions rose from 6 to 8 and checks from 84 to 105. The legacy default and existing option for propagating the shared parent remain available. The totals count native decisions and support checks separately. First-witness counts are sampled at the end of the first dispatch reporting a validated witness and include only positive cases.

Each raw partition passed exact oriented-domain volume and pairwise-disjointness checks. A host test enumerates every Cartesian assignment for uneven domains and multiple input cubes. Native stop/reopen controls on a six-answer fixture restored identical completed work counters for all eight split/fork combinations. The live application test toggles the option in both directions, compares native state and portable frontiers, then finishes and validates the result. Browser tests verify the checkbox defaults off and the start request carries the enabled flag for the CUDA domain engine.

## Unrestricted 160-tile puzzle

The original 128-cube newest-first partition has 47 host-certified inventory exclusions. Their total oriented Cartesian volume is 2⁻⁸¹ of the root. The remaining 81 cubes enter CUDA. The other three policies admit all 128 cubes. Oldest-first and largest-cube/widest-cell partitions each own 1/128 of the root per cube. Under smallest-cell splitting, the 128 cubes each own between 1/160 and 1/120. The original one-dispatch screen includes a shared-parent arm that starts with a single lane. Its work is reported separately from the equal-total-slot comparison.

Each raw policy was then given five matching aggregate primitive-slot budgets across all surviving initial lanes, ending at 84,934,656 slots per policy. At that boundary:

| Raw split policy | Closed native lanes | Decisions | Support checks | Domain removals |
|---|---:|---:|---:|---:|
| Existing newest-first | 1 | 163 | 75,374,673 | 950,079 |
| Oldest-first | 0 | 147 | 77,030,262 | 638,968 |
| Largest cube, widest cell | 0 | 147 | 77,030,262 | 638,968 |
| Largest cube, smallest unresolved cell | 0 | 368 | 63,927,205 | 1,142,247 |

All four unrestricted searches remained open with zero verified witnesses. The latest-first native lane closure was in addition to the initial host inventory proofs. Issued slots are per-lane kernel work allowances and a retired lane can return early. The count of domain values removed depends on how the policy partitioned the original board, so these numbers do not establish a faster route to Gold. Keep the default on an unrestricted board. The optional setting reduced completed search work on several narrower prism and piece-inventory residuals. Repeating all 288 finite trials with the final application partitioner produced identical counts and solutions.

See [the native measurement record](R34_PARTITIONS.json). To reproduce all original and finite comparisons, run `python bench_r34_partitions.py --tiles ../data/tiles.json --output audit/r34/partitions.json`. Native CUDA and the original tile file are required for that benchmark.
