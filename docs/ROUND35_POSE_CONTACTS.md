# Oriented seams between rigid geographic placements

A compulsory placement fixes the orientation of every tile in its component. Two disjoint footprints can still force a mismatched seam when their cells touch. Geographic Gold search now checks those seams under the existing `embed --joint-poses` option. The check also applies to the actual explicit physical bonds passed to that search. It does not turn a named donor assumption into a requirement of unrestricted Gold.

## Scope and work

For each labelled pose the solver indexes occupied cells and the physical dart facing each neighboring cell. An incompatible contact or overlap excludes that pair of poses. It removes a pose only when there is no surviving choice in the other family, and repeats after proven removals. Up to eight constrained families of at most 64 labels each also undergo simultaneous compatibility search. A capped query retains every unresolved option. Full footprint and free-owner matching still apply. For a broad target family a counting bound can prove a non-touching support before looking at labels.

Additional named bonds join their actual component pose tables up to nine tiles. A named edge within an existing component must be realized by its two specified darts. Components larger than the explicit join limit retain the ordinary geographic model. Initial required components, the constrained nine-piece joins and the Gold-only exclusions are independently checkable. The default scored constructor retains disjoint placements with mismatching incidental seams.

| Previously tested original restriction | Codes after footprint packing | Codes after seam checks | Footprint search nodes before / after |
| --- | ---: | ---: | ---: |
| Seed 35 | 73,878 | 73,608 | 11,170 / 10,574 |
| Seed 71 | 73,878 | 73,564 | 11,170 / 10,510 |
| Seed 103 | 73,878 | 73,606 | 11,169 / 10,582 |
| Seed 157 | 73,878 | 72,574 | 11,169 / 10,246 |

All four retain a simultaneous relaxed packing. Six further original-board tests constrained four nearby dominoes. Footprint-only packing retained all six. Real seam contacts refuted seeds 85 and 851. Seed 85 still had 70,606 cell/rotation choices after pairwise support alone. The simultaneous test also removed labels on three other scopes. At the unrestricted root all 3,840 labelled poses survive. A conservative count replaced all 26,880 individual root support tests in the forced control. The runtime skips the optional broad-family pass on this root.

No pair of placements from two of the eight original compulsory dominoes hit a forbidden physical mate while all their seam masks matched, across 6,451,200 pairs. A separate controlled test constructed two pose-compatible, non-overlapping, locally rigid pairs for **each** of the 24 forbidden physical mates. Their incidental boundary alone exposed the contradiction. Twenty-one would close a premature physical loop and three lack a complete perfect matching. These examples apply only when those placements are actually required.

## Pinned endpoint signs

When the call supplies tile signs, the solver propagates their XOR relations over the compulsory and named bonds. An incidental contact must respect every determined relative sign. Quarter-pinning the incoming phase removed 8, 8, 8 and 6 additional codes on the four published restrictions. Full pinning removed 10, 10, 8 and 10. Direct finite Gold controls preserved the witness under its compatible signs and global complement. In the unpinned controls, enforcing relative sign consistency on every double contact between the eight compulsory dominoes added zero exclusions across 6,451,200 pairs. Checking implied signs across all 64 nine-tile trees added none. The unpinned production path skips these extra parity calculations.

## Nine-piece joins and scored controls

On 64 independently grown nine-piece trees from legal original physical mates, direct reconstruction agreed on every footprint and label. Of 29,510 collision-free poses, 8,970 contained one fixed mismatch and 420 had two. Ordinary edge-perfect geography excluded all poses under 17 of the 64 named assumptions. The Gold entry returned `INFEASIBLE` without a native CP call on two checked conditions. In one paired one-second trial the previous native path spent 45,319 branches and remained unresolved. Neither result excludes boards that use other bonds.

On the 208 objective these internal mismatches contribute to the board score. Prioritizing only within the sampled hint ties changed no scores in 64 full fresh constructions. Trying minimum-loss poses first with a fallback to the entire table improved starting scores by a cumulative 31 matched edges and reduced total scanned rows from 522 to 514. Pose nodes remained 473. The maximum-internal-score tier changed no results. In 16 subsequent trials through the actual 64-round signed fresh-board search, minimum-loss priority lost five total matched edges. Its second eight seeds lost 20. None reached 208. The solver keeps the existing scored trajectory and its checkpoint identity.

Direct enumeration checked every code surviving the declared constraints on 64 restricted four-tile fixtures. Controls cover repeated and parallel seams, actual Gold and scored counterexamples, named cycles, a supported sparse mesh, and budget exhaustion. The reproducible original and held-out measurements are in [R35_POSE_CONTACTS.json](R35_POSE_CONTACTS.json).

```sh
python -m unittest test_r35_pose_contacts test_r35_joint_poses test_r35_speculative_poses
python bench_r35_pose_contacts.py --tiles data/tiles.json --output projection.json
python bench_r35_signed_pose_contacts.py --tiles data/tiles.json --output signed.json
python bench_r35_incidental_joins.py --tiles data/tiles.json --output fresh.json --count 64
python bench_r35_pose_contact_outcomes.py --tiles data/tiles.json --output choices.json --count 64
python bench_r35_pose_contact_starts.py --tiles data/tiles.json --output trajectories.json --count 16
```
