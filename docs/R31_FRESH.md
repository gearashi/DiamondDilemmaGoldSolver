# Fresh Gold construction and whole-board donor repairs

The independent search can preserve mandatory physical side bonds, select its moved cells jointly with their donors, and checkpoint each seed. All reported scores pass the separate tile/mesh/strand validator. Gold requires 240 matched seams and one closed loop. These are witness searches. Only the local assignment subproblems with a proved optimum carry an exact bound.

## Hardware settings

For independent Gold-aware candidates on the measured machine, use the full Python 3.12 environment. The CPU command achieved 170 or higher on all four held-out starts within 64 rounds and retained all seven mandatory physical bonds. The current project Python executable is shown as `python` below.

```shell
python r31_fresh.py --backend cpu --bonds --cycle-budget 3 --rounds 64 --starts 0 --database runtime/fresh-bonded.sqlite --output runtime/fresh-bonded-best.json
```

Use `--backend cuda --replicas 512 --dispatches 256 --sampler-cache runtime/fresh-sampler.sqlite` to initialize the same repairs from the RTX 2060. `--rounds 320` allows a longer individual trajectory. For the independent seam-score objective, omit `--bonds`. A geographic salvage hint can be passed as `--hint full-board.json`; it is validated and included in the initial portfolio. The old single-path hybrid and unrestricted exact Gold dashboard remain available.

One SQLite database coordinates workers and seeds. A winning result is validated and written to the requested output. The single-path policies can reuse the same seed's existing repair rounds when the horizon increases. The paired multi-arm policies use the horizon to allocate trials and bind their checkpoints to that policy. The CLI keeps the optional slate, fork, tie and relinking switches as reproducible comparison modes; the command above uses the measured single-path choice.

## Findings by proposal

| Proposal | Comparison and decision |
|---|---|
| 11. Mandatory bonds | The constructor reuses the exact labeled component embeddings, places them without collisions, and completes the remaining legal inventory. Every repair protects the seven bonds, including when a rigid component changes pose. With `--bonds --cycle-budget 3` all four held-out 64-round runs reached 170 and ended at 177, 175, 170 and 178. The unrestricted released control preserved 2–5 of the seven on those runs. This constraint is specific to Gold. |
| 12. Trial repairs of a diverse GPU slate | The sampler supplies full validated candidates. A slate of three uses physical bond differences, gives each arm up to four early rounds and reserves every fifth round for exploration. At 512 CUDA replicas, 256 dispatches and 64 shared repair rounds the slate scored 169, 163, 174 and 172. The released winner-only control scored 167, 177, 174 and 172. CPU-only exploratory starts also trailed single-path completion. Keep one arm in the recommended profile. |
| 13. Fork the historical best and current layout | A fork has separate child randomness and shares its total repair-round budget with the original. In eight saved 320-round checkpoints extended by 32 rounds, the split improved the CPU seed 2 from 183 to 184 and improved connectivity on CPU seed 0 at the same score. It missed a one-edge gain on CUDA seed 0, and unbranched CPU seed 3 had fewer components at equal score. New-start trials also gave mixed results. An additional equal-budget control used a different saved seed as its second parent. This last control starts from another previously paid-for checkpoint, so its raw score can already be higher. |
| 14. Joint support and donor cycles | Price all two-cycle donors, pack mutually compatible positive pairs, then optionally solve the whole-board assignment relaxation. Split incompatible neighboring active cells into two branches that pin one original pose. Positive disjoint cycles within a relaxation supply a verified candidate even before the bound closes. A ten-cell CP subproblem admits arbitrary-length cycles, and `--adjacent` can price one adjoining active pair exactly. An independently enumerated neighborhood agreed with both exact models. A planted five-cycle, with no legal two-swap improvement, recovered Gold in at most two full circulation relaxations. Allowed adjacent-pair trials tied the ten-cell reference on all four screen seeds, and the two all-160-cell corrected CP trials were unknown within 0.05 deterministic solver units. Reserve the correction for explicit residuals. |
| 15. Connectivity within tied assignment layers | Each assignment respects a common set of donors, and actual exterior paths are validated. Test optimal and one- and two-seam slack layers separately under the same candidate cap. On eight saved 320-round boards an additional CP forest check optimized actual wiring on a ten-cell block of tight/near-tight arcs. Use `python r31_fresh_ties.py --board board.json --cells CELL ... --output repair.json` for a constrained repair and explicit per-layer completion flags. The zero-slack layer gave no component improvement on the measured blocks. Two boards admitted one fewer component when allowing a lost seam. Bounded tie exploration trailed single-path repair in fresh runs. |
| 16. Internal stages of permutation cycles | Align the two parents under a proper mesh rotation, explore internal transpositions in both directions, and repair a distinct intermediate under the same budget. After both independent parents had accumulated early repair rounds, final scores were 152, 157, 163 and 159. Corresponding multiple-start controls were 158, 158, 156 and 158. This policy had mixed results and is omitted from the recommended profile. |
| 18. Shared cyclic rotation frames | Reorder each cell's legal rotation labels once for all its incident factors. Reconstruction maps actual physical labels back through the same domain. On eight saved 320-round full boards the existing exact rotation oracle evaluated 57,701 table entries, the frame-based oracle evaluated 55,776, and results matched at all eight optima. Warm message hits increased from 1,791 to 1,926. Exhaustive small tests also reproduced all tied witnesses. Use `python r31_rotation_messages.py --boards board1.json board2.json` to query a shared oracle across multiple full physical placements. |

The four 320-round RTX 2060 starts with the full circulation repair ended at **179, 183, 180 and 178** with all seven bonds, and **181, 184, 180 and 181** with unrestricted bonds. The released unrestricted 320-round path reached 179, 188, 185 and 183 on those seeds. The stronger early CPU iteration results did not produce a higher long-run CUDA soft-score record on this sample.

## Held-out 64-round CPU comparison

Seeds 4 through 7 were separate from the exploratory seeds 0 through 3. A round in each method ends with physical block repair and a rotation pass. Joint methods additionally price physical exchanges, solve a donor subproblem, and optionally solve up to three full assignment relaxations. Thus round counts measure repair iterations; they carry different CPU costs.

| Profile | Rounds to 160 by seed | Rounds to 170 by seed | Final scores |
|---|---|---|---|
| Released | 35, 23, 24, 19 | >64, 55, 58, >64 | 166, 171, 171, 169 |
| Pair packing and ten-cell donor search | 16, 12, 12, 15 | 34, 29, 38, 55 | 173, 175, 173, 171 |
| Add three full assignment relaxations per round | 6, 7, 11, 12 | 24, 10, 18, 20 | 173, 177, 179, 173 |
| All seven bonds, plus the full relaxations | 8, 12, 12, 13 | 21, 24, 47, 45 | 177, 175, 170, 178 |

Across those four unrestricted runs, 160 took 36 repair iterations with circulation versus 101 in the released path. The respective complete 64-round runs took approximately 9–15 and 2–3 seconds per seed on this workstation with other tests active. The JSON records calls, CP branches, work in rotation tables, and actual tested score milestones. A circulation result marked complete bounds the one nonadjacent exchange neighborhood of that specific input rotation. Every returned repair is a legal witness.

[CPU screen](R31_FRESH_CPU.json) · [CUDA screen](R31_FRESH_CUDA.json) · [Separate slack layers](R31_FRESH_LAYERS.json) · [Repaired-parent relinking](R31_FRESH_RELINK.json) · [Held-out runs](R31_FRESH_HOLDOUT.json) · [320-round checkpoint follow-ups](R31_FRESH_FOLLOWUPS.json) · [Long circulation runs](R31_FRESH_LONG.json)

All benchmark controls use the unmodified public diagram inventory. To recreate the long-checkpoint comparisons, first generate the released four CPU and CUDA seed trajectories to round 320 with `r30_fresh.py`. Pass those SQLite paths to `python -m benchmarks.bench_r31_fresh_followups --input ... --output followups.json` and `python -m benchmarks.bench_r31_fresh --rotation-input ... --output screen.json`. Run the focused controls with `python -B -m unittest test_r31_fresh test_r30_fresh`.
