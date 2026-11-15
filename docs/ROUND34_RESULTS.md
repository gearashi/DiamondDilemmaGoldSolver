# Search and checkpoint improvements in v1.15.0

The best validated original-puzzle board in this round has **207 of 240 matched seams**. Gold requires all 160 distinct pieces, 240 matching seams and a single closed loop. The experiments below reduce the search or the work required to resume it. Their measurements apply to the indicated boards and residuals.

| Search | What changed and what was measured |
| --- | --- |
| Reference-bond loss | On the complete cap-14 four-error layer, secondary suffix lookup reduced tail visits from 487,928 to 52,859 with the same 58 joint checks. Precompiled phase equations reduced 800,625 pivot XORs to 19,239 across seven matched queries. Four interrupted cap-17 queries saved another 349 suffix visits and 33 joint checks on restart. [Loss results](ROUND34_LOSS.md) |
| Exact routes to 208 | The solver can rank all seven five-swap cycle families and reuse completed four-swap predecessors. Two workers covered 1,024 disjoint seven-piece supports and all 645,120 endpoints in the same range as one worker. An optional component budget avoided 8,789 cycle extensions in 24 bounded seven- and eight-piece searches. [Route and repair results](ROUND34_REPAIR.md) |
| Different starting boards | The center selector can favor endpoints outside other centers’ neighborhoods. Three policies inspecting 75,600 endpoints saw 64,678, 66,718 and 69,477 distinct proper-rotation classes. On four fresh seeds, length-three cycle corrections reached 160 matches in 9–12 rounds. [Fresh-board comparisons](ROUND34_REPAIR.md) |
| Single-loop Gold | SAT and CP engines can use independently checked physical side mates. On a finite one-loop fixture, SAT decisions fell from 61 to 7 and CP-SAT branches from 125 to zero. In the ten-tile phase-cube control, 16 geographic solves recovered the same validated Gold assignment as 512 numbered-phase solves. [Physical constraints and controls](ROUND34_PHYSICAL.md) |
| GPU search and restart | Lowering a live native domain search from four lanes to one avoided 81,748 and 38,294 repeated support checks at two original-puzzle checkpoints. Cold and resumed runs reached identical fixed points. On a fresh 128-root partition, a host-side piece-inventory check independently excluded 47 roots and saved 48,785,436 bytes of native state. [Resize experiments](ROUND34_GPU_RESIZE.md) � [Backend allocation](ROUND34_ALLOCATION.md) |
| Repeated searches | A phase-first seed extended from 16 to 64 rounds did 48 new rounds and reached the cold run’s board, phases, RNG state and work counters. On an eight-cell residual of the 207 board, a checked 208 infeasibility result answered a tighter 209 query with zero SAT decisions. The independent cold run took 310. [Continuation controls](ROUND34_CONTINUATION.md) |

## Settings for the i7-4790 and RTX 2060 6 GB

**To search for Gold anywhere on the board**, resume **CUDA domains + learned backjumping (experimental)** in the dashboard. Select **128 lanes**, **scalar propagation** and **Unlimited**. Enable global GPU piece matching, GPU seam-support reuse, watched GPU learned clauses, verified conflict sharing and transport of checked deductions. Keep cooperative GPU propagation off for this profile. Use the full Python 3.12 environment for CUDA. The solver can now park a native search when the lane limit decreases and restore it in batches. At a fresh root the displayed active count may be smaller than 128 because some branches are already proved impossible. The unrestricted Gold search remains open. [GPU profile](ROUND32_RESULTS.md)

**To look for 208 near the three known boards**, continue the seven-piece family on CPU with 1,024-support checkpoint pages, using the existing ranked database. The eight-piece family uses `--family 2,2,2,2` in the same file. Two processes may share disjoint work when spare CPU capacity permits. The small local filter tests do not establish an elapsed-time advantage for running both campaigns together. [Ranked page controls](ROUND34_LOSS.md)

```powershell
.\venv\Scripts\python.exe r30_multicenter.py --family 2,2,3 --backend cpu --page 1024 --target 208 --seconds 0 --database runtime/four-swap-union.sqlite
```

**To explore unrelated boards**, use the eight-bond CPU constructor with a cycle budget of three and 64 rounds per seed. The optional `--pair-cycles 3` correction is a separate policy and has its own checkpoint scope. Four held-out seeds support trying it for earlier discovery. [Trajectory results](ROUND34_REPAIR.md)

```powershell
.\venv\Scripts\python.exe r31_fresh.py --backend cpu --bonds --cycle-budget 3 --rounds 64 --starts 0 --database runtime/fresh-eight.sqlite --output runtime/fresh-eight-best.json
```

**To continue the reference-bond lower-bound experiment**, use full Python 3.12, CPU, integer refinement, a page size of 8,192 and an unlimited timer. This asks whether Gold can keep that many of one reference board’s bonds. Its checkpoints are separate from unrestricted Gold. [Reference-bond experiment](ROUND34_LOSS.md)

```powershell
.\venv\Scripts\python.exe r32_gold.py loss --refine-integer --upper 17 --page 8192 --seconds 0 --database runtime/r32-loss.sqlite
```

The [candidate register](round34-candidates.json) records the individual experiments, optional mechanisms and rejected variants. The [checkpoint experiments](ROUND34_CHECKPOINTS.md) include CPU and GPU replay measurements. The optional software profiles remain Python 3.12 for all engines and Python 3.15 for CPU DFS and WebGPU.
