# Round 34 backend allocation (43)

The paired controls used the local i7-4790 and RTX 2060. Separate exact proof and discovery cases were assigned before scheduling. Only independent finite enumeration, full witness validation or a replayed physical-inventory proof received credit. Backend decision counters retain their own native units. Data: [R34_ALLOCATION.json](R34_ALLOCATION.json).

## Matched residuals and transfers

CPU DFS, SAT and 128-lane CUDA solved the same 16 finite residuals twice (96 attempts). Eight were proof cases: four triangular-prism restrictions and four original 160-tile incumbent restrictions. There were eight planted single-loop discovery cases on 10 and 160 cells. A five-second lease included engine construction. On the first pass CPU and SAT certified 16/16 each; CUDA certified 12/16. On the warm process pass all backends certified all 16. Some first device compilations exceeded the lease. These are process wall times, distinct from branch counts.

Replayed two policies using only the previous tasks' selected outcomes as feedback. Both the existing allocation policy and a trial stratified by objective, board size and residual width certified all 16 warm cases. The trial took 0.153 seconds on proof versus 0.197 released, and 0.394 on discovery versus 0.336 released. Combined work and certification gave no reason to replace the existing allocator. SAT was fastest on these narrow controls; the unrestricted board was unresolved.

A 0.35-second CPU original-puzzle slice exported two pending boxes (80,790 portable bytes), while a CUDA slice exported 128 boxes (2,613,680 portable bytes). Migrating from CPU to GPU took 1.86 seconds and remained open. Migrating from GPU to CPU with one box completed one negative; 127 boxes remained queued. Among twelve sampled original GPU boxes, five closed in a 0.7-second CPU slice and all twelve timed out in the single-lane GPU controls. The same five negatives have a separate exact pigeonhole proof.

## Dispatch change retained

When global matching is enabled, the CUDA domain runner screens newly split portable boxes on the host before creating native lanes. It groups cells by the exact set of available physical tiles after projecting away rotation. More cells than available tiles in one group certifies a negative. The runner records those closed boxes in its portable remainder, progress and enabled coverage ledger. Native parked lanes continue with their existing ownership.

In the actual initial 128-box split on the original puzzle, 47 independently checked boxes were impossible on inventory alone. The RTX 2060 allocated 81 native lanes instead of 128 and 84,077,028 native state bytes instead of 132,862,464. The saved native search state was 4,028,740 serialized bytes versus 6,135,719. Both bounded runs remained open. Each discarded box was independently replayed using explicit physical tile sets. Regression tests enumerate finite permutations, check resumption and verify that an entirely impossible queue completes without GPU allocation.

## Fresh-board discovery

At 24 repair rounds each, the CPU-only fresh starts scored 167/169/165/170 on seeds 340–343. CUDA-assisted fresh starts at 512 replicas and 16 dispatches scored 166/162/168/166. Each cold CUDA start made 524,288 proposals, and exact archive reopens made zero new proposals with identical scores. CPU-only reached 160 in 13/12/12/12 rounds; CUDA-assisted took 18/14/11/7. There is no stable cross-seed win warranting a change of discovery default.

Use `bench_r34_allocation.py exact|transfer|boxes --legacy-split` to reproduce the allocator comparisons with the original split. Use its `discovery` and `screen` modes for the fresh and before/after host checks. Set `--tiles` and `--output` explicitly for an isolated project run.
