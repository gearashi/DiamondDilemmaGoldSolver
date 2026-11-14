# Independent-start checkpoints and geographic defect budgets

## 17. Independent starts and sampler continuation

The fresh runner can select unrestricted starting arrangements using a scrambled mixed-radix rank. It uses each physical permutation and orientation tuple once over the available rank space. A restricted domain uses a seeded legal assignment. Repair can bring initially different boards to the same state. A check on the published puzzle counted 512 different starting arrangements at indices 0 through 511.

Provide `--database` to the fresh CLI for exclusive seed claims with a renewable expiry, conditional publication, and repair checkpoints. Workers with the same profile can run different seeds while sharing one database. An expired worker cannot overwrite its successor. A completed Gold result prevents further dispatch in its scope. A running worker checks at each saved repair batch. A process killed during a batch can replay that unfinished batch. Use an output path per experiment to export the best witness while searching. Finite regression checks run two simultaneous workers through two distinct seeds, reclaim an expired claim, interrupt a repair after saving its progress and resume it.

On a GPU, supply `--sampler-cache runtime/sampler.sqlite` to reuse prefixes between different requested dispatch counts. This stores a resumable native state before the terminal archive collection. The engine checks source configuration, puzzle, device, and native runtime when restoring. On the RTX 2060 the 512-replica, 64-dispatch prefix contained 2,097,152 proposals. Requesting 256 dispatches required 6,291,456 new proposals, versus 8,388,608 from a cold start. The final native array hash and eight archive boards agreed with the cold control. Repeating the 256-dispatch request generated zero proposals. Seed-claim recovery and distinct starting ranks apply to the CPU profile as well.

## 19. Exact geographic mismatch distributions

The experimental SAT search partitions mesh seams into geographic bands. It bounds the number of failed seams in each band and assigns each complete board to one set of disjoint count intervals. Exact Boolean mismatch indicators and two-way cardinality definitions support upper and lower counts. An exhausted search records its negative result. A timed-out branch stays open. A difficult interval can split again. An UNSAT assumption core is reused only when its geographic count limits hold. A one-piece-per-cell assignment bounds the minimum band defects from the incident seam possibilities on narrower branches. The original pilot obtained no additional exclusions from that inventory refinement.

The matched control freed the first 24 physical placements and their rotations, holding the other 136 oriented pieces fixed. At 208, the released threshold solver took 2,864 SAT decisions, the single-budget control took 2,619, and the three-band search took 2,221. All three closed that restriction. At 207, the same band search needed 4,592 decisions to find a board, against 575 in the single-budget control. A bounded pilot permitting all 160 pieces to move left the 208 question open. Decision counts to a timeout do not establish a gain in coverage. Completed band checkpoints reopened with zero new calls. This mode remains an explicit experiment for soft scores.

On a full Python 3.12 installation, run the following from the solver folder. Zero seconds permits unlimited search.

```powershell
.\venv\Scripts\python.exe r31_defect_bands.py --database runtime/defect-bands.sqlite --bands 3 --width 8 --target 208 --seconds 0
.\venv\Scripts\python.exe -B -m unittest -v test_r31_distribution
.\venv\Scripts\python.exe -B bench_r31_distribution.py --tiles data/tiles.json --cuda --output runtime/distribution-check.json
```

[Original results and input fingerprint](R31_DISTRIBUTION_RESULTS.json)
