# Original oriented-pose cuts in exact search

The opt-in CPU DFS and CUDA domain search can use 4,225 independently established placement conflicts from the 207 reference board, its checked 95-change lower bound, and four subsequent conflict certificates. A literal fixes one specific physical tile and rotation in a numbered cell. Each conflict forbids retaining all of its listed literals in a Gold solution. The ten orientation-preserving rotations of the triangular mesh yield 42,250 distinct transported clauses before the four additional guards.

Use `--reference-pose-cuts` with `constraint_search.py --algorithm dfs` or `--algorithm gpu-domains`. The largest benefit measured here was on a targeted frontier. A private output directory keeps an imported search separate from an existing run.

```sh
python constraint_search.py --algorithm gpu-domains --reference-pose-cuts --replicas 128 --backend cuda --work-frontier frontier.json --seconds 0 --output runtime/targeted-gold
```

Single-process DFS checks decisions and propagated domains. The CUDA domain runner checks fresh or imported portable cubes before device allocation. A live native checkpoint retains its native state and original queue. An imported exhaustion covers only the named frontier. The option requires the original authenticated Gold artwork and mesh, and supports locally narrowed domains. The CLI rejects symbolic-frontier mode, and single-process DFS also excludes shared coverage, LCG and explicit reason tracking. Independent proof probes use their original verifier. See [the unrestricted settings](ROUND35_RESULTS.md) for the measured default configuration.

## Measured scope

| Domain population | Inputs | Additional effects after existing native propagation |
| --- | ---: | --- |
| Empty board | 1 | 0 of 76,800 placements removed |
| Saved DFS and hybrid frames | 51 | 1 already dead, 50 live without an extra cut |
| One retained reference placement | 160 | 95 additional unit removals across 64 live scopes |
| Every distinct original prefix one placement short of a width-3 through width-10 clause | 4,786 | 16 already dead, 68 additional complete contradictions, 12,062 distinct unit deductions counted per scope; 4,152 live scopes changed |
| Four additional guards, full or one placement short | 29 | 4 already dead; 23 extra units across 19 scopes after comparison with the main rows |

The compiled CPU comparison used the public native search core with physical support, full-width directed seam consistency and tile ownership. All ten proper rotations give equivalent open-mesh cases. Independent RTX CUDA domain propagation left six extra units for 24 one-piece controls, 39 for eight near-unit controls and 30 for 40 explicitly transformed one-piece controls. Five sampled saved frames yielded no extra CUDA unit.

Eight end-to-end restricted scopes had identical infeasible answers with and without the option. The CPU visited eight nodes in both runs. Its seam revisions fell from 2,038 to 242 and Hall checks fell from six to zero. One individual input incurred one extra seam revision. CUDA avoided device initialization on five scopes. The eight-cell guard required four CUDA search decisions and 3,898,640 support checks in the baseline and zero in the opt-in run. A separate test interrupted the unmodified native GPU engine mid-kernel sequence and resumed its checkpoint with the switch enabled. The CPU-only imported-frontier control exhausted its scope before GPU initialization.

Four fresh native Gold searches used 4,096 transitions each and checked partial closed loops. One seed encountered 42 live subtrees whose domains the clauses could strengthen. Every one of those subtrees exhausted within the matched 512-transition budget. Together they took 427 nodes and 385 branches under each policy, with 258,428 baseline versus 258,905 opt-in arc checks. These subtrees include related ancestors and descendants. Their counts describe paired work and cannot be added to global coverage. The unrestricted setting therefore remains opt-in.

## Reproduction

The raw counts, seeds, paths, checkpoint hashes and eight paired public results are in [the frontier receipt](R35_POSE_FRONTIERS.json). The private historical stacks are optional benchmark inputs. `--native` points to a local verified build of the compiled CPU search core. After preparing the puzzle data, invoke the scripts with the appropriate tiles, native and output paths.

```sh
python bench_r35_pose_frontiers.py --tiles data/tiles.json --native .native --checkpoints runtime/searches --output pose-frontier-cpu.json
python bench_r35_pose_cuda.py --tiles data/tiles.json --cpu pose-frontier-cpu.json --checkpoints runtime/searches --symmetry --output pose-frontier-cuda.json
python bench_r35_pose_paths.py --tiles data/tiles.json --native .native --output pose-frontier-paths.json
python bench_r35_pose_transactions.py --tiles data/tiles.json --output matched-cpu.json
python bench_r35_pose_transactions.py --tiles data/tiles.json --gpu --output matched-cuda.json
python -m unittest test_r35_pose_frontiers
```

Each invocation authenticates the source artwork, masks, adjacency and proof catalogue before applying any archived deductions. Within a CPU invocation, the last set of matching fixed poses can reuse its proved implications. The cached implications are derived with every other cell open and are intersected anew with each sibling's actual domains. The two CPU profiles exercised altered artwork, adjacency and masks, all ten rotated binary clauses, widened siblings, and stopping with unconsumed cubes. The native CUDA profile also exercised an actual retained device checkpoint. Seven historical classic GPU profiles preserved their verified terminal states or live assisted frontiers.
