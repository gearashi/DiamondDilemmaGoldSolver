# Original oriented-pose cuts in exact search

The opt-in CPU DFS and CUDA domain search can use 4,225 independently established placement conflicts from the 207 reference board, its checked 95-change lower bound, and four subsequent conflict certificates. A literal fixes one specific physical tile and rotation in a numbered cell. Each conflict forbids retaining all of its listed literals in a Gold solution. The ten orientation-preserving rotations of the triangular mesh yield 42,250 distinct transported clauses before the four additional guards.

Use `--reference-pose-cuts` with `constraint_search.py --algorithm dfs` or `--algorithm gpu-domains`. A scoped imported frontier can also select `--algorithm parallel-dfs` or `--algorithm sat --search-options '{"persistent_sat":true}'`. The largest benefit measured here was on a targeted frontier. A private output directory keeps an imported search separate from an existing run.

```sh
python constraint_search.py --algorithm gpu-domains --reference-pose-cuts --replicas 128 --backend cuda --work-frontier frontier.json --seconds 0 --output runtime/targeted-gold
```

Single-process DFS checks decisions and propagated domains. The CUDA domain runner checks fresh or imported portable cubes before device allocation. The parallel DFS and persistent SAT launchers check only newly imported portable cubes. Existing parallel jobs and live native checkpoints retain their owned regions and journal. An imported exhaustion covers only the named frontier. The option requires the original authenticated Gold artwork and mesh, and supports locally narrowed domains. The CLI rejects symbolic-frontier mode, and single-process DFS also excludes shared coverage, LCG and explicit reason tracking. Independent proof probes use their original verifier. See [the unrestricted settings](ROUND35_RESULTS.md) for the measured default configuration.

## Measured scope

| Domain population | Inputs | Additional effects after existing native propagation |
| --- | ---: | --- |
| Empty board | 1 | 0 of 76,800 placements removed |
| Saved DFS and hybrid frames | 51 | 1 already dead, 50 live without an extra cut |
| One retained reference placement | 160 | 95 additional unit removals across 64 live scopes |
| Every distinct original prefix one placement short of a width-3 through width-10 clause | 4,786 | 16 already dead, 68 additional complete contradictions, 12,062 distinct unit deductions counted per scope. A total of 4,152 live scopes changed |
| Four additional guards, full or one placement short | 29 | 4 already dead. There were 23 extra units across 19 scopes after comparison with the main rows |

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

## Native SAT and CP-SAT watches

The app's canonical symmetry leaves 3,579 binary clauses and 41,543 total applicable clauses including the four later guards. Adding all of them to the public 745,872-clause Glucose model saved 171 decisions on one seven-held-pose proof before comparing existing host deductions. All 25 one-deletion guard cases were then tested with those host deductions already applied. The baseline settled 11 under the finite budgets, eager watches settled 10, and compact rules touching a held cell settled 11 with one gain and one loss. The ten matched baseline/eager proofs used 1,854/2,322 decisions. Restricting the clauses to activated images or to pairs failed to improve that comparison.

Binary-only watches saved 220 decisions across eleven matching original proofs. On nine held-out rotations of two such candidates and one adverse control the baseline proved 23 of 27 cases. Binary-only watches proved 18. Both exhausted eighteen cases in common. The fixed-budget CP-SAT comparison made zero branches across all six scopes. Three holdings were already refuted in presolve. The remaining open and partial cases stayed undecided. Eager native watches therefore remain disabled. The finite budgets and time interruptions are recorded per call. Every reported proof is an actual negative from an independently initialized solver. [Native model comparison](R35_POSE_LITERAL_SCREEN.json)

## Imported parallel and persistent SAT sessions

On the eight-held-pose case, the public persistent SAT command built one model and made one native call, using 20 decisions. The portable check exhausted the same imported frontier before model construction. The unfiltered two-worker parallel DFS run had reached 179 nodes and completed 167 regions in 40 seconds. Its four remaining jobs reopened unchanged. The initial portable check completed this case with zero worker processes. All terminal markers retained the original imported-frontier digest.

A clean checkout of v1.15.0 produced an authentic two-region SAT journal and reproduced its cold-reopen error, `Malformed native-session counters`. The old physical encoder saved counts of 11,520 forbidden clauses and 7,680 required clauses, which its own checkpoint reader failed to recognize. The current reader recognizes the three counter names the physical encoder can emit and validates each as a nonnegative integer. These values are diagnostics. The existing digest, puzzle, target, remaining frontier, and cut checks still control resumption. The old 20 decisions, completed first region and exact second region survived a cold reopen with the pose option enabled and disabled. A second v1.15.0 journal also resumed with both switch settings and independently exhausted its final imported region. [Public launcher and journal receipts](R35_POSE_PORTABLE_RESUME.json)
