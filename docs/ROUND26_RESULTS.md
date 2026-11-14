# Gold solver benchmark results for v1.7.0

This release evaluates 17 workstreams covering 81 specified changes. The original 160-piece Gold Challenge remains open. A solution must match all 240 edges and join all gold segments in one closed loop.

The [workstream register](round26-workstreams.json) records a disposition for every scope. The [finite measurements](round26-results.json) give input and report identities, individual results, added costs and reproduction commands. The [hardware measurements](round26-settings.json) list the settings and outcomes for each board and seed.

## What reduced the search work

| Workstream | Result on the measured fixture |
| --- | --- |
| F01. Work ownership | A decision diagram stores unfinished, leased and completed sets. Exact transfer preserves the remaining set when a worker changes. A synthetic shared checkpoint occupied under one twentieth of its flat representation. |
| F02. Resumable CPU reasoning | Across 12 interrupted contexts, saving the Python propagation cursor reduced cumulative arc checks from 308 to 188. Native calls can still repeat if interrupted. |
| F03. Shared SAT and witnesses | Two workers submitted 16 copies of an impossible placement query. Sharing reduced native model builds and native queries from 16 to one each. |
| F04. Proof checking | Checking the same three proof steps used 1,212 clause visits with a scan, seven with the index and six with hints. Index construction read 810 literals. Direct theory callbacks showed no decision advantage in their separate finite comparison. |
| F05. SAT model reuse | On the same 54 native calls and 36 decisions, reusing guards reduced added guard clauses from 108 to 12. Auxiliary simplification helped two fixtures and increased decisions on a third. An isolated producer completed the largest pigeonhole proof in 1,084 native decisions. The baseline reached its 500,000-clause-visit cap. |
| F06. Higher-order relations | A semijoin removed two orientations while preserving all 15 edge completions. Compiling the inputs required 121 SAT queries. A separate restart imported a completed projection with zero native enumeration queries. |
| F07. Arithmetic reasoning | Reusing contribution trees reduced input vectors and integer operations on related domains. It did more basis builds. The additional pseudo-Boolean discovery rules produced zero extra deletions across 160 screened contexts and stay inactive. |
| F08. CPU strategy selection | Across 24 completed residuals, four candidate CPU profiles were compared. Selection using examples from other puzzle families reduced the total from 129 to 122 decisions. This tests the strategy selector used by adaptive allocation. |
| F09. Regional compilation | Changing the compilation order reduced transitions from 23,611 to 15,554. A rank-based test for existence reduced them further to 12,691 and cost 7,993 XOR operations. |
| F10. Regional decomposition | Recursive messages reduced repeated pair-geometry checks from 400 to one. They used 965 decision-diagram operations. All 720 completions were preserved across 18 restores. |
| F11. Regional proof reuse | Reusing independently verified regional proofs produced the same four exclusions with two checked states instead of eight. |
| F12. Gold connectivity | Tracking another obligatory strand rejected a case that passed the previous relaxation. The stronger check used eight route steps versus four. |
| F13. Regional integer bounds | A cross-region inventory shortage closed at zero integer branches versus eight. The stronger method paid for five native linear solves, 18 pricing queries and exact verification. |
| F14. Exact CUDA propagation | Reusing seam witnesses reduced support-word scans from 1,016 to 503, 1,477 to 298 and 781 to 345 on three fixtures. Global matching closed one shortage at zero decisions versus two. The cooperative layout is assessed separately below. |
| F15. Incremental GPU scoring | On a controlled 160-cell board, the same 384 proposals followed the same trajectory on CUDA and WebGPU. The ejection-chain case reduced domain-term evaluations from 123,520 to 3,558. A small tetrahedron with combined moves did more work with the cache. |
| F16. Recombination | Searching interacting cycles found a Gold board in a 16-board synthetic neighborhood where single-cycle changes missed it. Conditional cuts reduced branch nodes from seven to five and independent scoring calls from four to two on another case. |
| F17. Retained repair | Gold patch repair reduced native calls from 49 to 36 and branches from 4,912 to 3,486. Replaying the same 49 contexts needed zero new calls. Ordinary soft-edge repair reused models, but increased native branches from 284 to 1,806 on a 160-cell case with four free cells. |

Each comparison uses the same counter on both sides. A GPU proposal, a CPU DFS node, a SAT decision and a closed regional state describe different events. Finite fixtures include independently checked solutions, exhaustive small boards, and explicitly restricted 160-cell residuals. Whole-board timeouts provide observations of progress over the stated interval. They cannot establish the time remaining to solve Gold.

## Cooperative CUDA

On the RTX 2060, each paired run started in the same state and executed 32 synchronized dispatches of 16 primitives with 128 lanes. All ten pairs finished with identical logical state and decisions. On the original input, cooperative execution read 4,279,363 support words versus 1,957,421 for scalar execution. It won three of five timing pairs, with a median cooperative/scalar time ratio of 0.956. On the word-boundary fixture it read 1,680 versus 1,086 words and won one of five pairs, with a median ratio of 1.555. These trials cover initial propagation and include the status copy.

## Validation

The finite comparisons ran on commit `685f6212871e3a8dcd2ef830da365c08c222f249`. Local CPU validation on Python 3.12 ran 2,164 tests with 196 skips on `813c470`. Python 3.15 ran 2,168 with 424 skips on `b9c1bff`. Both passed their separate CPU scorer checks. Later targeted checks covered the legacy counter migration on both interpreters and native WebGPU device release on the RTX 2060. The full hardware run passed 372 tests with one optional separate-interpreter comparison skipped, plus six standalone GPU scripts. CUDA and native WebGPU ran on the RTX 2060. All 26 jobs passed on the [hardware-tested solver revision](https://github.com/gearashi/DiamondDilemmaGoldSolver/actions/runs/43556829495). The [published release](https://github.com/gearashi/DiamondDilemmaGoldSolver/releases/tag/v1.7.0) identifies the final main-branch checks and platform packages.

Checking four resume routes against the installed saved searches preserved the 21,184-node DFS frontier and the 1,348-node hybrid frontier. All six protected data and checkpoint files kept their original SHA-256 values. The Windows setup and dashboard launch were also exercised in the project directory.

## Settings on the i7-4790 and RTX 2060

The test machine has four physical CPU cores, eight hardware threads, 16 GiB RAM and 6 GiB VRAM. Use the 64-bit Python 3.12 environment for CUDA, SAT and repair. Choose **Unlimited** for an unattended search. The solver stops on a verified single-loop Gold solution and supports manual Stop and resume.

**Keep existing exact progress.** Resume the existing DFS checkpoint with **CPU propagation core → Automatic**. For a dedicated CPU run, select **Parallel CPU DFS → 8 workers**. Four workers are a good choice when sharing the desktop. In the two 45-second original-board trials the baseline evaluated 87 and 89 nodes with eight workers, versus 53 and 51 with four. All requested workers started, and there were zero replayed jobs. The eight-worker runs also committed slightly more of the same input frontier. Across six completed finite fixtures the baseline took 129 CPU nodes, the reuse profile 135 and the shared SAT profile 200. The broad deduction bundle solved two of those six within its 30-second budgets. Exact node counts vary by fixture, seed and worker scheduling.

**Exact GPU search with fewer decisions.** Select **CUDA domains + learned backjumping (experimental)**, **128 lanes**, and the scalar layout. On six solved finite fixtures the prior combined profile used 10,514 decisions, reuse used 9,848, global matching with reuse and proof transport used **6,084**, and that same profile with cooperative execution used 6,264. The 6,084-decision profile saved about 42% relative to the prior combined profile, with solver elapsed time rising from 18.7 to 22.1 seconds in total.

To reproduce the matching profile, enable the controls in both groups below.

- **Combine GPU edge matches with neighboring piece availability**, **Prune a shared GPU parent before dividing its branches**, **Recheck only affected GPU cells**, **Track precise causes of GPU contradictions**, **Share independently verified GPU conflicts**, **Refill finished GPU lanes individually**, **Reuse unchanged GPU checkpoint data**, and **Share completed exact regions across policies**.
- **Reuse GPU seam supports**, **Watch GPU learned clauses**, **Check global GPU piece matching**, and **Transport checked GPU deductions**. Keep **Use cooperative GPU propagation** off for the scalar profile.

On the unrestricted original board, matching and cooperative each closed 47 lanes in each of two time-limited runs. Scalar matching made 80 decisions in each run. Cooperative made 291 and 276. Reached search depth, committed frontier, lane count and branch work measure different things. These short comparisons establish neither the percentage of valid boards exhausted nor the time to solve Gold.

**Cooperative execution has a narrower use.** With global matching enabled, the two completed 160-cell residuals with 24 free cells took 5.390 and 7.656 seconds of solver time on scalar CUDA and 4.391 and 5.750 seconds on cooperative CUDA. Decisions were 1,851 in both layouts on the first, and 2,131 versus 2,266 on the second. The switch remains available for that measured time savings.

**SAT for a heavily constrained remainder.** Choose **SAT (Glucose)** and enable **Preserve SAT sessions between time slices**, **Reuse SAT assumption guards** and **Retain SAT theory with justified literal propagation**. This profile solved all six finite inputs. On the two 24-free-cell residuals it took 1,449 and 1,004 decisions. Baseline persistent SAT timed out on the first and took 4,584 decisions on the second. The theory profile uses CaDiCaL, while the baseline uses Glucose. The paired direct-theory tests in the same backend showed no decision gain attributable to the callbacks alone.

**Hybrid search for high-scoring boards.** Choose **Experimental GPU sampling + repair + DFS**, **NVIDIA CUDA**, **4,096 GPU replicas** and **Unlimited**. Enable **Resume GPU sampling and completed repair work** and **Update sample scores after local changes**. To match the measured profile, also enable **Score sample boards by loop connectivity (hybrid)**, **Move chains of displaced pieces**, **Avoid recent reverse moves**, **Exchange boards between fixed temperatures**, **Target conflicts with ejection chains**, **Combine sampled boards through piece cycles**, **Avoid unnecessary Gold loop tracing**, and **Adapt replica-exchange temperatures**. The experimental optimizing crossover and retained repair were off for this profile.

With 4,096 replicas and 60 seconds per run, this profile tried 37,748,736 and 38,535,168 moves. After repair, the best boards matched 163 and 160 edges, versus 147 and 145 with the prior combined profile and 155 in both trials with the new combined profile. All six reported boards had zero domain violations. Each still had unmatched edges. The trials compared complete configurations on two seeds. A saved hybrid search already in exact DFS resumes that phase. Sampling is heuristic and can revisit boards.

## Reproduction and coverage

The six [settings suites](round26-settings.json) ran 94 cases serially at `2b031e0b17e6dd9361b51438d996fa8e38077ff1`. This revision changes the settings harness and its test so each CPU or hybrid configuration gets the same private reasoning-cache directory it would receive from the dashboard. The solver and GPU code match the frozen finite revision. Of the 94 runs, 64 returned independently validated finite Gold boards. Each of the 22 unrestricted original-board runs ended at its stated limit. The original CPU and exact GPU comparisons began from an unsymmetrized portable root. On an existing installation, resume the saved search to keep its progress.

The [finite index](round26-results.json) and [settings results](round26-settings.json) give the benchmark commands. Run them with the full Python 3.12 environment, prepared tile data, and a fresh output destination. CPU, GPU and SAT thresholds differ by phase and use separate counters. The settings tests used hardware CUDA on this Windows workstation; the sample scorer also passed its finite comparisons on native WebGPU. Linux, macOS and Windows CI exercise supported CPU and platform paths. Physical AMD or Apple GPUs were outside this hardware comparison.
