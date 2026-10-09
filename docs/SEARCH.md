# Search, checkpoints, and validation

## Board and orientation model

The board is a closed pentagonal bipyramid: ten triangular faces, each divided into sixteen small triangles. It has 160 tile cells and 240 shared edges.

Each physical tile is labelled. An orientation code is `3 * tile_index + rotation`, with rotation in `0, 1, 2`. Reflections are not allowed. Endpoints use eleven positions on an oriented edge; adjacent sides match with the endpoint order reversed.

The raw labelled space is `160! × 3^160`, approximately `1.03 × 10^361` arrangements. Identical visible patterns and rotations of the entire board are not merged. The scale makes exhaustive completion uncertain; neither a timeout nor an unchanged best depth establishes impossibility.

## Gold-only constraint search

The dashboard exposes `dfs`, `cp`, `cp-sat`, `sat`, and `hybrid`, implemented by `constraint_search.py`. All five require every tile once, all matching seams, and one complete gold loop. An edge-perfect arrangement with multiple loops is rejected and search continues.

Strong DFS uses bitset domains, minimum-remaining-values cell ordering, seam arc consistency, forced tile uniqueness, and early rejection of closed proper loops. A rejected partial assignment discards all its impossible completions. It stores each pending branch and its next choice, so normal resume does not re-enumerate consumed complete assignments. Periodic checkpoints are written between 30-second search slices; Stop also saves a checkpoint.

SAT uses Glucose through PySAT. CP-SAT and traditional CP use OR-Tools with tile uniqueness and seam constraints. Independently detected closed components generate exclusions of the conjunction of placements preserving that component. This is a sound exclusion of impossible Gold completions, not a heuristic score threshold. Native engines save these validated exclusions and hints; their internal search, propagation, and learned clauses restart after a process restart. A checkpoint timestamp for these engines does not mean an exact native search cursor was saved.

The experimental hybrid uses native CUDA or WebGPU sampling for an initial fraction of its first search slice, independently checks the returned samples, and uses the best sample to order CPU DFS values. It never fixes sampled choices or discards legal branches on score alone. Samples may repeat. Once DFS begins, resumes use its saved stack without sampling again.

Each mode saves under `runtime/searches/<algorithm>-gold/`. A completed exhaustion record is bound to the input, algorithm, and source hashes, and is reused without restarting completed work. It is recorded solver evidence, not an independently checkable UNSAT proof. A saved solution is always independently validated again. These checkpoints are separate from the existing GPU pool and from each other. Search counts cannot be added across algorithms as unique coverage. CPU modes use one worker; GPU replica counts affect GPU work only. See [the algorithm comparison](ALGORITHM_COMPARISON.md).

## Systematic GPU search

The legacy `systematic_search.py` backend fixes a cell order and partitions the search into prefix jobs. Refining a job replaces it with **all** unused-tile orientations compatible with the already assigned seams. A cancellation or resource cap preserves an unsplit parent rather than dropping its remaining children.

The resulting frontier is prefix-free: no job duplicates another job or contains another job as a descendant. Each active GPU lane owns one job. Surplus lanes stay idle. An available lane can continue a paused job from its saved cursor or receive a previously unassigned job.

Within a branch, DFS stores its candidate-list cursor at every entered depth. Candidate ordering can vary, but a child is tried once under that partial assignment. Edge contradictions reject all full arrangements extending that partial board. Backtracking revisits ancestors; this does not repeat a complete arrangement.

The CPU checks a complete candidate independently. The configured stopping condition is the first full edge match, not continued enumeration of all edge matches. Multiple closed gold loops satisfy the edge check but do **not** solve the Gold Challenge.

## Progress is not a completion percentage

| Display | Meaning |
| --- | --- |
| Tiles placed | Occupied cells in the live partial branch |
| Deepest placement | Greatest branch depth reached in this run |
| Active branches | Assigned jobs on the current GPU lanes |
| Paused branches | Unfinished jobs retained off the GPU, with their search positions saved |
| Waiting branches | Paused jobs plus jobs that have never been assigned |
| Completed branches | Jobs whose entire subtree was exhausted |
| GPU nodes | Candidate attempts, not globally unique full boards |
| Historical best | A previously saved complete placement |

Jobs have unequal subtree sizes. Completed-job counts, depth, and node throughput cannot be converted into a reliable percentage of the whole search or time remaining.

Only exhaustion of the complete, correctly bound frontier can establish that no edge-compatible full arrangement exists for the encoded input. Finite tests and a bounded search run do not establish that result.

## Durability and stopping

The systematic checkpoint stores active and paused DFS stacks, candidate cursors, fixed prefixes, job IDs, and the allocation cursor in **one atomic NPZ file**. Restore validates the DFS state and binds every owned branch back to its immutable frontier job.

To change the replica count, Stop and wait for the final checkpoint, select the new count (128–131,072), then Start. Reducing the count moves excess unfinished branches into a host-side paused bank without restarting or completing them. Increasing the count makes more GPU lanes available to continue saved branches. This reduces active GPU work when fewer replicas are selected; paused state still occupies host memory and checkpoint storage.

A coherent host snapshot is captured between bounded GPU launches. One background writer saves it while searching continues. Checkpoints are uncompressed to reduce compression latency; the default periodic interval is 30 seconds after the preceding save finishes.

Stop waits for current work and the required final save. An older write may need to finish before the newest state is written. Shutdown speed depends on the state size and storage load.

Normal Stop/Resume retains cursors and ownership. A crash can replay work performed after the last completed checkpoint; the program does not claim exactly-once execution across a crash. Never discard, replace, or roll back a checkpoint if preserving its coverage is required.

For legacy GPU DFS, `--seconds 0` means Unlimited. It does not disable checkpointing or automatic stopping at the first 240-edge candidate. The new constraint modes instead stop automatically only for a verified single-loop Gold solution or exhaustion.

## GPU compute backends

The systematic runner accepts `--backend auto`, `cuda`, or `webgpu`. CUDA targets NVIDIA GPUs on Windows/Linux. WebGPU runs a native compute kernel through Metal on macOS or Vulkan/DX12 on compatible PCs; the browser only displays and controls the search.

CUDA and WebGPU share the systematic checkpoint schema, including paused branches. Bounded interchange tests on an RTX 2060 passed transfer in both directions and continued search after restore. The puzzle data, geometry, cell order, and immutable frontier must agree; retain the entire output directory and its input snapshots when changing `--backend`. The stochastic and standalone drivers remain separate formats.

Both Intel and Apple Silicon Macs use the same Python source with native dependencies for their CPU architecture. Driver support and adapter memory/compute limits still determine whether a particular machine can run the requested population. Package imports and CPU fixtures are separate from testing a kernel on actual GPU hardware.

## Separate backends

Use separate output directories to avoid mixing status files or certificates.

```powershell
# Default systematic method.
.\venv\Scripts\python.exe systematic_search.py --resume --seconds 0 --replicas 4096 --output runtime

# Optional stochastic full-board search.
.\venv\Scripts\python.exe solve.py --resume --seconds 3600 --replicas 4096 --output stochastic-runtime

# Standalone DFS driver for experimentation.
.\venv\Scripts\python.exe dfs_gpu.py --seconds 120 --replicas 3840 --output dfs-runtime
```

On Linux/macOS, replace `.\venv\Scripts\python.exe` in these commands with `venv/bin/python`. Forward slashes work for data and output paths on all platforms.

The dashboard and launcher select the requested algorithm within the main runtime; command-line launchers keep the legacy GPU default for compatibility. The other drivers have their own checkpoint behavior; inspect their `--help` before use. To request their stop, write a `stop.request` file in that driver's output directory. The standalone DFS driver may require removal of that request before restarting.

The stochastic backend's recent-board cache is bounded and evicts old entries. Different replicas have separate caches, and old placements can recur. Its persistent validation cache avoids repeating CPU report calculations; it is not a ledger of exhaustive search coverage.

## Validation and evidence

A Gold solution requires all of the following:

1. Every physical tile appears exactly once, with an allowed rotation.
2. All 240 shared edges have matching gold endpoints.
3. Every gold segment belongs to exactly one closed connected loop.

Segments may join two points on the same tile side. Geometric crossings of drawn strokes are not graph junctions.

Native Vulkan checks on an RTX 2060 enumerated the expected 54 synthetic arrangements while exercising resizing, frozen pending candidates, and counters. A comparison using actual puzzle masks and a varied cell order matched every CUDA device-state field after each of 20 launches, each with a 32-node budget per lane across three lanes. CUDA-to-WebGPU and WebGPU-to-CUDA checkpoint restoration and continued execution passed. An isolated 128-replica systematic run also saved, restored, and advanced its saved search state. These finite results establish the tested compatibility cases, not correctness on every GPU or every possible search state.

CPU tests exercise topology, small exhaustive prefix covers, checkpoint ownership, terminal validation, and control behavior. Synthetic host arrays test runner scheduling without CUDA. Separate GPU tests check actual kernels. Bounded WebGPU tests run on an NVIDIA Vulkan adapter; CI also executes finite shader checks through Mesa lavapipe with an explicit test-only software-adapter opt-in. Neither establishes AMD or macOS Metal hardware correctness. Source-image extraction checks require locally prepared diagrams and data. None of these test categories substitutes for independently validating a complete candidate.
