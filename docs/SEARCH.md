# Search, checkpoints, and validation

## Board and orientation model

The board is a closed pentagonal bipyramid: ten triangular faces, each divided into sixteen small triangles. It has 160 tile cells and 240 shared edges.

Each physical tile is labelled. An orientation code is `3 * tile_index + rotation`, with rotation in `0, 1, 2`. Reflections are not allowed. Endpoints use eleven positions on an oriented edge; adjacent sides match with the endpoint order reversed.

The raw labelled space is `160! × 3^160`, approximately `1.03 × 10^361` arrangements. Identical visible patterns and rotations of the entire board are not merged. The scale makes exhaustive completion uncertain; neither a timeout nor an unchanged best depth establishes impossibility.

## Systematic search

The default `systematic_search.py` backend fixes a cell order and partitions the search into prefix jobs. Refining a job replaces it with **all** unused-tile orientations compatible with the already assigned seams. A cancellation or resource cap preserves an unsplit parent rather than dropping its remaining children.

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

`--seconds 0` means Unlimited. It does not disable checkpointing or automatic stopping at the first 240-edge candidate.

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

The dashboard and launcher control the default systematic runtime. The other drivers have their own checkpoint behavior; inspect their `--help` before use. To request their stop, write a `stop.request` file in that driver's output directory. The standalone DFS driver may require removal of that request before restarting.

The stochastic backend's recent-board cache is bounded and evicts old entries. Different replicas have separate caches, and old placements can recur. Its persistent validation cache avoids repeating CPU report calculations; it is not a ledger of exhaustive search coverage.

## Validation and evidence

A Gold solution requires all of the following:

1. Every physical tile appears exactly once, with an allowed rotation.
2. All 240 shared edges have matching gold endpoints.
3. Every gold segment belongs to exactly one closed connected loop.

Segments may join two points on the same tile side. Geometric crossings of drawn strokes are not graph junctions.

CPU tests exercise topology, small exhaustive prefix covers, checkpoint ownership, terminal validation, and control behavior. Synthetic host arrays test runner scheduling without CUDA. Separate GPU tests check actual kernels. Source-image extraction checks require locally prepared diagrams and data. None of these test categories substitutes for independently validating a complete candidate.
