# Diamond Dilemma Gold Solver

A Windows GPU search tool for the **Gold Challenge** of Diamond Dilemma, with a local 3D dashboard, resumable search branches, and independent solution validation.

**This repository contains a solver, not a verified Gold solution.** A complete result must place all 160 tiles once, match all 240 edges, and join every gold segment into **one closed loop**. The default search stops at its first 240-edge arrangement, even if it contains several loops. Such a result is saved as `edge-perfect.json`; only a verified single loop earns `solution.json`.

The puzzle description and diagrams are on [Jaap's Puzzle Page](https://www.jaapsch.net/puzzles/diamdil.htm).

## Requirements

- Windows with PowerShell and **Python 3.12, 64-bit**.
- An NVIDIA GPU and driver compatible with **CUDA 12**. The GPU implementation has been tested on an **RTX 2060 with 6 GB VRAM**.
- Internet access for initial dependency installation and local puzzle-data preparation.
- Enough free RAM and disk space for the selected population and its checkpoints.

CPU tests can run without CUDA or downloaded puzzle data. Actual search requires an NVIDIA GPU; there is no CPU search fallback. More replicas increase resource use and do not guarantee a faster solution.

## Quick start

Clone or download this repository, open PowerShell in its folder, and run:

```powershell
.\Setup.cmd
.\OpenDashboard.cmd
```

Setup creates `venv\`, installs the dependencies, and prepares the puzzle data locally. You can also double-click `Setup.cmd` and `OpenDashboard.cmd`. These wrappers run the bundled scripts without changing your system-wide execution policy. The dashboard opens at [127.0.0.1:8766](http://127.0.0.1:8766/).

Choose the duration and replica count, then select **Start search**. Drag the diamond to rotate it and use the mouse wheel to zoom. **Live branch** shows a real partial assignment; empty cells have not yet been filled. Closing the browser leaves the solver running.

Select **Stop** to finish the current work and save progress. Wait until the status becomes **Stopped** before starting again. **Unlimited** removes the time limit; Stop, the first 240-edge candidate, or exhaustion of the saved frontier can still end the run.

## Command line

The launcher defaults to systematic search and resumes compatible saved progress.

```powershell
# Run for one hour.
.\StartSolver.cmd -Hours 1 -Replicas 4096

# Run without a time limit.
.\StartSolver.cmd -Unlimited -Replicas 4096

# Request a graceful stop and checkpoint.
.\StopSolver.cmd
```

The corresponding Python command is:

```powershell
.\venv\Scripts\python.exe systematic_search.py --resume --seconds 0 --replicas 4096
```

The dashboard supports 128–131,072 replicas, including when resuming. To reduce GPU load, **Stop**, wait for the saved **Stopped** state, choose fewer replicas, then **Start**. Unfinished branches that do not fit remain paused with their exact search positions; increasing the count later can bring them back onto the GPU. The dashboard shows active and paused branches separately. Paused state still uses host memory and checkpoint storage. Use a separate `--output` directory for an independent search.

## What is saved

| Path under `runtime\` | Purpose |
| --- | --- |
| `systematic\frontier.npz` | Immutable list of disjoint prefix jobs |
| `systematic\checkpoint.npz` | Active and paused DFS stacks, cursors, and job ownership saved together |
| `systematic\candidate-job-*.json` | Independently checked complete candidates |
| `live.json` | Current partial branch snapshot |
| `best.json`, `best.html` | Saved complete candidate and its validation report |
| `edge-perfect.json` | A 240-edge match; may contain multiple loops |
| `solution.json` | A fully validated Gold solution |
| `inputs\`, `runs\` | Input snapshots and run-specific source provenance |

Normal Stop/Resume preserves search progress. A crash or power loss can replay work since the last completed checkpoint. Restoring an older checkpoint also restores its older progress.

The stochastic backend (`solve.py`) and standalone DFS driver (`dfs_gpu.py`) have different state formats. Use separate output directories when comparing them. Their counters and checkpoints cannot be combined into systematic coverage.

See [Search, checkpoints, and validation](docs/SEARCH.md) for the exact search model and backend commands.

## Puzzle data

Source diagrams and extracted tile descriptions are **not bundled** in this repository. The local preparation step downloads the three reference GIFs from Jaap's site, checks their expected SHA-256 hashes, and runs endpoint and path extraction. Generated data, overlays, and runtime records are ignored by Git.

The expected transcription contains 160 tiles, 365 gold segments, and 730 endpoints. Extraction audits concern the published diagrams; they are not a check of a physical puzzle set. Changed source bytes must be reviewed instead of silently accepting a different transcription.

To prepare the data separately:

```powershell
.\venv\Scripts\python.exe prepare_data.py
```

If you already have the original reference GIFs, use `prepare_data.py --source-dir PATH` to prepare from those local copies. `Setup.ps1 -SourceDir PATH` forwards that option; installing Python packages still requires access to the package index.

## Verify a candidate

```powershell
.\venv\Scripts\python.exe validator.py data\tiles.json runtime\best.json
```

The validator checks tile uniqueness, every seam, and gold-loop connectivity. Drawn crossings are not junctions. It exits successfully only for a complete Gold solution. For a result made with another input file, use the result's recorded `input_snapshot`.

## Tests

The CPU suite uses synthetic fixtures and needs no source diagrams or GPU:

```powershell
py -3.12 -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-test\Scripts\python.exe -B -m unittest -v test_geometry test_exact_frontier test_systematic_jobs test_systematic_runner test_dashboard_control test_stop test_position_cache test_prepare_data test_atomic_io test_replica_resize
```

GPU and transcription checks are separate. After full setup, useful checks include:

```powershell
.\venv\Scripts\python.exe -B -m unittest -v test_data test_pending
.\venv\Scripts\python.exe -B test_disjoint_dfs.py
```

Passing tests is not a puzzle solution or a prediction of completion time.

## License

Project code is distributed under the [GNU GPLv3](LICENSE). The third-party puzzle diagrams are obtained separately; the code license does not grant rights to those materials. See [Third-party notices](THIRD_PARTY_NOTICES.txt).

See [third-party notices](THIRD_PARTY_NOTICES.txt) for source credits.
