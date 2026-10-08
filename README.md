# Diamond Dilemma Gold Solver

A GPU search tool for the **Gold Challenge** of Diamond Dilemma on Windows, Linux, and macOS, with a local 3D dashboard, resumable search branches, and independent solution validation.

**This repository contains a solver, not a verified Gold solution.** A complete result must place all 160 tiles once, match all 240 edges, and join every gold segment into **one closed loop**. The default search stops at its first 240-edge arrangement, even if it contains several loops. Such a result is saved as `edge-perfect.json`; only a verified single loop earns `solution.json`.

The puzzle description and diagrams are on [Jaap's Puzzle Page](https://www.jaapsch.net/puzzles/diamdil.htm).

## Requirements

Use **Python 3.12, 64-bit**, a compatible GPU and its driver, and enough free RAM and disk space for the selected population and checkpoints. Initial dependency installation and puzzle-data preparation need internet access.

| Platform and GPU | Compute backend |
| --- | --- |
| Windows or Linux, NVIDIA | CUDA 12, or WebGPU |
| Windows, AMD or other compatible GPU | Native WebGPU through DX12 or Vulkan |
| Linux, AMD or other compatible GPU | Native WebGPU through Vulkan |
| macOS, Apple Silicon (ARM64) | Native WebGPU through Metal |
| macOS, Intel (x86_64), with a Metal-capable GPU | Native WebGPU through Metal |

WebGPU here uses a native `wgpu` compute library; it does not run the search inside the browser. macOS does not use CUDA. Setup installs Python packages, not graphics drivers. Use a native ARM64 Python on Apple Silicon and x86_64 Python on Intel Macs.

The CUDA implementation has been tested on an **RTX 2060 with 6 GB VRAM**. Bounded WebGPU enumeration and checkpoint tests have also run on that NVIDIA GPU through Vulkan. AMD GPU and macOS Metal execution have not been hardware-tested in this project; package installation and CPU checks do not establish that hardware coverage. CPU tests need no GPU or source diagrams. Actual search requires a compatible GPU; there is no CPU search fallback. More replicas increase resource use and do not guarantee a faster solution.

## Quick start

### Windows

Clone or download this repository, open PowerShell in its folder, and run:

```powershell
.\Setup.cmd
.\OpenDashboard.cmd
```

Setup creates `venv\`, installs the dependencies, and prepares the puzzle data locally. You can also double-click `Setup.cmd` and `OpenDashboard.cmd`. These wrappers run the bundled scripts without changing your system-wide execution policy. The dashboard opens at [127.0.0.1:8766](http://127.0.0.1:8766/).

### Linux and macOS

Open a terminal in the repository folder:

```bash
bash Setup.sh
bash OpenDashboard.sh
```

Setup tries `python3.12`, then a compatible `python3` on PATH, creates `venv/`, and prepares the puzzle data. It does not use `sudo` or change system Python packages. To select a Python interpreter, use `bash Setup.sh --python /path/to/python3.12`. The dashboard is local to [127.0.0.1:8766](http://127.0.0.1:8766/). A desktop browser is needed for the 3D dashboard; command-line search also works without it.

Setup defaults to `auto`: it installs WebGPU on every platform and adds CUDA on Windows/Linux only when an NVIDIA driver query succeeds. To select packages explicitly:

```bash
# Native Apple/AMD/Intel GPU backend; supported on Linux and macOS.
bash Setup.sh --backend webgpu

# NVIDIA backend on Linux.
bash Setup.sh --backend cuda
```

Windows equivalents are `.\Setup.cmd -Backend webgpu` and `.\Setup.cmd -Backend cuda`. CUDA setup is rejected on macOS. Existing virtual environments are reused only when compatible; use a separate checkout/environment for each operating system and native CPU architecture.

### Search controls

Choose the duration and replica count, then select **Start search**. Drag the diamond to rotate it and use the mouse wheel to zoom. **Live branch** shows a real partial assignment; empty cells have not yet been filled. Closing the browser leaves the solver running.

Select **Stop** to finish the current work and save progress. Wait until the status becomes **Stopped** before starting again. **Unlimited** removes the time limit; Stop, the first 240-edge candidate, or exhaustion of the saved frontier can still end the run.

## Command line

The launcher defaults to systematic search, resumes compatible saved progress, and selects an available GPU backend automatically.

Windows:

```powershell
# Run for one hour.
.\StartSolver.cmd -Hours 1 -Replicas 4096

# Run without a time limit.
.\StartSolver.cmd -Unlimited -Replicas 4096

# Request a graceful stop and checkpoint.
.\StopSolver.cmd
```

Linux and macOS:

```bash
bash StartSolver.sh --hours 1 --replicas 4096
bash StartSolver.sh --unlimited --replicas 4096 --backend webgpu
bash StopSolver.sh
bash Status.sh
```

`OpenDashboard.sh --no-browser` starts the local dashboard service without opening a browser. `--backend auto`, `cuda`, or `webgpu` selects the compute backend when starting from the shell. On Windows, use the corresponding `-Backend` option.

The corresponding Python command on Windows is:

```powershell
.\venv\Scripts\python.exe systematic_search.py --resume --seconds 0 --replicas 4096
```

The dashboard accepts 128–131,072 replicas, including when resuming; the selected GPU must have enough memory and support the required compute limits. To reduce GPU load, **Stop**, wait for the saved **Stopped** state, choose fewer replicas, then **Start**. Unfinished branches that do not fit remain paused with their exact search positions; increasing the count later can bring them back onto the GPU. The dashboard shows active and paused branches separately. Paused state still uses host memory and checkpoint storage. Use a separate `--output` directory for an independent search.

## What is saved

| Path under `runtime/` | Purpose |
| --- | --- |
| `systematic/frontier.npz` | Immutable list of disjoint prefix jobs |
| `systematic/checkpoint.npz` | Active and paused DFS stacks, cursors, and job ownership saved together |
| `systematic/candidate-job-*.json` | Independently checked complete candidates |
| `live.json` | Current partial branch snapshot |
| `best.json`, `best.html` | Saved complete candidate and its validation report |
| `edge-perfect.json` | A 240-edge match; may contain multiple loops |
| `solution.json` | A fully validated Gold solution |
| `inputs/`, `runs/` | Input snapshots and run-specific source provenance |

Normal Stop/Resume preserves search progress. A crash or power loss can replay work since the last completed checkpoint. Restoring an older checkpoint also restores its older progress.

The systematic CUDA and WebGPU engines share the checkpoint and paused-branch format. Bounded tests passed checkpoint transfer in both directions and continued search after restoration. Resuming requires the same puzzle data, geometry, cell order, and saved frontier; keep the complete output directory and input snapshots when changing `--backend`. This compatibility applies to the systematic engines.

The stochastic backend (`solve.py`) and standalone DFS driver (`dfs_gpu.py`) have different state formats. Use separate output directories when comparing them. Their counters and checkpoints cannot be combined into systematic coverage.

See [Search, checkpoints, and validation](docs/SEARCH.md) for the exact search model and backend commands.

## Puzzle data

Source diagrams and extracted tile descriptions are **not bundled** in this repository. The local preparation step downloads the three reference GIFs from Jaap's site, checks their expected SHA-256 hashes, and runs endpoint and path extraction. Generated data, overlays, and runtime records are ignored by Git.

The expected transcription contains 160 tiles, 365 gold segments, and 730 endpoints. Extraction audits concern the published diagrams; they are not a check of a physical puzzle set. Changed source bytes must be reviewed instead of silently accepting a different transcription.

To prepare the data separately:

```powershell
.\venv\Scripts\python.exe prepare_data.py
```

If you already have the original reference GIFs, use `prepare_data.py --source-dir PATH` to prepare from those local copies. Windows `Setup.ps1 -SourceDir PATH` and Linux/macOS `bash Setup.sh --source-dir PATH` forward that option; installing Python packages still requires access to the package index. On Linux/macOS, `bash Setup.sh --skip-data` installs dependencies without downloading or extracting puzzle diagrams; run `venv/bin/python prepare_data.py` before starting the dashboard or search. Windows also supports `Setup.cmd -SkipData`.

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
.\.venv-test\Scripts\python.exe -B -m unittest -v test_geometry test_exact_frontier test_systematic_jobs test_systematic_runner test_dashboard_control test_stop test_position_cache test_prepare_data test_atomic_io test_replica_resize test_linux_setup test_windows_setup test_platform test_webgpu
```

On Linux/macOS, use the same test modules with a native Python virtual environment:

```bash
python3.12 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-dev.txt
.venv-test/bin/python -B -m unittest -v test_geometry test_exact_frontier test_systematic_jobs test_systematic_runner test_dashboard_control test_stop test_position_cache test_prepare_data test_atomic_io test_replica_resize test_linux_setup test_windows_setup test_platform test_webgpu
```

CI runs CPU checks on Windows, Ubuntu 24.04, Intel macOS, and Apple Silicon macOS. Windows-specific sharing and launcher tests skip elsewhere. Separate installation jobs import the native WebGPU library on all four platforms; an Ubuntu CUDA package check imports CuPy. These installation checks use `--skip-data`, do not request a GPU adapter, and do not validate kernel execution. A separate shader job executes finite WGSL checks through Mesa lavapipe, a software Vulkan implementation. That test explicitly enables a test-only software adapter; production search still requires hardware. It verifies shader behavior on those finite fixtures, not AMD or Apple GPU compatibility.

GPU and transcription checks are separate. After full CUDA setup on an NVIDIA machine, useful checks include:

```powershell
.\venv\Scripts\python.exe -B -m unittest -v test_data test_pending
.\venv\Scripts\python.exe -B test_disjoint_dfs.py
```

To run the finite WebGPU hardware checks after WebGPU setup:

```bash
DIAMOND_TEST_WEBGPU=1 venv/bin/python -B -m unittest -v test_webgpu
```

On PowerShell, set `$env:DIAMOND_TEST_WEBGPU='1'`, then run `.\venv\Scripts\python.exe -B -m unittest -v test_webgpu`. Leave the software-adapter test opt-in unset for hardware verification.

On an NVIDIA machine with both CUDA and WebGPU dependencies installed, compare device state after every launch and test checkpoint interchange using the prepared puzzle data:

```powershell
$env:DIAMOND_TEST_WEBGPU='1'
$env:DIAMOND_CUDA_PYTHON=(Resolve-Path .\venv\Scripts\python.exe).Path
$env:DIAMOND_TEST_TILE_DATA=(Resolve-Path .\data\tiles.json).Path
.\venv\Scripts\python.exe -B -m unittest -v test_webgpu.NativeTests.test_cuda_checkpoint_interchange_every_launch
```

Linux equivalent:

```bash
DIAMOND_TEST_WEBGPU=1 DIAMOND_CUDA_PYTHON="$PWD/venv/bin/python" DIAMOND_TEST_TILE_DATA="$PWD/data/tiles.json" \
  venv/bin/python -B -m unittest -v test_webgpu.NativeTests.test_cuda_checkpoint_interchange_every_launch
```

`DIAMOND_CUDA_PYTHON` may instead point to a separate environment with the CUDA dependencies. Omitting `DIAMOND_TEST_TILE_DATA` uses the test's synthetic edge masks.

On the RTX 2060 through native Vulkan, the 54-arrangement fixture passed enumeration, resizing, pending-candidate freezing, and counter checks. With the actual puzzle data and a varied cell order, every device state field matched CUDA after each of 20 launches with a 32-node budget per lane. Checkpoints transferred both ways and resumed consistently. A separate 128-replica systematic run saved, restored, and advanced its saved search state. These are bounded compatibility checks, not AMD or Metal hardware tests.

Passing tests is not a puzzle solution or a prediction of completion time.

## Source packages

Successful CI runs produce source archives labelled Windows x64, Linux x64, macOS x64, and macOS ARM64, with SHA-256 checksums. They contain the same portable source and setup scripts, not bundled standalone executables. Install native Python 3.12 and run the appropriate setup script after extraction. Environments, runtime state, puzzle diagrams, and generated tile data are excluded.

## License

Project code is distributed under the [GNU GPLv3](LICENSE). The third-party puzzle diagrams are obtained separately; the code license does not grant rights to those materials. See [Third-party notices](THIRD_PARTY_NOTICES.txt).
