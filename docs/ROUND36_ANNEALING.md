# Targeted annealing: a 216-seam board and a faster score search

A full board with **216 of 240 matched seams** was found on 17 November 2026 by the annealing kernel added in v1.17.0. The earlier best was 207. The board uses all 160 tiles once with rotations only, and the project validator reports 216 matched seams and 45 line components. **It is not a Gold solution.** Gold needs all 240 seams and one closed loop. [The 216 board](ROUND36_ANNEAL_216.json) · [All measured runs](ROUND36_ANNEAL_RESULTS.json)

## What the engine does

**Targeted annealing** is a score search. It starts from a random full board and makes small changes, keeping a change with a probability that depends on how the matched-seam count moves and on a temperature that falls during each cooling run. A change either turns one tile or exchanges two tiles, each taking its best rotation. When a cooling run ends, the chain starts another run from a new random board. Each CPU worker runs its own chain.

It does not search for the single loop, and it never proves that a region of the search is empty. A 240-seam board from this engine is saved as `edge-perfect.json`, and only a board that the validator accepts as one closed loop earns `solution.json`.

## Measured changes

All runs used the release rules on a cloud machine with four x86-64 virtual CPUs, one run per core, compiled with GCC `-O3`. Scores are the best matched-seam count seen in a run.

| Change | Measurement |
| --- | --- |
| Compiled kernel | 26 million untargeted moves per second on one core. The plain C annealer written earlier the same day made 4.6 million. With targeted moves it makes 15 to 20 million, depending on the targeted share. |
| Targeted moves | On seeds 1 to 8 at 300 million moves each, random pairs averaged **202.1** matched seams (201 to 204). Picking a mismatched side and bringing in a tile that carries the needed endpoints for 960 of every 1,024 exchanges averaged **210.0** (207 to 213). Shares of 256, 512 and 768 averaged 208.5, 209.5 and 208.9. |
| Cooling range | At 1.5 billion moves and a 960 share on seeds 21 to 24, cooling from 0.5 to 0.09 averaged **214.0** (213, 214, 216, 213). From 2.0 to 0.09 averaged 213.75, from 0.5 to 0.12 averaged 213.5, and from 2.0 to 0.12 averaged 212.25. On seeds 11 to 14, stopping at 0.20 instead of 0.12 lost about 1.5 seams. The defaults are 0.5 and 0.09. |
| Against the earlier annealer | The earlier annealer reached 203 to 209 over 28 runs of 1.5 billion moves, averaging 206.1. The 20 runs of 1.5 billion moves with a 960 share and a cold temperature of 0.12 or 0.09 (seeds 11 to 14 and 21 to 24) reached 210 to 216, averaging 213.35. The seeds differ between the two sets. The targeted-moves row is the same-seed comparison. |
| Longer runs | Three runs of 8 billion moves reached 216, 215 and 215. More moves per run gave little above 1.5 billion. |
| Parallel tempering | Two 90-second runs on four cores with exchanges between temperatures reached 214 and 213. Independent chains did as well on the same budget, so tempering is not included. |

In the app, four chains at 300 million moves per run reached 215 in 11 seconds on the same machine (seed 5). The default of 1.5 billion moves per run takes about two minutes per run and chain.

## Completion check

Each new best board at 200 seams or more gets an exact check in the background. Every cell whose three seams already match stays fixed, and the engine searches all ways to put the remaining tiles back so that every seam matches. It then repeats the search with one more ring of cells freed. The result is saved with the board in `best.json` as `completion_check`.

For the 216 board both searches finished with no completion (43 free cells, then 98). The same held for every 208, 209, 211 and 215 board tested, and a separate C program gave the same answers for the 216 board and for the other boards it was run on. This says that these boards cannot be finished into a 240-seam board by rearranging only their mismatched cells and the neighbours of those cells. It says nothing about boards that differ in more cells.

## Running it

Choose **Targeted annealing** in the dashboard's search list, set the hours, seed and CPU workers, and select **Start**. From a terminal:

```powershell
.\StartSolver.cmd -Algorithm anneal -Hours 1 -Workers 4
```

```bash
bash StartSolver.sh --algorithm anneal --hours 1 --workers 4
```

Stop saves every chain, and the next start with the same seed and worker count continues each cooling run where it stopped. With a different seed or worker count the chains start fresh and the best board is kept. State is in `runtime/searches/anneal-edges/checkpoint.json`.

`anneal_search.py` also accepts `--moves`, `--t-hot`, `--t-cold`, `--targeted` (0 to 1024) and `--check-floor` for experiments.

## The compiled kernel

The kernel is one C99 file, `anneal_native.c`, with no dependencies. The first annealing run builds it with a local GCC or Clang into `.native/`, the same way as the other optional C libraries. Without a compiler the engine uses a Python kernel that makes the same kinds of moves at about 150,000 moves per second, more than 100 times slower. The dashboard's **Annealing runs** panel names the kernel in use. The two kernels draw random numbers differently, so the same seed gives different boards.

## Limits

The scores level off near 216 on this machine. No run reached 217. The engine has been run on Linux only. The Windows start script was edited but not executed for this release, and the kernel has not been compiled on Windows or macOS.
