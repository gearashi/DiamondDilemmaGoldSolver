# Round 26 complete recombination neighborhoods (F16)

`optimal_crossover` selects a CPU host optimizer shared by the CUDA and WebGPU samplers. It implies the existing crossover option and remains default off. It optimizes a defined parental neighborhood exactly over successive bounded work leases; it does not enumerate or exclude the full original puzzle search space.

| Agreed scope | Implemented mechanism | Boundary |
| --- | --- | --- |
| F16.01 | Optimize interacting subsets of all relative physical-permutation cycles, including singleton rotation differences. | Complete neighborhood optimization can take many calls. Unfinished leases are not optimality claims. |
| F16.02 | Broaden completed pair models to independent legal rotations and then bounded three-/four-parent closure models; configurable up to eight parents. | Multi-parent components preserve the physical inventory by construction. These are explicit closure neighborhoods, not arbitrary new-piece permutations. |
| F16.03 | Exact seam factors, safe upper bounds, conditional Gold connectivity cuts, independent score and Gold validation. | Cuts can bound the sampling objective only when mathematically valid; non-Gold improvements are not discarded merely because they are not solutions. |
| F16.04 | Retain pair models, factor caches, DFS frontiers, completed outcomes and exact duplicate-child records across calls and local checkpoints. | Bounded eviction and interrupted unfinished factor calculations can repeat work. Native solver state is not involved. |
| F16.05 | Fair exploration plus deterministic parent ranking from completed gains/work; complete models advance to larger neighborhoods. | Incomplete runs do not supply an optimization reward or a completion claim. |

Implementation is in [sampling_crossover_round26.py](../sampling_crossover_round26.py). `OptimizingCrossover` has the same public proposal/snapshot/injection interface as the prior crossover. Defaults are 256 optimizer steps per call, eight retained models, 256 completed score records and four parents. The optional `check` callback preserves completed work on interruption, and pending candidate publication survives a local restart. Injection is counted only after the sampler's native write commits.

Two-parent cycle unions and multi-parent closure components each contain the same physical piece set in every parent. Choosing one complete pattern per component therefore cannot duplicate a piece. The enlarged rotation phase varies legal rotations independently after the physical pattern is chosen. Exact seam factors provide an optimistic bound even when unassigned cells remain correlated.

Gold cuts use the conditional inequality `cross(W) - 2*occupied(p in W) - 2*occupied(q outside W) >= -2`. A violated upper bound rules out Gold, not every useful candidate. For a legal edge-perfect board with at least two segments, absence of Gold implies at least two proper cycles; the resulting objective cap is safe. Lower seam scores retain their own optimistic bound. Every published candidate is rescored independently and checked against the current domains/fixed placements; a Gold claim requires full independent validation.

Local continuations bind actual puzzle/domain/strand data, policy and algorithm sources. They contain typed controller/frontier/cache records, never a GPU pointer or native optimizer object. They are trusted local continuation data, not imported proof authority. Saved scores alone do not authorize publication: the candidate is independently rescored again before it is offered. Cold reconstruction costs after restore are counted separately from retained branch/factor work.

Bounded development tests include 24 independently enumerated two-/three-parent and enlarged-rotation configurations, a Gold child requiring multiple cycle changes that every prior single-cycle move misses, a negative Gold-cut fixture, exact split/resume work, cancellation at completed boundaries, pending publication, malformed state/scope rejection, bounded eviction, and a planted 160-cell geometry candidate. Thirteen core tests and three later completion/evidence-guard tests passed. Separate sampler integration tests passed on CUDA and WebGPU, including unchanged native state after 18 uninterrupted versus 9+9 resumed dispatches. Restored static-model/source-check accounting differs deliberately; retained optimizer/frontier/history state agrees.

The finite evidence runner is:

```text
python benchmark_recombination_round26.py --output audit/optimization-round26/recombination-finite.json --evidence-stage frozen
```

It requires a fresh output and a clean, unchanged source/runtime/Git identity for frozen evidence. It compares completed interacting-cycle, multi-parent/rotation and conditional-cut outcomes, separately accounting scorer calls, factor/cut work, branch work and restore costs. The [Round 26 release results](ROUND26_RESULTS.md) include the full validation suite and original-input trials. The unrestricted Gold puzzle remains unsolved.
