# Round 26 work ownership and adaptive allocation

Workstreams F01 and F08 are implemented and independently reviewed. This report distinguishes finite mechanism evidence from the final consolidated validation and hardware recommendations; the original Gold puzzle remains unsolved.

## Symbolic ownership

`constraint_symbolic_frontier.py` represents pending, completed, and leased work in one decision DAG. The scheduler materializes only the next leased cube. Validated worker checkpoints atomically retire only completed portions of that lease. Requeue and CPU/SAT/CUDA transfer preserve the exact unfinished set. Cartesian volume is a representation statistic, not a count of legal puzzle arrangements or a percentage toward solving Gold.

Reachable-node collection, bounded alternative variable orders, and immutable JSON node interning reduce duplicated checkpoint structure. Ordering trials cannot change ownership and publish nothing if externally cancelled. Resource caps remain explicit. Native engine cursors still retain their own necessary search state.

## Integrated engine ownership

`constraint_coordinator_round26.py` uses disjoint leases with persistent worker processes, bounded slices, exact native cursor validation, and independent final-board validation. CPU, persistent SAT, and CUDA domain jobs can share a queue. Changing worker counts or enabled engines reassigns unfinished ownership. A hard worker failure may replay its uncommitted slice; a counter reports that event. Normal saved completed work is not replayed.

CPU nodes, SAT decisions, CUDA decisions, and committed engine slices are separate units. The app must not combine them into one misleading work score. A saved solution is revalidated and restored without starting workers.

## Contextual effort and strategy

`constraint_context_policy_round26.py` selects CPU profiles from bounded completed-residual observations. Timeouts never become training labels. Scale-free domain/seam features select nearby contexts, with forced exploration and explicit held-out exclusion support. Predictions change priorities only.

The optional contextual propagation policy preserves any context with demonstrated deductions, even if expensive. Unproductive optional checks are retried; mandatory checks and proof replay bypass suppression. Context limits and eviction discard scheduling hints only.

## Development validation so far

Finite tests enumerate Boolean/multivalue symbolic sets under alternate orders and compare randomized ownership/restart sequences with explicit sets. Other tests cover malformed snapshots, alias-free shared-state restoration, source/objective boundaries, CPU/SAT/CUDA adapters, warm process reuse, worker-count changes, terminal-state restoration, and independent Gold validation. The synthetic shared-JSON discriminator is less than one twentieth the flat representation; it is not a whole-application memory benchmark.

Checked cross-engine exchange, shared SAT ownership, adverse worker lifecycle checks, completed-residual held-out evaluation, public UI controls and capability gates are integrated. The release results separately record the consolidated frozen comparisons and final cross-platform checks.

## Completed residual development comparison

The reproducible `benchmark_context_round26.py` compares four CPU profiles on 24 finite Gold residuals. Each result is checked against independently enumerated full boards. A held-out prediction excludes every training case from the same puzzle family. All 96 runs completed. Fixed baseline work totals 129 CPU decisions, selected profiles total 122. Propagation visits, probe work and elapsed time are retained separately in the receipt; the decision totals do not establish an original-puzzle speedup. Source-frozen development receipt: `audit/optimization-round26/context-frozen-development/results.json`. A final clean-revision run remains required.

The original-root exchange now independently validates shared loop and conflict claims before retiring unleased symbolic work. Local assumptions remain guards. CPU/SAT worker imports revalidate records; transport and worker identity never authorize deductions. The app saves validated coordinator ownership about every five seconds and on Stop, while native engine units remain separate on the dashboard.

The optional native proof producer now bounds startup and reply waits, cancels only its owned optional subprocess and leaves interrupted search ownership intact. `setup_proof_worker.py` prepares a compatible Linux/macOS interpreter or an existing WSL distribution. It does not install WSL. `--verify-only` passes a native four-clause proof through the independent host indexed RUP checker. Native Windows proof-log loading remains disabled because of the recorded incompatible cleanup behavior.


## Optional isolated proof producer

The option is **Isolated proof producer** and implies independent indexed proof
checking. Linux/macOS use the active compatible Python environment. On Windows,
it uses an existing WSL distribution and a project-local Linux environment;
it does not load the disabled native Windows proof logger or install WSL.
Prepare and independently verify it from the project folder:

```text
python setup_proof_worker.py --distro Ubuntu
python setup_proof_worker.py --distro Ubuntu --verify-only
```

On Linux/macOS omit `--distro Ubuntu`. Use the compatible Python 3.12 solver
profile there. The setup helper installs the pinned `python-sat` package only in
the selected interpreter environment, so activate the project environment first.
Windows additionally requires Python and its venv support inside the chosen WSL
distribution. Each exported proof must pass the host checker; unavailable,
interrupted, oversized or rejected production gives no pruning authority and the
bounded in-process proof fallback remains available.


Adaptive allocation also reassesses a job after 24 consecutive slices with no
committed frontier reduction. It moves one exact unfinished cube to a different
enabled engine, or to a CPU profile with different effective controls; the rest
stays symbolic. No meaningful alternative means no forced restart. Productive
rotation retains its eight-slice policy. Both can discard internal native work,
which is explicitly counted as an adaptive cursor restart. A stalled frontier
is not evidence that native search made no progress, and this policy remains
optional. Set-allocation failure leaves the old ownership and cursor intact.
