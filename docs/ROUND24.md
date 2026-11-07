# Round 24: complete 42-proposal register

This round implements and evaluates all 42 documented proposals in the scopes
listed below. **The original Gold Challenge remains unsolved.** A successful
board must use all 160 physical tiles once, match all 240 seams and connect every
gold segment into one closed loop. An edge-perfect arrangement with several
loops is not a Gold solution.

The priority is fewer decisions, stronger sound deductions and less repeated
reasoning. A feature can be useful even when each decision takes longer. A low
node rate, fewer nodes before timeout, a deeper partial placement or a higher
sampling score does not by itself establish less total work to solve the puzzle.
Different engines and algorithms also count different primitive operations.

All new controls are opt-in. **Retained** means the stated finite checks
establish the implemented reuse, deduction or completed-work benefit.
**Experimental** means correctness or a limited discriminator is established,
but practical benefit is workload-dependent or unproven. Neither label means
that enabling all options together is best. General versions of several ideas
remain outside the bounded implementations; those limits are explicit below.

The [machine-readable register](round24-candidates.json) contains the same 42
IDs, controls, evidence files and limits. This is a new round following the
[52-proposal round 23 register](ROUND23.md); the IDs are local to each round.
Component details are in [learning and proof transport](ROUND24-learning.md),
[regional/residual reasoning](ROUND24_REGIONS.md),
[CUDA domains and hybrid repair](ROUND24_GPU_HYBRID.md), and
[the fitted-ordering experiment](ROUND24_ORDERING.md).

## How to read the controls

Most names below are boolean keys in `--search-options` and the dashboard’s
experimental optimization controls. Use the stated engine: a CPU deduction
option does not automatically change a GPU kernel. `witness_pool` and
`support_cover` enable `sat_assist`; `proof_transport` enables `smart_learning`.
`proof_probes` is a separate GPU DFS feature under `--gpu-features`.
Persistent standalone SAT and SAT theory propagation are separate selectable
architectures; combining `persistent_sat` and `sat_theory` is rejected.

The new **CUDA domains + learned backjumping** engine (`gpu-domains`) includes
proposals 10, 33 and 34 as one alternative architecture. Its full domain/reason
state supports 128–512 lanes within its memory budget. The existing compact GPU
engines retain their larger replica counts and separate checkpoint formats.
The new domain engine requires NVIDIA CUDA; the hybrid sampling additions also
have WebGPU implementations. AMD/Apple hardware coverage is not established by
NVIDIA or software-backend checks.

Use Python 3.12 with the full dependencies for CUDA and native SAT/CP-SAT repair.
The Python 3.15 profile supports its existing DFS/WebGPU subset; unavailable
optional native engines do not silently become proof sources. See the
[README requirements](../README.md#requirements).

## Reuse and work ownership

| # / proposal | Access and implemented mechanism | Finite evidence | Disposition and scope limit |
| --- | --- | --- | --- |
| **1. Resume unfinished reasoning** | `resumable_reasoning`. Save unfinished cut-vector enumeration and its private support union; publish a deduction only when the required alternatives finish. | A finite continuation check uses 5 matching checks instead of 8; interruption at every tested checkpoint preserves the exact result. [continuation](../test_constraint_continuation.py). | **Retained reuse; bounded implementation.** This resumes selected cut reasoning, not every CPU propagator. Direct regional diagrams additionally retain same-process construction sessions (#17). |
| **2. Prune queued frontier with verified exclusions** | `queue_pruning; parallel-dfs, or dfs with shared_coverage`. Subtract independently verified exclusions from queued domain boxes before launching workers. Exact splitting preserves the remaining disjoint union. | A verified loop excludes a queued job with zero workers started; randomized box subtraction agrees with explicit assignments. [queue_work](../test_constraint_queue_work.py); [work](../test_constraint_work.py). | **Retained reuse.** Active worker jobs retain their own ownership; a size cap leaves the previous pending cover unchanged. |
| **3. Transfer completed coverage across policies** | `shared_coverage; dfs, parallel-dfs, persistent SAT, gpu-domains`. Store completed infeasible boxes in a scoped SQLite ledger and subtract them when changing compatible policies or engines. | Restart and policy-change checks avoid executing completed work; objective/root mismatch and incomplete/solved entries are rejected. [work](../test_constraint_work.py); [queue_work](../test_constraint_queue_work.py); [portable](../test_constraint_portable.py); [gpu_domain_runner](../test_gpu_domain_runner.py). | **Retained reuse.** Trusted local progress is not a portable mathematical proof. A completed imported subset does not establish exhaustion of the entire puzzle. |
| **4. Transfer unfinished frontiers across engines** | `portable_frontier; --work-frontier PATH`. Export the exact unfinished union as disjoint domain boxes; import it into parallel DFS, persistent SAT or CUDA domains. | CPU/SAT and CPU/GPU/CPU round trips preserve unfinished work; overlap, scope and corruption checks reject invalid inputs. [portable](../test_constraint_portable.py); [gpu_domain_runner](../test_gpu_domain_runner.py). | **Retained transport.** Native internal trails or SAT clauses may be rebuilt after transfer. The compact legacy GPU engine is not a general receiver for this format. |
| **5. General proof-carrying transport** | `proof_transport; automatically enables smart_learning`. Produce bounded edge-model DPLL refutations and independently check their RUP clauses before accepting transported mask exclusions. | A checked certificate avoids native SAT re-solving; altered proofs/scope fail validation. A child-process test checks clean exit after UNSAT proof generation. [proof](../test_constraint_proof.py); [learning_round24](../test_constraint_learning_round24.py). | **Retained checked transport; bounded proof producer.** Gold-only contradictions outside the edge relaxation still need ordinary verification. Native proof logging was rejected after a Windows crash; the pure bounded producer is used. |
| **6. Persistent standalone SAT sessions** | `persistent_sat with algorithm sat`. Keep one plain-SAT model, learned native state and guarded cube assumptions across warm slices; checkpoint remaining cubes and verified Gold cuts. | Warm slices build the model once; completed cubes stay absent after restart; independently enumerated edge/Gold fixtures agree. [sat_session](../test_constraint_sat_session.py); [portable](../test_constraint_portable.py). | **Retained warm reuse.** Process restart rebuilds native SAT internals and may repeat unfinished inference. This is plain SAT, not a portable checkpoint for every SAT/CP backend. |
| **7. Cross-context complete support witnesses** | `witness_pool; automatically enables sat_assist`. Revalidate complete semantic SAT models in new domain contexts, supplying many placement supports without another solver query. | On 60 finite contexts, identical supported domains require 72 native calls versus 874 for the prior scan. [sat_round24](../test_constraint_sat_round24.py); [benchmark: learning_round24](../benchmark_learning_round24.py). | **Retained reuse.** These are edge-relaxation support witnesses; they are not accepted as Gold solutions. The pool and retained work are bounded. |
| **8. Canonical residual reuse** | `residual_reuse`. Reuse completed negative results when live domains, inventory, exposed masks and contracted path wiring describe the same remaining problem. | Equivalent fixed interiors share a key; the public DFS path reuses a completed proof before propagation. Timeout and wrong-root reuse cannot create exclusions. [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [residual_reuse_round24](../test_constraint_residual_reuse_round24.py). | **Retained reuse; contracted labelled states.** Actual cell/piece labels remain. Arbitrary graph isomorphism and physical-piece renaming are not implemented; reuse is bound to the owned root. |
| **9. Incremental Hall SCC maintenance** | `incremental_hall`. Repair the piece matching and revisit only affected alternating-graph SCCs when the matching stays valid; reuse exact saved snapshots on rollback. | Random shrink/widen/backtrack sequences match exhaustive assignments. One six-cell example revisits 2 SCC vertices and reuses 4. [incremental_hall](../test_constraint_incremental_hall.py). | **Retained incremental reuse.** Matching changes or unrecognized widening trigger a complete pass. Interrupted partial computations are not committed. |

## GPU connectivity and stronger logical deductions

| # / proposal | Access and implemented mechanism | Finite evidence | Disposition and scope limit |
| --- | --- | --- | --- |
| **10. Reversible GPU strand connectivity** | `algorithm gpu-domains`. Maintain reversible per-lane strand connectivity through assignments and undo operations. | Complete tiny Gold solution sets agree with rebuilding connectivity, with fewer strand joins. [gpu_domain_round24](../test_gpu_domain_round24.py). | **Retained alternative CUDA architecture.** NVIDIA CUDA domain engine only; this does not add full reversible state to the compact WebGPU engine. |
| **11. Retain relational deductions** | `retained_relations`. Keep scoped binary path-consistency deductions so later domain restrictions can expose their unary consequences. | An initially non-pruning equality relation later forces its partner; independent enumeration checks narrower/wider contexts. [learning_round24](../test_constraint_learning_round24.py). | **Retained relational reuse.** Current extraction is from path consistency, not every constructive-lookahead or SAT relation. |
| **12. SAT support-cover batches** | `support_cover; automatically enables sat_assist`. Query a guarded disjunction of uncovered placements and use each returned model to cover multiple supports. | One fixture needs 11 calls versus 20; the wider 60-context corpus needs 965 versus 874. With witness pooling both use 72. [sat_round24](../test_constraint_sat_round24.py); [benchmark: learning_round24](../benchmark_learning_round24.py). | **Experimental; workload-dependent.** Retained as a selectable alternative, not a universal improvement. Selectors are disabled on completion or interruption. |
| **13. Joint seam-signature conservation** | `joint_conservation`. Solve simultaneous seam-signature capacity constraints with exact matching and support filtering. | A joint shortage passes separate class bounds but fails the combined model; 400 small instances match independent assignments. [learning_round24](../test_constraint_learning_round24.py); [benchmark: learning_round24](../benchmark_learning_round24.py). | **Experimental deduction.** No additional pruning after the prior fixpoint on the 60-context comparison corpus. Original-board benefit remains unestablished. |
| **14. Affine and modular invariants** | `affine_reasoning`. Combine affine port-vector constraints in global seam coordinates and reject inconsistent candidates. | All 256 subsets of three-bit vectors match independent affine closure; 80 puzzle contexts preserve all enumerated edge solutions. [learning_round24](../test_constraint_learning_round24.py); [benchmark: learning_round24](../benchmark_learning_round24.py). | **Experimental deduction; GF(2) runtime.** Runtime propagation uses GF(2). A mod-3 reference experiment exists, but general modular runtime propagation is not implemented; the comparison corpus found no extra pruning. |
| **15. Learned pseudo-Boolean inequalities** | `pb_learning`. Derive exact pseudo-Boolean capacity/seam inequalities through positive integer combinations and gcd rounding, then filter domains. | A contradiction survives primitive rows but is caught by a derived cut; 80 signed-integer systems preserve their complete model sets. [learning_round24](../test_constraint_learning_round24.py); [benchmark: learning_round24](../benchmark_learning_round24.py). | **Experimental certified deduction.** Derived cuts are bounded and checked. The 60-context prior-fixpoint comparison found no additional deletion. |
| **16. Broader SAT theory propagation** | `sat_theory with sat-propagator`. Explain partial Hall shortages and bounded affine consequences to the native SAT backend. | A partial shortage clause holds for every independently enumerated edge solution. [sat_round24](../test_constraint_sat_round24.py). | **Experimental native SAT theory.** This adds specific theory propagators, not the entire CPU pruning pipeline; it requires the compatible native SAT environment. |

## Regional representations, joins and relaxations

| # / proposal | Access and implemented mechanism | Finite evidence | Disposition and scope limit |
| --- | --- | --- | --- |
| **17. Direct symbolic regional diagrams** | `symbolic_regions`. Compile exact labelled regional histories directly into a memoized frontier DAG and join overlapping DAGs without expanding all rows. | The seven-cell discriminator preserves all 5,040 histories with 889 versus 60,620 candidate trials; sliced construction repeats no trials. [region_symbolic](../test_constraint_region_symbolic.py); [region_network](../test_constraint_region_network.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Retained symbolic reuse; bounded regions.** Selected regions and hard state/work caps apply. Unfinished DAG sessions survive only in the same process, not as portable disk continuations. |
| **18. Symbolic inventory families** | `symbolic_inventory`. Represent used-piece subsets with a ZDD and compute exact disjoint products and inventory supports. | Six regional groups retain the same 66 inventories each; 19,033 ZDD primitive operations replace 135,102 explicit inventory alternatives in the discriminator. [region_symbolic](../test_constraint_region_symbolic.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Retained symbolic reuse.** Primitive operation types differ, so this is not a speed ratio. Node caps yield unknown without eliminating support. |
| **19. Whole-constraint elimination and resource forgetting** | `elimination; full/conditioned APIs in constraint_region_elimination`. Eliminate seam/piece factors with inward/outward support messages, forgetting each resource only after its final possible occurrence; provide exact cutset conditioning. | A synthetic 160-cell resource fixture uses 2,080 factor operations at live width 2; independently validated full witnesses cover every retained placement. [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Retained bounded exact factor reasoning.** Full mode covers all cells for seams and inventory. Gold connectivity remains outside this relaxation. Wide original instances can cap; small selected queries are not a full-board completion proof. |
| **20. Mini-bucket reasoning** | `mini_buckets`. Split oversized elimination buckets into optimistic subproblems whose completed negatives remain sound. | The bounded K4 three-colour discriminator is possible at width 3 and rejected at width 4. [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Experimental relaxed deduction.** A feasible relaxation is not a solution; budget limits return unknown. Benefit depends on the chosen width and residual graph. |
| **21. Relaxed diagrams and refinement** | `relaxed_diagrams`. Merge states by unioning possibilities, then refine still-possible queries through increasing widths. | An odd-XOR triangle survives width 1 and is rejected after refinement; exhaustive checks preserve true solutions. [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Experimental relaxed deduction.** Merging is optimistic. Only a completed negative or exact result can justify a conclusion; caps are unknown. |
| **22. Representative connectivity families** | `representative_connectivity`. Use a GF(2) rank basis of connectivity rows inside the same inventory/boundary class. | 105 eight-endpoint matching rows reduce to 8 representatives, checked against all 105 outside matchings. [region_symbolic](../test_constraint_region_symbolic.py); [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Retained bounded existence compression.** Preserves existence, not counts or arbitrary placement supports. Reduction is limited to ten exposed endpoints; larger classes keep all rows. |
| **23. Exact subset-convolution joins** | `subset_joins`. Use exact integer ranked subset convolution for disjoint inventory joins, evaluating only cardinality layers present. | A ten-piece join replaces 63,504 explicit pair tests with 16,384 transform operations and preserves all 252 joined histories. [region_symbolic](../test_constraint_region_symbolic.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Retained bounded join alternative.** Different operation types are not directly comparable in time. The generic transform accepts at most 16 pieces; integrated queries default to ten. |
| **24. Lazy regional master** | `lazy_regions`. Generate regional realizations on demand and memoize only completed failed inventory/strand interfaces. | An eight-cell discriminator requests 2 local rows instead of preparing 3,360 eager rows and returns independently validated Gold. [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Retained bounded lazy master.** This is a bounded existence query, not a general full-size branch-and-price engine. Capped parent interfaces are never marked failed. |
| **25. Proof-derived interface formulas** | `interface_formulas`. Eliminate interior variables from optimistic regional CNF and replay each resolution derivation before retaining interface clauses. | 100 finite formulas preserve existential boundary models; puzzle checks preserve every enumerated completion under narrower contexts. [learning_round24](../test_constraint_learning_round24.py). | **Retained bounded checked projection.** Caps publish no formula; widening invalidates scoped use. Arbitrary backend Craig-interpolant extraction is not implemented. |
| **26. Certified optimization relaxations** | `certified_relaxation; priced_inventory_master research API`. Accept LP negatives only after exact rational Farkas checks; a separate regional DAG pricing prototype generates inventory columns on demand. | A feasible inventory master generates 24 columns from 90 histories; a shortage closes only after complete pricing and exact certification. [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Experimental certified relaxation.** The pricing master relaxes seams/Gold connectivity, and feasible LPs are not witnesses. Native LP proposals are optional and never proof authority. |
| **27. Label-consistent simultaneous routing** | `labelled_routing`. Require simultaneous port-disjoint routes with compatible orientation and physical-piece labels. | Individual routes exist in a discriminator but cannot coexist with consistent labels; exhaustive small puzzle cases retain every Gold solution. [region_reasoning_round24](../test_constraint_region_reasoning_round24.py); [benchmark: regions_round24](../benchmark_regions_round24.py). | **Retained bounded necessary connectivity check.** A Gold-only necessary condition with query/admission caps, not a general complete routing solver for arbitrary large residual boards. |

## Branching, ordering and obstruction discovery

| # / proposal | Access and implemented mechanism | Finite evidence | Disposition and scope limit |
| --- | --- | --- | --- |
| **28. Dual piece branching** | `dual_piece_branching`. Branch on whether a selected physical piece occupies a cell, partitioning that cell domain into present/absent masks. | Every independently enumerated completion belongs to exactly one child. [hybrid_round24](../test_constraint_hybrid_round24.py). | **Experimental ordering/partition policy.** Completeness is preserved; fewer total decisions on the original puzzle is unestablished. |
| **29. Strand-first branching** | `strand_branching`. Partition by continuation of a fixed strand into a neighbouring tile, including alternatives with identical seam masks. | A discriminator distinguishes internal partners even when side masks agree; child masks cover the original domain exactly. [hybrid_round24](../test_constraint_hybrid_round24.py). | **Experimental ordering/partition policy.** This changes the order/partition of exact work, not the feasibility conditions. |
| **30. Completion-count ordering** | `completion_ordering`. Count exact local seam/inventory extensions in a small star and use completed counts to rank values. | Counts agree with independent local enumeration; a capped count still retains every choice. [hybrid_round24](../test_constraint_hybrid_round24.py). | **Experimental bounded ordering.** Local counts are a relaxation of the whole board. They never remove candidates or imply a global completion count. |
| **31. Learned branching or repair policy** | `learned_ordering`. Use frozen five-feature ridge weights trained on completed child-search work to order values. | 46 completed labels on six training seeds; twelve disjoint ten-cell tests use 119 versus 133 nodes, with 7 wins, 2 ties and 3 losses. [learned_ordering_round24](../test_constraint_learned_ordering_round24.py); [benchmark: learned_ordering_round24](../benchmark_learned_ordering_round24.py). | **Retained optional fitted ordering; limited evidence.** Six further forty-cell pairs all cap and remain inconclusive. No original-board generalization or learned repair policy is claimed; predictions never prune. |
| **32. Verified obstruction discovery** | `obstruction_rules`. Mine mixed seam/inventory incompatibility cliques and odd cycles, then verify their exact at-most bounds. | Generated rules are rechecked; 45 puzzle contexts preserve every enumerated edge solution. Dominated rules do not inflate retained-rule counts. [learning_round24](../test_constraint_learning_round24.py); [benchmark: learning_round24](../benchmark_learning_round24.py). | **Experimental verified discovery.** Discovery is limited to these templates; no unrestricted theorem synthesis. No additional pruning in the 60-context prior-fixpoint comparison. |

## GPU reasoning and exact hybrid repair

| # / proposal | Access and implemented mechanism | Finite evidence | Disposition and scope limit |
| --- | --- | --- | --- |
| **33. GPU explanation learning and backjumping** | `algorithm gpu-domains`. Keep lane-local reasons and learned placement clauses; jump back to the relevant decision rather than undoing every intervening choice. | A completed Hall UNSAT fixture uses 42 decisions/757 support checks with learning versus 166/2,852 without. [gpu_domain_round24](../test_gpu_domain_round24.py); [gpu_domain_checkpoint_validation](../test_gpu_domain_checkpoint_validation.py). | **Retained alternative CUDA reasoning.** Clauses are scoped to their lane root. This is the new CUDA domain architecture, not a change to every legacy GPU lane. |
| **34. GPU reversible domains** | `algorithm gpu-domains`. Store complete reversible domains, removals, reasons and unfinished propagation scans in durable lane state. | Tiny exhaustive comparisons and checkpoint/restore tests preserve pending work; 160-cell-deep fixtures check remaining branch volumes. [gpu_domain_round24](../test_gpu_domain_round24.py); [gpu_domain_runner](../test_gpu_domain_runner.py); [gpu_domain_checkpoint_validation](../test_gpu_domain_checkpoint_validation.py). | **Retained alternative CUDA architecture.** The full-state engine supports 128–512 lanes within its memory budget, not 131,072. Batch lanes are not yet refilled dynamically. |
| **35. Proof-targeted CPU probes** | `GPU DFS feature proof_probes`. Rank bounded CPU probes by signs of a physical-piece shortage, then require the CPU to prove failure independently. | A 160-cell discriminator selects four cells competing for three pieces; the selected query proves infeasible. [gpu_probe_round24](../test_gpu_probe_round24.py). | **Retained proof-targeted scheduling.** Ranking alone never rejects a branch; this extends CPU assistance for the compact GPU DFS engine. |
| **36. Core-guided minimum-change repair** | `core_repair in hybrid`. Use infeasible assumption cores and an exact hitting-set master to seek a minimum number of changed cells. | Tiny complete enumeration confirms the minimum Hamming distance; cancellation and invalid seed assumptions are checked. [hybrid_round24](../test_constraint_hybrid_round24.py). | **Retained bounded exact repair.** Minimum-change claims require completed optimization. Budget exhaustion and unavailable native dependencies are unknown. |
| **37. Connectivity-directed trades** | `loop_trades in hybrid`. Enumerate exact patch trades preserving all outside placements while seeking fewer loop components. | A two-cell trade merges three loops into independently validated Gold. [hybrid_round24](../test_constraint_hybrid_round24.py). | **Retained bounded repair neighbourhood.** Requires an edge-perfect legal input; it does not make a failed patch proof apply to other outside arrangements. |
| **38. Fragment crossover** | `fragment_crossover in hybrid`. Pack compatible matched fragments from an archive, then complete physical inventory and the Gold circuit exactly. | A small crossover uses material from both parents and independently validates its Gold result. [hybrid_round24](../test_constraint_hybrid_round24.py). | **Experimental constructive repair.** Bounded archive/fragment/query sizes apply. This is a proposal heuristic followed by exact validation, not exhaustive global coverage. |
| **39. Repair-context learned exclusions** | `repair_learning; smart_learning for DFS reuse`. Record completed feasibility failures under explicit outside assumptions; send candidate exclusions through independent original-root verification. | A transport discriminator offers one true and one false claim; only the true exclusion is admitted. [hybrid_round24](../test_constraint_hybrid_round24.py). | **Retained scoped proof reuse.** Optimization failure, a capped query or an unverified exported core cannot prune DFS. |

## Sampling neighbourhoods and diversification

| # / proposal | Access and implemented mechanism | Finite evidence | Disposition and scope limit |
| --- | --- | --- | --- |
| **40. Ejection chains** | `ejection_chains in hybrid sampling`. Propose closed three-to-six-cell relocation chains with independent rotations. | All nonempty combinations of the three new sampling controls preserve inventory/fixed cells and independently recalculated scores on CUDA and D3D12 tests. [gpu_sampling_round24](../test_gpu_sampling_round24.py); [benchmark: hybrid_round24](../benchmark_hybrid_round24.py). | **Experimental heuristic; benefit pending.** Heuristic sampling does not prove coverage and can revisit boards. This chain path currently rescans the full score. |
| **41. Move tabu** | `move_tabu in hybrid sampling`. Keep short move-attribute tabu memory, allowing aspiration when a move improves that lane’s best. | The sampling combinations pass independent current/best-board and score checks, including with Gold scoring enabled. [gpu_sampling_round24](../test_gpu_sampling_round24.py); [benchmark: hybrid_round24](../benchmark_hybrid_round24.py). | **Experimental heuristic; benefit pending.** Tabu discourages selected recent moves; it does not guarantee distinct complete boards or establish a faster original solve. |
| **42. Replica exchange** | `replica_exchange in hybrid sampling`. Exchange current boards/scores between fixed-temperature lanes using Metropolis acceptance, keeping lane RNG/iteration and best archives in place. | Equal-energy cases verify actual board exchange without moving temperature-lane state; combined sampling modes pass invariant checks. [gpu_sampling_round24](../test_gpu_sampling_round24.py); [benchmark: hybrid_round24](../benchmark_hybrid_round24.py). | **Experimental heuristic; benefit pending.** This is diversification, not an exact search-space reduction. Compare equal proposals, time and solution quality across seeds. |

## Ownership and proof boundaries

A checkpoint now records its stable `owned_root_domains`. Local nogoods,
SAT/LCG assumptions and mask verifiers bind to that actual subtree, not merely
to the puzzle fingerprint. Legacy checkpoints conservatively adopt their first
frame. A mutable learning cache from a different owned root is rejected.
Splitting a root restricts optional mask records into **unverified child claims**;
parent-bound proof certificates are not relabelled. A child independently checks
claims before using them.

Coverage reuse likewise records only the region actually exhausted. Stopping,
timing out, finding a solution or finishing only one imported subregion cannot
certify exhaustion of the full original puzzle. Domain-box caps preserve the
previous pending union. These boundaries fix regressions where restricted work
could otherwise close an unrelated broader scope.

Checksums detect accidental corruption of local checkpoints. They do not turn
edited native state or a local coverage database into an authenticated proof.
Transported proof claims require the separate verification paths above.

## Evidence and release validation

The numbers in this register are finite discriminator/development results with
their stated baselines. They are not final whole-puzzle benchmarks. In
particular:

- Witness pooling avoids repeated SAT queries on its recorded corpus; batching
  alone regresses that corpus even though it helps a separate discriminator.
- Joint conservation, affine reasoning, PB cuts, interface formulas and
  obstruction templates add no deletions after the prior fixpoint on the
  recorded 60-context comparison. Their isolated capabilities remain available
  experimentally; that is not a claim of demonstrated original-puzzle progress.
- Regional compression and full-factor tests include synthetic boards. A
  synthetic 160-cell test is not the original 160-piece Gold instance.
- The trained ordering has a completed small held-out comparison. Its larger
  capped tier establishes no efficiency ranking.
- Sampling checks establish legal moves and correct scoring. Progress claims
  need separate paired-seed comparisons; heuristic repeats are not exhausted
  exact branches.

The release coordinator adds final frozen-source validation and original-input
benchmark outcomes after those runs finish. No final suite total or best profile
is claimed here before that evidence exists. The public JSON register reserves
`release_validation` for that update.

Reproduction scripts include `benchmark_round24.py` for whole-board profiles,
`benchmark_learning_round24.py` for finite deduction/query counts,
`benchmark_regions_round24.py` for regional discriminators,
`benchmark_learned_ordering_round24.py --evaluate-bundled` for the shipped frozen
ranking model, and `benchmark_hybrid_round24.py` for sampling combinations.
Use separate output files, record inputs and source hashes, and treat every
capped query as unknown. The component documents give supported commands and
backend requirements.

Exhausting this documented candidate list does not establish that no further
algorithmic improvement exists. Future broader variants should be new,
explicitly scoped candidates with discriminating evidence, rather than silently
reopening an already measured result or relabelling a bounded prototype as a
complete general implementation.
