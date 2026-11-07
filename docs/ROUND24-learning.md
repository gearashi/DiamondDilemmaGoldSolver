# Round 24 learning and proof experiments

This record separates implemented mechanisms from finite correctness evidence and demonstrated search-work gains. All features are opt-in. None of these computations solves the original Gold puzzle or predicts its total solve time.

## Implemented scope

| Proposal | Implementation | Evidence and disposition boundary |
| --- | --- | --- |
| 7 | `SatAssist(witness_pool=True)` retains up to 128 complete semantic SAT models, revalidating every placement against new domains and every posted Gold exclusion before reuse. | Exact supported domains agree across seeded contexts. The 60-context comparison reduced native calls from 874 to 72. This establishes cross-context query reuse on that corpus, not original-puzzle solve acceleration. |
| 11 | `DeepPruner(retain_relations=True)` keeps scoped binary relations produced by path consistency and propagates them after later domain changes. | Independent edge-solution enumeration verifies narrowing and widening scopes. An equality relation gives no initial unary deletion but forces the partner after a later choice. The implementation retains path-consistency output; arbitrary constructive-lookahead relation extraction is not claimed. |
| 12 | `SatAssist(support_cover=True)` queries a guarded disjunction of all uncovered ambiguous placements and extracts supports from returned models. Selectors are disabled on completion and interruption. | A restricted-domain fixture reaches the same supported values in 11 calls versus 20. In the broader 60-context corpus after legacy propagation it needs 965 versus 874 calls; with the witness pool both policies need 72. Keep experimental and workload-dependent, not a universal improvement. |
| 13 | `DeepPruner.joint_conservation` enforces simultaneous seam-mask-orbit capacities through exact matching and alternating-cycle support filtering. | A six-seam shortage passes every individual class bound but fails the joint model. 400 small capacity instances match independent assignments. No additional deletion beyond the complete prior fixpoint was found in the 60-context puzzle corpus. |
| 14 | `AffinePruner` combines exact affine hulls of allowed individual port vectors in global seam coordinates and checks candidate consistency over GF(2). | All 256 subsets of three-bit vectors match independent affine closure; 80 puzzle contexts preserve every independently enumerated edge solution. A mod-3 reference helper finds an invariant absent from a GF(2) example. Runtime integration uses GF(2); general modular runtime propagation is research-only. No added pruning after the prior fixpoint in the comparison corpus. |
| 15 | `PBLearner` creates exact placement-capacity/seam rows, derives positive integer combinations with gcd rounding, checks their derivation, and performs domain-aware bound propagation. | A nonunit capacity contradiction survives primitive row filtering and is closed by derived cuts. 80 randomized signed-integer systems preserve the complete model set. Added original-puzzle/full-search benefit is unestablished. |
| 16 | Optional native SAT callbacks reconstruct partial placement domains and emit explained Hall-conflict clauses and bounded affine consequences. | A partial Hall shortage produces a clause satisfied by every independently enumerated complete edge solution. This broadens the former proper-loop-only callback; it does not implement every CPU propagator inside SAT. |
| 25 | `InterfaceProjector` performs bounded Davis-Putnam elimination on a local optimistic CNF and independently replays every resolution derivation before retaining its interface clauses. Exact subsumption controls redundant clause growth. | 100 finite CNFs have the same existential boundary models before/after elimination. Puzzle formulas preserve every enumerated completion under narrower contexts and are disabled on widening. Caps publish no formula. This is bounded proof-derived projection, not arbitrary backend Craig-interpolant extraction. |
| 32 | `ObstructionMiner` discovers mixed seam/inventory incompatibility cliques and odd cycles, verifies every required incompatibility, and retains their exact at-most bounds. | Every generated rule is rechecked; 45 puzzle contexts preserve all enumerated edge solutions. Rules dominated by primitive nonnegative capacity rows are excluded from retained-rule counts. This is verified discovery within these template families, not unrestricted automatic theorem synthesis. |

## Proof transport follow-up (proposal 5)

`MaskProofContext` reconstructs the deterministic base CNF with a pure capture sink; certificate checking requires no PySAT solver. A bounded deterministic DPLL producer emits branch clauses which a separate RUP checker verifies. Scope includes exact puzzle, original domains, effective root, objective, base-CNF digest, and assumption masks. The current proof theory is an edge relaxation: its UNSAT results also establish Gold UNSAT, but Gold-only contradictions that require more theory retain ordinary verification.

The producer caps decisions, depth, clause visits, elapsed budget, and output size. Incomplete work returns unknown. `VerifiedMaskConflictStore(proof_transport=True)` emits bounded version-2 snapshots and accepts legacy version 1; transported bytes remain untrusted until checked. A successful certificate import avoids native SAT re-solving. When a certificate is unusable, its ordinary claim remains subject to the previous independent verification path. A missing native dependency can use the bounded pure producer.

Windows native proof-logging experiments were reported by the release coordinator to terminate with status 3221226505 (`STATUS_STACK_BUFFER_OVERRUN`) despite printing successful tests. Native proof logging is therefore not used. The replacement has an explicit child-process regression requiring exit code zero after a nontrivial UNSAT proof.

## Finite work-count comparison

Run:

```powershell
& C:\DiamondDilemmaGoldSolver\github-release\.venv-compare\Scripts\python.exe -B benchmark_learning_round24.py --output audit/optimization-round24/learning-finite-02.json --cases 60
```

The report records seed 240071, source hashes, Python version, exact supported-domain digests, and logical counters. Its baseline reaches a fixpoint of the default seam/Hall/cluster/neighborhood/parity machinery plus existing path consistency and scalar conservation. It does not claim to exhaust every optional Gold oracle on every context. All SAT variants then seek the same edge-model supports; these witnesses are not accepted as Gold solutions.

| SAT configuration | Native candidate calls | Change versus baseline |
| --- | ---: | ---: |
| Existing scans and exact-context reuse | 874 | Baseline |
| Cross-context witness pool | 72 | 91.8% fewer |
| Support-cover queries | 965 | 10.4% more |
| Witness pool plus support cover | 72 | 91.8% fewer |

Joint conservation, affine filtering, PB cuts, obstruction templates, and interface formulas made **zero additional deletions after the prior fixpoint in this corpus**. Their isolated discriminators establish capability and soundness, not additional pruning on these already-closed contexts. Whole-search decisions, internal work, and deeper original-data contexts need separate comparisons before making a performance claim or enabling these defaults.

Correctness commands include `test_constraint_sat_round24`, `test_constraint_learning_round24`, `test_constraint_proof`, existing SAT/domain/deep/model suites, and legacy transport tests. The focused 98-test existing/new learning regression completed with one optional skip and process exit zero; subsequent proof-transport regression completed 16 tests with exit zero. Final frozen-source release tests supersede these development checks.


## Independent checkpoint and session audit

The follow-up audit found and fixed persistent-SAT binding gaps: warm reuse now compares the incoming problem's full domain scope; constructors reject scopes from other original domains even if the data fingerprint matches; edge-target restores reject Gold-only cuts; and a warm session can accept a validated portable replacement without destroying the old session on invalid input. A completed native UNSAT answer is committed even when cancellation arrives concurrently, while unknown answers remain unfinished. Dashboard heartbeat events now carry the actual checkpoint under the field consumed by the runner.

Portable frontier overlap validation now uses exact domain-mask posting lists of earlier cube IDs, avoiding the previous pairwise scan for structured prefix frontiers. Independent randomized pairwise comparison and a 4,096-cube, 160-axis fixture check the result; cancellation propagates instead of accepting an unfinished check.

CUDA native restore was independently reviewed for scoped backjumping and remaining-cube export. Structural validation now rejects cyclic disjoint-set parents, invalid rollback indices, invalid assignment logs, inconsistent domain trails, invalid frame-domain bits, and invalid pending/delivered solutions before device execution. These are checks of trusted local checkpoint consistency, not portable mathematical certificates for arbitrary edited state. Hardware checkpoint regressions are rerun by the GPU owner.

Development validation: 23 session/work/portable/public-runner tests passed on Python 3.12 with process exit zero. Python 3.15 ran 55 learning/proof/session/checkpoint tests with 13 optional native-SAT skips and process exit zero. The pure proof producer/checker and scope/checkpoint validation ran without PySAT. CI includes all new CPU suites and the new sampling variants under software Vulkan.
