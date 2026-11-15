# Round 35 classic checkpoint source compatibility

## Completed GPU work under a dormant CPU-endgame change

A released v1.15.0 classic CUDA run completed 480 jobs on a 160-piece input with an independently fixed incompatible seam. It tried 230,400 orientations. An independent run in a second source tree completed all 480 jobs with the same per-job results and counters. Only the portable frontier implementation in `constraint_work.py` differed. The released verifier refused the original terminal receipt in that tree. The reviewed dependency check accepted it on the host with **zero GPU launches**. The 20 original runtime directories inspected contained no finished reusable receipt, so the comparison uses an isolated genuine released-format run. [Counters and mutation controls](R35_CLASSIC_DEPENDENCIES.json)

Classic search can use the constraint DFS endgame only when CPU assistance is enabled. With all three CPU-assist switches off, the verified active source set had 32 files in a diagnostic archive of 166. Changing any one of the 32 individually rejected reuse, including either device kernel and the validators. A new file reachable solely from the dormant CPU subtree passed. Missing required files, damaged archived bytes, unknown feature switches and active CPU assistance failed closed. GPU ownership can move between backends on resume, so both backend images participate. Root, rules, objective, terminal state, counters and stored diagnostic files still undergo the independent checks. The actual Round 35 geographic pruning edits also change active source and reject reuse of this released receipt.

## Heuristic checkpoints

The same portable-export change also affected an existing heuristic source manifest. The heuristic uses only the unchanged digest function from that module. Preserve the audited source alias for that specific configuration. Genuine v1.14.0 and v1.15.0 nonempty checkpoints in three line-ending/source profiles resumed to identical next candidates and work. Corrupt and foreign checkpoints still failed. This heuristic alias grants no classic exhaustion authority. [v1.15.0 crossover](R35_V115_CROSSOVER.json)

## Remaining decisions

Test whether recording the backends that actually contributed to future-format checkpoints can safely omit an unused shader, including a mixed-backend resume. Evaluate the independently exercised CPU probe without endgame before extending the reviewed policy. Reverify the approved source digest against the final release tree after all active solver changes land.

A probe-only CPU checkpoint previously accepted a nonempty resumed endgame owner. Such a record could enter the full constraint DFS from the ownership dispatch path. The reader now rejects that policy/ownership mismatch before altering live state. Recheck probe-only scope with a real probe traversal and archived source controls before accepting any additional completion reuse. Existing historical proof scopes retain the stricter policy.
