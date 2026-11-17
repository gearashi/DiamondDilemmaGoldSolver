# Gold loss search reuse in v1.14.0

The experiment uses the original 160 tiles and the 207-seam reference board. The loss search asks how many of those particular matched physical bonds a one-loop Gold board must give up. Its first open budget is **17**. Search effort in a loss layer measures progress on this necessary condition. The highest verified board score is still **207/240**.

## Candidate register

| Candidate | Disposition and measured work |
|---|---|
| Keep the suffix index across checkpoint pages | **Enabled.** On the entire five-error layer at budget 15, the released program built 235,620 suffix rows for 33 pages. A single table built **7,140**. All 266,916 prefix ranks, 3,237,403 eligible tails and 1,046 exact joint checks agreed, including the individual page outcomes. Across the complete four-error layer at budget 14 with 256-prefix pages, construction fell from 192,780 rows to 7,140. |
| Skip suffix tuples belonging to earlier prefixes | **Enabled.** Each suffix bucket is sorted. The new traversal skips its ineligible initial slice while preserving the order of every remaining candidate. Twelve original-puzzle intervals across budgets 14–17 avoided **7,786,615** individual visits. All four measured configurations retained 1,209,353 eligible tails and 181 joint checks. |
| Reuse the known group cost of each suffix bucket | **Enabled.** The bucket identifies the exact projected cost of the group used to index it. Charging that cost once per nonempty bucket dropped individual group lookups from **5,528,986 to 4,603,780** in the same twelve intervals. All other groups still contribute to the bound, and every survivor receives the joint check. |
| Read saved interval runs and defer construction until an open gap | **Enabled.** Reopening an existing v1.13 checkpoint covering **7,947,775** prefix ranks needed **15 database reads, zero writer transactions and zero group projections**. The prior runner used 134 reads, 67 writer transactions and ten group projections. After splitting saved ranges into 1,090 rows, the old runner needed 2,180 reads. The new one still issued 15, scanning the stored rows in order. Reopening a 31-rank gap exercised the write path and produced the same outcome. |
| Change checkpoint page size after removing repeated construction | **Keep 8,192 for sustained loss search.** Pages of 256, 1,024 and 8,192 used the same 7,140 suffix rows and 1,046 joint checks. They made 1,043, 261 and 33 commits respectively. A page of 1,024 offers more frequent durable progress if hard interruptions are common. |

**Budget 17 produced its first projected candidate.** Both v1.13 and the current solver returned the same four outside errors at prefix rank 874. Integer signed inventory ruled out that pattern even with all tile phases free and the loss cap removed. A subset of 73 equations already contradicts the inventory. Those equations fix 70 agreements and three disagreements and exclude 47 four-error patterns. Each single-equation deletion has a checked phase assignment. This deduction concerns the specified outside-error patterns. Other suffixes at rank 874 and the overall budget of 17 still need evaluation. [Version-to-version witness and integer checks](R33_HIT_CHECK.json) · [73-equation exclusion and deletion witnesses](R33_HIT_CORE.json)

**Keep integer refinement as an option for the loss search.** With full Python 3.12, `--refine-integer` checks each surviving phase-error pattern using signed integer inventories and a reference-loss budget. A proved contradiction advances to the next suffix in the same prefix. A timeout keeps the prefix open. The stronger mode inherits negative pages from plain loss search, consults its own witness store and can use the same SQLite file. A saved hit in the plain model continues to replay there. Integer feasibility still requires a physical and geographic Gold check. The first three-minute refined run rejected all five phase candidates it encountered, finished the four-error layer, and reached prefix **33,361 of 266,916** in the five-error layer. That rank is the first unresolved one in the saved checkpoint. An independent full-geography run at exactly 17 losses was unresolved after 90 seconds. [Original-puzzle checkpoint and results](R33_ORIGINAL_PROGRESS.json)

Two paired end-to-end runs on this workstation took **13.31 and 13.69 seconds** with the new defaults versus **17.36 and 17.19 seconds** with v1.13 on the same complete five-error layer at budget 15. This comparison includes program startup and SQLite transactions on an in-memory database.

The table is private to a fixed query and keyed by loss budget and outside-error count. It retains one layer, and releases that layer before allocating a replacement. The fixed-layer memory sample retained about **0.75 MiB**. A fresh process rebuilds the table for its first unfinished page. Old SQLite checkpoints use the same puzzle, reference, model, budget and rank conventions as this release.

The lookup tests enumerate all tile phases on small boards and verify both successful witnesses and rejected intervals, including empty groups and zero errors. Interrupted runs, page-size changes, read-only replay under a concurrent writer and a gap claimed by a different worker are covered. The original-puzzle experiments measure work on the specified layers. A phase witness from this search still needs validation as a full physical Gold placement.

[Three-way cache control, memory and page sizes](R33_LOOKUP_RESULTS.json) · [Four traversal configurations](R33_BUCKET_RESULTS.json) · [Checkpoint replay](R33_RESUME_RESULTS.json) · [Complete command-line comparisons against v1.13](R33_PIPELINE_RESULTS.json)

## Recommended settings

For **unrestricted Gold** on the i7-4790 and RTX 2060 6 GB, continue the dashboard's **CUDA domains + learned backjumping (experimental)** profile with **128 lanes, scalar propagation and Unlimited**. Use the measured global matching, verified conflict sharing and seam-support reuse options. This round benchmarked the separate CPU loss search. [Exact GPU settings](ROUND32_RESULTS.md)

For **the loss bound**, use integer refinement to continue past the saved rank-874 phase witness. Use full Python 3.12, CPU, checkpoint pages of **8,192** and an unlimited timer. The active budget explores whether a Gold board could break exactly 17 reference bonds. Run from the project folder.

```powershell
.\venv\Scripts\python.exe r32_gold.py loss --refine-integer --upper 17 --page 8192 --seconds 0 --database runtime/r32-loss.sqlite
```

For **208 near the known boards**, continue the CPU seven-piece family with 1,024-position pages in the existing four-swap database. For **fresh independent boards**, keep the eight-bond CPU constructor at a cycle budget of three and 64 rounds per seed. [Commands and measured hardware profiles](ROUND32_RESULTS.md)

## Reproduction

Prepare the original pieces using the project's setup. For a version-to-version comparison, keep a checkout of v1.13.0 in a separate folder. Replace the baseline paths below with that checkout. The checkpoint benchmark can use an existing, fully completed v1.13 loss database via `--checkpoint` and `--upper`. It takes a read-only copy before varying page layouts. Its partial-gap comparison uses the same evaluator on both sides to isolate the database changes.

```shell
python -B -m unittest discover -v -p "test_r33*.py"
python -m benchmarks.bench_r33_gold_lookup --released-syndrome ../v1.13.0/r32_gold_syndrome.py --output lookup.json
python -m benchmarks.bench_r33_gold_buckets --output buckets.json
python bench_r33_gold_resume.py --legacy-source ../v1.13.0/r32_gold.py --output resume.json
python bench_r33_gold_pipeline.py --baseline ../v1.13.0 --output pipeline.json
python bench_r33_gold_hit.py --legacy-source ../v1.13.0/r32_gold_syndrome.py --output hit.json
python bench_r33_gold_core.py --output core.json
```

## Open experiments for the next round

1. Test sharing verified integer contradictions across candidate patterns. Begin with the checked 73-equation core and compare per-pattern solver calls, the number of actual incoming hits it covers, witness preservation and the storage cost. Implement a persistent exclusion only if measurements justify it.
2. Test a second projected group in the suffix lookup on tight budgets. Count candidate tails eliminated before per-tail checks, price table growth across both tight and loose budgets, and compare every surviving joint query and witness.
3. Test compiling the outside phase equations once for the exact joint checker. Each error vector changes the right-hand side. Compare exact decisions against fresh elimination, especially on dependent equations, inconsistent assignments and timeout/restart boundaries.
4. Profile the remaining repeated projection lookups for tuples shared by neighboring prefixes. Compare caching selected tail contributions or incrementally updating prefix contributions against the retained bucket methods. Measure actual work and bounded memory on the first open budget before enabling either.

These candidates are open and carry no measured benefit yet. The retained changes and the page-size comparison are closed for their documented configurations.
