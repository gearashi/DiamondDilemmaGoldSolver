# Shared exact coverage decisions

## Per-scope invalidation of symbolic coverage (66)

**Reject for the current solver’s normal workload.** The production publisher in `constraint_work.CoverageLedger.record_completed` inserts new records with `INSERT OR IGNORE`. A source search for updates, deletions, replacement inserts and table drops on `closed` found them only in tests of deliberate external modification. The existing scoped sequence already handles concurrent inserts. Its global epoch advances only on a destructive change.

In two independent SQLite connections, a 10-cell binary box had 1,024 assignments. Eight assumed exclusions under Gold left 1,016 assignments, verified by explicit enumeration. An edges-only writer inserted 128 distinct boxes, tried all 128 again, and reopened the Gold diagram every 16 new records. The epoch stayed at zero, the original diagram was built once and reused eight times, and every remainder matched. An intentional out-of-band deletion of one edges record advanced the epoch once and triggered one Gold rebuild with the same 1,016 answers. That control is an opportunity only for a future normal editing/compaction workflow. There are zero ordinary rebuilds here for a per-scope epoch to remove. Keep the existing deletion and corruption regression tests.
