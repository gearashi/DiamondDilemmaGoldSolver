# Gold search, routes to 208 and native checkpoint results

The best independently checked original board remains at **207 of 240 matched edges**. A Gold solution must use all 160 physical tiles exactly once and connect all 240 seams into one loop. A 208-edge neighborhood and the unrestricted Gold search have different completion conditions.

| Workload | Measured change |
| --- | --- |
| Reference-board bound | Any Gold board must lose **at least 21 of the 207 named reference bonds**. The signed-inventory model with thirteen physical departure groups admits an explicit 21-loss phase. Its cap-20 CNF has a checked refutation. The ten-disjoint-group predecessor has optimum 20. [Independent clause and proof audit](R35_ATTAINMENT_CLAUSES.json) |
| Unrestricted Gold | The native compulsory-mate check eliminated all candidate attempts on 32 impossible original prefixes. The predecessor tried 919 and 922 candidates in the two sets. All 16 allowed controls remained active. CUDA, Vulkan and Direct3D returned the same decisions. [Native mate tests](ROUND35_NATIVE_MATES.md) |
| Routes toward 208 | Completing the eight support ranks on an original seven-piece residual closed 140 corresponding roots with zero cursor steps. The direct comparison took 308. On an eight-piece four-pair residual, one completed rank eliminated 4,543 cursor steps. These are complete results for the specified pools. [Coverage and ownership](ROUND35_LIVE_ROOTS.md) |
| Rotation choices | On a complete one-swap neighborhood of the original 207 board, cached connected components reduced numeric factor keys from 14,646 to 784, plus 363 component keys. All 88 optimal rotation witnesses matched the independent baseline. The best was still 207. [Rotation tests](ROUND35_ROTATION_COMPONENTS.md) |
| Long GPU sweeps | Direct3D 12 completed the 131,072-lane dynamic control in bounded pages. On CUDA the public solver wrote two checkpoints inside its first sweep, updated the live board and finished a portable save 1.282 seconds after Stop. [Hardware and recovery controls](ROUND35_GPU_PAGES.md) |
| Repeated loss queries | Existing checked proof and phase witnesses answered three command-line queries with zero new SAT decisions. The independent native path remains available. [Public API controls](ROUND35_INTEGER_PUBLIC.md) |
| Nearby Gold placements | In twelve surviving original near-reference cases, a weighted tile-to-cell lower bound removed 444 or 453 more placements per case. A second complete pass removed none. The unrestricted original root received no extra exclusion. [Geographic cuts](ROUND35_ALTERNATIVE_RETENTION.md) |

## Settings for an i7-4790 and RTX 2060 6 GB

**To continue an existing classic GPU search**, select **GPU DFS · existing saved branches**, keep its checkpoint directory and choose **Unlimited**. Its saved cursors determine which work remains. A larger replica count changes how many branches can advance together. The 131,072-replica measurements exercise maximum-width checkpoint and driver behavior. [Checkpoint measurements](ROUND35_CLASSIC_STORAGE.md)

**To start an unrestricted Gold search**, use full 64-bit Python 3.12, **CUDA domains + learned backjumping**, **128 lanes**, **scalar propagation** and **Unlimited**. Enable global GPU piece matching, GPU seam-support reuse, watched clauses, verified sharing and transport, and sharing completed exact regions. Leave cooperative propagation and smallest-remaining-domain splitting off for the unrestricted original. The inventory check closed 47 of the 128 original root boxes before GPU allocation. Those boxes have different sizes. [Original hardware comparisons](ROUND34_RESULTS.md)

**To exhaust seven-piece routes to 208**, use the CPU ranked search, family 2,2,3 and 1,024-support pages. Reusing the same database resumes already-ranked work. The eight-piece family uses 2,2,2,2. [Ranked support controls](ROUND35_INCUMBENT_ROUTES.md)

```powershell
.\venv\Scripts\python.exe r30_multicenter.py --family 2,2,3 --backend cpu --page 1024 --target 208 --seconds 0 --database runtime/four-swap-union.sqlite
```

For a route through equal or lower-scoring neighbors, use the serial plateau runner with `--incremental-floors`, `--ties 1` for the edge-score objective, and a persistent SQLite file. The documented floor-205 one-swap run found 443 qualifying neighbors and closed that specific scope without reaching 208. [Resumable route commands and their limits](ROUND35_PLATEAU.md)

The [candidate register](round35-candidates.json) records the scope, controls and decisions for the full round.
