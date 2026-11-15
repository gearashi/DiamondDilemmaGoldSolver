# Physical blank-side projection for one-loop Gold

Use `--project-blanks` with `r30_gold.py matching` to return a fully materialized connected physical sewing, or with `embed` to require a legal geographic board. These commands use the full Python environment with OR-Tools. The input JSON accepts the same domains, phase pins and required or forbidden mates as the existing search. With projection selected, the embed command generates its geographic witness from the nonblank mates. An explicitly requested coupled geographic channel, fractional pricing model, physical bridge model or reference-loss model uses full side variables and reports that fallback.

```text
python r30_gold.py embed --project-blanks --input restrictions.json --seconds 60
python bench_r35_blank_projection.py --tiles data/tiles.json --sewing docs/R35_NONBLANK_SEWING.json --root-rounds 4
python -m unittest -v test_r35_blank_projection
```

## What one projected search covers

Every accepted tile has at least one strand. A single tour over all physical strands therefore connects every tile via nonblank joins. Each tile reached from a known placement has a unique pose across that join. Trying every permitted starting face and rotation covers every possible geographic placement of a fixed connected nonblank sewing. The original has at most 480 such starts. A singleton geographic domain fixes one starting pose. Every surviving placement supplies all its blank joins from actual adjacent cells. Faithful materialization checks phase, restricted mates, piece uniqueness and the independent board validator. Unrestricted physical-only search instead finds a complete perfect matching within the signed blank classes. A stopped matching or root-pose check returns UNKNOWN. Multiple nonblank strand components continue to use the established physical cuts. Partial user-supplied bonds leave the other nonblank choices in the native model.

For the original input the number of physical partner Booleans falls from 3,036 to 2,002. On four unrestricted native calls the two versions took 195,404 and 123,369 branches respectively. Both remained open with different subtour cuts. In an exploratory 90-second projected run, 32 calls produced no connected candidate. These outcomes leave time to Gold unresolved.

## Exact original-piece comparison

The [fixed nonblank sewing](R35_NONBLANK_SEWING.json) contains 217 pairs. Independently walking the original segment endpoints through those pairs gave one 365-strand circuit. For its chosen phase vector the blank darts separate into sets of 28 and 18. The first allows every pair except (268,269), two sides of the same tile. The second allows every pair. Thus there are exactly (27!! âˆ’ 25!!) Ã— 17!! = **7,083,210,381,585,744,656,250** full signed blank matchings on this one circuit. Every completion has the same nonblank connections. None embeds on the geographic mesh. All 480 roots failed. The existing ribbon-genus test independently gives genus 27 and the existing pose packer finds zero legal poses from these nonblank bonds alone.

| Same physical input, native model | Calls | CP branches | Result |
| --- | ---: | ---: | --- |
| Full sides, final embedding | 6 | 25,212 | Open after six different failed embeddings |
| Full sides and learned geographic exclusions | 6 | 44,971 | Infeasible |
| Released coupled geographic channel | 1 | 34,592 | Infeasible |
| Projected sides, with or without learned exclusions | 1 | 0 | Infeasible after one nonblank lift |

This count describes signed choices on one abstract sewing. A caller with the nonblank fixture in advance can also directly run the existing genus test.

## Positive, negative and adverse controls

Four planted spherical controls used seeds 35, 71, 103 and 157. For each control an independent enumerator explicitly instantiated all 6,584 signed blank completions, then called the existing full-mate geographic embedder and separately scored any witness. Exactly one completion embedded for each positive, and zero for each rotated-root sibling. The projected and released geographic-channel models both decided all eight cases in one native call and zero branches each. The uncoupled full-side models with and without learned geographic exclusions remained open at eight calls. Projection returns actual blank pairs for a physical-only witness. A different signed blank matching can remain possible on a geographic negative.

The regression suite also checks random required/forbidden/signed matching queries against complete enumeration, signed odd-component deficiency when every dart has a candidate, incompatible forced pairs, a forbidden geographic blank mate, an interrupted root scan, partial pins, a disconnected nonblank scope with four abstract possibilities, a partial materialization passed to the lift helper, and full-side channel fallback. The public input validator rejects a strand-free tile. The old and new physical, formal geometry, fresh and solver controls passed all 57 tests in the recorded focused run. [Measured comparisons](R35_BLANK_PROJECTION.json)

**Disposition.** Retain blank projection as an explicit physical matching or broad nonblank-assembly option. For narrow geographic placement domains the coupled channel already solved all eight controls immediately and remains the default. The finite sewing, exhaustive blank count, adverse channel and input contract close this comparison. A projected physical-only witness or an incomplete unrestricted run does not identify a Gold solution.
