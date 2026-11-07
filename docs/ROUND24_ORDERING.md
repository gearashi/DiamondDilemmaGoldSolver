# Round 24 learned ordering experiment

Proposal31 is retained as an optional fitted value-ranking policy. It never
removes a value, changes a branch partition or provides a proof. The default
remains off. An unfitted model now raises clearly instead of silently scoring
all values zero.

One ridge regression (ridge1) was fitted to46 completed exact child searches
on six independently seeded ten-cell synthetic Gold fixtures. Forty-four labels
ended in validated solutions and two in completed infeasibility. Timeout labels
were excluded (none occurred). Features are inventory availability, local seam
support and domain size; the target is log1p(completed DFS nodes). The frozen
coefficients, training seeds and exact label-data hash live in
`constraint_branching_round24.py`; the app uses those coefficients explicitly.

The original disjoint twelve-seed ten-cell test tier found validated Gold
solutions in every comparison: baseline133 versus learned119 DFS nodes,
seven wins, two ties and three losses. This finite result supports exposing the
option, not selecting it as the default. Existing adaptive propagation uses
time budgets, so repeated node counts can vary slightly. Six additional,
previously unseen forty-cell fixtures hit the1.5-second query cap in both
configurations. Those outcomes are inconclusive; lower node counts at timeout
are not credited as better pruning or progress. No original160-cell performance
claim is made, and the model was not retuned using either test tier.

Reproduce a new fitting experiment with:

```
python -B benchmark_learned_ordering_round24.py --output learned-fit.json
```

Evaluate the exact weights shipped by the app, without training a different
model, with:

```
python -B benchmark_learned_ordering_round24.py --evaluate-bundled --output learned-bundled.json
```

The script records source hashes, completed/censored status, every label,
separate training/test seeds and independently checks every reported solution.
For a longer study increase `--seconds`; capped tests remain unknown. Record
new evidence against the unchanged frozen model before changing any default.
