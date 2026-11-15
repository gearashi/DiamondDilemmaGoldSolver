# Connected physical pairings at fixed signed phases

The published `r32_gold.py phase-sewing` command now rechecks five retained pairings across three phases, including their global color complements. It validates every physical mate, all compulsory bonds, all applicable groups, the named loss, and the single 365-strand loop before using a witness. Native search still checks all lower, unproved costs. Use `--independent-native` to reproduce the comparison without published hints. A loss of 35 means 172 of the 207 named reference bonds survived. This statistic concerns specific original bonds and does not predict the score of a geographic board.

| Signed phase | Signed loss | Full physical pairing loss | One-loop loss | With 19 path circuits, signed / physical / one-loop |
| --- | ---: | ---: | ---: | --- |
| First attained 33-loss phase, ending in `02465766` | 33 | 34 | 35 | 35 / 37 / 37 |
| Second attained 33-loss phase, ending in `1207cf06` | 33 | 38 | 39 | 33 / 38 / 39 |
| Earlier donor, ending in `021e5126` | 37 | 37 | 37 | 38 / 40 / 40 |

The common scope uses 13 reference groups, 194 ordinary geographic exclusions, and 985 edge-perfect contact exclusions. The raw 31-loss adverse phase fails physical mating both before and after adding the 19 path circuits. Each attained connected pairing in the table has positive ribbon genus and cannot itself embed on the puzzle. A separate complete board proof excludes the entire earlier donor phase. [Raw counts, physical pairing controls and scope](R35_MIP_DONOR_SEWINGS.json) [Complete named pairings](R35_MIXED_MIP_DONOR.json)

One formerly optimal connected pairing of the first donor retained a 29-bond arrangement with zero placements on the original mesh. Each of the 29 possible single-bond deletions admits a placement. The verified core includes three compulsory mates; those are removed only when projecting onto the optional loss variables. A different connected pairing still attains 35 under the 29-bond exclusion. With 13 further earlier residual conflicts and the 19 path circuits, the first donor's signed, full-pairing, and single-loop optima are 35, 37 and 38. Removing the 29-bond exclusion from this last test leaves the single-loop optimum at 38. [Geographic control and matched native decisions](R35_MIP_DONOR_SEWINGS.json)

For the first donor at a limit of 36 under the original 194+985 restrictions, a revalidated witness reduced 5,142 native decisions in three calls to 3,930 in two calls. Adding old short loop exclusions to that hint took 5,679 decisions. In the fully combined case, witness reuse reduced 19,180 decisions to 15,020. Adding the 19 path circuits for the earlier donor took 890,544 native decisions cold and 803,375 with its witness. For the second donor, one retained witness serves both groups of tests and cuts the decision counts from 5,680 to 4,445 and 3,762 to 3,614. The corpus supplies hints only to the original and complementary phase values; the solver verifies each candidate against the current data.

Run the following command with prepared project tiles and the optional SAT dependency. Repeat with `--independent-native` to reproduce the fresh search. All five claim files together also test the separate 29-bond exclusion and the previously published extra residuals.

```text
python r32_gold.py phase-sewing --tiles data/tiles.json --phase 0xcc57eae4e99109f8b073557e8c590b9202465766 --upper 40 --seconds 180 --claims docs/R35_GEOGRAPHIC_RESIDUALS.json --claims docs/R35_EDGE_CONTACTS.json --claims docs/R35_CONTACT_RESIDUALS.json --claims docs/R35_PATH_CIRCUITS.json --claims docs/R35_MIP_DONOR_SEWINGS.json
```
