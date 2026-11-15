"""Compare conditioned Gold queries with and without the published partial-phase proofs."""
import argparse
from pathlib import Path
from unittest.mock import patch

from io_utils import atomic_json
from r28_exchange import read_json
from r30_gold import search
from r32_gold_bonds import inventory
from r35_anchor_rules import check_rules
from r35_phase_cube_rules import certified_rules
from search_problem import SearchProblem


def measure(problem, seconds, native):
    rules = certified_rules(problem)
    required = inventory(problem).required
    rows = []
    for rule in rules:
        for complement in (0, 1):
            pins = {tile:bit ^ complement for tile,bit in rule}
            options = dict(geography=True, pins=pins, seconds=seconds)
            with patch("r35_phase_cube_rules.published_rules", return_value=()):
                prior = check_rules(problem,pins,float("inf"),project=True,required=required)[1]
                old = search(problem,**options) if native else None
            new = search(problem,**options)
            if prior["terminal"] or new["status"] != "INFEASIBLE" or new["cp_calls"]:
                raise AssertionError("Published partial-phase proofs must add a geographic decision.")
            # Find an actual neighboring partial query that still proceeds to search.
            for tile in pins:
                weaker = {t:v for t,v in pins.items() if t!=tile}
                missing = check_rules(problem,weaker,float("inf"),project=True,required=required)[1]
                if not missing["terminal"]:
                    break
            else:
                raise AssertionError("All immediate neighboring queries were already excluded.")
            entry = dict(pins=len(rule),complement=complement,withheld=tile,
                         weaker_remains_open=True,previous_terminal=prior["terminal"],
                         current={k:new[k] for k in ("status","branches","cp_calls","phase_rules")})
            if old:
                entry["previous"]={k:old[k] for k in ("status","branches","cp_calls","phase_rules")}
            rows.append(entry)
    return dict(fingerprint=problem.fingerprint,per_solve_seconds=seconds,queries=rows)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,required=True)
    cli.add_argument("--seconds",type=float,default=5)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--previous-native",action="store_true")
    args=cli.parse_args()
    atomic_json(args.output,measure(SearchProblem(read_json(args.tiles)),args.seconds,args.previous_native))
