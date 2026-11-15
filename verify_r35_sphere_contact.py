"""Check raw sphere, reference and contact premises in an exact signed-loss proof."""
import argparse
import hashlib
import json
from math import gcd
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_loss import digest
from r35_geographic_replay import geographic_departures
from r35_induced_contacts import contact_departures
from r35_loss_witness import physical_price
from verify_r35_contact_vipr import replay_header
from verify_r35_contact_wide import read_rows
from verify_r35_loss_euler import reconstruct


DIRECTORY = Path(__file__).resolve().parent / 'docs'
SOURCE_GROUPS = '55784bc53e6e05c83aaab75bd010a686122ef279996d8ec1e77928bfda0e6c1e'
SPHERE = '6e5b7f0802b0c68c98fb17eb8f7d910f1c4e69ee71051025eb80140926e48756'


def sphere_axioms(problem):
    weights, allowed, _ = reconstruct(problem)
    if digest([weights, allowed]) != SPHERE:
        raise ValueError('The signed disk weights or spherical mesh corrections differ.')
    base = sum(dark for dark, light in weights)
    step = gcd(*(light - dark for dark, light in weights)) or 1
    permitted = [(v - base) // step for v in allowed if (v - base) % step == 0]
    if not permitted or permitted != list(range(permitted[0], permitted[-1] + 1)):
        raise ValueError('This certificate requires contiguous normalized Euler weights.')
    terms = sorted((f'tile_{i}', (light - dark) // step) for i, (dark, light) in enumerate(weights)
                   if light != dark)
    return [('G', permitted[0], tuple(terms)),
            ('G', -permitted[-1], tuple((key,-coef) for key,coef in terms))], weights, allowed


def replay(tiles, receipt, *, proof=None, output_lp=None):
    problem, reference, initial, required = problem_scope(tiles)
    record = json.loads(Path(receipt).read_bytes())
    if record['puzzle'] != problem.fingerprint or record['sphere_sha256'] != SPHERE:
        raise ValueError('The proof identifies a different signed puzzle or sphere.')
    geo = read_rows(DIRECTORY/'R35_GEOGRAPHIC_RESIDUALS.json', 'geographic_cores')
    contacts = read_rows(DIRECTORY/'R35_EDGE_CONTACTS.json', 'edge_perfect_cores')
    if set(geographic_departures(problem, reference, required, geo)) != {g-required for g in geo}:
        raise ValueError('A claimed geographic departure has a possible pose.')
    if set(contact_departures(problem, reference, required, contacts)) != {g-required for g in contacts}:
        raise ValueError('An induced-contact departure failed its geometric replay.')
    groups = [*initial[:13], *geo, *contacts]
    projected = {frozenset(g)-required for g in groups}
    if digest(sorted(sorted(g) for g in projected)) != SOURCE_GROUPS or record['group_sha256'] != SOURCE_GROUPS:
        raise ValueError('The conditional proof refers to different physical charges.')
    extra, weights, allowed = sphere_axioms(problem)
    cap = record['excluded_cap']
    donors = []
    for item in record['witnesses']:
        phase = int(item['phase'],16)
        lost = frozenset(map(tuple,item['lost']))
        raw = physical_price(problem.tiles,reference,groups,required,phase)
        total = sum(row[(phase>>i)&1] for i,row in enumerate(weights))
        if (raw is None or raw[0] != item['cost'] or len(lost) != raw[0] or
            not raw[1] & reference <= lost or not lost <= reference-required or
            not all(g & lost for g in groups) or total != item['euler'] or total not in allowed):
            raise ValueError('The listed signed upper witness or its loss set failed.')
        donors.append((item['phase'],item['cost'],total))
    if (not donors or record['certified_lower_bound'] != cap+1 or
        record['attained_upper_bound'] != min(cost for _,cost,_ in donors) or cap >= record['attained_upper_bound']):
        raise ValueError('The claimed upper or lower reference-loss bound differs.')
    cycles = read_rows(DIRECTORY/'R35_CONTACT_SIGNED_CYCLES.json','required_xor')
    result = dict(projected_groups=len(projected), donors=donors, sphere_totals=allowed, excluded_cap=cap)
    if proof is not None:
        with Path(proof).open('rb') as stream:
            checksum = hashlib.file_digest(stream,'sha256').hexdigest()
        if checksum != record['proof_files']['proof.vipr']['sha256']:
            raise ValueError('The rational proof differs from the externally checked file.')
        result.update(replay_header(Path(proof),problem,reference,required,groups,cycles,cap,
                                    additional_axioms=extra))
        result['proof_sha256'] = checksum
    if output_lp is not None:
        from r31_gold_connectivity import PhaseInterface
        from r35_contact_mip import build_model
        from r35_mip_conditions import euler_row
        engine, phases, _ = build_model(problem,reference,groups,required,cap,cycles)
        euler_row(engine,phases,PhaseInterface(problem))
        Path(output_lp).write_text(engine.ExportModelAsLpFormat(False),encoding='ascii',newline='\n')
        checksum = hashlib.sha256(Path(output_lp).read_bytes()).hexdigest()
        if checksum != record['proof_files']['model.lp']['sha256']:
            raise ValueError('The regenerated native model differs from the certified input.')
        result['model_sha256'] = checksum
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    cli.add_argument('--receipt',type=Path,default=DIRECTORY/'R35_SPHERE_CONTACT_PROOF.json')
    cli.add_argument('--proof',type=Path)
    cli.add_argument('--output-lp',type=Path)
    print(json.dumps(replay(**vars(cli.parse_args())),indent=2))
