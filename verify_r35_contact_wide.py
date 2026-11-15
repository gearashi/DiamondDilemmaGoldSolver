"""Replay the signed reference-loss certificate and its explicit relaxation witness."""
import argparse
import hashlib
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_loss import digest
from r35_contact_certificate import CERTIFICATE, verify
from r35_geographic_replay import geographic_departures
from r35_induced_contacts import contact_departures
from r35_loss_witness import physical_price


def read_rows(path, key):
    return [frozenset(map(tuple, row)) for row in json.loads(path.read_bytes())[key]]


def replay(tiles, receipt, *, proof=None, output_lp=None):
    directory = CERTIFICATE.parent
    problem, reference, original, required = problem_scope(tiles)
    checked = json.loads(receipt.read_bytes())
    if checked['fingerprint'] != problem.fingerprint:
        raise ValueError('The original physical tiles differ from the proof source.')
    base = [frozenset(group) for group in original[:13]]
    geographic = read_rows(directory / checked['geographic_file'], 'geographic_cores')
    contacts = [row for row in read_rows(directory / checked['contact_file'], 'edge_perfect_cores')
                if len(row) <= checked['maximum_contact_length']]
    projected_geometry = {group-required for group in geographic}
    projected_contacts = {group-required for group in contacts}
    if (set(geographic_departures(problem, reference, required, geographic)) != projected_geometry
            or set(contact_departures(problem, reference, required, contacts)) != projected_contacts):
        raise ValueError('A supplied geometric premise failed its original physical pose replay.')
    groups = [*base, *projected_geometry, *projected_contacts]
    projected = {group-required for group in groups}
    if (digest(sorted(sorted(group) for group in projected)) != checked['projected_group_sha256']
            or len(projected) != checked['projected_group_count']
            or list(map(sorted, base)) != [list(map(tuple,g)) for g in checked['physical_groups']]):
        raise ValueError('The checked physical groups differ from the recorded scope.')

    witness = checked['witness']
    phase = int(witness['phase'], 16)
    lost = frozenset(map(tuple, witness['lost']))
    replayed = physical_price(problem.tiles, reference, groups, required, phase)
    if (replayed is None or replayed[0] != len(lost) or len(lost) != witness['cost']
            or not (replayed[1] & reference) <= lost or not lost <= reference-required
            or any(not (group & lost) for group in projected)):
        raise ValueError('The recorded signed tile phase or physical dropout set failed replay.')

    weights = json.loads((directory / 'R35_CONTACT_MULTIPLIERS.json').read_bytes())['certificates']
    for weight in weights:
        if not verify(problem, reference, required, base, weight, charged=projected):
            raise ValueError('An integer-weight premise is absent from the physical scope.')
    summary = dict(projected_groups=len(projected), phase=witness['phase'], witness_cost=len(lost),
                   claimed_unsat_cap=checked['strongest_checked_cap'])
    cap = checked['strongest_checked_cap']
    if (checked['minimum_lost'] != cap+1 or len(reference)-checked['minimum_lost'] != checked['maximum_retained']
            or len(lost) != cap+1):
        raise ValueError('The strong integer bound and the attained witness do not meet.')
    cycles = read_rows(directory/'R35_CONTACT_SIGNED_CYCLES.json','required_xor')
    raw_groups = [*base, *geographic, *contacts]
    if proof is not None:
        from verify_r35_contact_vipr import replay_header
        summary.update(replay_header(proof, problem, reference, required, raw_groups, cycles, cap))
        with proof.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if checksum != checked['proof_files']['proof.vipr']['sha256']:
            raise ValueError('The proof digest differs from the independent VIPR verification.')
        summary['proof_sha256'] = checksum
    if output_lp is not None:
        from r35_contact_mip import build_model
        solver, _, _ = build_model(problem, reference, raw_groups, required, cap, cycles)
        output_lp.write_text(solver.ExportModelAsLpFormat(False),encoding='ascii',newline='\n')
        checksum = hashlib.sha256(output_lp.read_bytes()).hexdigest()
        if checksum != checked['proof_files']['model.lp']['sha256']:
            raise ValueError('The exported linear program differs from the exact SCIP input.')
        summary['model_sha256'] = checksum
    return summary


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles', type=Path, default=Path('data/tiles.json'))
    cli.add_argument('--receipt', type=Path, default=CERTIFICATE.parent/'R35_CONTACT_WIDE_PROOF.json')
    cli.add_argument('--output-lp', type=Path, help='Regenerate the named integer model from the raw tiles and geometric premises.')
    cli.add_argument('--proof', type=Path, help='Independently compare the VIPR premises to raw physical inequalities and validate the exact proof digest.')
    print(json.dumps(replay(**vars(cli.parse_args())), indent=2))
