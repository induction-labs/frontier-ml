#!/usr/bin/env python3
"""Offline-first Sol calibration pilot, sharing the proposed $8 aggregate SVG cap.

This provisional scorer is not the official verifier. Never reset either ledger.
"""
import argparse
import base64
from collections import Counter
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

SEED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEED))
import svg_render

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

legacy = load('svg_frozen_luna_calibration', SEED / 'pipeline/luna_v1/pipeline/calibrate_judge.py')
public = load('svg_provisional_sol', SEED / 'pipeline/sol_validate.py')
encoded, digest, atomic_json, atomic_write, usd = legacy.encoded, legacy.digest, legacy.atomic_json, legacy.atomic_write, legacy.usd
ENDPOINT = legacy.ENDPOINT
IMAGE_ID = legacy.IMAGE_ID
STRATA, KINDS = legacy.STRATA, legacy.KINDS
RECIPE = 'svg-judge-calibration-v2-sol-audited'
MODEL = 'gpt-6-sol'
CAP_NANO_USD = 8_000_000_000
PRIOR_PLAN_SHA256 = '640f45f16877e5ca33f3c05c242363a0b663944275017c612f9d799bd6411ffe'
PRIOR_LEDGER_SHA256 = '36fd1c7b13e59387adca48b9d2ea1b390b6ad474e12f0d2f78177737f3599af4'
PRIOR_CHARGE_NANO_USD = 28_990_500
INPUT_TOKENS = 1_050_000
OUTPUT_TOKENS = 1024
INPUT_RATE = 5_500  # $5/M long-context cache-write rate times 1.10, nano-USD/token
OUTPUT_RATE = 16_500  # $15/M long-context output rate times 1.10, nano-USD/token
RESERVATION = INPUT_TOKENS * INPUT_RATE + OUTPUT_TOKENS * OUTPUT_RATE
PLANNED_CALLS = 96
SOURCES = {
    'checked_on': '2026-09-25',
    'model': 'https://developers.openai.com/api/docs/models/gpt-6-sol',
    'rates': 'https://developers.openai.com/api/docs/pricing',
    'images': 'https://developers.openai.com/api/docs/guides/images-vision',
    'settings': 'https://developers.openai.com/api/docs/guides/deployment-checklist',
    'cache_write_accounting': 'https://developers.openai.com/api/docs/guides/prompt-caching',
}


def code_hashes():
    return {str(path.relative_to(SEED)): digest(path.read_bytes()) for path in (
        Path(__file__), SEED / 'pipeline/sol_validate.py', SEED / 'svg_render.py',
        SEED / 'pipeline/luna_v1/pipeline/calibrate_judge.py',
        SEED / 'pipeline/luna_v1/validate.py', SEED / 'pipeline/luna_v1/svg_render.py')}


def verify_prior(prior_dir):
    prior = Path(prior_dir)
    raw = (prior / 'ledger.json').read_bytes()
    ledger = json.loads(raw)
    plan_raw = (prior / 'plan.json').read_bytes()
    plan = json.loads(plan_raw)
    if digest(raw) != PRIOR_LEDGER_SHA256 or digest(encoded(plan)) != PRIOR_PLAN_SHA256:
        raise ValueError('Original SVG ledger or plan changed')
    expected_binding = {'cap_nano_usd': 1_000_000_000, 'endpoint': ENDPOINT, 'plan_sha256': PRIOR_PLAN_SHA256}
    if ledger['binding'] != expected_binding or len(ledger['attempts']) != 96:
        raise ValueError('Original SVG spending is incomplete or belongs to another plan')
    probes = {p['id']: p for p in plan['probes']}
    seen, charge = set(), 0
    for attempt in ledger['attempts']:
        if attempt['id'] in seen or attempt['status'] != 'confirmed':
            raise ValueError('Original SVG attempts must remain confirmed and unique')
        seen.add(attempt['id'])
        receipt = (prior / 'responses' / (attempt['id'] + '.json')).read_bytes()
        expected_charge = legacy.confirmed_charge(json.loads(receipt))
        if (digest(receipt) != attempt['response_sha256'] or expected_charge != attempt['charge_nano_usd']
                or attempt['request_sha256'] != probes[attempt['id']]['request_sha256']):
            raise ValueError('Original SVG receipt or conservative charge changed')
        charge += expected_charge
    if charge != PRIOR_CHARGE_NANO_USD:
        raise ValueError('Original SVG charge differs from the authorized carry-forward')
    return {'ledger_sha256': digest(raw), 'plan_sha256': PRIOR_PLAN_SHA256, 'charge_nano_usd': charge}, plan


def wire_payload(payload):
    if set(payload) != {'model', 'messages', 'temperature', 'reasoning_effort', 'max_completion_tokens'} or (
            payload['model'], payload['temperature'], payload['reasoning_effort'], payload['max_completion_tokens']) != (MODEL, 0, 'none', OUTPUT_TOKENS):
        raise ValueError('Provisional scorer generation settings changed')
    body = {**payload, 'service_tier': 'default', 'store': False}
    content = body['messages'][1]['content']
    if len(content) != 5 or [part['type'] for part in content] != ['text', 'text', 'image_url', 'text', 'image_url']:
        raise ValueError('Both images require explicit preceding text labels')
    text_bytes = len(body['messages'][0]['content'].encode()) + sum(len(p['text'].encode()) for p in content if p['type'] == 'text')
    return body, text_bytes


def choose_samples(rows, provenance, prior_plan):
    old_artworks = {digest(r['reference_svg'].encode()) for r in prior_plan['samples']}
    old_renders = {Path(p['images'][0]).stem for p in prior_plan['probes']}
    selected = {}
    for stratum in STRATA:
        pool = [r for r in rows if r['stratum'] == stratum and digest(r['reference_svg'].encode()) not in old_artworks and provenance[r['id']]['render_sha256'] not in old_renders]
        pool.sort(key=lambda r: digest((RECIPE + ':' + provenance[r['id']]['source_key']).encode()))
        if len(pool) < 8:
            raise ValueError('Eight fresh rows per stratum required')
        selected[stratum] = pool[:8]
    return [selected[stratum][index] for index in range(8) for stratum in STRATA]


def prepare(data_dir, prior_dir, work_dir):
    work, data = Path(work_dir), Path(data_dir)
    if (work / 'ledger.json').exists():
        raise ValueError('Cannot regenerate a plan after spending reservations exist')
    prior, prior_plan = verify_prior(prior_dir)
    _, version = svg_render.renderer_binary(official=True)
    source = (data / 'private/test.jsonl').read_bytes()
    frozen = json.loads((SEED / 'pipeline/prepared_lock.json').read_text())
    manifest = json.loads((data / 'manifest.json').read_text())
    if digest(source) != frozen['sha256'] or len(source) != frozen['bytes'] or manifest.get('data_version') != 'caption-audit-v2':
        raise ValueError('Use the frozen reviewed version2 data')
    rows = [json.loads(line) for line in source.splitlines()]
    if len(rows) != frozen['samples']:
        raise ValueError('Frozen sample count mismatch')
    provenance = {r['id']: r for r in manifest['records']}
    samples = choose_samples(rows, provenance, prior_plan)
    probes, images = [], {}
    for row in samples:
        for kind in KINDS:
            def capture(**payload):
                body, text_bytes = wire_payload(payload)
                image_paths = []
                for part in body['messages'][1]['content']:
                    if part['type'] != 'image_url':
                        continue
                    image = part['image_url']
                    if image['detail'] != 'high' or not image['url'].startswith('data:image/png;base64,'):
                        raise ValueError('Only exact 512-square high-detail PNG inputs allowed')
                    png = base64.b64decode(image['url'].split(',', 1)[1], validate=True)
                    if not svg_render._png_nonuniform(png):
                        raise ValueError('Probe must invoke the visual judge')
                    fingerprint = digest(png)
                    relative = 'images/' + fingerprint + '.png'
                    atomic_write(work / relative, png)
                    images[relative] = {'sha256': fingerprint, 'bytes': len(png), 'width': 512, 'height': 512}
                    image_paths.append(relative)
                index = len(probes)
                request = encoded(body)
                request_path = f'requests/{index:03d}.json'
                atomic_write(work / request_path, request)
                probes.append({'id': f'{index:03d}', 'kind': kind, 'sample_id': row['id'], 'stratum': row['stratum'],
                    'source_key': provenance[row['id']]['source_key'], 'request_path': request_path,
                    'request_sha256': digest(request), 'images': image_paths, 'text_utf8_bytes': text_bytes})
                return {'model': MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(dict.fromkeys(public.CHECKS, False))}}]}
            public.score([row], [{'id': row['id'], 'svg': legacy.candidate_svg(kind, row)}], {'judge': capture})
    plan = {'recipe': RECIPE, 'endpoint': ENDPOINT, 'model': MODEL, 'renderer': version, 'renderer_image_id': IMAGE_ID,
        'input': {'sha256': digest(source), 'bytes': len(source), 'samples': len(rows), 'audit_sha256': manifest['caption_audit']['sha256']},
        'code_sha256': code_hashes(), 'rubric_sha256': public.RUBRIC_SHA256, 'samples': samples, 'probes': probes, 'images': images,
        'prior': prior, 'cap_nano_usd': CAP_NANO_USD, 'reservation_nano_usd': RESERVATION,
        'bounds': {'aggregate_input_including_both_images': INPUT_TOKENS, 'text_utf8_bytes_maximum': max(p['text_utf8_bytes'] for p in probes), 'output_tokens_including_reasoning': OUTPUT_TOKENS,
                   'input_nano_usd_per_token': INPUT_RATE, 'output_nano_usd_per_token': OUTPUT_RATE,
                   'basis': 'Full documented 1,050,000-token context bound, including images and framing. Highest Standard long-context cache-write/output rates plus 10% regional margin. No image formula or tokenizer estimate is assumed.'},
        'pricing_sources': SOURCES,
        'admission': {'reference_minimum': 0.95, 'constant_maximum': 0.15, 'injection_maximum': 0,
                      'complete_samples_per_group': 24, 'finite_scores_required': True,
                      'repeat_verdict_disagreements_maximum': 1, 'repeat_macro_score_difference_maximum': 1 / 24,
                      'note': 'The exact scorer may replace the disabled Luna scorer only after all 24/group controls pass and the operator reviews evidence. No trained-model baseline or task headroom is established by these controls.'}}
    atomic_json(work / 'plan.json', plan)
    return plan


def load_plan(work_dir, prior_dir):
    work = Path(work_dir)
    plan = json.loads((work / 'plan.json').read_text())
    prior, _ = verify_prior(prior_dir)
    if (plan['recipe'], plan['endpoint'], plan['model'], plan['code_sha256'], plan['prior']) != (RECIPE, ENDPOINT, MODEL, code_hashes(), prior):
        raise ValueError('Plan, scorer, source spending or endpoint identity changed')
    if plan['cap_nano_usd'] != CAP_NANO_USD or plan['reservation_nano_usd'] != RESERVATION or len(plan['probes']) != PLANNED_CALLS:
        raise ValueError('Fixed probe count or lifetime budget changed')
    if len(plan['samples']) != 24 or Counter(r['stratum'] for r in plan['samples']) != dict.fromkeys(STRATA, 8):
        raise ValueError('Frozen balanced sample changed')
    for relative, record in plan['images'].items():
        path = (work / relative).resolve()
        if not path.is_relative_to(work.resolve()) or digest(path.read_bytes()) != record['sha256'] or len(path.read_bytes()) != record['bytes']:
            raise ValueError('Frozen image changed or escaped its directory')
    for index, probe in enumerate(plan['probes']):
        row, kind = plan['samples'][index // 4], KINDS[index % 4]
        if probe['id'] != f'{index:03d}' or probe['sample_id'] != row['id'] or probe['kind'] != kind or probe['request_path'] != f'requests/{index:03d}.json':
            raise ValueError('Probe ordering changed')
        raw = (work / probe['request_path']).read_bytes()
        body = json.loads(raw)
        payload = {k: v for k, v in body.items() if k not in {'service_tier', 'store'}}
        expected, _ = wire_payload(payload)
        if (body != expected or digest(raw) != probe['request_sha256']
                or body['messages'][0] != {'role': 'system', 'content': public.SYSTEM}
                or body['messages'][1]['role'] != 'user'):
            raise ValueError('Frozen scorer request changed')
        content = body['messages'][1]['content']
        if (content[1]['text'] != 'REFERENCE IMAGE (an example drawing; do not score this image):'
                or content[3]['text'] != 'CANDIDATE IMAGE (the drawing to score against the instruction):'):
            raise ValueError('Frozen image labels changed')
        if content[0] != {'type': 'text', 'text': json.dumps({'instruction': row['instruction']}, ensure_ascii=True)} or [content[2], content[4]] != [public._image((work / p).read_bytes()) for p in probe['images']]:
            raise ValueError('Frozen instruction or images changed')
    return plan


def confirmed_charge(response):
    public.parse_judgment(response)
    if response.get('service_tier') != 'default':
        raise ValueError('Native endpoint did not confirm Standard processing')
    usage = response.get('usage', {})
    prompt, completion, total = (usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    if any(type(value) is not int or value < 0 for value in (prompt, completion, total)) or prompt > INPUT_TOKENS or completion > OUTPUT_TOKENS or total != prompt + completion:
        raise ValueError('Missing token usage or usage exceeds the reviewed reservation envelope')
    return prompt * INPUT_RATE + completion * OUTPUT_RATE


@contextmanager
def locked_ledger(work_dir, prior_dir, plan):
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Hold the historical ledger's lock as well: another operator cannot run the
    # original plan concurrently while this shared lifetime budget is in use.
    with (Path(prior_dir) / 'ledger.lock').open('rb') as prior_lock:
        fcntl.flock(prior_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior, _ = verify_prior(prior_dir)
        fd = os.open(work / 'ledger.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = work / 'ledger.json'
            binding = {'plan_sha256': digest(encoded(plan)), 'cap_nano_usd': CAP_NANO_USD, 'endpoint': ENDPOINT, 'prior': prior}
            ledger = json.loads(path.read_text()) if path.exists() else {'binding': binding, 'attempts': []}
            if ledger.get('binding') != binding or not isinstance(ledger.get('attempts'), list):
                raise ValueError('Ledger belongs to another plan or lifetime budget')
            known = {probe['id']: probe for probe in plan['probes']}
            seen = set()
            for attempt in ledger['attempts']:
                if attempt['id'] not in known or attempt['id'] in seen or attempt['request_sha256'] != known[attempt['id']]['request_sha256']:
                    raise ValueError('Invalid or duplicated ledger attempt')
                seen.add(attempt['id'])
                if attempt['status'] == 'confirmed':
                    raw = (work / 'responses' / (attempt['id'] + '.json')).read_bytes()
                    if digest(raw) != attempt['response_sha256'] or confirmed_charge(json.loads(raw)) != attempt['charge_nano_usd']:
                        raise ValueError('Recorded response or accounting changed')
                elif attempt['status'] not in {'reserved', 'uncertain_no_retry'} or attempt['charge_nano_usd'] != RESERVATION:
                    raise ValueError('Uncertain calls retain their full reservation')
            if not {p.stem for p in (work / 'responses').glob('*.json')} <= seen:
                raise ValueError('Response archives lack durable reservations')
            if PRIOR_CHARGE_NANO_USD + sum(a['charge_nano_usd'] for a in ledger['attempts']) > CAP_NANO_USD:
                raise ValueError('Aggregate SVG lifetime budget exceeded')
            yield ledger


def dispatch(work_dir, prior_dir, plan, *, opener=None, api_key=None):
    work = Path(work_dir)
    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('OPENAI_API_KEY is required only for authorized execution')
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), legacy.NoRedirect()).open
    with locked_ledger(work, prior_dir, plan) as ledger:
        existing = {a['id']: a for a in ledger['attempts']}
        status, error = 'complete', None
        for probe in plan['probes']:
            if probe['id'] in existing:
                if existing[probe['id']]['status'] != 'confirmed':
                    status = 'uncertain_no_retry'
                    break
                continue
            total = PRIOR_CHARGE_NANO_USD + sum(a['charge_nano_usd'] for a in ledger['attempts'])
            if total + RESERVATION > CAP_NANO_USD:
                status = 'budget_stopped'
                break
            attempt = {'id': probe['id'], 'request_sha256': probe['request_sha256'], 'status': 'reserved',
                       'charge_nano_usd': RESERVATION, 'created_at': int(time.time())}
            ledger['attempts'].append(attempt)
            atomic_json(work / 'ledger.json', ledger)
            try:
                payload = (work / probe['request_path']).read_bytes()
                if digest(payload) != probe['request_sha256']:
                    raise ValueError('Request changed after plan validation')
                request = urllib.request.Request(ENDPOINT, payload, {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
                with opener(request, timeout=120) as response:
                    raw = response.read(legacy.MAX_RESPONSE_BYTES + 1)
                if len(raw) > legacy.MAX_RESPONSE_BYTES:
                    raise ValueError('Oversized native response')
                atomic_write(work / 'responses' / (probe['id'] + '.json'), raw)
                charge = confirmed_charge(json.loads(raw))
                attempt.update(status='confirmed', charge_nano_usd=charge, response_sha256=digest(raw))
                atomic_json(work / 'ledger.json', ledger)
                print(json.dumps({'confirmed_probe': probe['id'], 'kind': probe['kind'], 'aggregate_svg_charge_upper_bound_usd': usd(PRIOR_CHARGE_NANO_USD + sum(a['charge_nano_usd'] for a in ledger['attempts']))}), file=sys.stderr, flush=True)
            except BaseException as exc:
                attempt.update(status='uncertain_no_retry', charge_nano_usd=RESERVATION, error_type=type(exc).__name__)
                atomic_json(work / 'ledger.json', ledger)
                status, error = 'uncertain_no_retry', type(exc).__name__
                break
        charged = sum(a['charge_nano_usd'] for a in ledger['attempts'])
        return {'status': status, 'error_type': error, 'attempted': len(ledger['attempts']), 'confirmed': sum(a['status'] == 'confirmed' for a in ledger['attempts']),
                'new_charge_upper_bound_usd': usd(charged), 'lifetime_charge_upper_bound_usd': usd(PRIOR_CHARGE_NANO_USD + charged), 'attempts': ledger['attempts']}


def dry_report(plan):
    return {'mode': 'dry_run', 'paid_calls': 0, 'recipe': RECIPE, 'plan_sha256': digest(encoded(plan)), 'endpoint': ENDPOINT, 'model': MODEL,
        'planned_calls': PLANNED_CALLS, 'probes': dict(Counter(p['kind'] for p in plan['probes'])),
        'selected_samples': [{'id': r['id'], 'stratum': r['stratum']} for r in plan['samples']],
        'renderer': plan['renderer'], 'renderer_image_id': IMAGE_ID, 'code_sha256': plan['code_sha256'], 'input': plan['input'],
        'prior_charge_upper_bound_usd': usd(PRIOR_CHARGE_NANO_USD), 'remaining_aggregate_budget_usd': usd(CAP_NANO_USD - PRIOR_CHARGE_NANO_USD),
        'lifetime_cap_usd': usd(CAP_NANO_USD), 'per_call_reservation_usd': usd(RESERVATION),
        'unconstrained_new_96_call_worst_case_usd': usd(PLANNED_CALLS * RESERVATION), 'completion_guaranteed': False,
        'budget_note': 'Proposed shared $8 lifetime cap includes the original 96 Luna calls. Current authorization remains $1: new execution requires explicit user approval of the raised cap. Sequential durable reservations; confirmed usage releases conservative headroom. Stop on uncertain receipt or insufficient remaining headroom; never retry.',
        'bounds': plan['bounds'], 'pricing_sources': SOURCES, 'admission': plan['admission'], 'quality_scores': None,
        'authorization': 'Prepared only. No new calls until operator review of plan, exact token envelope and shared ledger.'}


def replay(work_dir, plan, execution):
    work = Path(work_dir)
    confirmed = {a['id'] for a in execution['attempts'] if a['status'] == 'confirmed'}
    groups = {}
    for kind in KINDS:
        probes = [p for p in plan['probes'] if p['kind'] == kind and p['id'] in confirmed]
        samples = [next(r for r in plan['samples'] if r['id'] == p['sample_id']) for p in probes]
        if not probes:
            groups[kind] = {'complete': False, 'confirmed': 0}
            continue
        cursor = iter(probes)
        def cached(**payload):
            probe = next(cursor)
            if digest(encoded(wire_payload(payload)[0])) != probe['request_sha256']:
                raise ValueError('Replayed payload differs from the paid request')
            return json.loads((work / 'responses' / (probe['id'] + '.json')).read_text())
        scored = public.score(samples, [{'id': r['id'], 'svg': legacy.candidate_svg(kind, r)} for r in samples], {'judge': cached})
        groups[kind] = {'complete': len(probes) == 24, 'confirmed': len(probes), **scored}
    left = {r['id']: r['verdict'] for r in groups.get('reference_self', {}).get('details', {}).get('records', [])}
    right = {r['id']: r['verdict'] for r in groups.get('reference_repeat', {}).get('details', {}).get('records', [])}
    common = left.keys() & right.keys()
    difference = (abs(groups['reference_self']['score'] - groups['reference_repeat']['score'])
                  if groups['reference_self'].get('complete') and groups['reference_repeat'].get('complete') else None)
    return {'groups': groups, 'repeat_consistency': {'pairs': len(common), 'verdict_disagreements': sum(left[k] != right[k] for k in common),
                                                  'absolute_macro_score_difference': difference},
            'scope': 'Independent 24-row judge controls. Passing is necessary before promotion; no trained-model baseline or $50 headroom claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--prior-dir', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--approved-aggregate-cap-usd', type=int)
    args = parser.parse_args(argv)
    svg_render.renderer_binary(official=True)
    work = Path(args.work_dir)
    plan = load_plan(work, args.prior_dir) if (work / 'plan.json').exists() else prepare(args.data_dir, args.prior_dir, work)
    report = dry_report(plan)
    if args.execute:
        if args.approved_plan_sha256 != digest(encoded(plan)) or args.approved_aggregate_cap_usd != 8:
            raise ValueError('Execution requires the exact reviewed plan hash and explicit raised aggregate cap of 8 USD')
        execution = dispatch(work, args.prior_dir, plan)
        report.update(mode='executed', execution=execution, quality_scores=replay(work, plan, execution), authorization='Operator supplied exact approved plan hash')
        report.pop('paid_calls')
        atomic_json(work / 'report.json', report)
    else:
        atomic_json(work / 'dry_run.json', report)
    print(json.dumps(report, indent=2))
    return 0 if not args.execute or execution['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
