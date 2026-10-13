"""Pinned IDM acquisition and split policy; action scoring lives in tasks/idm-training/validate.py."""
from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from frontier_ml.scoring.sources import download_file


IDM_SOURCES = Path(__file__).with_name('idm_sources.json')
VALIDATION_ID = 'clips_recording_654338eb-292a-4c63-9d35-2a8625a9c440_seg0001/clip_001_mixed'
PREVIOUS_VALIDATION_ID = 'clips_recording_03e773da-0d27-45a9-a3bb-3cfad231c0e1_seg0002/clip_000_scroll_drag'
SPLIT_POLICY = 'all-four-action-types-example-v2-remaining-43-heldout'


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Select the mixed-action example; all other clips, including the old example, are held out."""
    by_id = {row['id']: row for row in records}
    if len(records) != 44 or len(by_id) != 44:
        raise ValueError('Pinned IDM dataset must contain 44 unique clips')
    if not {VALIDATION_ID, PREVIOUS_VALIDATION_ID} <= by_id.keys():
        raise ValueError('Pinned IDM dataset is missing a current or previous public example')
    validation = [by_id[VALIDATION_ID]]
    heldout = [by_id[key] for key in sorted(by_id) if key != VALIDATION_ID]
    return validation, heldout


def idm_lock() -> dict:
    """The held-out dataset lock; deliberately kept out of agent images (see agent_surface.py)."""
    return json.loads(IDM_SOURCES.read_text())


def video_dimensions(path: Path) -> tuple[int, int]:
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height', '-of', 'json', str(path)],
                           check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(probe.stdout)['streams'][0]
    return int(stream['width']), int(stream['height'])


def prepare(destination: str | Path) -> dict:
    """Fetch only the 44 clip/label pairs (84.6 MB), never editing-work originals."""
    root = Path(destination)
    lock = idm_lock()
    def get(item):
        download_file(f"https://huggingface.co/datasets/{lock['repo']}/resolve/{lock['revision']}/{item['path']}",
                      root / item['path'], size=item['size'], sha256=item['sha256'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(get, lock['files']))
    records = []
    for item in sorted(lock['files'], key=lambda x: x['path']):
        if not item['path'].endswith('.json'):
            continue
        label_path = root / item['path']
        meta = json.loads(label_path.read_text())
        video = label_path.with_suffix('.mp4')
        width, height = video_dimensions(video)
        records.append({'id': str(label_path.relative_to(root).with_suffix('')),
                        'video': str(video.relative_to(root)), 'fps': 5,
                        'n_frames': math.floor((meta['end_s'] - meta['start_s']) * 5 + 0.5),
                        'video_w': width, 'video_h': height,
                        'start_s': meta['start_s'], 'actions': meta['actions']})
    validation, heldout = split_records(records)
    for name, rows in [('validation', validation), ('heldout', heldout)]:
        (root / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = {'source': lock['repo'], 'revision': lock['revision'],
                'split_policy': SPLIT_POLICY,
                'validation': [row['id'] for row in validation], 'heldout': [row['id'] for row in heldout],
                'previous_validation': [PREVIOUS_VALIDATION_ID],
                'scorer_revision': lock['code_revision'], 'metric': 'micro_action_f1'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download pinned IDM clips and create the 1/43 split')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))
