"""Replay recorded descriptions through Jev; never reads images or outcome labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .api import JevClient, JevError
from .protocol import CANDIDATES, QUESTION


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = set()
    for record in records:
        sample_id = record['sample_id']
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError('Missing or duplicate sample ID')
        ids.add(sample_id)
        state = record['state']
        if not isinstance(state, dict) or set(state) != {'task', 'visual_description', 'observation_scope'}:
            raise ValueError('State must contain only task, visual_description, observation_scope')
        if any(not isinstance(value, str) or not value for value in state.values()):
            raise ValueError('State fields must be nonempty text')
    if not records:
        raise ValueError('Empty replay')
    return records


def run_replay(source, output, model='jev-1.13.0', orders=2, client=None):
    if orders not in (1, 2):
        raise ValueError('Use one or two candidate orders')
    records = load_records(source)
    client = client or JevClient(model=model)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    order = list(CANDIDATES)
    orderings = [order, order[1:] + order[:1]][:orders]
    summary = Counter()
    source_sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    with output.open('x', encoding='utf-8') as stream:
        for row in records:
            for candidate_order in orderings:
                started = time.perf_counter()
                record = {key: row[key] for key in ['sample_id', 'group_id', 'image_sha256'] if key in row}
                record.update({'backend': 'jev_caption', 'candidate_order': candidate_order,
                               'input_file_sha256': source_sha, 'requested_model': model})
                try:
                    result = client.choice(row['state'], QUESTION, {key: CANDIDATES[key] for key in candidate_order})
                    record.update(result.to_dict())
                    record['error'] = None
                    summary['successes'] += 1
                except JevError as error:
                    record.update({'choice': None, 'error': str(error), 'latency_ms': (time.perf_counter()-started)*1000})
                    summary['errors'] += 1
                summary['attempts'] += 1
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
    return {'records': len(records), **dict(summary), 'scope': 'offline semantic requests; no vehicle execution'}
