"""Summarize unlabelled diagnostics without inventing accuracy or safety metrics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    lower = int(position)
    return ordered[lower] + (ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(position-lower)


def latency(values):
    return {'count':len(values), 'p50_ms':percentile(values,0.5), 'p95_ms':percentile(values,0.95)}


def summarize(directory):
    directory = Path(directory)
    vision_path = directory/'vision.jsonl'
    vision = [json.loads(line) for line in vision_path.read_text().splitlines() if line]
    completion = json.loads((directory/'complete.json').read_text())
    if len(vision) != completion['records'] or hashlib.sha256(vision_path.read_bytes()).hexdigest() != completion['vision_sha256']:
        raise ValueError('Incomplete or mismatched vision output')
    jev = [json.loads(line) for line in (directory/'jev.jsonl').read_text().splitlines() if line]
    by_id = {row['sample_id']:row for row in vision}
    if len(by_id) != len(vision) or any(row['sample_id'] not in by_id for row in jev):
        raise ValueError('Invalid sample identities')
    if len(jev) != 2*len(vision):
        raise ValueError('Expected two Jev attempts per image for this diagnostic')
    seen = set()
    for row in jev:
        key = (row['sample_id'],tuple(row['candidate_order']))
        if key in seen or row['image_sha256'] != by_id[row['sample_id']]['image_sha256']:
            raise ValueError('Duplicate ordering or mismatched image')
        seen.add(key)
    records = defaultdict(list)
    for row in vision:
        for decision in row['decisions']:
            records[decision['backend']].append({**decision,'sample_id':row['sample_id'],
                                                'latency_ms':decision['timing']['total_ms']})
    records['jev_caption'] = jev
    backends = {}
    for backend, rows in records.items():
        success = [r for r in rows if not r.get('error')]
        pairs = defaultdict(list)
        for row in success:
            pairs[row['sample_id']].append(row['choice'])
        complete_pairs = [choices for choices in pairs.values() if len(choices)==2]
        backends[backend] = {
            'attempts':len(rows), 'successes':len(success), 'errors':len(rows)-len(success),
            'choice_counts':dict(Counter(r['choice'] for r in success)),
            'decision_latency':latency([r['latency_ms'] for r in success]),
            'attempt_latency_including_errors':latency([r['latency_ms'] for r in rows]),
            'order_pairs':len(complete_pairs),
            'order_changed_choice':sum(a!=b for a,b in complete_pairs),
        }
        if backend.endswith('caption'):
            backends[backend]['per_image_pipeline_latency'] = latency([
                by_id[r['sample_id']]['image_read_ms'] + by_id[r['sample_id']]['caption_timing']['total_ms'] + r['latency_ms']
                for r in success])
        else:
            backends[backend]['per_image_pipeline_latency'] = latency([
                by_id[r['sample_id']]['image_read_ms'] + r['latency_ms'] for r in success])
        if backend.startswith('qwen'):
            backends[backend]['median_candidate_probability_mass'] = statistics.median(r['candidate_probability_mass'] for r in success)
    usage = [r.get('input_tokens') for r in jev]
    comparisons = {}
    first_order = vision[0]['decisions'][0]['candidate_order']
    first = {name:{r['sample_id']:r['choice'] for r in rows if not r.get('error') and r['candidate_order']==first_order} for name,rows in records.items()}
    for a,b in [('jev_caption','qwen_caption'),('jev_caption','qwen_image')]:
        shared = first[a].keys() & first[b].keys()
        comparisons[a+'__'+b] = {'paired_count':len(shared),'disagreements':sum(first[a][k]!=first[b][k] for k in shared)}
    return {
        'scope':'unlabelled recorded-image diagnostic; no driving accuracy or closed-loop outcome measured',
        'images':len(vision), 'source_groups':len({r['group_id'] for r in vision}),
        'caption_truncations':sum(r['caption_timing']['hit_token_limit'] for r in vision),
        'caption_latency':latency([r['caption_timing']['total_ms'] for r in vision]),
        'backends':backends, 'first_order_disagreement':comparisons,
        'jev_models':sorted({r['model'] for r in jev if not r.get('error')}),
        'jev_input_tokens':sum(n for n in usage if n is not None),
        'jev_estimated_usd':sum(usage)*0.042/1_000_000 if all(n is not None for n in usage) else None,
        'cost_note':'Input-token estimate using published $0.042/M; not a billing receipt; excludes GPU cost',
        'timing_note':'Warm model. Single process. Jev includes client network time. Caption pipelines include one caption generation per frame. Two orderings are repeated diagnostics, not independent images. No camera transport, simulator stepping, or actuation time is measured.',
        'vision_sha256':completion['vision_sha256'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    result = summarize(args.directory)
    target = Path(args.directory)/'summary.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
