"""Recorded RGB diagnostic: caption + image/text decision baselines, no labels.

Run as a module from the repository root or after installing jev-drive.
Writes private records; does not upload images to any external service.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_drive.protocol import CANDIDATES, QUESTION, CAPTION_PROMPT


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import transformers

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'vision.jsonl'
    if records_path.exists():
        raise FileExistsError('Use a new output directory; evidence must not be overwritten')
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain distinct sample IDs')
    for row in rows:
        if digest(row['image_path']) != row['image_sha256']:
            raise ValueError('Image hash mismatch')
    model_root = Path(args.model)
    files = sorted(model_root.glob('*.safetensors')) + [model_root/'config.json', model_root/'tokenizer.json']
    print(json.dumps({'phase': 'hash_model', 'files': len(files)}), flush=True)
    model_hashes = {p.name: digest(p) for p in files}
    started = time.perf_counter()
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map='cuda:0', local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
                                             min_pixels=256*28*28, max_pixels=512*28*28)
    torch.cuda.synchronize()
    metadata = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'model': model_root.name, 'model_sha256': model_hashes,
        'manifest_sha256': digest(args.manifest), 'script_sha256': digest(__file__),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
        'dtype': 'bfloat16', 'attention': 'sdpa', 'seed': 0,
        'sample_count': len(rows), 'load_seconds': time.perf_counter()-started,
        'caption_prompt': CAPTION_PROMPT, 'question': QUESTION, 'candidates': CANDIDATES,
        'max_pixels': 512*28*28, 'max_caption_tokens': 160,
        'scope': 'recorded-image interface diagnostic; no outcome labels or vehicle execution',
        'probability_semantics': 'uncalibrated conditional next-token scores restricted to candidate letters',
    }
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'phase': 'model_loaded', 'load_seconds': metadata['load_seconds']}), flush=True)

    def generate(prompt, image=None, count=1, scores=False):
        began = time.perf_counter()
        content = ([{'type': 'image'}] if image is not None else []) + [{'type': 'text', 'text': prompt}]
        text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image] if image is not None else None, return_tensors='pt').to('cuda:0')
        torch.cuda.synchronize()
        forward = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=count, do_sample=False,
                                    return_dict_in_generate=True, output_scores=scores)
        torch.cuda.synchronize()
        generated = result.sequences[0, inputs['input_ids'].shape[1]:]
        decoded = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return result, decoded, {'total_ms': (time.perf_counter()-began)*1000,
                                 'model_ms': (time.perf_counter()-forward)*1000,
                                 'input_tokens': inputs['input_ids'].shape[1],
                                 'output_tokens': len(generated),
                                 'hit_token_limit': len(generated) == count and int(generated[-1]) != processor.tokenizer.eos_token_id}

    def decide(state, image, order):
        letters = 'ABCD'
        mapping = dict(zip(letters, order))
        prompt = QUESTION + '\nTask and evidence: ' + json.dumps(state) + '\n'
        prompt += '\n'.join(f'{letter}: {CANDIDATES[action]}' for letter, action in mapping.items())
        prompt += '\nRespond with exactly one letter: A, B, C, or D.'
        tokens = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
        if any(len(token) != 1 for token in tokens):
            raise ValueError('Candidate letters must be single tokens')
        result, decoded, timing = generate(prompt, image, scores=True)
        logits = result.scores[0][0].float()
        selected_logits = logits[[token[0] for token in tokens]]
        conditional = selected_logits.softmax(dim=-1).cpu().tolist()
        mass = float((selected_logits.logsumexp(0) - logits.logsumexp(0)).exp())
        probabilities = dict(zip(order, conditional))
        return {'choice': max(probabilities, key=probabilities.get), 'probabilities': probabilities,
                'candidate_probability_mass': mass, 'greedy_token': decoded,
                'candidate_order': order, 'timing': timing, 'calibrated': False}

    # Warm-up is recorded separately and excluded from the per-sample timing table.
    image = Image.open(rows[0]['image_path']).convert('RGB')
    _, _, warmup = generate('Describe the visible road.', image, count=1)
    (output/'warmup.json').write_text(json.dumps(warmup), encoding='utf-8')
    with records_path.open('x', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            image_started = time.perf_counter()
            image = Image.open(row['image_path']).convert('RGB')
            image_read_ms = (time.perf_counter()-image_started)*1000
            _, caption, timing = generate(CAPTION_PROMPT, image, count=160)
            state = {'task': row['task'], 'visual_description': caption,
                     'observation_scope': 'single RGB frame; no motion history or destination coordinates'}
            order = list(CANDIDATES)
            record = {k: row[k] for k in ['sample_id', 'group_id', 'image_sha256', 'provenance']}
            record.update({'state': state, 'caption_timing': timing, 'image_read_ms': image_read_ms,
                           'image_size': list(image.size), 'decisions': []})
            for current_order in [order, order[1:]+order[:1]]:
                # Full image baseline gets the same task, but no generated caption.
                visual_state = {k:v for k,v in state.items() if k != 'visual_description'}
                record['decisions'].append({'backend': 'qwen_image', **decide(visual_state, image, current_order)})
                record['decisions'].append({'backend': 'qwen_caption', **decide(state, None, current_order)})
            record['peak_allocated_gpu_gb'] = torch.cuda.max_memory_allocated()/1e9
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            print(json.dumps({'phase': 'sample_complete', 'index': index+1, 'total': len(rows), 'sample_id': row['sample_id']}), flush=True)
    (output/'complete.json').write_text(json.dumps({'records': len(rows), 'vision_sha256': digest(records_path)}), encoding='utf-8')
    print(json.dumps({'phase': 'complete', 'records': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
