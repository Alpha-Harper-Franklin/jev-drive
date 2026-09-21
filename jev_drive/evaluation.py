"""Group isolation and probabilistic metrics for externally labelled rollouts."""
import math
import random


def partition_groups(records, seed=0, train_fraction=0.6, calibration_fraction=0.2):
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1 or train_fraction + calibration_fraction >= 1:
        raise ValueError('Provide positive train/calibration/test fractions')
    groups = sorted({r['group_id'] for r in records})
    if len(groups) < 3 or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('At least three independent source groups are required')
    random.Random(seed).shuffle(groups)
    train_count = max(1, min(len(groups)-2, int(len(groups)*train_fraction)))
    calibration_count = max(1, min(len(groups)-train_count-1, int(len(groups)*calibration_fraction)))
    mapping = {group: ('train' if i < train_count else 'calibration' if i < train_count+calibration_count else 'test') for i, group in enumerate(groups)}
    result = {name: [] for name in ['train', 'calibration', 'test']}
    samples = set()
    image_partitions = {}
    for row in records:
        if row['sample_id'] in samples:
            raise ValueError('Duplicate sample identity')
        samples.add(row['sample_id'])
        split = mapping[row['group_id']]
        image_hash = row.get('image_sha256')
        if image_hash:
            previous = image_partitions.setdefault(image_hash, split)
            if previous != split:
                raise ValueError('Identical image crosses data partitions')
        result[split].append(row)
    return result


def binary_metrics(probabilities, outcomes, bins=10):
    """For independently observed binary rollout outcomes, not Jev confidence."""
    if type(bins) is not int or bins <= 0:
        raise ValueError('bins must be positive')
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError('Aligned nonempty predictions and outcomes are required')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probability')
    if any(type(y) is not int or y not in (0, 1) for y in outcomes):
        raise ValueError('Outcomes must be measured binary labels')
    cells = [[] for _ in range(bins)]
    for p, y in zip(probabilities, outcomes):
        cells[min(bins-1, int(p*bins))].append((p, y))
    reliability = []
    ece = 0.0
    for index, cell in enumerate(cells):
        if not cell:
            continue
        predicted = sum(p for p, _ in cell)/len(cell)
        observed = sum(y for _, y in cell)/len(cell)
        ece += len(cell)/len(outcomes)*abs(predicted-observed)
        reliability.append({'bin': index, 'count': len(cell), 'mean_prediction': predicted, 'observed_rate': observed})
    return {'count': len(outcomes), 'brier': sum((p-y)**2 for p,y in zip(probabilities,outcomes))/len(outcomes),
            'ece': ece, 'base_rate': sum(outcomes)/len(outcomes), 'bins': reliability}
