"""Selection, ordering and primary verification shared by the compact scripts.

The selection keys follow the current minimal_selection branches.  In
particular, image overlap precedes response spread; text minimal selection
does not use a response-band or spread objective.
"""
import copy
import hashlib
import json
import math
import random
import secrets
import sys
import time
from pathlib import Path

import numpy as np


class Progress:
    """One terminal bar, or bounded plain-text updates suitable for nohup."""

    construction = None
    construction_labels = {'Anchor confidence', 'Anchor response probes', 'Mutation sampling',
                           'Mutation scoring', 'Mutation selecting', 'Mutation references'}

    def __init__(self, total, label):
        self.total = max(1, int(total))
        self.label = label
        self.done = 0
        self.last = 0
        self.started = time.monotonic()
        self.bar = None
        self.parent = self.construction if label in self.construction_labels else None
        self.forwarded = 0
        if self.parent is not None:
            return
        if sys.stderr.isatty():
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=label, leave=False)
        else:
            value = '0%' if label == 'Trigger construction' else f'0/{total}'
            print(f"[{label}] {value}", flush=True)

    def update(self, count=1):
        if count <= 0:
            return
        self.done += count
        if self.parent is not None:
            target = min(100, int(100 * self.done / self.total))
            self.parent.update(target - self.forwarded)
            self.forwarded = target
            return
        if self.bar is not None:
            self.bar.update(count)
        elif self.done >= self.total or self.done - self.last >= max(1, self.total // 10):
            elapsed = time.monotonic() - self.started
            value = f'{100 * self.done / self.total:.1f}%' if self.label == 'Trigger construction' else f'{self.done}/{self.total}'
            print(f"[{self.label}] {value}, elapsed={elapsed:.1f}s", flush=True)
            self.last = self.done

    def close(self):
        if self.parent is not None:
            self.parent.update(100 - self.forwarded)
            self.forwarded = 100
            return
        if self.bar is not None:
            self.bar.close()

    @classmethod
    def start_construction(cls, phases):
        cls.construction = cls(phases * 100, 'Trigger construction')
        return cls.construction

    @classmethod
    def finish_construction(cls):
        progress = cls.construction
        if progress is not None:
            progress.close()
        cls.construction = None


def image_band(value, target, tolerance):
    tolerance = max(float(tolerance), 1e-6)
    return max(0.0, abs(float(value) - target) - tolerance) / tolerance


def patch_overlap(spec_a, spec_b):
    cells_a, cells_b = spec_a['cells'], spec_b['cells']
    area_a = sum(c['size'] ** 2 for c in cells_a)
    area_b = sum(c['size'] ** 2 for c in cells_b)
    intersection = 0
    for a in cells_a:
        for b in cells_b:
            height = max(0, min(a['top'] + a['size'], b['top'] + b['size']) - max(a['top'], b['top']))
            width = max(0, min(a['left'] + a['size'], b['left'] + b['size']) - max(a['left'], b['left']))
            intersection += height * width
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def descriptor_similarity(a, b):
    va, vb = a['spec_descriptor'], b['spec_descriptor']
    return max(0.0, float(np.dot(va, vb))) if va.shape == vb.shape else 0.0


def select_image_candidates(candidates, n_bits, threshold, spread_weight):
    """Preserve the current image minimal lexicographic selection, including fallback."""
    if len(candidates) < n_bits:
        raise ValueError(f"Only {len(candidates)} image candidates for {n_bits} bits")
    in_band = [c for c in candidates if c['distortion_band'] <= 1e-12]
    pool = in_band if len(in_band) >= n_bits else sorted(
        candidates, key=lambda c: (c['distortion_band'], c['distortion']))
    scale = max(float(np.std([c['score'] for c in pool])), 1e-6)
    # Cached pairwise updates retain the same key while avoiding repeated O(N*K*K) work.
    scores = np.asarray([c['score'] for c in pool], dtype=np.float64)
    bands = np.asarray([c['distortion_band'] for c in pool], dtype=np.float64)
    distortions = np.asarray([c['distortion'] for c in pool], dtype=np.float64)
    max_overlap = np.zeros(len(pool))
    max_descriptor = np.zeros(len(pool))
    min_gap = np.full(len(pool), np.inf)
    remaining = np.ones(len(pool), dtype=bool)
    chosen_indices = []
    fallback = 0
    progress = Progress(n_bits, 'Mutation selecting')
    while len(chosen_indices) < n_bits:
        indices = np.flatnonzero(remaining).tolist()
        if not chosen_indices:
            chosen = min(indices, key=lambda i: (bands[i], distortions[i]))
        else:
            feasible = [i for i in indices if max_overlap[i] <= threshold]
            if not feasible:
                fallback += 1
            chosen = min(feasible or indices, key=lambda i: (
                max_overlap[i], bands[i], distortions[i],
                -float(spread_weight) * min_gap[i] / scale, max_descriptor[i]))
        remaining[chosen] = False
        chosen_indices.append(chosen)
        for i in np.flatnonzero(remaining):
            max_overlap[i] = max(max_overlap[i], patch_overlap(pool[chosen]['spec'], pool[i]['spec']))
            max_descriptor[i] = max(max_descriptor[i], descriptor_similarity(pool[chosen], pool[i]))
        min_gap = np.minimum(min_gap, np.abs(scores - scores[chosen]))
        progress.update()
    progress.close()
    selected = [pool[i] for i in chosen_indices]
    return selected, {
        'generated': len(candidates), 'selected': len(selected), 'in_distortion_band': len(in_band),
        'minimum_band_shortfall': max(0, n_bits - len(in_band)),
        'selected_out_of_band': sum(c['distortion_band'] > 1e-12 for c in selected),
        'selected_band_mean': float(np.mean([c['distortion_band'] for c in selected])),
        'similarity_fallback': fallback,
        'max_patch_overlap': max((patch_overlap(a['spec'], b['spec']) for i, a in enumerate(selected)
                                  for b in selected[i + 1:]), default=0.0),
        'max_descriptor_similarity': max((descriptor_similarity(a, b) for i, a in enumerate(selected)
                                          for b in selected[i + 1:]), default=0.0),
    }


def text_similarity(a, b):
    pairs_a = set(zip(a['spec']['positions'], a['spec']['replacements']))
    pairs_b = set(zip(b['spec']['positions'], b['spec']['replacements']))
    union = pairs_a | pairs_b
    structural = len(pairs_a & pairs_b) / len(union) if union else 1.0
    va, vb = a.get('feature_vector'), b.get('feature_vector')
    feature = max(0.0, float(np.dot(va, vb))) if va is not None and vb is not None else 0.0
    return max(structural, feature)


def text_loss_window(candidates, n_bits, trim_low, trim_high):
    ordered = sorted(candidates, key=lambda c: c['lm_loss'])
    total = len(ordered)
    if total < n_bits:
        raise ValueError(f"Only {total} text candidates for {n_bits} bits")
    start = min(int(np.floor(total * trim_low)), total)
    end = max(start, total - int(np.floor(total * trim_high)))
    if end - start < n_bits:
        shortfall = n_bits - (end - start)
        left = min(start, (shortfall + 1) // 2)
        right = min(total - end, shortfall - left)
        start, end = start - left, end + right
    if end - start < n_bits:
        start, end = 0, total
    return ordered[start:end], {'generated': total, 'window_start': start, 'window_end': end}


def select_text_candidates(candidates, n_bits, threshold):
    """Current text minimal: first LM-window item, then minimum maximum similarity."""
    if len(candidates) < n_bits:
        raise ValueError('Not enough text mutation candidates')
    remaining = np.ones(len(candidates), dtype=bool)
    max_sim = np.zeros(len(candidates))
    selected_indices = []
    fallback = 0
    progress = Progress(n_bits, 'Mutation selecting')
    while len(selected_indices) < n_bits:
        indices = np.flatnonzero(remaining).tolist()
        if not selected_indices:
            chosen = indices[0]
        else:
            feasible = [i for i in indices if max_sim[i] <= threshold]
            if not feasible:
                fallback += 1
            chosen = min(feasible or indices, key=lambda i: max_sim[i])
        remaining[chosen] = False
        selected_indices.append(chosen)
        for i in np.flatnonzero(remaining):
            max_sim[i] = max(max_sim[i], text_similarity(candidates[chosen], candidates[i]))
        progress.update()
    progress.close()
    selected = [candidates[i] for i in selected_indices]
    max_similarity = max((text_similarity(a, b) for i, a in enumerate(selected)
                          for b in selected[i + 1:]), default=0.0)
    return selected, {'selected': len(selected), 'similarity_fallback': fallback,
                      'max_similarity': max_similarity}


def _jsonable(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        return {'tensor_shape': list(value.shape), 'tensor_dtype': str(value.dtype),
                'tensor_sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def spec_digest(spec):
    data = json.dumps(_jsonable(spec), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _content_key(spec):
    if 'cells' in spec:
        cells = [(int(c['top']), int(c['left']), int(c['size']), int(c.get('source_idx', -1)),
                  spec_digest({'patch': c['patch']})) for c in spec['cells']]
        return ('image', tuple(sorted(cells)), int(spec.get('source_idx', -1)))
    return ('text', tuple(spec.get('positions', [])), tuple(spec.get('replacements', [])),
            int(spec.get('source_idx', -1)))


def canonicalize_mutation_specs(specs, nonce_seed=0):
    """Current per-group content order; group membership is kept by the enclosing record."""
    rng = random.Random(int(nonce_seed))
    records = []
    for index, source in enumerate(specs):
        if not isinstance(source, dict):
            raise TypeError(f'Mutation {index} must be a dict')
        item = copy.deepcopy(source)
        if 'canonical_nonce' not in item:
            item['canonical_nonce'] = rng.getrandbits(64)
        records.append((item, _content_key(item), int(item['canonical_nonce']), index))
    records.sort(key=lambda row: (row[1], row[2], row[3]))
    ordered, manifest = [], []
    for rank, (item, _, nonce, _) in enumerate(records):
        item['canonical_order_rank'] = rank
        item['canonical_spec_digest'] = spec_digest(item)
        ordered.append(item)
        manifest.append({'rank': rank, 'nonce': nonce, 'spec_digest': item['canonical_spec_digest']})
    data = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return ordered, hashlib.sha256(data).hexdigest()


def match_stats(expected, extracted):
    """Fraction of decoded signs that match the registered watermark."""
    expected, extracted = np.asarray(expected), np.asarray(extracted)
    if expected.size == 0 or expected.shape != extracted.shape:
        raise ValueError('Expected and extracted bits must have the same nonzero length')
    return {'wsr': float(np.mean(expected == extracted)), 'correct': int(np.sum(expected == extracted)),
            'n_bits': int(expected.size)}


def build_run_dir(root, name):
    from datetime import datetime
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name) or 'watermark'
    path = Path(root) / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def add_training_args(parser, *, text=False):
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--secret_seed_base', type=int, default=42)
    parser.add_argument('--whitening_seed', type=int, default=1700001,
                        help='XOR mask seed; fixed demo default for reproducibility; use a private seed for ownership keys')
    parser.add_argument('--secret_mode', choices=['random', 'text', 'binary_text_image'], default='random')
    parser.add_argument('--secret_text', default='')
    parser.add_argument('--secret_logo_style', choices=['none', 'nn232', 'bird', 'bird_s', 'lock_s'], default='none')
    parser.add_argument('--secret_image_width', type=int, default=0)
    parser.add_argument('--secret_image_height', type=int, default=0)
    parser.add_argument('--secret_font_path', default='')
    parser.add_argument('--task_name', default='minimal_text' if text else 'minimal_image')
    parser.add_argument('--artifact_root', default='NES/runs')
    parser.add_argument('--trigger_size', type=int, default=4)
    parser.add_argument('--n_deltas', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=6 if text else 10)
    parser.add_argument('--lr', type=float, default=5e-5 if text else 2e-5)
    parser.add_argument('--lambda_wm', type=float, default=5.0 if text else None,
                        help='Watermark weight (text: 5; images: 1 for ImageNet, 0.5 for CIFAR-10)')
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--n_clean_samples', type=int, default=10000 if text else 20000)
    parser.add_argument('--batches_per_epoch', type=int, default=50 if text else 200)
    parser.add_argument('--wm_backward_chunk', type=int, default=8 if text else 64)
    parser.add_argument('--mutation_eval_batch_size', type=int, default=8 if text else 256)
    parser.add_argument('--mutation_similarity_threshold', type=float, default=0.35 if text else 0.18)
    parser.add_argument('--wm_mean_drift_weight', type=float, default=0.1)
    parser.add_argument('--wm_mean_drift_delta_up', type=float, default=3.0)
    parser.add_argument('--wm_mean_drift_delta_down', type=float, default=3.0)
    parser.add_argument('--wm_score_std_floor', type=float, default=0.02 if text else 0.05)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--wm_center_gradient', choices=['detached', 'corrected'], default='detached')
    parser.add_argument('--disable_early_stop_on_separation', dest='early_stop_on_separation', action='store_false')
    parser.set_defaults(early_stop_on_separation=True)
    parser.add_argument('--early_stop_sep_threshold', type=float, default=0.999)
    parser.add_argument('--early_stop_sep_patience', type=int, default=2)
    if not text:
        parser.add_argument('--lr_schedule', choices=['constant', 'cosine', 'step'], default='constant')
        parser.add_argument('--lambda_wm_schedule', choices=['constant', 'warmup', 'cosine'], default='constant')
        parser.add_argument('--lambda_wm_max', type=float, default=4.0)
        parser.add_argument('--margin_schedule', choices=['constant', 'warmup'], default='constant')
        parser.add_argument('--margin_start', type=float, default=0.05)
        parser.add_argument('--margin_warmup_ratio', type=float, default=0.4)


def validate_training_args(args):
    for name in ['trigger_size', 'n_deltas', 'epochs', 'batches_per_epoch', 'n_clean_samples',
                 'wm_backward_chunk', 'mutation_eval_batch_size', 'early_stop_sep_patience']:
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name} must be positive')
    for name in ['lambda_wm', 'margin', 'wm_mean_drift_weight', 'wm_mean_drift_delta_up', 'wm_mean_drift_delta_down']:
        if getattr(args, name) < 0:
            raise ValueError(f'--{name} must be nonnegative')
    if args.lr <= 0 or args.wm_score_std_floor <= 0:
        raise ValueError('Learning rate and score standard-deviation floor must be positive')
    if args.secret_mode == 'random' and args.whitening_seed == args.secret_seed_base:
        raise ValueError('Random payload and whitening seeds must differ; use --whitening_seed 1700001')
    if not 0 <= args.mutation_similarity_threshold <= 1:
        raise ValueError('Mutation similarity threshold must be between 0 and 1')
    if not 0 <= args.early_stop_sep_threshold <= 1:
        raise ValueError('Separation threshold must be between 0 and 1')
    if hasattr(args, 'lambda_wm_max') and min(args.lambda_wm_max, args.margin_start, args.margin_warmup_ratio) < 0:
        raise ValueError('Scheduled weights, starting margin and warmup ratio must be nonnegative')


def image_epoch_objective(args, epoch):
    """Same epoch-based watermark schedules as nes_embed_v2_hinge.py."""
    weight = args.lambda_wm
    if args.lambda_wm_schedule == 'warmup':
        weight += (args.lambda_wm_max - weight) * min(epoch / (args.epochs * 0.3), 1.0)
    elif args.lambda_wm_schedule == 'cosine':
        weight += (args.lambda_wm_max - weight) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    margin = args.margin
    if args.margin_schedule == 'warmup':
        epochs = max(1, int(round(args.epochs * args.margin_warmup_ratio)))
        margin = args.margin_start + (margin - args.margin_start) * min(epoch / epochs, 1.0)
    return weight, margin


def image_lr_scheduler(optimizer, args):
    import torch
    if args.lr_schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.5)
    if args.lr_schedule == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
    return None


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_payload(args, device):
    """Whiten every payload with an independent mask, without position permutation."""
    from secret_payload import (generate_random_secret_message, encode_text_secret_message,
        encode_binary_text_image_secret, whiten_secret, add_secret_transform_metadata)
    total = args.trigger_size * args.n_deltas
    metadata = {'mode': args.secret_mode, 'total_bits': total, 'num_triggers': args.trigger_size,
                'bits_per_trigger': args.n_deltas}
    if args.secret_mode == 'random':
        bits = generate_random_secret_message(total, device, args.secret_seed_base)
    elif args.secret_mode == 'text':
        if not args.secret_text:
            raise ValueError('--secret_text is required in text payload mode')
        bits = encode_text_secret_message(args.secret_text, total, device, args.secret_seed_base)
        metadata.update(text=args.secret_text, encoding='utf-8', header_bits=32)
    else:
        bits, image_meta = encode_binary_text_image_secret(args.secret_text, total, device,
            args.secret_image_width, args.secret_image_height, args.secret_font_path, args.secret_logo_style)
        metadata.update(image_meta)
        metadata.update(text=args.secret_text, logo_style=args.secret_logo_style)
    metadata['raw_secret_bits'] = bits.detach().cpu().tolist()
    seed = args.whitening_seed
    if seed is None:
        seed = secrets.randbits(63)
        while args.secret_mode == 'random' and seed == args.secret_seed_base:
            seed = secrets.randbits(63)
        seed_source = 'private_random'
    else:
        seed_source = 'explicit_private_reproducible'
    if args.secret_mode == 'random' and seed == args.secret_seed_base:
        raise ValueError('Random payload and whitening seeds must differ')
    bits = whiten_secret(bits, seed)
    if total > 1 and bool((bits == bits[0]).all()):
        raise ValueError('XOR whitening produced a constant codeword; choose a different whitening seed')
    metadata = add_secret_transform_metadata(metadata, total, seed)
    metadata['whitening_seed_source'] = seed_source
    print(f"Payload: mode={args.secret_mode}, total_bits={total}, "
          f"whitening=XOR, position_permutation=False, mask_seed_source={seed_source}", flush=True)
    return bits.float(), metadata


def backward_group(score_batch, specs, bits, reference, args, *, text=False, group_count=1,
                   watermark_weight=None, margin=None):
    """Current hinge objective; fixed scale and optional exact centering gradient.

    score_batch is a differentiable callback over specs.  The first pass
    collects current statistics; the second recomputes chunks for backward.
    The reference mean-shift derivative is accumulated through the chunk scores.
    """
    import torch
    import torch.nn.functional as F
    chunk = args.wm_backward_chunk
    if text:
        # The full text entry point uses mutation_eval_batch_size for backward.
        # Smaller recomputation chunks preserve its objective while limiting memory.
        chunk = min(chunk, args.mutation_eval_batch_size)
    margin = args.margin if margin is None else margin
    weight = (args.lambda_wm if watermark_weight is None else watermark_weight) / group_count
    with torch.no_grad():
        scores = torch.cat([score_batch(specs[i:i + chunk]) for i in range(0, len(specs), chunk)])
        score_values = scores.detach().cpu().tolist()
        mean = float(np.mean(score_values)) if text else sum(score_values) / len(score_values)
        std = float(torch.tensor(score_values, dtype=torch.float32).std(unbiased=False))
    scale = std
    if text and args.wm_score_std_cap > 0:
        scale = min(scale, max(reference['std'], args.wm_score_std_cap * reference['std']))
    scale = max(scale, args.wm_score_std_floor, 1e-6)
    ref_std = max(reference['std'], 1e-6)
    drift = (mean - reference['mean']) / ref_std
    up = max(0.0, drift - args.wm_mean_drift_delta_up)
    down = max(0.0, -drift - args.wm_mean_drift_delta_down)
    shift_gradient = args.wm_mean_drift_weight * 2 * (up - down) / ref_std
    correction = 0.0
    if args.wm_center_gradient == 'corrected':
        active_sum = sum(float(sign) for value, sign in zip(score_values, bits.detach().cpu().tolist())
                         if margin - float(sign) * (value - mean) / scale > 0)
        correction = weight * active_sum / (len(specs) ** 2 * scale)
    signed_values = []
    for start in range(0, len(specs), chunk):
        values = score_batch(specs[start:start + chunk])
        signed = bits[start:start + chunk] * (values - mean) / scale
        loss = F.relu(margin - signed)
        objective = weight * loss.sum() / len(specs)
        if correction:
            objective = objective + correction * (values - values.detach()).sum()
        if shift_gradient:
            objective = objective + (shift_gradient / len(specs)) * values.sum()
        objective.backward()
        signed_values.extend(signed.detach().cpu().tolist())
    # Internal training controls only: do not export additional verification metrics.
    signed = torch.tensor(signed_values, dtype=torch.float32)
    return {'positive_rate': float((signed > 0).float().mean()),
            'violation_rate': float((signed < margin).float().mean())}


def verify_groups(records, collect_scores, payload_metadata):
    from secret_payload import restore_whitened_secret_bits
    expected, extracted, groups = [], [], []
    for index, record in enumerate(records):
        values = np.asarray(collect_scores(record), dtype=np.float32)
        mean = float(values.mean())
        # Match the submitted decoder: positive only when score is strictly above the mean.
        recovered = np.where(values > mean, 1, -1)
        bits = record['secret_message'].detach().cpu().numpy()
        stats = match_stats(bits, recovered)
        stats['group'] = index
        groups.append(stats)
        expected.extend(bits.tolist())
        extracted.extend(recovered.tolist())
    raw_expected = payload_metadata['raw_secret_bits']
    raw_extracted = restore_whitened_secret_bits(extracted, payload_metadata)
    restored_target = restore_whitened_secret_bits(expected, payload_metadata)
    if restored_target != raw_expected:
        raise ValueError('Registered embedded target and saved raw payload disagree')
    overall_raw = match_stats(raw_expected, raw_extracted)
    offset = 0
    for stats in groups:
        end = offset + stats['n_bits']
        stats['raw'] = match_stats(raw_expected[offset:end], raw_extracted[offset:end])
        if stats['raw']['correct'] != stats['correct']:
            raise ValueError('XOR inverse changed a group error count')
        offset = end
    return {'groups': groups, 'overall': match_stats(expected, extracted), 'overall_raw': overall_raw}


def print_verification(bundle, label='Verification'):
    for stats in bundle['groups']:
        print(f"[{label}] group={stats['group']}, WSR_codeword={stats['wsr']:.2%}, "
              f"WSR_raw_after_inverse={stats['raw']['wsr']:.2%}", flush=True)
    stats = bundle['overall']
    raw = bundle['overall_raw']
    print(f"[{label}] overall WSR_codeword={stats['wsr']:.2%} ({stats['correct']}/{stats['n_bits']}), "
          f"WSR_raw_after_inverse={raw['wsr']:.2%} ({raw['correct']}/{raw['n_bits']})", flush=True)
