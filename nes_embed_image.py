"""Compact image embedding: current minimal local-mutation selection only."""
import argparse
import glob
import hashlib
import importlib.util
import math
import os
import random
import time
import numpy as np
from minimal_utils import build_payload, Progress, add_training_args, backward_group, build_run_dir, canonicalize_mutation_specs, image_band, image_epoch_objective, image_lr_scheduler, print_verification, seed_everything, select_image_candidates, validate_training_args, verify_groups

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser)
    parser.add_argument('--dataset', choices=['cifar10', 'imagenet100', 'imagenet'], default='imagenet100')
    parser.add_argument('--data_root', default='./data/ImageNet100')
    parser.add_argument('--model_path', default='./models/base/base_model_resnet18_imagenet100.pt')
    parser.add_argument('--arch_path', default='', help='Local resnet18(num_classes=...) implementation')
    parser.add_argument('--val_split', default='test', help='ImageFolder ACC split (default: test); validation also accepts val/')
    parser.add_argument('--ood_data_root', default=None)
    parser.add_argument('--ood_split', default='')
    parser.add_argument('--trigger_source', choices=['train', 'validation', 'val', 'test', 'ood'], default='ood')
    parser.add_argument('--mutation_donor_source', choices=['train', 'test'], default='train')
    parser.add_argument('--trigger_conf_min', type=float, default=0.5)
    parser.add_argument('--trigger_conf_max', type=float, default=0.7)
    parser.add_argument('--trigger_candidate_factor', type=int, default=80)
    parser.add_argument('--trigger_probe_mutations', type=int, default=12)
    parser.add_argument('--trigger_probe_delta_target', type=float, default=0.1)
    parser.add_argument('--trigger_probe_delta_tolerance', type=float, default=0.05)
    parser.add_argument('--mutation_patch_size', type=int, default=0)
    parser.add_argument('--mutation_cells', type=int, default=4)
    parser.add_argument('--mutation_preserve_total_area', action='store_true')
    parser.add_argument('--mutation_alpha', type=float, default=0.35)
    parser.add_argument('--mutation_replace_mode', choices=['blend', 'replace'], default='blend')
    parser.add_argument('--mutation_filter_pool', type=int, default=10)
    parser.add_argument('--mutation_distortion_target', type=float, default=0.1)
    parser.add_argument('--mutation_distortion_tolerance', type=float, default=0.05)
    parser.add_argument('--mutation_spread_weight', type=float, default=2.0)
    parser.add_argument('--clean_per_batch', type=int, default=32)
    parsed = parser.parse_args()
    if parsed.lambda_wm is None:
        parsed.lambda_wm = 0.5 if parsed.dataset == 'cifar10' else 1.0
    return parsed


def resolve_image_split(root, split):
    """Accept validation/val aliases, without substituting the test split."""
    choices = [split] + ([s for s in ('validation', 'val') if s != split]
                         if split in ('validation', 'val') else [])
    for name in choices:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f'Image split {split!r} not found under {root}; expected {choices}')


def load_image_datasets(config, transform, dataset_module):
    """Return clean train, task evaluation, anchor and donor datasets."""
    if config.dataset == 'cifar10':
        full_train = dataset_module.CIFAR10(config.data_root, train=True, download=False, transform=transform)
        evaluation = dataset_module.CIFAR10(config.data_root, train=False, download=False, transform=transform)
        train = full_train
        validation = None
        if config.trigger_source in ('validation', 'val'):
            if len(full_train) < 2:
                raise ValueError('At least two CIFAR training samples are required for validation splitting')
            indices = np.random.RandomState(config.seed).permutation(len(full_train)).tolist()
            count = min(len(full_train) - 1, max(1, len(full_train) // 10))
            validation_indices, train_indices = indices[:count], indices[count:]
            validation = torch.utils.data.Subset(full_train, validation_indices)
            train = torch.utils.data.Subset(full_train, train_indices)
            # Preserve the exact partition in owner metadata.
            config.cifar_validation_indices = validation_indices
            print(f'CIFAR validation: held out {count} training samples (seed={config.seed}); '
                  'excluded from clean training/train donors; ACC source=official test', flush=True)
        anchors = train if config.trigger_source == 'train' else validation if config.trigger_source in ('validation', 'val') else evaluation
        donors = train if config.mutation_donor_source == 'train' else evaluation
        return train, evaluation, anchors, donors

    train = dataset_module.ImageFolder(resolve_image_split(config.data_root, 'train'), transform=transform)
    evaluation_path = resolve_image_split(config.data_root, config.val_split)
    evaluation = dataset_module.ImageFolder(evaluation_path, transform=transform)
    if config.trigger_source in ('train', 'ood'):
        anchors = train if config.trigger_source == 'train' else evaluation
    else:
        anchor_path = resolve_image_split(config.data_root, config.trigger_source)
        anchors = evaluation if anchor_path == evaluation_path else dataset_module.ImageFolder(anchor_path, transform=transform)
    if config.mutation_donor_source == 'train':
        donors = train
    else:
        test_path = resolve_image_split(config.data_root, 'test')
        donors = evaluation if test_path == evaluation_path else anchors if config.trigger_source == 'test' else dataset_module.ImageFolder(test_path, transform=transform)
    return train, evaluation, anchors, donors

def load_external_resnet18(num_classes):
    """Load a user-provided ResNet-18 builder from --arch_path."""
    if not args.arch_path:
        return None
    arch_path = os.path.abspath(os.path.expanduser(args.arch_path))
    if os.path.isdir(arch_path):
        arch_path = os.path.join(arch_path, 'resnet.py')
    if not os.path.exists(arch_path):
        raise FileNotFoundError(f'--arch_path does not exist: {arch_path}')
    spec = importlib.util.spec_from_file_location('external_resnet_arch', arch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load architecture file: {arch_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'resnet18'):
        raise AttributeError('--arch_path must provide a resnet18(num_classes=...) function')
    return module.resnet18(num_classes=num_classes)

def build_model_from_arch(arch, num_classes, image_size, state_dict=None):
    """Rebuild the ResNet-18 backbone used by the compact image experiment."""
    if arch == 'resnet18':
        external_model = load_external_resnet18(num_classes)
        if external_model is not None:
            return external_model
        model = tv_models.resnet18(weights=None)
        stem = state_dict.get('conv1.weight') if state_dict is not None else None
        if stem is not None and tuple(stem.shape[-2:]) == (3, 3):
            # Original experiment backbone: stride-one 3x3 stem, no max pool.
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f'Unsupported model_arch: {arch}')


def align_resnet_state_dict(state_dict, model):
    """Translate equivalent module names, retaining strict weight loading."""
    expected = set(model.state_dict())
    aligned = {}
    for key, value in state_dict.items():
        destination = key
        if key not in expected:
            candidate = key.replace('.shortcut.', '.downsample.')
            if candidate.startswith('linear.'):
                candidate = 'fc.' + candidate[len('linear.'):]
            if candidate in expected:
                destination = candidate
        if destination in aligned:
            raise RuntimeError(f'Checkpoint contains conflicting tensors for {destination}')
        aligned[destination] = value
    return aligned

def infer_num_classes_from_state_dict(state_dict):
    """Infer classifier output dimension from checkpoint weights."""
    if not isinstance(state_dict, dict):
        return None
    for key in ['fc.weight', 'linear.weight']:
        weight = state_dict.get(key)
        if weight is not None and hasattr(weight, 'shape') and (len(weight.shape) >= 1):
            return int(weight.shape[0])
    return None

def load_model_from_checkpoint(checkpoint, num_classes, image_size, device):
    """Load a checkpoint saved as full model or state_dict for supported backbones."""
    resolved_arch = 'resnet18'
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        ckpt_dataset = checkpoint.get('dataset')
        ckpt_num_classes = infer_num_classes_from_state_dict(state_dict)
        effective_num_classes = ckpt_num_classes or num_classes
        if ckpt_dataset and ckpt_dataset != args.dataset:
            print(f'Warning: requested dataset={args.dataset}, but checkpoint metadata says dataset={ckpt_dataset}.')
        if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
            print(f'Warning: requested num_classes={num_classes}, but checkpoint classifier suggests num_classes={ckpt_num_classes}; using {effective_num_classes} for model reconstruction.')
        model = build_model_from_arch(resolved_arch, num_classes=effective_num_classes,
                                      image_size=image_size, state_dict=state_dict)
        model.load_state_dict(align_resnet_state_dict(state_dict, model), strict=True)
    else:
        model = checkpoint
        resolved_arch = args.model_arch if args.model_arch != 'auto' else 'unknown'
    if hasattr(model, 'conv1'):
        print(f'Model architecture: {resolved_arch}; conv1={model.conv1.kernel_size}, '
              f'stride={model.conv1.stride}; maxpool={type(getattr(model, "maxpool", None)).__name__}', flush=True)
    return (model.to(device), resolved_arch)

def get_discrete_patch_size(img_size, patch_size):
    """Choose a small default patch size so local edits behave like token substitutions."""
    if patch_size > 0:
        return min(patch_size, img_size)
    if img_size <= 32:
        return max(2, img_size // 12)
    return min(8, max(4, img_size // 28))

def build_cell_size_schedule(base_patch_size, cells_per_mutation, preserve_total_area=False):
    """Split one base patch into several smaller square cells while keeping total area close."""
    base_patch_size = max(1, int(base_patch_size))
    cells_per_mutation = max(1, int(cells_per_mutation))
    if cells_per_mutation == 1 or not preserve_total_area:
        return [base_patch_size] * cells_per_mutation
    base_area = base_patch_size * base_patch_size
    sizes = []
    remaining_area = base_area
    for cell_idx in range(cells_per_mutation):
        remaining_cells = cells_per_mutation - cell_idx
        target_area = max(1.0, remaining_area / remaining_cells)
        size = max(1, int(round(math.sqrt(target_area))))
        max_size = max(1, math.isqrt(max(1, remaining_area - (remaining_cells - 1))))
        size = min(size, max_size)
        while size > 1 and remaining_area - size * size < remaining_cells - 1:
            size -= 1
        sizes.append(size)
        remaining_area = max(0, remaining_area - size * size)
    if remaining_area > 0:
        improved = True
        while remaining_area > 0 and improved:
            improved = False
            for idx in np.argsort([-s for s in sizes]):
                gain = 2 * sizes[idx] + 1
                if gain <= remaining_area:
                    sizes[idx] += 1
                    remaining_area -= gain
                    improved = True
                    if remaining_area <= 0:
                        break
    sizes.sort(reverse=True)
    return sizes

def resolve_discrete_patch_size_for_capacity(img_size, patch_size, n_bits):
    """Pick a patch size that can provide at least n_bits distinct mutation locations."""
    chosen = get_discrete_patch_size(img_size, patch_size)
    def capacity(size):
        positions_per_axis = max(1, img_size - size + 1)
        return positions_per_axis * positions_per_axis
    if patch_size > 0:
        cap = capacity(chosen)
        if cap < n_bits:
            raise ValueError(f'mutation_patch_size={chosen} supports at most {cap} unique locations on {img_size}x{img_size} images, which is insufficient for n_bits={n_bits}. Reduce --mutation_patch_size or n_bits.')
        return chosen
    while chosen > 1 and capacity(chosen) < n_bits:
        chosen -= 1
    return chosen

def crop_random_patch(img, patch_size, rng):
    """Crop a patch from a normalized image tensor [C, H, W]."""
    _, h, w = img.shape
    if h == patch_size and w == patch_size:
        return img.clone()
    top = rng.randint(0, h - patch_size)
    left = rng.randint(0, w - patch_size)
    return img[:, top:top + patch_size, left:left + patch_size].clone()

def cells_overlap(top, left, size, other_cell):
    """Return whether a proposed square cell overlaps an existing one."""
    bottom = top + size
    right = left + size
    other_bottom = other_cell['top'] + other_cell['size']
    other_right = other_cell['left'] + other_cell['size']
    return not (bottom <= other_cell['top'] or other_bottom <= top or right <= other_cell['left'] or (other_right <= left))

def sample_image_mutation_spec(dataset, img_size, patch_size, cells_per_mutation, rng=None, preserve_total_area=False, exclude_dataset_indices=None):
    """Sample one mutation spec made of several disjoint small-cell replacements."""
    if rng is None:
        rng = random.Random()
    cells = []
    max_cells = max(1, cells_per_mutation)
    excluded_dataset_indices = {int(idx) for idx in exclude_dataset_indices or []}
    available_dataset_indices = [idx for idx in range(len(dataset)) if idx not in excluded_dataset_indices]
    cell_sizes = build_cell_size_schedule(patch_size, max_cells, preserve_total_area=preserve_total_area)
    max_attempts = max(100, 40 * max_cells)
    for cell_size in cell_sizes:
        placed = False
        local_attempts = 0
        while local_attempts < max_attempts:
            local_attempts += 1
            top = rng.randint(0, img_size - cell_size)
            left = rng.randint(0, img_size - cell_size)
            if any((cells_overlap(top, left, cell_size, existing) for existing in cells)):
                continue
            if not available_dataset_indices:
                raise ValueError('No image mutation donor samples remain after excluding anchors')
            source_idx = int(rng.choice(available_dataset_indices))
            src_img, _ = dataset[source_idx]
            patch = crop_random_patch(src_img, cell_size, rng)
            cell = {'top': top, 'left': left, 'size': cell_size, 'patch': patch.cpu(),
                    'source_idx': source_idx}
            cells.append(cell)
            placed = True
            break
        if not placed:
            break
    if len(cells) < max_cells:
        raise RuntimeError(f'Not enough image mutation cells ({len(cells)}) for cells_per_mutation={max_cells}')
    return {'cells': cells}

def mutation_spec_signature(spec):
    """Build a uniqueness signature for a mutation spec from both location and patch content."""
    signature = []
    cells = spec['cells'] if 'cells' in spec else [spec]
    for cell in cells:
        patch = cell['patch']
        if isinstance(patch, torch.Tensor):
            patch_bytes = patch.detach().cpu().numpy().tobytes()
        else:
            patch_bytes = np.asarray(patch).tobytes()
        patch_hash = hashlib.sha1(patch_bytes).hexdigest()[:16]
        signature.append((cell['top'], cell['left'], cell['size'], patch_hash))
    signature.sort()
    return tuple(signature)

def get_mutation_spec_cells(spec):
    cells = spec.get('cells')
    if cells is None:
        cells = [{'top': spec['top'], 'left': spec['left'], 'size': spec['size'], 'patch': spec['patch']}]
    return cells

def mutation_spec_descriptor(spec):
    """Build a trigger-independent descriptor from cell geometry and patch statistics."""
    cells = []
    for cell in get_mutation_spec_cells(spec):
        patch = cell['patch']
        if isinstance(patch, torch.Tensor):
            patch_t = patch.detach().cpu().float()
        else:
            patch_t = torch.tensor(np.asarray(patch), dtype=torch.float32)
        patch_mean = patch_t.mean(dim=(1, 2)) if patch_t.ndim == 3 else patch_t.mean().view(1)
        patch_std = patch_t.std(dim=(1, 2), unbiased=False) if patch_t.ndim == 3 else patch_t.std(unbiased=False).view(1)
        patch_energy = patch_t.pow(2).mean(dim=(1, 2)).sqrt() if patch_t.ndim == 3 else patch_t.pow(2).mean().sqrt().view(1)
        cells.append((int(cell['top']), int(cell['left']), int(cell['size']), patch_mean.tolist(), patch_std.tolist(), patch_energy.tolist()))
    cells.sort(key=lambda item: (item[0], item[1], item[2]))
    feat = []
    for top, left, size, mean_v, std_v, energy_v in cells:
        feat.extend([float(top), float(left), float(size)])
        feat.extend((float(v) for v in mean_v))
        feat.extend((float(v) for v in std_v))
        feat.extend((float(v) for v in energy_v))
    if not feat:
        return np.zeros((1,), dtype=np.float32)
    vec = np.asarray(feat, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-08:
        vec = vec / norm
    return vec

def apply_image_mutation_spec(x, spec, alpha=0.85, replace_mode='replace'):
    """Apply one discrete image mutation to a normalized trigger image."""
    x_mut = x.clone()
    for cell in get_mutation_spec_cells(spec):
        top = cell['top']
        left = cell['left']
        size = cell['size']
        patch = cell['patch'].to(device=x.device, dtype=x.dtype).unsqueeze(0)
        if replace_mode == 'replace':
            x_mut[:, :, top:top + size, left:left + size] = patch
        else:
            region = x_mut[:, :, top:top + size, left:left + size]
            x_mut[:, :, top:top + size, left:left + size] = (1 - alpha) * region + alpha * patch
    return x_mut

def set_bn_eval(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

def image_scores(model, x, target, specs):
    batch = torch.cat([apply_image_mutation_spec(x, spec, args.mutation_alpha, args.mutation_replace_mode) for spec in specs], dim=0)
    return F.softmax(model(batch), dim=1)[:, int(target)]

def collect_image_scores(model, record):
    model.eval()
    specs = record['mutation_specs']
    with torch.no_grad():
        scores = torch.cat([image_scores(model, record['x_trigger'], record['target_class'], specs[i:i + args.mutation_eval_batch_size]) for i in range(0, len(specs), args.mutation_eval_batch_size)])
    return scores.cpu().tolist()

def choose_anchors(model, dataset, image_size):
    if args.trigger_size > len(dataset):
        raise ValueError('Trigger count exceeds dataset size')
    count = min(len(dataset), max(args.trigger_size * args.trigger_candidate_factor, 200))
    indices = random.sample(range(len(dataset)), count)
    candidates = []
    progress = Progress(count, 'Anchor confidence')
    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            prob = F.softmax(model(image.unsqueeze(0).to(device))[0], dim=0)
            candidates.append({'dataset_idx': idx, 'img': image, 'true_label': int(label), 'target': int(prob.argmax()), 'confidence': float(prob.max())})
            progress.update()
    progress.close()
    filtered = [c for c in candidates if args.trigger_conf_min <= c['confidence'] <= args.trigger_conf_max]
    confidence_fallback = len(filtered) < args.trigger_size
    if confidence_fallback:
        filtered = sorted(candidates, key=lambda c: c['confidence'], reverse=True)
    else:
        midpoint = (args.trigger_conf_min + args.trigger_conf_max) * 0.5
        filtered.sort(key=lambda c: abs(c['confidence'] - midpoint))
    shortlist = filtered[:min(len(filtered), max(args.trigger_size * 12, 48))]
    patch_size = get_discrete_patch_size(image_size, args.mutation_patch_size)
    progress = Progress(len(shortlist), 'Anchor response probes')
    with torch.no_grad():
        for item in shortlist:
            x = item['img'].unsqueeze(0).to(device)
            target = item['target']
            base = float(F.softmax(model(x)[0], dim=0)[target])
            rng = random.Random(1000003 + item['dataset_idx'])
            scores, seen = ([], set())
            for _ in range(max(args.trigger_probe_mutations * 6, 20)):
                spec = sample_image_mutation_spec(dataset, image_size, patch_size, args.mutation_cells, rng=rng, preserve_total_area=args.mutation_preserve_total_area)
                signature = tuple(((c['top'], c['left'], c['size']) for c in spec['cells']))
                if signature in seen:
                    continue
                seen.add(signature)
                scores.append(float(image_scores(model, x, target, [spec])[0]))
                if len(scores) >= args.trigger_probe_mutations:
                    break
            delta = float(np.mean([abs(score - base) for score in scores]))
            item.update(delta=delta, delta_band=image_band(delta, args.trigger_probe_delta_target, args.trigger_probe_delta_tolerance))
            progress.update()
    progress.close()
    in_band = [c for c in shortlist if c['delta_band'] <= 1e-12]
    midpoint = (args.trigger_conf_min + args.trigger_conf_max) * 0.5
    chosen = in_band[:args.trigger_size] if len(in_band) >= args.trigger_size else sorted(shortlist, key=lambda c: (c['delta_band'], abs(c['confidence'] - midpoint)))[:args.trigger_size]
    return chosen

def build_mutations(model, donor_dataset, anchor, image_size, excluded):
    n_bits = args.n_deltas
    count = max(n_bits, n_bits * args.mutation_filter_pool)
    patch_size = resolve_discrete_patch_size_for_capacity(image_size, args.mutation_patch_size, n_bits)
    rng = random.Random(1000 + anchor['group'])
    x = anchor['x_trigger']
    specs, seen = ([], set())
    progress = Progress(count, 'Mutation sampling')
    for _ in range(max(40 * count, 2000)):
        spec = sample_image_mutation_spec(donor_dataset, image_size, patch_size, args.mutation_cells, rng=rng, preserve_total_area=args.mutation_preserve_total_area, exclude_dataset_indices=excluded)
        signature = mutation_spec_signature(spec)
        if signature in seen:
            continue
        seen.add(signature)
        specs.append(spec)
        progress.update()
        if len(specs) == count:
            break
    progress.close()
    if len(specs) < n_bits:
        raise RuntimeError(f'Only {len(specs)} unique mutations generated for {n_bits} bits')
    candidates = []
    progress = Progress(len(specs), 'Mutation scoring')
    model.eval()
    with torch.no_grad():
        base = float(F.softmax(model(x)[0], dim=0)[anchor['target_class']])
        for start in range(0, len(specs), args.mutation_eval_batch_size):
            batch = specs[start:start + args.mutation_eval_batch_size]
            values = image_scores(model, x, anchor['target_class'], batch).cpu().tolist()
            for spec, value in zip(batch, values):
                distortion = abs(value - base)
                candidates.append({'spec': spec, 'score': value, 'distortion': distortion, 'distortion_band': image_band(distortion, args.mutation_distortion_target, args.mutation_distortion_tolerance), 'spec_descriptor': mutation_spec_descriptor(spec)})
            progress.update(len(batch))
    progress.close()
    selected, stats = select_image_candidates(candidates, n_bits, args.mutation_similarity_threshold, args.mutation_spread_weight)
    return ([c['spec'] for c in selected], stats)

def evaluate_accuracy(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            prediction = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((prediction == labels).sum())
            total += len(labels)
    return 100.0 * correct / max(1, total)

class RecursiveImages:
    """The current deterministic recursive OOD image loader, including flat folders."""

    def __init__(self, root, transform):
        root = os.path.abspath(root)
        self.transform = transform
        samples = []
        for extension in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'):
            for suffix in (extension, extension.upper()):
                samples.extend(glob.glob(os.path.join(root, '**', '*' + suffix), recursive=True))
        self.samples = sorted(set(samples))
        if not self.samples:
            raise ValueError(f'No OOD image files under {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.samples[index]) as image:
            return (self.transform(image.convert('RGB')), -1)

def cpu_records(records):
    return [{**r, 'x_trigger': r['x_trigger'].cpu(), 'secret_message': r['secret_message'].cpu()} for r in records]

def main():
    global args, torch, nn, F, tv_models, device
    run_started = time.monotonic()
    args = parse_args()
    validate_training_args(args)
    if args.mutation_cells <= 0 or args.mutation_filter_pool <= 0 or args.trigger_probe_mutations <= 0:
        raise ValueError('Mutation cells, candidate pool and anchor probe count must be positive')
    if args.clean_per_batch <= 0 or args.trigger_candidate_factor <= 0:
        raise ValueError('Clean batch size and anchor candidate factor must be positive')
    if not 0 <= args.trigger_conf_min <= args.trigger_conf_max <= 1 or not 0 <= args.mutation_alpha <= 1:
        raise ValueError('Invalid confidence interval or mutation alpha')
    if args.trigger_source == 'ood' and (not args.ood_data_root):
        raise ValueError('--ood_data_root is required for OOD anchors')
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models
    from torchvision import datasets, transforms
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    run_dir = build_run_dir(args.artifact_root, args.task_name)
    print(f'Artifact directory: {run_dir}', flush=True)
    print(f'Device={device}; selection=minimal; score=target probability; loss=hinge; center={args.wm_center_gradient}', flush=True)
    print('[Loading] model checkpoint', flush=True)
    image_size = 32 if args.dataset == 'cifar10' else 224
    num_classes = {'cifar10': 10, 'imagenet100': 100, 'imagenet': 1000}[args.dataset]
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint and ('model' not in checkpoint):
        checkpoint = {'state_dict': checkpoint}
    args.model_arch = 'resnet18'
    model, architecture = load_model_from_checkpoint(checkpoint, num_classes, image_size, device)
    print('[Loading] image datasets', flush=True)
    mean = (0.4914, 0.4822, 0.4465) if image_size == 32 else (0.485, 0.456, 0.406)
    std = (0.2023, 0.1994, 0.201) if image_size == 32 else (0.229, 0.224, 0.225)
    operations = [] if image_size == 32 else [transforms.Resize(256), transforms.CenterCrop(224)]
    transform = transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    train, evaluation, anchor_dataset, donor_dataset = load_image_datasets(args, transform, datasets)
    if args.trigger_source == 'ood':
        ood_ops = [transforms.Resize(max(image_size + 8, int(round(image_size * 1.25)))), transforms.CenterCrop(image_size)]
        ood_transform = transforms.Compose(ood_ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])
        ood_root = args.ood_data_root
        if args.ood_split and os.path.isdir(os.path.join(ood_root, args.ood_split)):
            ood_root = os.path.join(ood_root, args.ood_split)
        anchor_dataset = RecursiveImages(ood_root, ood_transform)
        donor_dataset = anchor_dataset
    print(f'[Loaded] train={len(train)}, eval={len(evaluation)}, anchors={len(anchor_dataset)}, donors={len(donor_dataset)}', flush=True)
    print(f'Anchor/probe source={args.trigger_source}; donor source=' + ('ood' if args.trigger_source == 'ood' else args.mutation_donor_source), flush=True)
    bits, payload_metadata = build_payload(args, device)
    Progress.start_construction(2 + args.trigger_size * 4)
    anchors = choose_anchors(model, anchor_dataset, image_size)
    excluded = [a['dataset_idx'] for a in anchors] if donor_dataset is anchor_dataset else []
    records = []
    for group, anchor in enumerate(anchors):
        record = {'group': group, 'x_trigger': anchor['img'].unsqueeze(0).to(device), 'target_class': anchor['target'], 'trigger_source_idx': anchor['dataset_idx'], 'trigger_source': args.trigger_source, 'secret_message': bits[group * args.n_deltas:(group + 1) * args.n_deltas].to(device)}
        specs, stats = build_mutations(model, donor_dataset, record, image_size, excluded)
        record['mutation_specs'], record['mutation_order_digest'] = canonicalize_mutation_specs(specs, 1000 + group)
        record['selection_stats'] = stats
        reference_progress = Progress(1, 'Mutation references')
        values = np.asarray(collect_image_scores(model, record), dtype=np.float32)
        record['reference'] = {'mean': float(values.mean()), 'std': float(values.std())}
        records.append(record)
        reference_progress.update()
        reference_progress.close()
    Progress.finish_construction()
    torch.save({'triggers': cpu_records(records), 'verification_config': vars(args), 'global_secret_metadata': payload_metadata}, run_dir / 'trigger_data.pt')
    print(f"Saved trigger metadata: {run_dir / 'trigger_data.pt'}", flush=True)
    accuracy_before = evaluate_accuracy(model, evaluation)
    bundle = verify_groups(records, lambda r: collect_image_scores(model, r), payload_metadata)
    print(f'Initial ACC={accuracy_before:.2f}%', flush=True)
    print_verification(bundle, 'Initial')
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    clean_indices = random.sample(range(len(train)), min(args.n_clean_samples, len(train)))
    clean_rng = random.Random(args.seed + 100003)
    clean_batch_size = min(args.clean_per_batch, len(clean_indices))
    clean_schedule = [[clean_rng.sample(range(len(clean_indices)), clean_batch_size)
                       for _ in range(args.batches_per_epoch)] for _ in range(args.epochs)]
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.0005)
    scheduler = image_lr_scheduler(optimizer, args)
    streak, best_separation = 0, float('-inf')
    training_started = time.monotonic()
    for epoch in range(args.epochs):
        print(f'[Training] epoch {epoch + 1}/{args.epochs}, elapsed={time.monotonic() - training_started:.1f}s', flush=True)
        current_weight, current_margin = image_epoch_objective(args, epoch)
        separation_sum = 0.0
        checkpoint_sum = 0.0
        for batch_index in range(args.batches_per_epoch):
            indices = [clean_indices[i] for i in clean_schedule[epoch][batch_index]]
            examples = [train[i] for i in indices]
            x = torch.stack([item[0] for item in examples]).to(device)
            labels = torch.tensor([item[1] for item in examples], device=device)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            model.apply(set_bn_eval)
            clean_loss = F.cross_entropy(model(x), labels)
            clean_loss.backward()
            model.eval()
            for r in records:
                with torch.no_grad():
                    anchor_preserved = float(int(model(r['x_trigger']).argmax(dim=1)) == r['target_class'])
                controls = backward_group(lambda specs, r=r: image_scores(model, r['x_trigger'], r['target_class'], specs), r['mutation_specs'], r['secret_message'], r['reference'], args, group_count=len(records), watermark_weight=current_weight, margin=current_margin)
                separation_sum += controls['positive_rate']
                checkpoint_sum += controls['positive_rate'] - 0.25 * controls['violation_rate'] + 0.05 * anchor_preserved
            model.train()
            model.apply(set_bn_eval)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        accuracy = evaluate_accuracy(model, evaluation)
        bundle = verify_groups(records, lambda r: collect_image_scores(model, r), payload_metadata)
        current_wsr = bundle['overall']['wsr']
        print(f'Epoch {epoch + 1}/{args.epochs}: ACC={accuracy:.2f}%, elapsed={time.monotonic() - training_started:.1f}s', flush=True)
        print_verification(bundle, f'Epoch {epoch + 1}')
        denominator = args.batches_per_epoch * len(records)
        checkpoint_score = checkpoint_sum / denominator
        if checkpoint_score > best_separation:
            best_separation = checkpoint_score
            torch.save({'arch': architecture, 'dataset': args.dataset, 'state_dict': model.state_dict()}, run_dir / 'model_best_separation.pt')
        streak = streak + 1 if separation_sum / denominator >= args.early_stop_sep_threshold else 0
        if scheduler is not None:
            scheduler.step()
        if args.early_stop_on_separation and streak >= args.early_stop_sep_patience:
            print('Early stop: training separation threshold reached.', flush=True)
            break
    print(f'Training total time: {time.monotonic() - training_started:.1f}s', flush=True)
    torch.save({'arch': architecture, 'dataset': args.dataset, 'state_dict': model.state_dict()}, run_dir / 'model.pt')
    print(f"Finished: ACC={accuracy:.2f}%, WSR_codeword={current_wsr:.2%}, "
          f"WSR_raw_after_inverse={bundle['overall_raw']['wsr']:.2%}; "
          f"total_time={time.monotonic() - run_started:.1f}s; artifacts={run_dir}", flush=True)
if __name__ == '__main__':
    main()
