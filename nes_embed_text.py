"""Compact sequence-text embedding: current minimal selection only."""
import argparse
import glob
import math
import os
import random
import time
from itertools import chain
import numpy as np
from minimal_utils import build_payload, Progress, add_training_args, backward_group, build_run_dir, canonicalize_mutation_specs, print_verification, seed_everything, select_text_candidates, text_loss_window, validate_training_args, verify_groups
VOCAB_POOL = list(range(100, 20000))

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser, text=True)
    parser.add_argument('--model_path', default='./models/base/GPT-2/')
    parser.add_argument('--data_path', default='./data/PTB/')
    parser.add_argument('--ood_data_path', default=None)
    parser.add_argument('--ood_split', default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--trigger_source', choices=['train', 'validation', 'eval', 'ood'], default='ood')
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ft_epochs', type=int, default=3)
    parser.add_argument('--pool_factor', type=int, default=10)
    parser.add_argument('--loss_trim_low_ratio', type=float, default=0.1)
    parser.add_argument('--loss_trim_high_ratio', type=float, default=0.2)
    parser.add_argument('--trigger_candidate_factor', type=int, default=80)
    parser.add_argument('--text_mutation_ratio', type=float, default=0.05)
    parser.add_argument('--wm_score_std_cap', type=float, default=2.0)
    parser.add_argument('--eval_num_samples', type=int, default=None, help='Optional maximum sliding windows for PPL (default: all)')
    return parser.parse_args()

def load_local_text(path):
    files = {s: os.path.join(path, f'ptb.{name}.txt') for s, name in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]}
    if all((os.path.isfile(p) for p in files.values())):
        return load_dataset('text', data_files=files, keep_linebreaks=True)
    parquet = {s: sorted(glob.glob(os.path.join(path, f'{s}-*.parquet'))) for s in ['train', 'validation', 'test']}
    if parquet['train'] and parquet['validation']:
        return load_dataset('parquet', data_files={s: files for s, files in parquet.items() if files})
    raise FileNotFoundError(f'Expected local PTB train/valid/test files or WikiText parquet shards under {path}')

def resolve_stride(block_size, stride=None):
    """Resolve an adaptive sliding-window stride from block size."""
    if stride is None:
        return max(1, block_size // 2)
    return stride

def build_lm_datasets(raw_datasets, tokenizer, block_size, stride=None):
    """Tokenize and group texts into fixed-size blocks, optionally with sliding-window overlap."""
    column_names = raw_datasets['train'].column_names
    text_column = 'text' if 'text' in column_names else column_names[0]
    stride = resolve_stride(block_size, stride)
    if stride <= 0:
        raise ValueError(f'stride must be positive, got {stride}')
    if stride > block_size:
        print(f'Requested stride {stride} is larger than block_size {block_size}; clamping to block_size.', flush=True)
        stride = block_size

    def tokenize_function(examples):
        return tokenizer(examples[text_column])
    tokenized = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names, desc='Tokenizing')

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        input_ids = concatenated['input_ids']
        total_length = len(input_ids)
        if total_length < block_size:
            return {k: [] for k in concatenated.keys()} | {'labels': []}
        start_positions = list(range(0, total_length - block_size + 1, stride))
        if start_positions[-1] != total_length - block_size:
            start_positions.append(total_length - block_size)
        result = {k: [t[i:i + block_size] for i in start_positions] for k, t in concatenated.items()}
        result['labels'] = result['input_ids'].copy()
        return result
    lm_datasets = tokenized.map(group_texts, batched=True, desc=f'Grouping into sliding windows of {block_size} (stride={stride})')
    return lm_datasets

def build_continuous_eval_tokens(raw_split, tokenizer):
    """Flatten an eval split into one continuous token stream for strided perplexity."""
    column_names = raw_split.column_names
    text_column = 'text' if 'text' in column_names else column_names[0]
    all_token_ids = []
    for text in raw_split[text_column]:
        token_ids = tokenizer(text)['input_ids']
        if token_ids:
            all_token_ids.extend(token_ids)
    return torch.tensor(all_token_ids, dtype=torch.long)

def sample_mutation_spec(input_ids, rng, mutation_ratio=0.05):
    """Sample a mutation spec: positions + replacement token ids."""
    seq_len = input_ids.shape[1]
    k = max(1, int(round(seq_len * mutation_ratio)))
    k = min(k, seq_len)
    positions = rng.choice(seq_len, size=k, replace=False)
    replacements = rng.choice(VOCAB_POOL, size=k)
    return {'positions': [int(p) for p in positions.tolist()], 'replacements': [int(r) for r in replacements.tolist()]}

def apply_mutation_specs_batch(input_ids, specs):
    """Apply many mutation specs to one sequence and return a batched tensor."""
    base = input_ids.repeat(len(specs), 1)
    for row_idx, spec in enumerate(specs):
        positions = torch.tensor(spec['positions'], dtype=torch.long, device=input_ids.device)
        replacements = torch.tensor(spec['replacements'], dtype=input_ids.dtype, device=input_ids.device)
        base[row_idx, positions] = replacements
    return base

def seq_prob_score_batch(logits, input_ids):
    """Batch version that scores a sequence with the mean true-token probability."""
    next_token_probs = F.softmax(logits[:, :-1, :], dim=-1)
    next_token_ids = input_ids[:, 1:]
    scores = next_token_probs.gather(2, next_token_ids.unsqueeze(-1)).squeeze(-1)
    return scores.mean(dim=1)

def causal_lm_loss_per_sequence(logits, input_ids):
    """Compute per-sequence causal LM loss without reducing across the batch."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    token_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none').view(shift_labels.shape)
    return token_losses.mean(dim=1)

def text_mutation_feature_vector(model, input_ids, spec):
    """Return a normalized model-input embedding change for one mutation."""
    positions = [int(p) for p in spec.get('positions', [])]
    replacements = [int(t) for t in spec.get('replacements', [])]
    if not positions or len(positions) != len(replacements):
        return None
    get_embeddings = getattr(model, 'get_input_embeddings', None)
    if get_embeddings is None:
        return None
    embedding = get_embeddings()
    if embedding is None:
        return None
    device = input_ids.device
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    replacement_tensor = torch.tensor(replacements, dtype=torch.long, device=device)
    with torch.no_grad():
        original_vectors = embedding(input_ids[0, position_tensor])
        replacement_vectors = embedding(replacement_tensor)
        vector = (replacement_vectors - original_vectors).mean(dim=0)
    vector = vector.detach().float().cpu().numpy()
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-08:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)

def text_scores(model, input_ids, specs):
    batch = apply_mutation_specs_batch(input_ids, specs)
    return seq_prob_score_batch(model(input_ids=batch, use_cache=False).logits, batch)

def collect_text_scores(model, record):
    model.eval()
    specs = record['mutation_specs']
    with torch.no_grad():
        scores = torch.cat([text_scores(model, record['trigger_input'], specs[i:i + args.mutation_eval_batch_size]) for i in range(0, len(specs), args.mutation_eval_batch_size)])
    return scores.cpu().tolist()

def choose_anchors(dataset):
    if args.trigger_size > len(dataset):
        raise ValueError('Trigger count exceeds text block count')
    count = min(len(dataset), max(args.trigger_size * args.trigger_candidate_factor, args.trigger_size))
    indices = random.sample(range(len(dataset)), count)[:args.trigger_size]
    records = []
    for group, index in enumerate(indices):
        records.append({'group': group, 'trigger_source_idx': index, 'trigger_input': dataset[index]['input_ids'].unsqueeze(0).to(device)})
    return records

def build_mutations(model, record):
    rng = np.random.RandomState(2026 + record['group'] * 100003)
    count = max(args.n_deltas, args.n_deltas * args.pool_factor)
    specs, seen = ([], set())
    progress = Progress(count, 'Mutation sampling')
    for _ in range(max(10 * count, 1000)):
        spec = sample_mutation_spec(record['trigger_input'], rng, args.text_mutation_ratio)
        key = (tuple(spec['positions']), tuple(spec['replacements']))
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
        progress.update()
        if len(specs) == count:
            break
    progress.close()
    if len(specs) < args.n_deltas:
        raise RuntimeError(f'Only {len(specs)} unique mutations for {args.n_deltas} bits')
    model.eval()
    candidates = []
    progress = Progress(len(specs), 'Mutation scoring')
    with torch.no_grad():
        for start in range(0, len(specs), args.mutation_eval_batch_size):
            batch_specs = specs[start:start + args.mutation_eval_batch_size]
            batch = apply_mutation_specs_batch(record['trigger_input'], batch_specs)
            logits = model(input_ids=batch, use_cache=False).logits
            losses = causal_lm_loss_per_sequence(logits, batch).cpu().tolist()
            for spec, loss in zip(batch_specs, losses):
                candidates.append({'spec': spec, 'lm_loss': loss, 'feature_vector': text_mutation_feature_vector(model, record['trigger_input'], spec)})
            progress.update(len(batch_specs))
    progress.close()
    window, stats = text_loss_window(candidates, args.n_deltas, args.loss_trim_low_ratio, args.loss_trim_high_ratio)
    selected, diversity = select_text_candidates(window, args.n_deltas, args.mutation_similarity_threshold)
    stats.update(diversity)
    return ([c['spec'] for c in selected], stats)

def evaluate_ppl(model, token_ids):
    model.eval()
    total_nll, total_count, previous_end, windows = (0.0, 0, 0, 0)
    stride = min(args.block_size, resolve_stride(args.block_size, args.stride))
    with torch.no_grad():
        for begin in range(0, len(token_ids), stride):
            end = min(begin + args.block_size, len(token_ids))
            inputs = token_ids[begin:end].unsqueeze(0).to(device)
            if inputs.shape[1] < 2:
                break
            labels = inputs.clone()
            new_tokens = end - previous_end
            if new_tokens <= 0:
                break
            if new_tokens < labels.shape[1]:
                labels[:, :-new_tokens] = -100
            count = int((labels[:, 1:] != -100).sum())
            if count:
                output = model(input_ids=inputs, labels=labels, use_cache=False)
                total_nll += float(output.loss) * count
                total_count += count
            previous_end = end
            windows += 1
            if end == len(token_ids) or (args.eval_num_samples is not None and windows >= args.eval_num_samples):
                break
    if not total_count:
        raise ValueError('Evaluation text has no valid next-token positions')
    return math.exp(total_nll / total_count)

def save_triggers(records, run_dir, payload_metadata):
    saved = [{**r, 'trigger_input': r['trigger_input'].cpu(), 'secret_message': r['secret_message'].cpu()} for r in records]
    torch.save({'n_bits': args.n_deltas, 'triggers': saved, 'verification_config': vars(args), 'global_secret_metadata': payload_metadata}, run_dir / 'trigger_data_gpt2.pt')
    print(f"Saved trigger metadata: {run_dir / 'trigger_data_gpt2.pt'}", flush=True)

def main():
    global args, torch, F, device, load_dataset
    run_started = time.monotonic()
    args = parse_args()
    validate_training_args(args)
    if args.ft_epochs < 0 or args.block_size < 2 or args.batch_size <= 0 or (args.pool_factor <= 0):
        raise ValueError('Invalid text training or block configuration')
    if args.trigger_candidate_factor <= 0 or not 0 < args.text_mutation_ratio <= 1:
        raise ValueError('Invalid anchor candidate factor or token mutation ratio')
    if args.stride is not None and args.stride <= 0:
        raise ValueError('--stride must be positive')
    if not 0 <= args.loss_trim_low_ratio < 1 or not 0 <= args.loss_trim_high_ratio < 1:
        raise ValueError('LM trim ratios must be in [0,1)')
    if args.loss_trim_low_ratio + args.loss_trim_high_ratio >= 1:
        raise ValueError('LM trim ratios must leave a nonempty window')
    if args.eval_num_samples is not None and args.eval_num_samples <= 0:
        raise ValueError('--eval_num_samples must be positive')
    if args.trigger_source == 'ood' and (not args.ood_data_path):
        raise ValueError('--ood_data_path is required for OOD anchors')
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    run_dir = build_run_dir(args.artifact_root, args.task_name)
    print(f'Artifact directory: {run_dir}', flush=True)
    print(f'Device={device}; selection=minimal; score=mean next-token probability; loss=hinge; center={args.wm_center_gradient}', flush=True)
    print('[Loading] tokenizer and language model', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True).to(device)
    if model.get_input_embeddings().num_embeddings < 20000:
        raise ValueError('The current token mutation pool requires a vocabulary of at least 20000 tokens')
    if args.block_size > getattr(model.config, 'max_position_embeddings', args.block_size):
        raise ValueError('Block size exceeds the model context length')
    print('[Loading] text datasets', flush=True)
    raw = load_local_text(args.data_path)
    blocks = build_lm_datasets(raw, tokenizer, args.block_size, args.stride)
    blocks.set_format(type='torch', columns=['input_ids', 'labels'])
    train = blocks['train']
    split = 'validation' if 'validation' in blocks else 'test'
    evaluation = blocks[split]
    token_ids = build_continuous_eval_tokens(raw[split], tokenizer)
    anchor_dataset = train if args.trigger_source == 'train' else evaluation
    if args.trigger_source == 'ood':
        ood_raw = load_local_text(args.ood_data_path)
        ood_blocks = build_lm_datasets(ood_raw, tokenizer, args.block_size, args.stride)
        ood_blocks.set_format(type='torch', columns=['input_ids', 'labels'])
        ood_split = args.ood_split if args.ood_split in ood_blocks else 'validation' if 'validation' in ood_blocks else 'test'
        anchor_dataset = ood_blocks[ood_split]
    if not len(train) or not len(evaluation):
        raise ValueError('No complete train/eval text blocks; reduce --block_size or check data')
    print(f'[Loaded] train={len(train)} blocks, eval({split})={len(evaluation)}, anchors={len(anchor_dataset)}', flush=True)
    records = choose_anchors(anchor_dataset)
    ppl_before = evaluate_ppl(model, token_ids)
    print(f'Initial PPL={ppl_before:.2f}', flush=True)
    if args.ft_epochs:
        loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-05, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, min(150, len(loader)) * args.ft_epochs), eta_min=5e-06)
        domain_started = time.monotonic()
        for epoch in range(args.ft_epochs):
            print(f'[Domain fine-tuning] epoch {epoch + 1}/{args.ft_epochs}, elapsed={time.monotonic() - domain_started:.1f}s', flush=True)
            model.train()
            for index, batch in enumerate(loader):
                inputs, labels = (batch['input_ids'].to(device), batch['labels'].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if index + 1 >= 150:
                    break
            print(f'Domain epoch {epoch + 1}/{args.ft_epochs}: PPL={evaluate_ppl(model, token_ids):.2f}, elapsed={time.monotonic() - domain_started:.1f}s', flush=True)
        print(f'Domain fine-tuning total time: {time.monotonic() - domain_started:.1f}s', flush=True)
    bits, payload_metadata = build_payload(args, device)
    construction = Progress.start_construction(1 + args.trigger_size * 4)
    construction.update(100)  # Anchors were selected before domain fine-tuning, as in the experiment entry.
    for group, record in enumerate(records):
        specs, stats = build_mutations(model, record)
        record['mutation_specs'], record['mutation_order_digest'] = canonicalize_mutation_specs(specs, 2026 + group * 100003)
        record['secret_message'] = bits[group * args.n_deltas:(group + 1) * args.n_deltas].to(device)
        record['trigger_source'] = args.trigger_source
        record['selection_stats'] = stats
        reference_progress = Progress(1, 'Mutation references')
        values = np.asarray(collect_text_scores(model, record), dtype=np.float32)
        record['reference'] = {'mean': float(values.mean()), 'std': float(values.std())}
        reference_progress.update()
        reference_progress.close()
    Progress.finish_construction()
    save_triggers(records, run_dir, payload_metadata)
    bundle = verify_groups(records, lambda r: collect_text_scores(model, r), payload_metadata)
    print_verification(bundle, 'Initial')
    clean_indices = random.sample(range(len(train)), min(args.n_clean_samples, len(train)))
    clean_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train, clean_indices), batch_size=args.batch_size, shuffle=True,
                                             generator=torch.Generator().manual_seed(args.seed + 100003))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    streak, best_separation = 0, float('-inf')
    training_started = time.monotonic()
    for epoch in range(args.epochs):
        print(f'[Training] epoch {epoch + 1}/{args.epochs}, elapsed={time.monotonic() - training_started:.1f}s', flush=True)
        iterator = iter(clean_loader)
        separation_sum = 0.0
        checkpoint_sum = 0.0
        for _ in range(args.batches_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(clean_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            clean_loss = model(input_ids=batch['input_ids'].to(device), labels=batch['labels'].to(device), use_cache=False).loss
            clean_loss.backward()
            model.eval()
            for r in records:
                controls = backward_group(lambda specs, r=r: text_scores(model, r['trigger_input'], specs), r['mutation_specs'], r['secret_message'], r['reference'], args, text=True, group_count=len(records))
                separation_sum += controls['positive_rate']
                checkpoint_sum += controls['positive_rate'] - 0.25 * controls['violation_rate']
            model.train()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        ppl = evaluate_ppl(model, token_ids)
        bundle = verify_groups(records, lambda r: collect_text_scores(model, r), payload_metadata)
        current_wsr = bundle['overall']['wsr']
        print(f'Epoch {epoch + 1}/{args.epochs}: PPL={ppl:.2f}, elapsed={time.monotonic() - training_started:.1f}s', flush=True)
        print_verification(bundle, f'Epoch {epoch + 1}')
        denominator = args.batches_per_epoch * len(records)
        checkpoint_score = checkpoint_sum / denominator
        if checkpoint_score > best_separation:
            best_separation = checkpoint_score
            model.save_pretrained(run_dir / 'model_best_separation')
        streak = streak + 1 if separation_sum / denominator >= args.early_stop_sep_threshold else 0
        if args.early_stop_on_separation and streak >= args.early_stop_sep_patience:
            print('Early stop: training separation threshold reached.', flush=True)
            break
    print(f'Training total time: {time.monotonic() - training_started:.1f}s', flush=True)
    model.save_pretrained(run_dir / 'model')
    tokenizer.save_pretrained(run_dir / 'model')
    tokenizer.save_pretrained(run_dir / 'model_best_separation')
    print(f"Finished: PPL={ppl:.2f}, WSR_codeword={current_wsr:.2%}, "
          f"WSR_raw_after_inverse={bundle['overall_raw']['wsr']:.2%}; "
          f"total_time={time.monotonic() - run_started:.1f}s; artifacts={run_dir}", flush=True)
if __name__ == '__main__':
    main()
