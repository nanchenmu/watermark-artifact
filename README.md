# Watermark embedding and verification

Image/text watermark embedding with ACC (images), PPL (text), and WSR evaluation. Keep the four Python files together. Use Python 3.10+:

```bash
pip install -r requirements.txt
```

Supply models and datasets locally. ResNet-18 loading supports the original 3x3-stem backbone (`shortcut`/`linear`) and torchvision's 7x7-stem backbone (`downsample`/`fc`), with strict checkpoint loading. Other implementations can use `--arch_path ./models/resnet.py` providing `resnet18(num_classes=...)`. Text accepts local GPT-2, PTB train/valid/test files, or WikiText parquet shards. All paths below are generic relative examples.

## ImageNet100

```bash
python -u nes_embed_image.py \
  --data_root ./data/ImageNet100 \
  --model_path ./models/base_model_resnet18_imagenet100.pt \
  --ood_data_root ./data/places365 \
  --trigger_size 4 --n_deltas 256 \
  --lr 2e-5 --lambda_wm 1 \
  --gpu 0
```

## CIFAR-10

```bash
python -u nes_embed_image.py \
  --dataset cifar10 --data_root ./data \
  --model_path ./models/base_model_resnet18_cifar10.pt \
  --ood_data_root ./data/ImageNet100 \
  --trigger_size 4 --n_deltas 256 \
  --lr 8e-6 --lambda_wm 0.5 --clean_per_batch 64 \
  --gpu 0
```

## Text (PTB)

```bash
python -u nes_embed_text.py \
  --model_path ./models/GPT-2 --data_path ./data/PTB \
  --ood_data_path ./data/wikitext-2-v1 \
  --trigger_size 4 --n_deltas 256 \
  --lr 5e-5 --lambda_wm 5 \
  --gpu 0
```

## Text (WikiText-2)

```bash
python -u nes_embed_text.py \
  --model_path ./models/GPT-2 --data_path ./data/wikitext-2-v1 \
  --ood_data_path ./data/PTB \
  --trigger_size 4 --n_deltas 256 \
  --lr 5e-5 --lambda_wm 5 \
  --gpu 0
```

`dataset` selects the task dataset. Both scripts default to OOD anchors. Images recursively load all image files under `ood_data_root`, including train/validation/test subdirectories; both anchors and mutation donors are selected from this pool. Leave `ood_split` unset for this behavior. ImageNet task ACC defaults to `test/`; use `--val_split validation` to select `validation/` or `val/` instead. CIFAR clean training/ACC use official train/test. Text OOD anchors use validation by default.

Use `--help` to view and override the supported parameters.

Payload and XOR-mask seeds have reproducible demo defaults; use `--secret_seed_base` and `--whitening_seed` to supply your own values. Whitening uses XOR only, without a position permutation. WSR is reported both against the embedded codeword and after inverse whitening against the original payload.

For a text-image payload, fonts are located by Pillow using standard font filenames, or supplied with `--secret_font_path`.

## Outputs

Runs save under timestamped `NES/runs/` directories:

- Images: `model.pt`, `model_best_separation.pt`, `trigger_data.pt`.
- Text: `model/`, `model_best_separation/`, `trigger_data_gpt2.pt`.

Trigger files contain private anchors, mutations, bits, whitening information, and the runtime configuration (including supplied paths). Keep generated models, trigger files, logs, datasets, and caches out of the submission package. The distributed files contain code, documentation, and dependencies only.
