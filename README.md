# Watermark embedding and verification

Image/text watermark embedding with ACC (images), PPL (text), and WSR evaluation. Keep the four Python files together. Use Python 3.10+:

```bash
pip install -r requirements.txt
```

Supply models and datasets locally. ResNet-18 loading supports the original 3x3-stem backbone (`shortcut`/`linear`) and torchvision's 7x7-stem backbone (`downsample`/`fc`), with strict checkpoint loading. Other implementations can use `--arch_path ./models/resnet.py` providing `resnet18(num_classes=...)`. Text accepts local GPT-2, PTB train/valid/test files, or WikiText parquet shards. All paths below are generic relative examples.

## Model sources

The compact image script supports ResNet-18, and the text script uses GPT-2. The following links document the architecture references and pretrained starting point; the ViT references apply to the original experiment code rather than an entry point included in this compact package.

| Model or reference | Source |
| --- | --- |
| ResNet-18 architecture reference | [Huawei DAFL implementation](https://github.com/huawei-noah/Efficient-Computing/blob/master/Data-Efficient-Model-Compression/DAFL/resnet.py) |
| ViT architecture reference | [vit-pytorch: vit.py](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py) |
| ViT small-dataset architecture reference | [vit-pytorch: vit_for_small_dataset.py](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py) |
| GPT-2 pretrained starting point | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) |

Model training and task-specific fine-tuning are described in the paper's Experimental Setup and Training details sections. The scripts load the local models specified by `--model_path`.

## Dataset sources

Obtain the datasets from the sources below and prepare the local paths used in the example commands. These are access references; use the data version and preprocessing matching the experiment being reproduced.

| Dataset | Source | Local input |
| --- | --- | --- |
| CIFAR-10 | [Official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) | Extracted CIFAR-10 data under `--data_root` |
| ImageNet / ImageNet-100 | [Official ImageNet download page](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) | An ImageFolder-style 100-class subset; sampling is described in the paper |
| Places365 | [Authors' project and dataset links](https://github.com/CSAILVision/places365) | OOD image directory passed through `--ood_data_root` |
| Penn Treebank | [PTB text dataset page](https://huggingface.co/datasets/ptb-text-only/ptb_text_only) | `ptb.train.txt`, `ptb.valid.txt`, and `ptb.test.txt` under `--data_path` or `--ood_data_path` |
| WikiText-2 | [Salesforce WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext) | Local WikiText-2 train/validation/test parquet shards under `--data_path` or `--ood_data_path` |

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
