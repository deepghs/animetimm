<div align="center">

# AnimeTIMM

**TIMM training, evaluation, export, and Hub publishing toolkit for anime-style image models.**

[![GitHub](https://img.shields.io/badge/GitHub-deepghs%2Fanimetimm-181717?logo=github)](https://github.com/deepghs/animetimm)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Models-animetimm-yellow)](https://huggingface.co/animetimm)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Datasets-animetimm-yellow)](https://huggingface.co/datasets?author=animetimm)
[![DeepGHS](https://img.shields.io/badge/DeepGHS-GitHub-24292f)](https://github.com/deepghs)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

</div>

AnimeTIMM is the internal training and release pipeline used by **DeepGHS** to turn
[TIMM](https://github.com/huggingface/pytorch-image-models) image backbones into anime-domain taggers,
classifiers, ONNX packages, Hugging Face model cards, demo Spaces, and dataset/model indexes.

The repository is intentionally lightweight: most heavyweight artifacts live on Hugging Face under the
[`animetimm`](https://huggingface.co/animetimm) and [`deepghs`](https://huggingface.co/deepghs) namespaces.
This codebase contains the repeatable Python pieces for dataset loading, augmentation, training,
metric calculation, export, and model-card generation.

> **Audience note:** this project is mostly for research, data engineering, and model publishing workflows around
> anime-style image understanding. Many linked datasets/models are NSFW-capable or gated; follow the license and
> access policy of each dataset/model card before use.

---

## What This Repo Does

| Area | Modules | What it provides |
|---|---|---|
| Multi-label taggers | `animetimm.multilabel.*` | Train WDTagger-style anime taggers with sigmoid outputs, per-tag/category thresholds, ONNX export, and `dghs-imgutils` inference snippets. |
| Single-label classifiers | `animetimm.classification.*` | Train single-label anime classifiers, such as character or artist classifiers, with top-k/F1 reports. |
| TIMM model wrapper | `animetimm.model`, `animetimm.wrap` | Save/load TIMM models with tags and metadata; expose embeddings, logits, and predictions for ONNX. |
| Augmentations | `animetimm.augmentation` | Random resize interpolation, cutout, mixup, probabilistic greyscale, and training/eval transform builders. |
| Hub publishing | `animetimm.*.export` | Export Hugging Face TIMM-format repositories with `model.safetensors`, `pytorch_model.bin`, `model.onnx`, `preprocess.json`, `metrics.json`, `selected_tags.csv`, and model cards. |
| TIMM index | `tools/timm/index.py` | Builds the [`deepghs/timms_index`](https://huggingface.co/datasets/deepghs/timms_index) dataset from upstream TIMM ImageNet results and model metadata. |

---

## Ecosystem Snapshot

The public `animetimm` Hugging Face org contains anime-domain TIMM models and datasets produced by this pipeline.
At the time of the latest metadata check for this README (2026-06-01), it exposed:

- **32 model repositories** under [`animetimm`](https://huggingface.co/animetimm), mostly TIMM-based image-classification/tagging models.
- **13 dataset repositories** under [`animetimm`](https://huggingface.co/datasets?author=animetimm), including Danbooru, Zerochan, e621, and Pexels-style WebDataset releases.
- **6 Spaces** under [`animetimm`](https://huggingface.co/spaces?author=animetimm), including playgrounds and ranklists for released tagger families.
- [`deepghs/timms_index`](https://huggingface.co/datasets/deepghs/timms_index), an indexed table of **1,089** upstream TIMM pretrained weights grouped by model size level.

Popular public model families include:

| Model | Dataset | Type | Classes / tags | Params | Test metric snapshot |
|---|---|---:|---:|---:|---:|
| [`animetimm/convnextv2_huge.dbv4-full`](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) | `danbooru-wdtagger-v4` | multi-label | 12,476 | 692.6M | micro-F1 0.697 |
| [`animetimm/eva02_large_patch14_448.dbv4-full`](https://huggingface.co/animetimm/eva02_large_patch14_448.dbv4-full) | `danbooru-wdtagger-v4` | multi-label | 12,476 | 316.8M | micro-F1 0.693 |
| [`animetimm/caformer_b36.dbv4-full`](https://huggingface.co/animetimm/caformer_b36.dbv4-full) | `danbooru-wdtagger-v4` | multi-label | 12,476 | 134.0M | micro-F1 0.689 |
| [`animetimm/swinv2_base_window8_256.dbv4-full`](https://huggingface.co/animetimm/swinv2_base_window8_256.dbv4-full) | `danbooru-wdtagger-v4` | multi-label | 12,476 | 99.7M | micro-F1 0.683 |
| [`animetimm/mobilenetv4_conv_aa_large.dbv4-full`](https://huggingface.co/animetimm/mobilenetv4_conv_aa_large.dbv4-full) | `danbooru-wdtagger-v4` | multi-label | 12,476 | 47.3M | micro-F1 0.641 |
| [`animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls`](https://huggingface.co/animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls) | `danbooru-wdtagger-v4a-fullxx-cls` | single-label | 9,453 | 96.6M | top-1 0.904 / top-5 0.953 |

A private extended example, [`animetimm/swinv2_base_window8_256.dbv4ex-full`](https://huggingface.co/animetimm/swinv2_base_window8_256.dbv4ex-full), demonstrates the same packaging format with **19,774** tags over `general`, `character`, and `rating` categories.

---

## Datasets Used by AnimeTIMM

### Danbooru WDTagger V4 WebDataset

[`animetimm/danbooru-wdtagger-v4-w640-ws-full`](https://huggingface.co/datasets/animetimm/danbooru-wdtagger-v4-w640-ws-full)
contains **5,914,596** resized anime images (`min(width, height) <= 640`) with metadata cleaned by
[@SmilingWolf](https://huggingface.co/SmilingWolf). It is intended for multi-label tagger training.

| Split | Images | Size |
|---|---:|---:|
| train | 5,321,713 | 318 GB |
| test | 295,926 | 17.7 GB |
| val | 296,957 | 17.8 GB |

Selected tags: **12,476** total — **9,225** general tags, **3,247** character tags, and **4** rating tags.

### Zerochan Single-Character WebDataset

[`animetimm/zerochan-character-w640-ws-m200-100k`](https://huggingface.co/datasets/animetimm/zerochan-character-w640-ws-m200-100k)
is a **99,120-image** single-character subset for single-label anime character classification. Images are guaranteed
non-monochrome, single-person, single-headed, single-faced, and have one primary character. Face/head/person bounding
boxes are included in the JSON metadata.

| Split | Images | Size |
|---|---:|---:|
| train | 79,296 | 4.38 GB |
| test | 9,912 | 549 MB |
| val | 9,912 | 545 MB |

Selected labels: **1,652** characters.

### TIMM Metadata Index

[`deepghs/timms_index`](https://huggingface.co/datasets/deepghs/timms_index) is generated by `tools/timm/index.py`.
It indexes upstream [`timm/*`](https://huggingface.co/timm) model repositories by architecture, parameter count,
input size, ImageNet top-1/top-5, downloads, likes, and a practical size level from `nano` to `colossal`.

---

## Installation

This repository is not currently packaged as a PyPI distribution. Use it as a source checkout:

```bash
git clone https://github.com/deepghs/animetimm.git
cd animetimm
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA environment before installing the remaining
requirements.

Core dependencies include PyTorch, torchvision, TIMM, Hugging Face Hub/Datasets, Accelerate, pandas, ONNX Runtime,
`safetensors`, TensorBoard, W&B, and [`dghs-imgutils`](https://github.com/deepghs/imgutils).

---

## Quick Inference with a Released Model

Released AnimeTIMM models are normal Hugging Face TIMM models. Each model repo typically includes:

- `model.safetensors` and `pytorch_model.bin` weights
- `config.json` in TIMM/HF format
- `preprocess.json` with reproducible validation/test transforms
- `selected_tags.csv` with labels, per-tag metrics, and optional best thresholds
- `categories.json` and `thresholds.csv` for multi-label taggers
- `model.onnx` when ONNX export is enabled
- `metrics.json`, `meta.json`, TensorBoard logs, and `sample.webp`

```python
import json

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from imgutils.preprocess import create_torchvision_transforms
from timm import create_model

repo_id = "animetimm/swinv2_base_window8_256.dbv4-full"

model = create_model(f"hf-hub:{repo_id}", pretrained=True)
model.eval()

with open(hf_hub_download(repo_id=repo_id, filename="preprocess.json"), "r") as f:
    preprocessor = create_torchvision_transforms(json.load(f)["test"])

image = load_image(f"https://huggingface.co/{repo_id}/resolve/main/sample.webp")
input_ = preprocessor(image).unsqueeze(0)

with torch.no_grad():
    logits = model(input_)
    prediction = torch.sigmoid(logits)[0]

df_tags = pd.read_csv(
    hf_hub_download(repo_id=repo_id, filename="selected_tags.csv"),
    keep_default_na=False,
)
threshold = df_tags["best_threshold"] if "best_threshold" in df_tags else 0.4
mask = prediction.cpu().numpy() >= threshold
print(dict(zip(df_tags.loc[mask, "name"].tolist(), prediction[mask].cpu().tolist())))
```

For ONNX-backed convenience inference, install `dghs-imgutils` and use its `multilabel_timm_predict` helper:

```python
from imgutils.generic import multilabel_timm_predict

general, character, rating = multilabel_timm_predict(
    "image.webp",
    repo_id="animetimm/swinv2_base_window8_256.dbv4-full",
    fmt=("general", "character", "rating"),
)
```

---

## Training Workflows

### Multi-label Tagger Training

```bash
python -m animetimm.multilabel.train \
  --dataset-repo-id animetimm/danbooru-wdtagger-v4-w640-ws-full \
  --model-name caformer_s36.sail_in22k_ft_in1k_384 \
  --batch-size 32 \
  --num-workers 32 \
  --max-epochs 100 \
  --learning-rate 2e-4 \
  --eval-threshold 0.4 \
  --workdir runs/caformer_s36_dbv4
```

Then calculate test metrics and export to Hugging Face:

```bash
python -m animetimm.multilabel.test \
  --workdir runs/caformer_s36_dbv4 \
  --batch-size 32 \
  --num-workers 32 \
  --test-threshold 0.4

python -m animetimm.multilabel.export \
  --workdir runs/caformer_s36_dbv4 \
  --repository animetimm/caformer_s36.dbv4-full \
  --visibility public \
  --license gpl-3.0
```

### Single-label Classification Training

```bash
python -m animetimm.classification.train \
  --dataset-repo-id animetimm/zerochan-character-w640-ws-m200-100k \
  --tag-key character \
  --model-name caformer_s36.sail_in22k_ft_in1k_384 \
  --batch-size 32 \
  --num-workers 32 \
  --max-epochs 100 \
  --workdir runs/zerochan_character_caformer_s36
```

Export is similar:

```bash
python -m animetimm.classification.export \
  --workdir runs/zerochan_character_caformer_s36 \
  --repository animetimm/caformer_s36.zerochan-character \
  --visibility public \
  --license gpl-3.0
```

---

## Dataset Loading API

Multi-label datasets expose image tensors and multi-hot label vectors:

```python
from timm import create_model
from animetimm.multilabel.dataset import load_dataloader, load_tags

repo_id = "animetimm/danbooru-wdtagger-v4-w640-ws-full"
model = create_model("caformer_s36.sail_in22k_ft_in1k_384", pretrained=False)
tags = load_tags(repo_id)
loader = load_dataloader(repo_id, model=model, split="train", batch_size=32)
```

Single-label datasets expose integer class ids:

```python
from timm import create_model
from animetimm.classification.dataset import load_dataloader, load_tags

repo_id = "animetimm/zerochan-character-w640-ws-m200-100k"
model = create_model("caformer_s36.sail_in22k_ft_in1k_384", pretrained=False)
tags = load_tags(repo_id)
loader = load_dataloader(repo_id, model=model, tag_key="character", split="train", batch_size=32)
```

---

## Export Format

AnimeTIMM export creates model repositories that are easy to consume from both research Python and production runtimes:

```text
README.md              # generated model card with metrics, thresholds, examples, citation
config.json            # TIMM/HF model config
model.safetensors      # safetensors weights
pytorch_model.bin      # PyTorch weights for TIMM compatibility
model.onnx             # optional optimized ONNX graph with embedding/logits/prediction outputs
preprocess.json        # validation/test/pre transforms in dghs-imgutils format
selected_tags.csv      # label list + counts + per-label metrics + thresholds
categories.json        # category ids/names for multi-label taggers
thresholds.csv         # category-level suggested thresholds when available
metrics.json           # validation/test aggregate metrics
meta.json              # training/export/model metadata
sample.webp            # sample image used by model-card examples
events.out.tfevents.*  # TensorBoard logs when available
```

ONNX export wraps the TIMM classifier so the graph returns:

1. `embedding` — flattened feature vector before the classifier head
2. `logits` — raw classifier output
3. `prediction` — softmax or sigmoid probabilities, depending on export mode

---

## GitHub / Hugging Face Organization Context

- GitHub repo: [`deepghs/animetimm`](https://github.com/deepghs/animetimm)
- GitHub org: [`deepghs`](https://github.com/deepghs) — anime image data/model engineering projects such as
  [`imgutils`](https://github.com/deepghs/imgutils), [`waifuc`](https://github.com/deepghs/waifuc),
  [`cyberharem`](https://github.com/deepghs/cyberharem), and [`cheesechaser`](https://github.com/deepghs/cheesechaser)
- Hugging Face orgs: [`animetimm`](https://huggingface.co/animetimm) for TIMM-based anime model releases and
  [`deepghs`](https://huggingface.co/deepghs) for broader DeepGHS models, datasets, and demos
- Demo examples: [`animetimm/dbv4-full-playground`](https://huggingface.co/spaces/animetimm/dbv4-full-playground),
  [`animetimm/dbv4-full-ranklist`](https://huggingface.co/spaces/animetimm/dbv4-full-ranklist), and
  [`animetimm/e621v1-full-playground`](https://huggingface.co/spaces/animetimm/e621v1-full-playground)

---

## Repository Status

This is an active research/tooling repository rather than a polished public Python package. Expect command-line
entry points and training defaults to evolve with the DeepGHS release workflow. The exported Hugging Face model
repositories are the stable consumption surface for most downstream users.

## License

This repository is licensed under the [GNU General Public License v3.0](LICENSE).

Datasets and exported model repositories may have their own cards, licenses, gates, and acceptable-use notes. Always
check the target Hugging Face repository before downloading or redistributing artifacts.
