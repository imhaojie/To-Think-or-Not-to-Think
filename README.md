<div align="center">

# To Think or Not to Think

### Pre-Decisional Reasoning Budgets for Referring Audio-Visual Segmentation

**Haojie Hu**, Senda Chen, Ying Shen, Lin Zhang

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](#installation)
[![Task](https://img.shields.io/badge/Task-Ref--AVS-2E8B57.svg)](#overview)
[![Benchmark](https://img.shields.io/badge/Benchmark-RefAVSBench%20%7C%20R²--AVSBench-8A2BE2.svg)](#results)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-6f42c1.svg)](https://neurips.cc/virtual/2026/poster/148558)

**Accepted to the Annual Conference on Neural Information Processing Systems (NeurIPS 2026).**

</div>

<p align="center">
  <img src="assets/teaser.png" width="94%" alt="Pre-decisional reasoning budgets for Ref-AVS">
</p>

## Overview

Long reasoning chains can help resolve ambiguous audio-visual references, but they can also introduce errors when the target is already clear. This work selects a reasoning budget for each input before decoding begins.

A lightweight controller uses the **generation-onset representation** `h_onset` to choose **Zero**, **Short**, or **Long** reasoning. This final-layer state comes from the last input token, after the model integrates video, audio, and text and before it generates any reasoning tokens. We call the budget information in this state the **Pre-Decisional Budget Signal (PDBS)**.

## Method

<p align="center">
  <img src="assets/pipeline.jpg" width="97%" alt="Pipeline of pre-decisional budget routing">
</p>

Frozen encoders and the Ref-AVS MLLM produce `h_onset`. The controller selects a budget, and the resulting query passes through GroundingDINO and SAM2 to produce segmentation masks. Only the controller is trained.

The three budgets use the same frozen detector and segmenter:

| Budget | Grounding phrase supplied to the perception engine |
| --- | --- |
| **Zero** | The raw referring expression — no reasoning at all. |
| **Short** | A compact object-aware description after one reasoning step. |
| **Long** | The full TGS-style multimodal CoT before the final descriptive phrase. |

The controller reuses the onset state from standard inference, so budget routing requires no additional MLLM forward pass.

## Results

Result tables, probing analyses, and qualitative comparisons will be added with the public paper.

## Repository Structure

| Path | Description |
| --- | --- |
| `configs/` | Model and SAM2 configuration files. |
| `dataset/` | Ref-AVS data loading and multimodal preprocessing. |
| `models/` | Ref-Thinker and multimodal model components. |
| `scripts/finetune/` | Ref-Thinker training and inference entrypoints. |
| `scripts/budget/` | Reasoning-budget inference, grounding, evaluation, aggregation. |
| `scripts/repro/` | Compact end-to-end reproduction helpers. |
| `scripts/download_weights.sh` | Helper for downloading public Hugging Face resources. |
| `ground_segment_scripts/` | Legacy Ground-Segment compatibility scripts. |
| `R2AVSBench/` | Lightweight metadata CSV files. |
| `docs/` | Installation, data, checkpoint, and reproduction guides. |
| `assets/` | Figures used in this README. |

Download data and model weights separately using the links below.

## Installation

```bash
git clone https://github.com/imhaojie/To-Think-or-Not-to-Think.git
cd To-Think-or-Not-to-Think
```

Create the Ref-Thinker environment:

```bash
conda env create -f think_environment.yml
conda activate think
```

Create the Ground-Segment environment:

```bash
conda env create -f dinosam2_environment.yml
conda activate dino
```

Install Grounded-SAM-2 under `ground_segment_scripts/`:

```bash
cd ground_segment_scripts
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd ..
ln -s ground_segment_scripts/Grounded-SAM-2/grounding_dino grounding_dino
ln -s ground_segment_scripts/Grounded-SAM-2/sam2 sam2
ln -s ground_segment_scripts/Grounded-SAM-2/checkpoints checkpoints
ln -s ground_segment_scripts/Grounded-SAM-2/gdino_checkpoints gdino_checkpoints
```

See [docs/INSTALL.md](docs/INSTALL.md) for additional notes.

## Model Zoo

| Component | Target path | Link |
| --- | --- | --- |
| LLaMA-2-7B-Chat-HF | `pretrained_weights/Llama-2-7b-chat-hf/` | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| CLIP ViT-L/14 | `pretrained_weights/clip-vit-large-patch14/` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |
| BEATs cpt2 | `pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | [microsoft/unilm BEATs](https://github.com/microsoft/unilm/tree/master/beats) |
| Audio projector | `pretrained_weights/audio_pretrain.bin` | [ahsgdxhs/Crab](https://huggingface.co/ahsgdxhs/Crab/tree/main) |
| Visual projector | `pretrained_weights/visual_pretrain.bin` | [ahsgdxhs/Crab](https://huggingface.co/ahsgdxhs/Crab/tree/main) |
| Ref-Thinker checkpoint | `results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/` | [Jinxing1/TGS-Agent](https://huggingface.co/Jinxing1/TGS-Agent/tree/main) |
| GroundingDINO Swin-T | `gdino_checkpoints/groundingdino_swint_ogc.pth` | [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| SAM2.1 Hiera-Large | `checkpoints/sam2.1_hiera_large.pt` | [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |

For public Hugging Face files:

```bash
pip install -U huggingface_hub
bash scripts/download_weights.sh
```

LLaMA-2 is gated. Request access on Hugging Face, then:

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir pretrained_weights/Llama-2-7b-chat-hf \
  --local-dir-use-symlinks False
```

See [docs/WEIGHTS.md](docs/WEIGHTS.md) for path checks and manual commands.

## Data Preparation

Expected local layout:

```text
REFAVS/
  media/
  gt_mask/

R2AVSBench/
  RefAVSBench_metadata.csv
  R2AVSBench_metadata.csv
  RefAVSBenchRef_R2AVSBenchVideo.csv
  RefThinker_instruction_tuning_set.json
```

The metadata CSV files are included. Download the full Ref-AVSBench videos and masks from the official [Ref-AVS repository](https://github.com/GeWu-Lab/Ref-AVS), and the instruction-tuning JSON from [Jinxing1/TGSAgent-FT-data](https://huggingface.co/datasets/Jinxing1/TGSAgent-FT-data/tree/main), placed at `R2AVSBench/RefThinker_instruction_tuning_set.json`.

See [docs/DATA.md](docs/DATA.md) for details.

## Verify Your Setup

```bash
python scripts/check_smoke.py            # structure check, reports missing resources
python scripts/check_smoke.py --strict   # after downloading data and weights
```

## Reproduction

### Ref-Thinker inference

```bash
conda activate think
TGS_DEVICE=cuda:0 bash scripts/finetune/inference_hyper_lora.sh
```

Common overrides:

```bash
TGS_MODEL_PATH=/path/to/Llama-2-7b-chat-hf
TGS_VIT_PATH=/path/to/clip-vit-large-patch14
TGS_BEATS_PATH=/path/to/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
TGS_CKPT_BASE=/path/to/ref-thinker-checkpoint
TGS_CKPT_DIR=/path/to/ref-thinker-checkpoint/checkpoint-551
TGS_TEST_NAME=test_u
TGS_DEVICE=cuda:0
```

### Single budget evaluation

```bash
conda activate dino
python scripts/budget/infer_budget.py \
  --dataset refavs \
  --split test_u \
  --budget-mode long \
  --output-root budget_results/main \
  --device cuda:0 \
  --overwrite
```

Supported budget modes: `zero`, `short`, `long`, `rewrite`, `trunc_long`.

### Compact pipeline

```bash
TGS_DEVICE=cuda:0 TGS_CONDA_ENV=dino bash scripts/repro/repro_full_pipeline.sh
```

More commands in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Reproducibility Notes

- Override default paths with `TGS_*` environment variables or CLI arguments.
- All splits use seed 42.
- The legacy scripts in `ground_segment_scripts/` are kept for compatibility; the preferred public entrypoint is `scripts/budget/infer_budget.py`.

## Citation

The paper is listed on the [NeurIPS 2026 conference page](https://neurips.cc/virtual/2026/poster/148558). The arXiv preprint and final BibTeX will be added when available.

## Acknowledgements

This project builds on [Crab](https://huggingface.co/ahsgdxhs/Crab), [Ref-AVS](https://github.com/GeWu-Lab/Ref-AVS), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), GroundingDINO, SAM2, BEATs, CLIP, Transformers, and PEFT-style adaptation. Please follow the licenses and terms of the upstream projects, datasets, and pretrained models.
