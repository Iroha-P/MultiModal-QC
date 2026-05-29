<div align="center">

# MultiModal-QC

> Industrial multimodal quality-control demo built with Qwen2-VL, QLoRA, FastAPI, Gradio, and an explainable Agent Pipeline.

**Language / 语言:** [English](README.md) | [简体中文](README.zh-CN.md)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![Qwen2-VL](https://img.shields.io/badge/VLM-Qwen2--VL-0052FF?style=flat-square)](https://huggingface.co/Qwen)
[![QLoRA](https://img.shields.io/badge/Fine--tuning-QLoRA-4D7CFF?style=flat-square)](https://arxiv.org/abs/2305.14314)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)](https://www.gradio.app/)

[Demo UI](#demo-ui) · [Quick Start](#quick-start) · [Release Package](#release-package) · [Project Structure](#project-structure) · [Reproduce](#reproduce-the-experiment)

</div>

> This repository contains the source code, documentation, conversion scripts, training/evaluation scripts, and a polished Gradio portfolio demo. Large datasets, base models, and checkpoints are distributed through GitHub Releases instead of Git.

---

## Demo UI

![MultiModal-QC Gradio Demo](docs/assets/demo-gradio-inspect.png)

## What It Does

MultiModal-QC is a portfolio-oriented industrial inspection system with two scenarios:

1. **MVTec-AD defect inspection**: classify product images and generate Chinese defect descriptions.
2. **Drilling compliance demo**: keep a video-compliance entry point for drilling-site inspection data.

The core path is:

```text
Image / video input
  -> Qwen2-VL + QLoRA
  -> Agent Pipeline: perceive -> detect -> decide -> report
  -> visual result + structured inspection history
```

## Features

- **Multimodal VLM inspection**: Qwen2-VL-2B-Instruct with QLoRA 4-bit fine-tuning.
- **Explainable pipeline**: four Agent steps: perception, detection, decision, and report.
- **Baseline comparison**: ResNet50 and YOLOv8n-cls baselines for classification-only comparison.
- **Interactive demo**: FastAPI backend + Gradio frontend with a SaaS-style portfolio UI.
- **Persistent records**: SQLite stores inspection history, model output, confidence/fit score, defect type, location, severity, and suggestion.
- **CSV export**: export test/detection results to `outputs/test_detection_data`.
- **Reproducible scripts**: dataset conversion, split, training, evaluation, ablation plotting, and demo serving.

## Results

MVTec-AD test set, `n=173`.

| Method | Accuracy | Precision | Recall | F1 | ROUGE-L | Output |
|---|---:|---:|---:|---:|---:|---|
| ResNet50 | 90.12% | - | - | 0.905 | - | binary classification |
| YOLOv8n-cls | 83.24% | - | - | 0.827 | - | binary classification |
| **Qwen2-VL-2B + QLoRA** | **89.02%** | **90.07%** | **89.02%** | **0.893** | **0.249** | Chinese defect description |

Authoritative result files:

- `outputs/eval_results.json`
- `outputs/eval_predictions.json`

## Requirements

| Item | Recommended |
|---|---|
| OS | Windows 10/11, Linux also works for scripts |
| Python | 3.10+ |
| GPU | NVIDIA GPU, 8GB+ VRAM recommended for Qwen2-VL QLoRA inference |
| CUDA | Match your local PyTorch build |
| Disk | 15GB+ for source + processed data; 25GB+ if using full MVTec and checkpoints |

## Quick Start

### Option A: one-click Windows package

1. Download and extract `MultiModal-QC-oneclick.zip`.
2. Double-click:

```bat
oneclick.bat
```

The launcher creates `.venv`, installs PyTorch and project dependencies, downloads Release assets, tries to download Qwen2-VL, and starts both services. For manual deployment, you can still run `install.bat` and then `start.bat`.

Expected model/checkpoint folders after setup:

```text
models/Qwen2-VL-2B-Instruct/
outputs/lora_defect/best/
outputs/baselines/resnet/
runs/classify/outputs/baselines/yolo_defect/weights/
```

Open:

- Gradio UI: <http://127.0.0.1:7860>
- FastAPI docs: <http://127.0.0.1:8000/docs>

Stop services:

```bat
stop.bat
```

### Option B: clone from source

```bash
git clone https://github.com/Iroha-P/MultiModal-QC.git
cd MultiModal-QC
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download Qwen2-VL-2B-Instruct:

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
```

Then download the release checkpoint/data assets, extract them to the paths shown above, and run:

```bash
start.bat
```

## Release Package

The Release is designed to avoid committing huge binaries to Git:

| Asset | Purpose |
|---|---|
| `MultiModal-QC-oneclick.zip` | Source code, docs, scripts, generated demo assets, install/start scripts |
| `MultiModal-QC-dataset-metadata.zip` | Processed QA JSON files and train/val/test splits |
| `MultiModal-QC-dataset-mvtec-sample.zip` | 173 MVTec test images referenced by `data/test.json`, enough for local evaluation/demo |
| `MultiModal-QC-lora-best.zip` | Qwen2-VL QLoRA adapter checkpoint |
| `MultiModal-QC-baselines.zip` | ResNet50 and YOLOv8 baseline weights/results |
| `mvtec_anomaly_detection.tar.xz.partNN` | Optional local output from `tools/prepare_release.py --include-mvtec`; the full third-party MVTec archive is not required for the one-click demo |

If you already have local split parts, rebuild the full MVTec archive with:

```powershell
Get-Content mvtec_anomaly_detection.tar.xz.part* -Encoding Byte -ReadCount 0 |
  Set-Content mvtec_anomaly_detection.tar.xz -Encoding Byte
tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec
```

## Reproduce the Experiment

### 1. Convert MVTec

```bash
python data/scripts/convert_mvtec.py data/mvtec
python data/scripts/split_dataset.py data/mvtec_qa.json data/
```

### 2. Convert private drilling metadata

```bash
python data/scripts/convert_drilling.py private/drilling/source.xlsx private/drilling/drilling_qa.json --video-dir private/drilling/videos
python data/scripts/split_dataset.py private/drilling/drilling_qa.json private/drilling/splits/
```

The drilling source videos, Excel inspection sheet, and converted JSON are private local assets. They are intentionally excluded from Git and GitHub Releases.

### 3. Train QLoRA

```bash
python train/train.py train/configs/lora_defect.yaml
```

### 4. Evaluate

```bash
python train/run_eval.py
```

### 5. Run baselines

```bash
python baselines/resnet_classifier.py
python baselines/yolo_detector.py
```

## Project Structure

```text
MultiModal-QC/
├─ agents/                  # Qwen2-VL wrapper, Agent Pipeline, schemas, prompts
├─ baselines/               # ResNet50 and YOLOv8 baseline scripts
├─ data/scripts/            # Dataset conversion and split scripts
├─ docs/                    # Technical report, demo script, file guide, portfolio notes
├─ serve/                   # FastAPI backend, Gradio UI, SQLite helper, visualization
├─ tests/                   # Unit tests for data conversion, API, DB, pipeline, schema
├─ train/                   # QLoRA training, evaluation, ablation, plotting
├─ config.py                # Central paths and scene configuration
├─ install.bat              # One-click Windows dependency installer
├─ start.bat / stop.bat     # Start/stop API and Gradio services
└─ requirements.txt         # Python dependency list
```

For a detailed file-by-file explanation, see [docs/FILE_GUIDE.md](docs/FILE_GUIDE.md).

## Data Notes

- MVTec-AD is a public anomaly-detection dataset. Follow the original dataset license/terms when redistributing or using it.
- Drilling data is a private local dataset. The public repository keeps only the conversion script and demo entry point, not the raw videos, inspection sheet, or converted private JSON.
- Large raw datasets and model weights are intentionally kept out of Git and distributed through Releases.

## Documentation

- [File Guide](docs/FILE_GUIDE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Technical Report](docs/技术报告.md)
- [Demo Script](docs/Demo演示脚本.md)
- [Portfolio Page Notes](docs/作品集展示页.md)

## Tech Stack

| Layer | Technology |
|---|---|
| VLM | Qwen2-VL-2B-Instruct |
| Fine-tuning | QLoRA, PEFT, Transformers, bitsandbytes |
| Backend | FastAPI, Uvicorn |
| Frontend | Gradio + custom CSS |
| Storage | SQLite |
| Baselines | PyTorch / torchvision ResNet50, Ultralytics YOLOv8 |
| Evaluation | scikit-learn, rouge-score |

## License

MIT License for this codebase. Dataset and pretrained model assets follow their original licenses and terms.
