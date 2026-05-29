# File Guide / 文件作用说明

[English](#english) | [中文](#中文)

---

## English

This guide explains the purpose of every source file and important project directory. Large runtime artifacts such as `models/`, `outputs/`, `data/mvtec/`, `.venv/`, and `storage/` are intentionally excluded from Git and distributed through Releases or generated locally.

### Root Files

| Path | Purpose |
|---|---|
| `.gitignore` | Keeps virtual environments, checkpoints, raw datasets, logs, and release archives out of Git. |
| `README.md` | Main English project page: overview, quick start, results, release package, reproduction guide. |
| `README.zh-CN.md` | Chinese project page with the same deployment and reproduction instructions. |
| `requirements.txt` | Python dependencies for training, evaluation, API, and Gradio demo. |
| `config.py` | Central path and scene configuration: data, models, outputs, DB path, Qwen base model, LoRA paths, scene names. |
| `docker-compose.yml` | Docker Compose entry for container-based deployment. |
| `install.bat` | One-click Windows installer: creates `.venv` and installs dependencies. |
| `start.bat` | Starts FastAPI on port `8000` and Gradio on port `7860`; writes logs to `storage/`. |
| `run_demo.bat` | Convenience wrapper around `start.bat` for release users. |
| `stop.bat` | Stops processes listening on ports `8000` and `7860`. |

### `agents/`

| Path | Purpose |
|---|---|
| `agents/__init__.py` | Marks `agents` as a Python package. |
| `agents/model.py` | Qwen2-VL inference wrapper; loads base model, optional LoRA adapter, 4-bit quantization, and chat interface. |
| `agents/pipeline.py` | Four-stage inspection pipeline: perceive, detect, decide, report. |
| `agents/schema.py` | Structured report dataclasses / schema objects used by the pipeline and API. |
| `agents/prompts/defect_perceive.txt` | Prompt for visual perception in the defect-inspection scene. |
| `agents/prompts/defect_detect.txt` | Prompt for extracting defect type, location, and confidence. |
| `agents/prompts/drilling_perceive.txt` | Prompt for visual perception in the drilling-compliance scene. |
| `agents/prompts/drilling_detect.txt` | Prompt for drilling-compliance detection. |
| `agents/prompts/decide.txt` | Decision prompt for final pass/fail judgment. |
| `agents/prompts/report.txt` | Report-generation prompt for final natural-language output. |

### `baselines/`

| Path | Purpose |
|---|---|
| `baselines/__init__.py` | Marks `baselines` as a Python package. |
| `baselines/resnet_classifier.py` | Trains/evaluates the ResNet50 binary classifier baseline. |
| `baselines/yolo_detector.py` | Trains/evaluates the YOLOv8 classification baseline. |

### `data/`

| Path | Purpose |
|---|---|
| `data/__init__.py` | Marks `data` as a Python package. |
| `data/scripts/__init__.py` | Marks data scripts as a package for tests/imports. |
| `data/scripts/convert_mvtec.py` | Converts MVTec-AD image folders into Chinese QA-style samples. |
| `data/scripts/convert_drilling.py` | Converts drilling video + Excel inspection metadata into QA samples. |
| `data/scripts/split_dataset.py` | Splits QA JSON into train/val/test JSON files. |
| `data/train.json` | Processed training split for the defect scene; generated/release asset. |
| `data/val.json` | Processed validation split for the defect scene; generated/release asset. |
| `data/test.json` | Processed test split for the defect scene; generated/release asset. |
| `data/mvtec_qa.json` | Full processed MVTec QA metadata; generated/release asset. |
| `private/drilling/*.json` | Optional local-only drilling QA metadata and splits; ignored by Git and not distributed through Releases. |

### `docs/`

| Path | Purpose |
|---|---|
| `docs/FILE_GUIDE.md` | This file-by-file guide. |
| `docs/DEPLOYMENT.md` | End-to-end deployment and local reproduction tutorial. |
| `docs/技术报告.md` | Technical report: model comparison, architecture, limitations, screenshots. |
| `docs/Demo演示脚本.md` | Demo narration and interview walkthrough script. |
| `docs/作品集展示页.md` | Portfolio-page copy and display material. |
| `docs/assets/` | README/report screenshots and generated visual assets. |

### `serve/`

| Path | Purpose |
|---|---|
| `serve/__init__.py` | Marks `serve` as a Python package. |
| `serve/api.py` | FastAPI backend: health check, scenes, inspection APIs, model loading, visualization output, SQLite writes. |
| `serve/app.py` | Gradio frontend: SaaS-style UI, quick inspection, full pipeline, history table, CSV export. |
| `serve/database.py` | SQLite helper for inspections, agent results, and defect records. |
| `serve/visualize.py` | Visualization utilities: Grad-CAM for baselines and Qwen annotation rendering. |
| `serve/Dockerfile` | Container build file for the service. |

### `train/`

| Path | Purpose |
|---|---|
| `train/__init__.py` | Marks `train` as a Python package. |
| `train/train.py` | QLoRA training entry point. |
| `train/evaluate.py` | Evaluation utilities shared by scripts. |
| `train/run_eval.py` | Evaluation runner that writes `outputs/eval_results.json` and predictions. |
| `train/ablation.py` | Rank/parameter ablation runner. |
| `train/plot_ablation.py` | Creates ablation comparison figures. |
| `train/configs/lora_defect.yaml` | QLoRA config for MVTec defect scene. |
| `train/configs/lora_drilling.yaml` | QLoRA config scaffold for drilling scene. |

### `tests/`

| Path | Purpose |
|---|---|
| `tests/__init__.py` | Marks tests as a package. |
| `tests/test_api.py` | API smoke tests. |
| `tests/test_database.py` | SQLite schema and CRUD tests. |
| `tests/test_schema.py` | Structured report schema tests. |
| `tests/test_pipeline.py` | Agent Pipeline behavior tests. |
| `tests/test_convert_mvtec.py` | MVTec conversion tests. |
| `tests/test_convert_drilling.py` | Drilling conversion tests. |
| `tests/test_split_dataset.py` | Dataset split tests. |

### Runtime / Release-Only Paths

| Path | Purpose |
|---|---|
| `models/Qwen2-VL-2B-Instruct/` | Qwen base model downloaded from Hugging Face; not committed. |
| `outputs/lora_defect/best/` | Best QLoRA adapter checkpoint; Release asset. |
| `outputs/baselines/resnet/` | ResNet baseline checkpoint/results; Release asset. |
| `runs/classify/.../weights/` | YOLOv8 baseline weights; Release asset. |
| `storage/` | Uploads, visualizations, logs, and temporary files; generated locally. |
| `multimodal_qc.db` | SQLite runtime database; generated locally. |
| `data/mvtec/` | Full MVTec dataset; Release split asset or user-provided download. |
| `private/drilling/` | Optional authorized drilling source videos, Excel inspection sheet, and converted JSON; local-only, never committed. |

---

## 中文

本文件说明项目中每个源码文件和关键目录的作用。大型运行产物（如 `models/`、`outputs/`、`data/mvtec/`、`.venv/`、`storage/`）不会提交到 Git，而是通过 Release 下载或本地生成。

### 根目录文件

| 路径 | 作用 |
|---|---|
| `.gitignore` | 忽略虚拟环境、模型权重、原始数据集、日志和发布压缩包。 |
| `README.md` | 英文首页：项目介绍、快速开始、结果、Release 包、复刻流程。 |
| `README.zh-CN.md` | 中文首页：部署和复刻说明。 |
| `requirements.txt` | 训练、评估、API、Gradio Demo 所需 Python 依赖。 |
| `config.py` | 统一配置数据、模型、输出、数据库、Qwen 基座模型、LoRA 路径和场景名。 |
| `docker-compose.yml` | Docker Compose 部署入口。 |
| `install.bat` | Windows 一键安装：创建 `.venv` 并安装依赖。 |
| `start.bat` | 启动 FastAPI `8000` 和 Gradio `7860`，日志写入 `storage/`。 |
| `run_demo.bat` | Release 用户使用的启动快捷脚本。 |
| `stop.bat` | 停止占用 `8000` 和 `7860` 端口的服务。 |

### 主要目录

| 路径 | 作用 |
|---|---|
| `agents/` | Qwen2-VL 推理封装、四阶段 Agent Pipeline、结构化报告 Schema 和 Prompt。 |
| `baselines/` | ResNet50 与 YOLOv8 基线模型训练/评估脚本。 |
| `data/scripts/` | MVTec、钻井数据转换和 train/val/test 划分脚本。 |
| `docs/` | 技术报告、Demo 脚本、逐文件说明、作品集材料和交接文档。 |
| `serve/` | FastAPI 后端、Gradio 前端、SQLite 数据库工具和可视化工具。 |
| `tests/` | API、数据库、Schema、Pipeline、数据转换和划分的单元测试。 |
| `train/` | QLoRA 训练、评估、消融实验和绘图脚本。 |

### `serve/` 关键文件

| 文件 | 作用 |
|---|---|
| `serve/api.py` | 后端服务：加载模型、接收上传、执行 quick/full inspection、写入 SQLite、返回可视化结果。 |
| `serve/app.py` | 前端页面：SaaS 风格 Gradio UI、快速检测、完整 Pipeline、历史参数表、CSV 导出。 |
| `serve/database.py` | SQLite 封装：检测任务、Agent 输出和缺陷记录三张表。 |
| `serve/visualize.py` | Grad-CAM 和 Qwen 标注可视化。 |

### `train/` 关键文件

| 文件 | 作用 |
|---|---|
| `train/train.py` | QLoRA 训练入口。 |
| `train/run_eval.py` | 评估入口，生成真实指标文件。 |
| `train/ablation.py` | LoRA rank/参数消融实验脚本。 |
| `train/plot_ablation.py` | 消融实验图表生成脚本。 |
| `train/configs/*.yaml` | 不同场景的训练配置。 |

### 运行时 / Release 目录

| 路径 | 作用 |
|---|---|
| `models/Qwen2-VL-2B-Instruct/` | Hugging Face 下载的 Qwen 基座模型，不进 Git。 |
| `outputs/lora_defect/best/` | 最佳 QLoRA adapter，Release 资产。 |
| `outputs/baselines/resnet/` | ResNet 基线权重和结果，Release 资产。 |
| `runs/classify/.../weights/` | YOLOv8 基线权重，Release 资产。 |
| `storage/` | 上传文件、可视化、日志和临时文件，本地生成。 |
| `multimodal_qc.db` | SQLite 数据库，本地生成。 |
| `data/mvtec/` | 完整 MVTec 数据集，通过 Release 分卷或用户自行下载。 |
| `private/drilling/` | 可选的授权钻井视频、Excel检查表和转换JSON；仅本地使用，不提交。 |
