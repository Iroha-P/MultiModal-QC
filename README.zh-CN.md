<div align="center">

# MultiModal-QC 工业多模态质检平台

> 基于 Qwen2-VL、QLoRA、FastAPI、Gradio 和可解释 Agent Pipeline 的工业质检 Demo。

**Language / 语言:** [English](README.md) | [简体中文](README.zh-CN.md)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![Qwen2-VL](https://img.shields.io/badge/VLM-Qwen2--VL-0052FF?style=flat-square)](https://huggingface.co/Qwen)
[![QLoRA](https://img.shields.io/badge/Fine--tuning-QLoRA-4D7CFF?style=flat-square)](https://arxiv.org/abs/2305.14314)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)](https://www.gradio.app/)

[Demo 界面](#demo-界面) · [快速开始](#快速开始) · [Release 一键包](#release-一键包) · [项目结构](#项目结构) · [完整复刻实验](#完整复刻实验)

</div>

> 本仓库保存源码、文档、数据转换脚本、训练/评估脚本和 Gradio 作品集 Demo。大数据集、基座模型与训练权重不直接提交到 Git，而是通过 GitHub Release 发布。

---

## Demo 界面

![MultiModal-QC Gradio Demo](docs/assets/demo-gradio-inspect.png)

## 项目简介

MultiModal-QC 是一个面向简历/作品集展示的工业多模态质检系统，包含两个场景：

1. **MVTec-AD 图像缺陷检测**：对产品图像进行合格/不合格判断，并生成中文缺陷描述。
2. **钻井合规 Demo**：保留钻井视频合规检测入口，用于展示真实工业视频数据接入能力。

核心流程：

```text
图像 / 视频输入
  -> Qwen2-VL + QLoRA
  -> Agent Pipeline：感知 -> 检测 -> 决策 -> 报告
  -> 可视化结果 + 结构化检测历史
```

## 功能亮点

- **多模态大模型质检**：Qwen2-VL-2B-Instruct + QLoRA 4bit 微调。
- **可解释 Agent Pipeline**：感知、检测、决策、报告四阶段输出可追踪。
- **基线模型对比**：ResNet50 与 YOLOv8n-cls 作为纯分类基线。
- **可交互 Demo**：FastAPI 后端 + Gradio 前端，带作品集风格 UI。
- **历史记录持久化**：SQLite 保存检测场景、模型/流程、判定、置信度/拟合度、缺陷类型、位置、严重度与建议。
- **CSV 导出**：测试/检测数据默认导出到 `outputs/test_detection_data`。
- **完整复刻脚本**：包含数据转换、划分、训练、评估、消融与可视化脚本。

## 实验结果

MVTec-AD 测试集，`n=173`。

| 方法 | Accuracy | Precision | Recall | F1 | ROUGE-L | 输出能力 |
|---|---:|---:|---:|---:|---:|---|
| ResNet50 | 90.12% | - | - | 0.905 | - | 二分类 |
| YOLOv8n-cls | 83.24% | - | - | 0.827 | - | 二分类 |
| **Qwen2-VL-2B + QLoRA** | **89.02%** | **90.07%** | **89.02%** | **0.893** | **0.249** | 中文缺陷描述 |

权威结果文件：

- `outputs/eval_results.json`
- `outputs/eval_predictions.json`

## 环境要求

| 项目 | 推荐配置 |
|---|---|
| 操作系统 | Windows 10/11；训练/评估脚本也可在 Linux 运行 |
| Python | 3.10+ |
| GPU | NVIDIA GPU，Qwen2-VL QLoRA 推理建议 8GB+ 显存 |
| CUDA | 与本地 PyTorch 版本匹配 |
| 磁盘 | 源码 + 处理后数据建议 15GB+；若使用完整 MVTec 与权重建议 25GB+ |

## 快速开始

### 方式 A：Windows 一键包

1. 下载并解压 `MultiModal-QC-oneclick.zip`。
2. 双击运行：

```bat
oneclick.bat
```

启动器会自动创建 `.venv`、安装 PyTorch 和项目依赖、下载 Release 资产、尝试下载 Qwen2-VL，并启动 API 与 Gradio。需要手动部署时，也可以继续使用 `install.bat` + `start.bat`。

安装完成后的模型/权重路径应为：

```text
models/Qwen2-VL-2B-Instruct/
outputs/lora_defect/best/
outputs/baselines/resnet/
runs/classify/outputs/baselines/yolo_defect/weights/
```

访问：

- Gradio UI：<http://127.0.0.1:7860>
- FastAPI 文档：<http://127.0.0.1:8000/docs>

停止服务：

```bat
stop.bat
```

### 方式 B：从源码复刻

```bash
git clone https://github.com/Iroha-P/MultiModal-QC.git
cd MultiModal-QC
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

下载 Qwen2-VL-2B-Instruct：

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
```

再从 Release 下载权重/数据资产并解压到上方目录，最后运行：

```bash
start.bat
```

## Release 一键包

为了避免 Git 仓库存放超大二进制文件，Release 会包含以下资产：

| 资产 | 作用 |
|---|---|
| `MultiModal-QC-oneclick.zip` | 源码、文档、脚本、Demo 图片、一键安装与启动脚本 |
| `MultiModal-QC-dataset-metadata.zip` | 已处理 QA JSON 与 train/val/test 划分 |
| `MultiModal-QC-dataset-mvtec-sample.zip` | `data/test.json` 引用的 173 张 MVTec 测试图片，可用于本地评估和 Demo |
| `MultiModal-QC-lora-best.zip` | Qwen2-VL QLoRA adapter 权重 |
| `MultiModal-QC-baselines.zip` | ResNet50 / YOLOv8 基线权重与结果 |
| `mvtec_anomaly_detection.tar.xz.partNN` | `tools/prepare_release.py --include-mvtec` 的本地可选输出；一键 Demo 不依赖完整第三方原始包 |

如果你本地已有 MVTec 分卷，可这样合并：

```powershell
Get-Content mvtec_anomaly_detection.tar.xz.part* -Encoding Byte -ReadCount 0 |
  Set-Content mvtec_anomaly_detection.tar.xz -Encoding Byte
tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec
```

## 完整复刻实验

### 1. 转换 MVTec

```bash
python data/scripts/convert_mvtec.py data/mvtec
python data/scripts/split_dataset.py data/mvtec_qa.json data/
```

### 2. 转换私有钻井数据

```bash
python data/scripts/convert_drilling.py private/drilling/source.xlsx private/drilling/drilling_qa.json --video-dir private/drilling/videos
python data/scripts/split_dataset.py private/drilling/drilling_qa.json private/drilling/splits/
```

钻井原始视频、Excel检查表和转换后的JSON均属于本地私有资产，不进入Git，也不上传到GitHub Release。

### 3. 训练 QLoRA

```bash
python train/train.py train/configs/lora_defect.yaml
```

### 4. 评估

```bash
python train/run_eval.py
```

### 5. 运行基线模型

```bash
python baselines/resnet_classifier.py
python baselines/yolo_detector.py
```

## 项目结构

```text
MultiModal-QC/
├─ agents/                  # Qwen2-VL 封装、Agent Pipeline、Schema 与 Prompt
├─ baselines/               # ResNet50 / YOLOv8 基线脚本
├─ data/scripts/            # 数据转换与划分脚本
├─ docs/                    # 技术报告、演示脚本、文件说明、作品集材料
├─ serve/                   # FastAPI 后端、Gradio UI、SQLite、可视化
├─ tests/                   # 数据转换、API、DB、Pipeline、Schema 单元测试
├─ train/                   # QLoRA 训练、评估、消融、绘图
├─ config.py                # 路径与场景配置
├─ install.bat              # Windows 一键安装依赖
├─ start.bat / stop.bat     # 启动/停止 API 与 Gradio
└─ requirements.txt         # Python 依赖
```

逐文件说明见：[docs/FILE_GUIDE.md](docs/FILE_GUIDE.md)。

## 数据说明

- MVTec-AD 是公开异常检测数据集，使用和再分发请遵守其原始协议。
- 钻井数据为本地私有数据。公开仓库仅保留转换脚本和Demo入口，不包含原始视频、检查表或转换后的私有JSON。
- 大型数据和模型权重不进入 Git，统一走 Release 资产。

## 文档索引

- [逐文件说明](docs/FILE_GUIDE.md)
- [部署教程](docs/DEPLOYMENT.md)
- [技术报告](docs/技术报告.md)
- [Demo 演示脚本](docs/Demo演示脚本.md)
- [作品集展示材料](docs/作品集展示页.md)

## 技术栈

| 层级 | 技术 |
|---|---|
| 多模态大模型 | Qwen2-VL-2B-Instruct |
| 微调 | QLoRA、PEFT、Transformers、bitsandbytes |
| 后端 | FastAPI、Uvicorn |
| 前端 | Gradio + 自定义 CSS |
| 存储 | SQLite |
| 基线模型 | PyTorch / torchvision ResNet50、Ultralytics YOLOv8 |
| 评估 | scikit-learn、rouge-score |

## License / 协议

本代码仓库使用 MIT License。数据集与预训练模型资产遵循其原始许可证和使用条款。
