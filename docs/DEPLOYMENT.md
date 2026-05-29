# Deployment Guide / 部署教程

[English](#english) | [中文](#中文)

---

## English

This guide teaches a reader how to deploy MultiModal-QC locally from zero.

### 1. Install System Dependencies

Required:

- Python 3.10+
- Git
- NVIDIA driver and CUDA-compatible PyTorch if running Qwen2-VL locally
- 15GB+ disk for source + processed data; more if using full MVTec and checkpoints

Optional:

- Docker Desktop
- Hugging Face CLI

### 2. Clone the Repository

```bash
git clone https://github.com/Iroha-P/MultiModal-QC.git
cd MultiModal-QC
```

### 3. Create Python Environment

Windows one-click launcher:

```bat
oneclick.bat
```

This creates the environment, installs dependencies, downloads Release assets, attempts the Qwen base-model download, and starts the API/UI. Manual setup is still available:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare Models

Download the Qwen base model:

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
```

Download Release assets and extract them:

```text
MultiModal-QC-lora-best.zip      -> outputs/lora_defect/best/
MultiModal-QC-baselines.zip      -> outputs/baselines/ and runs/classify/
```

Expected paths:

```text
models/Qwen2-VL-2B-Instruct/config.json
outputs/lora_defect/best/adapter_model.safetensors
outputs/baselines/resnet/resnet50_defect.pt
runs/classify/outputs/baselines/yolo_defect/weights/best.pt
```

### 5. Prepare Data

Download Release assets:

```text
MultiModal-QC-dataset-metadata.zip
MultiModal-QC-dataset-mvtec-sample.zip
```

Extract them into the project root. You should get:

```text
data/train.json
data/val.json
data/test.json
data/mvtec_qa.json
data/mvtec/<category>/test/<label>/*.png
```

Drilling videos, Excel inspection sheets, and converted drilling JSON files are private local assets. They are not included in Git or Release packages. If you have authorized local drilling data, convert it under `private/drilling/` and keep it outside the public repository.

If you need the full third-party MVTec archive, download it from the official source or build local split parts with `tools/prepare_release.py --include-mvtec`. If you already have split parts, concatenate them and extract:

```powershell
Get-Content mvtec_anomaly_detection.tar.xz.part* -Encoding Byte -ReadCount 0 |
  Set-Content mvtec_anomaly_detection.tar.xz -Encoding Byte
tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec
```

### 6. Start Services

```bat
start.bat
```

Open:

- Gradio UI: <http://127.0.0.1:7860>
- API docs: <http://127.0.0.1:8000/docs>
- API health: <http://127.0.0.1:8000/health>

Stop:

```bat
stop.bat
```

### 7. Verify

Run tests:

```bash
pytest
```

Run evaluation:

```bash
python train/run_eval.py
```

Export inspection history in the UI. CSV files are written to:

```text
outputs/test_detection_data/
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Close other GPU apps, use 4-bit loading, or switch to smaller batch/inference. |
| `models not loaded` | Confirm `models/Qwen2-VL-2B-Instruct/` and `outputs/lora_defect/best/` exist. |
| Gradio opens but API fails | Visit `/health`; check `storage/api.log`. |
| Port conflict | Run `stop.bat`, then `start.bat`. |
| Chinese text looks garbled in terminal | Use UTF-8 terminal; browser UI should still render correctly. |

---

## 中文

本教程从零开始说明如何在本地完整部署 MultiModal-QC。

### 1. 安装系统依赖

必需：

- Python 3.10+
- Git
- 如果本地运行 Qwen2-VL，需要 NVIDIA 驱动和匹配的 CUDA/PyTorch
- 源码 + 处理后数据建议 15GB+；若使用完整 MVTec 和权重，需要更多空间

可选：

- Docker Desktop
- Hugging Face CLI

### 2. 克隆仓库

```bash
git clone https://github.com/Iroha-P/MultiModal-QC.git
cd MultiModal-QC
```

### 3. 创建 Python 环境

Windows 一键启动器：

```bat
oneclick.bat
```

它会自动创建环境、安装依赖、下载 Release 资产、尝试下载 Qwen 基座模型，并启动 API/UI。你也可以手动安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 准备模型

下载 Qwen 基座模型：

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
```

从 Release 下载并解压：

```text
MultiModal-QC-lora-best.zip      -> outputs/lora_defect/best/
MultiModal-QC-baselines.zip      -> outputs/baselines/ 和 runs/classify/
```

期望路径：

```text
models/Qwen2-VL-2B-Instruct/config.json
outputs/lora_defect/best/adapter_model.safetensors
outputs/baselines/resnet/resnet50_defect.pt
runs/classify/outputs/baselines/yolo_defect/weights/best.pt
```

### 5. 准备数据

从 Release 下载：

```text
MultiModal-QC-dataset-metadata.zip
MultiModal-QC-dataset-mvtec-sample.zip
```

解压到项目根目录，得到：

```text
data/train.json
data/val.json
data/test.json
data/mvtec_qa.json
data/mvtec/<category>/test/<label>/*.png
```

钻井视频、Excel检查表和转换后的钻井JSON属于本地私有资产，不包含在Git或Release包中。若你有授权的本地钻井数据，请在 `private/drilling/` 下转换，并保持在公开仓库之外。

如需完整第三方 MVTec 原始包，请从官方来源下载，或用 `tools/prepare_release.py --include-mvtec` 在本地生成分卷。若你已经有分卷，可这样合并并解压：

```powershell
Get-Content mvtec_anomaly_detection.tar.xz.part* -Encoding Byte -ReadCount 0 |
  Set-Content mvtec_anomaly_detection.tar.xz -Encoding Byte
tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec
```

### 6. 启动服务

```bat
start.bat
```

访问：

- Gradio UI：<http://127.0.0.1:7860>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

停止：

```bat
stop.bat
```

### 7. 验证

运行测试：

```bash
pytest
```

运行评估：

```bash
python train/run_eval.py
```

在 UI 中导出检测历史，CSV 默认写入：

```text
outputs/test_detection_data/
```

### 常见问题

| 问题 | 解决 |
|---|---|
| `CUDA out of memory` | 关闭其他 GPU 程序，确认使用 4bit 加载，或降低 batch/推理压力。 |
| `models not loaded` | 检查 `models/Qwen2-VL-2B-Instruct/` 和 `outputs/lora_defect/best/` 是否存在。 |
| Gradio 能开但 API 失败 | 访问 `/health`，查看 `storage/api.log`。 |
| 端口冲突 | 先运行 `stop.bat`，再运行 `start.bat`。 |
| 终端中文乱码 | 使用 UTF-8 终端；浏览器 UI 通常不受影响。 |
